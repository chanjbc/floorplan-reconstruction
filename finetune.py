import json
import os
import shutil
import sys
from pathlib import Path
import subprocess
import random
from PIL import Image
import torch

def preprocess_image(img, target_size=256, pad_value=(255, 255, 255)):
    """
    Resizes image to fit within target_size x target_size.
    Returns: processed_image, scale, (x_offset, y_offset)
    """
    w, h = img.size
    
    # Calculate scale to fit the longest edge into target_size
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize the original image
    resized_img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    # Create a new blank background (square)
    new_img = Image.new("RGB", (target_size, target_size), pad_value)
    
    # Calculate centering offsets
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    
    # Paste the resized image at the calculated offsets (Centered)
    new_img.paste(resized_img, (x_offset, y_offset))
    
    return new_img, scale, (x_offset, y_offset)

def sanitize_checkpoint(ckpt_path):
    """
    Removes 'args' from checkpoint if present to avoid UnpicklingError 
    with weights_only=True in PyTorch 2.6+.
    Only removes room_class_embed to allow fine-tuning on new classes.
    KEEPS token_embed to preserve geometric knowledge.
    Returns path to sanitized checkpoint.
    """
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        return ckpt_path
        
    sanitized_path = ckpt_path.parent / f"{ckpt_path.stem}_sanitized{ckpt_path.suffix}"
    
    # Always regenerate to ensure we catch new mismatches
    if sanitized_path.exists():
        try:
            os.remove(sanitized_path)
        except OSError:
            pass
        
    print(f"Sanitizing checkpoint {ckpt_path}...")
    try:
        # Load with weights_only=False to allow reading the legacy file
        data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        
        # Remove args
        if 'args' in data:
            del data['args']

        # Remove ONLY class embeddings
        if 'model' in data:
            state_dict = data['model']
            keys_to_remove = []
            for k in state_dict.keys():
                # Remove class embeddings (12 classes -> 2 classes mismatch)
                if 'room_class_embed' in k:
                    keys_to_remove.append(k)
                
            for k in keys_to_remove:
                print(f"Removing key {k} from checkpoint to allow shape change.")
                del state_dict[k]
            
            data['model'] = state_dict
            
        torch.save(data, sanitized_path)
        return sanitized_path
    except Exception as e:
        print(f"Failed to sanitize checkpoint: {e}")
        return ckpt_path

def setup_finetune_dataset(val_split=0.1, test_split=0.1):
    print("Setting up fine-tuning dataset from combined annotations...")
    
    # Define paths relative to this script
    current_dir = Path(__file__).resolve().parent
    base_dir = current_dir.parent
    
    # Paths specified in prompt
    ann_file = base_dir / 'annotations' / 'annotations.json'
    img_source_dir = base_dir / 'original_size_images'
    finetune_dataset_dir = base_dir / 'finetune_dataset'
    
    if not ann_file.exists():
        print(f"Error: Annotation file not found at {ann_file}")
        return None, 0

    # Clean and create directories
    if finetune_dataset_dir.exists():
        print(f"Removing existing dataset directory: {finetune_dataset_dir}")
        shutil.rmtree(finetune_dataset_dir)
    
    img_dest_dir = finetune_dataset_dir / 'images'
    img_dest_dir.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {img_dest_dir}")
    
    # Containers for train, val, test
    datasets = {
        'train': {'images': [], 'annotations': []},
        'val': {'images': [], 'annotations': []},
        'test': {'images': [], 'annotations': []}
    }
    
    print(f"Loading annotations from {ann_file}...")
    with open(ann_file, 'r') as f:
        data = json.load(f)
        
    categories = data.get('categories', [])
    # Ensure categories have sequential IDs starting from 0
    cat_id_map = {cat['id']: idx for idx, cat in enumerate(categories)}
    new_categories = [{'id': idx, 'name': cat['name']} for idx, cat in enumerate(categories)]

    images = data.get('images', [])
    annotations = data.get('annotations', [])
    
    # Shuffle and split images
    random.seed(42)
    random.shuffle(images)
    
    total_samples = len(images)
    num_val = max(1, int(total_samples * val_split))
    num_test = max(1, int(total_samples * test_split))
    
    val_imgs = set(img['id'] for img in images[:num_val])
    test_imgs = set(img['id'] for img in images[num_val:num_val+num_test])
    
    TARGET_RES = 256
    global_img_id = 0
    ann_id_counter = 0
    
    # Map old_id -> (new_id, split, scale, offsets)
    id_mapping = {}
    
    print(f"Processing {len(images)} images...")
    for img in images:
        old_id = img['id']
        
        if old_id in val_imgs:
            split = 'val'
        elif old_id in test_imgs:
            split = 'test'
        else:
            split = 'train'
            
        # Parse filename to find source
        # Expected format: {folder}_{original_name}
        fname = img['file_name']
        parts = fname.split('_', 1)
        if len(parts) < 2:
            print(f"Warning: Filename {fname} does not match expected format {{folder}}_{{name}}. Skipping.")
            continue
            
        folder_name = parts[0]
        original_name = parts[1]
        
        src_path = img_source_dir / folder_name / original_name
        if not src_path.exists():
            # Fallback: try without folder prefix if it doesn't exist
            src_path = img_source_dir / fname
            if not src_path.exists():
                print(f"Warning: Image {src_path} not found. Skipping.")
                continue
        
        dst_path = img_dest_dir / fname
        
        scale = 1.0
        offsets = (0, 0)
        
        try:
            with Image.open(src_path) as pil_img:
                pil_img = pil_img.convert("RGB")
                processed_img, proc_scale, proc_offsets = preprocess_image(pil_img, target_size=TARGET_RES)
                processed_img.save(dst_path)
                
                # Determine if we need to transform annotations
                # If the JSON says the image is already 256x256, we assume annotations match that.
                # If the JSON says the image is larger, we assume annotations match the original.
                if img.get('width') == TARGET_RES and img.get('height') == TARGET_RES:
                    scale = 1.0
                    offsets = (0, 0)
                else:
                    scale = proc_scale
                    offsets = proc_offsets
                    
        except Exception as e:
            print(f"Error processing {src_path}: {e}")
            continue
            
        new_img = img.copy()
        new_img['id'] = global_img_id
        new_img['width'] = TARGET_RES
        new_img['height'] = TARGET_RES
        
        datasets[split]['images'].append(new_img)
        id_mapping[old_id] = (global_img_id, split, scale, offsets)
        global_img_id += 1

    print(f"Processing {len(annotations)} annotations...")
    for ann in annotations:
        if ann['image_id'] in id_mapping:
            new_img_id, split, scale, (ox, oy) = id_mapping[ann['image_id']]
            
            new_ann = ann.copy()
            new_ann['image_id'] = new_img_id
            new_ann['id'] = ann_id_counter
            
            # Remap category
            if new_ann['category_id'] in cat_id_map:
                new_ann['category_id'] = cat_id_map[new_ann['category_id']]
            else:
                continue
            
            # Transform coordinates if scale != 1.0 or offsets != (0,0)
            if scale != 1.0 or ox != 0 or oy != 0:
                if 'segmentation' in new_ann:
                    new_seg = []
                    for poly in new_ann['segmentation']:
                        new_poly = []
                        for i in range(0, len(poly), 2):
                            x = poly[i]
                            y = poly[i+1]
                            new_poly.append(x * scale + ox)
                            new_poly.append(y * scale + oy)
                        new_seg.append(new_poly)
                    new_ann['segmentation'] = new_seg
                
                if 'bbox' in new_ann and len(new_ann['bbox']) == 4:
                    x, y, w, h = new_ann['bbox']
                    new_ann['bbox'] = [x * scale + ox, y * scale + oy, w * scale, h * scale]

            ann_id_counter += 1
            datasets[split]['annotations'].append(new_ann)

    # Save JSONs
    for split in ['train', 'val', 'test']:
        output = {
            "images": datasets[split]['images'],
            "annotations": datasets[split]['annotations'],
            "categories": new_categories
        }
        out_file = finetune_dataset_dir / f"{split}.json"
        with open(out_file, 'w') as f:
            json.dump(output, f)
        print(f"Saved {split}.json with {len(datasets[split]['images'])} images and {len(datasets[split]['annotations'])} annotations")
        
    print(f"Dataset created at {finetune_dataset_dir}")
    
    return finetune_dataset_dir, len(new_categories)

def run_finetuning(dataset_dir, num_classes):
    base_dir = Path(__file__).resolve().parent.parent
    work_dir = base_dir / 'Raster2Seq'
    
    # Checkpoint path from prompt
    base_ckpt_path = base_dir / 'checkpoints' / 'checkpoint.pth'
    
    if not base_ckpt_path.exists():
        print(f"Error: Checkpoint not found at {base_ckpt_path}")
        return

    # Sanitize base checkpoint
    base_ckpt_path = sanitize_checkpoint(base_ckpt_path)
    
    output_dir = base_dir / 'checkpoints' / 'finetuned_combined'
    
    # Calculate semantic classes (Categories + Background)
    semantic_classes_arg = str(num_classes + 1) if num_classes > 0 else "21"
    print(f"Detected {num_classes} categories. Training with semantic_classes={semantic_classes_arg} (including background).")

    # Use 'cubicasa' as dataset name to ensure correct collate_fn and extras handling
    dataset_name_arg = "cubicasa"

    # Train
    train_cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node=1",
        "--master_port", str(random.randint(10000, 20000)),
        "main_ddp.py",
        "--dataset_name", dataset_name_arg,
        "--dataset_root", str(dataset_dir),
        "--start_from_checkpoint", str(base_ckpt_path), 
        "--output_dir", str(output_dir),
        "--image_size", "256",
        "--batch_size", "4",
        "--epochs", "20",
        "--lr", "1e-5",
        "--lr_backbone", "1e-6",
        "--semantic_classes", semantic_classes_arg,
        "--num_workers", "4",
        "--job_name", "finetune_combined",
        "--num_queries", "2800",
        "--num_polys", "50",
        "--input_channels", "3", 
        "--raster_loss_coef", "0",
        "--cls_loss_coef", "5",
        "--coords_loss_coef", "25",
        "--room_cls_loss_coef", "1",
        "--label_smoothing", "0.3",
        "--jointly_train",
        "--poly2seq",
        "--seq_len", "512",
        "--num_bins", "32",
        "--per_token_sem_loss",
        "--use_anchor",
        "--dec_attn_concat_src",
        "--disable_poly_refine",
        "--ema4eval",
        "--converter_version", "v3"
    ]
    
    print(f"Starting fine-tuning in {work_dir}...")
    print("Command:", " ".join(train_cmd))
    subprocess.run(train_cmd, cwd=work_dir)

    # Evaluate Fine-tuned Checkpoint on Test Set
    print("\nEvaluating Fine-tuned Checkpoint on Test Set...")
    finetuned_ckpt_path = output_dir / "finetune_combined" / "checkpoint.pth"
    
    if not finetuned_ckpt_path.exists():
        print(f"Error: Fine-tuned checkpoint not found at {finetuned_ckpt_path}")
        # Fallback check
        finetuned_ckpt_path = output_dir / "checkpoint.pth"
        if not finetuned_ckpt_path.exists():
             return

    # Sanitize fine-tuned checkpoint before loading it for evaluation
    finetuned_ckpt_path = sanitize_checkpoint(finetuned_ckpt_path)

    eval_ft_cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node=1",
        "--master_port", str(random.randint(10000, 20000)),
        "main_ddp.py",
        "--dataset_name", dataset_name_arg,
        "--dataset_root", str(dataset_dir),
        "--resume", str(finetuned_ckpt_path), 
        "--output_dir", str(output_dir / "eval_finetuned"),
        "--image_size", "256",
        "--batch_size", "4",
        "--semantic_classes", semantic_classes_arg,
        "--num_workers", "4",
        "--eval_only",
        "--eval_set", "test",
        "--job_name", "eval_finetuned",
        "--num_queries", "2800",
        "--num_polys", "50",
        "--poly2seq",
        "--seq_len", "512",
        "--num_bins", "32",
        "--per_token_sem_loss",
        "--use_anchor",
        "--dec_attn_concat_src",
        "--disable_poly_refine",
        "--ema4eval",
        "--converter_version", "v3"
    ]
    subprocess.run(eval_ft_cmd, cwd=work_dir)

    print("\n" + "="*50)
    print("FINE-TUNING COMPLETE")
    print("="*50)
    print(f"New checkpoint saved at: {finetuned_ckpt_path}")

if __name__ == "__main__":
    dataset_dir, num_classes = setup_finetune_dataset(val_split=0.1, test_split=0.1)
    if dataset_dir:
        run_finetuning(dataset_dir, num_classes)
