import json
import os
import zipfile
import io
from math import ceil
from PIL import Image

# TODO: Change configuration
IMAGES_DIR = "./original_size_images/" # Directory containing original, full-sized images
JSONS_DIR = "./predictions/"
OUTPUT_DIR = "./" 
BATCH_SIZE = 800                
TARGET_SIZE = 256               # Target dimension (256x256)
PAD_VALUE = (255, 255, 255)     # White padding (RGB)

def calculate_polygon_area(coords):
    """Calculates area using the Shoelace formula."""
    x = coords[0::2]
    y = coords[1::2]
    return 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(len(x)-1)) + x[-1] * y[0] - x[0] * y[-1])

def get_bbox(coords):
    """Calculates bounding box [min_x, min_y, w, h]."""
    x = coords[0::2]
    y = coords[1::2]
    if not x or not y: return [0,0,0,0]
    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)
    return [min_x, min_y, max_x - min_x, max_y - min_y]

def preprocess_image(img, target_size, pad_value):
    """
    Resizes image to fit within target_size x target_size.
    Returns: processed_image
    """
    w, h = img.size
    
    # Calculate scale to fit the longest edge into target_size
    scale = min(target_size / w, target_size / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    # Resize the original image
    # Use BILINEAR to match the training/inference pipeline (Detectron2 defaults)
    resized_img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)
    
    # Create a new blank background (square)
    new_img = Image.new("RGB", (target_size, target_size), pad_value)
    
    # Calculate centering offsets
    x_offset = (target_size - new_w) // 2
    y_offset = (target_size - new_h) // 2
    
    # Paste the resized image at the calculated offsets (Centered)
    new_img.paste(resized_img, (x_offset, y_offset))
    
    return new_img

def get_base_coco_info(batch_num):
    return {
        "info": {
            "description": f"Floorplan Batch {batch_num} (Img 256, Orig Labels)",
            "year": 2024,
            "version": "1.0"
        },
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "segmentation", "supercategory": "room"}]
    }

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    print("Scanning directories...")
    all_files = os.listdir(IMAGES_DIR)
    
    # Filter logic
    valid_images = [
        f for f in all_files 
        if f.lower().endswith('.jpg') 
        and '_pred_floorplan' not in f 
        and '_pred_room_map' not in f
        and 'Zone.Identifier' not in f
    ]
    valid_images.sort()
    
    total_images = len(valid_images)
    num_batches = ceil(total_images / BATCH_SIZE)
    
    print(f"Found {total_images} valid images. Splitting into {num_batches} batches.")

    global_ann_id = 1 

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min((batch_idx + 1) * BATCH_SIZE, total_images)
        current_batch_files = valid_images[start_idx:end_idx]
        
        batch_name = f"batch_{batch_idx+1}"
        json_filename = os.path.join(OUTPUT_DIR, f"{batch_name}.json")
        zip_filename = os.path.join(OUTPUT_DIR, f"{batch_name}.zip")
        
        coco_data = get_base_coco_info(batch_idx + 1)
        
        print(f"Processing Batch {batch_idx+1}/{num_batches} ({len(current_batch_files)} images)...")

        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for img_filename in current_batch_files:
                img_path = os.path.join(IMAGES_DIR, img_filename)
                name_no_ext = os.path.splitext(img_filename)[0]
                
                # Process Image
                try:
                    with Image.open(img_path) as img:
                        img = img.convert("RGB")
                        
                        # Preprocess image (resize + pad) but IGNORE scale factor
                        processed_img = preprocess_image(img, TARGET_SIZE, PAD_VALUE)
                        
                        # Save processed image to zip
                        img_buffer = io.BytesIO()
                        processed_img.save(img_buffer, format="PNG")
                        zipf.writestr(img_filename, img_buffer.getvalue())

                except Exception as e:
                    print(f"  Error processing {img_filename}: {e}")
                    continue

                # --- 2. CREATE COCO IMAGE ENTRY 
                try:
                    image_id = int(name_no_ext)
                except ValueError:
                    image_id = abs(hash(name_no_ext)) % (10 ** 8)

                # Note: We record the NEW dimensions (256x256) in the COCO JSON
                # even though the labels are for the OLD dimensions.
                coco_data["images"].append({
                    "id": image_id,
                    "file_name": img_filename, 
                    "width": TARGET_SIZE, 
                    "height": TARGET_SIZE
                })

                # Process annotations
                json_path = os.path.join(JSONS_DIR, f"{name_no_ext}.json")
                if os.path.exists(json_path):
                    with open(json_path, 'r') as f:
                        raw_anns = json.load(f)
                    
                    for ann in raw_anns:
                        raw_seg = ann.get('segmentation', [])
                        if not raw_seg or len(raw_seg) < 3: continue

                        # Flatten coordinates WITHOUT scaling
                        flattened_seg = [c for pt in raw_seg for c in pt]
                        
                        coco_data["annotations"].append({
                            "id": global_ann_id,
                            "image_id": image_id,
                            "category_id": 1,
                            "segmentation": [flattened_seg],
                            "area": calculate_polygon_area(flattened_seg),
                            "bbox": get_bbox(flattened_seg),
                            "iscrowd": 0
                        })
                        global_ann_id += 1

        with open(json_filename, 'w') as f:
            json.dump(coco_data, f)
            
    print(f"\nFinished processing {total_images} images.")
    print(f"Output files located in: {os.path.abspath(OUTPUT_DIR)}")

if __name__ == "__main__":
    main()