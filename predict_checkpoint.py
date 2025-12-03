import sys
import subprocess
from pathlib import Path
import os

def run_prediction():
    # Define paths
    current_dir = Path(__file__).resolve().parent
    base_dir = current_dir.parent
    work_dir = base_dir 
    
    # Images are in finetune_dataset/images
    dataset_dir = base_dir / 'finetune_dataset' / 'images'
    output_dir = base_dir / 'finetune' / 'predictions'
    
    # Locate checkpoint (prefer sanitized version if available)
    ckpt_dir = base_dir  / 'checkpoints' / 'finetune' / "checkpoint.pth"
    ckpt_path = ckpt_dir / 'checkpoint_sanitized.pth'
    if not ckpt_path.exists():
        ckpt_path = ckpt_dir / 'checkpoint.pth'
    
    if not ckpt_path.exists():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        print("Make sure training has finished successfully.")
        return

    # Semantic classes: 1 room label + 1 background = 2
    semantic_classes = "2" 

    print(f"Running inference on {dataset_dir}...")
    print(f"Using checkpoint: {ckpt_path}")
    
    cmd = [
        sys.executable, "Raster2Seq/predict.py",
        "--dataset_root", str(dataset_dir),
        "--checkpoint", str(ckpt_path),
        "--output_dir", str(output_dir),
        "--semantic_classes", semantic_classes,
        "--input_channels", "3",
        "--poly2seq",
        "--seq_len", "512",
        "--num_bins", "32",
        "--disable_poly_refine",
        "--dec_attn_concat_src",
        "--use_anchor",
        "--ema4eval",
        "--per_token_sem_loss",
        "--drop_wd",
        "--save_pred",
        "--one_color",
        "--image_scale", "1", # Use 1 for pre-resized images in finetune_dataset
        "--crop_white_space"
    ]
    
    print("Command:", " ".join(cmd))
    subprocess.run(cmd, cwd=work_dir)

if __name__ == "__main__":
    run_prediction()
