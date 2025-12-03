# Floor Plan Reconstruction *In-the-Wild*

James Chan (NetID: jbc292)

Helenna Yin (NetID: yy2294)


### Setup
1. First clone the `Raster2Seq` repository. Make sure to download one of the checkpoints, and place it in `./checkpoints/`. At the time of writing, this is currently a private repository; reach out to Hao Phung for more information (htp26 [at] cornell [dot] edu).
2. Run the following script to clone the repo:
```bash
git clone https://github.com/chanjbc/floorplan-reconstruction.git
cd floorplan-reconstruction
```
3. Run the following script to create a conda environment and install all dependencies:
```bash
conda env create -f environment.yml
```

### WAFFLE Dataset
The WAFFLE dataset is available [here](https://tau-vailab.github.io/WAFFLE/). Download the original size images, located in `data/original_size_images.tar.gz`, and extract.

### Generating Initial Predictions
Once you have the Raster2Seq checkpoint downloaded, run the following command, ensuring that paths are set properly:

```bash
python Raster2Seq/predict.py 
    --dataset_root={{PATH_TO_IMAGE}} 
    --checkpoint={{PATH_TO_CHECKPOINT}}
    --output_dir=preds 
    --semantic_classes=12 --input_channels 3 
    --poly2seq 
    --seq_len 512 
    --num_bins 32 
    --disable_poly_refine 
    --dec_attn_concat_src 
    --use_anchor 
    --ema4eval 
    --per_token_sem_loss 
    --drop_wd 
    --save_pred 
    --one_color 
    --image_scale 2 
    --crop_white_space
```

### COCO Serialization
To serialize the checkpoint predictions to the COCO format, run the following command, ensuring all paths are set properly:
```bash
python annotation_preprocess.py
```

### Supervised Fine-Tuning
To apply supervised fine-tuning, run the following script, ensuring all paths are set properly:
```bash
python finetune.py
```