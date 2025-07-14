# Water Body Detection Using GeoAI(Deep Learning) with Sentinel-2 Satellite Imagery

## Overview
This project focuses on semantic segmentation of surface water bodies in Sri Lanka (selected ROI: Maduru Oya Reservoir and surrounding) using high-resolution Sentinel-2 satellite imagery.The complete pipeline integrates satellite image preprocessing, mask generation, image tiling, model training, and result visualization.

## Objective
- To demonstrate how a deep learning-based approach,specifically using a U-Net architecture,can effectively detect and delineate water features by leveraging spectral indices such as the Normalized Difference Water Index (NDWI).

### Side-by-side visualization of water body segmentation
<img width="1870" height="674" alt="Screenshot (254)" src="https://github.com/user-attachments/assets/5b1e5519-062c-45d7-af17-b24cdb8dbc15" />


### Performance metrics of the U-Net model during training
<img width="1920" height="722" alt="Screenshot (256)" src="https://github.com/user-attachments/assets/0ceaeef5-fc57-4304-b78c-2f107fdd110a" />

1. The first panel shows the training and validation loss, indicating the model’s learning dynamics. 
2. The second panel displays the Intersection over Union (IoU), assessing segmentation accuracy.
3. The third panel presents the Dice score, which reflects the model’s ability to detect water bodies accurately, even in cases of class imbalance or sparse features.


## Pipeline
1. Generate cloud-free Sentinel-2 composites over selected areas of Sri Lanka using Google Earth Engine (GEE).
2. Derive water masks using NDWI as ground truth labels for training.
3. Prepare the dataset by tiling the composite and mask images into smaller patches suitable for training.
4. Train a U-Net segmentation model using the geoai Python library with a ResNet34 encoder.
5. Run inference on unseen image tiles and visualize predictions against ground truth and original images.


## Model Training
- Use geoai.train_segmentation_model() with:
    - architecture = "unet"
    - encoder = "resnet34"
    - 6 input channels (selected Sentinel-2 bands)
    - 2 output classes (water and non-water)

- Training can be done on Google Colab with GPU acceleration.


## Notes
1. Adjust the NDWI threshold depending on the seasonality and cloud coverage.
2. Ensure balanced classes during tiling to avoid generating only non-water tiles.
3. This method is scalable and can be adapted to different regions by changing the region of interest (ROI) in GEE.








