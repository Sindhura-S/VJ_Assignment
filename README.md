Train IoU=0.0295 | Val IoU=0.0214

Collab link: https://colab.research.google.com/drive/1UUj8YbY6gXgYe0mtIUH0NdV6cZTf0gJT?usp=sharing

# Image Segmentation Assignment (COCO-based)

This repo contains Task 1 and Task 2 of an image segmentation pipeline:

## 🔧 Task 1: Dataset Preprocessing
- Converts COCO annotations to PNG masks
- Handles edge cases (RLE, crowd masks, invalid coords)


## 🏗️ Task 2: Model Training
- Custom UNet in PyTorch
- Metrics: IoU, Accuracy (torchmetrics)
- Logging via TensorBoard

