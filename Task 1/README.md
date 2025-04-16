# COCO to Segmentation Mask Converter

This script processes COCO-style datasets to generate binary or multi-class masks suitable for PyTorch training.

## Instructions
1. Install dependencies using `uv` or 'pip', the dependencies are provided in the requirements.txt.
2. Run the script with paths to images and annotations.
3. Masks are saved as `.png` in `processed_dataset/masks/` with class indices.

## Notes
- Handles polygons, RLE, overlaps, and missing entries.
- Output structure is clean and reproducible.

