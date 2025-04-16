import numpy as np
import os
import subprocess
import json
from shutil import move
from glob import glob
import numpy as np
from pycocotools import mask as maskUtils
import logging
from PIL import Image
import argparse
from tqdm import tqdm


os.makedirs("coco_sample", exist_ok=True)
os.chdir("coco_sample")

# Download 2017 training images (~500MB for all, filter as needed later)
subprocess.run(["wget", "http://images.cocodataset.org/zips/train2017.zip"], check=True)
subprocess.run(["unzip", "-q", "train2017.zip"], check=True)

# Download full annotations
subprocess.run(["wget", "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"], check=True)
subprocess.run(["unzip", "-q", "annotations_trainval2017.zip"], check=True)

os.makedirs("mini_train2017", exist_ok=True)
image_list = sorted(glob("train2017/*.jpg"))[:100]  # Take first 100 images

for img_path in image_list:
    move(img_path, "mini_train2017/")




# Load full annotations
with open("annotations/instances_train2017.json", "r") as f:
    data = json.load(f)

# Keep only images we selected
mini_image_ids = set([int(os.path.basename(f).split('.')[0]) for f in image_list])
mini_images = [img for img in data['images'] if img['id'] in mini_image_ids]
mini_annotations = [ann for ann in data['annotations'] if ann['image_id'] in mini_image_ids]
mini_categories = data['categories']  # keep all classes

# Save new mini annotation file
mini_data = {
    "images": mini_images,
    "annotations": mini_annotations,
    "categories": mini_categories
}

with open("instances_mini_train2017.json", "w") as f:
    json.dump(mini_data, f)



def load_coco_annotations(annotation_file):
    """
    Loads COCO-style annotations from a JSON file.

    Args:
        annotation_file (str): Path to the COCO annotations file (.json).

    Returns:
        dict: Dictionary containing images, annotations, and categories.
    """
    if not os.path.exists(annotation_file):
        raise FileNotFoundError(f"Annotation file not found: {annotation_file}")

    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    # Basic sanity checks for key fields
    required_keys = ["images", "annotations", "categories"]
    for key in required_keys:
        if key not in coco_data:
            raise ValueError(f"Missing key '{key}' in COCO annotations.")

    return coco_data


def generate_mask_from_annotations(image_info, annotations, category_map, image_size):
    

    height, width = image_size
    mask = np.zeros((height, width), dtype=np.uint8)

    for ann in annotations:
        try:
            category_id = ann["category_id"]
            class_index = category_map.get(category_id, 0)

            # Handle RLE and polygon differently
            if ann.get("iscrowd", 0):
                rle = ann["segmentation"]
                if isinstance(rle, list):  # sometimes malformed
                    continue
                binary_mask = maskUtils.decode(rle)
            else:
                rle = maskUtils.frPyObjects(ann["segmentation"], height, width)
                binary_mask = maskUtils.decode(rle)

            # Handle masks with multiple instances (H, W, N)
            if binary_mask.ndim == 3:
                binary_mask = np.max(binary_mask, axis=2)

            binary_mask = binary_mask.squeeze()

            # Apply class index where mask == 1
            mask = np.where(binary_mask == 1, class_index, mask)

        except Exception as e:
            logging.warning(f"Annotation skipped due to error: {e}")

    return mask



def save_mask_and_image(mask, image_path, output_dir, file_name):
    """
    Saves the generated mask as a PNG and copies the original image to the output folder.

    Args:
        mask (np.ndarray): Segmentation mask with class indices.
        image_path (str): Path to the original image file.
        output_dir (str): Root output directory (e.g., "./processed_dataset/").
        file_name (str): Image file name (e.g., "000000000009.jpg").

    Saves:
        ./processed_dataset/images/file_name
        ./processed_dataset/masks/file_name (as .png)
    """
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'masks'), exist_ok=True)

    # Save original image
    image_dest = os.path.join(output_dir, 'images', file_name)
    if not os.path.exists(image_dest):
        os.system(f'cp "{image_path}" "{image_dest}"')

    # Save mask as PNG with same name
    mask_img = Image.fromarray(mask.astype(np.uint8))
    mask_dest = os.path.join(output_dir, 'masks', file_name.replace('.jpg', '.png'))
    mask_img.save(mask_dest)




def main():
    parser = argparse.ArgumentParser(description="COCO to Segmentation Mask Converter")
    parser.add_argument('--images_dir', required=True, help='Path to folder containing images')
    parser.add_argument('--annotation_file', required=True, help='Path to COCO-style annotation .json file')
    parser.add_argument('--output_dir', default='./processed_dataset', help='Path to save processed masks and images')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Load annotation file
    coco_data = load_coco_annotations(args.annotation_file)
    image_id_to_anns = {}
    for ann in coco_data['annotations']:
        image_id_to_anns.setdefault(ann['image_id'], []).append(ann)

    # Build category_id → class_index map
    category_map = {cat['id']: idx + 1 for idx, cat in enumerate(coco_data['categories'])}  # 0 is background

    for image_info in tqdm(coco_data['images'], desc="Processing images"):
        try:
            image_id = image_info['id']
            file_name = image_info['file_name']
            image_path = os.path.join(args.images_dir, file_name)
            height, width = image_info['height'], image_info['width']
            anns = image_id_to_anns.get(image_id, [])

            mask = generate_mask_from_annotations(image_info, anns, category_map, (height, width))
            save_mask_and_image(mask, image_path, args.output_dir, file_name)

        except Exception as e:
            logging.warning(f"Failed to process {file_name}: {e}")


def run_pipeline(images_dir, annotation_file, output_dir):
    """
    Runs the full segmentation preprocessing pipeline for a COCO-style dataset in Colab.

    Args:
        images_dir (str): Directory containing input images.
        annotation_file (str): Path to COCO-style annotation file (.json).
        output_dir (str): Output directory for processed masks and copied images.
    """

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    coco_data = load_coco_annotations(annotation_file)

    image_id_to_anns = {}
    for ann in coco_data['annotations']:
        image_id_to_anns.setdefault(ann['image_id'], []).append(ann)

    category_map = {cat['id']: idx + 1 for idx, cat in enumerate(coco_data['categories'])}

    for image_info in tqdm(coco_data['images'], desc="Processing images"):
        try:
            image_id = image_info['id']
            file_name = image_info['file_name']
            image_path = os.path.join(images_dir, file_name)
            height, width = image_info['height'], image_info['width']
            anns = image_id_to_anns.get(image_id, [])

            mask = generate_mask_from_annotations(image_info, anns, category_map, (height, width))
            save_mask_and_image(mask, image_path, output_dir, file_name)

        except Exception as e:
            logging.warning(f"Failed to process {file_name}: {e}")

run_pipeline(
    images_dir="/content/coco_sample/mini_train2017",
    annotation_file="/content/coco_sample/instances_mini_train2017.json",
    output_dir="/content/processed_dataset"
)

with open("/content/coco_sample/instances_mini_train2017.json", "r") as f:
    coco_data = json.load(f)

category_map = {cat['id']: idx + 1 for idx, cat in enumerate(coco_data['categories'])}
print(f"Number of classes (including background): {len(category_map) + 1}")


from PIL import Image
import matplotlib.pyplot as plt

img = Image.open("/content/processed_dataset/images/000000000009.jpg")
mask = Image.open("/content/processed_dataset/masks/000000000009.png")

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img)
axs[0].set_title("Original Image")
axs[1].imshow(mask, cmap='gray')
axs[1].set_title("Segmentation Mask")
plt.show()
