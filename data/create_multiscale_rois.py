#!/usr/bin/env python3
"""
Create multi-scale ROI datasets (3x and 5x) from downloaded full images.

This script processes the already-downloaded full images and creates additional
ROI crops at 3x and 5x the original bounding box size for multi-scale training.
"""

import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def expand_bbox(bbox, scale, image_width, image_height):
    """
    Expand a bounding box by a given scale factor, centered on the original bbox.

    Args:
        bbox: [x, y, width, height] original bounding box
        scale: Scale factor (e.g., 3.0 for 3x larger)
        image_width: Width of the full image
        image_height: Height of the full image

    Returns:
        Expanded bbox [x, y, width, height], clipped to image boundaries
    """
    x, y, w, h = bbox

    # Calculate center of original bbox
    center_x = x + w / 2
    center_y = y + h / 2

    # Calculate new dimensions
    new_w = w * scale
    new_h = h * scale

    # Calculate new top-left corner (centered on original)
    new_x = center_x - new_w / 2
    new_y = center_y - new_h / 2

    # Clip to image boundaries
    new_x = max(0, new_x)
    new_y = max(0, new_y)
    new_w = min(new_w, image_width - new_x)
    new_h = min(new_h, image_height - new_y)

    return [int(new_x), int(new_y), int(new_w), int(new_h)]


def crop_and_save_image(image_path, bbox, output_path):
    """
    Crop an image to the given bounding box and save it.

    Args:
        image_path: Path to the source image
        bbox: [x, y, width, height] bounding box
        output_path: Path to save the cropped image
    """
    with Image.open(image_path) as img:
        x, y, w, h = bbox
        cropped = img.crop((x, y, x + w, y + h))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cropped.save(output_path)


def process_dataset(dataset_json_path, images_dir, output_base_dir, scales):
    """
    Process the dataset and create multi-scale ROIs.

    Args:
        dataset_json_path: Path to the COCO dataset JSON file
        images_dir: Directory containing the full images
        output_base_dir: Base directory for output ROIs
        scales: List of scale factors to process (e.g., [3.0, 5.0])
    """
    # Load dataset
    print(f"Loading dataset from {dataset_json_path}...")
    with open(dataset_json_path, 'r') as f:
        dataset = json.load(f)

    # Create image ID to filename mapping
    image_map = {img['id']: img for img in dataset['images']}

    print(f"Processing {len(dataset['annotations'])} annotations at scales: {scales}")

    # Process each annotation
    for annotation in tqdm(dataset['annotations'], desc="Creating multi-scale ROIs"):
        image_id = annotation['image_id']
        annotation_id = annotation['id']
        bbox = annotation['bbox']  # [x, y, width, height]

        # Get image info
        image_info = image_map[image_id]
        image_width = image_info['width']
        image_height = image_info['height']

        # Construct image path (same logic as download.py)
        url_extension = image_info['coco_url'].split('.')[-1]
        image_filename = f"{image_id}.{url_extension}"
        image_path = images_dir / image_filename

        if not image_path.exists():
            print(f"Warning: Image not found: {image_path}")
            continue

        # Create ROIs at each scale
        for scale in scales:
            # Expand bbox
            scaled_bbox = expand_bbox(bbox, scale, image_width, image_height)

            # Create output path
            scale_dir = output_base_dir / f"{int(scale)}x"
            roi_filename = f"{image_id}_{annotation_id}.png"
            roi_path = scale_dir / roi_filename

            # Crop and save
            try:
                crop_and_save_image(image_path, scaled_bbox, roi_path)
            except Exception as e:
                print(f"Error processing {roi_path}: {e}")

    print(f"\nCompleted! Created ROIs at scales: {scales}")
    for scale in scales:
        scale_dir = output_base_dir / f"{int(scale)}x"
        num_files = len(list(scale_dir.glob("*.png")))
        print(f"  {int(scale)}x: {num_files} ROIs")


def main():
    parser = argparse.ArgumentParser(
        description="Create multi-scale ROI datasets from full images"
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=Path("data/dataset_train.json"),
        help="Path to the COCO dataset JSON file"
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("train/images"),
        help="Directory containing full images"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("train/rois"),
        help="Base directory for output ROIs"
    )
    parser.add_argument(
        "--scales",
        type=float,
        nargs='+',
        default=[3.0, 5.0],
        help="Scale factors to create (default: 3.0 5.0)"
    )

    args = parser.parse_args()

    # Verify inputs exist
    if not args.dataset_json.exists():
        print(f"Error: Dataset JSON not found: {args.dataset_json}")
        return 1

    if not args.images_dir.exists():
        print(f"Error: Images directory not found: {args.images_dir}")
        return 1

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Process
    process_dataset(
        args.dataset_json,
        args.images_dir,
        args.output_dir,
        args.scales
    )

    return 0


if __name__ == "__main__":
    exit(main())
