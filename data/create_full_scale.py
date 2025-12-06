#!/usr/bin/env python3
"""
Create full-image scale crops for multi-scale model.
For each annotation, the full image is resized to 224x224.
"""

import json
import os
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import argparse


def create_full_scale_crops(coco_json: str, image_dir: str, output_dir: str, img_size: int = 224):
    """Create full-image crops for all annotations."""

    with open(coco_json) as f:
        data = json.load(f)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Build image lookup
    images = {}
    for img in data["images"]:
        img_id = img["id"]
        # Try different extensions
        for ext in [".png", ".jpg", ".jpeg"]:
            img_path = os.path.join(image_dir, f"{img_id}{ext}")
            if os.path.exists(img_path):
                images[img_id] = img_path
                break

    print(f"Found {len(images)} images")
    print(f"Total annotations: {len(data['annotations'])}")

    # Process each annotation
    skipped = 0
    for ann in tqdm(data["annotations"], desc="Creating full-scale crops"):
        img_id = ann["image_id"]
        ann_id = ann["id"]

        if img_id not in images:
            skipped += 1
            continue

        img_path = images[img_id]
        output_path = os.path.join(output_dir, f"{img_id}_{ann_id}.png")

        # Skip if already exists
        if os.path.exists(output_path):
            continue

        try:
            img = Image.open(img_path).convert("RGB")
            # Resize full image to target size
            img_resized = img.resize((img_size, img_size), Image.LANCZOS)
            img_resized.save(output_path)
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            skipped += 1

    print(f"Done! Skipped {skipped} annotations")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-json", required=True, help="COCO annotation JSON")
    parser.add_argument("--image-dir", required=True, help="Directory with source images")
    parser.add_argument("--output-dir", required=True, help="Output directory for full crops")
    parser.add_argument("--img-size", type=int, default=224, help="Output image size")
    args = parser.parse_args()

    create_full_scale_crops(args.coco_json, args.image_dir, args.output_dir, args.img_size)


if __name__ == "__main__":
    main()
