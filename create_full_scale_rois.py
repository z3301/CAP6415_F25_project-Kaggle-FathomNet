#!/usr/bin/env python3
"""
Create full-image scale crops for multi-scale model.
For each annotation, the full image is resized to 224x224.
"""

import json
import os
from PIL import Image
from tqdm import tqdm
import argparse


def create_full_scale_crops(coco_json: str, image_dir: str, output_dir: str, img_size: int = 224):
    """Create full-image crops for all annotations."""

    with open(coco_json) as f:
        data = json.load(f)

    os.makedirs(output_dir, exist_ok=True)

    # Build image lookup - try multiple extensions
    images = {}
    for img in data["images"]:
        img_id = img["id"]
        for ext in [".png", ".jpg", ".jpeg"]:
            img_path = os.path.join(image_dir, f"{img_id}{ext}")
            if os.path.exists(img_path):
                images[img_id] = img_path
                break

    print(f"Found {len(images)} source images")
    print(f"Total annotations: {len(data['annotations'])}")

    # Cache resized images to avoid re-reading
    resized_cache = {}

    skipped = 0
    created = 0
    for ann in tqdm(data["annotations"], desc="Creating full-scale crops"):
        img_id = ann["image_id"]
        ann_id = ann["id"]

        if img_id not in images:
            skipped += 1
            continue

        output_path = os.path.join(output_dir, f"{img_id}_{ann_id}.png")

        # Skip if already exists
        if os.path.exists(output_path):
            continue

        try:
            # Use cached resized image if available
            if img_id not in resized_cache:
                img = Image.open(images[img_id]).convert("RGB")
                resized_cache[img_id] = img.resize((img_size, img_size), Image.LANCZOS)

            resized_cache[img_id].save(output_path)
            created += 1

            # Clear cache periodically to manage memory
            if len(resized_cache) > 100:
                resized_cache.clear()

        except Exception as e:
            print(f"Error processing {images.get(img_id, 'unknown')}: {e}")
            skipped += 1

    print(f"Done! Created {created}, skipped {skipped}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-json", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--img-size", type=int, default=224)
    args = parser.parse_args()

    create_full_scale_crops(args.coco_json, args.image_dir, args.output_dir, args.img_size)


if __name__ == "__main__":
    main()
