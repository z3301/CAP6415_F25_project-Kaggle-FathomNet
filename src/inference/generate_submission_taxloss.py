#!/usr/bin/env python3
"""
Generate Kaggle submission for FathomNet 2025 using taxonomic loss model.
"""

import argparse
import json
import os
import subprocess

import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Load environment variables from .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, rely on environment variables

from data.data import load_and_encode_taxonomy
from src.models.model_multiscale_taxloss import MultiScaleTaxonomicClassifier


def get_kaggle_env():
    """Get Kaggle credentials from environment."""
    kaggle_username = os.environ.get("KAGGLE_USERNAME")
    kaggle_key = os.environ.get("KAGGLE_KEY") or os.environ.get("KAGGLE_API_TOKEN")
    if kaggle_username and kaggle_key:
        return {**os.environ, "KAGGLE_USERNAME": kaggle_username, "KAGGLE_KEY": kaggle_key}
    return None


def get_latest_submission_score(env, max_wait: int = 60, poll_interval: int = 5):
    """Poll Kaggle for the latest submission score (private LB)."""
    import re
    import time

    print(f"\nWaiting for score (up to {max_wait}s)...", end="", flush=True)

    for _ in range(max_wait // poll_interval):
        time.sleep(poll_interval)
        print(".", end="", flush=True)

        try:
            result = subprocess.run(
                ["kaggle", "competitions", "submissions", "-c", "fathomnet-2025"],
                capture_output=True,
                text=True,
                env=env
            )

            if result.returncode == 0:
                lines = result.stdout.strip().split("\n")
                # Skip header and separator lines, get first data line
                for line in lines[2:]:
                    if not line.strip() or line.startswith("-"):
                        continue
                    # Parse whitespace-separated format
                    # Format: fileName  date  description  status  publicScore  privateScore
                    parts = re.split(r'\s{2,}', line.strip())
                    if len(parts) >= 4:
                        status = parts[3].strip() if len(parts) > 3 else ""
                        private_score = parts[-1].strip() if len(parts) > 4 else ""

                        if "COMPLETE" in status.upper() and private_score:
                            print(f"\n\n{'='*50}")
                            print(f"✓ KAGGLE PRIVATE SCORE: {private_score}")
                            print(f"{'='*50}")
                            return private_score
                        elif "ERROR" in status.upper():
                            print(f"\n✗ Submission error")
                            return None
                    break  # Only check the latest submission
        except Exception as e:
            pass

    print("\n⚠ Timed out waiting for score. Check Kaggle manually.")
    return None


def submit_to_kaggle(submission_file: str, message: str = "4-scale taxloss submission", wait_for_score: bool = True):
    """Submit to Kaggle using credentials from environment."""
    env = get_kaggle_env()

    if not env:
        print("\n⚠ Kaggle credentials not found in environment.")
        print("  Set KAGGLE_USERNAME and KAGGLE_KEY (or KAGGLE_API_TOKEN) in .env or environment.")
        return False

    kaggle_username = os.environ.get("KAGGLE_USERNAME")
    print(f"\nSubmitting to Kaggle as {kaggle_username}...")

    try:
        result = subprocess.run(
            [
                "kaggle", "competitions", "submit",
                "-c", "fathomnet-2025",
                "-f", submission_file,
                "-m", message
            ],
            capture_output=True,
            text=True,
            env=env
        )

        if result.returncode == 0:
            print("✓ Submission successful!")
            if result.stdout.strip():
                print(result.stdout)

            # Wait for and display score
            if wait_for_score:
                get_latest_submission_score(env)

            return True
        else:
            print(f"✗ Submission failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("✗ Kaggle CLI not found. Install with: pip install kaggle")
        return False


class MultiScaleTestDataset(Dataset):
    """Test dataset that loads multi-scale pre-cropped ROIs."""

    def __init__(
        self,
        coco_json_path: str,
        roi_root: str,
        scales: list,
        transform=None,
    ):
        with open(coco_json_path, "r") as f:
            coco_data = json.load(f)

        self.annotations = coco_data["annotations"]
        self.images = {img["id"]: img for img in coco_data["images"]}
        self.roi_root = roi_root
        self.scales = scales
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_id = ann["image_id"]
        annotation_id = ann["id"]

        # Load multi-scale images
        images = {}
        for scale in self.scales:
            scale_dir = os.path.join(self.roi_root, scale)
            img_path = os.path.join(scale_dir, f"{image_id}_{annotation_id}.png")

            if os.path.exists(img_path):
                img = Image.open(img_path).convert("RGB")
            else:
                # Fallback: create black image
                img = Image.new("RGB", (224, 224), (0, 0, 0))

            if self.transform:
                img = self.transform(img)
            images[scale] = img

        return images, annotation_id


def main():
    parser = argparse.ArgumentParser(description="Generate taxloss submission")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/multiscale_4scales_taxloss/checkpoints/best-epoch=02-val_tax_score=0.531.ckpt",
        help="Path to model checkpoint (default: 4-scale taxloss best checkpoint)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/experiment-multiscale.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--distance-matrix",
        type=str,
        default="data/distance_matrix.csv",
        help="Path to distance matrix",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="submission_taxloss.csv",
        help="Output submission file",
    )
    parser.add_argument(
        "--scales",
        type=str,
        nargs="+",
        default=["1x", "3x", "5x", "full"],
        help="Scales to use (default: 1x 3x 5x full for best results)",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument(
        "--submit",
        action="store_true",
        default=True,
        help="Submit to Kaggle after generating (default: True)",
    )
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Don't submit to Kaggle, just generate CSV",
    )
    parser.add_argument(
        "--message",
        type=str,
        default="4-scale multiscale + taxonomic distance loss",
        help="Submission message for Kaggle",
    )
    args = parser.parse_args()

    # Handle --no-submit flag
    if args.no_submit:
        args.submit = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    cfg = OmegaConf.load(args.config)

    # Load taxonomy to get id_to_name mappings
    print("Loading taxonomy...")
    taxonomy_df, encoders, class_counts, id_to_name = load_and_encode_taxonomy(
        cfg.paths.taxonomy_csv,
        list(cfg.data.taxonomy_levels),
    )
    print(f"Class counts: {class_counts}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = MultiScaleTaxonomicClassifier.load_from_checkpoint(
        args.checkpoint,
        cfg=cfg,
        class_counts=class_counts,
        id_to_name=id_to_name,
        scales=args.scales,
        distance_matrix_path=args.distance_matrix,
        strict=False,
        map_location=device,  # Load directly to CPU if no GPU available
    )
    model.to(device)
    model.eval()

    # Create test transform (no augmentation)
    test_transform = transforms.Compose(
        [
            transforms.Resize((cfg.data.img_size, cfg.data.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # Create test dataset
    print("Loading test data...")
    test_dataset = MultiScaleTestDataset(
        coco_json_path=cfg.paths.test_coco_json,
        roi_root=cfg.paths.test_roi_root,
        scales=args.scales,
        transform=test_transform,
    )
    print(f"Test samples: {len(test_dataset)}")

    def collate_fn(batch):
        images_list, annotation_ids = zip(*batch)
        # Stack images for each scale
        images = {}
        for scale in args.scales:
            images[scale] = torch.stack([img[scale] for img in images_list])
        return images, list(annotation_ids)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Avoid multiprocessing pickle issues
        collate_fn=collate_fn,
    )

    # Run inference
    print("Running inference...")
    all_predictions = []
    all_annotation_ids = []

    with torch.no_grad():
        for images, annotation_ids in tqdm(test_loader, desc="Predicting"):
            # Move images to device
            images = {k: v.to(device) for k, v in images.items()}

            outputs = model(images)

            # Get species predictions
            species_logits = outputs["species"]
            species_preds = torch.argmax(species_logits, dim=1)

            all_predictions.extend(species_preds.cpu().numpy())
            all_annotation_ids.extend(annotation_ids)

    # Convert predictions to concept names
    concept_names = [id_to_name["species"][pred] for pred in all_predictions]

    # Create submission DataFrame
    submission_df = pd.DataFrame(
        {
            "annotation_id": all_annotation_ids,
            "concept_name": concept_names,
        }
    )

    # Sort by annotation_id
    submission_df = submission_df.sort_values("annotation_id")

    # Save submission
    submission_df.to_csv(args.output, index=False)
    print(f"\nSubmission saved to {args.output}")
    print(f"Total predictions: {len(submission_df)}")

    # Show distribution
    print("\nPrediction distribution (top 10):")
    for name, count in submission_df["concept_name"].value_counts().head(10).items():
        print(f"  {name}: {count} ({count/len(submission_df)*100:.1f}%)")

    # Submit to Kaggle if requested
    if args.submit:
        submit_to_kaggle(args.output, args.message)


if __name__ == "__main__":
    main()
