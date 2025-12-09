#!/usr/bin/env python3
"""
Generate Kaggle submission for FathomNet 2025 using taxonomic loss model.
"""

import argparse
import json
import os

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

# Map alternative Kaggle env var names BEFORE importing kaggle
# The Kaggle API reads credentials at import time
kaggle_token = os.environ.get("KAGGLE_API_TOKEN", "")
if kaggle_token:
    os.environ["KAGGLE_KEY"] = kaggle_token

from data.data import load_and_encode_taxonomy
from src.models.model_multiscale_taxloss import MultiScaleTaxonomicClassifier


def get_submission_score(max_wait: int = 60, poll_interval: int = 5):
    """Poll Kaggle for the latest submission score."""
    import time

    print(f"\nWaiting for score (up to {max_wait}s)...", end="", flush=True)

    for _ in range(max_wait // poll_interval):
        time.sleep(poll_interval)
        print(".", end="", flush=True)

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.authenticate()
            subs = api.competition_submissions("fathomnet-2025")

            if subs:
                s = subs[0]
                status = str(s.status).lower()
                if "complete" in status:
                    score = s.private_score
                    if score is not None:
                        print(f"\n\n{'='*50}")
                        print(f"KAGGLE SCORE: {score}")
                        print(f"{'='*50}")
                        return score
                elif "error" in status:
                    print(f"\nSubmission error")
                    return None
        except Exception as e:
            print(f"\n(error: {e})")
            return None

    print("\nTimed out.")
    return None


def submit_to_kaggle(submission_file: str, message: str = "4-scale taxloss submission"):
    """Submit to Kaggle."""
    import shutil
    import subprocess

    kaggle_cmd = shutil.which("kaggle")
    if not kaggle_cmd:
        print(f"\nKaggle CLI not found. Submit manually:")
        print(f"  kaggle competitions submit -c fathomnet-2025 -f {submission_file} -m \"{message}\"")
        return False

    print(f"\nSubmitting to Kaggle...")

    try:
        result = subprocess.run(
            [kaggle_cmd, "competitions", "submit", "-c", "fathomnet-2025", "-f", submission_file, "-m", message],
            capture_output=True,
            text=True,
        )

        if result.returncode == 0:
            print("Submission successful!")
            if result.stdout.strip():
                print(result.stdout.strip())
            get_submission_score()
            return True
        else:
            print(f"Submission failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"Submission failed: {e}")
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

    # Select best available device: CUDA > MPS (Apple Silicon) > CPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
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
