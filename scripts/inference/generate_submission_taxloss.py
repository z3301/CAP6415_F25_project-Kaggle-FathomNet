#!/usr/bin/env python3
"""
Generate Kaggle submission for FathomNet 2025 using taxonomic loss model.
"""

import argparse
import json
import os
import sys

import pandas as pd
import torch
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from data.data import load_and_encode_taxonomy
from src.models.model_multiscale_taxloss import MultiScaleTaxonomicClassifier


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
        required=True,
        help="Path to model checkpoint",
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
        default=["1x", "3x", "5x"],
        help="Scales to use",
    )
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    args = parser.parse_args()

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
        num_workers=4,
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
            "concept": concept_names,  # Kaggle expects 'concept', not 'concept_name'
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
    for name, count in submission_df["concept"].value_counts().head(10).items():
        print(f"  {name}: {count} ({count/len(submission_df)*100:.1f}%)")


if __name__ == "__main__":
    main()
