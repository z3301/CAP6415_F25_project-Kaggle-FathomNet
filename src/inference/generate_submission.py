#!/usr/bin/env python3
"""
Generate Kaggle submission for FathomNet 2025 Competition
"""

import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path

from src.config import Config
from src.data import create_transforms
from src.model import TaxonomyAwareClassifier


def main():
    parser = argparse.ArgumentParser(description='Generate Kaggle submission')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='submission.csv',
                        help='Output submission file path')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load taxonomy
    print("Loading taxonomy...")
    taxonomy_df = pd.read_csv(Config.TAXONOMY_PATH)

    # Load class counts from checkpoint (they may differ from taxonomy.csv)
    print(f"Loading checkpoint to get class counts...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')

    # Extract class counts from checkpoint hparams
    if 'hyper_parameters' in checkpoint and 'class_counts' in checkpoint['hyper_parameters']:
        class_counts = checkpoint['hyper_parameters']['class_counts']
        print(f"Class counts from checkpoint: {class_counts}")
    else:
        # Infer from model weights
        state_dict = checkpoint['state_dict']
        class_counts = {
            'kingdom': state_dict['kingdom_head.weight'].shape[0],
            'phylum': state_dict['phylum_head.weight'].shape[0],
            'class': state_dict['class_head.weight'].shape[0],
            'order': state_dict['order_head.weight'].shape[0],
            'family': state_dict['family_head.weight'].shape[0],
            'genus': state_dict['genus_head.weight'].shape[0],
            'species': state_dict['species_head.weight'].shape[0],
        }
        print(f"Class counts inferred from weights: {class_counts}")

    # Build mappings from taxonomy.csv
    name_to_id = {}
    id_to_name = {}

    for level in Config.TAXONOMY_LEVELS:
        unique_values = sorted(taxonomy_df[level].unique())
        name_to_id[level] = {name: idx for idx, name in enumerate(unique_values)}
        id_to_name[level] = {idx: name for name, idx in name_to_id[level].items()}

    print(f"Class counts: {class_counts}")

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = TaxonomyAwareClassifier.load_from_checkpoint(
        args.checkpoint,
        class_counts=class_counts
    )
    model.to(device)
    model.eval()

    # Load test data
    print("Loading test data...")
    test_annotations = pd.read_csv(Config.TEST_ANNOTATIONS)
    print(f"Test samples: {len(test_annotations)}")
    print(f"Test columns: {test_annotations.columns.tolist()}")
    print(test_annotations.head())

    # Create test dataset
    _, test_transform = create_transforms()

    # Get image paths and ROI coordinates
    test_image_paths = []
    test_bboxes = []

    for _, row in test_annotations.iterrows():
        img_path = os.path.join(Config.TEST_IMAGE_DIR, f"{row.name}.png")
        test_image_paths.append(img_path)
        # Use full image as bbox since we have pre-cropped ROIs
        test_bboxes.append([0, 0, 224, 224])

    # Create a simple test dataset
    from PIL import Image

    class TestDataset(torch.utils.data.Dataset):
        def __init__(self, annotations_df, transform=None):
            self.annotations = annotations_df
            self.transform = transform

        def __len__(self):
            return len(self.annotations)

        def __getitem__(self, idx):
            row = self.annotations.iloc[idx]

            # Path is already the full path in the annotations
            img_path = row['path']

            image = Image.open(img_path).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image, idx

    test_dataset = TestDataset(test_annotations, test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Run inference
    print("Running inference...")
    all_predictions = []
    all_indices = []

    with torch.no_grad():
        for images, indices in tqdm(test_loader, desc="Predicting"):
            images = images.to(device)
            outputs = model(images)

            # Get species predictions (most specific level)
            species_logits = outputs['species']
            species_preds = torch.argmax(species_logits, dim=1)

            all_predictions.extend(species_preds.cpu().numpy())
            all_indices.extend(indices.numpy())

    # Convert predictions to concept names
    concept_names = [id_to_name['species'][pred] for pred in all_predictions]

    # Create submission DataFrame
    # Kaggle expects annotation_id starting from 1
    submission_df = pd.DataFrame({
        'annotation_id': [idx + 1 for idx in all_indices],
        'concept_name': concept_names
    })

    # Sort by annotation_id
    submission_df = submission_df.sort_values('annotation_id')

    # Save submission
    submission_df.to_csv(args.output, index=False)
    print(f"\nSubmission saved to {args.output}")
    print(f"Total predictions: {len(submission_df)}")

    # Show distribution
    print("\nPrediction distribution (top 10):")
    for name, count in submission_df['concept_name'].value_counts().head(10).items():
        print(f"  {name}: {count} ({count/len(submission_df)*100:.1f}%)")


if __name__ == '__main__':
    main()
