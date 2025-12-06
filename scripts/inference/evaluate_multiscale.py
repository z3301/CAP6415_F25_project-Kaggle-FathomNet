#!/usr/bin/env python3
"""
Evaluation Script for Multi-Scale FathomNet Models

Evaluates trained multi-scale models and generates comprehensive metrics.

Usage:
    python evaluate_multiscale.py --model outputs/exp1a_2scales/best.ckpt
    python evaluate_multiscale.py --model outputs/exp1b_3scales/best.ckpt --split val
"""

import os
import argparse
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

from src.config import Config
from src.data_multiscale import MultiScaleFathomNetDataset, collate_fn_multiscale
from src.model_multiscale import MultiScaleTaxonomyClassifier
from src.taxonomy import load_and_encode_taxonomy
from src.data import create_transforms
from tqdm import tqdm
from sklearn.metrics import classification_report
from sklearn.utils.multiclass import unique_labels


def evaluate_multiscale_model(model, dataloader, class_counts, id_to_name, name="Evaluation"):
    """
    Evaluate multi-scale model performance.

    Args:
        model: Trained multi-scale model
        dataloader: DataLoader for evaluation (returns dict with scale_X.X keys)
        class_counts: Dict of class counts per taxonomic level
        id_to_name: Dict mapping IDs to names for each level
        name: Name of evaluation set
    """
    print(f"\n{'='*60}")
    print(f"RUNNING {name.upper()} METRICS")
    print(f"{'='*60}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    all_preds = {level: [] for level in Config.TAXONOMY_LEVELS}
    all_targets = {level: [] for level in Config.TAXONOMY_LEVELS}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{name} loop"):
            # Multi-scale batch returns (images_dict, labels_dict)
            scale_images, labels = batch

            # Move images to device
            scale_images_gpu = {}
            for key, value in scale_images.items():
                scale_images_gpu[key] = value.to(device)

            # Forward pass with multi-scale images
            outputs = model(scale_images_gpu)

            for level in Config.TAXONOMY_LEVELS:
                if level in outputs and level in labels:
                    preds = torch.argmax(outputs[level], dim=1).cpu()
                    targets_level = labels[level].cpu()
                    all_preds[level].append(preds)
                    all_targets[level].append(targets_level)

    # Calculate and print metrics
    print("\nAccuracy by Taxonomic Level:")
    print("-" * 40)
    for level in Config.TAXONOMY_LEVELS:
        if all_preds[level]:
            preds = torch.cat(all_preds[level])
            targets = torch.cat(all_targets[level])

            acc = (preds == targets).float().mean().item()
            print(f"{level.capitalize():12s}: {acc:.4f}")

            # Get actual labels present in the data
            labels = sorted(set(unique_labels(targets, preds)))
            class_names = [id_to_name[level][i] for i in labels]

            report = classification_report(
                targets,
                preds,
                labels=labels,
                target_names=class_names,
                zero_division=0,
                output_dict=False
            )

            print(f"\n{level.capitalize()} Classification Report:")
            print(report)


def prepare_eval_data(scales, split='val'):
    """
    Prepare evaluation dataset.

    Args:
        scales: List of scale multipliers
        split: 'val' or 'eval' for different holdout sets

    Returns:
        eval_loader: DataLoader for evaluation
        taxonomy_df: Taxonomy DataFrame
        encoders: Label encoders
        class_counts: Class counts per level
        id_to_name: ID to name mappings
    """
    print(f"Preparing {split} dataset with scales: {scales}")

    # Load annotations
    annotations = pd.read_csv(Config.TRAIN_ANNOTATIONS)
    image_paths = [os.path.join(Config.TRAIN_IMAGE_DIR, p) for p in annotations['path']]
    species_names = annotations['label'].tolist()

    # Replicate the exact train/val/eval split from training
    # Step 1: 70% train, 30% temp
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(sss1.split(image_paths, species_names))
    temp_paths = [image_paths[i] for i in temp_idx]
    temp_species = [species_names[i] for i in temp_idx]

    # Step 2: 15% val, 15% eval
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, eval_idx = next(sss2.split(temp_paths, temp_species))

    if split == 'val':
        eval_paths = [temp_paths[i] for i in val_idx]
        eval_species = [temp_species[i] for i in val_idx]
    else:  # eval
        eval_paths = [temp_paths[i] for i in eval_idx]
        eval_species = [temp_species[i] for i in eval_idx]

    # Create transforms
    _, val_transform = create_transforms()

    # Load taxonomy
    taxonomy_df, encoders, class_counts, id_to_name, name_to_id = load_and_encode_taxonomy()

    # Create dataset
    eval_dataset = MultiScaleFathomNetDataset(
        eval_paths, eval_species, taxonomy_df, encoders,
        transform=val_transform, scales=scales
    )

    # Create dataloader
    eval_loader = DataLoader(
        eval_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=lambda b: collate_fn_multiscale(b, Config.TAXONOMY_LEVELS),
        pin_memory=True
    )

    print(f"{split.capitalize()} dataset: {len(eval_dataset)} images")

    return eval_loader, taxonomy_df, encoders, class_counts, id_to_name


def infer_scales_from_checkpoint(checkpoint_path):
    """
    Infer number of scales from checkpoint filename.

    Args:
        checkpoint_path: Path to model checkpoint

    Returns:
        scales: List of scale multipliers
        num_scales: Number of scales
    """
    if 'exp1a' in checkpoint_path or '2scales' in checkpoint_path:
        scales = [1.0, 3.0]
        num_scales = 2
    elif 'exp1b' in checkpoint_path or '3scales' in checkpoint_path:
        scales = [1.0, 3.0, 5.0]
        num_scales = 3
    else:
        # Default to 3 scales
        print("Warning: Could not infer scales from checkpoint path. Defaulting to 3 scales.")
        scales = [1.0, 3.0, 5.0]
        num_scales = 3

    return scales, num_scales


def main():
    parser = argparse.ArgumentParser(description='Evaluate multi-scale FathomNet model')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--split', type=str, default='val', choices=['val', 'eval'],
                       help='Which split to evaluate on (default: val)')
    parser.add_argument('--scales', type=float, nargs='+', default=None,
                       help='Scale multipliers (default: infer from checkpoint)')

    args = parser.parse_args()

    # Check if checkpoint exists
    if not os.path.exists(args.model):
        print(f"Error: Checkpoint not found at {args.model}")
        return

    # Infer scales if not provided
    if args.scales is None:
        scales, num_scales = infer_scales_from_checkpoint(args.model)
        print(f"Inferred {num_scales} scales from checkpoint: {scales}")
    else:
        scales = args.scales
        num_scales = len(scales)
        print(f"Using provided scales: {scales}")

    print("="*60)
    print(f"EVALUATING MULTI-SCALE MODEL")
    print("="*60)
    print(f"Checkpoint: {args.model}")
    print(f"Split: {args.split}")
    print(f"Scales: {scales}")
    print()

    # Prepare evaluation data
    eval_loader, taxonomy_df, encoders, class_counts, id_to_name = \
        prepare_eval_data(scales, split=args.split)

    # Load model
    print(f"Loading model from {args.model}...")
    model = MultiScaleTaxonomyClassifier.load_from_checkpoint(
        args.model,
        class_counts=class_counts,
        num_scales=num_scales
    )
    model.eval()

    # Run evaluation (multi-scale version)
    evaluate_multiscale_model(
        model, eval_loader, class_counts, id_to_name,
        name=f"{args.split.capitalize()} Set Evaluation"
    )

    print("\nEvaluation complete!")


if __name__ == '__main__':
    main()
