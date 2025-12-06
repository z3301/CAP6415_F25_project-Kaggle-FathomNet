#!/usr/bin/env python3
"""
Full Image Training Script for FathomNet Model

Trains on full images (largest annotation per image) instead of cropped ROIs.

Usage:
    python train_fullimg.py --epochs 30 --lr 3e-4
"""

import argparse
import os
import sys
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.config import Config, DATA_ROOT
from data.data import FathomNetTaxonomyDataset, collate_fn, create_transforms
from src.taxonomy import load_and_encode_taxonomy
from src.models.model import TaxonomyAwareClassifier


def prepare_fullimg_data(batch_size=32):
    """Prepare datasets using full images instead of ROIs."""
    print("Preparing full image datasets...")

    # Load full image annotations
    annotations_path = os.path.join(DATA_ROOT, "train", "annotations_fullimg.csv")
    annotations = pd.read_csv(annotations_path)

    # Paths are already absolute in the CSV
    image_paths = annotations['path'].tolist()
    species_names = annotations['label'].tolist()

    print(f"Total full images: {len(image_paths)}")

    # Use simple random split (stratified fails for rare classes at image level)
    import numpy as np
    np.random.seed(42)
    indices = np.random.permutation(len(image_paths))

    n_train = int(0.7 * len(indices))
    n_val = int(0.15 * len(indices))

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    eval_idx = indices[n_train + n_val:]

    train_paths = [image_paths[i] for i in train_idx]
    train_species = [species_names[i] for i in train_idx]
    val_paths = [image_paths[i] for i in val_idx]
    val_species = [species_names[i] for i in val_idx]
    eval_paths = [image_paths[i] for i in eval_idx]
    eval_species = [species_names[i] for i in eval_idx]

    # Create transforms
    train_transform, val_transform = create_transforms()

    # Load taxonomy
    taxonomy_df, encoders, class_counts, id_to_name, name_to_id = load_and_encode_taxonomy()

    # Create datasets
    train_dataset = FathomNetTaxonomyDataset(train_paths, train_species, taxonomy_df, encoders, transform=train_transform)
    val_dataset = FathomNetTaxonomyDataset(val_paths, val_species, taxonomy_df, encoders, transform=val_transform)
    eval_dataset = FathomNetTaxonomyDataset(eval_paths, eval_species, taxonomy_df, encoders, transform=val_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                             num_workers=Config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                           num_workers=Config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=Config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Validation dataset: {len(val_dataset)} images")
    print(f"Evaluation dataset: {len(eval_dataset)} images")

    return train_loader, val_loader, eval_loader, taxonomy_df, encoders, class_counts, id_to_name, name_to_id


def main():
    parser = argparse.ArgumentParser(description='Train on full images')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Maximum number of epochs (default: 30)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size (default: 32)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')

    args = parser.parse_args()

    print("="*60)
    print("FULL IMAGE TRAINING")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Early stopping patience: {args.patience}")
    print()

    # Prepare data
    train_loader, val_loader, eval_loader, taxonomy_df, encoders, class_counts, id_to_name, name_to_id = prepare_fullimg_data(args.batch_size)

    # Create model
    print("Creating hierarchical model for full images...")
    model = TaxonomyAwareClassifier(
        class_counts=class_counts,
        lr=args.lr
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters:")
    print(f"  - Total: {total_params:,}")
    print(f"  - Trainable: {trainable_params:,}")
    print()

    # Setup output directory
    output_dir = 'outputs/fullimg'
    os.makedirs(output_dir, exist_ok=True)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='fullimg-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3,
        verbose=True
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=args.patience,
        mode='min',
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Trainer
    print("Starting training...")
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        accelerator='auto',
        devices=1,
        precision='16-mixed',
        log_every_n_steps=10,
        default_root_dir=output_dir
    )

    # Train
    trainer.fit(model, train_loader, val_loader)

    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val_loss: {checkpoint_callback.best_model_score:.4f}")
    print()


if __name__ == '__main__':
    main()
