#!/usr/bin/env python3
"""
Baseline Training Script for Single-Scale FathomNet Model

Tests the hierarchical classifier without multi-scale to isolate issues.

Usage:
    python train_baseline.py --epochs 30 --lr 3e-4
"""

import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from src.train import prepare_data
from src.model import TaxonomyAwareClassifier


def main():
    parser = argparse.ArgumentParser(description='Train baseline single-scale hierarchical model')
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
    print("BASELINE SINGLE-SCALE TRAINING")
    print("="*60)
    print(f"Configuration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Early stopping patience: {args.patience}")
    print()

    # Prepare data
    train_loader, val_loader, eval_loader, taxonomy_df, encoders, class_counts, id_to_name, name_to_id = prepare_data()

    print(f"Dataset sizes:")
    print(f"  - Training: {len(train_loader.dataset)} images")
    print(f"  - Validation: {len(val_loader.dataset)} images")
    print(f"  - Evaluation: {len(eval_loader.dataset)} images")
    print()

    # Create model
    print("Creating single-scale hierarchical model...")
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
    output_dir = 'outputs/baseline_single_scale'
    os.makedirs(output_dir, exist_ok=True)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='baseline-{epoch:02d}-{val_loss:.4f}',
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
