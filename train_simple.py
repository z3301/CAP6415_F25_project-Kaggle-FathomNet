#!/usr/bin/env python3
"""
Quick test: Train independent heads (no hierarchical conditioning)

This will tell us if the conditioning is causing the collapse.
"""

import argparse
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from src.train import prepare_data
from src.model_simple import SimpleTaxonomyClassifier


def main():
    parser = argparse.ArgumentParser(description='Train simple independent-head model')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Maximum number of epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')

    args = parser.parse_args()

    print("="*60)
    print("SIMPLE INDEPENDENT-HEAD MODEL (NO CONDITIONING)")
    print("="*60)
    print(f"This tests if hierarchical conditioning is the problem")
    print(f"Epochs: {args.epochs}, LR: {args.lr}")
    print()

    # Prepare data
    train_loader, val_loader, eval_loader, taxonomy_df, encoders, class_counts, id_to_name, name_to_id = prepare_data()

    # Create model
    print("Creating simple model with independent heads...")
    model = SimpleTaxonomyClassifier(
        class_counts=class_counts,
        lr=args.lr
    )

    # Setup output directory
    output_dir = 'outputs/simple_independent'
    os.makedirs(output_dir, exist_ok=True)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename='simple-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        verbose=True
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=5,
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
