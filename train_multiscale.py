#!/usr/bin/env python3
"""
Multi-scale training script for FathomNet 2025 Competition.

Uses pre-cropped ROI images at multiple scales (1x, 3x, 5x) with
separate ConvNeXtV2 encoders per scale.
"""

import argparse
import os
import sys

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import (
    load_and_encode_taxonomy,
    build_multiscale_dataloaders,
)
from src.model_multiscale import MultiScaleTaxonomyClassifier


def main():
    parser = argparse.ArgumentParser(description='Train multi-scale model')
    parser.add_argument('--config', type=str, default='config/experiment-multiscale.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--scales', type=str, nargs='+', default=['1x', '3x', '5x'],
                        help='Scales to use (e.g., 1x 3x 5x)')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override max epochs')
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--exp-name', type=str, default='multiscale',
                        help='Experiment name for logging')
    args = parser.parse_args()

    # Load config
    cfg = OmegaConf.load(args.config)

    # Override config values if provided
    if args.batch_size:
        cfg.data.batch_size = args.batch_size
    if args.epochs:
        cfg.training.max_epochs = args.epochs

    # Set seed
    pl.seed_everything(cfg.project.seed)

    # Load taxonomy
    print("Loading taxonomy...")
    taxonomy_df, encoders, class_counts, id_to_name = load_and_encode_taxonomy(
        cfg.paths.taxonomy_csv,
        list(cfg.data.taxonomy_levels),
    )
    print(f"Class counts: {class_counts}")

    # Build dataloaders
    print(f"Building dataloaders with scales: {args.scales}")
    dataloaders, split_indices = build_multiscale_dataloaders(
        cfg, taxonomy_df, encoders, scales=args.scales
    )
    print(f"Train samples: {len(split_indices['train'])}")
    print(f"Val samples: {len(split_indices['val'])}")
    print(f"Eval samples: {len(split_indices['eval'])}")

    # Create model
    print(f"Creating multi-scale model with {len(args.scales)} scales...")
    model = MultiScaleTaxonomyClassifier(cfg, class_counts, scales=args.scales)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Setup output directory
    output_dir = os.path.join(cfg.paths.data_root, "outputs", args.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="best-{epoch:02d}-{val_loss:.3f}",
            monitor=cfg.callbacks.monitor_metric,
            mode=cfg.callbacks.monitor_mode,
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor=cfg.callbacks.monitor_metric,
            mode=cfg.callbacks.monitor_mode,
            patience=cfg.callbacks.early_stopping_patience,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Logger
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name="lightning_logs",
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        precision=cfg.training.precision,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
    )

    # Train
    print("\nStarting training...")
    trainer.fit(
        model,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["val"],
        ckpt_path=args.resume,
    )

    print(f"\nTraining complete!")
    print(f"Best checkpoint: {callbacks[0].best_model_path}")
    print(f"Best val_loss: {callbacks[0].best_model_score:.4f}")


if __name__ == "__main__":
    main()
