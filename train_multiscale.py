#!/usr/bin/env python3
"""
================================================================================
Multi-Scale Training Script for FathomNet 2025 Competition
================================================================================

This script trains a multi-scale hierarchical classifier for marine species
identification. It uses multiple ConvNeXtV2-Base encoders (one per spatial scale)
to capture both fine-grained ROI details and broader contextual information.

APPROACH OVERVIEW
-----------------
The key insight is that marine species classification benefits from multiple
spatial contexts:

1. ROI (1x): Fine-grained morphological details (tentacles, body shape, texture)
2. 3x context: Immediate surroundings, substrate type, nearby organisms
3. 5x context: Broader habitat context, depth indicators
4. Full image: Global scene understanding, environmental cues

Each scale is processed by a separate ConvNeXtV2-Base encoder, and features
are concatenated then passed through hierarchical classification heads.

ARCHITECTURE
------------
```
                        Multi-Scale Input
                    /     |      |      \\
                  1x     3x     5x    full
                   |      |      |      |
             [ConvNeXtV2-Base] x 4 (88M params each)
                   |      |      |      |
               [1024]  [1024] [1024] [1024]  (feature dims)
                    \\    |     |    /
                     CONCATENATE
                          |
                       [4096]
                          |
                   [SharedFeatures]
                          |
                       [1024]
                          |
            +-------------+-------------+
            |             |             |
        Kingdom       Phylum    ...  Species
         (2)           (8)           (80)
```

HIERARCHICAL CLASSIFICATION
---------------------------
The model predicts at 7 taxonomic levels, with each level receiving information
from the previous level (taxonomic cascade):

    Kingdom (2) → Phylum (8) → Class (22) → Order (43)
              → Family (66) → Genus (75) → Species (80)

Each head receives:
- Shared multi-scale features (1024d)
- Embedded prediction from parent level (e.g., phylum head gets kingdom embedding)

This cascade design helps propagate taxonomic constraints down the hierarchy.

TRAINING CONFIGURATION
----------------------
- Optimizer: AdamW with weight decay 0.01
- Learning rate: 3e-4 with CosineAnnealingWarmRestarts
- Precision: 16-bit mixed precision (AMP)
- Batch size: 8-16 depending on GPU memory
- Gradient clipping: 1.0
- Early stopping: patience 10 on val_loss

DATA PIPELINE
-------------
Uses pre-cropped multi-scale ROI images stored in:
    train/multiscale_rois/{annotation_id}/{scale}.jpg

This avoids expensive on-the-fly cropping during training.

USAGE EXAMPLES
--------------
# 3-scale training (default)
python train_multiscale.py --config config/experiment-multiscale.yaml

# 4-scale training with full image context
python train_multiscale.py --scales 1x 3x 5x full --exp-name multiscale_4scales

# Multi-GPU training with DDP
CUDA_VISIBLE_DEVICES=0,1 python train_multiscale.py \\
    --scales 1x 3x 5x full \\
    --batch-size 8 \\
    --gpus 2 \\
    --strategy ddp_find_unused_parameters_true

# Resume from checkpoint
python train_multiscale.py --resume outputs/multiscale/checkpoints/last.ckpt

COMPETITION RESULTS
-------------------
- 3-scale model: ~2.74 taxonomic distance score
- 4-scale model: ~2.05 taxonomic distance score (significant improvement)

The full-image scale provides crucial global context that improves species
discrimination, especially for habitat-dependent species.

Author: Dan Zimmerman
Course: CAP6415 - Computer Vision (Fall 2025)
================================================================================
"""

import argparse
import os
import sys

import pytorch_lightning as pl
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Add project root to path for src imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data import (
    load_and_encode_taxonomy,
    build_multiscale_dataloaders,
)
from src.model_multiscale import MultiScaleTaxonomyClassifier


def main():
    """
    Main training function.

    Workflow:
    1. Parse command-line arguments
    2. Load configuration from YAML file
    3. Load and encode taxonomy labels
    4. Build multi-scale dataloaders
    5. Create model with specified scales
    6. Setup callbacks (checkpointing, early stopping, LR monitoring)
    7. Train with PyTorch Lightning
    """
    # =========================================================================
    # ARGUMENT PARSING
    # =========================================================================
    parser = argparse.ArgumentParser(
        description='Train multi-scale hierarchical classifier for marine species',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic 3-scale training
  python train_multiscale.py

  # 4-scale with full image context
  python train_multiscale.py --scales 1x 3x 5x full --exp-name multiscale_4scales

  # Multi-GPU DDP training
  CUDA_VISIBLE_DEVICES=0,1 python train_multiscale.py --gpus 2 --strategy ddp_find_unused_parameters_true
        """
    )

    # Configuration
    parser.add_argument('--config', type=str, default='config/experiment-multiscale.yaml',
                        help='Path to YAML config file (default: config/experiment-multiscale.yaml)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')

    # Model architecture
    parser.add_argument('--scales', type=str, nargs='+', default=['1x', '3x', '5x'],
                        help='Scales to use. Options: 1x, 3x, 5x, full (default: 1x 3x 5x)')

    # Training hyperparameters (override config)
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size from config')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override max epochs from config')

    # Hardware configuration
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use (default: 1)')
    parser.add_argument('--strategy', type=str, default=None,
                        help='PyTorch Lightning strategy for multi-GPU training. '
                             'Use "ddp_find_unused_parameters_true" for multi-GPU.')

    # Experiment tracking
    parser.add_argument('--exp-name', type=str, default='multiscale',
                        help='Experiment name for output directory and logging')

    args = parser.parse_args()

    # =========================================================================
    # CONFIGURATION LOADING
    # =========================================================================
    # Load base config from YAML
    cfg = OmegaConf.load(args.config)

    # Override config values with command-line arguments if provided
    if args.batch_size:
        cfg.data.batch_size = args.batch_size
    if args.epochs:
        cfg.training.max_epochs = args.epochs

    # Set random seed for reproducibility
    # This seeds Python, NumPy, and PyTorch random number generators
    pl.seed_everything(cfg.project.seed)

    # =========================================================================
    # TAXONOMY LOADING
    # =========================================================================
    # Load taxonomy CSV and create label encoders for each taxonomic level
    # This maps species names to integer indices and builds the class hierarchy
    print("Loading taxonomy...")
    taxonomy_df, encoders, class_counts, id_to_name = load_and_encode_taxonomy(
        cfg.paths.taxonomy_csv,
        list(cfg.data.taxonomy_levels),
    )
    print(f"Class counts: {class_counts}")
    # Expected: {'kingdom': 2, 'phylum': 8, 'class': 22, 'order': 43,
    #            'family': 66, 'genus': 75, 'species': 80}

    # =========================================================================
    # DATA LOADING
    # =========================================================================
    # Build PyTorch dataloaders for train/val/eval splits
    # Uses stratified splitting to maintain class distribution
    print(f"Building dataloaders with scales: {args.scales}")
    dataloaders, split_indices = build_multiscale_dataloaders(
        cfg, taxonomy_df, encoders, scales=args.scales
    )
    print(f"Train samples: {len(split_indices['train'])}")  # ~18,959
    print(f"Val samples: {len(split_indices['val'])}")      # ~2,370
    print(f"Eval samples: {len(split_indices['eval'])}")    # ~2,370

    # =========================================================================
    # MODEL CREATION
    # =========================================================================
    # Create multi-scale model with one ConvNeXtV2-Base encoder per scale
    # Total params = 88M × num_scales + classification heads
    print(f"Creating multi-scale model with {len(args.scales)} scales...")
    model = MultiScaleTaxonomyClassifier(cfg, class_counts, scales=args.scales)

    # Log parameter counts for verification
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    # Expected for 4 scales: ~358M total parameters

    # =========================================================================
    # OUTPUT DIRECTORY SETUP
    # =========================================================================
    # Create output directory for checkpoints and logs
    output_dir = os.path.join(cfg.paths.data_root, "outputs", args.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # CALLBACKS
    # =========================================================================
    callbacks = [
        # Save best models based on validation loss
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="best-{epoch:02d}-{val_loss:.3f}",
            monitor=cfg.callbacks.monitor_metric,  # "val_loss"
            mode=cfg.callbacks.monitor_mode,        # "min"
            save_top_k=3,                           # Keep top 3 checkpoints
            save_last=True,                         # Always save last checkpoint
        ),
        # Stop training if validation loss doesn't improve
        EarlyStopping(
            monitor=cfg.callbacks.monitor_metric,
            mode=cfg.callbacks.monitor_mode,
            patience=cfg.callbacks.early_stopping_patience,  # Default: 10 epochs
            verbose=True,
        ),
        # Log learning rate for monitoring warmup/annealing
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # =========================================================================
    # LOGGER
    # =========================================================================
    # TensorBoard logger for training visualization
    # View with: tensorboard --logdir outputs/multiscale/lightning_logs
    logger = TensorBoardLogger(
        save_dir=output_dir,
        name="lightning_logs",
    )

    # =========================================================================
    # TRAINER SETUP
    # =========================================================================
    trainer_kwargs = dict(
        max_epochs=cfg.training.max_epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        precision=cfg.training.precision,  # "16-mixed" for AMP
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,  # Prevent exploding gradients
    )

    # Add distributed strategy for multi-GPU training
    # DDP (Distributed Data Parallel) is the recommended strategy
    # find_unused_parameters=True is needed because not all taxonomy heads
    # may be used in every forward pass (hierarchical cascade)
    if args.strategy:
        trainer_kwargs["strategy"] = args.strategy

    trainer = pl.Trainer(**trainer_kwargs)

    # =========================================================================
    # TRAINING
    # =========================================================================
    print("\nStarting training...")
    trainer.fit(
        model,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["val"],
        ckpt_path=args.resume,  # Resume from checkpoint if specified
    )

    # =========================================================================
    # TRAINING COMPLETE
    # =========================================================================
    print(f"\nTraining complete!")
    print(f"Best checkpoint: {callbacks[0].best_model_path}")
    print(f"Best val_loss: {callbacks[0].best_model_score:.4f}")


if __name__ == "__main__":
    main()
