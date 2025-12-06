#!/usr/bin/env python3
"""
================================================================================
Multi-Scale Training with Taxonomic Distance-Aware Loss
================================================================================

This script trains a multi-scale hierarchical classifier using taxonomy-aware
loss functions that better align with the competition metric (taxonomic distance).

MOTIVATION
----------
The FathomNet 2025 competition uses taxonomic distance as the evaluation metric:
- Misclassifying a species within the same genus costs less than
- Misclassifying across different genera, which costs less than
- Misclassifying across different families, etc.

Standard cross-entropy loss treats all misclassifications equally, which is
suboptimal for this metric. This script implements loss functions that
incorporate taxonomic relationships during training.

LOSS FUNCTIONS
--------------
1. **Taxonomic Distance Loss** (--loss-type distance):
   Penalizes misclassifications proportionally to taxonomic distance.

   L = (1-α) × CE(y, ŷ) + α × E[D(y, ŷ)]

   where:
   - CE is cross-entropy loss
   - D(y, ŷ) is the taxonomic distance between true and predicted species
   - α controls the weight of the distance penalty (default: 0.3)

   The distance penalty is computed as:
   E[D] = Σᵢ p(i) × D(y, i)  (expected distance under predicted distribution)

2. **Taxonomic Label Smoothing** (--loss-type smooth):
   Instead of one-hot targets, distributes probability mass to taxonomically
   similar species.

   q_i = (1-ε) × I(i=y) + ε × sim(i,y) / Z

   where:
   - ε is the smoothing factor (default: 0.1)
   - sim(i,y) = 1 / (1 + D(i,y)) is similarity based on taxonomic distance
   - Z is a normalization constant

3. **Combined Loss** (--loss-type both):
   Uses both distance penalty and label smoothing together.

4. **Baseline CE** (--loss-type ce):
   Standard cross-entropy for comparison.

DISTANCE MATRIX
---------------
The taxonomic distance matrix (distance_matrix.csv) is an 80×80 symmetric matrix
where D[i,j] represents the taxonomic distance between species i and j:

- Same species: D = 0
- Same genus: D = 1
- Same family: D = 2
- Same order: D = 3
- Same class: D = 4
- Same phylum: D = 5
- Same kingdom: D = 6
- Different kingdoms: D = 7

The matrix is generated from the taxonomy.csv file using:
    python src/create_distance_matrix.py

MONITORING
----------
This script monitors `val_tax_score` instead of `val_loss`:
- val_tax_score directly measures competition metric (taxonomic distance)
- Lower is better (0 = perfect predictions)
- Early stopping and checkpointing based on this metric

USAGE EXAMPLES
--------------
# Train with taxonomic distance loss (default settings)
python train_multiscale_taxloss.py --scales 1x 3x 5x full

# Higher distance penalty weight
python train_multiscale_taxloss.py --loss-type distance --alpha 0.5

# Train with taxonomic label smoothing
python train_multiscale_taxloss.py --loss-type smooth --smoothing 0.15

# Combined loss (both distance and smoothing)
python train_multiscale_taxloss.py --loss-type both --alpha 0.3 --smoothing 0.1

# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1 python train_multiscale_taxloss.py \\
    --scales 1x 3x 5x full \\
    --gpus 2 \\
    --strategy ddp_find_unused_parameters_true

EXPERIMENTAL RESULTS
--------------------
The taxonomic-aware losses help align training with the competition metric,
potentially improving the final score compared to standard cross-entropy.

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

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from data.data import (
    load_and_encode_taxonomy,
    build_multiscale_dataloaders,
)
from src.models.model_multiscale_taxloss import MultiScaleTaxonomicClassifier


def main():
    """
    Main training function for taxonomic distance-aware training.

    Workflow:
    1. Parse command-line arguments (including loss type and hyperparameters)
    2. Load configuration and taxonomy
    3. Build multi-scale dataloaders
    4. Create model with taxonomic loss function
    5. Setup callbacks (monitoring val_tax_score, not val_loss)
    6. Train with PyTorch Lightning
    """
    # =========================================================================
    # ARGUMENT PARSING
    # =========================================================================
    parser = argparse.ArgumentParser(
        description='Train multi-scale model with taxonomic distance-aware loss',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Loss Types:
  distance  - Cross-entropy + expected distance penalty (default)
  smooth    - Cross-entropy with taxonomic label smoothing
  both      - Combined distance penalty and label smoothing
  ce        - Baseline cross-entropy (for comparison)

Examples:
  # Default taxonomic distance loss
  python train_multiscale_taxloss.py --scales 1x 3x 5x full

  # Higher distance weight
  python train_multiscale_taxloss.py --loss-type distance --alpha 0.5

  # Label smoothing based on taxonomy
  python train_multiscale_taxloss.py --loss-type smooth --smoothing 0.15
        """
    )

    # Configuration
    parser.add_argument('--config', type=str, default='config/experiment-multiscale.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--distance-matrix', type=str, default='data/distance_matrix.csv',
                        help='Path to taxonomic distance matrix CSV (80x80 symmetric)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')

    # Model architecture
    parser.add_argument('--scales', type=str, nargs='+', default=['1x', '3x', '5x'],
                        help='Scales to use (e.g., 1x 3x 5x full)')

    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size from config')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override max epochs from config')

    # Hardware configuration
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--strategy', type=str, default=None,
                        help='PyTorch Lightning strategy (e.g., ddp_find_unused_parameters_true)')

    # Experiment tracking
    parser.add_argument('--exp-name', type=str, default='multiscale_taxloss',
                        help='Experiment name for output directory')

    # Taxonomic loss configuration
    parser.add_argument('--loss-type', type=str, default='distance',
                        choices=['distance', 'smooth', 'both', 'ce'],
                        help='Type of loss function to use (default: distance)')
    parser.add_argument('--alpha', type=float, default=0.3,
                        help='Weight for distance penalty: L = (1-α)×CE + α×E[D] '
                             '(0=pure CE, 1=pure distance). Default: 0.3')
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing factor for taxonomic smoothing. '
                             'Distributes ε probability mass to similar species. Default: 0.1')

    args = parser.parse_args()

    # =========================================================================
    # CONFIGURATION LOADING
    # =========================================================================
    cfg = OmegaConf.load(args.config)

    # Override config values with command-line arguments
    if args.batch_size:
        cfg.data.batch_size = args.batch_size
    if args.epochs:
        cfg.training.max_epochs = args.epochs

    # Set random seed for reproducibility
    pl.seed_everything(cfg.project.seed)

    # =========================================================================
    # TAXONOMY LOADING
    # =========================================================================
    print("Loading taxonomy...")
    taxonomy_df, encoders, class_counts, id_to_name = load_and_encode_taxonomy(
        cfg.paths.taxonomy_csv,
        list(cfg.data.taxonomy_levels),
    )
    print(f"Class counts: {class_counts}")

    # =========================================================================
    # DATA LOADING
    # =========================================================================
    print(f"Building dataloaders with scales: {args.scales}")
    dataloaders, split_indices = build_multiscale_dataloaders(
        cfg, taxonomy_df, encoders, scales=args.scales
    )
    print(f"Train samples: {len(split_indices['train'])}")
    print(f"Val samples: {len(split_indices['val'])}")

    # =========================================================================
    # MODEL CREATION WITH TAXONOMIC LOSS
    # =========================================================================
    print(f"\nCreating model with taxonomic {args.loss_type} loss")
    print(f"  Distance matrix: {args.distance_matrix}")
    print(f"  Alpha: {args.alpha}")
    print(f"  Smoothing: {args.smoothing}")

    # The model internally creates the appropriate loss function based on loss_type:
    # - 'distance': TaxonomicDistanceLoss
    # - 'smooth': TaxonomicLabelSmoothing
    # - 'both': Combined TaxonomicDistanceLoss + TaxonomicLabelSmoothing
    # - 'ce': Standard CrossEntropyLoss
    model = MultiScaleTaxonomicClassifier(
        cfg=cfg,
        class_counts=class_counts,
        id_to_name=id_to_name,
        scales=args.scales,
        distance_matrix_path=args.distance_matrix,
        loss_type=args.loss_type,
        alpha=args.alpha,
        smoothing=args.smoothing,
    )

    # Log parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")

    # =========================================================================
    # OUTPUT DIRECTORY SETUP
    # =========================================================================
    output_dir = os.path.join(cfg.paths.data_root, "outputs", args.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # CALLBACKS
    # =========================================================================
    # IMPORTANT: We monitor val_tax_score, NOT val_loss
    # val_tax_score is the actual competition metric (taxonomic distance)
    # Lower is better (0 = perfect predictions)
    callbacks = [
        ModelCheckpoint(
            dirpath=os.path.join(output_dir, "checkpoints"),
            filename="best-{epoch:02d}-{val_tax_score:.3f}",
            monitor="val_tax_score",  # Competition metric
            mode="min",               # Lower taxonomic distance is better
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_tax_score",
            mode="min",
            patience=cfg.callbacks.early_stopping_patience,
            verbose=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # =========================================================================
    # LOGGER
    # =========================================================================
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
        precision=cfg.training.precision,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
    )

    # Add distributed strategy for multi-GPU training
    if args.strategy:
        trainer_kwargs["strategy"] = args.strategy

    trainer = pl.Trainer(**trainer_kwargs)

    # =========================================================================
    # TRAINING
    # =========================================================================
    print("\nStarting training...")
    print("Monitoring: val_tax_score (lower is better)")
    trainer.fit(
        model,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["val"],
        ckpt_path=args.resume,
    )

    # =========================================================================
    # TRAINING COMPLETE
    # =========================================================================
    print(f"\nTraining complete!")
    print(f"Best checkpoint: {callbacks[0].best_model_path}")
    print(f"Best val_tax_score: {callbacks[0].best_model_score:.4f}")


if __name__ == "__main__":
    main()
