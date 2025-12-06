#!/usr/bin/env python3
"""
================================================================================
Multi-Scale Cross-Attention Training Script for FathomNet 2025 Competition
================================================================================

This script trains a cross-attention based multi-scale classifier that uses
the ROI features to attend to context features from surrounding scales. This
approach is inspired by the winning team's architecture from FathomNet 2023.

MOTIVATION
----------
The insight behind cross-attention for marine species classification:
- The ROI (Region of Interest) contains the organism to classify
- Context scales (3x, 5x, full) contain environmental/habitat information
- Not all context information is equally relevant for each ROI
- Cross-attention allows the model to dynamically weight context information
  based on the specific ROI being classified

ARCHITECTURE
------------
```
                ROI (1x)              Context Scales (3x, 5x, full)
                   |                         |
            [ConvNeXtV2-Base]           [ConvNeXtV2-Base] × N
                   |                         |
            [7×7 patch features]       [7×7 patch features each]
                   |                         |
               Q (Query)            K, V (Key, Value) concatenated
                   \\                       /
                    \\                     /
                     [Cross-Attention Layers]
                              |
                      [Fused Features]
                              |
                  [Hierarchical Classification]
                              |
        Kingdom → Phylum → Class → Order → Family → Genus → Species
```

CROSS-ATTENTION MECHANISM
-------------------------
For each ROI patch position (49 positions from 7×7 grid):

    Attention(Q, K, V) = softmax(QK^T / √d_k) × V

where:
- Q: ROI patch features (49 × 1024)
- K: Context patch features from all scales ((49 × N_scales) × 1024)
- V: Same as K
- d_k: Attention dimension (1024)

This allows each ROI patch to attend to relevant context patches across
all scales, enabling fine-grained ROI-context feature fusion.

MODEL VARIANTS
--------------
1. **MultiScaleCrossAttentionClassifier** (default):
   - Uses global ROI features as query
   - Context patch features as keys/values
   - Simpler, fewer parameters

2. **MultiScaleCrossAttentionWithROIPatches** (--use-roi-patches):
   - Uses ROI patch features as queries (49 queries)
   - More expressive but computationally heavier
   - Better for fine-grained spatial correspondence

ASYMMETRIC BACKBONES
--------------------
Different backbone architectures can be used for ROI vs context:

- **ROI Backbone**: Larger model (e.g., ConvNeXtV2-Large) for detailed
  morphological features
- **Context Backbones**: Smaller models (e.g., ConvNeXtV2-Base) since
  context information is coarser

Example:
    python train_multiscale_attention.py \\
        --roi-backbone convnextv2_large.fcmae_ft_in22k_in1k \\
        --context-backbone convnextv2_base.fcmae_ft_in22k_in1k

USAGE EXAMPLES
--------------
# Default 3-scale cross-attention
python train_multiscale_attention.py

# With ROI patch queries
python train_multiscale_attention.py --use-roi-patches

# Asymmetric backbones (larger ROI encoder)
python train_multiscale_attention.py \\
    --roi-backbone convnextv2_large.fcmae_ft_in22k_in1k \\
    --context-backbone convnextv2_base.fcmae_ft_in22k_in1k

# More attention layers
python train_multiscale_attention.py --attention-layers 6 --attention-heads 16

# Multi-GPU training
CUDA_VISIBLE_DEVICES=0,1 python train_multiscale_attention.py \\
    --gpus 2 \\
    --strategy ddp

HYPERPARAMETERS
---------------
- --attention-layers: Number of stacked cross-attention layers (default: 4)
- --attention-heads: Number of attention heads per layer (default: 8)
- These control the capacity of the attention mechanism

EXPERIMENTAL RESULTS
--------------------
The cross-attention model offers a different approach to multi-scale fusion
compared to simple concatenation. Results vary based on:
- Number of attention layers
- Choice of backbone architectures
- Whether ROI patches are used as queries

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
from src.models.model_multiscale_attention import (
    MultiScaleCrossAttentionClassifier,
    MultiScaleCrossAttentionWithROIPatches,
)


def main():
    """
    Main training function for cross-attention multi-scale model.

    Workflow:
    1. Parse arguments (including attention configuration)
    2. Load configuration and taxonomy
    3. Build multi-scale dataloaders
    4. Create cross-attention model (with or without ROI patches)
    5. Setup callbacks and trainer
    6. Train with PyTorch Lightning
    """
    # =========================================================================
    # ARGUMENT PARSING
    # =========================================================================
    parser = argparse.ArgumentParser(
        description='Train multi-scale cross-attention model for marine species',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model Variants:
  Default      - Global ROI features as query
  ROI Patches  - ROI patch features as queries (--use-roi-patches)

Examples:
  # Default cross-attention
  python train_multiscale_attention.py --scales 1x 3x 5x

  # With ROI patches as queries
  python train_multiscale_attention.py --use-roi-patches

  # Asymmetric backbones
  python train_multiscale_attention.py \\
      --roi-backbone convnextv2_large.fcmae_ft_in22k_in1k \\
      --context-backbone convnextv2_base.fcmae_ft_in22k_in1k
        """
    )

    # Configuration
    parser.add_argument('--config', type=str, default='config/experiment-multiscale.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')

    # Model architecture
    parser.add_argument('--scales', type=str, nargs='+', default=['1x', '3x', '5x'],
                        help='Scales to use (e.g., 1x 3x 5x)')

    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size from config')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override max epochs from config')

    # Hardware configuration
    parser.add_argument('--gpus', type=int, default=1,
                        help='Number of GPUs to use')
    parser.add_argument('--strategy', type=str, default='auto',
                        help='PyTorch Lightning strategy: auto, ddp, fsdp')

    # Experiment tracking
    parser.add_argument('--exp-name', type=str, default='attention_3scales',
                        help='Experiment name for output directory')

    # Cross-attention configuration
    parser.add_argument('--attention-layers', type=int, default=4,
                        help='Number of stacked cross-attention layers (default: 4)')
    parser.add_argument('--attention-heads', type=int, default=8,
                        help='Number of attention heads per layer (default: 8)')
    parser.add_argument('--use-roi-patches', action='store_true',
                        help='Use ROI patch features as queries instead of global features')

    # Backbone configuration (asymmetric)
    parser.add_argument('--roi-backbone', type=str, default=None,
                        help='Backbone for ROI encoder (e.g., convnextv2_large.fcmae_ft_in22k_in1k). '
                             'If None, uses default ConvNeXtV2-Base.')
    parser.add_argument('--context-backbone', type=str, default=None,
                        help='Backbone for context encoders (e.g., convnextv2_base.fcmae_ft_in22k_in1k). '
                             'If None, uses default ConvNeXtV2-Base.')

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
    print(f"Eval samples: {len(split_indices['eval'])}")

    # =========================================================================
    # MODEL CREATION
    # =========================================================================
    print(f"\nCreating cross-attention model:")
    print(f"  Scales: {args.scales}")
    print(f"  Attention layers: {args.attention_layers}")
    print(f"  Attention heads: {args.attention_heads}")
    print(f"  Use ROI patches: {args.use_roi_patches}")
    print(f"  ROI backbone: {args.roi_backbone or 'default (ConvNeXtV2-Base)'}")
    print(f"  Context backbone: {args.context_backbone or 'default (ConvNeXtV2-Base)'}")
    print(f"  GPUs: {args.gpus}, Strategy: {args.strategy}")

    # Choose model variant based on --use-roi-patches flag
    if args.use_roi_patches:
        # ROI patches as queries - more expressive but heavier
        model = MultiScaleCrossAttentionWithROIPatches(
            cfg, class_counts,
            scales=args.scales,
            num_attention_layers=args.attention_layers,
            num_attention_heads=args.attention_heads,
            roi_backbone=args.roi_backbone,
            context_backbone=args.context_backbone,
        )
    else:
        # Global ROI features as query - simpler
        model = MultiScaleCrossAttentionClassifier(
            cfg, class_counts,
            scales=args.scales,
            num_attention_layers=args.attention_layers,
            num_attention_heads=args.attention_heads,
            roi_backbone=args.roi_backbone,
            context_backbone=args.context_backbone,
        )

    # =========================================================================
    # PARAMETER COUNTING
    # =========================================================================
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Breakdown by component
    encoder_params = sum(
        p.numel() for n, p in model.named_parameters()
        if "encoder" in n
    )
    attention_params = sum(
        p.numel() for n, p in model.named_parameters()
        if "cross_attention" in n
    )
    head_params = total_params - encoder_params - attention_params
    print(f"  Encoder params: {encoder_params:,}")
    print(f"  Attention params: {attention_params:,}")
    print(f"  Head params: {head_params:,}")

    # =========================================================================
    # OUTPUT DIRECTORY SETUP
    # =========================================================================
    output_dir = os.path.join(cfg.paths.data_root, "outputs", args.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # =========================================================================
    # CALLBACKS
    # =========================================================================
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
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator="gpu" if args.gpus > 0 else "cpu",
        devices=args.gpus if args.gpus > 0 else 1,
        strategy=args.strategy if args.gpus > 1 else "auto",
        precision=cfg.training.precision,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
    )

    # =========================================================================
    # TRAINING
    # =========================================================================
    print("\nStarting training...")
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
    print(f"Best val_loss: {callbacks[0].best_model_score:.4f}")


if __name__ == "__main__":
    main()
