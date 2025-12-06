#!/usr/bin/env python3
"""
Training Script for DINOv2 Multi-Context Patch Attention Model

Trains the DINOv2-based attention model with:
- ROI (1×): CLS token as Query (global embedding)
- Context (3×, 5×): Patch tokens as Key/Value
- Scaled dot-product cross-attention (Attention Is All You Need)
- Hierarchical classification

Usage:
    python train_dinov2_attention.py --epochs 30 --batch-size 8
"""

import argparse
import os
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit


from src.config import Config, DATA_ROOT
from data.data import create_transforms
from data.data_multiscale import MultiScaleFathomNetDataset, collate_fn_multiscale
from src.taxonomy import load_and_encode_taxonomy
from src.models.model_dinov2_attention import DINOv2PatchAttentionClassifier


def prepare_data(batch_size=8, scales=[1.0, 3.0, 5.0]):
    """Prepare multi-scale datasets for DINOv2 attention model."""
    print("Preparing multi-scale datasets for DINOv2 attention model...")

    # Load annotations (1× ROI paths)
    annotations = pd.read_csv(Config.TRAIN_ANNOTATIONS)
    image_paths = annotations['path'].tolist()
    species_names = annotations['label'].tolist()

    print(f"Total samples: {len(image_paths)}")
    print(f"Scales: {scales}")

    # Stratified split
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(sss1.split(image_paths, species_names))
    train_paths = [image_paths[i] for i in train_idx]
    train_species = [species_names[i] for i in train_idx]
    temp_paths = [image_paths[i] for i in temp_idx]
    temp_species = [species_names[i] for i in temp_idx]

    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    val_idx, eval_idx = next(sss2.split(temp_paths, temp_species))
    val_paths = [temp_paths[i] for i in val_idx]
    val_species = [temp_species[i] for i in val_idx]
    eval_paths = [temp_paths[i] for i in eval_idx]
    eval_species = [temp_species[i] for i in eval_idx]

    # Create transforms
    train_transform, val_transform = create_transforms()

    # Load taxonomy
    taxonomy_df, encoders, class_counts, id_to_name, name_to_id = load_and_encode_taxonomy()

    # Create multi-scale datasets
    train_dataset = MultiScaleFathomNetDataset(
        train_paths, train_species, taxonomy_df, encoders,
        transform=train_transform, scales=scales
    )
    val_dataset = MultiScaleFathomNetDataset(
        val_paths, val_species, taxonomy_df, encoders,
        transform=val_transform, scales=scales
    )
    eval_dataset = MultiScaleFathomNetDataset(
        eval_paths, eval_species, taxonomy_df, encoders,
        transform=val_transform, scales=scales
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=Config.NUM_WORKERS, collate_fn=collate_fn_multiscale, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=Config.NUM_WORKERS, collate_fn=collate_fn_multiscale, pin_memory=True
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=batch_size, shuffle=False,
        num_workers=Config.NUM_WORKERS, collate_fn=collate_fn_multiscale, pin_memory=True
    )

    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Validation dataset: {len(val_dataset)} images")
    print(f"Evaluation dataset: {len(eval_dataset)} images")

    return train_loader, val_loader, eval_loader, taxonomy_df, encoders, class_counts, id_to_name, name_to_id


def main():
    parser = argparse.ArgumentParser(description='Train DINOv2 Patch Attention Model')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Maximum number of epochs (default: 30)')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate (default: 3e-4)')
    parser.add_argument('--batch-size', type=int, default=8,
                       help='Batch size (default: 8, smaller due to ViT memory)')
    parser.add_argument('--patience', type=int, default=10,
                       help='Early stopping patience (default: 10)')
    parser.add_argument('--backbone', type=str, default='vit_base_patch14_dinov2.lvd142m',
                       help='DINOv2 backbone (default: vit_base_patch14_dinov2.lvd142m)')
    parser.add_argument('--num-heads', type=int, default=8,
                       help='Number of attention heads (default: 8)')
    parser.add_argument('--scales', type=str, default='1.0,3.0,5.0',
                       help='Comma-separated scales (default: 1.0,3.0,5.0)')

    args = parser.parse_args()

    # Parse scales
    scales = [float(s) for s in args.scales.split(',')]
    num_context_scales = len(scales) - 1  # Exclude ROI (1.0)

    print("=" * 60)
    print("DINOV2 MULTI-CONTEXT PATCH ATTENTION MODEL")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Epochs: {args.epochs}")
    print(f"  - Learning rate: {args.lr}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Backbone: {args.backbone}")
    print(f"  - Attention heads: {args.num_heads}")
    print(f"  - Scales: {scales}")
    print(f"  - Context scales: {num_context_scales}")
    print()
    print("Architecture:")
    print("  - ROI (1×) -> CLS token (Query)")
    print("  - Context (3×, 5×) -> Patch tokens (Key, Value)")
    print("  - Cross-attention: softmax(Q·K^T / √d_k) · V")
    print()

    # Prepare data
    train_loader, val_loader, eval_loader, taxonomy_df, encoders, class_counts, id_to_name, name_to_id = prepare_data(
        batch_size=args.batch_size, scales=scales
    )

    # Create model
    print("Creating DINOv2 Patch Attention model...")
    model = DINOv2PatchAttentionClassifier(
        class_counts=class_counts,
        lr=args.lr,
        num_context_scales=num_context_scales,
        backbone_name=args.backbone,
        num_heads=args.num_heads,
        dropout=0.3
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters:")
    print(f"  - Total: {total_params:,}")
    print(f"  - Trainable: {trainable_params:,}")
    print()

    # Setup output directory
    output_dir = f'outputs/dinov2_attention_{num_context_scales + 1}scales'
    os.makedirs(output_dir, exist_ok=True)

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename=f'dinov2_attn-{{epoch:02d}}-{{val_loss:.4f}}',
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

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best checkpoint: {checkpoint_callback.best_model_path}")
    print(f"Best val_loss: {checkpoint_callback.best_model_score:.4f}")
    print()


if __name__ == '__main__':
    main()
