"""
Training utilities for FathomNet 2025 Competition

Contains data preparation and model training functions.
"""

import os
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

from src.config import Config
from src.data import FathomNetTaxonomyDataset, collate_fn, create_transforms
from src.taxonomy import load_and_encode_taxonomy
from src.model import TaxonomyAwareClassifier


def prepare_data():
    """
    Prepare training, validation, and evaluation datasets.

    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        eval_loader: Evaluation DataLoader
        taxonomy_df: Taxonomy DataFrame
        encoders: Label encoders
        class_counts: Class counts per taxonomic level
        id_to_name: ID to name mappings
        name_to_id: Name to ID mappings
    """
    print("Preparing datasets...")

    # Load annotations
    annotations = pd.read_csv(Config.TRAIN_ANNOTATIONS)
    image_paths = [os.path.join(Config.TRAIN_IMAGE_DIR, p) for p in annotations['path']]
    species_names = annotations['label'].tolist()

    # Step 1: 70% train, 30% temp (val + eval)
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, temp_idx = next(sss1.split(image_paths, species_names))
    train_paths = [image_paths[i] for i in train_idx]
    train_species = [species_names[i] for i in train_idx]
    temp_paths = [image_paths[i] for i in temp_idx]
    temp_species = [species_names[i] for i in temp_idx]

    # Step 2: 15% val, 15% eval (split 30% temp set into 50/50)
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

    # Create datasets
    train_dataset = FathomNetTaxonomyDataset(train_paths, train_species, taxonomy_df, encoders, transform=train_transform)
    val_dataset = FathomNetTaxonomyDataset(val_paths, val_species, taxonomy_df, encoders, transform=val_transform)
    eval_dataset = FathomNetTaxonomyDataset(eval_paths, eval_species, taxonomy_df, encoders, transform=val_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                             num_workers=Config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                           num_workers=Config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)
    eval_loader = DataLoader(eval_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, collate_fn=collate_fn, pin_memory=True)

    print(f"Train dataset: {len(train_dataset)} images")
    print(f"Validation dataset: {len(val_dataset)} images")
    print(f"Internal evaluation dataset: {len(eval_dataset)} images")

    return train_loader, val_loader, eval_loader, taxonomy_df, encoders, class_counts, id_to_name, name_to_id


def train_model(train_loader, val_loader, class_counts, lr=Config.LEARNING_RATE,
                backbone_name=Config.BACKBONE, max_epochs=Config.MAX_EPOCHS):
    """
    Train the model using PyTorch Lightning.

    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        class_counts: Dict of class counts per taxonomic level
        lr: Learning rate
        backbone_name: Name of backbone model
        max_epochs: Maximum number of training epochs

    Returns:
        best_model_path: Path to the best model checkpoint
    """
    print("Starting model training...")

    # Create model
    model = TaxonomyAwareClassifier(class_counts=class_counts, lr=lr, backbone_name=backbone_name)

    # Create callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=Config.OUTPUT_DIR,
        filename='fathomnet-{epoch:02d}-{val_loss:.4f}',
        save_top_k=3,
        monitor='val_loss',
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min'
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=1,
        precision=16,  # Mixed precision for faster training
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        log_every_n_steps=10,
        default_root_dir=Config.OUTPUT_DIR
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    # Return best model path
    return checkpoint_callback.best_model_path
