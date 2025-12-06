#!/usr/bin/env python3
"""
Debug script to compare model behavior between notebook and script versions.
"""

import os
import torch
import torch.nn as nn
import timm


from src.config import Config
from src.train import prepare_data
from src.models.model import TaxonomyAwareClassifier

# Test 1: Check backbone model name
print("="*60)
print("TEST 1: Backbone Configuration")
print("="*60)
print(f"Config.BACKBONE = {Config.BACKBONE}")

# Try to create the backbone
try:
    backbone_config = timm.create_model(
        Config.BACKBONE,
        pretrained=False,
        num_classes=0
    )
    print(f"✓ Backbone '{Config.BACKBONE}' loaded successfully")
    print(f"  - num_features: {backbone_config.num_features}")
except Exception as e:
    print(f"✗ Failed to load backbone: {e}")

# Test 2: Check data loading
print("\n" + "="*60)
print("TEST 2: Data Loading")
print("="*60)
train_loader, val_loader, eval_loader, taxonomy_df, encoders, class_counts, id_to_name, name_to_id = prepare_data()

# Get one batch
batch = next(iter(train_loader))
images, labels = batch

print(f"Batch images shape: {images.shape}")
print(f"Labels keys: {labels.keys()}")
for level in Config.TAXONOMY_LEVELS:
    if level in labels:
        print(f"  - {level}: shape={labels[level].shape}, unique values={len(torch.unique(labels[level]))}")

# Test 3: Model forward pass
print("\n" + "="*60)
print("TEST 3: Model Forward Pass")
print("="*60)

model = TaxonomyAwareClassifier(class_counts=class_counts)
model.eval()

with torch.no_grad():
    outputs = model(images)

print(f"Output keys: {outputs.keys()}")
for level in Config.TAXONOMY_LEVELS:
    if level in outputs:
        logits = outputs[level]
        probs = torch.softmax(logits, dim=1)
        print(f"  - {level}: logits shape={logits.shape}, max_prob={probs.max().item():.4f}")

# Test 4: Loss calculation
print("\n" + "="*60)
print("TEST 4: Loss Calculation")
print("="*60)

loss, level_losses = model.hierarchical_loss(outputs, labels)
print(f"Total loss: {loss.item():.4f}")
for level, level_loss in level_losses.items():
    print(f"  - {level}: {level_loss.item():.4f}")

# Test 5: Check predictions
print("\n" + "="*60)
print("TEST 5: Predictions")
print("="*60)

for level in Config.TAXONOMY_LEVELS:
    if level in outputs and level in labels:
        preds = torch.argmax(outputs[level], dim=1)
        targets = labels[level]
        acc = (preds == targets).float().mean()

        # Check distribution
        pred_counts = torch.bincount(preds, minlength=class_counts[level])
        pred_unique = (pred_counts > 0).sum().item()

        print(f"  - {level}:")
        print(f"      Accuracy: {acc.item():.4f}")
        print(f"      Unique predictions: {pred_unique} / {class_counts[level]}")
        print(f"      Most common pred: class {preds.mode().values.item()} (count={pred_counts.max().item()})")

print("\n" + "="*60)
print("Debug complete!")
print("="*60)
