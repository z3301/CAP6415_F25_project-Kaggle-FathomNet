#!/usr/bin/env python3
"""
Pure PyTorch training (no Lightning) to test if model can learn.
"""

import os
import sys
import torch
import torch.nn as nn
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.config import Config
from src.train import prepare_data
from src.models.model_simple import SimpleTaxonomyClassifier

print("="*80)
print("PURE PYTORCH TRAINING TEST (NO LIGHTNING)")
print("="*80)

# Prepare data
train_loader, val_loader, eval_loader, taxonomy_df, encoders, class_counts, id_to_name, name_to_id = prepare_data()

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")

model = SimpleTaxonomyClassifier(class_counts=class_counts, lr=3e-4)
model.to(device)

# Optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

# Training loop
num_epochs = 10
print(f"\nTraining for {num_epochs} epochs...")

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0.0
    train_steps = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        images, targets = batch
        images = images.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}

        optimizer.zero_grad()
        outputs = model(images)
        loss, level_losses = model.hierarchical_loss(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_steps += 1

        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_train_loss = train_loss / train_steps

    # Validation
    model.eval()
    val_loss = 0.0
    val_steps = 0
    val_correct = {level: 0 for level in Config.TAXONOMY_LEVELS}
    val_total = 0

    with torch.no_grad():
        for batch in val_loader:
            images, targets = batch
            images = images.to(device)
            targets = {k: v.to(device) for k, v in targets.items()}

            outputs = model(images)
            loss, level_losses = model.hierarchical_loss(outputs, targets)

            val_loss += loss.item()
            val_steps += 1

            # Calculate accuracies
            batch_size = images.shape[0]
            val_total += batch_size

            for level in Config.TAXONOMY_LEVELS:
                if level in outputs and level in targets:
                    preds = torch.argmax(outputs[level], dim=1)
                    val_correct[level] += (preds == targets[level]).sum().item()

    avg_val_loss = val_loss / val_steps
    val_accs = {level: val_correct[level] / val_total for level in Config.TAXONOMY_LEVELS}

    print(f"\nEpoch {epoch}:")
    print(f"  Train Loss: {avg_train_loss:.4f}")
    print(f"  Val Loss: {avg_val_loss:.4f}")
    print(f"  Val Accuracies:")
    for level in Config.TAXONOMY_LEVELS:
        print(f"    {level}: {val_accs[level]:.4f}")
    print()

print("="*80)
print("TRAINING COMPLETE")
print("="*80)
