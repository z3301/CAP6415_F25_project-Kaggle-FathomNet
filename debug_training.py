#!/usr/bin/env python3
"""
Comprehensive debugging script to find why training fails.

Tests:
1. Label encoding and data loading
2. Loss gradients
3. Weight updates
"""

import torch
import torch.nn as nn
from copy import deepcopy

from src.config import Config
from src.train import prepare_data
from src.model_simple import SimpleTaxonomyClassifier

print("="*80)
print("COMPREHENSIVE TRAINING DEBUG")
print("="*80)

# Prepare data
print("\n" + "="*80)
print("STEP 0: Preparing Data")
print("="*80)
train_loader, val_loader, eval_loader, taxonomy_df, encoders, class_counts, id_to_name, name_to_id = prepare_data()

print("\n" + "="*80)
print("STEP 1: CHECK LABEL ENCODING")
print("="*80)

# Get first batch
batch = next(iter(train_loader))
images, labels = batch

print(f"\nBatch size: {images.shape[0]}")
print(f"\nLabel shapes:")
for level in Config.TAXONOMY_LEVELS:
    if level in labels:
        print(f"  {level}: {labels[level].shape}")

print(f"\nLabel ranges (min, max):")
for level in Config.TAXONOMY_LEVELS:
    if level in labels:
        print(f"  {level}: ({labels[level].min().item()}, {labels[level].max().item()}) out of {class_counts[level]} classes")

print(f"\nUnique labels per level in this batch:")
for level in Config.TAXONOMY_LEVELS:
    if level in labels:
        unique = torch.unique(labels[level])
        print(f"  {level}: {len(unique)} unique values")
        print(f"    Values: {unique.tolist()[:20]}")

print(f"\nClass distribution in this batch:")
for level in Config.TAXONOMY_LEVELS:
    if level in labels:
        counts = torch.bincount(labels[level], minlength=class_counts[level])
        non_zero = (counts > 0).sum().item()
        print(f"  {level}: {non_zero}/{class_counts[level]} classes present")
        print(f"    Most common: class {counts.argmax().item()} (count={counts.max().item()})")

# Check multiple batches
print(f"\nChecking label diversity across 10 batches:")
all_labels = {level: [] for level in Config.TAXONOMY_LEVELS}
for i, batch in enumerate(train_loader):
    if i >= 10:
        break
    _, labels = batch
    for level in Config.TAXONOMY_LEVELS:
        if level in labels:
            all_labels[level].append(labels[level])

for level in Config.TAXONOMY_LEVELS:
    if all_labels[level]:
        all_level_labels = torch.cat(all_labels[level])
        unique = torch.unique(all_level_labels)
        print(f"  {level}: {len(unique)} unique labels across 10 batches")

print("\n" + "="*80)
print("STEP 2: CHECK MODEL FORWARD PASS AND LOSS GRADIENTS")
print("="*80)

# Create model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleTaxonomyClassifier(class_counts=class_counts)
model.to(device)
model.train()

# Get a batch
batch = next(iter(train_loader))
images, targets = batch
images = images.to(device)
targets = {k: v.to(device) for k, v in targets.items()}

print(f"\nForward pass...")
outputs = model(images)

print(f"\nOutput shapes:")
for level in Config.TAXONOMY_LEVELS:
    if level in outputs:
        print(f"  {level}: {outputs[level].shape}")

print(f"\nCalculating loss...")
total_loss, losses = model.hierarchical_loss(outputs, targets)

print(f"\nTotal loss: {total_loss.item():.4f}")
print(f"Individual losses:")
for level, loss in losses.items():
    print(f"  {level}: {loss.item():.4f}")

print(f"\nBackpropagating...")
total_loss.backward()

print(f"\nGradient norms (first 20 parameters):")
grad_norms = []
for i, (name, param) in enumerate(model.named_parameters()):
    if i >= 20:
        break
    if param.grad is not None:
        grad_norm = param.grad.norm().item()
        grad_norms.append(grad_norm)
        print(f"  {name}: {grad_norm:.6f}")
    else:
        print(f"  {name}: NO GRADIENT")

if grad_norms:
    print(f"\nGradient statistics:")
    print(f"  Mean: {sum(grad_norms)/len(grad_norms):.6f}")
    print(f"  Min: {min(grad_norms):.6f}")
    print(f"  Max: {max(grad_norms):.6f}")
    print(f"  Non-zero: {sum(1 for g in grad_norms if g > 1e-10)}/{len(grad_norms)}")

print("\n" + "="*80)
print("STEP 3: CHECK WEIGHT UPDATES")
print("="*80)

# Create fresh model and optimizer
model = SimpleTaxonomyClassifier(class_counts=class_counts)
model.to(device)
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

# Store initial weights
print("\nStoring initial weights...")
initial_weights = {}
for name, param in model.named_parameters():
    initial_weights[name] = param.data.clone()

print(f"\nSample of initial weights (first 5 parameters):")
for i, (name, param) in enumerate(model.named_parameters()):
    if i >= 5:
        break
    print(f"  {name}: mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")

# Train for one batch
print(f"\nTraining on one batch...")
batch = next(iter(train_loader))
images, targets = batch
images = images.to(device)
targets = {k: v.to(device) for k, v in targets.items()}

optimizer.zero_grad()
outputs = model(images)
total_loss, losses = model.hierarchical_loss(outputs, targets)
print(f"Loss before update: {total_loss.item():.4f}")

total_loss.backward()
optimizer.step()

# Check weight changes
print(f"\nWeight changes after 1 optimizer step (first 10 parameters):")
weight_changes = []
for i, (name, param) in enumerate(model.named_parameters()):
    if i >= 10:
        break
    diff = (param.data - initial_weights[name]).abs().mean().item()
    weight_changes.append(diff)
    print(f"  {name}: {diff:.8f}")

if weight_changes:
    print(f"\nWeight change statistics:")
    print(f"  Mean: {sum(weight_changes)/len(weight_changes):.8f}")
    print(f"  Min: {min(weight_changes):.8f}")
    print(f"  Max: {max(weight_changes):.8f}")
    print(f"  Non-zero: {sum(1 for w in weight_changes if w > 1e-10)}/{len(weight_changes)}")

# Forward pass again to see if loss changed
with torch.no_grad():
    outputs = model(images)
    total_loss, losses = model.hierarchical_loss(outputs, targets)
    print(f"\nLoss after update: {total_loss.item():.4f}")

print(f"\n✓ Weights ARE updating!" if any(w > 1e-8 for w in weight_changes) else "\n✗ Weights NOT updating!")

print("\n" + "="*80)
print("STEP 4: TRAIN FOR 3 STEPS AND CHECK LOSS PROGRESSION")
print("="*80)

model.train()
losses_over_time = []

for step in range(3):
    batch = next(iter(train_loader))
    images, targets = batch
    images = images.to(device)
    targets = {k: v.to(device) for k, v in targets.items()}

    optimizer.zero_grad()
    outputs = model(images)
    total_loss, level_losses = model.hierarchical_loss(outputs, targets)
    total_loss.backward()
    optimizer.step()

    losses_over_time.append(total_loss.item())
    print(f"Step {step}: loss={total_loss.item():.4f}")

print(f"\nLoss progression:")
for i, loss in enumerate(losses_over_time):
    change = "" if i == 0 else f" (change: {loss - losses_over_time[i-1]:+.4f})"
    print(f"  Step {i}: {loss:.4f}{change}")

if len(losses_over_time) > 1:
    total_change = losses_over_time[-1] - losses_over_time[0]
    print(f"\nTotal loss change: {total_change:+.4f}")
    print(f"✓ Loss is decreasing!" if total_change < -0.01 else "✗ Loss is NOT decreasing significantly")

print("\n" + "="*80)
print("STEP 5: CHECK PREDICTIONS")
print("="*80)

model.eval()
with torch.no_grad():
    batch = next(iter(val_loader))
    images, targets = batch
    images = images.to(device)
    targets = {k: v.to(device) for k, v in targets.items()}

    outputs = model(images)

    print(f"\nPrediction diversity (validation set):")
    for level in Config.TAXONOMY_LEVELS:
        if level in outputs and level in targets:
            preds = torch.argmax(outputs[level], dim=1)
            unique_preds = len(torch.unique(preds))
            acc = (preds == targets[level]).float().mean().item()

            print(f"  {level}:")
            print(f"    Accuracy: {acc:.4f}")
            print(f"    Unique predictions: {unique_preds}/{class_counts[level]}")
            print(f"    Most common prediction: class {preds.mode().values.item()}")

print("\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80)
