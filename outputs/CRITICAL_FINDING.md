# CRITICAL: All Models Fail to Train - Including Independent Heads

**Date**: 2025-12-03
**Status**: 🚨 SEVERE ISSUE - Even simplest architecture collapses

---

## Test Results

### All Models Show Identical Collapse

| Model | Architecture | Species Acc | Status |
|-------|-------------|-------------|--------|
| Exp 1a | 2-scale multi-scale | 1.27% | ❌ FAILED |
| Exp 1b | 3-scale multi-scale | 1.27% | ❌ FAILED |
| Baseline | Single-scale hierarchical | 1.27% | ❌ FAILED |
| **Simple (Independent)** | **No conditioning** | **1.27%** | **❌ FAILED** |

### Independent Heads Test (Just Completed)

Trained `SimpleTaxonomyClassifier` with **completely independent heads** (NO hierarchical conditioning):

```
Epoch 0: val_species_acc=0.0127
Epoch 1: val_species_acc=0.0127
Epoch 2: val_species_acc=0.0127
Epoch 3: val_species_acc=0.0127
Epoch 4: val_species_acc=0.0127
```

**The model did not learn AT ALL in 5 epochs!**

---

## What This Means

1. **NOT a hierarchical conditioning issue** - Independent heads also fail
2. **NOT a multi-scale issue** - Single-scale also fails
3. **NOT a gradient flow issue** - Simplest architecture also fails
4. **NOT a model initialization issue** - Different architectures, same failure

---

## The Real Problem

Since the notebook achieved 89% species accuracy with identical code, the issue must be:

### Hypothesis 1: Data Loading/Label Bug (MOST LIKELY)

Something is wrong with how labels are being encoded or loaded in the scripts vs notebook.

**Evidence:**
- All models predict only ONE class per taxonomic level
- This happens even at initialization
- Training does not improve anything
- Loss is high (~21) but not decreasing

**Possible causes:**
- Labels are all mapped to the same ID
- Label tensor shape mismatch
- Incorrect target encoding in dataloader

### Hypothesis 2: Loss Calculation Bug

The loss is being calculated incorrectly, so gradients don't flow properly.

**Evidence:**
- Loss stays at ~21 across all epochs
- Loss doesn't decrease with training
- All taxonomic levels show identical behavior

### Hypothesis 3: Optimizer/Learning Rate Issue

Something prevents the model from actually updating weights.

**Evidence:**
- Train loss also doesn't decrease
- No improvement across multiple epochs
- Happens across all model architectures

---

## Immediate Debug Steps

### Step 1: Check Label Encoding

```python
# In debug script
for batch in train_loader:
    images, labels = batch
    print("Label shapes:", {k: v.shape for k, v in labels.items()})
    print("Label ranges:", {k: (v.min().item(), v.max().item()) for k, v in labels.items()})
    print("Unique labels per level:")
    for level in Config.TAXONOMY_LEVELS:
        unique = torch.unique(labels[level])
        print(f"  {level}: {len(unique)} unique values out of {class_counts[level]} classes")
        print(f"    Values: {unique[:10].tolist()}")
    break
```

### Step 2: Check Loss Gradients

```python
# After loss.backward()
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad norm = {param.grad.norm().item():.6f}")
```

### Step 3: Check Weight Updates

```python
# Before training
initial_weights = {name: param.clone() for name, param in model.named_parameters()}

# After 1 epoch
for name, param in model.named_parameters():
    diff = (param - initial_weights[name]).abs().mean()
    print(f"{name}: weight change = {diff.item():.6f}")
```

### Step 4: Compare with Notebook Data Loading

Look at exactly how the notebook loads and processes labels vs our scripts.

---

## Why This is Critical

1. **Blocks all progress** - Can't test multi-scale or attention if basic training doesn't work
2. **Time-sensitive** - Day 3 evening, need working model by Day 4
3. **Identical code** - Notebook works, scripts don't - suggests subtle bug

---

## Next Action

**STOP ALL TRAINING** and focus on debugging the data/label pipeline.

The issue is NOT in the model architecture. It's somewhere in:
- Data loading ([src/data.py](src/data.py))
- Label encoding ([src/taxonomy.py](src/taxonomy.py))
- Training loop ([src/train.py](src/train.py))
- Or a subtle PyTorch/Lightning configuration issue

---

**Time to debug**: Estimated 30-60 minutes to identify and fix.

**Priority**: CRITICAL - This blocks everything else.
