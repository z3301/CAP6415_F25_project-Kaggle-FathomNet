# CRITICAL: Baseline Model Has Same Collapse Issue

**Date**: 2025-12-03
**Finding**: Single-scale baseline model shows identical collapse to multi-scale models

---

## Baseline Epoch 0 Results

```
val_loss=3.610
val_kingdom_acc=1.000  ✓
val_phylum_acc=0.304   ❌ (predicting mostly Cnidaria)
val_class_acc=0.152    ❌ (predicting mostly Octocorallia)
val_order_acc=0.127    ❌
val_family_acc=0.0506  ❌
val_genus_acc=0.0253   ❌
val_species_acc=0.0127 ❌ (RANDOM GUESSING)
```

##Identical to Multi-Scale Results

| Model | Species Acc | Val Loss |
|-------|-------------|----------|
| Exp 1a (2 scales) | 1.27% | 3.602 |
| Exp 1b (3 scales) | 1.27% | 3.602 |
| **Baseline (1 scale)** | **1.27%** | **3.609** |

---

## Conclusion

**The problem is NOT in multi-scale fusion.**
**The problem IS in the core hierarchical model or data.**

---

## Root Cause Possibilities (Ranked by Likelihood)

### 1. Hierarchical Conditioning Bug (HIGH)
The sequential conditioning logic has a bug that prevents proper gradient flow to later taxonomic levels.

**Evidence**:
- Kingdom: 100% (no conditioning, works)
- Phylum: 30% (one level of conditioning, partially works)
- Species: 1.27% (six levels of conditioning, completely collapsed)

**Check**: `src/model.py` lines where previous level output feeds into next level

### 2. Loss Weighting Imbalance (HIGH)
All taxonomic levels are weighted equally in the total loss, but they have very different numbers of classes (2 vs 80). The model may be ignoring fine-grained levels.

**Evidence**:
- All three models (1a, 1b, baseline) converge to val_loss ~3.60
- This loss might be dominated by easy levels (kingdom, phylum)

**Check**: How individual level losses are combined

### 3. Label Encoding Bug (MEDIUM)
Mismatch between how labels are encoded in data vs how model expects them.

**Evidence**:
- Consistent across all models suggests systematic issue
- Same classes predicted at each level

**Check**: Print actual label tensors and predictions

### 4. Gradient Vanishing Through Conditioning (MEDIUM)
The sequential dependencies create a very deep effective network, causing vanishing gradients for deeper levels.

**Evidence**:
- Performance degrades sharply after phylum
- Loss is reasonable but accuracy is terrible

**Check**: Gradient norms at each taxonomic level

---

## Next Steps (Immediate)

### Step 1: Check Loss Calculation (5 min)

Add debugging to training:
```python
# In training_step
for level in self.taxonomy_levels:
    print(f"{level}: loss={loss_dict[level].item():.4f}")
print(f"Total loss: {total_loss.item():.4f}")
```

Expected: Should see if species loss is tiny compared to kingdom loss

### Step 2: Test Without Conditioning (15 min)

Temporarily remove sequential conditioning:
```python
# Comment out the conditioning parts
# Train just independent heads
```

If this works → conditioning is the bug
If this fails → data or loss issue

### Step 3: Check Label Distribution (10 min)

```python
# Print batch labels
for batch in train_loader:
    _, labels = batch
    for level in taxonomy_levels:
        print(f"{level}: unique={len(torch.unique(labels[level]))}, values={labels[level][:10]}")
    break
```

Expected: Should see variety of labels, not just one class

---

## Quick Fixes to Try

### Fix 1: Remove Hierarchical Conditioning
Train independent classifiers for each level. If this works, we know conditioning is the problem.

### Fix 2: Weight Losses by Inverse Class Count
```python
level_weights = {
    'kingdom': 1.0 / 2,
    'phylum': 1.0 / 8,
    'class': 1.0 / 22,
    'order': 1.0 / 43,
    'family': 1.0 / 66,
    'genus': 1.0 / 75,
    'species': 1.0 / 80,
}
# Normalize
total = sum(level_weights.values())
level_weights = {k: v/total for k, v in level_weights.items()}

# In loss calculation:
total_loss = sum(level_weights[level] * loss_dict[level] for level in levels)
```

### Fix 3: Simplify Conditioning
Instead of concatenating previous level output, just add it:
```python
# Before:
combined = torch.cat([prev_output, features], dim=1)

# After:
combined = features + self.projection(prev_output)
```

---

## Timeline Impact

**Current time**: ~1 hour into baseline training

**Options**:

1. **Let baseline finish** (29 more epochs, ~30 min)
   - See if it improves over epochs
   - Probably won't - epoch 0 is already collapsed

2. **Kill and debug now** (recommended)
   - Add debug prints
   - Test fixes immediately
   - Could have working model by end of night

**Recommendation**: Kill baseline, add debugging, try Fix #1 (no conditioning) immediately.

---

## Success Criteria

A fix is working if:
- Species accuracy > 10% (vs current 1.27%)
- Each taxonomic level improves over previous
- Not all predicting same class

Target for moving forward:
- Species accuracy > 20%
- Validation score < 2.50

---

**Status**: Awaiting decision on whether to debug now or let baseline finish training.
