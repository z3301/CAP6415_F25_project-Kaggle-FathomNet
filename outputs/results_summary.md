# Multi-Scale Experiment Results Summary

**Date**: 2025-12-03
**Status**: ⚠️ CRITICAL ISSUE - Model Collapse Detected

---

## Experiment Results

### Exp 1a: 2-Scale Model (ROI + 3× context)
- **Architecture**: 2× ConvNeXtV2-Base encoders
- **Parameters**: 181M total (175M backbone + 6M heads)
- **Training**: Completed at epoch 14 (early stopping)
- **Best checkpoint**: `exp1a_2scales-epoch=14-val_loss=3.6019.ckpt`

**Validation Accuracy by Taxonomic Level:**
| Level   | Accuracy | Notes |
|---------|----------|-------|
| Kingdom | 1.0000   | Perfect (only 1 class: Animalia) |
| Phylum  | 0.3038   | Predicting mostly Cnidaria |
| Class   | 0.1519   | Predicting mostly Octocorallia |
| Order   | 0.0352   | Severe collapse |
| Family  | 0.0239   | Severe collapse |
| Genus   | 0.0127   | Severe collapse |
| **Species** | **0.0127** | **CRITICAL: Random guessing level** |

### Exp 1b: 3-Scale Model (ROI + 3× + 5× context)
- **Architecture**: 3× ConvNeXtV2-Base encoders
- **Parameters**: 269M total (263M backbone + 6M heads)
- **Training**: Completed at epoch 14 (early stopping)
- **Best checkpoint**: `exp1b_3scales-epoch=14-val_loss=3.6020.ckpt`

**Validation Accuracy by Taxonomic Level:**
| Level   | Accuracy | Notes |
|---------|----------|-------|
| Kingdom | 1.0000   | Perfect (only 1 class: Animalia) |
| Phylum  | 0.3038   | Predicting mostly Cnidaria |
| Class   | 0.1519   | Predicting mostly Octocorallia |
| Order   | 0.0352   | Severe collapse |
| Family  | 0.0239   | Severe collapse |
| Genus   | 0.0127   | Severe collapse |
| **Species** | **0.0127** | **CRITICAL: Identical to 1a, random level** |

---

## Problem Analysis

### Critical Issue: Model Collapse

Both models have collapsed to predicting primarily one class at each taxonomic level:
- **Phylum**: Cnidaria (30% of validation set)
- **Class**: Octocorallia (15% of validation set)
- **Species**: Essentially random guessing (1.27% vs 1.25% random baseline)

### Identical Results

The fact that Exp 1a (2 scales) and Exp 1b (3 scales) have **identical accuracy** suggests:
1. The issue is not in the multi-scale architecture itself
2. Something fundamental is wrong with the training process
3. Both models converged to the same degenerate solution

### Validation Loss vs Accuracy Disconnect

- Validation loss: 3.60 (both experiments)
- Validation accuracy: 1.27% (species level)

This low accuracy despite "reasonable" loss suggests:
- Loss is not properly reflecting classification performance
- Possible issue with hierarchical conditioning
- Label encoding or loss weighting problems

---

## Possible Root Causes

### 1. Hierarchical Conditioning Bug
The sequential conditioning (species depends on genus, genus on family, etc.) may have a bug causing gradients to not flow properly to lower levels.

### 2. Class Imbalance Not Addressed
- 80 species classes with varying support
- No class balancing or weighted loss
- Model defaulting to majority class prediction

### 3. Learning Rate Too High
- LR: 3e-4 may be too high for fine-tuning ConvNeXt
- Models converged quickly (14 epochs) to poor solution

### 4. Label Encoding Issues
- Possible mismatch between encoded labels and model outputs
- Hierarchical label consistency not enforced

### 5. Gradient Flow Issues
- 181M-269M parameters
- Potential vanishing gradients through hierarchical heads
- Batch normalization or layer norm issues

---

## Comparison to Baseline

From previous runs (single-scale ConvNeXtV2-Base):
- **Baseline validation score**: 2.74 (competition metric)
- **Target score**: < 2.50 (30% gap closure)

Current multi-scale performance is **significantly worse** than baseline, not better.

---

## Next Steps (Priority Order)

### Immediate: Debug and Fix (Day 3)

1. **Check training logs for Exp 1a/1b**:
   - Look at training accuracy vs validation accuracy
   - Check if training accuracy is also collapsed
   - Review loss curves for all taxonomic levels

2. **Verify label encoding**:
   - Print sample batch labels vs predictions
   - Ensure taxonomic hierarchy is consistent
   - Check for off-by-one errors in label indices

3. **Test single-scale model first**:
   - Train a single ConvNeXtV2 model without multi-scale
   - Verify hierarchical heads work correctly
   - Isolate whether issue is in multi-scale or base architecture

4. **Review hierarchical head implementation**:
   - Check gradient flow through conditioning
   - Verify sequential conditioning logic
   - Test with and without conditioning

### Fixes to Try:

1. **Lower learning rate**: Try 1e-4 or 5e-5
2. **Class-weighted loss**: Weight species loss by inverse class frequency
3. **Longer training**: Remove early stopping, train for full 50 epochs
4. **Simpler architecture first**: Single-scale with hierarchical heads
5. **Different optimizer**: Try SGD with momentum instead of AdamW
6. **Gradient clipping**: Add gradient norm clipping
7. **Warm-up schedule**: Add learning rate warm-up

---

## Timeline Impact

**Original plan**: Days 3-4 were for attention mechanism implementation.

**Revised plan**:
- **Day 3 (today)**: Debug and fix model collapse
- **Day 4**: Re-run Exp 1a/1b with fixes
- **Day 5**: Attention mechanism (compressed)
- **Day 6-7**: Evaluation, analysis, report

⚠️ **Risk**: If debugging takes longer than 1 day, attention mechanism may need to be descoped or simplified.

---

## Recommendations for User

1. **Prioritize getting any working multi-scale model** over perfect architecture
2. **Consider simpler baseline**: Single-scale hierarchical model that works
3. **If time-constrained**: Focus on getting good results from simpler model rather than complex attention
4. **Document the debugging process**: Show systematic problem-solving for CAP6415

---

## Files for Investigation

1. Training logs:
   - `outputs/exp1a_2scales/lightning_logs/version_0/`
   - `outputs/exp1b_3scales/lightning_logs/version_0/`

2. Model code:
   - `src/model_multiscale.py` - Multi-scale architecture
   - `train_multiscale.py` - Training loop
   - `src/data_multiscale.py` - Data loading

3. Evaluation logs:
   - `outputs/exp1a_eval.log`
   - `outputs/exp1b_eval.log`

---

**Status**: Awaiting user input on how to proceed.
