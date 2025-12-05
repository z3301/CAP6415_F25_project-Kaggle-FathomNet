# Debugging Plan: Model Collapse Investigation

**Date**: 2025-12-03
**Issue**: Both multi-scale experiments (Exp 1a & 1b) collapsed to 1.27% species accuracy

---

## Step 1: Test Baseline (In Progress)

**Goal**: Verify single-scale hierarchical model works correctly

**Action**: Training baseline single-scale model on GPU 2
- Script: `train_baseline.py`
- Configuration:
  - Model: Single ConvNeXtV2-Base (92M params)
  - Hierarchical heads with sequential conditioning
  - LR: 3e-4, Batch: 32, Epochs: 30, Patience: 10
- Output: `outputs/baseline_single_scale/`
- Log: `outputs/baseline_training.log`

**Expected Outcomes:**

### If Baseline Works (species acc > 10%):
→ Problem is in **multi-scale architecture**
- Check how scale features are fused
- Verify all encoders receive gradients
- Test with 2 scales first, then 3 scales

### If Baseline Also Collapses (species acc < 5%):
→ Problem is in **hierarchical conditioning** or **data pipeline**
- Bug in sequential conditioning logic
- Label encoding mismatch
- Loss calculation issue
- Data augmentation too aggressive

---

## Step 2: Diagnosis Based on Baseline Results

### Scenario A: Baseline Works

**Root cause**: Multi-scale fusion or encoder synchronization

**Fixes to try:**
1. Check projection layer dimensions match
2. Verify all scale inputs reach their encoders
3. Add gradient monitoring per encoder
4. Test concatenation vs other fusion methods
5. Try training one encoder at a time

**Quick test**:
```python
# In model forward, print:
print(f"Scale 1.0 features: {scale_features['scale_1.0'].shape}")
print(f"Scale 3.0 features: {scale_features['scale_3.0'].shape}")
print(f"Fused features: {fused.shape}")
```

---

### Scenario B: Baseline Also Fails

**Root cause**: Hierarchical conditioning or core architecture

**Debugging steps:**

1. **Check label encoding**:
```python
# Print sample from dataloader
for batch in train_loader:
    images, labels = batch
    print("Labels:", labels)
    break
```

2. **Test without hierarchical conditioning**:
   - Train species head independently
   - Remove sequential conditioning temporarily
   - If this works → bug in conditioning logic

3. **Verify loss calculation**:
```python
# In training_step, add:
for level in self.taxonomy_levels:
    print(f"{level}: logits shape {outputs[level].shape}, targets shape {targets[level].shape}")
    print(f"{level} loss: {loss_dict[level].item():.4f}")
```

4. **Check for label leakage**:
   - Ensure train/val/eval splits don't overlap
   - Verify stratification worked correctly

5. **Test with subset**:
   - Train on just 2-3 species
   - If this works → class imbalance issue

---

## Step 3: Implement Fixes

### Common Fixes

1. **Lower learning rate**: 1e-4 or 5e-5
2. **Add gradient clipping**: `trainer.gradient_clip_val=1.0`
3. **Warmup schedule**: Linear warmup for first 5 epochs
4. **Class weights**: Weight loss by inverse frequency
5. **Reduce model capacity**: Freeze early ConvNeXt layers
6. **Better initialization**: Xavier/Kaiming for heads

### Code Changes Needed

**If hierarchical conditioning is the issue**:
```python
# src/model.py or src/model_multiscale.py
# Change from hard conditioning to soft conditioning:

# Current (hard):
combined = torch.cat([prev_level_output, current_features], dim=1)

# Try (soft with learned gate):
gate = torch.sigmoid(self.gate_layer(prev_level_output))
combined = gate * prev_level_output + (1 - gate) * current_features
```

**If class imbalance is the issue**:
```python
# Calculate class weights
from sklearn.utils.class_weight import compute_class_weight
weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights))
```

---

## Step 4: Re-run Experiments

Once fixes identified:

1. **Test fix on baseline** (30 min - 1 hour)
2. **Apply to multi-scale** (modify `train_multiscale.py`)
3. **Re-run Exp 1a** (2-4 hours)
4. **If 1a works, run Exp 1b** (2-4 hours)

---

## Timeline Impact

**Current**: End of Day 3 (evening)

**Best case** (baseline works, simple multi-scale fix):
- Tonight: Identify and fix multi-scale issue
- Day 4 morning: Re-run Exp 1a/1b
- Day 4 afternoon: Results + start attention
- Day 5-7: Attention, eval, report ✅

**Moderate case** (hierarchical issue, need fixes):
- Tonight: Identify hierarchical bug
- Day 4: Fix and test baseline
- Day 4 evening: Re-run multi-scale
- Day 5: Get results, simplify attention
- Day 6-7: Quick attention test, report ⚠️

**Worst case** (fundamental architecture issue):
- Day 4-5: Major rewrites
- Day 6: Get working model
- Day 7: Report with limited results ❌
- May need to descope attention entirely

---

## Contingency Plans

### If can't fix by end of Day 4:

**Plan B**: Use working components
1. Report on single-scale baseline (if it works)
2. Document debugging process thoroughly
3. Show attempted fixes and analysis
4. Discuss why multi-scale failed (for CAP6415 learning)

**Plan C**: Different approach
1. Use pre-trained DinoV2 instead of ConvNeXt
2. Simpler classifier (no hierarchical conditioning)
3. Focus on getting any improvement over baseline
4. Document experiments and lessons learned

---

## Success Criteria

**Minimum for CAP6415**:
- Working model with > 10% species accuracy
- Thorough documentation of debugging process
- Clear explanation of approach and challenges
- Honest results and analysis

**Target**:
- Multi-scale model with > 20% species accuracy
- Validation score < 2.50
- Attention mechanism implemented (even if simple)

**Stretch**:
- Validation score < 2.30
- Meaningful improvements from multi-scale + attention

---

## Next Check-in

**Time**: ~30 minutes (check baseline progress)
- If epoch 0-1 shows reasonable training loss → good sign
- If stuck at high loss or NaN → dig deeper immediately

**Monitor**: `tail -f outputs/baseline_training.log`
