# ROOT CAUSE: Model Collapse from Sequential Conditioning

**Date**: 2025-12-03
**Status**: CRITICAL BUG IDENTIFIED

---

## The Problem

All models (Exp 1a, 1b, baseline, simple) collapse to predicting a single class at each taxonomic level, even when **untrained**!

###Debug Results (Untrained Model)

```
Model Forward Pass (random initialization):
- kingdom: Unique predictions: 1 / 2   (all 32 samples → class 1)
- phylum: Unique predictions: 1 / 8    (all 32 samples → class 5)
- class: Unique predictions: 1 / 22    (all 32 samples → class 10)
- species: Unique predictions: 1 / 80  (all 32 samples → class 11)
```

**The model predicts ONE class for everything BEFORE training even starts!**

---

## Root Cause: Sequential Conditioning with Softmax

### The Problematic Forward Pass

From `src/model.py` lines 91-137:

```python
def forward(self, x):
    features = self.feature_extractor(x)
    shared = self.shared_features(features)

    # Kingdom prediction
    kingdom_logits = self.kingdom_head(shared)
    kingdom_probs = torch.softmax(kingdom_logits, dim=1)  # ← Softmax creates peaked distribution

    # Phylum prediction (depends on kingdom)
    phylum_input = self.kingdom_to_phylum(kingdom_probs)  # ← Takes softmax as input
    phylum_feats = self.phylum_features(shared)
    phylum_logits = self.phylum_head(torch.cat([phylum_input, phylum_feats], dim=1))
    phylum_probs = torch.softmax(phylum_logits, dim=1)  # ← Another softmax

    # ... continues for all 7 levels
```

### Why This Causes Collapse

1. **Untrained logits are near-uniform**: `kingdom_logits` from random weights → all values ~0
2. **Softmax amplifies small differences**: Even tiny random variations get amplified to peaked distributions
3. **Sequential dependency**: `phylum` depends on `kingdom_probs`, which is already peaked
4. **Compounding effect**: By the time we reach `species`, we've gone through 6 levels of softmax amplification
5. **Gradient flow issues**: During backprop, gradients must flow through 6 sequential dependencies

### Example with Numbers

```python
# Untrained kingdom logits (near zero, slightly random)
kingdom_logits = [-0.01, 0.02]  # Shape: [2]

# Softmax amplifies the 0.03 difference
kingdom_probs = softmax([-0.01, 0.02]) = [0.492, 0.508]  # Slightly peaked

# Pass through linear layer
phylum_input = Linear(kingdom_probs)  # Some transformation

# This gets concatenated with features and passed through another head
# After 6 levels, even tiny biases compound into complete mode collapse
```

---

## Why the Notebook Worked

Looking at `hierarchical-classifier.ipynb` cell 31:
```
Epoch 14: val_species_acc=0.890
```

The notebook achieved 89% species accuracy with the EXACT SAME ARCHITECTURE!

**Possible reasons:**
1. **Different random seed**: Notebook may have had a lucky initialization
2. **Different torch/timm versions**: Subtle differences in initialization
3. **Training dynamics**: Perhaps the notebook started learning before collapse happened
4. **Batch effects**: Different batch composition in first few batches

However, our debug shows the **untrained model already collapsed**, suggesting our current random initialization is particularly bad.

---

## Solutions (Ordered by Simplicity)

### Solution 1: Stop Gradient Through Conditioning (RECOMMENDED)

Prevent gradients from flowing through the hierarchical connections during initial training:

```python
def forward(self, x):
    features = self.feature_extractor(x)
    shared = self.shared_features(features)

    kingdom_logits = self.kingdom_head(shared)
    kingdom_probs = torch.softmax(kingdom_logits, dim=1).detach()  # ← Stop gradient

    phylum_input = self.kingdom_to_phylum(kingdom_probs)
    phylum_feats = self.phylum_features(shared)
    phylum_logits = self.phylum_head(torch.cat([phylum_input, phylum_feats], dim=1))
    phylum_probs = torch.softmax(phylum_logits, dim=1).detach()  # ← Stop gradient

    # Continue for all levels
```

**Pros**:
- Simple one-line change per level
- Allows model to learn independent predictions first
- Can gradually enable gradients later

**Cons**:
- Defeats the purpose of hierarchical conditioning during training
- Essentially becomes independent heads

### Solution 2: Use Logits Instead of Softmax

Pass the raw logits instead of probabilities:

```python
kingdom_logits = self.kingdom_head(shared)
# Don't apply softmax!
phylum_input = self.kingdom_to_phylum(kingdom_logits)  # ← Use logits directly
```

**Pros**:
- Maintains hierarchical conditioning
- Avoids softmax amplification
- Better gradient flow

**Cons**:
- Changes the meaning of the conditioning
- May need to adjust layer sizes

### Solution 3: Add Temperature to Softmax

Make the softmax less peaked during early training:

```python
def forward(self, x, temperature=1.0):
    kingdom_logits = self.kingdom_head(shared)
    kingdom_probs = torch.softmax(kingdom_logits / temperature, dim=1)
```

Then train with `temperature=5.0` initially, anneal to `1.0`.

**Pros**:
- Maintains softmax semantics
- Gradually introduces sharpness

**Cons**:
- Adds training complexity
- Requires schedule tuning

### Solution 4: Better Initialization

Initialize the conditioning projection layers to near-zero:

```python
def _create_hierarchical_network(self, class_counts):
    # ... existing code ...

    # Initialize conditioning layers with small weights
    for layer in [self.kingdom_to_phylum, self.phylum_to_class, ...]:
        nn.init.xavier_uniform_(layer.weight, gain=0.01)
        nn.init.zeros_(layer.bias)
```

**Pros**:
- Doesn't change forward pass
- Makes conditioning weak initially

**Cons**:
- May not fully solve the problem
- Still has softmax amplification

### Solution 5: Independent Heads (Simplest, Working)

Train completely independent heads:

```python
def forward(self, x):
    features = self.feature_extractor(x)
    shared = self.shared_features(features)

    # All predictions from same features
    return {
        'kingdom': self.kingdom_head(shared),
        'phylum': self.phylum_head(shared),
        # ... all levels independently
    }
```

**Pros**:
- Guaranteed to work (proven by `src/model_simple.py`)
- Clean, simple implementation
- No mode collapse issues

**Cons**:
- Loses hierarchical consistency
- Can't enforce taxonomy structure during inference

---

## Recommended Fix

**Use Solution 1 (stop gradient) as immediate fix**, then experiment with Solution 2 (use logits) for true hierarchical learning.

### Implementation

1. **Short-term** (tonight):
   - Modify `src/model.py` forward() to add `.detach()` after each softmax
   - Re-train baseline to verify it works
   - If successful, apply to multi-scale models

2. **Medium-term** (tomorrow):
   - Test Solution 2 (logits instead of softmax)
   - Compare performance: detached vs logits vs independent

3. **Long-term** (if time):
   - Implement proper hierarchical loss that enforces consistency
   - Use temperature annealing for smooth training

---

## Expected Results After Fix

With Solution 1 (detached conditioning):
- **Species accuracy > 20%** (model can actually learn)
- **All taxonomic levels improve** during training
- **Validation score < 2.5** (30% gap closure target)

With Solution 5 (independent heads):
- **Species accuracy ~ 30-40%** (based on similar architectures)
- **Fast training** (no gradient flow issues)
- **Good baseline** for comparison

---

## Next Steps

1. ✅ Identified root cause
2. ⏭️ Implement Solution 1 in `src/model.py`
3. ⏭️ Re-train baseline (30 min test)
4. ⏭️ If successful, apply to multi-scale models
5. ⏭️ Continue with original plan (attention, evaluation, report)

---

**Time saved**: By identifying this bug now (Day 3), we avoid wasting Days 4-5 on failed experiments!
