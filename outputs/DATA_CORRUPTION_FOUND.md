# ROOT CAUSE IDENTIFIED: Complete Training Data Corruption

**Date**: 2025-12-03
**Status**: 🚨 CRITICAL - ALL training images are corrupted!

---

## The Discovery

ALL 23,699 training images in `train/rois/` are corrupted placeholder files of ~130 bytes each.

### Evidence

```bash
$ find train/rois -name "*.png" -size -10k | wc -l
23699

$ ls train/rois/*.png | wc -l
23699

$ ls -lh train/rois/1_1.png
-rw-rw-r-- 1 user user 129 Nov 18 21:27 train/rois/1_1.png
```

**Every single training image is 129-131 bytes** - these are not real images, they're corrupted/placeholder files!

### Actual Error Count

During training:
```
Error loading image .../1631_5996.png: cannot identify image file
Error loading image .../46_185.png: cannot identify image file
... (123,999 errors across 6 epochs = ~20,666 errors/epoch)
```

With 519 batches × 32 batch_size = 16,608 images/epoch, nearly EVERY image fails to load!

---

## Why This Explains Everything

### 1. Identical 1.27% Species Accuracy

- 80 species classes
- Random guessing = 1/80 = 1.25%
- Observed = 1.27% ≈ random chance

The model cannot learn because it's training on **corrupted data that fails to load**.

### 2. All Models Fail Identically

| Model | Architecture | Species Acc |
|-------|-------------|-------------|
| Exp 1a | 2-scale multi-scale | 1.27% |
| Exp 1b | 3-scale multi-scale | 1.27% |
| Baseline | Single-scale hierarchical | 1.27% |
| Simple | Independent heads | 1.27% |
| Pure PyTorch | No Lightning | 1.27% |

**All show IDENTICAL failure** because they're all using the same corrupted dataset!

### 3. Why the Notebook Worked

The notebook ([hierarchical-classifier.ipynb](hierarchical-classifier.ipynb)) achieved 89% species accuracy because it likely had access to the REAL uncorrupted images, not these placeholders.

### 4. Loss Behavior

- Loss DOES decrease slightly (21.15 → 21.03)
- But this is just the model learning the label distribution from the FEW images that do load
- Validation accuracy never improves because there's no signal in the corrupted images

---

## What Happened to the Data?

The images were likely:

1. **Never downloaded properly** - Download script failed silently
2. **Corrupted during transfer** - Files got truncated to placeholders
3. **Intentionally removed** - Someone cleaned up the data after notebook work
4. **Symbolic link broken** - Real images are elsewhere, these are broken links

The file timestamps (Nov 18 21:27) suggest they were all created/modified at once, likely as placeholders.

---

## Immediate Action Required

### Step 1: Locate Real Images

Check if real images exist elsewhere:

```bash
# Check for larger image files
find /mnt/beegfs/home/dzimmerman2021 -name "*.png" -size +50k | grep -i fathom

# Check for backup directories
find /mnt/beegfs/home/dzimmerman2021 -type d -name "*fathom*" -o -name "*backup*"

# Check if notebook points to different path
grep "train.*rois" hierarchical-classifier.ipynb
```

### Step 2: Re-download Dataset

If images can't be found, download from FathomNet competition:

```bash
# Download from Kaggle
kaggle competitions download -c fathomnet-2025

# Or use the original FathomNet URLs if available
```

### Step 3: Verify Image Integrity

After getting real images:

```bash
# Check file sizes
find train/rois -name "*.png" -size -10k

# Verify they can be loaded
python -c "
from PIL import Image
import os
for f in os.listdir('train/rois')[:100]:
    try:
        img = Image.open(f'train/rois/{f}')
        print(f'{f}: {img.size}')
    except Exception as e:
        print(f'{f}: ERROR - {e}')
"
```

---

## Time Impact

**All previous training (Days 1-3) was invalid** due to corrupted data:
- Exp 1a training: wasted
- Exp 1b training: wasted
- Baseline training: wasted
- All debugging efforts: addressing wrong problem

**Once real images are restored:**
- Re-run all experiments with valid data
- Expect dramatically better results (likely >50% species accuracy based on notebook)

---

## Why Debugging Couldn't Find This

The debugging focused on:
- Model architecture
- Loss calculation
- Gradient flow
- PyTorch Lightning configuration

But never checked: **Are the input images actually valid?**

The dataloader's error handling silently skipped corrupted images, so the model trained on whatever few images (if any) did load, leading to random-guess performance.

---

## Next Steps

1. **STOP all training immediately** ✅ (already done)
2. **Locate or re-download real training images** 🔴 URGENT
3. **Verify image integrity**
4. **Re-run ONE baseline experiment to verify improvement**
5. **If successful, re-run Exp 1a and 1b with real data**
6. **Continue with attention mechanism and report**

---

**Estimated time to fix**:
- If images exist elsewhere: 10 minutes
- If need to re-download: 30-60 minutes (depends on dataset size)
- Then 2-4 hours to re-run experiments with real data

**Priority**: CRITICAL - Cannot proceed without real training data
