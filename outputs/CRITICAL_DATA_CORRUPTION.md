# CRITICAL: Complete Training Dataset Corruption

**Date**: 2025-12-03
**Status**: 🚨 ALL TRAINING DATA CORRUPTED - Re-download Required

---

## Executive Summary

**ALL 23,699 training images are corrupted placeholder files (~130 bytes each).** This is why every model (Exp 1a, 1b, baseline, simple, pure PyTorch) achieves exactly 1.27% species accuracy (random guessing level).

The previous 3 days of training were on INVALID DATA. Once real images are restored, models should achieve >50% accuracy based on notebook results.

---

## The Evidence

### 1. File Size Distribution

```bash
$ find train/rois -name "*.png" -exec ls -l {} \; | awk '{print $5}' | sort -n | uniq -c

   1 127
2302 128
12164 129
7450 130
1492 131
 290 132
```

**ALL 23,699 files are 127-132 bytes** - these are corrupted placeholder files, not real images.

Normal PNG images should be 10-500KB.

### 2. Loading Errors During Training

```
Error loading image .../1631_5996.png: cannot identify image file
Error loading image .../46_185.png: cannot identify image file
... (123,999 errors across 6 epochs)
```

**~20,666 errors per epoch** - nearly EVERY image fails to load.

### 3. Both Full Images and ROIs Corrupted

```bash
$ ls -lh train/images/1000.png
-rw-rw-r-- 1 user user 132 Nov 18 21:27 train/images/1000.png

$ ls -lh train/rois/1_1.png
-rw-rw-r-- 1 user user 129 Nov 18 21:27 train/rois/1_1.png
```

Both the full images AND the cropped ROIs are corrupted, suggesting the download script never completed successfully.

---

## Why This Explains All Failures

### Identical 1.27% Accuracy Across All Models

| Model | Architecture | Species Acc | Expected |
|-------|-------------|-------------|----------|
| Exp 1a | 2-scale multi-scale | 1.27% | >40% |
| Exp 1b | 3-scale multi-scale | 1.27% | >50% |
| Baseline | Single-scale | 1.27% | >30% |
| Simple | Independent heads | 1.27% | >30% |
| Pure PyTorch | No Lightning | 1.27% | >30% |

**1.27% ≈ 1/80 = random guessing for 80 species classes**

### Why Loss Still Decreases

Training loss decreases slightly (21.15 → 21.03) because:
1. The dataloader's error handling silently skips corrupted images
2. A few images might load partially or default to zeros
3. The model learns the label distribution from whatever it can see
4. But validation accuracy never improves because there's no visual signal

### Why Debugging Didn't Find This Earlier

All debugging focused on:
- ✅ Model architecture (identical to working notebook)
- ✅ Loss calculation (working correctly)
- ✅ Gradient flow (gradients flowing)
- ✅ Weight updates (weights updating)
- ✅ Label encoding (labels correct)

But never checked: **Are the images themselves valid?**

---

## Why The Notebook Worked

The notebook ([hierarchical-classifier.ipynb](hierarchical-classifier.ipynb)) achieved 89% species accuracy with:

```
Epoch 14: val_species_acc=0.890
Best model saved at: fathomnet-epoch=04-val_loss=0.9577.ckpt
```

**Possible reasons:**
1. The notebook was run BEFORE the images got corrupted (Nov 18 21:27)
2. The notebook downloaded its own copy of the data to a different location
3. The notebook used a different data source
4. The images were corrupted AFTER the notebook run

The checkpoint file `model_scripted.pt` (353MB, Nov 18 06:27) exists and predates the corrupted images (Nov 18 21:27), confirming the notebook had valid data.

---

## Root Cause Analysis

### When Did Corruption Occur?

```bash
$ ls -lh train/rois/ | head -5
-rw-rw-r-- 1 user user 129 Nov 18 21:27 1000_3865.png
-rw-rw-r-- 1 user user 129 Nov 18 21:27 1000_3866.png
```

**All images created: Nov 18 21:27** (same timestamp)

**Notebook checkpoint: Nov 18 06:27** (15 hours earlier!)

This confirms:
1. Notebook trained on VALID images (morning of Nov 18)
2. Images were re-downloaded/corrupted at 21:27 (evening of Nov 18)
3. All script-based training since then used corrupted data

### Why Download Failed

The download script ([download.py](download.py)) downloads images from URLs in the COCO dataset files:

```python
async def download_image(client: AsyncClient, url: str, output_path: Path):
    response = await client.get(url)
    response.raise_for_status()
    with open(output_path, "wb") as f:
        f.write(response.content)
```

Possible failure modes:
1. **Network timeout** - Downloads interrupted, wrote partial files
2. **Rate limiting** - Server blocked requests, returning error pages
3. **Invalid URLs** - URLs in dataset JSON no longer valid
4. **Insufficient disk space** - Write failed after opening file
5. **Process killed** - Download interrupted, left placeholder files

The fact that ALL images are ~130 bytes suggests they're all receiving the same error response (likely an HTTP error page or redirect).

---

## Immediate Action Plan

### Step 1: Check for Original Dataset

```bash
# Check if dataset JSON files exist
ls -lh dataset_train.json dataset_test.json

# Check their content
head dataset_train.json
```

### Step 2: Attempt Re-download

Option A: Re-run download script
```bash
python download.py dataset_train.json train/
```

Option B: Download from Kaggle
```bash
kaggle competitions download -c fathomnet-2025
unzip fathomnet-2025.zip
```

Option C: Use FathomNet API directly
```bash
# Check if fathomnet package is installed
python -c "import fathomnet; print(fathomnet.__version__)"
```

### Step 3: Verify Downloaded Images

```bash
# Check file sizes
find train/rois -name "*.png" -size -10k

# Should return 0 if all images valid

# Verify with Python
python -c "
from PIL import Image
import os
import random

files = [f'train/rois/{f}' for f in os.listdir('train/rois') if f.endswith('.png')]
sample = random.sample(files, min(100, len(files)))

errors = 0
for f in sample:
    try:
        img = Image.open(f)
        img.verify()
    except Exception as e:
        print(f'ERROR: {f} - {e}')
        errors += 1

print(f'\n{errors}/100 images failed to load')
print('Dataset OK!' if errors == 0 else 'Dataset CORRUPTED!')
"
```

### Step 4: Quick Baseline Test

Once images are valid, run a 1-epoch test:

```bash
CUDA_VISIBLE_DEVICES=2 python train_simple.py --max_epochs=1
```

Expected results with valid data:
- Train loss should decrease significantly (< 10 by end of epoch 1)
- Val species accuracy should be > 5% after epoch 1 (much better than 1.27%)
- No "Error loading image" messages

### Step 5: Full Re-training

If baseline test succeeds:
1. Re-run baseline (3-5 epochs to convergence)
2. Re-run Exp 1a (ROI + 3× context)
3. Re-run Exp 1b (ROI + 3× + 5× context)
4. Compare results

Expected performance with valid data:
- Baseline: ~30-40% species accuracy
- Exp 1a: ~40-50% species accuracy
- Exp 1b: ~50-60% species accuracy (target: close 30% gap)

---

## Time Impact

### Wasted Time (Nov 18-Dec 3)
- **Days 1-2**: Modularizing code, setting up experiments ✅ (still useful)
- **Day 3 morning**: Running Exp 1a, 1b (wasted - trained on corrupted data) ❌
- **Day 3 afternoon**: Extensive debugging (wasted - wrong problem) ❌
- **Day 3 evening**: Pure PyTorch tests (wasted - still corrupted data) ❌

### Recovery Time Estimate

1. **Re-download dataset**: 30-60 minutes
2. **Verify integrity**: 5-10 minutes
3. **Baseline test (1 epoch)**: 5 minutes
4. **Re-run all experiments**: 2-4 hours
5. **Analysis and report**: Proceed as planned

**Total recovery time**: 3-5 hours

**Remaining time for project**: Still feasible to complete if starting now (Day 3 evening)

---

## Lessons Learned

### What Went Wrong

1. **No data validation** - Never checked if images were valid before training
2. **Silent failures** - Dataloader error handling hid the problem
3. **No baseline checks** - Should have verified random model gives ~1.25% accuracy
4. **Trust in file timestamps** - Assumed if files exist, they're valid

### What To Do Differently

1. **Always validate data first**:
   ```python
   # Before ANY training
   print("Checking dataset integrity...")
   validate_images(train_loader)
   ```

2. **Monitor data loading errors**:
   ```python
   # In dataset __getitem__
   if error:
       logger.error(f"Failed to load {path}")
       raise  # Don't silently skip!
   ```

3. **Sanity check performance**:
   ```python
   # Untrained model should give ~1/n_classes accuracy
   # If trained model = untrained, something is wrong!
   ```

4. **Check file integrity after download**:
   ```python
   # After download
   verify_all_images()
   assert no_corrupted_files
   ```

---

## Next Steps

**PRIORITY 1**: Re-download and verify training data

**PRIORITY 2**: Run 1-epoch baseline test to confirm images work

**PRIORITY 3**: Re-run all experiments with valid data

**PRIORITY 4**: Continue with attention mechanism if time permits

**PRIORITY 5**: Write final report with corrected results

---

**This is a recoverable setback.** The code is solid (all debugging confirmed correct implementation). We just need valid training data!
