# FathomNet Quick Reference

Quick commands for training and inference with the project.

---

## Installation

```bash
# Clone and install
git clone https://github.com/z3301/CAP6415_F25_project-Kaggle-FathomNet.git
cd CAP6415_F25_project-Kaggle-FathomNet
pip install -e .

# Download data (requires Kaggle API credentials)
kaggle competitions download -c fathomnet-2025
```

---

## Training

### Multi-Scale with Taxonomic Loss (Best Model)
```bash
# Using CLI entry point
fathomnet-train-taxloss \
  --config config/experiment-multiscale.yaml \
  --scales 1x 3x 5x full \
  --exp-name taxloss_4scales

# Or run directly
python -m src.training.train_multiscale_taxloss \
  --config config/experiment-multiscale.yaml
```

### Standard Multi-Scale Training
```bash
fathomnet-train \
  --config config/experiment-multiscale.yaml \
  --scales 1x 3x 5x
```

### Custom Hyperparameters
```bash
# Smaller batch size (if OOM)
python -m src.training.train_multiscale_taxloss --batch-size 8

# Different learning rate
python -m src.training.train_multiscale_taxloss --lr 1e-4
```

---

## Inference

### Generate Submission
```bash
# Using CLI entry point
fathomnet-predict-taxloss \
  --checkpoint outputs/taxloss_4scales/checkpoints/best.ckpt \
  --output submission.csv

# Or run directly
python -m src.inference.generate_submission_taxloss \
  --checkpoint outputs/taxloss_4scales/checkpoints/best.ckpt
```

---

## Monitoring Training

### Check GPU Usage
```bash
nvidia-smi
watch -n 1 nvidia-smi  # Continuous monitoring
```

### Check Training Processes
```bash
ps aux | grep train_multiscale
```

### View Training Logs
```bash
# Tail the latest log
tail -f outputs/*/lightning_logs/version_*/events.out.tfevents.*

# Or check nohup output
tail -f nohup.out
```

### Check Saved Checkpoints
```bash
ls -lth outputs/*/checkpoints/*.ckpt | head -10
```

---

## Troubleshooting

### Kill Training (Emergency)
```bash
# Find process IDs
ps aux | grep train_multiscale

# Kill specific process
kill <PID>

# Or kill all training processes
pkill -f train_multiscale
```

### Out of Memory (OOM)
```bash
# Reduce batch size
python -m src.training.train_multiscale_taxloss --batch-size 4

# Or use fewer scales
python -m src.training.train_multiscale --scales 1x 3x
```

---

## Python API

```python
# Import models
from src.models import (
    MultiScaleTaxonomicClassifier,     # Best model (taxonomic loss)
    MultiScaleTaxonomyClassifier,      # Multi-scale with CE loss
    TaxonomyAwareClassifier,           # Single-scale baseline
)

# Import data utilities
from data import (
    load_and_encode_taxonomy,
    build_multiscale_dataloaders,
    MultiScalePrecroppedDataset,
)

# Import losses
from src.losses import TaxonomicDistanceLoss
```

---

## File Locations

### Key Directories
- `src/training/` - Training scripts
- `src/inference/` - Submission generation scripts
- `src/models/` - Model architectures
- `src/losses.py` - Custom loss functions
- `src/config.py` - Default configuration
- `data/` - Data loading utilities
- `config/` - YAML configuration files
- `outputs/` - Training outputs and checkpoints

### Configuration Files
- `config/experiment-default.yaml` - Base single-scale config
- `config/experiment-multiscale.yaml` - Multi-scale training config

### Data Locations
- `~/Documents/data/train/` - Training images (default)
- `~/Documents/data/test/` - Test images (default)
- `data/taxonomy.csv` - Taxonomic hierarchy
- `data/distance_matrix.csv` - Taxonomic distance matrix

---

## Quick Status Check

Run this to see everything at a glance:
```bash
echo "=== GPU Usage ===" && \
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv && \
echo -e "\n=== Training Processes ===" && \
ps aux | grep train_multiscale | grep -v grep && \
echo -e "\n=== Checkpoints ===" && \
ls -lth outputs/*/checkpoints/*.ckpt 2>/dev/null | head -5
```

---

## Best Model Configuration

The best performing model uses:
- 4 scales: ROI (1x), 3x context, 5x context, full image
- ROI encoder: ConvNeXtV2-Large (197M params)
- Context encoders: ConvNeXtV2-Base (88M each)
- Taxonomic distance loss (alpha=0.3)
- Confidence threshold fallback at inference (70%)

**Result**: Private LB score of 1.94 (27% improvement over baseline)
