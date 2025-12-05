# FathomNet Project Status Summary

**Date**: 2025-12-03
**Project**: Multi-Scale Hierarchical Classifier for FathomNet 2025 (CAP6415)
**Deadline**: 1 week from start

---

## ✅ Completed Tasks

### 1. Code Modularization
Successfully extracted notebook code into modular structure:
- `src/config.py` - Configuration management
- `src/data.py` - Single-scale datasets
- `src/model.py` - Single-scale hierarchical model
- `src/taxonomy.py` - Taxonomy encoding utilities
- `src/train.py` - Training infrastructure
- `src/evaluate.py` - Evaluation metrics
- `src/inference.py` - Prediction/submission generation

### 2. Multi-Scale Architecture Implementation
Created multi-scale components:
- `src/data_multiscale.py` - Multi-scale datasets (supports configurable scales)
- `src/model_multiscale.py` - Multi-scale hierarchical classifier
  - Separate ConvNeXtV2-Base encoder per scale
  - Projection layers to shared 512-dim space
  - Concatenation fusion (e.g., 3 scales → 1536-dim)
  - Hierarchical heads with sequential conditioning
- `train_multiscale.py` - Training script with exp 1a/1b support
- `evaluate_multiscale.py` - Evaluation script for multi-scale models

### 3. Training Infrastructure
- PyTorch Lightning training framework
- Mixed precision (FP16) for memory efficiency
- Early stopping (patience=10 epochs)
- Model checkpointing (top-3 by val_loss)
- Learning rate monitoring

### 4. Parallel Experiment Launch
Successfully launched both experiments in parallel:
- **Exp 1a**: GPU 0, 2 scales (ROI + 3× context), 181M params
- **Exp 1b**: GPU 1, 3 scales (ROI + 3× + 5× context), 269M params

### 5. Monitoring Setup
- Created `monitor_training.sh` - Automated monitoring script
  - Checks progress every 30 minutes
  - Detects completion via checkpoint files and file modifications
  - Sends desktop notifications when experiments finish
  - Running in background (process 99012d)

### 6. Planning Documents
- `docs/1week_timeline.md` - Detailed 1-week accelerated timeline
- `docs/status_summary.md` - This file
- Updated todo list with 8 tasks

---

## 🏃 Currently Running

### Exp 1a (Background Process 70a36b)
- **Status**: Training Epoch 0 in progress
- **GPU**: 0 (CUDA device)
- **Configuration**:
  - Scales: [1.0, 3.0] (ROI + 3× context)
  - Encoders: 2× ConvNeXtV2-Base
  - Total params: 181M (175M backbone + 6M heads)
  - Batch size: 32
  - Learning rate: 3e-4 with cosine annealing
  - Max epochs: 50 with early stopping
- **Output**: `/mnt/beegfs/home/dzimmerman2021/Documents/fathomnet/outputs/exp1a_2scales/`
- **Training speed**: ~2.8 it/s
- **Observations**: Some corrupted image files (gracefully handled)

### Exp 1b (Background Process f91724)
- **Status**: Training Epoch 0 in progress
- **GPU**: 1 (CUDA_VISIBLE_DEVICES=1)
- **Configuration**:
  - Scales: [1.0, 3.0, 5.0] (ROI + 3× + 5× context)
  - Encoders: 3× ConvNeXtV2-Base
  - Total params: 269M (263M backbone + 6M heads)
  - Batch size: 32
  - Learning rate: 3e-4 with cosine annealing
  - Max epochs: 50 with early stopping
- **Output**: `/mnt/beegfs/home/dzimmerman2021/Documents/fathomnet/outputs/exp1b_3scales/`
- **Training speed**: ~4.2 it/s (GPU 1 appears faster)
- **Observations**: Same corrupted image files as Exp 1a

### Monitoring Script (Background Process 99012d)
- **Status**: Running
- **Log**: `/mnt/beegfs/home/dzimmerman2021/Documents/fathomnet/outputs/monitor.out`
- **Check interval**: 30 minutes
- **Alert**: Will notify when both experiments complete

---

## 📋 Next Steps

### Immediate (Automated)
- ⏳ Wait for training to complete (~24-48 hours expected)
- ⏳ Monitoring script will alert on completion

### Day 3 (After Training Completes)
1. **Morning: Evaluate Results (4 hours)**
   ```bash
   # Evaluate Exp 1a
   python evaluate_multiscale.py --model outputs/exp1a_2scales/<best_checkpoint>.ckpt

   # Evaluate Exp 1b
   python evaluate_multiscale.py --model outputs/exp1b_3scales/<best_checkpoint>.ckpt
   ```

2. **Afternoon: Design Attention Mechanism (4 hours)**
   - Create `src/model_attention.py`
   - Architecture:
     - 3× DinoV2-B/14 encoders (patch embeddings, not global features)
     - ROI patches → Query projection
     - Environmental patches (3×, 5×) → Key/Value projections
     - Multi-head attention (8 heads)
     - Fusion → hierarchical heads

### Day 4-5: Implement & Train Attention
- Implement `src/model_attention.py` and `src/data_attention.py`
- Create `train_attention.py`
- Launch Exp 2 (attention model)
- Target: validation score < 2.30

### Day 6: Final Evaluation
- Run comprehensive evaluation
- Create comparison table (baseline vs 1a vs 1b vs 2)
- Generate visualizations and attention heatmaps
- Error analysis

### Day 7: Report & Documentation
- Write CAP6415 final report (4-5 pages)
- Code cleanup and documentation
- Prepare submission

---

## 🎯 Success Criteria

### Minimum Viable (Pass CAP6415):
- ✅ Multi-scale architecture implemented and trained
- ⏳ Attention mechanism implemented (even if no improvement)
- ⏳ Complete evaluation with metrics and visualizations
- ⏳ Final report documenting approach and results

### Target (Strong Performance):
- 🎯 Validation score < 2.50 (close 30% gap to 1st place)
- 🎯 Attention mechanism shows measurable improvement
- 🎯 Clear visualizations of what attention learns
- 🎯 Thorough error analysis

### Stretch (Publication Quality):
- 🚀 Validation score < 2.30 (close 50% gap)
- 🚀 Attention weights align with ecological knowledge
- 🚀 Complete ablation study
- 🚀 High-quality figures and polished report

---

## 📊 Technical Details

### Dataset Splits
- Training: 70% (stratified by species)
- Validation: 15% (for early stopping and model selection)
- Evaluation: 15% (for final metrics)

### Architecture Details
- **Backbone**: ConvNeXtV2-Base (88M params per encoder)
- **Image size**: 224×224 per scale
- **Multi-scale processing**:
  - 1.0× scale: Original ROI
  - 3.0× scale: ROI + padding (2× max dimension)
  - 5.0× scale: ROI + padding (4× max dimension)
- **Fusion**: Concatenation (512-dim per scale)
- **Hierarchical heads**: Sequential conditioning
  - Kingdom → Phylum → Class → Order → Family → Genus → Species
  - Each level conditions on previous level features

### Training Configuration
- **Optimizer**: AdamW
- **Learning rate**: 3e-4 with cosine annealing
- **Batch size**: 32 (16-32 for attention model)
- **Mixed precision**: FP16
- **Early stopping**: Patience=10 epochs
- **Hardware**: H200 GPUs with 143GB VRAM

---

## 🐛 Known Issues

### Corrupted Images
Multiple image files cannot be loaded (PIL cannot identify):
- Examples: `8334_22438.png`, `5632_16750.png`, `1285_4856.png`, etc.
- Impact: Training continues with remaining images (no crash)
- Fix: Dataset handles gracefully with try-except in collate_fn

### Git LFS Warnings
- Non-fatal warnings on every git operation
- Operations complete despite warnings
- Not blocking progress

---

## 📝 Files Reference

### Training Scripts
- `train_multiscale.py` - Multi-scale training (Exp 1a/1b)
- `evaluate_multiscale.py` - Multi-scale evaluation
- `monitor_training.sh` - Automated monitoring

### Source Code
- `src/config.py` - Configuration
- `src/data_multiscale.py` - Multi-scale datasets
- `src/model_multiscale.py` - Multi-scale model
- `src/taxonomy.py` - Taxonomy utilities
- `src/evaluate.py` - Evaluation functions
- `src/train.py` - Single-scale training (for reference)

### Documentation
- `docs/1week_timeline.md` - Detailed weekly plan
- `docs/status_summary.md` - This file
- `docs/dissertation_report.md` - Research progress report

### Outputs
- `outputs/exp1a_2scales/` - Exp 1a checkpoints and logs
- `outputs/exp1b_3scales/` - Exp 1b checkpoints and logs
- `outputs/monitor.out` - Monitoring script log
- `outputs/training_monitor.log` - Monitoring summary

---

## 🔍 How to Check Progress

### Check Training Progress
```bash
# View latest output from Exp 1a
tail -f outputs/exp1a_2scales/lightning_logs/version_0/events.out.tfevents.*

# View latest output from Exp 1b
tail -f outputs/exp1b_3scales/lightning_logs/version_0/events.out.tfevents.*
```

### Check Monitoring Log
```bash
# View monitoring script output
cat outputs/training_monitor.log

# Or watch it live
tail -f outputs/training_monitor.log
```

### Check GPU Usage
```bash
nvidia-smi
```

### List Saved Checkpoints
```bash
# Exp 1a
ls -lh outputs/exp1a_2scales/*.ckpt

# Exp 1b
ls -lh outputs/exp1b_3scales/*.ckpt
```

---

## 💡 Tips

1. **Let it run**: Both experiments are configured for automatic early stopping. Don't interrupt unless there's an error.

2. **Check every 6-8 hours**: The monitoring script checks every 30 minutes, but you only need to check progress 2-3 times per day.

3. **Expect 24-48 hours**: With early stopping, experiments should complete within this timeframe.

4. **GPU availability**: 7 GPUs remain free (GPUs 2-7) for future experiments.

5. **Attention mechanism**: Can start implementation design today, but wait for multi-scale results before training.

---

**Last Updated**: 2025-12-03 (Experiments just started)
**Next Update**: When training completes (monitoring script will alert)
