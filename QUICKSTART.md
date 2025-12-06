# FathomNet Quick Reference

Quick commands for monitoring and working with the project.

---

## 📊 Check Training Progress

### View Current Status
```bash
# Check which processes are running
ps aux | grep train_multiscale

# Check GPU usage
nvidia-smi

# View monitoring log
cat outputs/training_monitor.log
```

### Check Latest Training Output
```bash
# Exp 1a latest progress (filter for epochs and validation)
tail -100 outputs/exp1a_2scales/lightning_logs/version_0/events.out.tfevents.* 2>/dev/null || echo "No logs yet"

# Exp 1b latest progress
tail -100 outputs/exp1b_3scales/lightning_logs/version_0/events.out.tfevents.* 2>/dev/null || echo "No logs yet"
```

### Check Saved Checkpoints
```bash
# List Exp 1a checkpoints
ls -lth outputs/exp1a_2scales/*.ckpt 2>/dev/null | head -5

# List Exp 1b checkpoints
ls -lth outputs/exp1b_3scales/*.ckpt 2>/dev/null | head -5
```

---

## 🔍 After Training Completes

### Find Best Checkpoint
```bash
# Exp 1a: Find checkpoint with lowest val_loss
ls outputs/exp1a_2scales/*.ckpt | sort -t= -k3 -n | head -1

# Exp 1b: Find checkpoint with lowest val_loss
ls outputs/exp1b_3scales/*.ckpt | sort -t= -k3 -n | head -1
```

### Run Evaluation
```bash
# Evaluate Exp 1a on validation set
python evaluate_multiscale.py \
  --model $(ls outputs/exp1a_2scales/*.ckpt | sort -t= -k3 -n | head -1) \
  --split val

# Evaluate Exp 1b on validation set
python evaluate_multiscale.py \
  --model $(ls outputs/exp1b_3scales/*.ckpt | sort -t= -k3 -n | head -1) \
  --split val
```

### Run on Evaluation Set (Final Metrics)
```bash
# Exp 1a evaluation set
python evaluate_multiscale.py \
  --model outputs/exp1a_2scales/<best_checkpoint>.ckpt \
  --split eval

# Exp 1b evaluation set
python evaluate_multiscale.py \
  --model outputs/exp1b_3scales/<best_checkpoint>.ckpt \
  --split eval
```

---

## 🚀 Launch New Experiments

### Re-run Experiments (if needed)
```bash
# Re-run Exp 1a
CUDA_VISIBLE_DEVICES=0 ~/miniconda/envs/fathomnet/bin/python train_multiscale.py \
  --exp 1a --epochs 50 --lr 3e-4

# Re-run Exp 1b
CUDA_VISIBLE_DEVICES=1 ~/miniconda/envs/fathomnet/bin/python train_multiscale.py \
  --exp 1b --epochs 50 --lr 3e-4
```

### Custom Hyperparameters
```bash
# Smaller batch size (if OOM)
python train_multiscale.py --exp 1b --batch-size 16 --epochs 30

# Different learning rate
python train_multiscale.py --exp 1a --lr 1e-4 --epochs 40
```

---

## 🛠️ Troubleshooting

### Check if Training is Still Running
```bash
# Check process status
ps aux | grep train_multiscale

# Check GPU usage
nvidia-smi

# If no GPU activity, training may have stopped
```

### View Full Logs
```bash
# Exp 1a full log
find outputs/exp1a_2scales/lightning_logs -name "*.log" -exec cat {} \;

# Exp 1b full log
find outputs/exp1b_3scales/lightning_logs -name "*.log" -exec cat {} \;
```

### Restart Monitoring Script
```bash
# Kill existing monitor
pkill -f monitor_training.sh

# Restart
bash monitor_training.sh > outputs/monitor.out 2>&1 &
```

### Kill Training (Emergency)
```bash
# Find process IDs
ps aux | grep train_multiscale

# Kill specific process
kill <PID>

# Or kill all training processes
pkill -f train_multiscale
```

---

## 📈 Quick Analysis

### Count Epochs Completed
```bash
# Exp 1a
find outputs/exp1a_2scales -name "*.ckpt" | wc -l

# Exp 1b
find outputs/exp1b_3scales -name "*.ckpt" | wc -l
```

### Estimate Time Remaining
```bash
# If training at ~3 it/s with 519 batches per epoch:
# Time per epoch ≈ 519 / 3 = 173 seconds ≈ 3 minutes
# For 50 epochs: ~150 minutes = 2.5 hours (without early stopping)
# With early stopping: expect 15-30 epochs = 45-90 minutes

echo "Expected completion: 24-48 hours from start (with early stopping)"
```

---

## 📁 File Locations

### Key Files
- Training script: `train_multiscale.py`
- Evaluation script: `evaluate_multiscale.py`
- Monitoring script: `monitor_training.sh`
- Timeline plan: `docs/1week_timeline.md`
- Status summary: `docs/status_summary.md`

### Output Directories
- Exp 1a: `outputs/exp1a_2scales/`
- Exp 1b: `outputs/exp1b_3scales/`
- Checkpoints: `outputs/exp*/exp*-epoch=XX-val_loss=X.XXXX.ckpt`
- Logs: `outputs/exp*/lightning_logs/version_0/`

### Data Locations
- Training images: `train/rois/`
- Test images: `test/rois/`
- Annotations: `train/annotations.csv`, `test/annotations.csv`
- Taxonomy: `taxonomy.csv`

---

## 🎯 Current Goals

### This Week (Day 1-2)
- ✅ Launch Exp 1a and 1b in parallel
- ✅ Set up monitoring
- ⏳ Wait for training completion (24-48h)

### Day 3
- Evaluate multi-scale results
- Design attention mechanism architecture

### Day 4-5
- Implement attention model
- Train Exp 2 (attention)

### Day 6
- Final evaluation and comparison
- Error analysis and visualizations

### Day 7
- Write CAP6415 report
- Code cleanup and documentation

---

## 💬 Quick Status Check

Run this one-liner to see everything at a glance:
```bash
echo "=== GPU Usage ===" && \
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv && \
echo -e "\n=== Training Processes ===" && \
ps aux | grep train_multiscale | grep -v grep && \
echo -e "\n=== Checkpoints ===" && \
echo "Exp 1a: $(ls outputs/exp1a_2scales/*.ckpt 2>/dev/null | wc -l) checkpoints" && \
echo "Exp 1b: $(ls outputs/exp1b_3scales/*.ckpt 2>/dev/null | wc -l) checkpoints" && \
echo -e "\n=== Latest Monitor Log ===" && \
tail -10 outputs/training_monitor.log 2>/dev/null || echo "Monitor log not found"
```

---

**Remember**: The monitoring script will alert you when experiments complete. You don't need to check constantly!
