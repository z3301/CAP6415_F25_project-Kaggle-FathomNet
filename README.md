![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=PyTorch&logoColor=white) ![PyTorch Lightning](https://img.shields.io/badge/Lightning-%23792EE5.svg?logo=Lightning&logoColor=white) ![Kaggle](https://img.shields.io/badge/Kaggle_Competition-FathomNet_2025-blue) ![Rank](https://img.shields.io/badge/Rank-8th_/_79_teams-success)


# FathomNet 2025 Hierarchical Classification Project

## Overview

This repository contains my submission for the **[Kaggle FathomNet 2025 Competition](https://www.kaggle.com/competitions/fathomnet-2025)** — a challenge focused on advancing machine learning models for **underwater image classification** across diverse marine species.

The competition aimed to develop algorithms capable of classifying underwater organisms using the **FathomNet** open-access image repository, which provides annotated imagery from oceanographic research institutions.

My solution placed **8th out of 79 teams**, achieving a **final public leaderboard score of 2.30** (lower is better).

---

## Model Architecture

My solution implements a **taxonomy-aware hierarchical classifier** built with **PyTorch Lightning**.  
Instead of predicting species independently, the model learns **seven interconnected taxonomic ranks**:

```
kingdom → phylum → class → order → family → genus → species
```

### Key Components
- **Backbone:** Pretrained **ConvNeXtV2-Base** from **timm**, fine-tuned for underwater imagery.
- **Multi-head architecture:** Separate linear heads for each taxonomic rank, sharing a common feature encoder.
- **Loss weighting:** Cross-entropy losses combined with hierarchical weighting to emphasize correct lineage predictions.
- **Optimizer & Scheduler:** AdamW with cosine annealing and early stopping.
- **Framework:** PyTorch Lightning for reproducibility, checkpointing, and GPU management.

This hierarchical approach improves consistency across taxonomic ranks while leveraging relationships between species and higher-order classes.

---

## Reproducing the Experiment

To reproduce the results:
 
### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/CAP6415_F25_project-Kaggle-FathomNet.git
cd CAP6415_F25_project-Kaggle-FathomNet
```

### 2. Install Dependencies
Ensure you are using **Python ≥ 3.10** and have **CUDA-enabled GPUs** configured.

```bash
# Install as editable package (recommended)
pip install -e .

# Or install dependencies only
pip install -r requirements.txt
```

### 3. Configure Data Path
Open `hierarchical-classifier.ipynb` and edit the following line to point to your local dataset:
```python
DATA_ROOT = "/path/to/your/local/data"
```
Make sure the image files are actually present (and not Git LFS pointer text files). If you copied the dataset from a Git LFS repo, run `git lfs pull` in the dataset location before training or inference. The default CLI config expects data under `~/Documents/data/{train,test}/`.

### 4. Repository Structure

```
fathomnet/
├── config/                          # Experiment configurations
│   ├── experiment-default.yaml      # Base single-scale config
│   └── experiment-multiscale.yaml   # Multi-scale training config
├── data/                            # Data loading and preprocessing
│   ├── data.py                      # Main data loading utilities
│   ├── data_multiscale.py           # Multi-scale dataset classes
│   ├── preprocessing.py             # ROI extraction scripts
│   └── extract_multiscale_rois.py   # Multi-scale ROI extraction
├── src/                             # Source code (installable package)
│   ├── models/                      # Model architectures
│   │   ├── model.py                 # Base taxonomy classifier
│   │   ├── model_multiscale.py      # Multi-scale ConvNeXtV2 model
│   │   └── model_multiscale_taxloss.py  # Taxonomic distance-aware loss
│   ├── training/                    # Training scripts
│   │   ├── train_multiscale.py      # Main multi-scale training
│   │   └── train_multiscale_taxloss.py  # Taxonomic loss training
│   ├── inference/                   # Submission generation scripts
│   │   ├── generate_submission_taxloss.py
│   │   └── generate_submission_multiscale.py
│   ├── eval.py                      # Evaluation functions
│   └── losses.py                    # Custom loss functions
├── pyproject.toml                   # Package configuration
└── outputs/                         # Training outputs (checkpoints, logs)
```

### 5. Command-line Workflow

After installing with `pip install -e .`, you can use the CLI commands:

```bash
# Train multi-scale model with taxonomic loss (best performing)
fathomnet-train-taxloss \
    --config config/experiment-multiscale.yaml \
    --scales 1x 3x 5x full \
    --exp-name taxloss_4scales

# Train standard multi-scale model
fathomnet-train \
    --config config/experiment-multiscale.yaml \
    --scales 1x 3x 5x

# Generate submission
fathomnet-predict-taxloss \
    --checkpoint outputs/taxloss_4scales/checkpoints/best.ckpt \
    --output submission.csv
```

Or run the scripts directly:

```bash
python -m src.training.train_multiscale_taxloss --help
python -m src.inference.generate_submission_taxloss --help
```

Edit `config/experiment-multiscale.yaml` to customize dataset paths, augmentation strength, backbone selection, optimizer settings, and hierarchy-loss weights.

### 6. Notebook (Legacy)
The original `hierarchical-classifier.ipynb` still ships for interactive exploration, but the scripted workflow above is the preferred, reproducible path for experiments.

---

## Dependencies
All required packages are listed in `requirements.txt`.  
This includes:
- PyTorch  
- torchvision  
- pytorch-lightning  
- timm  
- scikit-learn  
- pandas, numpy, matplotlib, seaborn  

Make sure your PyTorch and torchvision builds are compatible with your CUDA version.

---

## Results Summary

| Metric | Score |
|:--------|:------:|
| **Final Leaderboard Score** | **2.30 (lower is better)** |
| **Competition Rank** | **8 / 79 teams** |
| **Framework** | PyTorch Lightning |
| **Backbone** | ConvNeXtV2-Base (timm) |
