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
Ensure you are using **Python ≥ 3.9** and have **CUDA-enabled GPUs** configured.

```bash
pip install -r requirements.txt
```

### 3. Configure Data Path
Open `hierarchical-classifier.ipynb` and edit the following line to point to your local dataset:
```python
DATA_ROOT = "/path/to/your/local/data"
```

### 4. Run the Notebook
Launch Jupyter and execute all cells:
```bash
jupyter notebook hierarchical-classifier.ipynb
```

### 5. Outputs
The notebook will:
- Train the hierarchical multi-output classifier  
- Evaluate accuracy across all taxonomic ranks  
- Save trained weights, plots, and metrics in the `runs/` directory  

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
