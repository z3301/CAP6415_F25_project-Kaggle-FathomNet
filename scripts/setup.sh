#!/bin/bash
# FathomNet 2025 - One-command setup script
# Downloads test dataset and pre-trained checkpoint

set -e  # Exit on error

echo "=== FathomNet 2025 Setup ==="
echo ""

# Step 1: Install dependencies
echo "[1/3] Installing dependencies..."
pip install -e . --quiet
pip install gdown --quiet

# Step 2: Download test dataset
echo "[2/3] Downloading test dataset..."
python data/download.py data/dataset_test.json data/test

# Step 3: Download pre-trained checkpoint
echo "[3/3] Downloading pre-trained checkpoint (~4GB)..."
mkdir -p outputs/multiscale_4scales_taxloss/checkpoints
gdown 1roOwrSRXP93tZRiLqb4paKrmBnT4Jw1y -O "outputs/multiscale_4scales_taxloss/checkpoints/best-epoch=02-val_tax_score=0.531.ckpt"

echo ""
echo "=== Setup complete! ==="
echo ""
echo "To run inference:"
echo "  export KAGGLE_USERNAME=\"your_username\""
echo "  export KAGGLE_KEY=\"your_api_key\""
echo "  python -m src.inference.generate_submission_taxloss"
echo ""
echo "Or use --no-submit to skip Kaggle submission:"
echo "  python -m src.inference.generate_submission_taxloss --no-submit"
