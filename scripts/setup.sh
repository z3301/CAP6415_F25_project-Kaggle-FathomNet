#!/bin/bash
# FathomNet 2025 - One-command setup script
# Downloads test dataset and pre-trained checkpoint

set -e  # Exit on error

echo "=== FathomNet 2025 Setup ==="
echo ""

# Step 1: Install dependencies
echo "[1/4] Installing dependencies..."
pip install -e . --quiet
pip install gdown --quiet

# Step 2: Download test dataset (full images + 1x ROIs)
echo "[2/4] Downloading test dataset..."
python data/download.py data/dataset_test.json test

# Step 3: Create multi-scale ROIs (3x, 5x, full)
echo "[3/4] Creating multi-scale ROIs..."
# Copy 1x ROIs (already created by download.py as 'rois/')
mkdir -p test/rois/1x
cp test/rois/*.png test/rois/1x/ 2>/dev/null || true

# Create 3x and 5x scale ROIs
python data/create_multiscale_rois.py \
    --dataset-json data/dataset_test.json \
    --images-dir test/images \
    --output-dir test/rois \
    --scales 3.0 5.0

# Create full-image scale ROIs
python data/create_full_scale_rois.py \
    --coco-json data/dataset_test.json \
    --image-dir test/images \
    --output-dir test/rois/full

# Step 4: Download pre-trained checkpoint
echo "[4/4] Downloading pre-trained checkpoint (~4GB)..."
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
