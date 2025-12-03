"""
Configuration for FathomNet 2025 Competition

Centralized configuration for paths, training parameters, and model settings.
"""

import os

DATA_ROOT = "/mnt/beegfs/home/dzimmerman2021/Documents/fathomnet/"


class Config:
    """Configuration class for FathomNet project."""

    # Paths
    TAXONOMY_PATH = os.path.join(DATA_ROOT, "taxonomy.csv")
    TRAIN_ANNOTATIONS = os.path.join(DATA_ROOT, "train", "annotations.csv")
    TRAIN_IMAGE_DIR = os.path.join(DATA_ROOT, "train", "rois")
    TEST_ANNOTATIONS = os.path.join(DATA_ROOT, "test", "annotations.csv")
    TEST_IMAGE_DIR = os.path.join(DATA_ROOT, "test", "rois")
    OUTPUT_DIR = os.path.join(DATA_ROOT, "outputs")
    SUBMISSION_PATH = os.path.join(DATA_ROOT, "submission.csv")

    # Training parameters
    BATCH_SIZE = 32
    NUM_WORKERS = 8
    MAX_EPOCHS = 50
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 0.05
    LABEL_SMOOTHING = 0.1

    # Model parameters
    IMG_SIZE = 224
    BACKBONE = "convnextv2_base.fcmae_ft_in22k_in1k"

    # Taxonomic levels
    TAXONOMY_LEVELS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

    # Inference parameters
    CONFIDENCE_THRESHOLD = 0.7


def check_path(path, is_dir=False):
    """Check if a path exists and return a status message."""
    exists = os.path.exists(path)
    if not exists:
        return f"NOT FOUND: {path}"
    else:
        if is_dir:
            return f"Directory found: {path}"
        else:
            return f"File found: {path}"


def verify_paths():
    """Verify all configured paths exist."""
    print("Checking configured paths:")
    print(check_path(Config.TAXONOMY_PATH))
    print(check_path(Config.TRAIN_ANNOTATIONS))
    print(check_path(Config.TRAIN_IMAGE_DIR, is_dir=True))
    print(check_path(Config.TEST_ANNOTATIONS))
    print(check_path(Config.TEST_IMAGE_DIR, is_dir=True))
    print(f"Output will be saved to: {os.path.abspath(Config.OUTPUT_DIR)}")

    # Create output directory if it doesn't exist
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
