# FathomNet 2025 Default Configuration
"""Default configuration values for model hyperparameters."""


class Config:
    """Default configuration values."""

    # Training hyperparameters
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 0.01
    LABEL_SMOOTHING = 0.1
    MAX_EPOCHS = 50

    # Model defaults
    BACKBONE = "convnextv2_base"
    DROPOUT = 0.3
    HIDDEN_DIM = 512

    # Taxonomy levels
    TAXONOMY_LEVELS = ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

    # Data defaults
    IMAGE_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 4
