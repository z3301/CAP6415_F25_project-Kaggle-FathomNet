"""
Data Loading and Preprocessing for FathomNet 2025 Marine Species Classification.

================================================================================
OVERVIEW
================================================================================

This module provides comprehensive data loading functionality for the FathomNet
2025 competition, supporting multiple input modalities:

1. SINGLE-SCALE LOADING
   - ROI (Region of Interest): Tight crop around the organism
   - Context: Wider crop providing environmental context
   - On-the-fly cropping from full images

2. MULTI-SCALE PRE-CROPPED LOADING
   - Pre-computed crops at multiple scales (1x, 3x, 5x, full)
   - Faster training due to pre-processing
   - Supports variable number of scales

================================================================================
DATA ORGANIZATION
================================================================================

Competition Data Structure:
---------------------------
    data/
    ├── train/
    │   ├── images/               # Full training images
    │   └── rois/                 # Pre-cropped ROIs
    │       ├── 1x/               # Tight crop (original bbox + 10%)
    │       ├── 3x/               # 3× context (200% margin)
    │       ├── 5x/               # 5× context (400% margin)
    │       └── full/             # Full image resized
    ├── test/
    │   ├── images/
    │   └── rois/
    ├── train_coco.json           # COCO-format training annotations
    ├── test_coco.json            # COCO-format test annotations
    └── taxonomy.csv              # Hierarchical taxonomy mapping

Annotation Formats Supported:
-----------------------------
1. COCO JSON format (primary):
   - Standard format from Kaggle competition
   - Contains image metadata, bounding boxes, category IDs

2. CSV format (legacy):
   - Simple path/label pairs
   - Used for pre-cropped ROI loading

================================================================================
MULTI-SCALE APPROACH
================================================================================

The key insight from winning solutions is that marine organism identification
benefits from BOTH local details AND environmental context:

Scale   | Margin | Coverage    | Information
--------|--------|-------------|------------------------------------------
1x      | 10%    | ROI only    | Fine-grained morphological features
3x      | 200%   | Local       | Immediate surroundings, size context
5x      | 400%   | Regional    | Habitat type, nearby organisms
full    | N/A    | Entire      | Global scene context, depth cues

Pre-cropping Strategy:
    - Crops are computed once using create_multiscale_rois.py
    - Each scale is resized to 224×224 for consistency
    - Separate backbone encoders process each scale
    - Features are fused via concatenation or attention

================================================================================
TAXONOMY ENCODING
================================================================================

The competition uses a 7-level taxonomic hierarchy:

    Kingdom → Phylum → Class → Order → Family → Genus → Species

Each level is encoded using sklearn's LabelEncoder:
    - Species names map to integer IDs
    - Unknown/missing values map to a special "unknown" class
    - Encoders are fit on taxonomy.csv (all possible values)

The taxonomy.csv contains the complete hierarchy:
    species,genus,family,order,class,phylum,kingdom
    Abralia veranyi,Abralia,Enoploteuthidae,Oegopsida,Cephalopoda,Mollusca,Animalia
    ...

================================================================================
DATA AUGMENTATION
================================================================================

Training Augmentations:
    - RandomResizedCrop: Scale variation (0.8-1.0), maintains aspect
    - RandomHorizontalFlip: 50% probability
    - RandomRotation: ±15 degrees (configurable)
    - ColorJitter: Brightness, contrast, saturation, hue variation

Validation/Test:
    - Simple resize to target size (no augmentation)
    - Consistent normalization (ImageNet statistics)

Normalization (ImageNet pre-training compatibility):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

================================================================================
STRATIFIED SPLITTING
================================================================================

Data is split to preserve class distribution:

    Train: 80% (default, configurable)
    Val:   10%
    Eval:  10%

Uses sklearn's StratifiedShuffleSplit to ensure:
    - Each split has similar species distribution
    - Rare species appear in all splits
    - Reproducible with fixed seed

================================================================================
GIT LFS HANDLING
================================================================================

Competition images are stored using Git LFS. This module includes:
    - Detection of LFS pointer files (incomplete downloads)
    - Clear error messages when actual data is needed
    - Path resolution fallbacks for flexible directory structures

================================================================================
USAGE
================================================================================

Basic multi-scale data loading:
    from src.data import load_and_encode_taxonomy, build_multiscale_dataloaders

    taxonomy_df, encoders, class_counts, id_to_name = load_and_encode_taxonomy(
        "data/taxonomy.csv",
        ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
    )

    dataloaders, split_indices = build_multiscale_dataloaders(
        cfg, taxonomy_df, encoders, scales=["1x", "3x", "5x"]
    )

    for images, labels in dataloaders["train"]:
        # images: {"1x": (B,3,224,224), "3x": (B,3,224,224), "5x": (B,3,224,224)}
        # labels: {"kingdom": (B,), "phylum": (B,), ..., "species": (B,)}
        outputs = model(images)
        loss = compute_loss(outputs, labels)
"""

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


# =============================================================================
# ANNOTATION LOADING FUNCTIONS
# =============================================================================


def load_annotations(csv_path: str, image_root: str) -> pd.DataFrame:
    """
    Load annotations from a simple CSV file.

    This is a legacy format supporting basic path/label pairs.
    For full COCO-format annotations with bounding boxes, use load_coco_annotations.

    Expected CSV columns:
        - path: Relative or absolute path to image
        - label: Species name

    Args:
        csv_path: Path to annotations CSV file
        image_root: Root directory to resolve relative paths

    Returns:
        DataFrame with columns: path, label, image_path

    Raises:
        ValueError: If required columns are missing
    """
    df = pd.read_csv(csv_path)

    # Validate required columns
    required = {"path", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Annotations file {csv_path} missing columns: {missing}")

    # Clean string columns
    df["path"] = df["path"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()

    # Resolve image paths (handles various directory structures)
    df["image_path"] = df["path"].apply(
        lambda p: _resolve_image_path(p, image_root)
    )

    return df


def load_coco_annotations(
    coco_json: str,
    image_root: str,
    include_labels: bool = True,
) -> pd.DataFrame:
    """
    Load annotations from COCO-format JSON.

    COCO format is the standard for this competition, containing:
        - images: List of image metadata (id, file_name, width, height)
        - annotations: List of annotations (id, image_id, category_id, bbox)
        - categories: List of categories (id, name, supercategory)

    Args:
        coco_json: Path to COCO-format JSON file
        image_root: Root directory containing images
        include_labels: Whether to include category labels (False for test set)

    Returns:
        DataFrame with columns:
            - annotation_id: Unique identifier for each annotation
            - image_id: Source image ID
            - label: Species name (or None if include_labels=False)
            - bbox: [x, y, width, height] bounding box
            - image_path: Resolved path to image file

    Example:
        >>> df = load_coco_annotations("train_coco.json", "data/train/images")
        >>> print(df.columns)
        ['annotation_id', 'image_id', 'label', 'bbox', 'image_path']
    """
    with open(coco_json, "r") as f:
        data = json.load(f)

    # Build category lookup: {category_id: category_name}
    categories = {
        int(cat["id"]): str(cat.get("name") or "unknown")
        for cat in data.get("categories", [])
    }

    # Build image lookup: {image_id: {path, width, height}}
    images = {}
    for img in data.get("images", []):
        # Determine file extension from file_name or coco_url
        suffix = Path(img.get("file_name", "")).suffix
        if not suffix and img.get("coco_url"):
            suffix = Path(img["coco_url"]).suffix
        suffix = suffix or ".png"

        images[int(img["id"])] = {
            "path": _resolve_image_path(
                os.path.join(image_root, f"{img['id']}{suffix}"),
                image_root,
            ),
            "width": img.get("width"),
            "height": img.get("height"),
        }

    # Build annotation rows
    rows = []
    for ann in data.get("annotations", []):
        image_info = images.get(int(ann["image_id"]))
        if not image_info:
            continue

        # Get label if requested
        label = None
        if include_labels:
            label = categories.get(int(ann["category_id"] or -1), "unknown")

        rows.append({
            "annotation_id": int(ann["id"]),
            "image_id": int(ann["image_id"]),
            "label": label,
            "bbox": [float(x) for x in ann.get("bbox", [])] if ann.get("bbox") else None,
            "image_path": image_info["path"],
        })

    df = pd.DataFrame(rows)

    # Fill missing labels with "unknown"
    if include_labels and df["label"].isnull().any():
        df["label"] = df["label"].fillna("unknown")

    return df


# =============================================================================
# IMAGE PATH UTILITIES
# =============================================================================


def _is_git_lfs_pointer(path: str) -> bool:
    """
    Check whether a file contains a Git LFS pointer instead of actual image bytes.

    Git LFS pointers are small text files that reference the actual data stored
    elsewhere. Training on these will fail, so we detect them early.

    Pointer file format:
        version https://git-lfs.github.com/spec/v1
        oid sha256:abc123...
        size 12345

    Args:
        path: Path to file to check

    Returns:
        True if file appears to be an LFS pointer, False otherwise
    """
    try:
        with open(path, "rb") as f:
            header = f.read(256)
        return b"git-lfs" in header or b"git lfs" in header
    except OSError:
        # If we can't read the file, let the caller handle it
        return False


def _ensure_real_image(path: str):
    """
    Validate that a file exists and contains actual image data.

    Raises clear error messages for common issues:
        - Missing files
        - Git LFS pointers that need to be fetched

    Args:
        path: Path to image file

    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If file is a Git LFS pointer
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    if _is_git_lfs_pointer(path):
        raise RuntimeError(
            f"Image at '{path}' is a Git LFS pointer. "
            "Download or copy the actual files (e.g., run `git lfs pull` in the "
            "dataset source) before training or inference."
        )


def _validate_sample_images(frame: pd.DataFrame, sample_size: int = 5):
    """
    Check a sample of images early to fail fast on dataset issues.

    This catches problems like missing files or unfetched LFS pointers
    before training begins, rather than failing mid-epoch.

    Args:
        frame: DataFrame with 'image_path' column
        sample_size: Number of images to validate

    Raises:
        FileNotFoundError: If any sample image is missing
        RuntimeError: If any sample image is an LFS pointer
    """
    for path in frame["image_path"].head(sample_size):
        _ensure_real_image(path)


def _resolve_image_path(path: str, image_root: str) -> str:
    """
    Normalize an image path and resolve it against the image root.

    Handles various directory structures:
        1. Absolute paths that exist as-is
        2. Relative paths under image_root
        3. Paths with 'rois/' or 'images/' subdirectories
        4. Fallback to original path if nothing matches

    Args:
        path: Original path (may be relative or absolute)
        image_root: Root directory to search

    Returns:
        Resolved absolute path to image (may not exist)
    """
    path = os.path.normpath(path)

    # Check if path exists directly and is not an LFS pointer
    if os.path.exists(path) and not _is_git_lfs_pointer(path):
        return path

    # Try joining basename with image_root
    candidate = os.path.normpath(os.path.join(image_root, os.path.basename(path)))
    if os.path.exists(candidate) and not _is_git_lfs_pointer(candidate):
        return candidate

    # Try resolving paths with 'rois' subdirectory
    parts = path.split(os.sep)
    if "rois" in parts:
        suffix = parts[parts.index("rois") + 1:]
        candidate = os.path.normpath(os.path.join(image_root, *suffix))
        if os.path.exists(candidate) and not _is_git_lfs_pointer(candidate):
            return candidate

    # Try resolving paths with 'images' subdirectory
    if "images" in parts:
        suffix = parts[parts.index("images") + 1:]
        candidate = os.path.normpath(os.path.join(image_root, *suffix))
        if os.path.exists(candidate) and not _is_git_lfs_pointer(candidate):
            return candidate

    # Return original path as fallback
    return path


# =============================================================================
# IMAGE CROPPING UTILITIES
# =============================================================================


def crop_bbox(image: Image.Image, bbox: Optional[List[float]], margin: float) -> Image.Image:
    """
    Crop an image around a bounding box with an optional margin.

    The margin expands the crop beyond the original bounding box:
        - margin=0.0: Exact bounding box (1× crop)
        - margin=0.1: 110% of bbox size (1.1× crop)
        - margin=2.0: 300% of bbox size (3× crop)
        - margin=4.0: 500% of bbox size (5× crop)

    The formula is: crop_size = bbox_size × (1 + margin)

    Crops are clamped to image boundaries to avoid black borders.

    Args:
        image: PIL Image to crop
        bbox: [x, y, width, height] bounding box, or None to return full image
        margin: Expansion factor (0.0 = no expansion)

    Returns:
        Cropped PIL Image

    Example:
        >>> # Get 3× context crop
        >>> roi = crop_bbox(image, bbox=[100, 100, 50, 50], margin=2.0)
    """
    if not bbox:
        return image

    x, y, w, h = bbox

    # Compute center of bounding box
    cx = x + w / 2.0
    cy = y + h / 2.0

    # Apply margin expansion
    scale = 1.0 + margin
    half_w = (w * scale) / 2.0
    half_h = (h * scale) / 2.0

    # Compute crop boundaries, clamped to image bounds
    left = max(0.0, cx - half_w)
    top = max(0.0, cy - half_h)
    right = min(image.width, cx + half_w)
    bottom = min(image.height, cy + half_h)

    return image.crop((left, top, right, bottom))


# =============================================================================
# TAXONOMY ENCODING
# =============================================================================


def load_and_encode_taxonomy(
    taxonomy_csv: str,
    levels: List[str]
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder], Dict[str, int], Dict[str, Dict[int, str]]]:
    """
    Load taxonomy CSV and create label encoders for each hierarchy level.

    The taxonomy CSV contains the complete hierarchical classification:
        species,genus,family,order,class,phylum,kingdom
        Abralia veranyi,Abralia,Enoploteuthidae,...

    This function:
        1. Loads the CSV and cleans string values
        2. Creates a LabelEncoder for each level
        3. Adds an "unknown" class to handle missing/novel species
        4. Computes class counts for model initialization

    Args:
        taxonomy_csv: Path to taxonomy CSV file
        levels: List of hierarchy levels to encode
                e.g., ["kingdom", "phylum", "class", "order", "family", "genus", "species"]

    Returns:
        taxonomy_df: DataFrame with original columns plus {level}_id columns
        encoders: Dict mapping level names to fitted LabelEncoder objects
        class_counts: Dict mapping level names to number of classes
        id_to_name: Dict mapping level names to {id: name} dicts for decoding

    Example:
        >>> taxonomy_df, encoders, class_counts, id_to_name = load_and_encode_taxonomy(
        ...     "data/taxonomy.csv",
        ...     ["kingdom", "phylum", "class", "order", "family", "genus", "species"]
        ... )
        >>> print(class_counts)
        {'kingdom': 2, 'phylum': 8, 'class': 22, ..., 'species': 80}
    """
    # Load taxonomy
    taxonomy_df = pd.read_csv(taxonomy_csv)

    # Clean species column (primary key)
    taxonomy_df["species"] = taxonomy_df["species"].astype(str).str.strip()

    # Initialize return dictionaries
    encoders: Dict[str, LabelEncoder] = {}
    class_counts: Dict[str, int] = {}
    id_to_name: Dict[str, Dict[int, str]] = {}

    # Process each hierarchy level
    for level in levels:
        if level not in taxonomy_df.columns:
            raise ValueError(f"Column '{level}' missing from taxonomy CSV.")

        # Clean and fill missing values
        taxonomy_df[level] = taxonomy_df[level].fillna("unknown").astype(str).str.strip()

        # Create and fit encoder (include "unknown" for handling unseen labels)
        encoder = LabelEncoder()
        encoder.fit(list(taxonomy_df[level]) + ["unknown"])

        # Add encoded column to dataframe
        taxonomy_df[f"{level}_id"] = encoder.transform(taxonomy_df[level])

        # Store encoder and metadata
        encoders[level] = encoder
        class_counts[level] = len(encoder.classes_)
        id_to_name[level] = {int(i): cls for i, cls in enumerate(encoder.classes_)}

    # Remove duplicate species (keep first occurrence)
    taxonomy_df = taxonomy_df.drop_duplicates(subset=["species"]).reset_index(drop=True)

    return taxonomy_df, encoders, class_counts, id_to_name


# =============================================================================
# DATA AUGMENTATION
# =============================================================================


def create_transforms(cfg: DictConfig):
    """
    Create training and validation image transforms.

    Training transforms include augmentation for regularization:
        - RandomResizedCrop: Scale and aspect variation
        - RandomHorizontalFlip: Mirror augmentation
        - RandomRotation: Rotation augmentation
        - ColorJitter: Color variation

    Validation transforms are minimal (just resize and normalize).

    Both use ImageNet normalization for pretrained backbone compatibility.

    Args:
        cfg: Configuration with augmentation parameters

    Returns:
        train_transform: Composed transform for training
        val_transform: Composed transform for validation/test

    Config structure expected:
        data.img_size: Target image size (e.g., 224)
        augmentation:
            random_resized_crop_scale: [min, max] scale factors
            horizontal_flip: bool
            rotation_degrees: max rotation in degrees
            color_jitter:
                brightness: float
                contrast: float
                saturation: float
                hue: float
    """
    # Build training augmentation pipeline
    train_tfms = [
        transforms.RandomResizedCrop(
            cfg.data.img_size,
            scale=tuple(cfg.augmentation.random_resized_crop_scale),
        )
    ]

    if cfg.augmentation.horizontal_flip:
        train_tfms.append(transforms.RandomHorizontalFlip())

    train_tfms.append(transforms.RandomRotation(cfg.augmentation.rotation_degrees))

    if cfg.augmentation.color_jitter:
        cj = cfg.augmentation.color_jitter
        train_tfms.append(
            transforms.ColorJitter(
                brightness=cj.brightness,
                contrast=cj.contrast,
                saturation=cj.saturation,
                hue=cj.hue,
            )
        )

    # Add tensor conversion and normalization
    train_tfms.extend([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225],    # ImageNet std
        ),
    ])

    # Validation: simple resize + normalize (no augmentation)
    val_tfms = transforms.Compose([
        transforms.Resize((cfg.data.img_size, cfg.data.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return transforms.Compose(train_tfms), val_tfms


# =============================================================================
# DATASET CLASSES - SINGLE SCALE (LEGACY)
# =============================================================================


class FathomNetTaxonomyDataset(Dataset):
    """
    Dataset for on-the-fly ROI and context cropping.

    This dataset loads full images and crops them dynamically:
        - ROI: Tight crop around bounding box (e.g., 10% margin)
        - Context: Wider crop for environmental context (e.g., 200% margin)

    While flexible, this is slower than using pre-cropped images.
    For production training, use MultiScalePrecroppedDataset instead.

    Args:
        frame: DataFrame with columns [image_path, bbox, label]
        taxonomy_df: DataFrame with taxonomy hierarchy
        levels: List of taxonomy levels to include
        encoders: Dict of LabelEncoder for each level
        transform: Image transform to apply
        image_size: Target image size
        roi_margin: Margin for ROI crop (0.1 = 10% expansion)
        context_margin: Margin for context crop (2.0 = 3× size)
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        taxonomy_df: pd.DataFrame,
        levels: List[str],
        encoders: Dict[str, LabelEncoder],
        transform,
        image_size: int,
        roi_margin: float,
        context_margin: float,
    ):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.levels = levels
        self.encoders = encoders
        self.image_size = image_size
        self.roi_margin = roi_margin
        self.context_margin = context_margin

        # Build species name to taxonomy row lookup
        self.name_to_row = self._build_name_lookup(taxonomy_df)

        # Default IDs for unknown species
        self.default_ids = {
            level: int(np.where(encoder.classes_ == "unknown")[0][0])
            if "unknown" in encoder.classes_
            else 0
            for level, encoder in encoders.items()
        }

    def _build_name_lookup(self, taxonomy_df: pd.DataFrame):
        """Build lookup from species name to encoded taxonomy IDs."""
        lookup = {}
        for _, row in taxonomy_df.iterrows():
            row_ids = {level: int(row[f"{level}_id"]) for level in self.levels}
            for level in self.levels:
                key = str(row[level]).strip().lower()
                lookup.setdefault(key, row_ids)
        return lookup

    def __len__(self):
        return len(self.frame)

    def _load_image(self, path: str):
        """Load and validate image from path."""
        try:
            _ensure_real_image(path)
            image = Image.open(path).convert("RGB")
        except (FileNotFoundError, RuntimeError):
            # Propagate dataset issues clearly
            raise
        except Exception:
            # Create dummy for corrupt images
            image = Image.new("RGB", (self.image_size, self.image_size))
        return image

    def __getitem__(self, idx):
        """
        Load a single sample.

        Returns:
            dict with keys:
                roi: Tensor of shape (3, H, W) - tight crop
                context: Tensor of shape (3, H, W) - wide crop
                labels: dict mapping level names to integer IDs
        """
        row = self.frame.iloc[idx]
        image = self._load_image(row["image_path"])
        bbox = row.get("bbox")

        # Create crops at two scales
        roi_crop = crop_bbox(image, bbox, margin=self.roi_margin)
        context_crop = crop_bbox(image, bbox, margin=self.context_margin)

        # Apply transforms
        roi_tensor = self.transform(roi_crop)
        context_tensor = self.transform(context_crop)

        # Get taxonomy labels
        label = str(row["label"]).strip().lower()
        taxonomy_row = self.name_to_row.get(label)
        if taxonomy_row is None:
            taxonomy_row = {level: self.default_ids[level] for level in self.levels}

        labels = {level: int(taxonomy_row[level]) for level in self.levels}

        return {"roi": roi_tensor, "context": context_tensor, "labels": labels}


class FathomNetTestDataset(Dataset):
    """
    Test dataset for on-the-fly cropping (no labels).

    Similar to FathomNetTaxonomyDataset but returns annotation IDs
    instead of labels, for generating submission files.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        transform,
        image_size: int,
        roi_margin: float,
        context_margin: float
    ):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.image_size = image_size
        self.roi_margin = roi_margin
        self.context_margin = context_margin

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        path = row["image_path"]
        bbox = row.get("bbox")

        try:
            _ensure_real_image(path)
            image = Image.open(path).convert("RGB")
        except (FileNotFoundError, RuntimeError):
            raise
        except Exception:
            image = Image.new("RGB", (self.image_size, self.image_size))

        return {
            "roi": self.transform(crop_bbox(image, bbox, margin=self.roi_margin)),
            "context": self.transform(crop_bbox(image, bbox, margin=self.context_margin)),
            "annotation_id": int(row["annotation_id"]),
        }


# =============================================================================
# COLLATE FUNCTIONS - SINGLE SCALE
# =============================================================================


def collate_fn(batch):
    """
    Collate function for training datasets.

    Stacks ROI and context images into batched tensors,
    and converts label dicts to batched tensors.
    """
    roi_images = torch.stack([item["roi"] for item in batch])
    context_images = torch.stack([item["context"] for item in batch])
    labels = {
        level: torch.tensor([item["labels"][level] for item in batch], dtype=torch.long)
        for level in batch[0]["labels"]
    }
    return {"roi": roi_images, "context": context_images}, labels


def test_collate_fn(batch):
    """Collate function for test datasets (returns IDs instead of labels)."""
    roi_images = torch.stack([item["roi"] for item in batch])
    context_images = torch.stack([item["context"] for item in batch])
    ids = torch.tensor([item["annotation_id"] for item in batch], dtype=torch.long)
    return {"roi": roi_images, "context": context_images}, ids


# =============================================================================
# DATASET CLASSES - MULTI-SCALE (PRIMARY)
# =============================================================================


class MultiScalePrecroppedDataset(Dataset):
    """
    Dataset for loading pre-cropped multi-scale ROI images.

    This is the primary dataset for training, loading pre-computed crops
    at multiple scales (1x, 3x, 5x, full). Pre-cropping provides:
        - Faster training (no on-the-fly cropping)
        - Consistent crops across epochs
        - Support for arbitrary number of scales

    Directory Structure Expected:
        roi_root/
        ├── 1x/
        │   ├── {image_id}_{annotation_id}.png
        │   └── ...
        ├── 3x/
        │   └── ...
        ├── 5x/
        │   └── ...
        └── full/
            └── ...

    Args:
        frame: DataFrame with columns [annotation_id, image_id, label]
        taxonomy_df: DataFrame with taxonomy hierarchy
        levels: List of taxonomy levels
        encoders: Dict of LabelEncoder for each level
        transform: Image transform to apply
        roi_root: Path to directory containing scale subdirectories
        scales: List of scale names to load (e.g., ["1x", "3x", "5x"])

    Returns per item:
        dict with keys:
            scales: Dict mapping scale names to tensors (3, H, W)
            labels: Dict mapping level names to integer IDs
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        taxonomy_df: pd.DataFrame,
        levels: List[str],
        encoders: Dict[str, LabelEncoder],
        transform,
        roi_root: str,
        scales: List[str] = ["1x", "3x", "5x"],
    ):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.levels = levels
        self.encoders = encoders
        self.roi_root = roi_root
        self.scales = scales

        # Build name lookup
        self.name_to_row = self._build_name_lookup(taxonomy_df)

        # Default IDs for unknown species
        self.default_ids = {
            level: int(np.where(encoder.classes_ == "unknown")[0][0])
            if "unknown" in encoder.classes_
            else 0
            for level, encoder in encoders.items()
        }

    def _build_name_lookup(self, taxonomy_df: pd.DataFrame):
        """Build lookup from species name to encoded taxonomy IDs."""
        lookup = {}
        for _, row in taxonomy_df.iterrows():
            row_ids = {level: int(row[f"{level}_id"]) for level in self.levels}
            for level in self.levels:
                key = str(row[level]).strip().lower()
                lookup.setdefault(key, row_ids)
        return lookup

    def __len__(self):
        return len(self.frame)

    def _load_scale_image(self, image_id: int, annotation_id: int, scale: str) -> Image.Image:
        """
        Load a pre-cropped image at a specific scale.

        Tries two naming conventions:
            1. {image_id}_{annotation_id}.png (preferred)
            2. {annotation_id}.png (fallback)
        """
        filename = f"{image_id}_{annotation_id}.png"
        path = os.path.join(self.roi_root, scale, filename)

        if os.path.exists(path):
            return Image.open(path).convert("RGB")

        # Fallback: try annotation_id only
        path_ann = os.path.join(self.roi_root, scale, f"{annotation_id}.png")
        if os.path.exists(path_ann):
            return Image.open(path_ann).convert("RGB")

        raise FileNotFoundError(f"Multi-scale image not found: {path}")

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]

        # Get identifiers for finding images
        annotation_id = int(row.get("annotation_id", idx + 1))
        image_id = int(row.get("image_id", annotation_id))

        # Load all scales
        scale_tensors = {}
        for scale in self.scales:
            img = self._load_scale_image(image_id, annotation_id, scale)
            scale_tensors[scale] = self.transform(img)

        # Get taxonomy labels
        label = str(row["label"]).strip().lower()
        taxonomy_row = self.name_to_row.get(label)
        if taxonomy_row is None:
            taxonomy_row = {level: self.default_ids[level] for level in self.levels}
        labels = {level: int(taxonomy_row[level]) for level in self.levels}

        return {
            "scales": scale_tensors,
            "labels": labels,
        }


class MultiScaleTestDataset(Dataset):
    """
    Test dataset for pre-cropped multi-scale images.

    Same as MultiScalePrecroppedDataset but returns annotation_id
    instead of labels, for generating submission files.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        transform,
        roi_root: str,
        scales: List[str] = ["1x", "3x", "5x"],
    ):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.roi_root = roi_root
        self.scales = scales

    def __len__(self):
        return len(self.frame)

    def _load_scale_image(self, image_id: int, annotation_id: int, scale: str) -> Image.Image:
        """Load a pre-cropped image at a specific scale."""
        filename = f"{image_id}_{annotation_id}.png"
        path = os.path.join(self.roi_root, scale, filename)

        if os.path.exists(path):
            return Image.open(path).convert("RGB")

        # Fallback: try annotation_id only
        path_ann = os.path.join(self.roi_root, scale, f"{annotation_id}.png")
        if os.path.exists(path_ann):
            return Image.open(path_ann).convert("RGB")

        raise FileNotFoundError(f"Multi-scale image not found: {path}")

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]

        annotation_id = int(row.get("annotation_id", idx + 1))
        image_id = int(row.get("image_id", annotation_id))

        scale_tensors = {}
        for scale in self.scales:
            img = self._load_scale_image(image_id, annotation_id, scale)
            scale_tensors[scale] = self.transform(img)

        return {
            "scales": scale_tensors,
            "annotation_id": annotation_id,
        }


# =============================================================================
# COLLATE FUNCTIONS - MULTI-SCALE
# =============================================================================


def multiscale_collate_fn(batch):
    """
    Collate function for multi-scale training datasets.

    Stacks images from each scale into separate batched tensors.

    Returns:
        scale_images: Dict mapping scale names to tensors (B, 3, H, W)
        labels: Dict mapping level names to tensors (B,)
    """
    scales = list(batch[0]["scales"].keys())
    scale_images = {
        scale: torch.stack([item["scales"][scale] for item in batch])
        for scale in scales
    }
    labels = {
        level: torch.tensor([item["labels"][level] for item in batch], dtype=torch.long)
        for level in batch[0]["labels"]
    }
    return scale_images, labels


def multiscale_test_collate_fn(batch):
    """Collate function for multi-scale test datasets."""
    scales = list(batch[0]["scales"].keys())
    scale_images = {
        scale: torch.stack([item["scales"][scale] for item in batch])
        for scale in scales
    }
    ids = torch.tensor([item["annotation_id"] for item in batch], dtype=torch.long)
    return scale_images, ids


# =============================================================================
# DATA SPLITTING
# =============================================================================


def stratified_splits(labels: List[str], cfg: DictConfig):
    """
    Create stratified train/val/eval splits.

    Uses sklearn's StratifiedShuffleSplit to preserve class distribution
    across all splits. This is crucial for imbalanced datasets where
    rare species need to appear in validation data.

    Two-stage splitting:
        1. Split into train vs (val + eval)
        2. Split (val + eval) into val vs eval

    Args:
        labels: List of species labels for stratification
        cfg: Configuration with split ratios

    Returns:
        train_idx: Array of training indices
        val_idx: Array of validation indices
        eval_idx: Array of evaluation indices

    Config structure expected:
        data.train_ratio: float (e.g., 0.8)
        data.val_ratio: float (e.g., 0.1)
        data.eval_ratio: float (e.g., 0.1)
        project.seed: int (for reproducibility)

    Raises:
        ValueError: If ratios don't sum to 1.0
    """
    train_ratio = float(cfg.data.train_ratio)
    val_ratio = float(cfg.data.val_ratio)
    eval_ratio = float(cfg.data.eval_ratio)

    # Validate ratios
    if not np.isclose(train_ratio + val_ratio + eval_ratio, 1.0):
        raise ValueError("train/val/eval ratios must sum to 1.0")

    labels = np.array(labels)
    indices = np.arange(len(labels))

    # First split: train vs (val + eval)
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=(1 - train_ratio), random_state=cfg.project.seed
    )
    train_idx, temp_idx = next(sss1.split(indices, labels))

    temp_labels = labels[temp_idx]
    temp_indices = indices[temp_idx]

    # Handle edge case where val + eval is empty
    if val_ratio + eval_ratio == 0:
        return train_idx, np.array([], dtype=int), np.array([], dtype=int)

    # Second split: val vs eval
    relative_eval = eval_ratio / (val_ratio + eval_ratio)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=relative_eval, random_state=cfg.project.seed
    )
    val_idx, eval_idx = next(sss2.split(temp_indices, temp_labels))

    val_indices = temp_indices[val_idx]
    eval_indices = temp_indices[eval_idx]

    return train_idx, val_indices, eval_indices


# =============================================================================
# DATALOADER BUILDERS - SINGLE SCALE
# =============================================================================


def build_dataloaders(
    cfg: DictConfig,
    taxonomy_df: pd.DataFrame,
    encoders: Dict[str, LabelEncoder],
):
    """
    Build train/val/eval dataloaders for single-scale (ROI + context) training.

    This is the legacy loader using on-the-fly cropping.
    For multi-scale training, use build_multiscale_dataloaders instead.
    """
    roi_margin = float(getattr(cfg.data, "roi_margin", 0.1))
    context_margin = float(getattr(cfg.data, "context_margin", 2.0))

    # Load annotations
    if getattr(cfg.paths, "train_coco_json", None):
        annotations = load_coco_annotations(
            cfg.paths.train_coco_json,
            cfg.paths.train_full_image_dir,
            include_labels=True,
        )
    else:
        annotations = load_annotations(
            cfg.paths.train_annotations, cfg.paths.train_image_dir
        )
        annotations["bbox"] = None

    # Create stratified splits
    train_idx, val_idx, eval_idx = stratified_splits(annotations["label"].tolist(), cfg)
    train_df = annotations.iloc[train_idx]
    val_df = annotations.iloc[val_idx]
    eval_df = annotations.iloc[eval_idx]

    # Validate sample images early
    _validate_sample_images(train_df)

    # Create transforms
    train_tfms, val_tfms = create_transforms(cfg)

    # Create datasets
    datasets = {
        "train": FathomNetTaxonomyDataset(
            train_df, taxonomy_df, cfg.data.taxonomy_levels, encoders,
            train_tfms, cfg.data.img_size, roi_margin, context_margin
        ),
        "val": FathomNetTaxonomyDataset(
            val_df, taxonomy_df, cfg.data.taxonomy_levels, encoders,
            val_tfms, cfg.data.img_size, roi_margin, context_margin
        ),
        "eval": FathomNetTaxonomyDataset(
            eval_df, taxonomy_df, cfg.data.taxonomy_levels, encoders,
            val_tfms, cfg.data.img_size, roi_margin, context_margin
        ),
    }

    # Create dataloaders
    dataloaders = {
        split: DataLoader(
            ds,
            batch_size=cfg.data.batch_size,
            shuffle=(split == "train"),
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )
        for split, ds in datasets.items()
    }

    return dataloaders, {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "eval": eval_idx.tolist()
    }


def prepare_test_loader(cfg: DictConfig):
    """Create test dataloader for generating predictions."""
    roi_margin = float(getattr(cfg.data, "roi_margin", 0.1))
    context_margin = float(getattr(cfg.data, "context_margin", 2.0))

    if getattr(cfg.paths, "test_coco_json", None):
        df = load_coco_annotations(
            cfg.paths.test_coco_json,
            cfg.paths.test_full_image_dir,
            include_labels=False,
        )
    else:
        df = pd.read_csv(cfg.paths.test_annotations)
        if "path" not in df.columns:
            raise ValueError("Test annotations must include a 'path' column.")
        if "annotation_id" not in df.columns:
            df["annotation_id"] = np.arange(1, len(df) + 1)
        df["image_path"] = df["path"].astype(str).str.strip().apply(
            lambda p: _resolve_image_path(p, cfg.paths.test_image_dir)
        )
        df["bbox"] = None

    _validate_sample_images(df)
    _, val_tfms = create_transforms(cfg)

    dataset = FathomNetTestDataset(df, val_tfms, cfg.data.img_size, roi_margin, context_margin)
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=test_collate_fn,
    )
    return loader


# =============================================================================
# DATALOADER BUILDERS - MULTI-SCALE (PRIMARY)
# =============================================================================


def build_multiscale_dataloaders(
    cfg: DictConfig,
    taxonomy_df: pd.DataFrame,
    encoders: Dict[str, LabelEncoder],
    scales: List[str] = ["1x", "3x", "5x"],
):
    """
    Build dataloaders using pre-cropped multi-scale ROI images.

    This is the primary dataloader builder for training, using pre-computed
    crops at multiple scales for faster training.

    Args:
        cfg: Configuration object with paths and training settings
        taxonomy_df: DataFrame with taxonomy hierarchy
        encoders: Dict of LabelEncoder for each level
        scales: List of scale names to load (e.g., ["1x", "3x", "5x", "full"])

    Returns:
        dataloaders: Dict with train/val/eval DataLoaders
        split_indices: Dict with indices for each split

    Config structure expected:
        paths.train_roi_root: Path to rois/ directory with scale subdirs
        paths.train_coco_json or paths.train_annotations: Annotation source
        data.batch_size: Batch size
        data.num_workers: Number of data loading workers

    Example:
        >>> dataloaders, indices = build_multiscale_dataloaders(
        ...     cfg, taxonomy_df, encoders, scales=["1x", "3x", "5x", "full"]
        ... )
        >>> for images, labels in dataloaders["train"]:
        ...     # images: {"1x": (B,3,H,W), "3x": ..., "5x": ..., "full": ...}
        ...     outputs = model(images)
    """
    # Get ROI root directory
    roi_root = getattr(
        cfg.paths, "train_roi_root",
        os.path.join(cfg.paths.data_root, "train", "rois")
    )

    # Load annotations
    if getattr(cfg.paths, "train_coco_json", None) and os.path.exists(cfg.paths.train_coco_json):
        annotations = load_coco_annotations(
            cfg.paths.train_coco_json,
            cfg.paths.train_full_image_dir,
            include_labels=True,
        )
    else:
        annotations = load_annotations(
            cfg.paths.train_annotations, cfg.paths.train_image_dir
        )

    # Create stratified splits
    train_idx, val_idx, eval_idx = stratified_splits(annotations["label"].tolist(), cfg)
    train_df = annotations.iloc[train_idx]
    val_df = annotations.iloc[val_idx]
    eval_df = annotations.iloc[eval_idx]

    # Create transforms
    train_tfms, val_tfms = create_transforms(cfg)

    # Create multi-scale datasets
    datasets = {
        "train": MultiScalePrecroppedDataset(
            train_df, taxonomy_df, cfg.data.taxonomy_levels, encoders, train_tfms, roi_root, scales
        ),
        "val": MultiScalePrecroppedDataset(
            val_df, taxonomy_df, cfg.data.taxonomy_levels, encoders, val_tfms, roi_root, scales
        ),
        "eval": MultiScalePrecroppedDataset(
            eval_df, taxonomy_df, cfg.data.taxonomy_levels, encoders, val_tfms, roi_root, scales
        ),
    }

    # Create dataloaders
    dataloaders = {
        split: DataLoader(
            ds,
            batch_size=cfg.data.batch_size,
            shuffle=(split == "train"),
            num_workers=cfg.data.num_workers,
            pin_memory=True,
            collate_fn=multiscale_collate_fn,
        )
        for split, ds in datasets.items()
    }

    return dataloaders, {
        "train": train_idx.tolist(),
        "val": val_idx.tolist(),
        "eval": eval_idx.tolist()
    }
