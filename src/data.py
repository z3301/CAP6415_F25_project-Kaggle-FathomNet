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


def load_annotations(csv_path: str, image_root: str) -> pd.DataFrame:
    """Legacy CSV loader that only provides cropped ROI paths and labels."""
    df = pd.read_csv(csv_path)
    required = {"path", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Annotations file {csv_path} missing columns: {missing}")
    df["path"] = df["path"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
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
    Load annotations directly from the COCO-style dataset JSON so we can crop
    both the ROI and a larger context region on the fly.
    """
    with open(coco_json, "r") as f:
        data = json.load(f)

    categories = {int(cat["id"]): str(cat.get("name") or "unknown") for cat in data.get("categories", [])}
    images = {}
    for img in data.get("images", []):
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

    rows = []
    for ann in data.get("annotations", []):
        image_info = images.get(int(ann["image_id"]))
        if not image_info:
            continue
        label = None
        if include_labels:
            label = categories.get(int(ann["category_id"] or -1), "unknown")
        rows.append(
            {
                "annotation_id": int(ann["id"]),
                "image_id": int(ann["image_id"]),
                "label": label,
                "bbox": [float(x) for x in ann.get("bbox", [])] if ann.get("bbox") else None,
                "image_path": image_info["path"],
            }
        )
    df = pd.DataFrame(rows)
    if include_labels and df["label"].isnull().any():
        df["label"] = df["label"].fillna("unknown")
    return df


def _is_git_lfs_pointer(path: str) -> bool:
    """Check whether a file contains a Git LFS pointer instead of real image bytes."""
    try:
        with open(path, "rb") as f:
            header = f.read(256)
        return b"git-lfs" in header or b"git lfs" in header
    except OSError:
        # If the file can't be read, let the caller handle it.
        return False


def _ensure_real_image(path: str):
    """Fail fast when the target is missing or still an unfetched Git LFS pointer."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")
    if _is_git_lfs_pointer(path):
        raise RuntimeError(
            f"Image at '{path}' is a Git LFS pointer. "
            "Download or copy the actual files (e.g., run `git lfs pull` in the dataset source) before training or inference."
        )


def _validate_sample_images(frame: pd.DataFrame, sample_size: int = 5):
    """Check a handful of rows early so we fail fast when the dataset is missing."""
    for path in frame["image_path"].head(sample_size):
        _ensure_real_image(path)


def _resolve_image_path(path: str, image_root: str) -> str:
    """Normalize an image path and fall back to the provided root when needed."""
    path = os.path.normpath(path)
    if os.path.exists(path) and not _is_git_lfs_pointer(path):
        return path
    # Prefer the configured image_root when the original path is missing or points to an LFS stub.
    candidate = os.path.normpath(os.path.join(image_root, os.path.basename(path)))
    if os.path.exists(candidate) and not _is_git_lfs_pointer(candidate):
        return candidate
    parts = path.split(os.sep)
    if "rois" in parts:
        suffix = parts[parts.index("rois") + 1 :]
        candidate = os.path.normpath(os.path.join(image_root, *suffix))
        if os.path.exists(candidate) and not _is_git_lfs_pointer(candidate):
            return candidate
    if "images" in parts:
        suffix = parts[parts.index("images") + 1 :]
        candidate = os.path.normpath(os.path.join(image_root, *suffix))
        if os.path.exists(candidate) and not _is_git_lfs_pointer(candidate):
            return candidate
    return path


def crop_bbox(image: Image.Image, bbox: Optional[List[float]], margin: float) -> Image.Image:
    """
    Crop an image around a bounding box with an optional margin.

    margin=0.1 enlarges the box to 110% of its size. margin=2.0 yields a 3x crop.
    """
    if not bbox:
        return image
    x, y, w, h = bbox
    cx = x + w / 2.0
    cy = y + h / 2.0
    scale = 1.0 + margin
    half_w = (w * scale) / 2.0
    half_h = (h * scale) / 2.0
    left = max(0.0, cx - half_w)
    top = max(0.0, cy - half_h)
    right = min(image.width, cx + half_w)
    bottom = min(image.height, cy + half_h)
    return image.crop((left, top, right, bottom))


def load_and_encode_taxonomy(taxonomy_csv: str, levels: List[str]) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder], Dict[str, int], Dict[str, Dict[int, str]]]:
    taxonomy_df = pd.read_csv(taxonomy_csv)
    taxonomy_df["species"] = taxonomy_df["species"].astype(str).str.strip()
    encoders: Dict[str, LabelEncoder] = {}
    class_counts: Dict[str, int] = {}
    id_to_name: Dict[str, Dict[int, str]] = {}

    for level in levels:
        if level not in taxonomy_df.columns:
            raise ValueError(f"Column '{level}' missing from taxonomy CSV.")
        taxonomy_df[level] = taxonomy_df[level].fillna("unknown").astype(str).str.strip()
        encoder = LabelEncoder()
        encoder.fit(list(taxonomy_df[level]) + ["unknown"])
        taxonomy_df[f"{level}_id"] = encoder.transform(taxonomy_df[level])
        encoders[level] = encoder
        class_counts[level] = len(encoder.classes_)
        id_to_name[level] = {int(i): cls for i, cls in enumerate(encoder.classes_)}

    taxonomy_df = taxonomy_df.drop_duplicates(subset=["species"]).reset_index(drop=True)
    return taxonomy_df, encoders, class_counts, id_to_name


def create_transforms(cfg: DictConfig):
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
    train_tfms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    val_tfms = transforms.Compose(
        [
            transforms.Resize((cfg.data.img_size, cfg.data.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return transforms.Compose(train_tfms), val_tfms


class FathomNetTaxonomyDataset(Dataset):
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
        self.name_to_row = self._build_name_lookup(taxonomy_df)
        self.default_ids = {
            level: int(np.where(encoder.classes_ == "unknown")[0][0])
            if "unknown" in encoder.classes_
            else 0
            for level, encoder in encoders.items()
        }

    def _build_name_lookup(self, taxonomy_df: pd.DataFrame):
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
        try:
            _ensure_real_image(path)
            image = Image.open(path).convert("RGB")
        except (FileNotFoundError, RuntimeError):
            # Propagate missing data and Git LFS pointer issues so the user can fix the dataset.
            raise
        except Exception:
            image = Image.new("RGB", (self.image_size, self.image_size))
        return image

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        image = self._load_image(row["image_path"])
        bbox = row.get("bbox")
        roi_crop = crop_bbox(image, bbox, margin=self.roi_margin)
        context_crop = crop_bbox(image, bbox, margin=self.context_margin)
        roi_tensor = self.transform(roi_crop)
        context_tensor = self.transform(context_crop)
        label = str(row["label"]).strip().lower()
        taxonomy_row = self.name_to_row.get(label)
        if taxonomy_row is None:
            taxonomy_row = {level: self.default_ids[level] for level in self.levels}
        labels = {
            level: int(taxonomy_row[level])
            for level in self.levels
        }
        return {"roi": roi_tensor, "context": context_tensor, "labels": labels}


class FathomNetTestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform, image_size: int, roi_margin: float, context_margin: float):
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
            # Make the root cause explicit instead of silently returning a dummy image.
            raise
        except Exception:
            image = Image.new("RGB", (self.image_size, self.image_size))
        return {
            "roi": self.transform(crop_bbox(image, bbox, margin=self.roi_margin)),
            "context": self.transform(crop_bbox(image, bbox, margin=self.context_margin)),
            "annotation_id": int(row["annotation_id"]),
        }


def collate_fn(batch):
    roi_images = torch.stack([item["roi"] for item in batch])
    context_images = torch.stack([item["context"] for item in batch])
    labels = {level: torch.tensor([item["labels"][level] for item in batch], dtype=torch.long) for level in batch[0]["labels"]}
    return {"roi": roi_images, "context": context_images}, labels


def test_collate_fn(batch):
    roi_images = torch.stack([item["roi"] for item in batch])
    context_images = torch.stack([item["context"] for item in batch])
    ids = torch.tensor([item["annotation_id"] for item in batch], dtype=torch.long)
    return {"roi": roi_images, "context": context_images}, ids


def stratified_splits(labels: List[str], cfg: DictConfig):
    train_ratio = float(cfg.data.train_ratio)
    val_ratio = float(cfg.data.val_ratio)
    eval_ratio = float(cfg.data.eval_ratio)
    if not np.isclose(train_ratio + val_ratio + eval_ratio, 1.0):
        raise ValueError("train/val/eval ratios must sum to 1.0")
    labels = np.array(labels)
    indices = np.arange(len(labels))
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=(1 - train_ratio), random_state=cfg.project.seed
    )
    train_idx, temp_idx = next(sss1.split(indices, labels))
    temp_labels = labels[temp_idx]
    temp_indices = indices[temp_idx]
    if val_ratio + eval_ratio == 0:
        return train_idx, np.array([], dtype=int), np.array([], dtype=int)
    relative_eval = eval_ratio / (val_ratio + eval_ratio)
    sss2 = StratifiedShuffleSplit(
        n_splits=1, test_size=relative_eval, random_state=cfg.project.seed
    )
    val_idx, eval_idx = next(sss2.split(temp_indices, temp_labels))
    val_indices = temp_indices[val_idx]
    eval_indices = temp_indices[eval_idx]
    return train_idx, val_indices, eval_indices


def build_dataloaders(
    cfg: DictConfig,
    taxonomy_df: pd.DataFrame,
    encoders: Dict[str, LabelEncoder],
):
    roi_margin = float(getattr(cfg.data, "roi_margin", 0.1))
    context_margin = float(getattr(cfg.data, "context_margin", 2.0))
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
    train_idx, val_idx, eval_idx = stratified_splits(annotations["label"].tolist(), cfg)
    train_df = annotations.iloc[train_idx]
    val_df = annotations.iloc[val_idx]
    eval_df = annotations.iloc[eval_idx]
    # Fail early if the dataset only contains Git LFS pointers or missing files.
    _validate_sample_images(train_df)
    train_tfms, val_tfms = create_transforms(cfg)
    datasets = {
        "train": FathomNetTaxonomyDataset(train_df, taxonomy_df, cfg.data.taxonomy_levels, encoders, train_tfms, cfg.data.img_size, roi_margin, context_margin),
        "val": FathomNetTaxonomyDataset(val_df, taxonomy_df, cfg.data.taxonomy_levels, encoders, val_tfms, cfg.data.img_size, roi_margin, context_margin),
        "eval": FathomNetTaxonomyDataset(eval_df, taxonomy_df, cfg.data.taxonomy_levels, encoders, val_tfms, cfg.data.img_size, roi_margin, context_margin),
    }
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
    return dataloaders, {"train": train_idx.tolist(), "val": val_idx.tolist(), "eval": eval_idx.tolist()}


def prepare_test_loader(cfg: DictConfig):
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
