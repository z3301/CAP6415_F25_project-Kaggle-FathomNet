import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def load_annotations(csv_path: str, image_root: str) -> pd.DataFrame:
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


def _resolve_image_path(path: str, image_root: str) -> str:
    """Normalize an image path and fall back to the provided root when needed."""
    path = os.path.normpath(path)
    if os.path.exists(path):
        return path
    candidate = os.path.normpath(os.path.join(image_root, os.path.basename(path)))
    if os.path.exists(candidate):
        return candidate
    parts = path.split(os.sep)
    if "rois" in parts:
        suffix = parts[parts.index("rois") + 1 :]
        candidate = os.path.normpath(os.path.join(image_root, *suffix))
        if os.path.exists(candidate):
            return candidate
    return path


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
    ):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.levels = levels
        self.encoders = encoders
        self.image_size = image_size
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
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (self.image_size, self.image_size))
        return self.transform(image)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        image = self._load_image(row["image_path"])
        label = str(row["label"]).strip().lower()
        taxonomy_row = self.name_to_row.get(label)
        if taxonomy_row is None:
            taxonomy_row = {level: self.default_ids[level] for level in self.levels}
        labels = {
            level: int(taxonomy_row[level])
            for level in self.levels
        }
        return {"image": image, "labels": labels}


class FathomNetTestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, transform, image_size: int):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self.image_size = image_size

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        try:
            image = Image.open(row["image_path"]).convert("RGB")
        except Exception:
            image = Image.new("RGB", (self.image_size, self.image_size))
        return {
            "image": self.transform(image),
            "annotation_id": int(row["annotation_id"]),
        }


def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    labels = {level: torch.tensor([item["labels"][level] for item in batch], dtype=torch.long) for level in batch[0]["labels"]}
    return images, labels


def test_collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    ids = torch.tensor([item["annotation_id"] for item in batch], dtype=torch.long)
    return images, ids


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
    annotations = load_annotations(
        cfg.paths.train_annotations, cfg.paths.train_image_dir
    )
    train_idx, val_idx, eval_idx = stratified_splits(annotations["label"].tolist(), cfg)
    train_df = annotations.iloc[train_idx]
    val_df = annotations.iloc[val_idx]
    eval_df = annotations.iloc[eval_idx]
    train_tfms, val_tfms = create_transforms(cfg)
    datasets = {
        "train": FathomNetTaxonomyDataset(train_df, taxonomy_df, cfg.data.taxonomy_levels, encoders, train_tfms, cfg.data.img_size),
        "val": FathomNetTaxonomyDataset(val_df, taxonomy_df, cfg.data.taxonomy_levels, encoders, val_tfms, cfg.data.img_size),
        "eval": FathomNetTaxonomyDataset(eval_df, taxonomy_df, cfg.data.taxonomy_levels, encoders, val_tfms, cfg.data.img_size),
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
    df = pd.read_csv(cfg.paths.test_annotations)
    if "path" not in df.columns:
        raise ValueError("Test annotations must include a 'path' column.")
    if "annotation_id" not in df.columns:
        df["annotation_id"] = np.arange(1, len(df) + 1)
    df["image_path"] = df["path"].astype(str).str.strip().apply(
        lambda p: _resolve_image_path(p, cfg.paths.test_image_dir)
    )
    _, val_tfms = create_transforms(cfg)
    dataset = FathomNetTestDataset(df, val_tfms, cfg.data.img_size)
    loader = DataLoader(
        dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=True,
        collate_fn=test_collate_fn,
    )
    return loader
