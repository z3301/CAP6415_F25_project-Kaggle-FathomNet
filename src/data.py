"""
Dataset classes for FathomNet 2025 Competition

Single-scale dataset classes for training and inference.
"""

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from src.config import Config


class FathomNetTaxonomyDataset(Dataset):
    """
    Dataset for FathomNet with taxonomic information.

    Args:
        image_paths: List of image file paths
        species_names: List of species labels
        taxonomy_df: DataFrame with taxonomy information
        encoders: Dict of LabelEncoders for each taxonomic level
        transform: Transform to apply to images
    """

    def __init__(self, image_paths, species_names, taxonomy_df, encoders, transform=None):
        self.image_paths = image_paths
        self.species_names = species_names
        self.taxonomy_df = taxonomy_df
        self.encoders = encoders
        self.transform = transform

        # Pre-compute taxonomic info for faster access
        self.taxonomic_info = []
        for species in species_names:
            row = self.taxonomy_df[self.taxonomy_df['species'] == species]
            if row.empty:
                print(f"Warning: Species '{species}' not found in taxonomy_df")
                # Use default (unknown) class
                self.taxonomic_info.append({level: 0 for level in Config.TAXONOMY_LEVELS})
            else:
                row = row.iloc[0]
                self.taxonomic_info.append({
                    level: int(row[f'{level}_id'])
                    for level in Config.TAXONOMY_LEVELS
                    if f'{level}_id' in row
                })

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # Handle image loading errors gracefully
        try:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image of the correct size
            img = torch.zeros(3, Config.IMG_SIZE, Config.IMG_SIZE)

        # Return image and taxonomic labels
        result = {'image': img}
        result.update(self.taxonomic_info[idx])

        return result


class FathomNetTestDataset(Dataset):
    """
    Dataset for FathomNet test set without labels.

    Args:
        image_paths: List of image file paths
        annotation_ids: List of annotation IDs for submission
        transform: Transform to apply to images
    """

    def __init__(self, image_paths, annotation_ids, transform=None):
        self.image_paths = image_paths
        self.annotation_ids = annotation_ids
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        annotation_id = self.annotation_ids[idx]

        # Handle image loading errors gracefully
        try:
            img = Image.open(img_path).convert("RGB")
            if self.transform:
                img = self.transform(img)
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image of the correct size
            img = torch.zeros(3, Config.IMG_SIZE, Config.IMG_SIZE)

        return {'image': img, 'annotation_id': annotation_id}


def collate_fn(batch):
    """Custom collate function to handle the taxonomy dataset."""
    images = torch.stack([b['image'] for b in batch])

    # Collect labels for each taxonomic level
    labels = {}
    for level in Config.TAXONOMY_LEVELS:
        if level in batch[0]:
            labels[level] = torch.tensor([b[level] for b in batch])

    return images, labels


def test_collate_fn(batch):
    """Custom collate function for test dataset."""
    images = torch.stack([b['image'] for b in batch])
    annotation_ids = [b['annotation_id'] for b in batch]

    return images, annotation_ids


def create_transforms():
    """
    Create augmentation pipelines for train and validation/test sets.

    Returns:
        train_transform: Transform pipeline for training
        val_transform: Transform pipeline for validation/test
    """
    # Training - strong augmentation to help with model generalization
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(Config.IMG_SIZE, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Validation/Test - just resize and normalize
    val_transform = transforms.Compose([
        transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform
