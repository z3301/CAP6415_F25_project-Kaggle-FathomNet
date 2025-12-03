"""
Multi-Scale Dataset for FathomNet 2025 Competition

This module implements a dataset that extracts ROI + multiple context windows
from images to provide environmental context for hierarchical classification.

Issue: #24
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class MultiScaleFathomNetDataset(Dataset):
    """
    Dataset that loads ROI crop + multiple context windows at different scales.

    Args:
        image_paths: List of image file paths
        species_names: List of species labels
        taxonomy_df: DataFrame with taxonomy information
        encoders: Dict of LabelEncoders for each taxonomic level
        transform: Transform to apply to each scale
        scales: List of scale multipliers (1.0 = tight ROI, >1.0 = wider context)
        taxonomy_levels: List of taxonomic levels to encode
    """

    def __init__(self, image_paths, species_names, taxonomy_df, encoders,
                 transform=None, scales=[1.0, 3.0, 5.0],
                 taxonomy_levels=["kingdom", "phylum", "class", "order", "family", "genus", "species"]):
        self.image_paths = image_paths
        self.species_names = species_names
        self.taxonomy_df = taxonomy_df
        self.encoders = encoders
        self.transform = transform
        self.scales = scales
        self.taxonomy_levels = taxonomy_levels

        # Pre-compute taxonomic info for faster access
        self.taxonomic_info = []
        for species in species_names:
            row = self.taxonomy_df[self.taxonomy_df['species'] == species]
            if row.empty:
                print(f"Warning: Species '{species}' not found in taxonomy_df")
                # Use default (unknown) class
                self.taxonomic_info.append({level: 0 for level in self.taxonomy_levels})
            else:
                row = row.iloc[0]
                self.taxonomic_info.append({
                    level: int(row[f'{level}_id'])
                    for level in self.taxonomy_levels
                    if f'{level}_id' in row
                })

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return dummy images if loading fails
            dummy = torch.zeros(3, 224, 224)
            result = {f'scale_{s}': dummy for s in self.scales}
            result.update(self.taxonomic_info[idx])
            return result

        # Create multi-scale versions
        multi_scale_imgs = {}

        for scale in self.scales:
            if scale == 1.0:
                # Original ROI - no modification
                scaled_img = img
            elif scale > 1.0:
                # Larger context: simulate by padding
                # Note: In production, this should load from original frame with larger bbox
                w, h = img.size
                padding = int((scale - 1.0) * max(w, h) / 2)

                # Pad with black (simulates unknown context)
                # TODO: Replace with actual context from original frames when available
                scaled_img = transforms.Pad(padding, fill=0)(img)
            else:
                # Smaller than ROI: center crop
                w, h = img.size
                new_size = int(min(w, h) * scale)
                scaled_img = transforms.CenterCrop(new_size)(img)

            # Apply transforms if provided
            if self.transform:
                scaled_img = self.transform(scaled_img)

            multi_scale_imgs[f'scale_{scale}'] = scaled_img

        # Add taxonomic labels
        result = multi_scale_imgs
        result.update(self.taxonomic_info[idx])

        return result


def collate_fn_multiscale(batch, taxonomy_levels=["kingdom", "phylum", "class", "order", "family", "genus", "species"]):
    """
    Custom collate function for multi-scale batches.

    Args:
        batch: List of dataset items
        taxonomy_levels: List of taxonomic levels to collect

    Returns:
        images_dict: Dict mapping scale names to batched tensors
        labels: Dict mapping taxonomic levels to label tensors
    """
    # Extract scales from first batch item
    scale_keys = [k for k in batch[0].keys() if k.startswith('scale_')]

    # Stack images for each scale
    images = {}
    for scale_key in scale_keys:
        images[scale_key] = torch.stack([b[scale_key] for b in batch])

    # Collect labels for each taxonomic level
    labels = {}
    for level in taxonomy_levels:
        if level in batch[0]:
            labels[level] = torch.tensor([b[level] for b in batch])

    return images, labels


class MultiScaleTestDataset(Dataset):
    """
    Test dataset that returns multi-scale images without labels.

    Args:
        image_paths: List of image file paths
        annotation_ids: List of annotation IDs for submission
        transform: Transform to apply to each scale
        scales: List of scale multipliers
    """

    def __init__(self, image_paths, annotation_ids, transform=None, scales=[1.0, 3.0, 5.0]):
        self.image_paths = image_paths
        self.annotation_ids = annotation_ids
        self.transform = transform
        self.scales = scales

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        annotation_id = self.annotation_ids[idx]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            dummy = torch.zeros(3, 224, 224)
            result = {f'scale_{s}': dummy for s in self.scales}
            result['annotation_id'] = annotation_id
            return result

        # Create multi-scale versions (same logic as training dataset)
        multi_scale_imgs = {}

        for scale in self.scales:
            if scale == 1.0:
                scaled_img = img
            elif scale > 1.0:
                w, h = img.size
                padding = int((scale - 1.0) * max(w, h) / 2)
                scaled_img = transforms.Pad(padding, fill=0)(img)
            else:
                w, h = img.size
                new_size = int(min(w, h) * scale)
                scaled_img = transforms.CenterCrop(new_size)(img)

            if self.transform:
                scaled_img = self.transform(scaled_img)

            multi_scale_imgs[f'scale_{scale}'] = scaled_img

        multi_scale_imgs['annotation_id'] = annotation_id
        return multi_scale_imgs


def test_collate_fn_multiscale(batch):
    """Collate function for test dataset."""
    scale_keys = [k for k in batch[0].keys() if k.startswith('scale_')]

    images = {}
    for scale_key in scale_keys:
        images[scale_key] = torch.stack([b[scale_key] for b in batch])

    annotation_ids = [b['annotation_id'] for b in batch]

    return images, annotation_ids
