"""
Inference and submission generation for FathomNet 2025 Competition

Contains functions for generating predictions and submission files.
"""

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.data import FathomNetTestDataset, test_collate_fn, create_transforms
from src.model import TaxonomyAwareClassifier


def prepare_test_data(taxonomy_df):
    """
    Prepare test dataset and dataloader.

    Args:
        taxonomy_df: Taxonomy DataFrame

    Returns:
        test_loader: DataLoader for test data
    """
    print("Preparing test dataset...")

    # Load test annotations
    test_annotations = pd.read_csv(Config.TEST_ANNOTATIONS)
    test_image_paths = [os.path.join(Config.TEST_IMAGE_DIR, p) for p in test_annotations['path']]

    # Use row indices as IDs since there's no annotation_id column
    test_annotation_ids = list(range(len(test_annotations)))

    print(f"Using row indices (0 to {len(test_annotations)-1}) as annotation IDs")

    # Create transforms
    _, test_transform = create_transforms()

    # Create test dataset
    test_dataset = FathomNetTestDataset(
        test_image_paths, test_annotation_ids, transform=test_transform
    )

    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        collate_fn=test_collate_fn
    )

    print(f"Test dataset: {len(test_dataset)} images")

    return test_loader


def get_taxonomic_prediction(outputs, id_to_name, confidence_threshold, taxonomy_df):
    """
    Get best taxonomic prediction using hierarchical fallback strategy.

    Args:
        outputs: Model outputs with predictions and confidences
        id_to_name: ID to name mappings
        confidence_threshold: Confidence threshold for predictions
        taxonomy_df: Taxonomy DataFrame

    Returns:
        best_names: List of predicted species names
    """
    # Build fallback maps
    genus_map = taxonomy_df.groupby("genus")["species"].agg(lambda x: x.value_counts().idxmax()).to_dict()
    family_map = taxonomy_df.groupby("family")["species"].agg(lambda x: x.value_counts().idxmax()).to_dict()
    order_map = taxonomy_df.groupby("order")["species"].agg(lambda x: x.value_counts().idxmax()).to_dict()
    class_map = taxonomy_df.groupby("class")["species"].agg(lambda x: x.value_counts().idxmax()).to_dict()
    phylum_map = taxonomy_df.groupby("phylum")["species"].agg(lambda x: x.value_counts().idxmax()).to_dict()
    kingdom_map = taxonomy_df.groupby("kingdom")["species"].agg(lambda x: x.value_counts().idxmax()).to_dict()

    # Convert predictions and confidences
    predictions = {}
    confidences = {}
    for level in Config.TAXONOMY_LEVELS:
        preds = outputs[level]["pred"]
        confs = outputs[level]["conf"]
        predictions[level] = preds.tolist() if isinstance(preds, torch.Tensor) else preds
        confidences[level] = confs.tolist() if isinstance(confs, torch.Tensor) else confs

    # Remap ids to names
    id_to_name = {level: {int(k): str(v) for k, v in id_to_name[level].items()} for level in Config.TAXONOMY_LEVELS}

    best_names = []

    for i in range(len(predictions["species"])):
        # 1. Try species directly
        species_id = predictions["species"][i]
        species_conf = confidences["species"][i]
        species_name = id_to_name["species"].get(species_id)

        if species_name and species_conf >= confidence_threshold:
            best_names.append(species_name)
            continue

        # 2. Fallbacks from genus → kingdom
        fallback_order = [
            ("genus", genus_map),
            ("family", family_map),
            ("order", order_map),
            ("class", class_map),
            ("phylum", phylum_map),
            ("kingdom", kingdom_map)
        ]

        found = False
        for level, level_map in fallback_order:
            level_id = predictions[level][i]
            level_name = id_to_name[level].get(level_id)
            fallback_species = level_map.get(level_name)
            if fallback_species:
                best_names.append(fallback_species)
                found = True
                break

        if not found:
            best_names.append("fallback_species_unknown")

    return best_names


def generate_submission(model_path, test_loader, class_counts, id_to_name, taxonomy_df):
    """
    Generate submission file for the competition.

    Args:
        model_path: Path to trained model checkpoint
        test_loader: DataLoader for test data
        class_counts: Dict of class counts per taxonomic level
        id_to_name: ID to name mappings
        taxonomy_df: Taxonomy DataFrame

    Returns:
        submission_df: DataFrame with submission predictions
    """
    print("Generating submission file...")

    # Load the best model
    model = TaxonomyAwareClassifier.load_from_checkpoint(
        model_path, class_counts=class_counts
    )
    model.eval()

    # Use GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running inference on: {device}")
    model.to(device)

    # Process test data manually
    all_annotation_ids = []
    all_predictions = {level: [] for level in Config.TAXONOMY_LEVELS}
    all_confidences = {level: [] for level in Config.TAXONOMY_LEVELS}

    model.eval()
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing test data"):
            images, annotation_ids = batch
            images = images.to(device)

            # Forward pass
            outputs = model(images)

            # Process outputs
            all_annotation_ids.extend(annotation_ids)

            for level in Config.TAXONOMY_LEVELS:
                if level in outputs:
                    # Convert logits to probabilities
                    probs = torch.softmax(outputs[level], dim=1)
                    preds = torch.argmax(probs, dim=1)
                    conf = torch.gather(probs, 1, preds.unsqueeze(1)).squeeze(1)

                    all_predictions[level].extend(preds.cpu().numpy())
                    all_confidences[level].extend(conf.cpu().numpy())

    # Get final taxonomic predictions
    best_names = get_taxonomic_prediction(
        {
            level: {'pred': all_predictions[level], 'conf': all_confidences[level]}
            for level in Config.TAXONOMY_LEVELS
        },
        id_to_name,
        confidence_threshold=Config.CONFIDENCE_THRESHOLD,
        taxonomy_df=taxonomy_df
    )

    # Create submission DataFrame
    submission_df = pd.DataFrame({
        'annotation_id': [id + 1 for id in all_annotation_ids],
        'concept_name': best_names
    })

    # Save submission file
    submission_df.to_csv(Config.SUBMISSION_PATH, index=False)
    print(f"Submission saved to {Config.SUBMISSION_PATH}")

    # Display submission distribution
    print("\nPrediction distribution:")
    value_counts = submission_df['concept_name'].value_counts().head(10)
    for name, count in value_counts.items():
        print(f"  {name}: {count} ({count/len(submission_df)*100:.1f}%)")

    return submission_df
