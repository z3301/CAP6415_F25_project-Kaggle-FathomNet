"""
Taxonomy utilities for FathomNet 2025 Competition

Handles taxonomy loading, encoding, and hierarchical relationships.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import Config


def load_and_encode_taxonomy():
    """
    Load taxonomy data and create label encoders for each taxonomic level.

    Returns:
        taxonomy_df: DataFrame with taxonomy information and encoded IDs
        encoders: Dict of LabelEncoders for each taxonomic level
        class_counts: Dict mapping taxonomic levels to number of classes
        id_to_name: Dict mapping IDs to names for each level
        name_to_id: Dict mapping names to IDs for each level
    """
    print("Loading taxonomy data...")
    taxonomy_df = pd.read_csv(Config.TAXONOMY_PATH)

    encoders = {}
    class_counts = {}
    id_to_name = {}
    name_to_id = {}

    for level in Config.TAXONOMY_LEVELS:
        if level not in taxonomy_df.columns:
            print(f"Warning: '{level}' not found in taxonomy data!")
            continue

        # Fill missing values and normalize text
        taxonomy_df[level] = taxonomy_df[level].fillna("unknown").astype(str).str.strip()

        # Make sure "unknown" is included as a class
        unique_labels = taxonomy_df[level].unique().tolist()
        if "unknown" not in unique_labels:
            unique_labels.append("unknown")

        le = LabelEncoder()
        le.fit(unique_labels)

        # Encode and store
        taxonomy_df[f"{level}_id"] = le.transform(taxonomy_df[level])
        encoders[level] = le
        class_counts[level] = len(le.classes_)
        id_to_name[level] = {i: name for i, name in enumerate(le.classes_)}
        name_to_id[level] = {name: i for i, name in enumerate(le.classes_)}

        print(f"{level.capitalize():<8}: {class_counts[level]:>3} classes")

    return taxonomy_df, encoders, class_counts, id_to_name, name_to_id
