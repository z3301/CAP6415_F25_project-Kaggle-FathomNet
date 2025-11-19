import os
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from tqdm import tqdm


def _build_fallback_maps(taxonomy_df: pd.DataFrame, levels: List[str]):
    fallback_maps = {}
    for level in levels:
        mapping = (
            taxonomy_df.groupby(level)["species"]
            .agg(lambda x: x.value_counts().idxmax())
            .to_dict()
        )
        fallback_maps[level] = mapping
    return fallback_maps


def taxonomic_fallback(
    preds: Dict[str, List[int]],
    confidences: Dict[str, List[float]],
    id_to_name: Dict[str, Dict[int, str]],
    taxonomy_df: pd.DataFrame,
    cfg: DictConfig,
):
    fallback_enabled = cfg.inference.get("enable_fallback", False)
    threshold = cfg.inference.confidence_threshold

    # If fallback is disabled or threshold <= 0, just map species ids to names directly.
    if (not fallback_enabled) or threshold <= 0:
        return [
            id_to_name["species"].get(int(species_id), "UNKNOWN")
            for species_id in preds["species"]
        ]

    fallback_levels = ["genus", "family", "order", "class", "phylum", "kingdom"]
    fallback_maps = _build_fallback_maps(
        taxonomy_df, fallback_levels[::-1]
    )
    best_species = []
    for idx in range(len(preds["species"])):
        species_id = preds["species"][idx]
        species_conf = confidences["species"][idx]
        species_name = id_to_name["species"].get(int(species_id))
        if species_name and species_conf >= threshold:
            best_species.append(species_name)
            continue
        chosen = None
        for level in fallback_levels:
            level_id = preds[level][idx]
            level_name = id_to_name[level].get(int(level_id))
            fallback = fallback_maps.get(level, {}).get(level_name)
            if fallback:
                chosen = fallback
                break
        best_species.append(chosen or "UNKNOWN")
    return best_species


def generate_submission(
    model,
    dataloader,
    cfg: DictConfig,
    id_to_name: Dict[str, Dict[int, str]],
    taxonomy_df: pd.DataFrame,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    all_annotation_ids: List[int] = []
    preds = {level: [] for level in cfg.data.taxonomy_levels}
    confidences = {level: [] for level in cfg.data.taxonomy_levels}
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            images, annotation_ids = batch
            if isinstance(images, dict):
                images = {k: v.to(device) for k, v in images.items()}
            else:
                images = images.to(device)
            batch_ids = annotation_ids.cpu().tolist()
            outputs = model(images)
            all_annotation_ids.extend(batch_ids)
            for level in cfg.data.taxonomy_levels:
                level_logits = outputs[level]
                level_probs = torch.softmax(level_logits, dim=1)
                level_pred = torch.argmax(level_probs, dim=1)
                preds[level].extend(level_pred.cpu().tolist())
                confidence = torch.gather(level_probs, 1, level_pred.unsqueeze(1)).squeeze(1)
                confidences[level].extend(confidence.cpu().tolist())
    best_species = taxonomic_fallback(preds, confidences, id_to_name, taxonomy_df, cfg)
    submission = pd.DataFrame(
        {
            "annotation_id": all_annotation_ids,
            "concept_name": best_species,
        }
    ).sort_values("annotation_id")
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    submission_path = os.path.join(cfg.paths.output_dir, "submission.csv")
    submission.to_csv(submission_path, index=False)
    return submission_path
