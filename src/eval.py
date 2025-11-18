import os
from collections import defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from tqdm import tqdm


def evaluate_model(
    model,
    dataloader,
    cfg: DictConfig,
    id_to_name: Dict[str, Dict[int, str]],
    prefix: str,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    all_preds = defaultdict(list)
    all_targets = defaultdict(list)
    with torch.no_grad():
        for images, targets in tqdm(dataloader, desc=f"{prefix} evaluation"):
            images = images.to(device)
            outputs = model(images)
            for level in cfg.data.taxonomy_levels:
                if level in outputs and level in targets:
                    preds = torch.argmax(outputs[level], dim=1).cpu()
                    target = targets[level].cpu()
                    all_preds[level].append(preds)
                    all_targets[level].append(target)
    reports = {}
    for level in cfg.data.taxonomy_levels:
        if not all_preds[level]:
            continue
        preds = torch.cat(all_preds[level])
        targets = torch.cat(all_targets[level])
        acc = (preds == targets).float().mean().item()
        print(f"{prefix} {level} accuracy: {acc:.4f}")
        labels = list(range(len(id_to_name[level])))
        names = [id_to_name[level].get(int(i), "unknown") for i in labels]
        report = classification_report(
            targets,
            preds,
            labels=labels,
            target_names=names,
            zero_division=0,
        )
        reports[level] = report
        cm = confusion_matrix(targets, preds, labels=labels, normalize="true")
        fig, ax = plt.subplots(figsize=(8, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)
        disp.plot(ax=ax, xticks_rotation=90, values_format=".2f", cmap="Blues", colorbar=False)
        ax.set_title(f"{prefix} {level} confusion matrix")
        fig.tight_layout()
        fig_path = os.path.join(cfg.paths.output_dir, f"{prefix}_{level}_cm.png")
        fig.savefig(fig_path)
        plt.close(fig)
        with open(os.path.join(cfg.paths.output_dir, f"{prefix}_{level}_report.txt"), "w") as f:
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write(report)
    return reports


def save_ground_truths(dataset, id_to_name, cfg: DictConfig):
    rows: List[Dict] = []
    for idx in range(len(dataset)):
        item = dataset[idx]
        species_id = item["labels"]["species"]
        name = id_to_name["species"].get(int(species_id), "UNKNOWN")
        rows.append({"annotation_id": idx + 1, "concept_name": name})
    df = pd.DataFrame(rows)
    path = os.path.join(cfg.paths.output_dir, "ground_truths.csv")
    df.to_csv(path, index=False)
    return path


def save_predictions(model, dataloader, id_to_name, cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    rows: List[Dict] = []
    annotation_id = 1
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Saving predictions"):
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs["species"], dim=1).cpu().tolist()
            for pred in preds:
                rows.append(
                    {"annotation_id": annotation_id, "concept_name": id_to_name["species"].get(int(pred), "UNKNOWN")}
                )
                annotation_id += 1
    df = pd.DataFrame(rows)
    path = os.path.join(cfg.paths.output_dir, "predictions.csv")
    df.to_csv(path, index=False)
    return path
