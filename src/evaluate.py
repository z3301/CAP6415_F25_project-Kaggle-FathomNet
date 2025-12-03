"""
Evaluation and metrics for FathomNet 2025 Competition

Contains evaluation functions for model performance analysis.
"""

import os
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils.multiclass import unique_labels
from tqdm import tqdm

from src.config import Config


def evaluate_model(model, dataloader, class_counts, id_to_name, name="Evaluation"):
    """
    Evaluate model performance and generate reports.

    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        class_counts: Dict of class counts per taxonomic level
        id_to_name: Dict mapping IDs to names for each level
        name: Name of evaluation set (for file naming)
    """
    print(f"\n{'='*60}")
    print(f"RUNNING {name.upper()} METRICS")
    print(f"{'='*60}\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    name_slug = name.lower().replace(" ", "_")

    all_preds = {level: [] for level in Config.TAXONOMY_LEVELS}
    all_targets = {level: [] for level in Config.TAXONOMY_LEVELS}

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"{name} loop"):
            images, targets = batch
            images = images.to(device)
            outputs = model(images)

            for level in Config.TAXONOMY_LEVELS:
                if level in outputs and level in targets:
                    preds = torch.argmax(outputs[level], dim=1).cpu()
                    targets_level = targets[level].cpu()
                    all_preds[level].append(preds)
                    all_targets[level].append(targets_level)

    for level in Config.TAXONOMY_LEVELS:
        if all_preds[level]:
            preds = torch.cat(all_preds[level])
            targets = torch.cat(all_targets[level])

            acc = (preds == targets).float().mean().item()
            print(f"\n{level.capitalize()} Accuracy: {acc:.4f}")

            # Get actual labels present in the data
            labels = sorted(set(unique_labels(targets, preds)))
            class_names = [id_to_name[level][i] for i in labels]

            report = classification_report(
                targets,
                preds,
                labels=labels,
                target_names=class_names,
                zero_division=0
            )

            # Print to console
            print(f"\n{level.capitalize()} Classification Report:\n{report}")

            # Save report
            report_path = os.path.join(Config.OUTPUT_DIR, f"{name_slug}_{level}_report.txt")
            with open(report_path, "w") as f:
                f.write(f"{level.capitalize()} Accuracy: {acc:.4f}\n\n")
                f.write(report)
            print(f"Report saved to {report_path}")

            # Confusion matrix
            cm = confusion_matrix(targets, preds, normalize="true")
            fig, ax = plt.subplots(figsize=(12, 12))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
            disp.plot(ax=ax, cmap="Blues", xticks_rotation=90)

            # Remove text annotations for large matrices
            for text in disp.text_.ravel():
                text.set_visible(False)

            plt.title(f"{level.capitalize()} Confusion Matrix ({name})")
            plt.tight_layout()

            # Save
            cm_path = os.path.join(Config.OUTPUT_DIR, f"{name_slug}_{level}_confusion_matrix.png")
            plt.savefig(cm_path)
            print(f"Confusion matrix saved to {cm_path}")
            plt.close()
