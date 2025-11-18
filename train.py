import argparse
import json
import os

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint

from src.data import build_dataloaders, load_and_encode_taxonomy
from src.eval import evaluate_model
from src.model import TaxonomyAwareClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Train the taxonomy-aware classifier.")
    parser.add_argument("--config", type=str, default="config/experiment-default.yaml", help="Path to YAML config.")
    parser.add_argument("--resume_from", type=str, default=None, help="Optional checkpoint to resume from.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    os.makedirs(cfg.paths.output_dir, exist_ok=True)
    checkpoint_dir = os.path.join(cfg.paths.output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    if torch.cuda.is_available():
        # Enable Tensor Core friendly matmul kernels on Ampere+ GPUs
        torch.set_float32_matmul_precision("medium")
    pl.seed_everything(cfg.project.seed)
    taxonomy_df, encoders, class_counts, id_to_name = load_and_encode_taxonomy(
        cfg.paths.taxonomy_csv, cfg.data.taxonomy_levels
    )
    dataloaders, splits = build_dataloaders(cfg, taxonomy_df, encoders)
    model = TaxonomyAwareClassifier(cfg, class_counts)

    checkpoint_cb = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="fathomnet-{epoch:02d}-{val_loss:.4f}",
        monitor=cfg.callbacks.monitor_metric,
        mode=cfg.callbacks.monitor_mode,
        save_top_k=1,
        save_last=True,
    )
    early_stop_cb = EarlyStopping(
        monitor=cfg.callbacks.monitor_metric,
        patience=cfg.callbacks.early_stopping_patience,
        mode=cfg.callbacks.monitor_mode,
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=accelerator,
        devices=1,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        precision=cfg.training.precision,
        callbacks=[checkpoint_cb, early_stop_cb, lr_monitor],
        default_root_dir=cfg.paths.output_dir,
    )
    trainer.fit(
        model,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["val"],
        ckpt_path=args.resume_from,
    )
    metadata = {
        "config": OmegaConf.to_container(cfg, resolve=True),
        "class_counts": class_counts,
        "splits": splits,
        "best_model": checkpoint_cb.best_model_path,
    }
    with open(os.path.join(cfg.paths.output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    if checkpoint_cb.best_model_path and len(dataloaders["eval"].dataset) > 0:
        eval_model = TaxonomyAwareClassifier.load_from_checkpoint(
            checkpoint_cb.best_model_path,
            cfg=cfg,
            class_counts=class_counts,
        )
        evaluate_model(eval_model, dataloaders["eval"], cfg, id_to_name, prefix="eval")


if __name__ == "__main__":
    main()
