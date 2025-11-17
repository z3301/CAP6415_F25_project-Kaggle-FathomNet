import argparse

import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.data import build_dataloaders, load_and_encode_taxonomy
from src.eval import evaluate_model
from src.model import TaxonomyAwareClassifier


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained model on a dataset split.")
    parser.add_argument("--config", type=str, default="config/experiment-default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint (.ckpt).")
    parser.add_argument("--split", choices=["train", "val", "eval"], default="eval")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    pl.seed_everything(cfg.project.seed)
    taxonomy_df, encoders, class_counts, id_to_name = load_and_encode_taxonomy(
        cfg.paths.taxonomy_csv, cfg.data.taxonomy_levels
    )
    dataloaders, _ = build_dataloaders(cfg, taxonomy_df, encoders)
    loader = dataloaders[args.split]
    model = TaxonomyAwareClassifier.load_from_checkpoint(
        args.checkpoint,
        cfg=cfg,
        class_counts=class_counts,
    )
    evaluate_model(model, loader, cfg, id_to_name, prefix=args.split)


if __name__ == "__main__":
    main()
