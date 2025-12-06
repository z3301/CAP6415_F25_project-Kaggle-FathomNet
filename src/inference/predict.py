import argparse

import pytorch_lightning as pl
from omegaconf import OmegaConf

from src.data import load_and_encode_taxonomy, prepare_test_loader
from src.model import TaxonomyAwareClassifier
from src.submission import generate_submission


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Kaggle submission.")
    parser.add_argument("--config", type=str, default="config/experiment-01.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = OmegaConf.load(args.config)
    pl.seed_everything(cfg.project.seed)
    taxonomy_df, encoders, class_counts, id_to_name = load_and_encode_taxonomy(
        cfg.paths.taxonomy_csv, cfg.data.taxonomy_levels
    )
    test_loader = prepare_test_loader(cfg)
    model = TaxonomyAwareClassifier.load_from_checkpoint(
        args.checkpoint, cfg=cfg, class_counts=class_counts
    )
    submission_path = generate_submission(model, test_loader, cfg, id_to_name, taxonomy_df)
    print(f"Submission saved to {submission_path}")


if __name__ == "__main__":
    main()
