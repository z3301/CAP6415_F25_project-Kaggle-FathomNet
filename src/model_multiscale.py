"""
Multi-scale ConvNeXtV2 model for FathomNet 2025 Competition.

Uses separate encoders for each scale (1x, 3x, 5x) and fuses features
before hierarchical classification.
"""

from typing import Dict, List

import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
from omegaconf import DictConfig


class MultiScaleTaxonomyClassifier(pl.LightningModule):
    """
    Hierarchical taxonomy classifier with multi-scale ConvNeXtV2 backbones.

    Accepts input in format: {"1x": tensor, "3x": tensor, "5x": tensor}
    Each scale has its own encoder, features are concatenated before classification.
    """

    def __init__(self, cfg: DictConfig, class_counts: Dict[str, int], scales: List[str] = None):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.class_counts = class_counts
        self.levels = list(class_counts.keys())
        self.scales = scales or ["1x", "3x", "5x"]

        backbone_name = cfg.model.backbone

        # Create separate backbone for each scale
        self.backbones = nn.ModuleDict()
        for scale in self.scales:
            self.backbones[scale] = timm.create_model(
                backbone_name,
                pretrained=True,
                num_classes=0,
            )

        # Get feature dimension from backbone
        feature_dim = self.backbones[self.scales[0]].num_features
        in_features = feature_dim * len(self.scales)  # Concatenate all scale features

        hidden_dim = cfg.model.hidden_dim
        head_dim = cfg.model.head_dim

        # Fusion layer
        self.shared_features = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        self._build_heads(hidden_dim, head_dim)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)
        self.hierarchy_weights = dict(cfg.loss.hierarchy_weights)
        self.val_outputs = []

    def _build_heads(self, hidden_dim: int, head_dim: int):
        """Build hierarchical classification heads."""
        def block(out_dim: int):
            return nn.Linear(head_dim * 2, out_dim)

        self.kingdom_head = nn.Linear(hidden_dim, self.class_counts["kingdom"])
        self.kingdom_to_phylum = nn.Linear(self.class_counts["kingdom"], head_dim)
        self.phylum_features = nn.Linear(hidden_dim, head_dim)
        self.phylum_head = block(self.class_counts["phylum"])

        self.phylum_to_class = nn.Linear(self.class_counts["phylum"], head_dim)
        self.class_features = nn.Linear(hidden_dim, head_dim)
        self.class_head = block(self.class_counts["class"])

        self.class_to_order = nn.Linear(self.class_counts["class"], head_dim)
        self.order_features = nn.Linear(hidden_dim, head_dim)
        self.order_head = block(self.class_counts["order"])

        self.order_to_family = nn.Linear(self.class_counts["order"], head_dim)
        self.family_features = nn.Linear(hidden_dim, head_dim)
        self.family_head = block(self.class_counts["family"])

        self.family_to_genus = nn.Linear(self.class_counts["family"], head_dim)
        self.genus_features = nn.Linear(hidden_dim, head_dim)
        self.genus_head = block(self.class_counts["genus"])

        self.genus_to_species = nn.Linear(self.class_counts["genus"], head_dim)
        self.species_features = nn.Linear(hidden_dim, head_dim)
        self.species_head = block(self.class_counts["species"])

    def _extract_features(self, images: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Extract and concatenate features from all scales."""
        scale_features = []
        for scale in self.scales:
            if scale in images:
                feats = self.backbones[scale](images[scale])
                scale_features.append(feats)
        return torch.cat(scale_features, dim=1)

    def forward(self, images: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-scale backbones and hierarchical classifier.

        Args:
            images: Dict with keys like "1x", "3x", "5x" mapping to image tensors

        Returns:
            Dict of logits for each taxonomy level
        """
        # Extract and fuse features from all scales
        combined_features = self._extract_features(images)
        shared = self.shared_features(combined_features)

        # Hierarchical classification
        kingdom_logits = self.kingdom_head(shared)
        kingdom_probs = torch.softmax(kingdom_logits, dim=1)

        phylum_input = self.kingdom_to_phylum(kingdom_probs)
        phylum_feats = self.phylum_features(shared)
        phylum_logits = self.phylum_head(torch.cat([phylum_input, phylum_feats], dim=1))
        phylum_probs = torch.softmax(phylum_logits, dim=1)

        class_input = self.phylum_to_class(phylum_probs)
        class_feats = self.class_features(shared)
        class_logits = self.class_head(torch.cat([class_input, class_feats], dim=1))
        class_probs = torch.softmax(class_logits, dim=1)

        order_input = self.class_to_order(class_probs)
        order_feats = self.order_features(shared)
        order_logits = self.order_head(torch.cat([order_input, order_feats], dim=1))
        order_probs = torch.softmax(order_logits, dim=1)

        family_input = self.order_to_family(order_probs)
        family_feats = self.family_features(shared)
        family_logits = self.family_head(torch.cat([family_input, family_feats], dim=1))
        family_probs = torch.softmax(family_logits, dim=1)

        genus_input = self.family_to_genus(family_probs)
        genus_feats = self.genus_features(shared)
        genus_logits = self.genus_head(torch.cat([genus_input, genus_feats], dim=1))
        genus_probs = torch.softmax(genus_logits, dim=1)

        species_input = self.genus_to_species(genus_probs)
        species_feats = self.species_features(shared)
        species_logits = self.species_head(torch.cat([species_input, species_feats], dim=1))

        return {
            "kingdom": kingdom_logits,
            "phylum": phylum_logits,
            "class": class_logits,
            "order": order_logits,
            "family": family_logits,
            "genus": genus_logits,
            "species": species_logits,
        }

    def hierarchical_loss(self, outputs, targets):
        """Compute weighted loss across all taxonomy levels."""
        total_loss = 0.0
        level_losses = {}
        weight_sum = 0.0
        for level in self.levels:
            if level not in outputs or level not in targets:
                continue
            loss = self.criterion(outputs[level], targets[level])
            w = self.hierarchy_weights.get(level, 1.0)
            total_loss += w * loss
            weight_sum += w
            level_losses[level] = loss
        total_loss = total_loss / max(weight_sum, 1e-6)
        return total_loss, level_losses

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss, level_losses = self.hierarchical_loss(outputs, labels)
        self.log("train_loss", loss, prog_bar=True)
        for lvl, lvl_loss in level_losses.items():
            self.log(f"train_{lvl}_loss", lvl_loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss, level_losses = self.hierarchical_loss(outputs, labels)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        for lvl, lvl_loss in level_losses.items():
            self.log(f"val_{lvl}_loss", lvl_loss, sync_dist=True)
            preds = torch.argmax(outputs[lvl], dim=1)
            acc = (preds == labels[lvl]).float().mean()
            self.log(f"val_{lvl}_acc", acc, prog_bar=True, sync_dist=True)
        self.val_outputs.append({"outputs": outputs, "labels": labels})
        return loss

    def on_validation_epoch_end(self):
        self.val_outputs.clear()

    def predict_step(self, batch, batch_idx):
        images, annotation_ids = batch
        outputs = self(images)
        probs = {}
        preds = {}
        confidences = {}
        for level in self.levels:
            logits = outputs[level]
            level_probs = torch.softmax(logits, dim=1)
            level_preds = torch.argmax(level_probs, dim=1)
            probs[level] = level_probs.detach().cpu()
            preds[level] = level_preds.detach().cpu()
            confidences[level] = (
                torch.gather(level_probs, 1, level_preds.unsqueeze(1))
                .squeeze(1)
                .detach()
                .cpu()
            )
        return {
            "annotation_ids": annotation_ids,
            "probs": probs,
            "preds": preds,
            "confidences": confidences,
        }

    def configure_optimizers(self):
        backbone_params = []
        other_params = []
        for name, param in self.named_parameters():
            if "backbones" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.cfg.training.learning_rate * self.cfg.training.backbone_lr_scale},
                {"params": other_params, "lr": self.cfg.training.learning_rate},
            ],
            weight_decay=self.cfg.training.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
