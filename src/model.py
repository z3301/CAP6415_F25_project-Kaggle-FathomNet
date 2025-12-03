"""
Single-scale model for FathomNet 2025 Competition

Hierarchical taxonomy classifier with single ConvNeXtV2 backbone.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm

from src.config import Config


class TaxonomyAwareClassifier(pl.LightningModule):
    """
    Hierarchical taxonomy classifier with sequential conditioning.

    Args:
        class_counts: Dict mapping taxonomic levels to number of classes
        lr: Learning rate
        backbone_name: Name of timm backbone model
    """

    def __init__(self, class_counts, lr=Config.LEARNING_RATE,
                 backbone_name=Config.BACKBONE):
        super().__init__()
        self.save_hyperparameters()
        self.class_counts = class_counts
        self.lr = lr

        # Load ConvNeXt v2 from timm
        backbone = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0  # Remove classification head
        )
        self.feature_extractor = backbone
        in_features = backbone.num_features

        # Shared feature processing
        self.shared_features = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Create 7-level hierarchical classifier
        self._create_hierarchical_network(class_counts)

        # Loss
        self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

        # Tracking
        self.val_step_outputs = []

    def _create_hierarchical_network(self, class_counts):
        """Create sequential hierarchical classification heads."""
        # Kingdom head (top level)
        self.kingdom_head = nn.Linear(1024, class_counts['kingdom'])
        self.kingdom_to_phylum = nn.Linear(class_counts['kingdom'], 512)
        self.phylum_features = nn.Linear(1024, 512)
        self.phylum_head = nn.Linear(1024, class_counts['phylum'])

        # Phylum to Class
        self.phylum_to_class = nn.Linear(class_counts['phylum'], 512)
        self.class_features = nn.Linear(1024, 512)
        self.class_head = nn.Linear(1024, class_counts['class'])

        # Class to Order
        self.class_to_order = nn.Linear(class_counts['class'], 512)
        self.order_features = nn.Linear(1024, 512)
        self.order_head = nn.Linear(1024, class_counts['order'])

        # Order to Family
        self.order_to_family = nn.Linear(class_counts['order'], 512)
        self.family_features = nn.Linear(1024, 512)
        self.family_head = nn.Linear(1024, class_counts['family'])

        # Family to Genus
        self.family_to_genus = nn.Linear(class_counts['family'], 512)
        self.genus_features = nn.Linear(1024, 512)
        self.genus_head = nn.Linear(1024, class_counts['genus'])

        # Genus to Species
        self.genus_to_species = nn.Linear(class_counts['genus'], 512)
        self.species_features = nn.Linear(1024, 512)
        self.species_head = nn.Linear(1024, class_counts['species'])

    def forward(self, x):
        """Forward pass through backbone and hierarchical classifier."""
        features = self.feature_extractor(x)
        shared = self.shared_features(features)

        # Predict hierarchy with sequential conditioning
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
            'kingdom': kingdom_logits,
            'phylum': phylum_logits,
            'class': class_logits,
            'order': order_logits,
            'family': family_logits,
            'genus': genus_logits,
            'species': species_logits,
        }

    def hierarchical_loss(self, outputs, targets):
        """Calculate hierarchical loss with weighting for different levels."""
        losses = {}
        total_loss = 0.0
        weights = {
            'kingdom': 0.5,
            'phylum': 0.75,
            'class': 1.0,
            'order': 1.25,
            'family': 1.5,
            'genus': 2.0,
            'species': 2.5
        }

        # Calculate losses for each available level
        for level in Config.TAXONOMY_LEVELS:
            if level in outputs and level in targets:
                level_loss = self.criterion(outputs[level], targets[level])
                losses[level] = level_loss
                total_loss += weights[level] * level_loss

        # Normalize by total weight
        total_weight = sum(weights[l] for l in losses.keys())
        return total_loss / total_weight, losses

    def training_step(self, batch, batch_idx):
        """Training step."""
        x, y = batch
        outputs = self(x)
        loss, level_losses = self.hierarchical_loss(outputs, y)

        self.log("train_loss", loss, prog_bar=True)
        for level, level_loss in level_losses.items():
            self.log(f"train_{level}_loss", level_loss, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        x, y = batch
        outputs = self(x)
        loss, level_losses = self.hierarchical_loss(outputs, y)

        # Calculate accuracy for each level
        accuracy = {}
        for level in Config.TAXONOMY_LEVELS:
            if level in outputs and level in y:
                preds = torch.argmax(outputs[level], dim=1)
                acc = (preds == y[level]).float().mean()
                accuracy[level] = acc

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        for level, level_loss in level_losses.items():
            self.log(f"val_{level}_loss", level_loss, prog_bar=False, sync_dist=True)

        for level, acc in accuracy.items():
            self.log(f"val_{level}_acc", acc, prog_bar=True, sync_dist=True)

        self.val_step_outputs.append({'outputs': outputs, 'targets': y})
        return loss

    def on_validation_epoch_end(self):
        """Validation epoch end processing."""
        if not self.val_step_outputs:
            return

        all_preds = {level: [] for level in Config.TAXONOMY_LEVELS}
        all_targets = {level: [] for level in Config.TAXONOMY_LEVELS}

        for output in self.val_step_outputs:
            for level in Config.TAXONOMY_LEVELS:
                if level in output['outputs'] and level in output['targets']:
                    pred = torch.argmax(output['outputs'][level], dim=1).cpu()
                    target = output['targets'][level].cpu()
                    all_preds[level].append(pred)
                    all_targets[level].append(target)

        for level in Config.TAXONOMY_LEVELS:
            if all_preds[level]:
                all_preds[level] = torch.cat(all_preds[level])
                all_targets[level] = torch.cat(all_targets[level])
                acc = (all_preds[level] == all_targets[level]).float().mean()
                self.log(f"val_{level}_acc_epoch", acc, prog_bar=True)

        self.val_step_outputs.clear()

    def predict_step(self, batch, batch_idx):
        """Generate predictions for test data."""
        if isinstance(batch, tuple) and len(batch) == 2:
            images, annotation_ids = batch
        else:
            images = batch
            annotation_ids = None

        outputs = self(images)

        # Get probabilities and predictions for each level
        results = {}
        for level in Config.TAXONOMY_LEVELS:
            if level in outputs:
                probs = torch.softmax(outputs[level], dim=1)
                preds = torch.argmax(probs, dim=1)
                conf = torch.gather(probs, 1, preds.unsqueeze(1)).squeeze(1)

                results[level] = {
                    'pred': preds.cpu(),
                    'conf': conf.cpu(),
                    'probs': probs.cpu()
                }

        if annotation_ids is not None:
            results['annotation_ids'] = annotation_ids

        return results

    def configure_optimizers(self):
        """Configure optimizers with different learning rates for backbone vs heads."""
        # Group parameters by part of the model
        backbone_params = self.feature_extractor.parameters()
        classifier_params = [p for n, p in self.named_parameters()
                            if "feature_extractor" not in n]

        # Create parameter groups with different learning rates
        param_groups = [
            {'params': backbone_params, 'lr': self.lr / 10},  # Lower LR for backbone
            {'params': classifier_params, 'lr': self.lr}
        ]

        # Create optimizer
        optimizer = torch.optim.AdamW(param_groups, weight_decay=Config.WEIGHT_DECAY)

        # Create learning rate scheduler
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=5, T_mult=2, eta_min=1e-6
            ),
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
