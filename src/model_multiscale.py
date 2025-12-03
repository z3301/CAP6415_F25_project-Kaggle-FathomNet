"""
Multi-Scale Backbone Model for FathomNet 2025 Competition

This module implements a multi-scale hierarchical classifier with separate
encoders for each scale (ROI, 3×, 5×) to leverage environmental context.

Issue: #25
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm


class MultiScaleBackbone(nn.Module):
    """
    Multiple encoder branches for different spatial scales.

    Each scale gets its own ConvNeXtV2 encoder to learn specialized features:
    - ROI encoder: Fine-grained organism morphology
    - 3× encoder: Immediate environmental context
    - 5× encoder: Broader habitat/depth indicators

    Args:
        backbone_name: Name of timm model to use for each encoder
        num_scales: Number of scale encoders (default: 3)
        pretrained: Whether to load pretrained weights
    """

    def __init__(self, backbone_name='convnextv2_base.fcmae_ft_in22k_in1k',
                 num_scales=3, pretrained=True):
        super().__init__()

        self.num_scales = num_scales

        # Create separate backbone for each scale
        self.encoders = nn.ModuleList([
            timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
            for _ in range(num_scales)
        ])

        # Get feature dimension from first encoder
        self.feat_dim = self.encoders[0].num_features

    def forward(self, scale_images_dict):
        """
        Forward pass through all scale encoders.

        Args:
            scale_images_dict: Dict mapping scale names to image tensors
                              e.g., {'scale_1.0': tensor, 'scale_3.0': tensor, ...}

        Returns:
            List of feature tensors, one per scale
        """
        features = []

        # Sort scale keys to ensure consistent ordering
        scale_keys = sorted(scale_images_dict.keys())

        for i, scale_key in enumerate(scale_keys):
            if i >= self.num_scales:
                break
            feats = self.encoders[i](scale_images_dict[scale_key])
            features.append(feats)

        return features


class MultiScaleTaxonomyClassifier(pl.LightningModule):
    """
    Multi-scale hierarchical taxonomy classifier.

    Combines features from multiple spatial scales before hierarchical
    classification through 7 taxonomic ranks.

    Args:
        class_counts: Dict mapping taxonomic levels to number of classes
        lr: Learning rate
        num_scales: Number of scale encoders
        fusion: Feature fusion method ('concat' or 'attention')
        backbone_name: Backbone model name
        taxonomy_levels: List of taxonomic levels
    """

    def __init__(self, class_counts, lr=3e-4, num_scales=3,
                 fusion='concat', backbone_name='convnextv2_base.fcmae_ft_in22k_in1k',
                 taxonomy_levels=["kingdom", "phylum", "class", "order", "family", "genus", "species"]):
        super().__init__()
        self.save_hyperparameters()

        self.class_counts = class_counts
        self.lr = lr
        self.fusion = fusion
        self.taxonomy_levels = taxonomy_levels

        # Multi-scale backbone
        self.backbone = MultiScaleBackbone(
            backbone_name=backbone_name,
            num_scales=num_scales,
            pretrained=True
        )
        feat_dim = self.backbone.feat_dim

        # Projection layers for each scale to common dimension
        self.projections = nn.ModuleList([
            nn.Linear(feat_dim, 512) for _ in range(num_scales)
        ])

        # Fusion layer (concatenation for now, attention in future)
        if fusion == 'concat':
            fused_dim = 512 * num_scales
        else:
            raise NotImplementedError(f"Fusion method '{fusion}' not implemented")

        # Shared feature processing
        self.shared_features = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.3),  # Increased dropout for larger model
        )

        # Hierarchical classifier heads (sequential conditioning)
        self._create_hierarchical_network(class_counts)

        # Loss
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

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

    def forward(self, scale_images_dict):
        """
        Forward pass through multi-scale backbone and hierarchical classifier.

        Args:
            scale_images_dict: Dict of scale tensors

        Returns:
            Dict of logits for each taxonomic level
        """
        # Extract features from each scale
        scale_features = self.backbone(scale_images_dict)

        # Project each scale to common dimension
        projected = [
            proj(feats) for proj, feats in zip(self.projections, scale_features)
        ]

        # Fuse multi-scale features (concatenation)
        fused = torch.cat(projected, dim=1)

        # Shared processing
        shared = self.shared_features(fused)

        # Hierarchical prediction with sequential conditioning
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
        """Calculate weighted hierarchical loss across all taxonomic levels."""
        losses = {}
        total_loss = 0.0

        # Weight lower ranks more heavily
        weights = {
            'kingdom': 0.5,
            'phylum': 0.75,
            'class': 1.0,
            'order': 1.25,
            'family': 1.5,
            'genus': 2.0,
            'species': 2.5
        }

        for level in self.taxonomy_levels:
            if level in outputs and level in targets:
                level_loss = self.criterion(outputs[level], targets[level])
                losses[level] = level_loss
                total_loss += weights[level] * level_loss

        # Normalize by total weight
        total_weight = sum(weights[l] for l in losses.keys())
        return total_loss / total_weight, losses

    def training_step(self, batch, batch_idx):
        images_dict, labels = batch
        outputs = self(images_dict)
        loss, level_losses = self.hierarchical_loss(outputs, labels)

        self.log("train_loss", loss, prog_bar=True)
        for level, level_loss in level_losses.items():
            self.log(f"train_{level}_loss", level_loss, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        images_dict, labels = batch
        outputs = self(images_dict)
        loss, level_losses = self.hierarchical_loss(outputs, labels)

        # Calculate accuracy for each level
        accuracy = {}
        for level in self.taxonomy_levels:
            if level in outputs and level in labels:
                preds = torch.argmax(outputs[level], dim=1)
                acc = (preds == labels[level]).float().mean()
                accuracy[level] = acc

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        for level, level_loss in level_losses.items():
            self.log(f"val_{level}_loss", level_loss, prog_bar=False, sync_dist=True)

        for level, acc in accuracy.items():
            self.log(f"val_{level}_acc", acc, prog_bar=True, sync_dist=True)

        self.val_step_outputs.append({'outputs': outputs, 'targets': labels})
        return loss

    def on_validation_epoch_end(self):
        if not self.val_step_outputs:
            return

        all_preds = {level: [] for level in self.taxonomy_levels}
        all_targets = {level: [] for level in self.taxonomy_levels}

        for output in self.val_step_outputs:
            for level in self.taxonomy_levels:
                if level in output['outputs'] and level in output['targets']:
                    pred = torch.argmax(output['outputs'][level], dim=1).cpu()
                    target = output['targets'][level].cpu()
                    all_preds[level].append(pred)
                    all_targets[level].append(target)

        for level in self.taxonomy_levels:
            if all_preds[level]:
                all_preds[level] = torch.cat(all_preds[level])
                all_targets[level] = torch.cat(all_targets[level])
                acc = (all_preds[level] == all_targets[level]).float().mean()
                self.log(f"val_{level}_acc_epoch", acc, prog_bar=True)

        self.val_step_outputs.clear()

    def configure_optimizers(self):
        """Configure optimizer with different learning rates for backbone vs heads."""
        # Separate learning rates: lower for pretrained backbones
        backbone_params = list(self.backbone.parameters())
        other_params = [p for n, p in self.named_parameters() if 'backbone' not in n]

        param_groups = [
            {'params': backbone_params, 'lr': self.lr / 10},  # Lower LR for backbones
            {'params': other_params, 'lr': self.lr}
        ]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=0.05)

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=5, T_mult=2, eta_min=1e-6
            ),
            "interval": "epoch",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
