"""
Simple independent-head model for debugging

No hierarchical conditioning - just independent classifiers for each level.
This will help isolate if the conditioning is causing the collapse.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm

from src.config import Config


class SimpleTaxonomyClassifier(pl.LightningModule):
    """
    Simple taxonomy classifier with INDEPENDENT heads (no conditioning).

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
        self.taxonomy_levels = Config.TAXONOMY_LEVELS

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

        # Create INDEPENDENT heads for each level
        self.heads = nn.ModuleDict({
            level: nn.Linear(1024, class_counts[level])
            for level in self.taxonomy_levels
        })

        # Loss
        self.criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)

        # Tracking
        self.val_step_outputs = []

    def forward(self, x):
        """Forward pass - just extract features and classify independently."""
        features = self.feature_extractor(x)
        shared = self.shared_features(features)

        # Independent predictions for each level
        outputs = {}
        for level in self.taxonomy_levels:
            outputs[level] = self.heads[level](shared)

        return outputs

    def hierarchical_loss(self, outputs, targets):
        """Calculate loss with equal weighting for all levels."""
        losses = {}
        total_loss = 0.0

        # Equal weight for all levels
        for level in self.taxonomy_levels:
            if level in targets and level in outputs:
                losses[level] = self.criterion(outputs[level], targets[level])
                total_loss += losses[level]

        return total_loss, losses

    def training_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        total_loss, losses = self.hierarchical_loss(outputs, targets)

        # Log losses
        self.log('train_loss', total_loss, prog_bar=True)
        for level, loss in losses.items():
            self.log(f'train_{level}_loss', loss)

        return total_loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self(images)
        total_loss, losses = self.hierarchical_loss(outputs, targets)

        # Calculate accuracies
        accuracies = {}
        for level in self.taxonomy_levels:
            if level in targets and level in outputs:
                preds = torch.argmax(outputs[level], dim=1)
                acc = (preds == targets[level]).float().mean()
                accuracies[level] = acc

        # Store for epoch end
        self.val_step_outputs.append({
            'val_loss': total_loss,
            'accuracies': accuracies
        })

        # Log
        self.log('val_loss', total_loss, prog_bar=True)
        for level, acc in accuracies.items():
            self.log(f'val_{level}_acc', acc, prog_bar=True)

        return total_loss

    def on_validation_epoch_end(self):
        if not self.val_step_outputs:
            return

        # Average accuracies
        avg_accs = {}
        for level in self.taxonomy_levels:
            accs = [x['accuracies'].get(level, 0) for x in self.val_step_outputs if level in x['accuracies']]
            if accs:
                avg_accs[level] = sum(accs) / len(accs)
                self.log(f'val_{level}_acc_epoch', avg_accs[level])

        self.val_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=Config.MAX_EPOCHS
        )
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
