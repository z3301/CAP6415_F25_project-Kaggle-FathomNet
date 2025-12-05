"""
DINOv2-based model for FathomNet 2025 Competition

Hierarchical taxonomy classifier with DINOv2 backbone (for loading winning team checkpoint).
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl

from src.config import Config


class DINOv2TaxonomyClassifier(pl.LightningModule):
    """
    Hierarchical taxonomy classifier with DINOv2 backbone.

    Args:
        class_counts: Dict mapping taxonomic levels to number of classes
        lr: Learning rate
        model_name: DINOv2 model variant (e.g., 'dinov2_vitl14')
    """

    def __init__(self, class_counts, lr=Config.LEARNING_RATE,
                 model_name='dinov2_vitl14'):
        super().__init__()
        self.save_hyperparameters()
        self.class_counts = class_counts
        self.lr = lr

        # Load DINOv2 from torch hub
        self.feature_extractor = torch.hub.load(
            'facebookresearch/dinov2', model_name, pretrained=True
        )
        in_features = self.feature_extractor.embed_dim  # 1024 for ViT-L

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
        """Forward pass through DINOv2 backbone and hierarchical classifier."""
        # DINOv2 returns CLS token features
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

    def configure_optimizers(self):
        """Configure optimizers."""
        backbone_params = self.feature_extractor.parameters()
        classifier_params = [p for n, p in self.named_parameters()
                            if "feature_extractor" not in n]

        param_groups = [
            {'params': backbone_params, 'lr': self.lr / 10},
            {'params': classifier_params, 'lr': self.lr}
        ]

        optimizer = torch.optim.AdamW(param_groups, weight_decay=Config.WEIGHT_DECAY)
        return optimizer
