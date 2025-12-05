"""
Multi-Context Environmental Attention Model for FathomNet 2025 Competition

This module implements the architecture from the diagram:
- Shared ViT encoder for all scales
- ROI produces global embeddings (Query)
- Context regions produce patch embeddings (Key, Value)
- Cross-attention from ROI to each context scale
- Concatenated embeddings -> Projection -> Hierarchical classifier

Issue: #27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention module where ROI queries attend to context features.

    Q: ROI global embeddings
    K, V: Context patch embeddings
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, query, key_value):
        """
        Args:
            query: ROI embeddings [B, embed_dim]
            key_value: Context embeddings [B, N, embed_dim] (N = num patches or 1 for global)

        Returns:
            Attended features [B, embed_dim]
        """
        B = query.shape[0]

        # Add sequence dimension to query if needed
        if query.dim() == 2:
            query = query.unsqueeze(1)  # [B, 1, embed_dim]

        # Handle 2D key_value (global features)
        if key_value.dim() == 2:
            key_value = key_value.unsqueeze(1)  # [B, 1, embed_dim]

        # Project Q, K, V
        Q = self.q_proj(query)  # [B, 1, embed_dim]
        K = self.k_proj(key_value)  # [B, N, embed_dim]
        V = self.v_proj(key_value)  # [B, N, embed_dim]

        # Reshape for multi-head attention
        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, D]
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D]

        # Scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, 1, N]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, V)  # [B, H, 1, D]
        out = out.transpose(1, 2).contiguous().view(B, -1, self.embed_dim)  # [B, 1, embed_dim]
        out = self.out_proj(out)

        # Residual connection and layer norm
        out = self.layer_norm(query + out)

        return out.squeeze(1)  # [B, embed_dim]


class MultiContextAttentionModule(nn.Module):
    """
    Multi-Context Environmental Attention Module from the diagram.

    ROI embeddings attend to each context scale separately via cross-attention,
    then all attended features are concatenated.
    """

    def __init__(self, embed_dim, num_context_scales=3, num_heads=8, dropout=0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_context_scales = num_context_scales

        # Cross-attention block for each context scale
        self.cross_attention_blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_context_scales)
        ])

        # Projection network after concatenation
        concat_dim = embed_dim * (1 + num_context_scales)  # ROI + attended contexts
        self.projection = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, roi_features, context_features_list):
        """
        Args:
            roi_features: ROI global embeddings [B, embed_dim]
            context_features_list: List of context features, each [B, embed_dim]

        Returns:
            Fused features [B, embed_dim]
        """
        attended_features = [roi_features]  # Start with ROI features

        # Apply cross-attention from ROI to each context scale
        for i, (cross_attn, context_feats) in enumerate(
            zip(self.cross_attention_blocks, context_features_list)
        ):
            attended = cross_attn(roi_features, context_feats)
            attended_features.append(attended)

        # Concatenate all features
        concatenated = torch.cat(attended_features, dim=1)

        # Project to final embedding
        output = self.projection(concatenated)

        return output


class MultiContextAttentionClassifier(pl.LightningModule):
    """
    Multi-Context Environmental Attention Classifier with Hierarchical Classification.

    Architecture (from diagram):
    1. Shared ConvNeXt encoder for all scales
    2. ROI -> Global Embeddings (Query)
    3. Context scales -> Global Embeddings (Key, Value)
    4. Cross-attention: ROI queries each context scale
    5. Concatenate attended features + ROI
    6. Projection network
    7. Hierarchical auxiliary classification (7 taxonomic levels)

    Args:
        class_counts: Dict mapping taxonomic levels to number of classes
        lr: Learning rate
        num_context_scales: Number of context scales (3x, 5x, full)
        backbone_name: ConvNeXt backbone model name
        embed_dim: Embedding dimension (from backbone)
        num_heads: Number of attention heads
        taxonomy_levels: List of taxonomic levels
    """

    def __init__(self, class_counts, lr=3e-4, num_context_scales=3,
                 backbone_name='convnextv2_base.fcmae_ft_in22k_in1k',
                 embed_dim=1024, num_heads=8, dropout=0.3,
                 taxonomy_levels=["kingdom", "phylum", "class", "order", "family", "genus", "species"]):
        super().__init__()
        self.save_hyperparameters()

        self.class_counts = class_counts
        self.lr = lr
        self.taxonomy_levels = taxonomy_levels
        self.num_context_scales = num_context_scales

        # Shared ConvNeXt encoder for all scales
        self.encoder = timm.create_model(backbone_name, pretrained=True, num_classes=0)
        self.embed_dim = self.encoder.num_features

        # Multi-Context Attention Module
        self.attention_module = MultiContextAttentionModule(
            embed_dim=self.embed_dim,
            num_context_scales=num_context_scales,
            num_heads=num_heads,
            dropout=dropout
        )

        # Classification network
        self.classifier_input = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Hierarchical classifier heads
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
        Forward pass through shared encoder, attention module, and classifier.

        Args:
            scale_images_dict: Dict of scale tensors
                {'scale_1.0': [B,3,224,224], 'scale_3.0': ..., 'scale_5.0': ..., 'scale_full': ...}

        Returns:
            Dict of logits for each taxonomic level
        """
        # Sort scale keys for consistent ordering
        scale_keys = sorted(scale_images_dict.keys())

        # Encode all scales with shared encoder
        all_features = []
        for scale_key in scale_keys:
            features = self.encoder(scale_images_dict[scale_key])
            all_features.append(features)

        # ROI is first scale (1.0), contexts are remaining scales
        roi_features = all_features[0]
        context_features = all_features[1:]

        # Apply multi-context attention
        fused_features = self.attention_module(roi_features, context_features)

        # Classification
        shared = self.classifier_input(fused_features)

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

        # Weight lower ranks more heavily (species most important)
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

        total_weight = sum(weights[l] for l in losses.keys())
        return total_loss / total_weight, losses

    def training_step(self, batch, batch_idx):
        images_dict, labels = batch
        outputs = self(images_dict)
        loss, level_losses = self.hierarchical_loss(outputs, labels)

        self.log("train_loss", loss, prog_bar=True)
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
        """Configure optimizer with different learning rates for encoder vs heads."""
        encoder_params = list(self.encoder.parameters())
        other_params = [p for n, p in self.named_parameters() if 'encoder' not in n]

        param_groups = [
            {'params': encoder_params, 'lr': self.lr / 10},  # Lower LR for pretrained encoder
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
