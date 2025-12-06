"""
DINOv2 Multi-Context Patch Attention Model for FathomNet 2025 Competition

This module implements the architecture from the "Attention Is All You Need" paper
combined with DINOv2 vision transformer:
- Shared DINOv2 encoder for all scales
- ROI (1×): CLS token as Query (global embedding)
- Context (3×, 5×, full): Patch tokens as Key/Value
- Cross-attention: Scaled dot-product attention (Q·K^T / √d_k)·V
- Hierarchical auxiliary classification (7 taxonomic levels)

Reference: https://arxiv.org/abs/1706.03762 (Attention Is All You Need)

Issue: #27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import timm


class ScaledDotProductCrossAttention(nn.Module):
    """
    Scaled Dot-Product Cross-Attention from "Attention Is All You Need".

    Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads (h in paper)
        dropout: Dropout rate
    """

    def __init__(self, embed_dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Linear projections for Q, K, V (W^Q, W^K, W^V in paper)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection (W^O in paper)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5  # 1/√d_k

    def forward(self, query, key, value):
        """
        Cross-attention where query attends to key-value pairs.

        Args:
            query: ROI global embeddings [B, embed_dim] or [B, 1, embed_dim]
            key: Context patch embeddings [B, N, embed_dim]
            value: Context patch embeddings [B, N, embed_dim]

        Returns:
            Attended features [B, embed_dim]
        """
        B = query.shape[0]

        # Ensure query has sequence dimension
        if query.dim() == 2:
            query = query.unsqueeze(1)  # [B, 1, embed_dim]

        # Project Q, K, V
        Q = self.q_proj(query)  # [B, 1, embed_dim]
        K = self.k_proj(key)    # [B, N, embed_dim]
        V = self.v_proj(value)  # [B, N, embed_dim]

        # Reshape for multi-head attention: [B, H, seq_len, head_dim]
        Q = Q.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, 1, d_k]
        K = K.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d_k]
        V = V.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, d_v]

        # Scaled dot-product attention: softmax(Q·K^T / √d_k) · V
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, 1, N]
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)  # [B, H, 1, d_v]

        # Concatenate heads and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, -1, self.embed_dim)  # [B, 1, embed_dim]
        attn_output = self.out_proj(attn_output)

        return attn_output.squeeze(1)  # [B, embed_dim]


class MultiContextPatchAttention(nn.Module):
    """
    Multi-Context Patch Attention Module.

    ROI CLS token (query) attends to context patch tokens (key/value)
    for each context scale, then concatenates all attended features.
    """

    def __init__(self, embed_dim, num_context_scales=3, num_heads=8, dropout=0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_context_scales = num_context_scales

        # Cross-attention for each context scale
        self.cross_attention_blocks = nn.ModuleList([
            ScaledDotProductCrossAttention(embed_dim, num_heads, dropout)
            for _ in range(num_context_scales)
        ])

        # Layer norm for each attended output
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim) for _ in range(num_context_scales)
        ])

        # Projection after concatenation
        concat_dim = embed_dim * (1 + num_context_scales)  # ROI + attended contexts
        self.projection = nn.Sequential(
            nn.LayerNorm(concat_dim),
            nn.Linear(concat_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, roi_cls, context_patches_list):
        """
        Args:
            roi_cls: ROI CLS token (global embedding) [B, embed_dim]
            context_patches_list: List of context patch embeddings [B, N, embed_dim]

        Returns:
            Fused features [B, embed_dim]
        """
        attended_features = [roi_cls]  # Start with ROI global embedding

        for i, (cross_attn, layer_norm, context_patches) in enumerate(
            zip(self.cross_attention_blocks, self.layer_norms, context_patches_list)
        ):
            # Cross-attention: ROI queries context patches
            attended = cross_attn(
                query=roi_cls,
                key=context_patches,
                value=context_patches
            )
            # Add residual and normalize
            attended = layer_norm(roi_cls + attended)
            attended_features.append(attended)

        # Concatenate ROI + all attended features
        concatenated = torch.cat(attended_features, dim=1)

        # Project to final embedding
        output = self.projection(concatenated)

        return output


class DINOv2PatchAttentionClassifier(pl.LightningModule):
    """
    DINOv2 Multi-Context Patch Attention Classifier.

    Architecture:
    1. Shared DINOv2 encoder (ViT) for all scales
    2. ROI (1×) -> CLS token (Query)
    3. Context (3×, 5×, full) -> Patch tokens (Key, Value)
    4. Cross-attention: Q attends to K,V using scaled dot-product
    5. Concatenate + Project
    6. Hierarchical auxiliary classification (7 taxonomic levels)

    Args:
        class_counts: Dict mapping taxonomic levels to number of classes
        lr: Learning rate
        num_context_scales: Number of context scales
        backbone_name: DINOv2 model name from timm
        num_heads: Number of attention heads
        dropout: Dropout rate
    """

    def __init__(self, class_counts, lr=3e-4, num_context_scales=3,
                 backbone_name='vit_base_patch14_dinov2.lvd142m',
                 num_heads=8, dropout=0.3,
                 taxonomy_levels=["kingdom", "phylum", "class", "order", "family", "genus", "species"]):
        super().__init__()
        self.save_hyperparameters()

        self.class_counts = class_counts
        self.lr = lr
        self.taxonomy_levels = taxonomy_levels
        self.num_context_scales = num_context_scales

        # Shared DINOv2 encoder with dynamic image size support
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True  # Allow variable input sizes
        )
        self.embed_dim = self.encoder.embed_dim

        print(f"DINOv2 encoder: {backbone_name}")
        print(f"  - Embed dim: {self.embed_dim}")
        print(f"  - Dynamic image size: enabled")

        # Multi-Context Patch Attention Module
        self.attention_module = MultiContextPatchAttention(
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

    def encode_with_patches(self, x):
        """
        Encode image and return both CLS token and patch tokens.

        Args:
            x: Input image [B, 3, H, W]

        Returns:
            cls_token: Global CLS embedding [B, embed_dim]
            patch_tokens: Patch embeddings [B, N, embed_dim]
        """
        features = self.encoder.forward_features(x)  # [B, 1+N, embed_dim]
        cls_token = features[:, 0]      # [B, embed_dim]
        patch_tokens = features[:, 1:]  # [B, N, embed_dim]
        return cls_token, patch_tokens

    def forward(self, scale_images_dict):
        """
        Forward pass through DINOv2 encoder, patch attention, and classifier.

        Args:
            scale_images_dict: Dict of scale tensors
                {'scale_1.0': [B,3,224,224], 'scale_3.0': ..., 'scale_5.0': ...}

        Returns:
            Dict of logits for each taxonomic level
        """
        # Sort scale keys for consistent ordering
        scale_keys = sorted(scale_images_dict.keys())

        # Encode ROI (first scale = 1.0) -> get CLS token as Query
        roi_images = scale_images_dict[scale_keys[0]]
        roi_cls, _ = self.encode_with_patches(roi_images)

        # Encode context scales -> get patch tokens as Key/Value
        context_patches_list = []
        for scale_key in scale_keys[1:]:
            context_images = scale_images_dict[scale_key]
            _, patch_tokens = self.encode_with_patches(context_images)
            context_patches_list.append(patch_tokens)

        # Apply multi-context patch attention
        fused_features = self.attention_module(roi_cls, context_patches_list)

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
