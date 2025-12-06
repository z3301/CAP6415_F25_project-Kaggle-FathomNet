"""
Multi-scale Cross-Attention Model for FathomNet 2025 Marine Species Classification.

================================================================================
ARCHITECTURAL OVERVIEW
================================================================================

This module implements a sophisticated cross-attention mechanism for multi-scale
feature fusion, inspired by the winning team's approach from FathomNet 2023.
Instead of simple feature concatenation, this architecture allows the Region of
Interest (ROI) to selectively attend to relevant contextual information.

Key insight: Marine species identification often requires understanding BOTH the
organism's fine-grained features (from the ROI) AND its environmental context
(habitat, depth, surrounding organisms). Cross-attention enables dynamic
weighting of context based on what's in the ROI.

================================================================================
CROSS-ATTENTION MECHANISM
================================================================================

Standard Feature Fusion (Baseline Model):
    features = concat([ROI_global, Context3x_global, Context5x_global])

Cross-Attention Fusion (This Model):
    Query (Q):  ROI global features       → "What am I looking for?"
    Key (K):    Context patch features    → "What's available in the context?"
    Value (V):  Context patch features    → "What information to retrieve?"

    Attention(Q, K, V) = softmax(QK^T / √d) × V

Mathematical Formulation:
--------------------------
For ROI query q ∈ ℝ^D and context patches {k_1, ..., k_N} ∈ ℝ^{N×D}:

    α_i = softmax_i(q · k_i / √D)           # Attention weights
    output = Σ_i α_i × v_i                   # Weighted sum of values

This computes a data-dependent weighted average of context patches,
where patches more relevant to the ROI receive higher weights.

================================================================================
ARCHITECTURE DIAGRAM
================================================================================

Input Images (3 scales):
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│  1x (ROI)   │   │  3x Context │   │  5x Context │
│  224×224    │   │  224×224    │   │  224×224    │
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │                 │                 │
       ▼                 ▼                 ▼
┌─────────────┐   ┌─────────────┐   ┌─────────────┐
│ ROI Encoder │   │   Context   │   │   Context   │
│ (ConvNeXtV2)│   │   Encoder   │   │   Encoder   │
│   88M params│   │   88M params│   │   88M params│
└──────┬──────┘   └──────┬──────┘   └──────┬──────┘
       │                 │                 │
       ▼                 │                 │
   ┌───────┐            ▼                 ▼
   │Global │      ┌──────────────────────────┐
   │(1, D) │      │  Concatenate Patches     │
   │ Query │      │  (N_3x + N_5x, D)        │
   └───┬───┘      │  Keys & Values           │
       │          └────────────┬─────────────┘
       │                       │
       ▼                       ▼
    ┌─────────────────────────────────────┐
    │         Cross-Attention             │
    │   Q: ROI global features (1, D)     │
    │   K: Context patches (N_total, D)   │
    │   V: Context patches (N_total, D)   │
    │                                     │
    │   output = softmax(QK^T/√D) × V     │
    └──────────────────┬──────────────────┘
                       │
                       ▼
              ┌────────────────┐
              │   Attended     │
              │   Features     │
              │     (1, D)     │
              └───────┬────────┘
                      │
                      ▼
              ┌────────────────┐
              │   Feature      │
              │   Projection   │
              │   (D → 2048)   │
              └───────┬────────┘
                      │
                      ▼
    ┌─────────────────────────────────────┐
    │     Hierarchical Classification     │
    │   Kingdom → Phylum → ... → Species  │
    │                                     │
    │   Each level conditions on parent   │
    │   predictions for coherent output   │
    └─────────────────────────────────────┘

================================================================================
ASYMMETRIC BACKBONE SUPPORT
================================================================================

This model supports using different backbone sizes for ROI vs context:

    ROI Backbone:     ConvNeXtV2-Large (198M params, 1536D features)
    Context Backbone: ConvNeXtV2-Base  (88M params, 1024D features)

Rationale: Fine-grained species details are primarily in the ROI, so allocating
more capacity there may improve accuracy. Context provides coarse environmental
cues that don't require as much capacity.

When asymmetric, context features are projected to match ROI dimension:
    context_proj = Linear(context_dim → roi_dim)

================================================================================
PATCH FEATURE EXTRACTION
================================================================================

Unlike global average pooling which loses spatial information, this model
extracts per-patch features from the backbone:

ConvNeXtV2 Output: (B, D, H, W) where H=W=7 for 224×224 input

Patch Features:    Flatten spatial dims → (B, H×W, D) = (B, 49, D)
                   Each of 49 patches represents a 32×32 region

For 3x context (672×672 original area), each patch covers ~96×96 pixels
For 5x context (1120×1120 original area), each patch covers ~160×160 pixels

This spatial granularity allows attention to focus on specific regions of
the context (e.g., substrate type, nearby organisms) rather than averaging
everything together.

================================================================================
MULTI-LAYER CROSS-ATTENTION
================================================================================

We stack multiple cross-attention layers for progressive refinement:

    Layer 1: Initial context retrieval based on raw ROI features
    Layer 2: Refined retrieval based on layer 1 output
    ...
    Layer N: Final refined representation

Each layer includes:
    1. Pre-norm on query and keys
    2. Multi-head cross-attention
    3. Residual connection
    4. FFN (MLP) with residual

This follows the Transformer architecture pattern but with cross-attention
instead of self-attention.

================================================================================
TRAINING CONFIGURATION
================================================================================

Learning Rate Strategy:
    - Backbone encoders:  base_lr × 0.1 (slower, pretrained weights)
    - Cross-attention:    base_lr × 0.5 (moderate, new layers)
    - Classification:     base_lr       (fastest, new layers)

This graduated learning rate helps:
    1. Preserve pretrained knowledge in backbones
    2. Give attention layers time to learn alignment
    3. Allow heads to adapt quickly to task

================================================================================
EXPERIMENTAL RESULTS
================================================================================

Experiment: multiscale_attention (3 scales: 1x, 3x, 5x)
    - 4-layer cross-attention, 8 heads
    - ConvNeXtV2-Base for all encoders (symmetric)
    - Batch size 16, 50 epochs

    Training progress monitored via TensorBoard
    Validation loss and per-level accuracy tracked

================================================================================
VARIANT: ROI PATCH ATTENTION
================================================================================

MultiScaleCrossAttentionWithROIPatches extends the base model to use ROI
patch features as queries (instead of just global ROI):

    Q: ROI patches (B, N_roi, D)     # Each ROI patch queries context
    K: Context patches (B, N_ctx, D)
    V: Context patches (B, N_ctx, D)

    attended_patches = CrossAttention(Q, K, V)  # (B, N_roi, D)
    final = mean(attended_patches) + roi_global  # Residual connection

This captures more fine-grained correspondences between ROI regions and
context regions, potentially useful for species with distinctive patterns
in specific body parts.

================================================================================
USAGE
================================================================================

Basic usage:
    model = MultiScaleCrossAttentionClassifier(
        cfg=config,
        class_counts={"kingdom": 2, "phylum": 8, ..., "species": 80},
        scales=["1x", "3x", "5x"],
        num_attention_layers=4,
        num_attention_heads=8,
    )

Asymmetric backbones:
    model = MultiScaleCrossAttentionClassifier(
        cfg=config,
        class_counts=class_counts,
        roi_backbone="convnextv2_large.fcmae_ft_in22k_in1k_384",
        context_backbone="convnextv2_base.fcmae_ft_in22k_in1k",
    )

References:
    - Vaswani et al., "Attention Is All You Need" (2017)
    - FathomNet 2023 winning solution (team approach)
    - ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders
"""

from typing import Dict, List, Optional, Tuple

import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


class CrossAttentionBlock(nn.Module):
    """
    Cross-attention block where ROI queries attend to context patches.

    This implements a single layer of cross-attention following the standard
    Transformer architecture, but with separate query and key/value inputs:

        - Query (Q):  ROI global features - the "question"
        - Key (K):    Context patches - what to match against
        - Value (V):  Context patches - what to retrieve

    Mathematical formulation:
    -------------------------
    Given query q ∈ ℝ^{B×1×D} and context c ∈ ℝ^{B×N×D}:

        Q = W_q × LayerNorm(q)           # Project query
        K = W_k × LayerNorm(c)           # Project keys
        V = W_v × c                      # Project values (no norm)

        # Multi-head attention (h heads, d = D/h)
        Q_h, K_h, V_h ∈ ℝ^{B×h×1×d}, ℝ^{B×h×N×d}

        # Scaled dot-product attention per head
        A = softmax(Q_h × K_h^T / √d)    # (B, h, 1, N)
        head_out = A × V_h               # (B, h, 1, d)

        # Concatenate heads and project
        out = W_out × concat(head_1, ..., head_h)

        # Residual connections
        q = q + out                      # Attention residual
        q = q + MLP(LayerNorm(q))        # FFN residual

    Attention Visualization:
    ------------------------
    For a single sample, the attention weights A ∈ ℝ^{1×N} show which
    context patches the ROI is "looking at". High weights indicate patches
    deemed relevant for classifying this particular organism.

    Args:
        embed_dim: Feature dimension (D)
        num_heads: Number of attention heads (must divide embed_dim)
        dropout: Dropout probability for attention and MLP
        mlp_ratio: MLP hidden dimension = embed_dim × mlp_ratio

    Example:
        >>> block = CrossAttentionBlock(embed_dim=1024, num_heads=8)
        >>> query = torch.randn(4, 1, 1024)    # ROI features
        >>> context = torch.randn(4, 98, 1024) # Context patches from 2 scales
        >>> output = block(query, context)     # (4, 1, 1024)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        mlp_ratio: float = 4.0,
    ):
        super().__init__()
        # Store configuration
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # Dimension per head

        # Scaling factor for attention: 1/√(head_dim)
        # Prevents dot products from growing too large, which would cause
        # softmax to produce near-one-hot distributions
        self.scale = self.head_dim ** -0.5

        # Linear projections for Q, K, V
        # Each projects from embed_dim to embed_dim (split across heads internally)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        # Output projection after concatenating attention heads
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        # Pre-normalization layers (Pre-LN Transformer style)
        # Pre-LN is more stable for training than Post-LN
        self.norm1 = nn.LayerNorm(embed_dim)  # Query norm
        self.norm2 = nn.LayerNorm(embed_dim)  # FFN norm
        self.norm_k = nn.LayerNorm(embed_dim) # Key norm (separate from query)

        # Feed-forward network (MLP) with GELU activation
        # Standard Transformer FFN: Linear → GELU → Dropout → Linear → Dropout
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

        # Dropout applied to attention weights
        self.attn_dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,      # (B, 1, D) - ROI features
        context: torch.Tensor,    # (B, N, D) - Context patches
    ) -> torch.Tensor:
        """
        Apply cross-attention: query attends to context.

        The ROI global features (query) look up relevant information from
        the context patches by computing attention weights based on
        query-key similarity, then retrieving a weighted sum of values.

        Args:
            query: ROI global features of shape (B, 1, D)
                   Single vector per sample representing the organism
            context: Context patch features of shape (B, N, D)
                     N patches from concatenated 3x and 5x context scales

        Returns:
            Updated query features of shape (B, 1, D)
            Enriched with relevant context information
        """
        B, Nq, D = query.shape  # Nq = 1 for global query
        _, Nk, _ = context.shape  # Nk = total context patches

        # Pre-normalization (stabilizes training)
        q = self.norm1(query)
        k = self.norm_k(context)
        v = context  # Values use unnormalized context (empirically better)

        # Project to Q, K, V and reshape for multi-head attention
        # (B, N, D) -> (B, N, num_heads, head_dim) -> (B, num_heads, N, head_dim)
        q = self.q_proj(q).view(B, Nq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(k).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(v).view(B, Nk, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # attn[b, h, i, j] = how much query position i attends to key position j
        # Shape: (B, num_heads, 1, N)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)  # Normalize over key positions
        attn = self.attn_dropout(attn)

        # Apply attention to values and reshape back
        # (B, num_heads, 1, head_dim) -> (B, 1, num_heads, head_dim) -> (B, 1, D)
        out = (attn @ v).transpose(1, 2).reshape(B, Nq, D)
        out = self.out_proj(out)

        # Residual connection for attention output
        query = query + out

        # Feed-forward network with residual connection
        query = query + self.mlp(self.norm2(query))

        return query


class MultiLayerCrossAttention(nn.Module):
    """
    Stack of cross-attention blocks for progressive feature refinement.

    Multiple layers allow the model to:
    1. Iteratively refine which context regions are relevant
    2. Build up complex feature representations
    3. Capture multi-hop reasoning (e.g., "this organism is near coral,
       and coral indicates shallow reef, which suggests species X")

    Architecture:
        Input Query ─┬─> CrossAttention Block 1 ─┬─> Block 2 ─┬─> ... ─┬─> Block N ─┬─> FinalNorm ─> Output
                     │                           │            │        │            │
                     │      Context ─────────────┴────────────┴────────┴────────────┘

    Each block applies:
        query = query + Attention(query, context)
        query = query + MLP(query)

    The same context patches are provided to all layers, but the query
    is progressively refined at each layer.

    Args:
        embed_dim: Feature dimension
        num_layers: Number of cross-attention blocks to stack
        num_heads: Number of attention heads per block
        dropout: Dropout probability

    Example:
        >>> attn = MultiLayerCrossAttention(embed_dim=1024, num_layers=4)
        >>> query = torch.randn(4, 1, 1024)
        >>> context = torch.randn(4, 98, 1024)
        >>> output = attn(query, context)  # (4, 1, 1024)
    """

    def __init__(
        self,
        embed_dim: int,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Create stack of identical cross-attention blocks
        self.layers = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        # Final normalization before output
        self.final_norm = nn.LayerNorm(embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply multiple layers of cross-attention.

        Args:
            query: Initial query features (B, 1, D)
            context: Context patches to attend to (B, N, D)

        Returns:
            Refined query features (B, 1, D)
        """
        # Iteratively refine query through all layers
        for layer in self.layers:
            query = layer(query, context)

        # Final normalization for stable output distribution
        return self.final_norm(query)


class PatchFeatureExtractor(nn.Module):
    """
    Wrapper around a backbone to extract patch-level features.

    Instead of just global average pooling, this module extracts features
    from every spatial position in the backbone's output, providing a
    sequence of patch embeddings that can be used in attention.

    Backbone Type Support:
    ----------------------
    ConvNeXtV2 (and most CNNs):
        Output shape: (B, D, H, W) where H=W=7 for 224×224 input
        Global: mean over (H, W) → (B, D)
        Patches: flatten (H, W) → (B, H×W, D) = (B, 49, D)

    Vision Transformer (ViT):
        Output shape: (B, N+1, D) where N = number of patches
        Global: CLS token at position 0 → (B, D)
        Patches: positions 1..N → (B, N, D)

    Spatial Resolution:
    -------------------
    For 224×224 input with stride 32 (typical for ConvNeXtV2):
        - 7×7 = 49 patches
        - Each patch represents a 32×32 pixel region
        - Receptive field is larger due to convolutions

    This spatial information is crucial for attention - it allows the
    model to focus on specific regions of the context (e.g., the substrate
    below the organism, or nearby organisms).

    Args:
        backbone_name: Name of the timm backbone model
        pretrained: Whether to load pretrained weights

    Example:
        >>> extractor = PatchFeatureExtractor("convnextv2_base.fcmae_ft_in22k_in1k")
        >>> images = torch.randn(4, 3, 224, 224)
        >>> global_feat, patch_feats = extractor(images)
        >>> print(global_feat.shape)   # (4, 1024)
        >>> print(patch_feats.shape)   # (4, 49, 1024)
    """

    def __init__(self, backbone_name: str, pretrained: bool = True):
        super().__init__()
        # Create backbone without classification head
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,       # Remove classifier
            features_only=False, # We want the full feature map, not multi-scale
        )
        # Store feature dimension for downstream layers
        self.feature_dim = self.backbone.num_features
        self.backbone_name = backbone_name

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract both global and patch features from input image.

        Args:
            x: Input image tensor of shape (B, 3, H, W)

        Returns:
            global_feat: Global pooled features of shape (B, D)
                        Represents the entire image as a single vector
            patch_feats: Per-patch features of shape (B, N, D)
                        N patches, each with a D-dimensional feature vector
        """
        # Get spatial features before final pooling
        # forward_features returns the backbone output before the classifier
        spatial = self.backbone.forward_features(x)

        if len(spatial.shape) == 4:
            # CNN-style backbone: (B, D, H, W)
            # Examples: ConvNeXt, ResNet, EfficientNet
            B, D, H, W = spatial.shape

            # Global average pooling for global features
            global_feat = spatial.mean(dim=(2, 3))  # (B, D)

            # Flatten spatial dimensions for patch features
            # (B, D, H, W) -> (B, D, H*W) -> (B, H*W, D)
            patch_feats = spatial.flatten(2).transpose(1, 2)

        else:
            # Transformer-style backbone: (B, N+1, D)
            # Examples: ViT, DeiT, Swin Transformer
            # Position 0 is CLS token, positions 1..N are patch tokens
            global_feat = spatial[:, 0]   # CLS token
            patch_feats = spatial[:, 1:]  # Patch tokens

        return global_feat, patch_feats


class MultiScaleCrossAttentionClassifier(pl.LightningModule):
    """
    Multi-scale classifier with cross-attention fusion.

    This is the main model class that combines:
    1. Separate backbone encoders for each scale
    2. Patch-level feature extraction from context
    3. Cross-attention for dynamic context fusion
    4. Hierarchical classification heads

    The key innovation over simple concatenation is that the ROI can
    selectively attend to relevant context regions rather than treating
    all context equally.

    Architecture Summary:
    ---------------------
    1. ROI Encoder: Extracts global features from 1x (tight crop)
    2. Context Encoders: Extract patch features from 3x and 5x context
    3. Cross-Attention: ROI queries context patches for relevant info
    4. Feature Projection: Maps attended features to classifier space
    5. Hierarchical Heads: Predict all 7 taxonomy levels

    Asymmetric Backbone Support:
    ----------------------------
    Different backbones can be used for ROI vs context:
        - roi_backbone: Typically larger (e.g., ConvNeXtV2-Large)
        - context_backbone: Typically smaller (e.g., ConvNeXtV2-Base)

    Rationale: ROI contains fine-grained species features requiring more
    capacity, while context provides coarser environmental cues.

    When asymmetric, a projection layer aligns context features to ROI
    dimension for attention compatibility.

    Args:
        cfg: OmegaConf configuration object with model/training settings
        class_counts: Dict mapping taxonomy level names to class counts
        scales: List of scale names (default: ["1x", "3x", "5x"])
        num_attention_layers: Number of cross-attention layers (default: 4)
        num_attention_heads: Number of attention heads (default: 8)
        roi_backbone: Optional backbone name for ROI (overrides cfg)
        context_backbone: Optional backbone name for context (overrides cfg)

    Input Format:
        Dictionary with scale keys mapping to image tensors:
        {
            "1x": torch.Tensor of shape (B, 3, 224, 224),  # ROI
            "3x": torch.Tensor of shape (B, 3, 224, 224),  # 3x context
            "5x": torch.Tensor of shape (B, 3, 224, 224),  # 5x context
        }

    Output Format:
        Dictionary with logits for each taxonomy level:
        {
            "kingdom": torch.Tensor of shape (B, 2),
            "phylum": torch.Tensor of shape (B, 8),
            ...
            "species": torch.Tensor of shape (B, 80),
        }

    Example:
        >>> model = MultiScaleCrossAttentionClassifier(
        ...     cfg=config,
        ...     class_counts={"kingdom": 2, ..., "species": 80},
        ...     scales=["1x", "3x", "5x"],
        ...     num_attention_layers=4,
        ... )
        >>> images = {
        ...     "1x": torch.randn(4, 3, 224, 224),
        ...     "3x": torch.randn(4, 3, 224, 224),
        ...     "5x": torch.randn(4, 3, 224, 224),
        ... }
        >>> outputs = model(images)
        >>> print(outputs["species"].shape)  # (4, 80)
    """

    def __init__(
        self,
        cfg: DictConfig,
        class_counts: Dict[str, int],
        scales: List[str] = None,
        num_attention_layers: int = 4,
        num_attention_heads: int = 8,
        roi_backbone: Optional[str] = None,
        context_backbone: Optional[str] = None,
    ):
        super().__init__()
        # Save hyperparameters for checkpoint loading (exclude non-serializable cfg)
        self.save_hyperparameters(ignore=["cfg"])
        self.cfg = cfg
        self.class_counts = class_counts
        self.levels = list(class_counts.keys())  # ["kingdom", "phylum", ..., "species"]
        self.scales = scales or ["1x", "3x", "5x"]

        # =================================================================
        # ENCODER SETUP
        # =================================================================

        # Separate ROI scale from context scales
        # ROI is the tight crop (1x), everything else is context
        self.roi_scale = "1x"
        self.context_scales = [s for s in self.scales if s != self.roi_scale]

        # Support asymmetric backbones (different sizes for ROI vs context)
        default_backbone = cfg.model.backbone
        roi_backbone_name = roi_backbone or default_backbone
        context_backbone_name = context_backbone or default_backbone

        # ROI encoder - produces global features that become the query
        print(f"Creating ROI encoder: {roi_backbone_name}")
        self.roi_encoder = PatchFeatureExtractor(roi_backbone_name, pretrained=True)
        roi_feature_dim = self.roi_encoder.feature_dim

        # Context encoders - one per context scale, produce patch features
        print(f"Creating {len(self.context_scales)} context encoders: {context_backbone_name}")
        self.context_encoders = nn.ModuleDict()
        for scale in self.context_scales:
            self.context_encoders[scale] = PatchFeatureExtractor(
                context_backbone_name, pretrained=True
            )
        context_feature_dim = self.context_encoders[self.context_scales[0]].feature_dim

        # =================================================================
        # ASYMMETRIC BACKBONE HANDLING
        # =================================================================

        # Check if ROI and context have different feature dimensions
        self.asymmetric = roi_feature_dim != context_feature_dim

        # Use ROI dimension as the attention space
        # (projecting smaller context up to larger ROI dimension)
        attention_dim = roi_feature_dim

        if self.asymmetric:
            print(f"Asymmetric mode: ROI dim={roi_feature_dim}, Context dim={context_feature_dim}")
            print(f"Projecting context features to {attention_dim}")
            # Linear projection to align context features with ROI features
            self.context_proj = nn.Linear(context_feature_dim, attention_dim)
        else:
            # Identity for symmetric case (no projection needed)
            self.context_proj = nn.Identity()

        # =================================================================
        # CROSS-ATTENTION MODULE
        # =================================================================

        print(f"Creating cross-attention with {num_attention_layers} layers, {num_attention_heads} heads")
        self.cross_attention = MultiLayerCrossAttention(
            embed_dim=attention_dim,
            num_layers=num_attention_layers,
            num_heads=num_attention_heads,
            dropout=0.1,
        )

        # =================================================================
        # CLASSIFICATION HEAD SETUP
        # =================================================================

        feature_dim = attention_dim  # Output dimension for classification

        # Project attended features to classification space
        hidden_dim = cfg.model.hidden_dim  # e.g., 2048
        head_dim = cfg.model.head_dim      # e.g., 512

        self.feature_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
        )

        # Build hierarchical classification heads
        self._build_heads(hidden_dim, head_dim)

        # =================================================================
        # LOSS AND TRAINING SETUP
        # =================================================================

        # Cross-entropy loss with optional label smoothing
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=cfg.training.label_smoothing
        )

        # Hierarchy weights for combining losses across levels
        self.hierarchy_weights = dict(cfg.loss.hierarchy_weights)

        # Storage for validation outputs (cleared each epoch)
        self.val_outputs = []

    def _build_heads(self, hidden_dim: int, head_dim: int):
        """
        Build hierarchical classification heads.

        Architecture: Each level (except kingdom) receives conditioning from
        the parent level's predicted distribution plus direct features.

        For level L with parent P:
            conditioning = Linear(P_num_classes → head_dim)(softmax(P_logits))
            features = Linear(hidden_dim → head_dim)(shared_features)
            logits_L = Linear(2 × head_dim → L_num_classes)([conditioning; features])

        This allows each level to use parent predictions to constrain its
        output space, improving taxonomic coherence.

        Args:
            hidden_dim: Dimension of shared features from feature_proj
            head_dim: Dimension for parent conditioning and level features
        """
        def block(out_dim: int):
            """Classification block: concatenated input → logits"""
            return nn.Linear(head_dim * 2, out_dim)

        # Kingdom head: no parent conditioning (top of hierarchy)
        self.kingdom_head = nn.Linear(hidden_dim, self.class_counts["kingdom"])

        # Phylum head: conditioned on kingdom
        self.kingdom_to_phylum = nn.Linear(self.class_counts["kingdom"], head_dim)
        self.phylum_features = nn.Linear(hidden_dim, head_dim)
        self.phylum_head = block(self.class_counts["phylum"])

        # Class head: conditioned on phylum
        self.phylum_to_class = nn.Linear(self.class_counts["phylum"], head_dim)
        self.class_features = nn.Linear(hidden_dim, head_dim)
        self.class_head = block(self.class_counts["class"])

        # Order head: conditioned on class
        self.class_to_order = nn.Linear(self.class_counts["class"], head_dim)
        self.order_features = nn.Linear(hidden_dim, head_dim)
        self.order_head = block(self.class_counts["order"])

        # Family head: conditioned on order
        self.order_to_family = nn.Linear(self.class_counts["order"], head_dim)
        self.family_features = nn.Linear(hidden_dim, head_dim)
        self.family_head = block(self.class_counts["family"])

        # Genus head: conditioned on family
        self.family_to_genus = nn.Linear(self.class_counts["family"], head_dim)
        self.genus_features = nn.Linear(hidden_dim, head_dim)
        self.genus_head = block(self.class_counts["genus"])

        # Species head: conditioned on genus (final level)
        self.genus_to_species = nn.Linear(self.class_counts["genus"], head_dim)
        self.species_features = nn.Linear(hidden_dim, head_dim)
        self.species_head = block(self.class_counts["species"])

    def _fuse_context_patches(
        self,
        context_patches: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Combine patch features from multiple context scales.

        Simple concatenation along the sequence dimension, with optional
        projection for asymmetric backbones.

        For 3x and 5x scales with 7×7 patch grids each:
            3x patches: (B, 49, D)
            5x patches: (B, 49, D)
            Combined:   (B, 98, D)

        The cross-attention will learn to weight these patches appropriately.

        Args:
            context_patches: Dict mapping scale names to patch tensors

        Returns:
            Combined patch tensor of shape (B, N_total, D)
        """
        all_patches = []
        for scale in self.context_scales:
            if scale in context_patches:
                patches = context_patches[scale]
                # Project to attention dimension if asymmetric
                patches = self.context_proj(patches)
                all_patches.append(patches)

        if not all_patches:
            raise ValueError("No context patches available")

        # Concatenate along sequence dimension: (B, N1+N2+..., D)
        return torch.cat(all_patches, dim=1)

    def forward(self, images: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass with cross-attention fusion.

        Data flow:
        1. Extract ROI global features (query for attention)
        2. Extract context patch features (keys and values)
        3. Apply cross-attention (ROI attends to context)
        4. Project attended features to classification space
        5. Apply hierarchical classification heads

        Args:
            images: Dict with scale keys ("1x", "3x", "5x") mapping to
                   image tensors of shape (B, 3, 224, 224)

        Returns:
            Dict of logits for each taxonomy level
        """
        # =================================================================
        # FEATURE EXTRACTION
        # =================================================================

        # Extract ROI features (will be the query for attention)
        # roi_global: (B, D) - single vector representing the organism
        # roi_patches: (B, 49, D) - spatial features (not used in base model)
        roi_global, roi_patches = self.roi_encoder(images[self.roi_scale])

        # Reshape to attention format: (B, 1, D)
        query = roi_global.unsqueeze(1)

        # Extract context patch features (keys and values for attention)
        context_patches = {}
        for scale in self.context_scales:
            if scale in images:
                # We only need patches, not global features for context
                _, patches = self.context_encoders[scale](images[scale])
                context_patches[scale] = patches

        # =================================================================
        # CROSS-ATTENTION FUSION
        # =================================================================

        # Combine patches from all context scales: (B, N_total, D)
        all_context = self._fuse_context_patches(context_patches)

        # Apply cross-attention: ROI queries context
        # attended: (B, 1, D) - ROI enriched with context information
        attended = self.cross_attention(query, all_context)

        # Remove sequence dimension: (B, D)
        attended = attended.squeeze(1)

        # =================================================================
        # CLASSIFICATION
        # =================================================================

        # Project to classification space
        shared = self.feature_proj(attended)

        # Hierarchical classification with parent conditioning
        # Kingdom (top level - no conditioning)
        kingdom_logits = self.kingdom_head(shared)
        kingdom_probs = torch.softmax(kingdom_logits, dim=1)

        # Phylum (conditioned on kingdom)
        phylum_input = self.kingdom_to_phylum(kingdom_probs)
        phylum_feats = self.phylum_features(shared)
        phylum_logits = self.phylum_head(torch.cat([phylum_input, phylum_feats], dim=1))
        phylum_probs = torch.softmax(phylum_logits, dim=1)

        # Class (conditioned on phylum)
        class_input = self.phylum_to_class(phylum_probs)
        class_feats = self.class_features(shared)
        class_logits = self.class_head(torch.cat([class_input, class_feats], dim=1))
        class_probs = torch.softmax(class_logits, dim=1)

        # Order (conditioned on class)
        order_input = self.class_to_order(class_probs)
        order_feats = self.order_features(shared)
        order_logits = self.order_head(torch.cat([order_input, order_feats], dim=1))
        order_probs = torch.softmax(order_logits, dim=1)

        # Family (conditioned on order)
        family_input = self.order_to_family(order_probs)
        family_feats = self.family_features(shared)
        family_logits = self.family_head(torch.cat([family_input, family_feats], dim=1))
        family_probs = torch.softmax(family_logits, dim=1)

        # Genus (conditioned on family)
        genus_input = self.family_to_genus(family_probs)
        genus_feats = self.genus_features(shared)
        genus_logits = self.genus_head(torch.cat([genus_input, genus_feats], dim=1))
        genus_probs = torch.softmax(genus_logits, dim=1)

        # Species (conditioned on genus - final level)
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
        """
        Compute weighted loss across all taxonomy levels.

        Total loss = Σ_L (w_L × CE(outputs[L], targets[L])) / Σ_L w_L

        Typical weights emphasize species (most important for ranking):
            kingdom: 0.05, phylum: 0.05, class: 0.1, order: 0.1,
            family: 0.15, genus: 0.2, species: 0.35

        Args:
            outputs: Dict of logits for each level
            targets: Dict of ground truth labels for each level

        Returns:
            total_loss: Weighted average loss across levels
            level_losses: Dict of individual losses per level
        """
        total_loss = 0.0
        level_losses = {}
        weight_sum = 0.0

        for level in self.levels:
            if level not in outputs or level not in targets:
                continue

            # Standard cross-entropy for each level
            loss = self.criterion(outputs[level], targets[level])
            w = self.hierarchy_weights.get(level, 1.0)

            total_loss += w * loss
            weight_sum += w
            level_losses[level] = loss

        # Normalize by total weight
        total_loss = total_loss / max(weight_sum, 1e-6)
        return total_loss, level_losses

    def training_step(self, batch, batch_idx):
        """
        Single training step.

        Args:
            batch: Tuple of (images_dict, labels_dict)
            batch_idx: Index of current batch

        Returns:
            Training loss for this batch
        """
        images, labels = batch
        outputs = self(images)
        loss, level_losses = self.hierarchical_loss(outputs, labels)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        for lvl, lvl_loss in level_losses.items():
            self.log(f"train_{lvl}_loss", lvl_loss, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Single validation step with accuracy tracking.

        Args:
            batch: Tuple of (images_dict, labels_dict)
            batch_idx: Index of current batch

        Returns:
            Validation loss for this batch
        """
        images, labels = batch
        outputs = self(images)
        loss, level_losses = self.hierarchical_loss(outputs, labels)

        # Log loss
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        # Log per-level loss and accuracy
        for lvl, lvl_loss in level_losses.items():
            self.log(f"val_{lvl}_loss", lvl_loss, sync_dist=True)

            # Compute accuracy
            preds = torch.argmax(outputs[lvl], dim=1)
            acc = (preds == labels[lvl]).float().mean()
            self.log(f"val_{lvl}_acc", acc, prog_bar=True, sync_dist=True)

        # Store for epoch-end analysis
        self.val_outputs.append({"outputs": outputs, "labels": labels})
        return loss

    def on_validation_epoch_end(self):
        """Clear stored validation outputs at end of epoch."""
        self.val_outputs.clear()

    def predict_step(self, batch, batch_idx):
        """
        Prediction step for generating submissions.

        Args:
            batch: Tuple of (images_dict, annotation_ids)
            batch_idx: Index of current batch

        Returns:
            Dict with annotation IDs, probabilities, predictions, and confidences
        """
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

            # Confidence = probability of predicted class
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
        """
        Configure optimizers with parameter-specific learning rates.

        Learning Rate Strategy:
        -----------------------
        1. Encoder parameters: base_lr × backbone_lr_scale (0.1)
           - Pretrained weights, need careful fine-tuning

        2. Cross-attention parameters: base_lr × 0.5
           - New layers, but need to align with encoder representations

        3. Classification head parameters: base_lr
           - New layers, can learn quickly

        Scheduler: CosineAnnealingWarmRestarts
            - T_0=5: First restart after 5 epochs
            - T_mult=2: Double period after each restart
            - Helps escape local minima and explore loss landscape

        Returns:
            Dict with optimizer and lr_scheduler configuration
        """
        # Separate parameters into groups based on module type
        encoder_params = []
        attention_params = []
        head_params = []

        for name, param in self.named_parameters():
            if "roi_encoder" in name or "context_encoders" in name:
                encoder_params.append(param)
            elif "cross_attention" in name:
                attention_params.append(param)
            else:
                head_params.append(param)

        # Create optimizer with different learning rates per group
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": encoder_params,
                    "lr": self.cfg.training.learning_rate * self.cfg.training.backbone_lr_scale,
                },
                {
                    "params": attention_params,
                    "lr": self.cfg.training.learning_rate * 0.5,  # Moderate for attention
                },
                {
                    "params": head_params,
                    "lr": self.cfg.training.learning_rate,
                },
            ],
            weight_decay=self.cfg.training.weight_decay,
        )

        # Cosine annealing with warm restarts
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


class MultiScaleCrossAttentionWithROIPatches(MultiScaleCrossAttentionClassifier):
    """
    Variant that uses ROI patch features as queries (instead of just global).

    Key Difference from Base Model:
    --------------------------------
    Base model:  Q = ROI_global (1 query vector)
    This model:  Q = ROI_patches (49 query vectors, one per spatial position)

    Each ROI patch independently attends to all context patches, then
    results are averaged and combined with ROI global via residual.

    Potential Benefits:
    -------------------
    - Fine-grained correspondence: Different parts of the organism can
      attend to different context regions (e.g., fins attend to water
      column, body attends to substrate)
    - Richer representations: 49× more attention operations means more
      opportunities to extract relevant context information

    Potential Drawbacks:
    --------------------
    - Higher memory/compute: 49× more attention operations
    - May overfit to spurious local correspondences

    Architecture:
        ROI patches (B, 49, D) ────────────┐
                                           │ Query
        Context patches (B, 98, D) ─────── CrossAttention ──> attended (B, 49, D)
                                                                      │
                                                                 mean over patches
                                                                      │
        ROI global (B, D) ───────────────────── + ──────────> final (B, D)
                                          (residual)

    Usage:
        Same as MultiScaleCrossAttentionClassifier
    """

    def forward(self, images: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass using ROI patches as queries.

        Instead of a single global query, each of the 49 ROI patches
        independently queries the context. Results are averaged and
        combined with global features.

        Args:
            images: Dict with scale keys mapping to image tensors

        Returns:
            Dict of logits for each taxonomy level
        """
        # =================================================================
        # FEATURE EXTRACTION
        # =================================================================

        # Extract ROI features - now we use BOTH global and patches
        roi_global, roi_patches = self.roi_encoder(images[self.roi_scale])
        # roi_patches: (B, N_roi, D) where N_roi = 49 typically

        # Extract context patch features
        context_patches = {}
        for scale in self.context_scales:
            if scale in images:
                _, patches = self.context_encoders[scale](images[scale])
                context_patches[scale] = patches

        # Combine all context patches
        all_context = self._fuse_context_patches(context_patches)

        # =================================================================
        # PATCH-LEVEL CROSS-ATTENTION
        # =================================================================

        # Each ROI patch queries all context patches
        # Input: (B, 49, D) queries, (B, 98, D) key/values
        # Output: (B, 49, D) attended patches
        attended_patches = self.cross_attention(roi_patches, all_context)

        # Average pooling over attended patches
        # This combines information from all 49 patch-level attention operations
        attended = attended_patches.mean(dim=1)  # (B, D)

        # Residual connection with global ROI features
        # This ensures we don't lose the global representation
        attended = attended + roi_global

        # =================================================================
        # CLASSIFICATION (same as base class)
        # =================================================================

        # Project to classification space
        shared = self.feature_proj(attended)

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
