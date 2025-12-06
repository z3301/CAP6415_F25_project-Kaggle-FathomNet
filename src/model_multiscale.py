"""
Multi-scale ConvNeXtV2 Model for FathomNet 2025 Competition
============================================================

This module implements a hierarchical taxonomy classifier that uses multiple
image scales (ROI + context) to improve marine species classification.

Architecture Overview
---------------------

The key insight is that marine species identification benefits from both:
1. Fine-grained ROI features (organism details)
2. Contextual environmental features (habitat, depth cues, nearby organisms)

This is implemented via:
1. **Multi-scale Feature Extraction**: Separate ConvNeXtV2-Base encoders process
   each scale independently. For N scales, we use N independent backbones.

2. **Feature Fusion via Concatenation**: Features from all scales are concatenated
   into a single vector: [f_1x || f_3x || f_5x] where each f_i ∈ ℝ^1024 for
   ConvNeXtV2-Base, giving a fused representation ∈ ℝ^(N×1024).

3. **Hierarchical Classification**: The fused features pass through a cascade
   of classification heads that respect the taxonomic hierarchy:
   Kingdom → Phylum → Class → Order → Family → Genus → Species

Scale Definitions
-----------------

Scales are defined relative to the bounding box annotation:
- **1x**: Original ROI crop (just the annotated organism)
- **3x**: 3× expansion around ROI center (organism + immediate context)
- **5x**: 5× expansion (organism + broader environmental context)
- **full**: Entire image resized (global scene context)

The pre-cropped images are stored in:
    {data_root}/train/rois/{scale}/{image_id}_{annotation_id}.png

Hierarchical Classification
---------------------------

The model predicts all 7 taxonomic levels simultaneously using a cascading
architecture where predictions at higher levels inform lower-level predictions:

    shared_features → kingdom_logits
                   ↘
    [kingdom_probs || shared_features] → phylum_logits
                                      ↘
    [phylum_probs || shared_features] → class_logits
                                     ↘
                                     ... (continues to species)

This cascade allows the model to:
1. Use coarse predictions to inform fine-grained predictions
2. Naturally encode the constraint that species must be consistent with genus, etc.
3. Learn hierarchically-structured representations

The concatenation of parent probabilities with shared features is implemented as:

    parent_embedding = linear_projection(parent_probs)  # Maps (K_parent,) → (head_dim,)
    level_features = linear_projection(shared)          # Maps (hidden_dim,) → (head_dim,)
    level_input = concat([parent_embedding, level_features])  # (2 * head_dim,)
    level_logits = classification_head(level_input)     # (2 * head_dim,) → (K_level,)

Loss Function
-------------

We use a weighted hierarchical cross-entropy loss:

    L_total = Σ_i (w_i × CE(logits_i, targets_i)) / Σ_i w_i

Where weights are configured in cfg.loss.hierarchy_weights, typically giving
higher weight to species-level predictions since that's the competition metric.

Training Details
----------------

- **Optimizer**: AdamW with separate learning rates for backbone and heads
- **Backbone LR**: base_lr × backbone_lr_scale (typically 0.1)
- **Head LR**: base_lr (typically 3e-4)
- **Scheduler**: CosineAnnealingWarmRestarts with T_0=5, T_mult=2
- **Regularization**: Dropout (0.3), label smoothing (0.1)
- **Precision**: Mixed precision (fp16) for faster training

Experimental Results (FathomNet 2025)
-------------------------------------

| Configuration        | Val Loss | Private Score | Notes                    |
|---------------------|----------|---------------|--------------------------|
| 3-scale (1x,3x,5x)  | ~0.8     | 2.06          | 264M params             |
| 3-scale + TaxLoss   | ~0.75    | 2.00          | Best single model       |
| 4-scale (+ full)    | TBD      | TBD           | Currently training      |

References
----------

- ConvNeXtV2: https://arxiv.org/abs/2301.00808
- FathomNet Competition: https://www.kaggle.com/competitions/fathomnet-2025
- 1st Place Solution: Uses DINOv2-Large + MCEAM (cross-attention on patches)

Author: Dan Zimmerman
Course: CAP6415 - Computer Vision (Fall 2025)
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

    This is the primary model architecture for the FathomNet 2025 competition.
    It processes multiple context scales of each ROI image through independent
    encoders, concatenates the features, and predicts all 7 taxonomic levels.

    Architecture Diagram
    --------------------

                    ┌─────────────────┐
        1x ROI ───▶ │ ConvNeXtV2-Base │ ───▶ f_1x (1024-d)
                    └─────────────────┘              │
                    ┌─────────────────┐              │
        3x ctx ───▶ │ ConvNeXtV2-Base │ ───▶ f_3x ──┼──▶ concat ───▶ [f_1x || f_3x || f_5x]
                    └─────────────────┘              │         (3072-d for 3 scales)
                    ┌─────────────────┐              │              │
        5x ctx ───▶ │ ConvNeXtV2-Base │ ───▶ f_5x ──┘              │
                    └─────────────────┘                             ▼
                                                          ┌─────────────────┐
                                                          │  LayerNorm      │
                                                          │  Linear(→2048)  │
                                                          │  GELU + Dropout │
                                                          └────────┬────────┘
                                                                   │
                                                                   ▼
                                                          shared_features (2048-d)
                                                                   │
                                            ┌──────────────────────┼──────────────────────┐
                                            ▼                      ▼                      ▼
                                    Kingdom Head            Phylum Head    ...    Species Head
                                        (2)                 (8 classes)           (80 classes)

    Parameters
    ----------
    cfg : DictConfig
        Configuration object containing model, training, and loss settings.
        Key fields:
        - cfg.model.backbone: timm model name (e.g., "convnextv2_base.fcmae_ft_in22k_in1k")
        - cfg.model.hidden_dim: Dimension of shared features (default: 2048)
        - cfg.model.head_dim: Dimension of per-level embeddings (default: 256)
        - cfg.training.label_smoothing: Label smoothing factor (default: 0.1)
        - cfg.loss.hierarchy_weights: Dict of per-level loss weights

    class_counts : Dict[str, int]
        Number of classes at each taxonomic level.
        Example: {"kingdom": 2, "phylum": 8, "class": 22, ..., "species": 80}

    scales : List[str], optional
        Which scales to use. Default: ["1x", "3x", "5x"]
        Each scale requires a corresponding pre-cropped image directory.

    Input Format
    ------------
    The forward() method expects a dict mapping scale names to image tensors:

        images = {
            "1x": torch.Tensor,  # (B, 3, 224, 224)
            "3x": torch.Tensor,  # (B, 3, 224, 224)
            "5x": torch.Tensor,  # (B, 3, 224, 224)
        }

    Output Format
    -------------
    Returns a dict mapping taxonomy level names to logits:

        outputs = {
            "kingdom": torch.Tensor,  # (B, 2)
            "phylum": torch.Tensor,   # (B, 8)
            "class": torch.Tensor,    # (B, 22)
            "order": torch.Tensor,    # (B, 43)
            "family": torch.Tensor,   # (B, 66)
            "genus": torch.Tensor,    # (B, 75)
            "species": torch.Tensor,  # (B, 80)
        }

    Attributes
    ----------
    backbones : nn.ModuleDict
        Dictionary of ConvNeXtV2 encoders, one per scale.
        Keys are scale names ("1x", "3x", "5x", "full").

    shared_features : nn.Sequential
        Fusion network that processes concatenated scale features.
        Architecture: LayerNorm → Linear → GELU → Dropout

    {level}_head : nn.Linear
        Classification head for each taxonomic level.

    {level}_to_{child} : nn.Linear
        Projection from parent level logits to child level input.

    {level}_features : nn.Linear
        Projection from shared features to level-specific features.
    """

    def __init__(self, cfg: DictConfig, class_counts: Dict[str, int], scales: List[str] = None):
        """
        Initialize the multi-scale classifier.

        Memory Requirements
        -------------------
        For 3 scales with ConvNeXtV2-Base (88M params each):
        - Total backbone params: ~264M
        - Fusion + heads: ~8M
        - Total: ~272M params (~1GB fp32, ~500MB fp16)

        For 4 scales: ~360M params (~1.4GB fp32)
        """
        super().__init__()

        # Save hyperparameters for checkpoint loading (excludes cfg for flexibility)
        self.save_hyperparameters(ignore=["cfg"])

        self.cfg = cfg
        self.class_counts = class_counts
        self.levels = list(class_counts.keys())  # ["kingdom", "phylum", ..., "species"]
        self.scales = scales or ["1x", "3x", "5x"]

        backbone_name = cfg.model.backbone

        # ──────────────────────────────────────────────────────────────────────
        # Multi-Scale Backbone Initialization
        # ──────────────────────────────────────────────────────────────────────
        # Create separate backbone for each scale. Using independent encoders
        # (rather than shared) allows each to specialize:
        # - 1x encoder learns fine-grained organism features
        # - 3x/5x encoders learn contextual/environmental features
        # - full encoder learns global scene features
        #
        # Alternative approaches considered but not used:
        # - Shared backbone + scale-specific adapters (fewer params, less expressive)
        # - Asymmetric backbones (Large for ROI, Base for context) - see attention model
        # ──────────────────────────────────────────────────────────────────────

        self.backbones = nn.ModuleDict()
        for scale in self.scales:
            self.backbones[scale] = timm.create_model(
                backbone_name,
                pretrained=True,  # Use ImageNet-pretrained weights
                num_classes=0,    # Remove classification head, return features
            )

        # ──────────────────────────────────────────────────────────────────────
        # Feature Fusion Layer
        # ──────────────────────────────────────────────────────────────────────
        # After backbone extraction, we have N feature vectors of dimension D
        # (D=1024 for ConvNeXtV2-Base). We concatenate and project to hidden_dim.
        #
        # Concatenation is a simple but effective fusion strategy:
        # - Preserves information from all scales without mixing
        # - Downstream layers can learn to weight scales appropriately
        # - Computationally efficient (no attention overhead)
        #
        # The 1st-place solution uses cross-attention instead, which allows
        # the ROI to selectively attend to relevant context patches.
        # ──────────────────────────────────────────────────────────────────────

        feature_dim = self.backbones[self.scales[0]].num_features  # 1024 for ConvNeXtV2-Base
        in_features = feature_dim * len(self.scales)  # 3072 for 3 scales, 4096 for 4 scales

        hidden_dim = cfg.model.hidden_dim  # Typically 2048
        head_dim = cfg.model.head_dim      # Typically 256

        self.shared_features = nn.Sequential(
            nn.LayerNorm(in_features),           # Normalize concatenated features
            nn.Linear(in_features, hidden_dim),   # Project to hidden dimension
            nn.GELU(),                            # Non-linearity (GELU works well with transformers/ConvNeXt)
            nn.Dropout(0.3),                      # Regularization to prevent overfitting
        )

        # Build the hierarchical classification heads
        self._build_heads(hidden_dim, head_dim)

        # ──────────────────────────────────────────────────────────────────────
        # Loss Function
        # ──────────────────────────────────────────────────────────────────────
        # Standard cross-entropy with label smoothing.
        # Label smoothing prevents overconfident predictions and improves
        # calibration, which is important for the taxonomic distance metric.
        # ──────────────────────────────────────────────────────────────────────

        self.criterion = nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)
        self.hierarchy_weights = dict(cfg.loss.hierarchy_weights)

        # Storage for validation outputs (cleared each epoch)
        self.val_outputs = []

    def _build_heads(self, hidden_dim: int, head_dim: int):
        """
        Build hierarchical classification heads for all taxonomic levels.

        Hierarchical Head Architecture
        -------------------------------

        For the root level (Kingdom):
            kingdom_logits = Linear(hidden_dim → K_kingdom)(shared_features)

        For all other levels:
            parent_embedding = Linear(K_parent → head_dim)(softmax(parent_logits))
            level_features = Linear(hidden_dim → head_dim)(shared_features)
            level_input = concat([parent_embedding, level_features])  # (2 × head_dim)
            level_logits = Linear(2 × head_dim → K_level)(level_input)

        The parent embedding allows the model to condition predictions on
        the taxonomic context (e.g., knowing it's a fish helps predict species).

        Parameters
        ----------
        hidden_dim : int
            Dimension of the shared feature representation (typically 2048)

        head_dim : int
            Dimension of the per-level embeddings (typically 256)

        Notes
        -----
        Using softmax(parent_logits) rather than hard predictions allows
        gradient flow during training and captures uncertainty.
        """

        def block(out_dim: int):
            """Classification block that takes concatenated parent + level features."""
            return nn.Linear(head_dim * 2, out_dim)

        # Kingdom (root level) - directly from shared features
        self.kingdom_head = nn.Linear(hidden_dim, self.class_counts["kingdom"])

        # Phylum - conditioned on Kingdom predictions
        self.kingdom_to_phylum = nn.Linear(self.class_counts["kingdom"], head_dim)
        self.phylum_features = nn.Linear(hidden_dim, head_dim)
        self.phylum_head = block(self.class_counts["phylum"])

        # Class - conditioned on Phylum predictions
        self.phylum_to_class = nn.Linear(self.class_counts["phylum"], head_dim)
        self.class_features = nn.Linear(hidden_dim, head_dim)
        self.class_head = block(self.class_counts["class"])

        # Order - conditioned on Class predictions
        self.class_to_order = nn.Linear(self.class_counts["class"], head_dim)
        self.order_features = nn.Linear(hidden_dim, head_dim)
        self.order_head = block(self.class_counts["order"])

        # Family - conditioned on Order predictions
        self.order_to_family = nn.Linear(self.class_counts["order"], head_dim)
        self.family_features = nn.Linear(hidden_dim, head_dim)
        self.family_head = block(self.class_counts["family"])

        # Genus - conditioned on Family predictions
        self.family_to_genus = nn.Linear(self.class_counts["family"], head_dim)
        self.genus_features = nn.Linear(hidden_dim, head_dim)
        self.genus_head = block(self.class_counts["genus"])

        # Species - conditioned on Genus predictions (this is the competition target)
        self.genus_to_species = nn.Linear(self.class_counts["genus"], head_dim)
        self.species_features = nn.Linear(hidden_dim, head_dim)
        self.species_head = block(self.class_counts["species"])

    def _extract_features(self, images: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract and concatenate features from all scales.

        This is the multi-scale feature extraction step. Each scale's image
        is passed through its dedicated backbone, and the resulting feature
        vectors are concatenated.

        Parameters
        ----------
        images : Dict[str, torch.Tensor]
            Dictionary mapping scale names to image tensors.
            Each tensor has shape (B, 3, 224, 224).

        Returns
        -------
        torch.Tensor
            Concatenated features with shape (B, N × D) where:
            - N is the number of scales
            - D is the backbone feature dimension (1024 for ConvNeXtV2-Base)

        Feature Extraction Details
        --------------------------
        For ConvNeXtV2-Base:
        - Input: (B, 3, 224, 224)
        - Stem: 4×4 conv, stride 4 → (B, 128, 56, 56)
        - Stage 1: 3 blocks → (B, 128, 56, 56)
        - Stage 2: 3 blocks, downsample → (B, 256, 28, 28)
        - Stage 3: 27 blocks, downsample → (B, 512, 14, 14)
        - Stage 4: 3 blocks, downsample → (B, 1024, 7, 7)
        - Global pool → (B, 1024)
        """
        scale_features = []

        for scale in self.scales:
            if scale in images:
                # Extract features using the scale-specific backbone
                # Result: (B, feature_dim) where feature_dim=1024 for ConvNeXtV2-Base
                feats = self.backbones[scale](images[scale])
                scale_features.append(feats)

        # Concatenate along feature dimension: (B, N × feature_dim)
        return torch.cat(scale_features, dim=1)

    def forward(self, images: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-scale backbones and hierarchical classifier.

        This implements the full inference pipeline:
        1. Extract features from each scale using dedicated backbones
        2. Concatenate and project to shared representation
        3. Cascade through hierarchical classification heads

        Parameters
        ----------
        images : Dict[str, torch.Tensor]
            Dictionary with keys like "1x", "3x", "5x" mapping to image tensors.
            Each tensor should have shape (B, 3, 224, 224).

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary mapping taxonomic level names to logit tensors.
            Keys: "kingdom", "phylum", "class", "order", "family", "genus", "species"
            Values: Tensors of shape (B, num_classes_at_level)

        Hierarchical Cascade Computation
        ---------------------------------

        Let h = shared_features(concat(backbone(img) for img in images))

        For each level after kingdom:
            parent_probs = softmax(parent_logits)
            parent_embed = W_parent→child @ parent_probs  # (head_dim,)
            level_feats = W_level @ h                      # (head_dim,)
            level_logits = W_head @ [parent_embed || level_feats]  # (num_classes,)
        """
        # Step 1: Extract and fuse features from all scales
        combined_features = self._extract_features(images)
        shared = self.shared_features(combined_features)

        # Step 2: Hierarchical classification cascade
        # Kingdom (root level)
        kingdom_logits = self.kingdom_head(shared)
        kingdom_probs = torch.softmax(kingdom_logits, dim=1)

        # Phylum (conditioned on Kingdom)
        phylum_input = self.kingdom_to_phylum(kingdom_probs)
        phylum_feats = self.phylum_features(shared)
        phylum_logits = self.phylum_head(torch.cat([phylum_input, phylum_feats], dim=1))
        phylum_probs = torch.softmax(phylum_logits, dim=1)

        # Class (conditioned on Phylum)
        class_input = self.phylum_to_class(phylum_probs)
        class_feats = self.class_features(shared)
        class_logits = self.class_head(torch.cat([class_input, class_feats], dim=1))
        class_probs = torch.softmax(class_logits, dim=1)

        # Order (conditioned on Class)
        order_input = self.class_to_order(class_probs)
        order_feats = self.order_features(shared)
        order_logits = self.order_head(torch.cat([order_input, order_feats], dim=1))
        order_probs = torch.softmax(order_logits, dim=1)

        # Family (conditioned on Order)
        family_input = self.order_to_family(order_probs)
        family_feats = self.family_features(shared)
        family_logits = self.family_head(torch.cat([family_input, family_feats], dim=1))
        family_probs = torch.softmax(family_logits, dim=1)

        # Genus (conditioned on Family)
        genus_input = self.family_to_genus(family_probs)
        genus_feats = self.genus_features(shared)
        genus_logits = self.genus_head(torch.cat([genus_input, genus_feats], dim=1))
        genus_probs = torch.softmax(genus_logits, dim=1)

        # Species (conditioned on Genus) - this is the competition target
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
        Compute weighted cross-entropy loss across all taxonomy levels.

        The total loss is a weighted sum of per-level losses:

            L_total = (Σ_i w_i × CE(outputs_i, targets_i)) / (Σ_i w_i)

        This allows emphasizing certain levels (e.g., species) during training.

        Parameters
        ----------
        outputs : Dict[str, torch.Tensor]
            Dictionary of logits from forward(), keyed by level name.

        targets : Dict[str, torch.Tensor]
            Dictionary of ground truth labels, keyed by level name.
            Each tensor has shape (B,) with integer class indices.

        Returns
        -------
        total_loss : torch.Tensor
            Scalar tensor containing the weighted average loss.

        level_losses : Dict[str, torch.Tensor]
            Dictionary of per-level losses for logging.

        Hierarchy Weights
        -----------------
        Typical configuration (from experiment-multiscale.yaml):
            kingdom: 0.5
            phylum: 0.7
            class: 0.8
            order: 0.9
            family: 1.0
            genus: 1.0
            species: 1.5  # Higher weight because this is the competition metric
        """
        total_loss = 0.0
        level_losses = {}
        weight_sum = 0.0

        for level in self.levels:
            if level not in outputs or level not in targets:
                continue

            # Compute cross-entropy loss for this level
            loss = self.criterion(outputs[level], targets[level])

            # Apply hierarchy weight
            w = self.hierarchy_weights.get(level, 1.0)
            total_loss += w * loss
            weight_sum += w

            level_losses[level] = loss

        # Normalize by total weight
        total_loss = total_loss / max(weight_sum, 1e-6)

        return total_loss, level_losses

    def training_step(self, batch, batch_idx):
        """
        Execute one training step.

        Parameters
        ----------
        batch : Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
            Tuple of (images, labels) where:
            - images: Dict mapping scale names to image tensors
            - labels: Dict mapping level names to label tensors

        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor for backpropagation.
        """
        images, labels = batch

        # Forward pass
        outputs = self(images)

        # Compute hierarchical loss
        loss, level_losses = self.hierarchical_loss(outputs, labels)

        # Log metrics
        self.log("train_loss", loss, prog_bar=True)
        for lvl, lvl_loss in level_losses.items():
            self.log(f"train_{lvl}_loss", lvl_loss, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Execute one validation step.

        In addition to loss, we also compute per-level accuracy.

        Parameters
        ----------
        batch : Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]
            Same format as training_step.

        batch_idx : int
            Index of the current batch.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        images, labels = batch

        outputs = self(images)
        loss, level_losses = self.hierarchical_loss(outputs, labels)

        # Log loss with sync_dist=True for multi-GPU
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        # Log per-level loss and accuracy
        for lvl, lvl_loss in level_losses.items():
            self.log(f"val_{lvl}_loss", lvl_loss, sync_dist=True)

            # Compute accuracy
            preds = torch.argmax(outputs[lvl], dim=1)
            acc = (preds == labels[lvl]).float().mean()
            self.log(f"val_{lvl}_acc", acc, prog_bar=True, sync_dist=True)

        self.val_outputs.append({"outputs": outputs, "labels": labels})
        return loss

    def on_validation_epoch_end(self):
        """Clear validation outputs at end of epoch to free memory."""
        self.val_outputs.clear()

    def predict_step(self, batch, batch_idx):
        """
        Execute one prediction step for inference.

        Parameters
        ----------
        batch : Tuple[Dict[str, torch.Tensor], torch.Tensor]
            Tuple of (images, annotation_ids) where:
            - images: Dict mapping scale names to image tensors
            - annotation_ids: Tensor of annotation IDs for submission

        batch_idx : int
            Index of the current batch.

        Returns
        -------
        Dict
            Dictionary containing:
            - annotation_ids: IDs for matching with submission format
            - probs: Per-level probability distributions
            - preds: Per-level predicted class indices
            - confidences: Per-level prediction confidences
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

            # Confidence is the probability of the predicted class
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
        Configure optimizer and learning rate scheduler.

        Optimization Strategy
        ---------------------

        We use different learning rates for different parts of the model:

        1. **Backbone parameters** (90% of params): Lower LR (base × 0.1)
           - Pretrained weights are already good, fine-tune slowly
           - Prevents catastrophic forgetting of ImageNet features

        2. **Head parameters** (10% of params): Higher LR (base)
           - Random initialization, needs faster learning
           - Task-specific, no pretrained knowledge

        Scheduler: CosineAnnealingWarmRestarts
        - T_0=5: First restart after 5 epochs
        - T_mult=2: Each subsequent period is 2× longer
        - eta_min=1e-6: Minimum learning rate

        This gives a schedule like:
        - Epochs 1-5: Full LR → min
        - Epochs 6-15: Full LR → min (restart)
        - Epochs 16-35: Full LR → min (restart)
        - etc.

        Returns
        -------
        Dict
            Dictionary with 'optimizer' and 'lr_scheduler' keys.
        """
        # Separate parameters by learning rate group
        backbone_params = []
        other_params = []

        for name, param in self.named_parameters():
            if "backbones" in name:
                backbone_params.append(param)
            else:
                other_params.append(param)

        # Create optimizer with parameter groups
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": backbone_params,
                    "lr": self.cfg.training.learning_rate * self.cfg.training.backbone_lr_scale,
                },
                {
                    "params": other_params,
                    "lr": self.cfg.training.learning_rate,
                },
            ],
            weight_decay=self.cfg.training.weight_decay,
        )

        # Learning rate scheduler with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=5,        # First restart after 5 epochs
            T_mult=2,     # Double period after each restart
            eta_min=1e-6  # Minimum learning rate
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }
