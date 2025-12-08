"""
Multi-scale Model with Taxonomic Distance-Aware Loss for FathomNet 2025.

================================================================================
MOTIVATION AND COMPETITION METRIC
================================================================================

The FathomNet 2025 competition uses a hierarchical scoring metric that
penalizes misclassifications based on taxonomic distance:

    Score = (1/N) × Σ_i d(predicted_species_i, true_species_i)

Where d(s1, s2) is the taxonomic distance between species, defined as the
minimum number of edges in the taxonomy tree to traverse from s1 to s2.

Example distances:
    - Same species:        d = 0
    - Same genus:          d = 2  (up to genus, down to other species)
    - Same family:         d = 4  (up 2 levels, down 2 levels)
    - Same order:          d = 6  (up 3, down 3)
    - Different phylum:    d = 12 (up 6, down 6)

Key Insight:
------------
Standard cross-entropy loss treats all misclassifications equally. But for
this competition, predicting a closely related species (e.g., same genus)
is MUCH better than predicting a distant species (e.g., different phylum).

This model incorporates the taxonomic distance directly into the loss
function, encouraging the model to make "smarter" mistakes when uncertain.

================================================================================
ARCHITECTURE OVERVIEW
================================================================================

This model extends MultiScaleTaxonomyClassifier with:

1. TAXONOMIC DISTANCE LOSS
   - Penalizes predictions proportional to their taxonomic distance from truth
   - Mathematical formulation: L = (1-α)×CE + α×E[D|p]

2. TAXONOMIC LABEL SMOOTHING
   - Smooths labels based on taxonomic similarity
   - Related species get higher soft targets than distant ones

3. VALIDATION METRIC TRACKING
   - Logs expected taxonomic score during validation
   - Allows direct monitoring of competition metric

Architecture Diagram:
---------------------
┌─────────────────────────────────────────────────────────────────────┐
│                    Input Images (Multiple Scales)                    │
│                   1x (ROI), 3x, 5x [, full]                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Backbone   │     │   Backbone   │     │   Backbone   │
│  (1x Scale)  │     │  (3x Scale)  │     │  (5x Scale)  │
│  ConvNeXtV2  │     │  ConvNeXtV2  │     │  ConvNeXtV2  │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       └────────────────────┼────────────────────┘
                            ▼
              ┌─────────────────────────┐
              │   Feature Concatenation  │
              │   [f_1x; f_3x; f_5x]     │
              │   (B, 3×1024) = (B, 3072)│
              └────────────┬────────────┘
                           ▼
              ┌─────────────────────────┐
              │    Fusion Network       │
              │    LayerNorm → Linear   │
              │    → GELU → Dropout     │
              └────────────┬────────────┘
                           ▼
              ┌─────────────────────────┐
              │  Hierarchical Classifier │
              │  Kingdom → ... → Species │
              └────────────┬────────────┘
                           ▼
              ┌─────────────────────────┐
              │   Taxonomic-Aware Loss  │
              │   Species: Tax Loss     │
              │   Others: Standard CE   │
              └─────────────────────────┘

================================================================================
LOSS FUNCTION DETAILS
================================================================================

This model supports 4 loss configurations for the species level:

1. "ce" - Standard Cross-Entropy
   L = -log(p_y)

2. "distance" - Taxonomic Distance Loss
   L = (1-α) × CE + α × E_p[D(·, y)]
     = (1-α) × (-log p_y) + α × Σ_i p_i × D(i, y)

   The second term penalizes probability mass on distant species.
   α controls the trade-off (default: 0.3).

3. "smooth" - Taxonomic Label Smoothing
   Instead of hard labels [0,0,1,0,0], use soft labels based on similarity:
   q_i = (1-ε)×I(i=y) + ε × sim(i, y) / Z

   Where sim(i, y) = exp(-β × D(i, y)) for some temperature β.
   Related species get higher soft targets.

4. "both" - Combined Distance + Smoothing
   L = 0.5 × TaxDistanceLoss + 0.5 × TaxLabelSmoothing

================================================================================
EXPECTED TAXONOMIC SCORE COMPUTATION
================================================================================

During validation, we compute an approximation of the competition metric:

    val_tax_score = (1/N) × Σ_i D(argmax(p_i), y_i)

This uses argmax predictions (not soft predictions) to match the actual
submission format. Lower is better.

Key benefit: We can monitor the actual competition metric during training,
rather than relying on cross-entropy loss as a proxy.

================================================================================
DISTANCE MATRIX
================================================================================

The taxonomic distance matrix is precomputed and stored in a CSV file:
    taxonomic_distance_matrix.csv

Format:
    - Rows and columns are species names
    - Cell (i, j) contains D(species_i, species_j)
    - Symmetric matrix with zeros on diagonal

The matrix is loaded and converted to a PyTorch tensor registered as a
buffer (moves to GPU automatically, not treated as a parameter).

================================================================================
EXPERIMENTAL RESULTS
================================================================================

Training Configuration:
    - Scales: 1x, 3x, 5x
    - Backbone: ConvNeXtV2-Base
    - Loss type: "distance" (α=0.3)
    - Batch size: 16, 50 epochs

Results:
    - val_tax_score converged to ~1.7-1.8 (vs ~2.0+ for standard CE)
    - Private leaderboard: 2.00 (our best submission)
    - Improvement from taxonomic awareness: ~0.2-0.3 points

The taxonomic loss consistently outperformed standard cross-entropy,
especially on difficult samples where the model was uncertain.

================================================================================
USAGE
================================================================================

Basic usage:
    model = MultiScaleTaxonomicClassifier(
        cfg=config,
        class_counts={"kingdom": 2, ..., "species": 80},
        id_to_name=id_to_name_mapping,  # Required for loss setup
        scales=["1x", "3x", "5x"],
        distance_matrix_path="taxonomic_distance_matrix.csv",
        loss_type="distance",  # or "smooth", "both", "ce"
        alpha=0.3,             # Weight for distance term
        smoothing=0.1,         # Amount of label smoothing
    )

Training:
    trainer = pl.Trainer(...)
    trainer.fit(model, train_loader, val_loader)

References:
    - Deng et al., "Hierarchical Semantic Indexing for Large Scale Image Retrieval"
    - Redmon and Farhadi, "YOLO9000: Better, Faster, Stronger" (WordTree)
    - Wu et al., "Making Better Mistakes: Leveraging Class Hierarchies"
"""

import os
from typing import Dict, List, Optional

import pandas as pd
import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from ..losses import TaxonomicDistanceLoss, TaxonomicLabelSmoothing


class MultiScaleTaxonomicClassifier(pl.LightningModule):
    """
    Multi-scale classifier with taxonomic distance-aware training.

    This model combines multi-scale feature extraction with loss functions
    that incorporate taxonomic knowledge. The key innovation is that the
    species-level loss penalizes predictions proportional to their taxonomic
    distance from the ground truth.

    Key Features:
    -------------
    1. Multi-scale backbones: Separate ConvNeXtV2 encoders for each scale
    2. Feature fusion: Concatenation followed by learned projection
    3. Hierarchical classification: Parent-conditioned predictions
    4. Taxonomic loss: Distance-aware penalty for species misclassification
    5. Metric tracking: Logs expected taxonomic score during validation

    Loss Types:
    -----------
    - "ce": Standard cross-entropy (baseline)
    - "distance": CE + expected distance penalty
    - "smooth": Taxonomic label smoothing
    - "both": Combination of distance and smoothing

    Args:
        cfg: OmegaConf configuration with model/training settings
        class_counts: Dict mapping level names to number of classes
        id_to_name: Dict mapping level names to {class_id: class_name} dicts
            Required for setting up the taxonomic distance matrix
        scales: List of scale names (default: ["1x", "3x", "5x"])
        distance_matrix_path: Path to CSV with taxonomic distances
        loss_type: Which loss to use ("distance", "smooth", "both", "ce")
        alpha: Weight for distance term in TaxonomicDistanceLoss (default: 0.3)
        smoothing: Amount of label smoothing for TaxonomicLabelSmoothing

    Input Format:
        Dictionary with scale keys mapping to image tensors:
        {
            "1x": torch.Tensor of shape (B, 3, 224, 224),
            "3x": torch.Tensor of shape (B, 3, 224, 224),
            "5x": torch.Tensor of shape (B, 3, 224, 224),
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
        >>> from src.data import load_and_encode_taxonomy
        >>> taxonomy_df, encoders, class_counts, id_to_name = load_and_encode_taxonomy(...)
        >>> model = MultiScaleTaxonomicClassifier(
        ...     cfg=config,
        ...     class_counts=class_counts,
        ...     id_to_name=id_to_name,
        ...     distance_matrix_path="taxonomic_distance_matrix.csv",
        ...     loss_type="distance",
        ... )
    """

    def __init__(
        self,
        cfg: DictConfig,
        class_counts: Dict[str, int],
        id_to_name: Dict[str, Dict[int, str]],
        scales: List[str] = None,
        distance_matrix_path: Optional[str] = None,
        loss_type: str = "distance",  # "distance", "smooth", "both", "ce"
        alpha: float = 0.3,
        smoothing: float = 0.1,
    ):
        super().__init__()
        # Save hyperparameters for checkpoint loading
        # Exclude cfg (not serializable) and id_to_name (large dict)
        self.save_hyperparameters(ignore=["cfg", "id_to_name"])

        # Store configuration
        self.cfg = cfg
        self.class_counts = class_counts
        self.id_to_name = id_to_name  # Needed to map class IDs to species names
        self.levels = list(class_counts.keys())  # ["kingdom", "phylum", ..., "species"]
        self.scales = scales or ["1x", "3x", "5x"]
        self.loss_type = loss_type

        # =================================================================
        # MULTI-SCALE BACKBONE SETUP
        # =================================================================

        backbone_name = cfg.model.backbone  # e.g., "convnextv2_base.fcmae_ft_in22k_in1k"

        # Create separate backbone for each scale
        # Each backbone is independently pretrained and will be fine-tuned
        self.backbones = nn.ModuleDict()
        for scale in self.scales:
            self.backbones[scale] = timm.create_model(
                backbone_name,
                pretrained=True,
                num_classes=0,  # Remove classifier head, keep feature extractor
            )

        # =================================================================
        # FEATURE FUSION NETWORK
        # =================================================================

        # Get feature dimension from backbone (e.g., 1024 for ConvNeXtV2-Base)
        feature_dim = self.backbones[self.scales[0]].num_features

        # Total input features = feature_dim × number of scales
        # e.g., 3 scales × 1024 = 3072
        in_features = feature_dim * len(self.scales)

        hidden_dim = cfg.model.hidden_dim  # e.g., 2048
        head_dim = cfg.model.head_dim      # e.g., 512

        # Fusion layer: project concatenated features to classification space
        # LayerNorm helps with different feature magnitudes across scales
        self.shared_features = nn.Sequential(
            nn.LayerNorm(in_features),          # Normalize concatenated features
            nn.Linear(in_features, hidden_dim), # Project to hidden dim
            nn.GELU(),                          # Non-linearity
            nn.Dropout(0.3),                    # Regularization
        )

        # =================================================================
        # HIERARCHICAL CLASSIFICATION HEADS
        # =================================================================

        self._build_heads(hidden_dim, head_dim)

        # =================================================================
        # LOSS FUNCTION SETUP
        # =================================================================

        # Hierarchy weights for combining losses across taxonomy levels
        self.hierarchy_weights = dict(cfg.loss.hierarchy_weights)

        # Setup taxonomic-aware losses
        self._setup_losses(distance_matrix_path, loss_type, alpha, smoothing)

        # Storage for validation outputs
        self.val_outputs = []

    def _setup_losses(
        self,
        distance_matrix_path: Optional[str],
        loss_type: str,
        alpha: float,
        smoothing: float,
    ):
        """
        Setup loss functions based on configuration.

        For non-species levels, we use standard cross-entropy with label
        smoothing. For species level, we optionally use taxonomic-aware losses.

        The distance matrix is loaded from CSV and converted to a tensor
        that's registered as a buffer (for automatic GPU transfer).

        Args:
            distance_matrix_path: Path to taxonomic distance CSV
            loss_type: Which loss to use for species level
            alpha: Weight for distance term in TaxonomicDistanceLoss
            smoothing: Smoothing amount for TaxonomicLabelSmoothing
        """
        # Standard cross-entropy for non-species levels
        # Uses label smoothing for calibration (from config)
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=self.cfg.training.label_smoothing
        )

        # Setup taxonomic-aware loss for species level
        if distance_matrix_path and os.path.exists(distance_matrix_path):
            # Build list of species names in order of class IDs
            # This ensures distance matrix rows/columns align with logit positions
            species_names = [
                self.id_to_name["species"][i]
                for i in range(self.class_counts["species"])
            ]

            # Create TaxonomicDistanceLoss if requested
            if loss_type in ["distance", "both"]:
                self.tax_distance_loss = TaxonomicDistanceLoss(
                    distance_matrix_path,
                    species_names,
                    alpha=alpha,  # Weight for distance term
                )
            else:
                self.tax_distance_loss = None

            # Create TaxonomicLabelSmoothing if requested
            if loss_type in ["smooth", "both"]:
                self.tax_smooth_loss = TaxonomicLabelSmoothing(
                    distance_matrix_path,
                    species_names,
                    smoothing=smoothing,
                )
            else:
                self.tax_smooth_loss = None

            # =========================================================
            # LOAD DISTANCE MATRIX FOR VALIDATION METRIC
            # =========================================================
            # We need the full distance matrix to compute val_tax_score
            dm = pd.read_csv(distance_matrix_path, index_col=0)
            n_species = self.class_counts["species"]
            dist_tensor = torch.zeros(n_species, n_species)

            # Fill in the distance matrix
            for i, name_i in enumerate(species_names):
                for j, name_j in enumerate(species_names):
                    if name_i in dm.index and name_j in dm.columns:
                        dist_tensor[i, j] = dm.loc[name_i, name_j]
                    elif i == j:
                        # Same species = distance 0
                        dist_tensor[i, j] = 0.0
                    else:
                        # Unknown pair = max distance (conservative)
                        dist_tensor[i, j] = dm.values.max()

            # Register as buffer: moves to GPU with model, not a trainable parameter
            self.register_buffer("distance_matrix", dist_tensor)
            self.use_taxonomic_loss = loss_type != "ce"

        else:
            # No distance matrix available - fall back to standard CE
            self.tax_distance_loss = None
            self.tax_smooth_loss = None
            self.use_taxonomic_loss = False

            # Create dummy distance matrix (all zeros)
            self.register_buffer(
                "distance_matrix",
                torch.zeros(self.class_counts["species"], self.class_counts["species"])
            )

    def _build_heads(self, hidden_dim: int, head_dim: int):
        """
        Build hierarchical classification heads.

        Architecture: Each level (except kingdom) receives conditioning from
        the parent level's predicted distribution plus direct features.

        For level L with parent P:
            conditioning = Linear(P_num_classes → head_dim)(softmax(P_logits))
            features = Linear(hidden_dim → head_dim)(shared_features)
            logits_L = Linear(2 × head_dim → L_num_classes)([conditioning; features])

        This allows taxonomic coherence: if phylum prediction is confident,
        it restricts which classes are plausible, and so on down the tree.

        Args:
            hidden_dim: Dimension of shared features from fusion layer
            head_dim: Dimension for parent conditioning embeddings
        """
        def block(out_dim: int):
            """Create classification block: [conditioning; features] → logits"""
            return nn.Linear(head_dim * 2, out_dim)

        # Kingdom: Top level, no parent conditioning
        self.kingdom_head = nn.Linear(hidden_dim, self.class_counts["kingdom"])

        # Phylum: Conditioned on kingdom
        self.kingdom_to_phylum = nn.Linear(self.class_counts["kingdom"], head_dim)
        self.phylum_features = nn.Linear(hidden_dim, head_dim)
        self.phylum_head = block(self.class_counts["phylum"])

        # Class: Conditioned on phylum
        self.phylum_to_class = nn.Linear(self.class_counts["phylum"], head_dim)
        self.class_features = nn.Linear(hidden_dim, head_dim)
        self.class_head = block(self.class_counts["class"])

        # Order: Conditioned on class
        self.class_to_order = nn.Linear(self.class_counts["class"], head_dim)
        self.order_features = nn.Linear(hidden_dim, head_dim)
        self.order_head = block(self.class_counts["order"])

        # Family: Conditioned on order
        self.order_to_family = nn.Linear(self.class_counts["order"], head_dim)
        self.family_features = nn.Linear(hidden_dim, head_dim)
        self.family_head = block(self.class_counts["family"])

        # Genus: Conditioned on family
        self.family_to_genus = nn.Linear(self.class_counts["family"], head_dim)
        self.genus_features = nn.Linear(hidden_dim, head_dim)
        self.genus_head = block(self.class_counts["genus"])

        # Species: Conditioned on genus (final level, most important for scoring)
        self.genus_to_species = nn.Linear(self.class_counts["genus"], head_dim)
        self.species_features = nn.Linear(hidden_dim, head_dim)
        self.species_head = block(self.class_counts["species"])

    def _extract_features(self, images: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Extract and concatenate features from all scales.

        Each scale goes through its dedicated backbone, producing a feature
        vector. These are concatenated to form the combined representation.

        Args:
            images: Dict mapping scale names to image tensors (B, 3, 224, 224)

        Returns:
            Concatenated features of shape (B, num_scales × feature_dim)
            e.g., (B, 3072) for 3 scales with 1024-dim backbone
        """
        scale_features = []
        for scale in self.scales:
            if scale in images:
                # Extract features through scale-specific backbone
                feats = self.backbones[scale](images[scale])  # (B, 1024)
                scale_features.append(feats)

        # Concatenate along feature dimension
        return torch.cat(scale_features, dim=1)  # (B, 3072)

    def forward(self, images: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass through multi-scale backbones and hierarchical classifier.

        Data flow:
        1. Extract features from each scale using dedicated backbones
        2. Concatenate features from all scales
        3. Project through fusion network
        4. Apply hierarchical classification with parent conditioning

        Args:
            images: Dict with scale keys mapping to image tensors

        Returns:
            Dict of logits for each taxonomy level
        """
        # =================================================================
        # FEATURE EXTRACTION AND FUSION
        # =================================================================

        # Extract and concatenate features from all scales
        combined_features = self._extract_features(images)  # (B, 3072)

        # Project through fusion network
        shared = self.shared_features(combined_features)  # (B, 2048)

        # =================================================================
        # HIERARCHICAL CLASSIFICATION
        # =================================================================

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

        # Species (conditioned on genus - final and most important level)
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
        Compute weighted loss with taxonomic awareness for species.

        For non-species levels: standard cross-entropy with label smoothing
        For species level: taxonomic-aware loss (if configured)

        Loss combination:
            total_loss = Σ_L (w_L × loss_L) / Σ_L w_L

        When loss_type == "both":
            species_loss = 0.5 × TaxDistanceLoss + 0.5 × TaxLabelSmoothing

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

            logits = outputs[level]
            target = targets[level]
            w = self.hierarchy_weights.get(level, 1.0)

            # Species level gets special taxonomic-aware loss
            if level == "species" and self.use_taxonomic_loss:
                if self.tax_distance_loss and self.tax_smooth_loss:
                    # Combine both losses equally
                    loss = 0.5 * self.tax_distance_loss(logits, target) + \
                           0.5 * self.tax_smooth_loss(logits, target)
                elif self.tax_distance_loss:
                    # Just distance-aware loss
                    loss = self.tax_distance_loss(logits, target)
                elif self.tax_smooth_loss:
                    # Just taxonomic label smoothing
                    loss = self.tax_smooth_loss(logits, target)
                else:
                    # Fallback to standard CE
                    loss = self.ce_loss(logits, target)
            else:
                # Non-species levels use standard CE
                loss = self.ce_loss(logits, target)

            total_loss += w * loss
            weight_sum += w
            level_losses[level] = loss

        # Normalize by total weight
        total_loss = total_loss / max(weight_sum, 1e-6)
        return total_loss, level_losses

    def _compute_taxonomic_score(self, species_logits, species_targets):
        """
        Compute expected taxonomic distance (competition metric approximation).

        This is an approximation of the actual competition metric:
            score = mean(D(predicted, true)) for all samples

        We use argmax predictions to match submission format.

        Args:
            species_logits: Model predictions of shape (B, num_species)
            species_targets: Ground truth labels of shape (B,)

        Returns:
            Mean taxonomic distance (lower is better)
        """
        # Get predicted species (argmax)
        preds = species_logits.argmax(dim=1)  # (B,)

        # Look up distance from distance matrix
        # distance_matrix[i, j] = distance between species i and j
        distances = self.distance_matrix[preds, species_targets]

        return distances.float().mean()

    def training_step(self, batch, batch_idx):
        """
        Single training step.

        Logs:
            - train_loss: Overall weighted loss
            - train_{level}_loss: Loss for each taxonomy level
            - train_tax_score: Taxonomic distance metric

        Args:
            batch: Tuple of (images_dict, labels_dict)
            batch_idx: Index of current batch

        Returns:
            Training loss for this batch
        """
        images, labels = batch
        outputs = self(images)
        loss, level_losses = self.hierarchical_loss(outputs, labels)

        # Log overall loss
        self.log("train_loss", loss, prog_bar=True)

        # Log per-level losses
        for lvl, lvl_loss in level_losses.items():
            self.log(f"train_{lvl}_loss", lvl_loss, prog_bar=False)

        # Log taxonomic score (competition metric approximation)
        if "species" in outputs and "species" in labels:
            tax_score = self._compute_taxonomic_score(
                outputs["species"], labels["species"]
            )
            self.log("train_tax_score", tax_score, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        """
        Single validation step with accuracy and taxonomic score tracking.

        Logs:
            - val_loss: Overall weighted loss
            - val_{level}_loss: Loss for each taxonomy level
            - val_{level}_acc: Accuracy for each level
            - val_tax_score: Taxonomic distance metric (MOST IMPORTANT)

        The val_tax_score is highlighted in prog_bar because it directly
        corresponds to the competition evaluation metric.

        Args:
            batch: Tuple of (images_dict, labels_dict)
            batch_idx: Index of current batch

        Returns:
            Validation loss for this batch
        """
        images, labels = batch
        outputs = self(images)
        loss, level_losses = self.hierarchical_loss(outputs, labels)

        # Log overall loss
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)

        # Log per-level loss and accuracy
        for lvl, lvl_loss in level_losses.items():
            self.log(f"val_{lvl}_loss", lvl_loss, sync_dist=True)

            # Compute accuracy
            preds = torch.argmax(outputs[lvl], dim=1)
            acc = (preds == labels[lvl]).float().mean()

            # Show species accuracy in progress bar (most important level)
            self.log(f"val_{lvl}_acc", acc, prog_bar=(lvl == "species"), sync_dist=True)

        # Log taxonomic score (MOST IMPORTANT METRIC!)
        # This directly corresponds to the competition evaluation
        if "species" in outputs and "species" in labels:
            tax_score = self._compute_taxonomic_score(
                outputs["species"], labels["species"]
            )
            self.log("val_tax_score", tax_score, prog_bar=True, sync_dist=True)

        # Store for epoch-end analysis
        self.val_outputs.append({"outputs": outputs, "labels": labels})
        return loss

    def on_validation_epoch_end(self):
        """Clear stored validation outputs at end of epoch."""
        self.val_outputs.clear()

    def configure_optimizers(self):
        """
        Configure optimizers with parameter-specific learning rates.

        Learning Rate Strategy:
        -----------------------
        1. Backbone parameters: base_lr × backbone_lr_scale (e.g., 0.1)
           - Pretrained weights, need careful fine-tuning
           - Lower learning rate preserves pretrained knowledge

        2. Other parameters: base_lr
           - New layers (fusion, heads) can learn faster
           - No pretrained weights to preserve

        Scheduler: CosineAnnealingWarmRestarts
            - T_0=5: First restart after 5 epochs
            - T_mult=2: Double period after each restart (5, 10, 20, ...)
            - eta_min=1e-6: Minimum learning rate

        Returns:
            Dict with optimizer and lr_scheduler configuration
        """
        # Separate parameters into backbone vs other
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
