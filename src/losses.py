"""
Taxonomic Distance-Aware Loss Functions for FathomNet 2025
============================================================

This module implements custom loss functions that leverage the taxonomic
hierarchy to improve marine species classification. The key insight is that
not all misclassifications are equally bad - confusing two species in the
same genus is less severe than confusing species from different phyla.

Competition Metric
------------------

The FathomNet 2025 competition uses a **taxonomic distance metric**:

    score = mean(distance(predicted_species, true_species))

Where distance is computed from the taxonomic tree:
- Same species: 0
- Same genus, different species: 1
- Same family, different genus: 2
- Same order, different family: 3
- Same class, different order: 4
- Same phylum, different class: 5
- Same kingdom, different phylum: 6
- Different kingdom: 7

Lower scores are better (0 = perfect).

Loss Functions Implemented
--------------------------

1. **TaxonomicDistanceLoss**: Combines cross-entropy with expected taxonomic
   distance, encouraging the model to make "safer" mistakes when uncertain.

2. **TaxonomicLabelSmoothing**: Distributes smoothing mass based on taxonomic
   similarity rather than uniformly.

3. **HierarchicalTaxonomicLoss**: Wrapper that applies these losses within
   the hierarchical training framework.

Mathematical Formulation
------------------------

Standard Cross-Entropy:
    CE(p, y) = -log(p_y)

Where p is the predicted probability distribution and y is the true class.

Taxonomic Distance Loss:
    L_tax = (1 - α) × CE(p, y) + α × E[D(pred, y)]

Where:
    - α is the weighting between CE and distance (typically 0.3)
    - E[D(pred, y)] = Σ_i p_i × distance(i, y)
    - The expected distance term penalizes spreading probability mass
      to taxonomically distant classes

Taxonomic Label Smoothing:
    Instead of uniform smoothing: q_i = (1-ε)·I(i=y) + ε/K
    We use similarity-weighted smoothing: q_i = (1-ε)·I(i=y) + ε·sim(i,y)/Z

Where sim(i,y) = max_dist - distance(i,y) measures taxonomic similarity.

Experimental Results
--------------------

| Loss Type                | Val Loss | Private Score | Improvement |
|-------------------------|----------|---------------|-------------|
| Standard CE             | 0.82     | 2.06          | Baseline    |
| CE + Distance (α=0.3)   | 0.78     | 2.00          | +0.06       |
| Taxonomic Smoothing     | 0.79     | 2.02          | +0.04       |
| Both                    | 0.77     | TBD           | Testing     |

The taxonomic distance loss improved our private leaderboard score from
2.06 to 2.00, which is a meaningful improvement given the competition
range (1st place: ~1.38, 8th place: 2.00).

Implementation Notes
--------------------

The distance matrix is loaded from a CSV file with species names as both
row and column indices. We align the matrix with the model's class order
at initialization to ensure efficient lookup during training.

Author: Dan Zimmerman
Course: CAP6415 - Computer Vision (Fall 2025)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Dict, List, Optional


class TaxonomicDistanceLoss(nn.Module):
    """
    Loss function that penalizes predictions based on taxonomic distance.

    This loss combines standard cross-entropy with an expected taxonomic
    distance term. The intuition is that when the model is uncertain, it
    should hedge toward taxonomically similar classes rather than spreading
    probability mass uniformly.

    Mathematical Formulation
    ------------------------

    For a single sample with true class y and predicted probabilities p:

        L = (1 - α) × CE(p, y) + α × E_p[D(·, y)]

    Where:
        CE(p, y) = -log(p_y)                    # Cross-entropy
        E_p[D(·, y)] = Σ_i p_i × D(i, y)       # Expected taxonomic distance

    The expected distance term is differentiable w.r.t. the logits through
    the softmax probabilities, allowing gradient-based optimization.

    Gradient Analysis
    -----------------

    ∂L/∂z_i = (1-α) × (p_i - I(i=y)) + α × p_i × (D(i,y) - E_p[D])

    The first term is the standard cross-entropy gradient.
    The second term pushes probability mass away from classes with
    above-average distance and toward classes with below-average distance.

    Parameters
    ----------
    distance_matrix_path : str
        Path to CSV file containing the taxonomic distance matrix.
        Expected format: rows and columns indexed by species names,
        values are integer distances (0-7 for standard taxonomy).

    class_names : List[str]
        List of species names in the order used by the model.
        Must match the model's output dimension.

    alpha : float, default=0.5
        Weight for the distance penalty term.
        - α=0: Pure cross-entropy (ignore taxonomy)
        - α=1: Pure distance minimization (ignore likelihood)
        - α=0.3: Recommended value (primarily CE with distance regularization)

    temperature : float, default=1.0
        Temperature for softmax when computing expected distance.
        Higher temperature = softer probabilities = more conservative.

    normalize_distances : bool, default=True
        Whether to normalize distances to [0, 1] range.
        Recommended for stable training when combining with CE.

    Attributes
    ----------
    distances : torch.Tensor
        Registered buffer containing the distance matrix (C, C).
        distances[i, j] = taxonomic distance between class i and class j.

    Example
    -------
    >>> loss_fn = TaxonomicDistanceLoss(
    ...     "taxonomic_distance_matrix.csv",
    ...     ["Aurelia aurita", "Chrysaora fuscescens", ...],
    ...     alpha=0.3
    ... )
    >>> logits = model(images)["species"]  # (B, 80)
    >>> targets = batch["labels"]["species"]  # (B,)
    >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        distance_matrix_path: str,
        class_names: List[str],
        alpha: float = 0.5,
        temperature: float = 1.0,
        normalize_distances: bool = True,
    ):
        """
        Initialize the taxonomic distance loss.

        This constructor loads the distance matrix from disk and aligns it
        with the model's class ordering. Any species not found in the matrix
        receives the maximum distance to all other classes.
        """
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

        # ──────────────────────────────────────────────────────────────────────
        # Load and Align Distance Matrix
        # ──────────────────────────────────────────────────────────────────────
        # The distance matrix CSV has species names as row/column indices.
        # We need to reorder it to match the model's class order.
        # ──────────────────────────────────────────────────────────────────────

        dm = pd.read_csv(distance_matrix_path, index_col=0)

        n_classes = len(class_names)
        distances = torch.zeros(n_classes, n_classes)

        for i, name_i in enumerate(class_names):
            for j, name_j in enumerate(class_names):
                if name_i in dm.index and name_j in dm.columns:
                    # Found in matrix - use actual distance
                    distances[i, j] = dm.loc[name_i, name_j]
                elif i == j:
                    # Same class - distance is 0
                    distances[i, j] = 0.0
                else:
                    # Not found - use maximum distance (conservative)
                    distances[i, j] = dm.values.max()

        # ──────────────────────────────────────────────────────────────────────
        # Normalization
        # ──────────────────────────────────────────────────────────────────────
        # Normalizing distances to [0, 1] ensures the distance term is on a
        # similar scale to cross-entropy, making α more interpretable.
        # Without normalization, you'd need very different α for different
        # taxonomic depths.
        # ──────────────────────────────────────────────────────────────────────

        if normalize_distances:
            max_dist = distances.max()
            if max_dist > 0:
                distances = distances / max_dist

        # Register as buffer (moves to GPU with model, but not a parameter)
        self.register_buffer("distances", distances)

        # Standard cross-entropy (per-sample, for combining with distance)
        self.ce_loss = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined cross-entropy and taxonomic distance loss.

        Parameters
        ----------
        logits : torch.Tensor
            Raw model outputs (before softmax) with shape (B, C).
            B = batch size, C = number of classes.

        targets : torch.Tensor
            Ground truth class indices with shape (B,).
            Values in range [0, C-1].

        Returns
        -------
        torch.Tensor
            Scalar loss tensor (mean over batch).

        Implementation Details
        ----------------------

        1. Compute standard CE loss per sample
        2. Compute softmax probabilities (optionally with temperature)
        3. Index into distance matrix to get distances from each target
        4. Compute expected distance as dot product of probs and distances
        5. Combine losses with weighting α
        """
        # ──────────────────────────────────────────────────────────────────────
        # Step 1: Standard Cross-Entropy
        # ──────────────────────────────────────────────────────────────────────
        ce = self.ce_loss(logits, targets)  # (B,)

        # ──────────────────────────────────────────────────────────────────────
        # Step 2: Compute Probabilities (with optional temperature scaling)
        # ──────────────────────────────────────────────────────────────────────
        # Temperature > 1 makes the distribution softer, which increases the
        # expected distance (more mass on incorrect classes).
        # Temperature < 1 makes it sharper (closer to argmax).
        # ──────────────────────────────────────────────────────────────────────
        probs = F.softmax(logits / self.temperature, dim=1)  # (B, C)

        # ──────────────────────────────────────────────────────────────────────
        # Step 3: Get Distance from Each Target
        # ──────────────────────────────────────────────────────────────────────
        # distances[targets] indexes into the distance matrix using the target
        # class for each sample, giving us the distance from that target to
        # every other class.
        #
        # Shape: (B,) -> (B, C)
        # target_distances[b, c] = distance between class c and targets[b]
        # ──────────────────────────────────────────────────────────────────────
        target_distances = self.distances[targets]  # (B, C)

        # ──────────────────────────────────────────────────────────────────────
        # Step 4: Expected Taxonomic Distance
        # ──────────────────────────────────────────────────────────────────────
        # E[D] = Σ_c p_c × D(c, y)
        #
        # This is a differentiable approximation of the actual taxonomic
        # distance we'd incur by sampling from the predicted distribution.
        # ──────────────────────────────────────────────────────────────────────
        expected_distance = (probs * target_distances).sum(dim=1)  # (B,)

        # ──────────────────────────────────────────────────────────────────────
        # Step 5: Combine Losses
        # ──────────────────────────────────────────────────────────────────────
        # L = (1-α)×CE + α×E[D]
        #
        # When α=0: Pure cross-entropy (maximum likelihood)
        # When α=1: Pure distance minimization (ignores likelihood)
        # When α=0.3: Primarily likelihood, regularized by distance
        # ──────────────────────────────────────────────────────────────────────
        loss = (1 - self.alpha) * ce + self.alpha * expected_distance

        return loss.mean()


class TaxonomicLabelSmoothing(nn.Module):
    """
    Label smoothing that distributes mass based on taxonomic similarity.

    Standard label smoothing spreads ε probability mass uniformly across
    all incorrect classes. This ignores the taxonomic structure - spreading
    mass to a distantly related species is just as likely as a close relative.

    Taxonomic label smoothing instead distributes the smoothing mass
    according to taxonomic similarity. Classes in the same genus get more
    mass than classes in different families, etc.

    Mathematical Formulation
    ------------------------

    Standard label smoothing (uniform):
        q_i = (1 - ε) × I(i = y) + ε / K

    Taxonomic label smoothing (similarity-weighted):
        q_i = (1 - ε) × I(i = y) + ε × sim(i, y) / Z

    Where:
        sim(i, y) = max_distance - distance(i, y)   # Similarity (higher = closer)
        Z = Σ_{j≠y} sim(j, y)                       # Normalization constant

    Loss (soft cross-entropy):
        L = -Σ_i q_i × log(p_i)

    Intuition
    ---------

    Consider predicting a species "Aurelia aurita" (moon jellyfish):
    - Uniform smoothing: Equal mass to sharks, fish, other jellies
    - Taxonomic smoothing: More mass to other Aurelia species, then other
      Scyphozoa, then other Cnidaria, etc.

    This provides a "softer" training signal that reflects the actual
    structure of the problem - the model gets partial credit for being
    "close" even when wrong.

    Parameters
    ----------
    distance_matrix_path : str
        Path to CSV file with taxonomic distances.

    class_names : List[str]
        Species names in model output order.

    smoothing : float, default=0.1
        Total mass to distribute to non-target classes.
        Typical values: 0.1 (light smoothing) to 0.3 (heavy smoothing).

    temperature : float, default=2.0
        Temperature for converting similarities to smoothing distribution.
        Higher temperature = more uniform distribution.
        Lower temperature = more mass on closest relatives.

    Example
    -------
    >>> loss_fn = TaxonomicLabelSmoothing(
    ...     "taxonomic_distance_matrix.csv",
    ...     species_names,
    ...     smoothing=0.1,
    ...     temperature=2.0
    ... )
    >>> logits = model(images)["species"]
    >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        distance_matrix_path: str,
        class_names: List[str],
        smoothing: float = 0.1,
        temperature: float = 2.0,
    ):
        """
        Initialize taxonomic label smoothing.

        The smoothing distribution is precomputed at initialization for
        efficiency. For each possible target class, we compute a distribution
        over non-target classes based on taxonomic similarity.
        """
        super().__init__()
        self.smoothing = smoothing

        # Load and process distance matrix
        dm = pd.read_csv(distance_matrix_path, index_col=0)

        n_classes = len(class_names)
        distances = torch.zeros(n_classes, n_classes)

        for i, name_i in enumerate(class_names):
            for j, name_j in enumerate(class_names):
                if name_i in dm.index and name_j in dm.columns:
                    distances[i, j] = dm.loc[name_i, name_j]
                elif i == j:
                    distances[i, j] = 0.0
                else:
                    distances[i, j] = dm.values.max()

        # ──────────────────────────────────────────────────────────────────────
        # Convert Distances to Similarities
        # ──────────────────────────────────────────────────────────────────────
        # similarity = max_distance - distance
        # This inverts the relationship: higher similarity = closer in tree
        # ──────────────────────────────────────────────────────────────────────
        max_dist = distances.max()
        similarities = (max_dist - distances) / max_dist  # Normalize to [0, 1]

        # ──────────────────────────────────────────────────────────────────────
        # Compute Smoothing Distribution
        # ──────────────────────────────────────────────────────────────────────
        # For each target class y, we want a distribution over non-y classes
        # that gives more weight to similar classes.
        #
        # We use softmax over similarities, but set the diagonal to -inf
        # to exclude the target class from receiving smoothing mass.
        # ──────────────────────────────────────────────────────────────────────
        similarities_no_diag = similarities.clone()
        similarities_no_diag.fill_diagonal_(float("-inf"))

        # Softmax converts similarities to a proper distribution
        # Temperature controls sharpness: higher = more uniform
        smooth_dist = F.softmax(similarities_no_diag / temperature, dim=1)

        # Register as buffer (smooth_dist[y, :] is the smoothing distribution
        # to use when the target is class y)
        self.register_buffer("smooth_dist", smooth_dist)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy with taxonomic label smoothing.

        Parameters
        ----------
        logits : torch.Tensor
            Model outputs with shape (B, C).

        targets : torch.Tensor
            Ground truth class indices with shape (B,).

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.

        Implementation
        --------------

        1. Create one-hot encoding of targets
        2. Look up smoothing distribution for each target
        3. Blend: q = (1-ε)×one_hot + ε×smooth_dist
        4. Compute soft cross-entropy: L = -Σ q_i × log(p_i)
        """
        n_classes = logits.size(1)

        # ──────────────────────────────────────────────────────────────────────
        # Step 1: One-Hot Encoding
        # ──────────────────────────────────────────────────────────────────────
        one_hot = F.one_hot(targets, n_classes).float()  # (B, C)

        # ──────────────────────────────────────────────────────────────────────
        # Step 2: Get Smoothing Distribution for Each Target
        # ──────────────────────────────────────────────────────────────────────
        # smooth_dist[targets] indexes using the target class for each sample
        # Result: (B, C) where smooth_mass[b, :] is the smoothing distribution
        # for sample b's target class
        # ──────────────────────────────────────────────────────────────────────
        smooth_mass = self.smooth_dist[targets]  # (B, C)

        # ──────────────────────────────────────────────────────────────────────
        # Step 3: Create Smoothed Target Distribution
        # ──────────────────────────────────────────────────────────────────────
        # q_i = (1-ε)×I(i=y) + ε×smooth_dist[y, i]
        #
        # The target class gets probability (1-ε) plus any mass it would
        # receive from smoothing (which is 0 since we set diagonal to -inf).
        # All other classes get mass proportional to their similarity.
        # ──────────────────────────────────────────────────────────────────────
        smoothed = (1 - self.smoothing) * one_hot + self.smoothing * smooth_mass

        # ──────────────────────────────────────────────────────────────────────
        # Step 4: Soft Cross-Entropy
        # ──────────────────────────────────────────────────────────────────────
        # L = -Σ_i q_i × log(p_i)
        #
        # This is equivalent to KL divergence (up to an additive constant)
        # between the smoothed target distribution and the predicted distribution.
        # ──────────────────────────────────────────────────────────────────────
        log_probs = F.log_softmax(logits, dim=1)
        loss = -(smoothed * log_probs).sum(dim=1)

        return loss.mean()


class HierarchicalTaxonomicLoss(nn.Module):
    """
    Combined hierarchical loss with taxonomic distance awareness.

    This is a convenience wrapper that applies taxonomic-aware losses
    specifically to the species level while using standard cross-entropy
    for higher taxonomic levels. The motivation is that:

    1. The competition metric is computed at the species level only
    2. Higher levels (kingdom, phylum, etc.) have fewer classes and
       simpler structure - taxonomic smoothing is less beneficial
    3. Separating loss types allows different hyperparameters per level

    Architecture
    ------------

    For each taxonomic level:
    - Kingdom through Genus: Standard cross-entropy with label smoothing
    - Species: Taxonomic distance loss and/or taxonomic label smoothing

    The total loss is a weighted sum across levels:

        L_total = Σ_l w_l × L_l

    Where w_l are the hierarchy weights (typically higher for species).

    Parameters
    ----------
    distance_matrix_path : str
        Path to species-level distance matrix CSV.

    class_names : Dict[str, List[str]]
        Dictionary mapping level names to lists of class names.
        Only "species" is used for taxonomic losses.

    hierarchy_weights : Dict[str, float]
        Weight for each taxonomic level in the combined loss.

    loss_type : str, default="distance"
        Which taxonomic loss to use at species level:
        - "distance": TaxonomicDistanceLoss
        - "smooth": TaxonomicLabelSmoothing
        - "both": Average of both
        - "ce": Standard cross-entropy (no taxonomic awareness)

    alpha : float, default=0.3
        Weight for distance term in TaxonomicDistanceLoss.

    smoothing : float, default=0.1
        Smoothing factor for TaxonomicLabelSmoothing.

    Example
    -------
    >>> loss_fn = HierarchicalTaxonomicLoss(
    ...     "taxonomic_distance_matrix.csv",
    ...     class_names={"kingdom": [...], ..., "species": [...]},
    ...     hierarchy_weights={"kingdom": 0.5, ..., "species": 1.5},
    ...     loss_type="distance",
    ...     alpha=0.3
    ... )
    >>> outputs = model(images)  # Dict[str, Tensor]
    >>> labels = batch["labels"]  # Dict[str, Tensor]
    >>> total_loss, level_losses = loss_fn(outputs, labels)
    """

    def __init__(
        self,
        distance_matrix_path: str,
        class_names: Dict[str, List[str]],
        hierarchy_weights: Dict[str, float],
        loss_type: str = "distance",  # "distance", "smooth", "both", "ce"
        alpha: float = 0.3,
        smoothing: float = 0.1,
    ):
        """
        Initialize the hierarchical taxonomic loss.

        The species-level taxonomic loss is only created if a valid distance
        matrix path is provided and loss_type is not "ce".
        """
        super().__init__()
        self.hierarchy_weights = hierarchy_weights
        self.levels = list(class_names.keys())

        # Species names for taxonomic loss
        self.species_names = class_names.get("species", [])

        # ──────────────────────────────────────────────────────────────────────
        # Create Species-Level Taxonomic Losses
        # ──────────────────────────────────────────────────────────────────────

        if loss_type in ["distance", "both"] and self.species_names:
            self.distance_loss = TaxonomicDistanceLoss(
                distance_matrix_path,
                self.species_names,
                alpha=alpha,
            )
        else:
            self.distance_loss = None

        if loss_type in ["smooth", "both"] and self.species_names:
            self.smooth_loss = TaxonomicLabelSmoothing(
                distance_matrix_path,
                self.species_names,
                smoothing=smoothing,
            )
        else:
            self.smooth_loss = None

        # Standard CE for non-species levels (and fallback)
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> tuple:
        """
        Compute weighted hierarchical loss with taxonomic awareness.

        Parameters
        ----------
        outputs : Dict[str, torch.Tensor]
            Model outputs for each taxonomic level.
            Keys: "kingdom", "phylum", ..., "species"
            Values: Logit tensors with shape (B, num_classes)

        targets : Dict[str, torch.Tensor]
            Ground truth labels for each level.
            Same keys as outputs, values are (B,) integer tensors.

        Returns
        -------
        total_loss : torch.Tensor
            Scalar weighted average loss across all levels.

        level_losses : Dict[str, torch.Tensor]
            Dictionary of per-level losses for logging and analysis.
        """
        total_loss = 0.0
        level_losses = {}
        weight_sum = 0.0

        for level in self.levels:
            if level not in outputs or level not in targets:
                continue

            logits = outputs[level]
            target = targets[level]
            weight = self.hierarchy_weights.get(level, 1.0)

            # ──────────────────────────────────────────────────────────────────
            # Level-Specific Loss Selection
            # ──────────────────────────────────────────────────────────────────
            if level == "species" and (self.distance_loss or self.smooth_loss):
                # Use taxonomic-aware loss for species
                if self.distance_loss and self.smooth_loss:
                    # Average both loss types
                    loss = 0.5 * self.distance_loss(logits, target) + \
                           0.5 * self.smooth_loss(logits, target)
                elif self.distance_loss:
                    loss = self.distance_loss(logits, target)
                else:
                    loss = self.smooth_loss(logits, target)
            else:
                # Standard CE for higher taxonomic levels
                loss = self.ce_loss(logits, target)

            total_loss += weight * loss
            weight_sum += weight
            level_losses[level] = loss

        # Normalize by total weight
        total_loss = total_loss / max(weight_sum, 1e-6)
        return total_loss, level_losses


def compute_expected_taxonomic_score(
    probs: torch.Tensor,
    targets: torch.Tensor,
    distance_matrix: torch.Tensor,
) -> torch.Tensor:
    """
    Compute expected taxonomic distance score for validation.

    This function approximates the competition metric using the model's
    hard predictions (argmax) rather than expected value. It's useful for
    monitoring during training to see how well we're optimizing the actual
    evaluation criterion.

    Parameters
    ----------
    probs : torch.Tensor
        Predicted probability distribution with shape (B, C).
        Note: We use argmax, so only the ranking matters.

    targets : torch.Tensor
        Ground truth class indices with shape (B,).

    distance_matrix : torch.Tensor
        Taxonomic distance matrix with shape (C, C).
        Should NOT be normalized (use actual integer distances 0-7).

    Returns
    -------
    torch.Tensor
        Scalar tensor containing the mean taxonomic distance.

    Note
    ----
    This is NOT differentiable because of argmax. Use TaxonomicDistanceLoss
    for training. This function is for evaluation/logging only.
    """
    # Get hard predictions (argmax of probabilities)
    preds = probs.argmax(dim=1)

    # Look up distances between predictions and ground truth
    # distance_matrix[preds, targets] selects element (pred[i], target[i])
    # for each sample i
    distances = distance_matrix[preds, targets]

    return distances.float().mean()
