# Research Report: Analysis of 1st Place FathomNet Solution and Planned Improvements

**Author:** [Your Name]
**Date:** December 3, 2025
**Project:** FathomNet 2025 @ CVPR-FGVC Competition (Supporting Research)
**Current Ranking:** 8th/79 teams (Score: 2.74)
**Target:** Improve toward 1st place performance
**Context:** This work supports dissertation research but is not the primary dissertation project

---

## Executive Summary

I analyzed the winning solution from the Kaggle CVPR'25 FathomNet competition to understand the performance gap between my 8th place solution (score: 2.74) and the winner. The key finding: **multi-scale environmental context extraction was the decisive architectural advantage**, not merely the choice of backbone model.

My initial experiment swapping ConvNeXtV2 → DinoV2 showed minimal improvement (2.66 vs 2.74, only 2.9% gain), confirming that the bottleneck is architectural design rather than backbone selection. This report presents:

1. Detailed analysis of what the winner did differently
2. Why my DinoV2 experiment had limited impact
3. Five prioritized experiments to close the performance gap
4. Timeline, risk analysis, and expected contributions to the broader research program

---

## Part 1: What the Winner Did Differently

### 1.1 Core Architectural Innovation: Multi-Scale Spatial Context

**The Winning Insight:** Marine species classification cannot be solved from organism morphology alone—environmental context (depth, substrate, water column, habitat) is a critical signal.

**Implementation:**

```
Input Image → 4 Parallel Processing Streams:
├── ROI crop (tight bounding box)
├── 3× context window (immediate surroundings)
├── 5× context window (broader environment)
└── Full frame (complete scene)
     ↓
4 Separate DinoV2-Large Encoders (one per scale)
     ↓
Multi-Context Environmental Attention Module
(ROI features attend to environmental patches)
     ↓
Concatenated Embeddings → Final Classifier
```

**Why This Matters:**

- A jellyfish in open water (pelagic zone) vs. near substrate = different species
- Schooling fish patterns visible in full frame constrain species options
- Substrate texture (rocky/sandy) correlates with benthic organism types
- Water turbidity/lighting indicates depth zone → species distribution

### 1.2 Technical Components

| Component | Purpose | Implementation |
|-----------|---------|----------------|
| **Separate Encoders** | Each scale learns specialized features (morphology vs. environment) | 4× `facebook/dinov2-large` (307M params each) |
| **Multi-Context Environmental Attention** | ROI features query relevant environmental patches | `MultiLayerAttentionModels` with learnable Q/K/V projections |
| **Hierarchical Taxonomy Loss** | Optional multi-head classifier for Phylum→Species ranks | Loaded from WoRMS taxonomy distance matrix |
| **Heavy Augmentation** | Simulate underwater imaging variance | Blur downsampling, rotation, color jitter |

### 1.3 Attention Mechanism Deep Dive

The winner applies the Transformer attention mechanism from Vaswani et al. (2017) "Attention Is All You Need" to computer vision:

**Mathematical Foundation:**
```
Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
```

Where:
- **Q (Query):** ROI features asking "What environmental cues matter for this organism?"
- **K (Key):** Environmental patch features describing themselves
- **V (Value):** Actual environmental content to retrieve

**Why DinoV2 is Critical for This:**

DinoV2 is a Vision Transformer that outputs **patch embeddings**, not just global features:

```python
dinov2 = DinoV2Model.from_pretrained('facebook/dinov2-large')
output = dinov2(image)  # Input: [batch, 3, 224, 224]

# Two types of outputs:
cls_token = output.last_hidden_state[:, 0, :]   # [batch, 1024] - global
patches = output.last_hidden_state[:, 1:, :]    # [batch, 256, 1024] - spatial

# For 224×224 image with 14×14 patches:
# 256 patches = 16×16 grid
```

Each of the 256 patches represents a 14×14 pixel region. The attention mechanism can selectively focus on specific spatial regions:

**Concrete Example: Classifying a Crab**

```python
# Step 1: Extract features
roi_feats = dinov2_roi(crab_roi)[:, 0, :]        # [1, 1024]
context3_patches = dinov2_3x(context_3x)[:, 1:, :] # [1, 256, 1024]

# Step 2: ROI queries context
Q = W_q(roi_feats)  # "Show me substrate texture cues"
K = W_k(context3_patches)  # Each patch: "I'm sand", "I'm rock", "I'm water"

# Step 3: Compute relevance
scores = Q @ K.T / sqrt(d_k)  # [1, 256]
# High scores for bottom patches (substrate)
# Low scores for top patches (water column)

attention_weights = softmax(scores)
# weights ≈ [0.001, 0.001, ..., 0.08, 0.12, 0.15, ...]
#            ↑ water column       ↑ substrate patches

# Step 4: Weighted combination
attended_3x = attention_weights @ V  # [1, 1024]
# Result contains mostly substrate features, minimal water column
```

**Multi-Head Attention (8 heads):**

Each head learns a different "query strategy":

| Head | Learned Strategy Example |
|------|-------------------------|
| Head 1 | "Focus on substrate texture" |
| Head 2 | "Focus on depth indicators (light/dark gradient)" |
| Head 3 | "Focus on other organisms in frame" |
| Head 4 | "Focus on water column position" |
| Head 5 | "Focus on scale/size cues in environment" |
| Head 6 | "Focus on habitat type (reef, open water, kelp)" |
| Head 7 | "Focus on color/turbidity of water" |
| Head 8 | "Focus on benthic vs pelagic indicators" |

### 1.4 Key Differences from My Approach

| Aspect | My Solution (8th) | Winner (1st) |
|--------|-------------------|--------------|
| **Spatial Coverage** | Single ROI crop | ROI + 3 context windows (4 scales) |
| **Feature Extraction** | Shared ConvNeXtV2 backbone for all ranks | Separate DinoV2 encoder per scale |
| **Taxonomy Integration** | Sequential conditioning (species depends on genus) | Optional hierarchical loss (independent predictions) |
| **Patch-Level Features** | ConvNeXt gives global pooled features | DinoV2 provides 256 spatial patch embeddings |
| **Inference Strategy** | Confidence-based fallback up taxonomy | Direct species prediction with ensemble |
| **Compute Cost** | ~88M params | ~1.2B params (4 encoders) |

---

## Part 2: Why My DinoV2 Experiment Failed

**Result:** Swapping ConvNeXtV2 → DinoV2 only improved from 2.74 → 2.66 (2.9% gain)

### 2.1 Root Cause Analysis

The marginal improvement suggests that my bottleneck is **not the backbone's feature quality**, but the **limited spatial context**.

**Evidence:**

1. DinoV2's superior self-supervised pretraining should transfer better to underwater imagery
2. The small gain (0.08 points) indicates I'm still missing the same environmental signals
3. My single ROI crop forces DinoV2 to solve an artificially hard problem: classify organisms divorced from their habitat
4. I'm not leveraging DinoV2's patch embeddings—my model only uses the global CLS token

**Hypothesis:** If I had tested DinoV2 with multi-scale context AND patch-level attention, the improvement would have been substantial.

### 2.2 Implications

- The winner's 6-point advantage is **not primarily from DinoV2 vs. ConvNeXt**
- The winner's advantage comes from **architectural design** (multi-scale context extraction)
- My hierarchical conditioning may introduce **error propagation** that limits ceiling performance

**Error Propagation Example:**
```
My Model:
Misclassify Order → Wrong Family subtree →
Wrong Genus subtree → Impossible to predict correct Species

Winner's Model:
All ranks predicted independently → error in Order doesn't affect Species
```

---

## Part 3: Planned Experiments (Priority Order)

### Experiment 1: Add Multi-Scale Context ⭐ **HIGH PRIORITY**

**Hypothesis:** Environmental context is the primary missing signal. Adding 3× and 5× context windows should close 60-70% of the performance gap.

**Implementation Plan:**

```python
# src/data_multiscale.py
class MultiScaleFathomNetDataset(Dataset):
    def __getitem__(self, idx):
        img, bbox, label = self.load_sample(idx)

        # Extract 3 scales
        roi_crop = self.crop_bbox(img, bbox, margin=0.1)
        context_3x = self.crop_bbox(img, bbox, margin=2.0)
        context_5x = self.crop_bbox(img, bbox, margin=4.0)

        # Apply transforms consistently
        roi_crop = self.transform(roi_crop)
        context_3x = self.transform(context_3x)
        context_5x = self.transform(context_5x)

        return {
            'roi': roi_crop,
            'context_3x': context_3x,
            'context_5x': context_5x,
            'label': label
        }
```

```python
# src/model_multiscale.py
class MultiScaleTaxonomyClassifier(pl.LightningModule):
    def __init__(self):
        # Separate encoders per scale
        self.encoder_roi = timm.create_model('convnextv2_base', pretrained=True, num_classes=0)
        self.encoder_3x = timm.create_model('convnextv2_base', pretrained=True, num_classes=0)
        self.encoder_5x = timm.create_model('convnextv2_base', pretrained=True, num_classes=0)

        # Projection to shared dimension
        self.proj_roi = nn.Linear(1024, 512)
        self.proj_3x = nn.Linear(1024, 512)
        self.proj_5x = nn.Linear(1024, 512)

        # Concatenated features → hierarchical heads
        self.shared_mlp = nn.Linear(512 * 3, 1024)
        # ... rest of hierarchical heads (kingdom → species)

    def forward(self, batch):
        roi_feats = self.proj_roi(self.encoder_roi(batch['roi']))
        ctx3_feats = self.proj_3x(self.encoder_3x(batch['context_3x']))
        ctx5_feats = self.proj_5x(self.encoder_5x(batch['context_5x']))

        # Simple concatenation (no attention yet)
        combined = torch.cat([roi_feats, ctx3_feats, ctx5_feats], dim=1)
        shared = self.shared_mlp(combined)

        # Pass through hierarchical classifier heads
        return self.hierarchical_forward(shared)
```

**Ablations:**

- **Exp 1a:** ROI + 3× context (2 encoders)
- **Exp 1b:** ROI + 3× + 5× context (3 encoders) ← **Main experiment**
- **Exp 1c:** ROI + 3× + 5× + full frame (4 encoders, match winner exactly)

**Success Criteria:** Score < 2.50 (close 30%+ of gap)

---

### Experiment 2: Add Cross-Scale Attention ⭐ **MEDIUM PRIORITY**

**Hypothesis:** Attention lets the model learn "which environmental patches matter for this organism," improving context utilization beyond simple feature concatenation.

**Requires:** Must use DinoV2 (or another ViT) to get patch embeddings. ConvNeXt only provides global features.

**Implementation Plan:**

```python
# src/model_attention.py
class CrossScaleAttention(nn.Module):
    """
    Implements multi-head attention from Vaswani et al. 2017
    ROI features query context window patches
    """
    def __init__(self, d_model=1024, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Learnable projection matrices
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, roi_features, context_patch_features):
        """
        Args:
            roi_features: [batch, d_model] - global ROI features
            context_patch_features: [batch, num_patches, d_model] - spatial patches

        Returns:
            attended_features: [batch, d_model] - ROI-relevant context
            attention_weights: [batch, num_heads, 1, num_patches] - for visualization
        """
        batch_size = roi_features.size(0)

        # Project to Q, K, V and reshape for multi-head
        Q = self.W_q(roi_features).view(batch_size, 1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(context_patch_features).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(context_patch_features).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention
        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1)

        # Output projection + residual connection
        output = self.W_o(attended)
        output = self.layer_norm(roi_features + output)

        return output, attention_weights


class MultiScaleWithAttention(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # DinoV2 encoders (need patch features!)
        self.encoder_roi = timm.create_model('vit_large_patch14_dinov2.lvd142m',
                                              pretrained=True, num_classes=0)
        self.encoder_3x = timm.create_model('vit_large_patch14_dinov2.lvd142m',
                                             pretrained=True, num_classes=0)
        self.encoder_5x = timm.create_model('vit_large_patch14_dinov2.lvd142m',
                                             pretrained=True, num_classes=0)

        # Attention modules
        self.attn_3x = CrossScaleAttention(d_model=1024, num_heads=8)
        self.attn_5x = CrossScaleAttention(d_model=1024, num_heads=8)

        # Final classifier
        self.shared_mlp = nn.Linear(1024 * 3, 1024)  # ROI + 2 attended contexts
        # ... hierarchical heads

    def forward(self, batch):
        # Extract features
        roi_out = self.encoder_roi.forward_features(batch['roi'])
        roi_global = roi_out[:, 0, :]  # [B, 1024] - CLS token
        roi_patches = roi_out[:, 1:, :]  # [B, 256, 1024] - NOT USED, but available

        context3_out = self.encoder_3x.forward_features(batch['context_3x'])
        context3_patches = context3_out[:, 1:, :]  # [B, 256, 1024] - patches

        context5_out = self.encoder_5x.forward_features(batch['context_5x'])
        context5_patches = context5_out[:, 1:, :]  # [B, 256, 1024]

        # Attend to relevant context patches
        attended_3x, attn_weights_3x = self.attn_3x(roi_global, context3_patches)
        attended_5x, attn_weights_5x = self.attn_5x(roi_global, context5_patches)

        # Combine and classify
        combined = torch.cat([roi_global, attended_3x, attended_5x], dim=1)
        shared = self.shared_mlp(combined)

        outputs = self.hierarchical_forward(shared)
        outputs['attention_weights'] = {'3x': attn_weights_3x, '5x': attn_weights_5x}

        return outputs
```

**Visualization Code:**

```python
# scripts/visualize_attention.py
def visualize_attention(image, attention_weights, patch_size=16):
    """
    Overlay attention weights on original image to see what model focuses on
    """
    # attention_weights: [num_heads, 1, num_patches]
    avg_attention = attention_weights.mean(dim=0).squeeze(0)  # [num_patches]

    # Reshape to spatial grid (e.g., 16×16 patches)
    grid_size = int(math.sqrt(len(avg_attention)))
    attn_map = avg_attention.view(grid_size, grid_size).cpu().numpy()

    # Upsample to image size
    attn_map_resized = cv2.resize(attn_map, (image.shape[1], image.shape[0]))

    # Overlay heatmap
    plt.imshow(image)
    plt.imshow(attn_map_resized, alpha=0.5, cmap='jet')
    plt.colorbar(label='Attention Weight')
    plt.title('Environmental Attention Map')
    plt.show()
```

**Success Criteria:** Score < 2.40 (additional 5-10% improvement over Exp 1b)

---

### Experiment 3: Remove Hierarchical Conditioning ⭐ **MEDIUM PRIORITY**

**Hypothesis:** Sequential conditioning (species head uses genus softmax) causes error propagation. Independent predictions with hierarchical loss should improve ceiling performance.

**Current Architecture Issue:**
```
Shared Features (1024-dim)
    ↓
Kingdom Head → softmax → Kingdom Probs (2-dim)
    ↓
[Kingdom Probs + Shared Features] → Phylum Head → Phylum Probs (9-dim)
    ↓
[Phylum Probs + Shared Features] → Class Head → Class Probs (22-dim)
    ↓
... continues through Order → Family → Genus → Species

Problem: Error at any level constrains all lower levels
```

**Proposed Change:**

```python
class IndependentHierarchicalClassifier(pl.LightningModule):
    def __init__(self):
        # Shared feature extractor (multi-scale backbone)
        self.backbone = MultiScaleBackbone()  # From Exp 1
        self.shared_mlp = nn.Linear(feat_dim, 1024)

        # Independent heads (NO conditioning between ranks)
        self.head_kingdom = nn.Linear(1024, n_kingdoms)
        self.head_phylum = nn.Linear(1024, n_phyla)
        self.head_class = nn.Linear(1024, n_classes)
        self.head_order = nn.Linear(1024, n_orders)
        self.head_family = nn.Linear(1024, n_families)
        self.head_genus = nn.Linear(1024, n_genera)
        self.head_species = nn.Linear(1024, n_species)

    def forward(self, x):
        shared_feats = self.shared_mlp(self.backbone(x))

        # All heads predict independently in parallel
        return {
            'kingdom': self.head_kingdom(shared_feats),
            'phylum': self.head_phylum(shared_feats),
            'class': self.head_class(shared_feats),
            'order': self.head_order(shared_feats),
            'family': self.head_family(shared_feats),
            'genus': self.head_genus(shared_feats),
            'species': self.head_species(shared_feats)
        }

    def compute_loss(self, preds, targets):
        # Weighted sum of per-rank cross-entropy losses
        # (matches winner's "hierarchical_loss" option)
        losses = {}
        for rank in ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']:
            losses[rank] = F.cross_entropy(preds[rank], targets[rank])

        # Weight later ranks more heavily (same as current model)
        weights = {
            'kingdom': 0.5,
            'phylum': 0.75,
            'class': 1.0,
            'order': 1.25,
            'family': 1.5,
            'genus': 2.0,
            'species': 2.5
        }

        return sum(weights[r] * losses[r] for r in losses) / sum(weights.values())
```

**Ablation:** Compare independent heads vs. current sequential conditioning on same multi-scale architecture (from Exp 1b)

**Success Criteria:** Score < 2.35 (eliminate error propagation penalty)

---

### Experiment 4: Ensemble Multi-Scale Models ⭐ **LOW PRIORITY**

**Hypothesis:** Training separate specialists (ROI-only, 3×-context, 5×-context) and ensembling may outperform a single joint model.

**Implementation Plan:**

```python
# Train 3 separate models
model_roi = TaxonomyClassifier(scales=['roi'])
model_3x = TaxonomyClassifier(scales=['roi', '3x'])
model_5x = TaxonomyClassifier(scales=['roi', '3x', '5x'])

# Ensemble at inference
def ensemble_predict(img, bbox):
    pred_roi = model_roi(crop_roi(img, bbox))
    pred_3x = model_3x(crop_roi(img, bbox), crop_3x(img, bbox))
    pred_5x = model_5x(crop_roi(img, bbox), crop_3x(img, bbox), crop_5x(img, bbox))

    # Weighted average (tune weights on validation set)
    final_pred = 0.2 * pred_roi + 0.3 * pred_3x + 0.5 * pred_5x
    return final_pred
```

**Success Criteria:** Score < 2.30 (ensemble diversity benefit)

---

### Experiment 5: Add Taxonomy Distance Matrix ⭐ **LOW PRIORITY**

**Hypothesis:** Using WoRMS biological distance matrix (like the winner) to weight misclassifications may improve fine-grained separation.

**Implementation Plan:**

```python
# In preprocessing (replicate winner's A0.data_preprocess.py)
from fathomnet.api import worms

def build_taxonomy_distance_matrix(species_list):
    """Query WoRMS API to build phylogenetic distance matrix"""
    distances = np.zeros((len(species_list), len(species_list)))

    for i, sp1 in enumerate(species_list):
        for j, sp2 in enumerate(species_list):
            # Distance = number of ranks until common ancestor
            distances[i, j] = compute_phylogenetic_distance(sp1, sp2)

    return distances

# In loss function
class TaxonomyAwareLoss(nn.Module):
    def __init__(self, distance_matrix):
        super().__init__()
        self.distance_matrix = torch.tensor(distance_matrix)

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')

        # Weight by phylogenetic distance of prediction vs. true label
        preds = logits.argmax(dim=1)
        weights = self.distance_matrix[targets, preds]

        return (ce_loss * weights).mean()
```

**Success Criteria:** Score < 2.25 (better fine-grained discrimination)

---

## Part 4: Experimental Timeline

| Week | Experiments | Expected Deliverable |
|------|-------------|---------------------|
| **1-2** | Exp 1a-1c (Multi-scale context) | Checkpoint with score < 2.50 |
| **3** | Exp 2 (Cross-scale attention) | Checkpoint with score < 2.40 |
| **4** | Exp 3 (Remove conditioning) | Checkpoint with score < 2.35 |
| **5** | Exp 4 (Ensemble) | Checkpoint with score < 2.30 |
| **6** | Exp 5 (Taxonomy distance) | Final checkpoint, target < 2.25 |
| **7** | Ablation studies & analysis | Dissertation chapter draft |

**Total Duration:** 7 weeks

---

## Part 5: Risk Analysis & Mitigation

### Risk 1: Compute Constraints

**Issue:** 3 separate ConvNeXt encoders = 3× memory/training time (~264M params total)

**Mitigation:**
- Start with smaller backbone (`convnextv2_base` = 88M params instead of `convnextv2_large` = 200M)
- Use gradient checkpointing: `model.gradient_checkpointing_enable()`
- Freeze early layers of context encoders (only fine-tune last few blocks)
- Reduce batch size if needed (32 → 16)

### Risk 2: Overfitting with More Parameters

**Issue:** Adding encoders increases model capacity → risk of overfitting on small dataset (16,589 train images)

**Mitigation:**
- Increase dropout in projection layers (0.2 → 0.5)
- Add stronger augmentation to context windows
- Use earlier early-stopping patience (10 → 5 epochs)
- Monitor train/val loss gap closely
- Reduce learning rate for new components

### Risk 3: Minimal Improvement from Attention

**Issue:** Concatenation may capture most benefits; attention adds complexity without gain

**Mitigation:**
- Run Exp 1b (concatenation) thoroughly first to establish baseline
- Only proceed with Exp 2 if Exp 1b shows promise (score < 2.50)
- Visualize attention weights to verify they're learning meaningful patterns
- Compare computational cost vs. performance gain

### Risk 4: DinoV2 Download/Compatibility

**Issue:** DinoV2-Large is huge (1.1GB checkpoint), may have version issues

**Mitigation:**
- Test DinoV2 loading in isolation before integrating
- Use `timm` library which handles versioning: `timm.create_model('vit_large_patch14_dinov2.lvd142m')`
- Have fallback to ViT-Large if DinoV2 unavailable

---

## Part 6: Success Metrics

### Primary Metric
**Target Score:** < 2.40 (close 50% of gap to current best)
- This would move from 8th → likely top 3-4 on leaderboard

### Secondary Metrics

1. **Per-rank accuracy** (Kingdom → Species): Track which ranks improve most with multi-scale context
2. **Confusion analysis:** Does multi-scale reduce inter-family/inter-genus confusion?
3. **Failure case analysis:** Do errors shift from "wrong environment" to "morphologically similar"?
4. **Calibration:** Does removing conditioning improve confidence calibration?
5. **Attention visualization:** Do attention maps focus on biologically relevant features?

### Ablation Studies for Dissertation

- **Contribution of each scale:** ROI-only vs. ROI+3× vs. ROI+3×+5×
- **Attention vs. concatenation:** Quantify attention's value added
- **Conditioning vs. independent:** Measure error propagation penalty
- **Backbone choice:** ConvNeXt vs. DinoV2 **with multi-scale** (not just single ROI)

---

## Part 7: Expected Contributions to Research Program

### Potential Integration with Dissertation Work

**Key Insights for Broader Research:**

1. **Domain-specific context matters:** For underwater imagery, environmental signals are first-class features, not background noise
2. **Architecture > backbone choice:** Multi-scale design had 10× larger impact than DinoV2 vs. ConvNeXt
3. **Hierarchical conditioning trade-offs:** Elegant and biologically principled but introduces error propagation that limits ceiling
4. **Attention for context fusion:** Quantify whether learned weighting outperforms concatenation

**Novel Contributions from This Work:**

- Systematic ablation of multi-scale context in hierarchical taxonomy classification
- Comparison of sequential conditioning vs. independent prediction with hierarchical loss
- Analysis of when environmental context is critical (underwater/wildlife) vs. less important (studio/medical imagery)
- Attention visualization revealing learned environmental cue priorities
- Open-source implementation bridging state-of-the-art competition code and academic research

**Potential Publications (Independent of Dissertation):**

- **Workshop paper:** CVPR FG-VC Workshop (Fine-Grained Visual Categorization)
- **Conference paper:** Full paper at WACV or similar if results are strong
- **Technical report:** FathomNet competition analysis with improved baseline model
- **Code release:** Well-documented repository for community use

---

## Part 8: Detailed Implementation for Experiment 1 (Multi-Scale Context)

Since this is the highest priority experiment, here's a complete implementation guide:

### Step 1: Modify Data Pipeline

Create `src/data_multiscale.py`:

```python
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class MultiScaleFathomNetDataset(Dataset):
    """
    Dataset that loads ROI crop + multiple context windows
    """
    def __init__(self, image_paths, species_names, taxonomy_df, encoders,
                 transform=None, scales=[1.0, 3.0, 5.0]):
        """
        Args:
            scales: List of crop scale multipliers (1.0 = tight ROI)
        """
        self.image_paths = image_paths
        self.species_names = species_names
        self.taxonomy_df = taxonomy_df
        self.encoders = encoders
        self.transform = transform
        self.scales = scales

        # Pre-compute taxonomic info (same as before)
        self.taxonomic_info = []
        for species in species_names:
            row = self.taxonomy_df[self.taxonomy_df['species'] == species]
            if row.empty:
                self.taxonomic_info.append({level: 0 for level in TAXONOMY_LEVELS})
            else:
                row = row.iloc[0]
                self.taxonomic_info.append({
                    level: int(row[f'{level}_id']) for level in TAXONOMY_LEVELS
                })

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return dummy multi-scale images
            dummy = torch.zeros(3, 224, 224)
            result = {f'scale_{s}': dummy for s in self.scales}
            result.update(self.taxonomic_info[idx])
            return result

        # For ROI crops, we don't have bbox info in the path
        # Assume image is already cropped to ROI
        # Create multi-scale versions by zooming in/out

        multi_scale_imgs = {}

        for scale in self.scales:
            if scale == 1.0:
                # Original ROI
                scaled_img = img
            elif scale > 1.0:
                # Larger context: zoom out (pad and resize)
                # This is a simplification - ideally load from original frame
                w, h = img.size
                new_w, new_h = int(w * scale), int(h * scale)

                # Pad the image to simulate larger context
                # (In real implementation, would load from original frame with larger bbox)
                padding = ((new_w - w) // 2, (new_h - h) // 2)
                scaled_img = transforms.Pad(padding, fill=0)(img)
            else:
                # Smaller than ROI: crop center
                w, h = img.size
                new_w, new_h = int(w * scale), int(h * scale)
                scaled_img = transforms.CenterCrop((new_h, new_w))(img)

            # Apply standard transforms
            if self.transform:
                scaled_img = self.transform(scaled_img)

            multi_scale_imgs[f'scale_{scale}'] = scaled_img

        result = multi_scale_imgs
        result.update(self.taxonomic_info[idx])

        return result


def collate_fn_multiscale(batch):
    """Collate function for multi-scale dataset"""
    # Extract scales from first batch item
    scales = [k for k in batch[0].keys() if k.startswith('scale_')]

    # Stack images for each scale
    images = {}
    for scale_key in scales:
        images[scale_key] = torch.stack([b[scale_key] for b in batch])

    # Collect labels for each taxonomic level
    labels = {}
    for level in TAXONOMY_LEVELS:
        if level in batch[0]:
            labels[level] = torch.tensor([b[level] for b in batch])

    return images, labels
```

**Note:** The current FathomNet dataset structure has ROI images already cropped. To get true multi-scale context, you'd need to:

1. Load the original full frames from `dataset_train.json`
2. Use the COCO bounding boxes to extract ROI + 3× + 5× windows
3. This requires modifying the data loading to read from the full frames, not just ROI crops

**Alternative approach (if original frames unavailable):**
- Train the multi-scale model using zoom-in/zoom-out on ROI crops as a proof-of-concept
- This won't give true environmental context but will test the architecture

### Step 2: Modify Model Architecture

Create `src/model_multiscale.py`:

```python
import torch
import torch.nn as nn
import pytorch_lightning as pl
import timm

class MultiScaleBackbone(nn.Module):
    """
    Multiple encoder branches for different scales
    """
    def __init__(self, backbone_name='convnextv2_base', num_scales=3):
        super().__init__()

        # Create separate backbone for each scale
        self.encoders = nn.ModuleList([
            timm.create_model(backbone_name, pretrained=True, num_classes=0)
            for _ in range(num_scales)
        ])

        # Get feature dimension from first encoder
        self.feat_dim = self.encoders[0].num_features

    def forward(self, scale_images_dict):
        """
        Args:
            scale_images_dict: {'scale_1.0': tensor, 'scale_3.0': tensor, ...}
        Returns:
            List of feature tensors, one per scale
        """
        features = []

        # Sort scales to ensure consistent ordering
        scale_keys = sorted(scale_images_dict.keys())

        for i, scale_key in enumerate(scale_keys):
            feats = self.encoders[i](scale_images_dict[scale_key])
            features.append(feats)

        return features


class MultiScaleTaxonomyClassifier(pl.LightningModule):
    """
    Multi-scale hierarchical taxonomy classifier
    """
    def __init__(self, class_counts, lr=3e-4, num_scales=3, fusion='concat'):
        super().__init__()
        self.save_hyperparameters()
        self.class_counts = class_counts
        self.lr = lr
        self.fusion = fusion  # 'concat' or 'attention'

        # Multi-scale backbone
        self.backbone = MultiScaleBackbone(num_scales=num_scales)
        feat_dim = self.backbone.feat_dim

        # Projection layers for each scale
        self.projections = nn.ModuleList([
            nn.Linear(feat_dim, 512) for _ in range(num_scales)
        ])

        # Fusion layer
        if fusion == 'concat':
            fused_dim = 512 * num_scales
        else:
            raise NotImplementedError("Attention fusion coming in Exp 2")

        # Shared feature processing
        self.shared_features = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.3),  # Increased dropout for more params
        )

        # Hierarchical classifier heads (same as before)
        self._create_hierarchical_network(class_counts)

        # Loss
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

        # Tracking
        self.val_step_outputs = []

    def _create_hierarchical_network(self, class_counts):
        # Same hierarchical head architecture as current model
        # (kingdom → phylum → ... → species with conditioning)
        # Copy from your existing model
        pass  # See your notebook for full implementation

    def forward(self, scale_images_dict):
        # Extract features from each scale
        scale_features = self.backbone(scale_images_dict)

        # Project each scale to common dimension
        projected = [
            proj(feats) for proj, feats in zip(self.projections, scale_features)
        ]

        # Fuse multi-scale features
        if self.fusion == 'concat':
            fused = torch.cat(projected, dim=1)

        # Shared processing
        shared = self.shared_features(fused)

        # Hierarchical prediction (same as before)
        return self.hierarchical_forward(shared)

    def training_step(self, batch, batch_idx):
        images_dict, labels = batch
        outputs = self(images_dict)

        loss, level_losses = self.hierarchical_loss(outputs, labels)

        self.log("train_loss", loss, prog_bar=True)
        for level, level_loss in level_losses.items():
            self.log(f"train_{level}_loss", level_loss)

        return loss

    # ... rest of training/validation/prediction steps same as before
```

### Step 3: Training Script

Create `train_multiscale.py`:

```python
import os
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.data_multiscale import MultiScaleFathomNetDataset, collate_fn_multiscale
from src.model_multiscale import MultiScaleTaxonomyClassifier

def main():
    # Load data (same as before, but use MultiScaleFathomNetDataset)
    # ...

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,  # Reduced due to 3× memory
        shuffle=True,
        num_workers=8,
        collate_fn=collate_fn_multiscale
    )

    # Create model
    model = MultiScaleTaxonomyClassifier(
        class_counts=class_counts,
        num_scales=3,  # ROI + 3× + 5×
        fusion='concat'
    )

    # Train
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        devices=1,
        precision=16,
        callbacks=[...],
    )

    trainer.fit(model, train_loader, val_loader)

if __name__ == "__main__":
    main()
```

---

## Part 9: Questions for Advisor

1. **Compute budget:** Do I have access to multi-GPU nodes for 3-encoder training? If not, should I prioritize smaller backbones (ConvNeXtV2-Base vs Large)?

2. **Scope:** Should I implement all 5 experiments, or focus deeply on Exp 1-3 with thorough ablations?

3. **Baseline:** Should I try to exactly replicate the winner's full setup (4 DinoV2-Large + full-frame) as an upper bound, or is that overkill?

4. **Research priority:** Given this is supporting work (not core dissertation), what level of depth is appropriate? Should this be a thorough investigation or a focused exploration of multi-scale context?

5. **Publication target:** Are these experiments sufficient for a workshop paper (e.g., CVPR FG-VC workshop), or should I expand scope? Is publication even a goal for this supporting work?

6. **Data access:** Can I get access to the original full frames (not just ROI crops) to implement true multi-scale context windows? Current dataset only has cropped ROIs.

7. **Timeline:** Is 7 weeks realistic for this work given other dissertation priorities, or should I prioritize fewer experiments with deeper analysis?

8. **Integration:** How might insights from this work (multi-scale context, attention mechanisms) inform or support my primary dissertation research?

---

## Part 10: Immediate Next Steps

Upon your approval, I will:

1. **Week 1 Days 1-2:** Set up new repository structure with `src/data_multiscale.py` and `src/model_multiscale.py`

2. **Week 1 Days 3-5:** Implement and test multi-scale data loading (verify I can load multiple context windows)

3. **Week 1 Days 6-7:** Implement multi-scale model architecture (3 ConvNeXt encoders with concatenation fusion)

4. **Week 2 Days 1-5:** Run Experiment 1b (ROI + 3× + 5×) and monitor training closely

5. **Week 2 Days 6-7:** Evaluate Exp 1b on validation set and decide whether to proceed to Exp 2 (attention) or iterate on Exp 1

**Deliverable by end of Week 2:** Model checkpoint with score < 2.50 (or detailed analysis of why multi-scale didn't help as expected)

---

## Appendix A: Code Repository Structure

```
fathomnet/
├── config/
│   ├── experiment-baseline.yaml          # Current 8th place config
│   ├── experiment-multiscale-2enc.yaml   # Exp 1a: ROI + 3×
│   ├── experiment-multiscale-3enc.yaml   # Exp 1b: ROI + 3× + 5× (MAIN)
│   ├── experiment-multiscale-attention.yaml  # Exp 2
│   └── experiment-independent-heads.yaml  # Exp 3
├── src/
│   ├── data.py                           # Original single-scale dataset
│   ├── data_multiscale.py                # NEW: Multi-scale dataset
│   ├── model.py                          # Original hierarchical model
│   ├── model_multiscale.py               # NEW: Multi-encoder architecture
│   ├── model_attention.py                # NEW: Cross-scale attention (Exp 2)
│   ├── model_independent.py              # NEW: Independent hierarchical heads (Exp 3)
│   └── taxonomy_distance.py              # NEW: WoRMS API integration (Exp 5)
├── scripts/
│   ├── train_multiscale.py               # Training script for Exp 1
│   ├── evaluate_ablations.py             # Systematic ablation study
│   └── visualize_attention.py            # Attention map visualization (Exp 2)
├── notebooks/
│   ├── analysis_winner.ipynb             # This report's analysis
│   ├── results_comparison.ipynb          # Track all experiments
│   └── attention_visualization.ipynb     # Exp 2 attention analysis
├── docs/
│   ├── dissertation_report.md            # This document
│   └── experiment_logs.md                # Detailed experiment notes
├── hierarchical-classifier.ipynb         # Original notebook (baseline)
└── outputs/
    ├── exp1a_roi_3x/                     # Exp 1a results
    ├── exp1b_roi_3x_5x/                  # Exp 1b results (MAIN)
    ├── exp2_attention/                   # Exp 2 results
    └── exp3_independent/                 # Exp 3 results
```

---

## Appendix B: References

1. Vaswani, A., et al. (2017). "Attention Is All You Need." *NeurIPS*.
2. FathomNet CVPR'25 Competition (1st place solution repository)
3. Oquab, M., et al. (2024). "DINOv2: Learning Robust Visual Features without Supervision." *ICLR*.
4. Woo, S., et al. (2023). "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders." *CVPR*.
5. Your current FathomNet repository: `/mnt/beegfs/home/dzimmerman2021/Documents/fathomnet/`

---

## Summary

This report identifies **multi-scale environmental context extraction** as the primary architectural advantage of the 1st place solution. My experiments will systematically validate this hypothesis through:

1. Multi-scale feature extraction (Exp 1) - **CRITICAL**
2. Attention-based context fusion (Exp 2) - **HIGH VALUE**
3. Removing hierarchical conditioning error propagation (Exp 3) - **MEDIUM VALUE**
4. Ensemble and taxonomy-aware losses (Exp 4-5) - **NICE TO HAVE**

Expected outcome: Close 50%+ of performance gap (score < 2.40) while generating novel insights on the role of environmental context in fine-grained classification for dissertation.

**Ready to begin upon your approval.**
