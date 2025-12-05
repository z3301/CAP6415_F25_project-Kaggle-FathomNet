# 1-Week Accelerated Timeline for CAP6415 FathomNet Project

**Project Goal**: Complete multi-scale hierarchical classifier with transformer attention for environmental context integration

**Hard Deadline**: 7 days from start
**Non-negotiable Requirements**:
- Multi-scale architecture (Exp 1a + 1b)
- Transformer attention mechanism
- Evaluation metrics and visualizations
- Final report/documentation for CAP6415

---

## Day 1-2: Multi-Scale Training (IN PROGRESS)

### Current Status
- ✅ Exp 1a: ROI + 3× context (2 encoders, 181M params) - Running on GPU 0
- ✅ Exp 1b: ROI + 3× + 5× context (3 encoders, 269M params) - Running on GPU 1
- ⏳ Expected completion: 24-48 hours (50 epochs with early stopping)

### Monitoring Tasks
- [x] Launch both experiments in parallel
- [ ] Check training progress every 6-8 hours
- [ ] Verify no OOM errors or crashes
- [ ] Watch for early stopping trigger (patience=10 epochs)

**Deliverable**: Best checkpoints for Exp 1a and 1b saved to `outputs/exp1a_2scales/` and `outputs/exp1b_3scales/`

---

## Day 3: Rapid Evaluation + Attention Design (8 hours)

### Morning: Evaluate Multi-Scale Results (4 hours)
**Script**: Create `evaluate_multiscale.py`

Tasks:
1. Load best checkpoints from Exp 1a and 1b
2. Run evaluation on validation set (15% holdout)
3. Generate metrics for all 7 taxonomic ranks
4. Create comparison table: baseline vs 1a vs 1b
5. Quick visualization: accuracy per rank, confusion matrices

**Success Criteria**:
- Exp 1b validation score < 2.50 (target: close 30% of gap to 1st place)
- Improved performance over Exp 1a (2 scales)

### Afternoon: Design Attention Mechanism (4 hours)
**File**: Create `src/model_attention.py`

Architecture decisions:
1. **Feature extraction**:
   - ROI features: ConvNeXt global pooling [batch, 1024]
   - Environmental patches: DinoV2 patch embeddings [batch, num_patches, 768]
   - Context scales: 3× and 5× crops processed through separate DinoV2

2. **Attention mechanism**:
   - ROI features → Query projection: Linear(1024 → 512)
   - Environmental patches → Key/Value projections: Linear(768 → 512)
   - Multi-head attention: 8 heads, dropout 0.1
   - Output fusion: Concat(ROI, attended_context) → hierarchical heads

3. **Integration**:
   - Reuse existing hierarchical heads from `model_multiscale.py`
   - Keep training infrastructure from PyTorch Lightning
   - Target model size: ~350M params (3× DinoV2 + attention + heads)

**Deliverable**: Architecture sketch, pseudo-code for attention module

---

## Day 4-5: Implement and Train Attention Model (16 hours)

### Day 4 Morning: Implementation (4 hours)
**Files**: `src/model_attention.py`, `src/data_attention.py`

1. `MultiScaleAttentionBackbone`:
   - 3× DinoV2-B/14 encoders (1 per scale: 1.0×, 3.0×, 5.0×)
   - Extract patch embeddings (not global features)
   - Projection layers for query/key/value

2. `CrossScaleAttention`:
   - PyTorch `nn.MultiheadAttention` (8 heads)
   - ROI patches query environmental patches from 3× and 5× scales
   - Residual connections + layer norm

3. `AttentionTaxonomyClassifier`:
   - Inherit from `pl.LightningModule`
   - Reuse hierarchical sequential conditioning from existing model
   - Same optimizer/scheduler setup as multi-scale experiments

### Day 4 Afternoon: Data Pipeline (4 hours)
**File**: `src/data_attention.py`

1. Modify `MultiScaleFathomNetDataset`:
   - Return **patch embeddings** instead of global features
   - DinoV2 image size: 224×224 → 16×16 patches (14×14 grid = 196 patches)
   - Store patches per scale: `{scale_1.0: [196, 768], scale_3.0: [196, 768], scale_5.0: [196, 768]}`

2. Update `collate_fn_attention`:
   - Stack patch tensors: `[batch, num_scales, 196, 768]`
   - Keep taxonomic labels unchanged

### Day 5: Training Attention Model (8 hours)
**Script**: `train_attention.py`

Configuration:
- Experiment: `exp2_attention`
- Scales: [1.0, 3.0, 5.0]
- Batch size: 12 (reduced due to patch memory)
- Learning rate: 3e-4 with cosine annealing
- Max epochs: 30 (attention should converge faster)
- Early stopping: patience=8

Launch command:
```bash
CUDA_VISIBLE_DEVICES=2 ~/miniconda/envs/fathomnet/bin/python train_attention.py \
  --exp 2 --epochs 30 --lr 3e-4 --batch-size 12
```

**Expected training time**: 6-8 hours (fewer epochs, similar dataset size)

**Success Criteria**:
- Validation score < 2.30 (further improvement over multi-scale baseline)
- Attention weights visualize meaningful environmental context
- No gradient explosion or NaN losses

**Deliverable**: Best checkpoint saved to `outputs/exp2_attention/`

---

## Day 6: Final Evaluation and Analysis (8 hours)

### Morning: Comprehensive Evaluation (4 hours)

1. **Metrics for all experiments**:
   - Run `evaluate_multiscale.py --model outputs/exp1a_2scales/best.ckpt`
   - Run `evaluate_multiscale.py --model outputs/exp1b_3scales/best.ckpt`
   - Run `evaluate_attention.py --model outputs/exp2_attention/best.ckpt`

2. **Comparison table** (save to `outputs/results_summary.csv`):
   ```
   | Experiment        | Species Acc | Genus Acc | Family Acc | Val Score | Params |
   |-------------------|-------------|-----------|------------|-----------|--------|
   | Baseline          | 0.XXX       | 0.XXX     | 0.XXX      | 2.74      | 88M    |
   | Exp 1a (2 scales) | 0.XXX       | 0.XXX     | 0.XXX      | 2.XX      | 181M   |
   | Exp 1b (3 scales) | 0.XXX       | 0.XXX     | 0.XXX      | 2.XX      | 269M   |
   | Exp 2 (attention) | 0.XXX       | 0.XXX     | 0.XXX      | 2.XX      | 350M   |
   ```

3. **Visualizations**:
   - Confusion matrices per taxonomic rank (all experiments)
   - Training curves: loss and accuracy over epochs
   - Attention heatmaps: ROI attending to environmental context (5-10 examples)

### Afternoon: Error Analysis (4 hours)

1. **Analyze failure cases**:
   - Find worst-performing species classes
   - Check if attention mechanism helps disambiguation
   - Identify systematic errors (e.g., morphologically similar species)

2. **Ablation study** (if time permits):
   - Remove 5× scale from Exp 2 → Does 3× alone suffice?
   - Vary number of attention heads (4 vs 8 vs 16)
   - Document in `docs/ablations.md`

**Deliverable**:
- `outputs/results_summary.csv`
- `outputs/visualizations/` folder with all plots
- `docs/error_analysis.md` with findings

---

## Day 7: Final Report and Documentation (8 hours)

### Morning: Write CAP6415 Report (4 hours)
**File**: `docs/cap6415_final_report.md`

Sections:
1. **Introduction** (0.5 pages)
   - Problem: Fine-grained marine species classification
   - Challenge: Similar morphology, environmental context crucial
   - Goal: Incorporate multi-scale spatial context with attention

2. **Methods** (1.5 pages)
   - Multi-scale architecture (Exp 1a/1b)
   - Transformer attention for environmental context (Exp 2)
   - Hierarchical sequential conditioning across 7 taxonomic ranks
   - Training details: dataset splits, hyperparameters, hardware

3. **Results** (1 page)
   - Comparison table (baseline vs 1a vs 1b vs 2)
   - Confusion matrices and attention visualizations
   - Error analysis findings

4. **Discussion** (1 page)
   - Why multi-scale context helps (ecological background)
   - Attention mechanism interpretation (what patterns did it learn?)
   - Limitations and future work

5. **Conclusion** (0.5 pages)
   - Summary of contributions
   - Performance improvement quantified

**Target length**: 4-5 pages + figures

### Afternoon: Code Cleanup and Submission (4 hours)

1. **Code organization**:
   - Ensure all imports work
   - Add docstrings to key functions
   - Create `requirements.txt` with exact versions
   - Update `README.md` with usage instructions

2. **Reproducibility**:
   - Document exact commands to reproduce experiments
   - Save experiment configs to `outputs/*/config.json`
   - Archive best checkpoints (compress if needed)

3. **Final checklist**:
   - [ ] All code runs without errors
   - [ ] Report is complete and well-formatted
   - [ ] Figures are high-quality and labeled
   - [ ] GitHub repository is organized
   - [ ] Submission files ready for CAP6415

**Deliverable**: Complete project package ready for submission

---

## Contingency Plans

### If Exp 1a/1b Training Fails (Day 3)
- **Fallback**: Use baseline model + implement attention directly
- Skip multi-scale comparison, focus on attention as main contribution
- Reduces scope but keeps core innovation (attention mechanism)

### If Attention Training OOMs (Day 5)
- **Fix 1**: Reduce batch size to 8 or 6
- **Fix 2**: Use gradient checkpointing to save memory
- **Fix 3**: Reduce number of attention heads (8 → 4)
- **Fix 4**: Process scales sequentially instead of parallel

### If Training Takes Longer Than Expected (Day 5)
- **Fix 1**: Reduce max_epochs from 30 to 20
- **Fix 2**: Use smaller DinoV2 (DinoV2-S instead of DinoV2-B)
- **Fix 3**: Train on subset of data (50% or 70%)

### If No Improvement from Attention (Day 6)
- **Analysis**: Debug attention weights (are they learning meaningful patterns?)
- **Pivot**: Frame as "explored attention but multi-scale sufficient"
- Still valid contribution: systematic study of multi-scale architecture

---

## Daily Time Budget

| Day   | Hours | Focus Area                          |
|-------|-------|-------------------------------------|
| 1-2   | 24-48h| Multi-scale training (automated)    |
| 3     | 8h    | Evaluation + attention design       |
| 4     | 8h    | Attention implementation            |
| 5     | 8h    | Attention training (6-8h automated) |
| 6     | 8h    | Final evaluation and analysis       |
| 7     | 8h    | Report writing and submission       |
| **Total** | **40h** | **Active work time**            |

---

## Success Metrics

### Minimum Viable (Pass CAP6415):
- ✅ Multi-scale architecture implemented and trained
- ✅ Attention mechanism implemented (even if no improvement)
- ✅ Complete evaluation with metrics and visualizations
- ✅ Final report documenting approach and results

### Target (Strong Performance):
- 🎯 Validation score < 2.50 (close 30% gap to 1st place)
- 🎯 Attention mechanism shows measurable improvement
- 🎯 Clear visualizations of what attention learns
- 🎯 Thorough error analysis and ablation studies

### Stretch (Publication Quality):
- 🚀 Validation score < 2.30 (close 50% gap)
- 🚀 Attention weights align with ecological knowledge
- 🚀 Complete ablation study of design choices
- 🚀 High-quality figures and polished report

---

## Key Risk Mitigations

1. **Parallel experiments**: Exp 1a and 1b running simultaneously saves 24 hours
2. **Pre-built infrastructure**: Reuse data loaders, training loops, evaluation code
3. **Simple attention first**: Use PyTorch built-in `MultiheadAttention` instead of custom implementation
4. **Continuous monitoring**: Check training every 6-8 hours to catch errors early
5. **Staged deliverables**: Each day produces usable output (not all-or-nothing)

---

## Questions for User (Optional)

1. Do you need a formal presentation/slides for CAP6415, or just the written report?
2. Are there specific formatting requirements (LaTeX, page limits, citation style)?
3. Should I prioritize code quality or analysis depth if time gets tight?

---

**Timeline Status**: Day 1-2 in progress, both experiments training successfully
**Next Milestone**: Day 3 evaluation when training completes (~24-48 hours from now)
