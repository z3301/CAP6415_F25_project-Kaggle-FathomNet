# Documentation & Publication Plan 

## Overall Objective
To transform my Kaggle FathomNet 2025 solution (a hierarchical, taxonomy-aware classifier for underwater species) into a fully documented and reproducible experiment suitable for both academic grading and peer-reviewed publication.

---

## 1. Repository Organization (publishable structure)

```
CAP6415_F25_project-FathomNet2025/
│
├── README.md                  ← overview, abstract, reproduction guide
├── requirements.txt            ← dependencies
├── hierarchical-classifier.ipynb
│
├── docs/                       ← scientific documentation
│   ├── abstract.md
│   ├── methodology.md
│   ├── results.md
│   ├── discussion.md
│   └── references.bib
│
├── results/                    ← generated plots, confusion matrices, sample images
│   ├── metrics_table.csv
│   ├── hierarchy_accuracy.png
│   ├── confusion_matrix_species.png
│   └── qualitative_samples/
│
├── logs/
│   ├── week1log.txt
│   ├── week2log.txt
│   ├── week3log.txt
│   ├── week4log.txt
│   └── week5log.txt
│
└── video_demo/                 ← screen recording and slides (Week 5)
```

This structure doubles as both:
- A **grading-compliant submission**, and  
- A **journal-ready supplementary dataset** (all reproducibility materials neatly versioned).

---

## 2. Week-by-Week Deliverables

| Week | Deadline | Deliverables | Focus Area | Notes |
|------|-----------|--------------|-------------|--------|
| **Week 1** | Nov 9 | `README.md`, `week1log.txt` | Repository setup & initial literature review | Include a short project abstract in README and confirm CUDA setup. |
| **Week 2** | Nov 16 | `week2log.txt`, initial code upload | Data loading, taxonomy CSV preprocessing | Document any data cleaning or taxonomy-mapping scripts. |
| **Week 3** | Nov 23 | `week3log.txt`, model implementation | Build the hierarchical classifier notebook | Comment every major code block for publication readiness. |
| **Week 4** | Nov 30 | `week4log.txt`, results folder | Train model, generate graphs and qualitative samples | Save metrics plots, confusion matrices, and example predictions in `/results/`. |
| **Week 5** | Dec 7 | `week5log.txt`, `video_demo/` | Record 10–20 min video demo, finalize documentation | Include narration explaining code flow, results, and limitations. |
| **Final** | Dec 8 | Push final repo | Submission + publication draft | Export docs as a single PDF (`docs/fathomnet_hierarchical_paper_draft.pdf`). |

---

## 3. Scientific Publication Alignment

To prepare this project for a scientific venue (e.g., *IEEE OCEANS*, *Frontiers in Marine Science*, *MDPI Sensors*), the repository’s documentation should be convertible into a paper using your `/docs` folder.

| Section | Repository Source | Journal Equivalent |
|----------|-------------------|--------------------|
| **Abstract** | `README.md` intro | Paper abstract |
| **Methodology** | `docs/methodology.md` (with figures from notebook) | Section 2 |
| **Results** | `/results/` figures + `docs/results.md` | Section 3 |
| **Discussion** | `docs/discussion.md` | Section 4 |
| **References** | `docs/references.bib` | Bibliography |

Use Markdown now; convert later to LaTeX with:
```bash
pandoc docs/*.md -o fathomnet_paper_draft.pdf --bibliography=docs/references.bib
```

---

## 4. Reproducibility Checklist

To ensure the TA or a reviewer can run your code successfully:
- `requirements.txt` lists all dependencies (verified via `pip install -r requirements.txt`).
- The notebook runs end-to-end on a CUDA GPU without external credentials.
- Random seeds are fixed (`torch.manual_seed(42)`).
- `DATA_ROOT` path is the only user-specific edit.
- Output plots and metrics are auto-saved to `/results/`.
- Execution time (approx. minutes per epoch) is documented in the video demo.

---

## 5. Video Demo Plan (10–20 minutes total)

| Segment | Duration | Content |
|----------|-----------|----------|
| Intro | 1–2 min | Briefly describe competition and problem setup. |
| Repository Tour | 2–3 min | Walk through folder structure and README highlights. |
| Code Run | 5–8 min | Execute notebook cells (or show key snippets) and narrate outputs. |
| Results | 3–5 min | Show confusion matrices, taxonomy-level accuracy plots, sample predictions. |
| Discussion & Future Work | 2–3 min | Mention improvements (e.g., domain adaptation, semi-supervised learning). |

---

## 6. Grading → Publication Mapping

| Grading Component | Journal-Ready Equivalent |
|--------------------|--------------------------|
| **Development log (10%)** | Supplementary Appendix A: Research timeline |
| **Description (10%)** | Abstract + Introduction |
| **Documentation (20%)** | Methods and Code Availability |
| **Reproducibility (30%)** | Results Reproduction section |
| **Video Demo (30%)** | Supplementary Video (Visualization of Experiment) |
