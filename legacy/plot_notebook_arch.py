#!/usr/bin/env python3
"""Generate architecture diagram for the notebook's TaxonomyAwareClassifier using PlotNeuralNet."""

import sys
import os

# Use the installed PlotNeuralNet package
from PlotNeuralNet.PyCore.TikzGen import (
    ToHead, ToCor, ToBegin, ToEnd, ToGenerate,
    ToConv, ToConvConvRelu, ToPool, ToSoftMax, ToConnection, ToSkip, ToFullyConnected
)

# Path to PlotNeuralNet Layers directory
PLOTNN_PATH = "/mnt/beegfs/home/dzimmerman2021/miniconda/envs/fathomnet/lib/python3.10/site-packages/PlotNeuralNet"

# Sample image path
SAMPLE_IMAGE = "/mnt/beegfs/home/dzimmerman2021/Documents/fathomnet/legacy/sample_roi.png"

# Custom preamble
CUSTOM_PREAMBLE = r"""
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
"""

# Architecture for TaxonomyAwareClassifier from the notebook
arch = [
    # Header
    ToHead(PLOTNN_PATH),
    ToCor(),
    CUSTOM_PREAMBLE,
    ToBegin(),

    # ============ INPUT IMAGE ============
    rf"""
\node[canvas is zy plane at x=0] (input_img) at (-3,0,0) {{\includegraphics[width=2.5cm,height=2.5cm]{{{SAMPLE_IMAGE}}}}};
\node[below, font=\scriptsize] at (input_img.south) {{ROI (1x)}};
""",

    # ============ ConvNeXtV2-Base Backbone ============
    # Stem
    ToConvConvRelu(
        name='stem',
        sFilter="224",
        nFilter=(3, 3),
        offset="(0,0,0)",
        to="(0,0,0)",
        width=(1, 1),
        height=18,
        depth=18,
        caption="ConvNeXtV2-B"
    ),

    # Connection from image to stem
    r"\draw [connection] (input_img.east) -- node {\midarrow} (stem-west);",

    # Stage 1: 56x56
    ToConvConvRelu(
        name='stage1',
        sFilter="56",
        nFilter=(128, 128),
        offset="(2.5,0,0)",
        to="(stem-east)",
        width=(2, 2),
        height=14,
        depth=14,
        caption="Stage 1"
    ),
    ToConnection("stem", "stage1"),

    # Stage 2: 28x28
    ToConvConvRelu(
        name='stage2',
        sFilter="28",
        nFilter=(256, 256),
        offset="(2.5,0,0)",
        to="(stage1-east)",
        width=(2, 2),
        height=11,
        depth=11,
        caption="Stage 2"
    ),
    ToConnection("stage1", "stage2"),

    # Stage 3: 14x14
    ToConvConvRelu(
        name='stage3',
        sFilter="14",
        nFilter=(512, 512),
        offset="(2.5,0,0)",
        to="(stage2-east)",
        width=(3, 3),
        height=8,
        depth=8,
        caption="Stage 3"
    ),
    ToConnection("stage2", "stage3"),

    # Stage 4: 7x7
    ToConvConvRelu(
        name='stage4',
        sFilter="7",
        nFilter=(1024, 1024),
        offset="(2.5,0,0)",
        to="(stage3-east)",
        width=(4, 4),
        height=5,
        depth=5,
        caption="Stage 4"
    ),
    ToConnection("stage3", "stage4"),

    # Global Average Pooling
    ToPool(name="gap", offset="(1.5,0,0)", to="(stage4-east)", width=1, height=3, depth=3, opacity=0.8, caption="GAP"),
    ToConnection("stage4", "gap"),

    # Shared Features (FC layer)
    ToFullyConnected(
        name="shared",
        sFilter="1024",
        offset="(2,0,0)",
        to="(gap-east)",
        width=2,
        height=3,
        depth=20,
        caption="Shared\\\\Features"
    ),
    ToConnection("gap", "shared"),

    # ============ HIERARCHICAL CLASSIFICATION HEADS ============
    ToSoftMax(name="kingdom", sFilter="2", offset="(4,9,0)", to="(shared-east)",
              width=1, height=2, depth=2, caption="Kingdom"),

    ToSoftMax(name="phylum", sFilter="8", offset="(4,6,0)", to="(shared-east)",
              width=1, height=2, depth=4, caption="Phylum"),

    ToSoftMax(name="class_head", sFilter="22", offset="(4,3,0)", to="(shared-east)",
              width=1, height=2, depth=6, caption="Class"),

    ToSoftMax(name="order", sFilter="43", offset="(4,0,0)", to="(shared-east)",
              width=1, height=2, depth=8, caption="Order"),

    ToSoftMax(name="family", sFilter="66", offset="(4,-3,0)", to="(shared-east)",
              width=1, height=2, depth=10, caption="Family"),

    ToSoftMax(name="genus", sFilter="75", offset="(4,-6,0)", to="(shared-east)",
              width=1, height=2, depth=12, caption="Genus"),

    ToSoftMax(name="species", sFilter="79", offset="(4,-9,0)", to="(shared-east)",
              width=1, height=2, depth=14, caption="Species"),

    # Connections from shared to all heads
    ToConnection("shared", "kingdom"),
    ToConnection("shared", "phylum"),
    ToConnection("shared", "class_head"),
    ToConnection("shared", "order"),
    ToConnection("shared", "family"),
    ToConnection("shared", "genus"),
    ToConnection("shared", "species"),

    # Hierarchical connections between heads (showing information flow)
    ToSkip(of="kingdom", to="phylum", pos=1.25),
    ToSkip(of="phylum", to="class_head", pos=1.25),
    ToSkip(of="class_head", to="order", pos=1.25),
    ToSkip(of="order", to="family", pos=1.25),
    ToSkip(of="family", to="genus", pos=1.25),
    ToSkip(of="genus", to="species", pos=1.25),

    # ============ CROSS-ENTROPY LOSS (annotation) ============
    r"""
\node[draw, rounded corners, fill=red!20, minimum width=3cm, minimum height=1.2cm, align=center]
    (celoss) at ([xshift=4cm, yshift=-6cm]species-east) {
    \textbf{Cross-Entropy Loss}\\[2pt]
    $\mathcal{L} = -\sum_i y_i \log(p_i)$
};
""",

    # Arrow from species to loss
    r"\draw [->, thick, dashed, red!60] (species-south) |- (celoss.west);",

    # ============ CONFIDENCE FALLBACK (annotation) ============
    r"""
\node[draw, rounded corners, fill=blue!15, minimum width=2.5cm, minimum height=1cm, align=center]
    (fallback) at ([xshift=4cm, yshift=4cm]species-east) {
    \textbf{Confidence Fallback}\\[2pt]
    If conf $< 70\%$:\\
    S $\to$ G $\to$ F $\to$ O $\to$ C $\to$ P
};
""",

    r"\draw [->, thick, dashed, blue!60] (species-east) |- (fallback.west);",

    # ============ TITLE ============
    r"""
\node[align=center, font=\Large\bfseries] at (8, 14) {
    Single-Scale Baseline Architecture\\[4pt]
    \normalfont\normalsize Total Parameters: 88M (ConvNeXtV2-Base)
};
""",

    ToEnd()
]

def main():
    output_path = "/mnt/beegfs/home/dzimmerman2021/Documents/fathomnet/legacy/notebook_architecture.tex"
    ToGenerate(arch, output_path)
    print(f"Generated {output_path}")
    print(f"Run: cd /mnt/beegfs/home/dzimmerman2021/Documents/fathomnet/legacy && tectonic notebook_architecture.tex")

if __name__ == '__main__':
    main()
