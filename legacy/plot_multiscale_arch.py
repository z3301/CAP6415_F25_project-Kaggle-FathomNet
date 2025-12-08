#!/usr/bin/env python3
"""Generate architecture diagram for the 4-scale Multi-Scale model with Taxonomic Loss using PlotNeuralNet."""

import sys
import os

# Use the installed PlotNeuralNet package
from PlotNeuralNet.PyCore.TikzGen import (
    ToHead, ToCor, ToBegin, ToEnd, ToGenerate, ToInput,
    ToConv, ToConvConvRelu, ToPool, ToSoftMax, ToConnection, ToSkip, ToFullyConnected, ToSum
)

# Path to PlotNeuralNet Layers directory
PLOTNN_PATH = "/mnt/beegfs/home/dzimmerman2021/miniconda/envs/fathomnet/lib/python3.10/site-packages/PlotNeuralNet"

# Sample image path (relative to where tex is compiled)
SAMPLE_IMAGE = "/mnt/beegfs/home/dzimmerman2021/Documents/fathomnet/legacy/sample_roi.png"

# Custom colors for the diagram
CUSTOM_COLORS = r"""
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{graphicx}
\def\ROIColor{rgb:orange,5;red,3;white,2}
\def\ContextColor{rgb:blue,4;cyan,2;white,3}
\def\FusionColor{rgb:green,4;yellow,2;white,2}
\def\TaxLossColor{rgb:red,5;magenta,2;white,2}
"""

# Helper function to create a 4-stage ConvNeXt encoder with input image
def convnext_encoder(prefix, y_offset, is_large=False, input_caption="Input", scale_label="1x"):
    """Generate a 4-stage ConvNeXt encoder at given y_offset.

    Args:
        prefix: Name prefix for all nodes (e.g., 'roi', 'ctx3')
        y_offset: Vertical position offset
        is_large: If True, use Large variant (197M), else Base (88M)
        input_caption: Caption for input image
        scale_label: Scale label for the input image (e.g., "1x", "3x")
    """
    # Dimensions for Large vs Base
    if is_large:
        channels = [(192, 192), (384, 384), (768, 768), (1536, 1536)]
        widths = [(2, 2), (3, 3), (4, 4), (5, 5)]
        final_dim = 1536
        model_name = "ConvNeXtV2-L"
    else:
        channels = [(128, 128), (256, 256), (512, 512), (1024, 1024)]
        widths = [(2, 2), (2, 2), (3, 3), (4, 4)]
        final_dim = 1024
        model_name = "ConvNeXtV2-B"

    layers = []

    # Input image node
    layers.append(rf"""
\node[canvas is zy plane at x=0] ({prefix}_img) at (-3,{y_offset},0) {{\includegraphics[width=2.5cm,height=2.5cm]{{{SAMPLE_IMAGE}}}}};
\node[below, font=\scriptsize] at ({prefix}_img.south) {{{input_caption}}};
""")

    # First conv block (stem)
    layers.append(ToConvConvRelu(
        name=f'{prefix}_input',
        sFilter="224",
        nFilter=(3, 3),
        offset=f"(0,{y_offset},0)",
        to="(0,0,0)",
        width=(1, 1),
        height=18,
        depth=18,
        caption=model_name
    ))

    # Connection from image to input
    layers.append(rf"\draw [connection] ({prefix}_img.east) -- node {{\midarrow}} ({prefix}_input-west);")

    # Stage 1: 56x56
    layers.append(ToConvConvRelu(
        name=f'{prefix}_s1',
        sFilter="56",
        nFilter=channels[0],
        offset="(2.5,0,0)",
        to=f"({prefix}_input-east)",
        width=widths[0],
        height=14,
        depth=14,
        caption="Stage 1"
    ))
    layers.append(ToConnection(f"{prefix}_input", f"{prefix}_s1"))

    # Stage 2: 28x28
    layers.append(ToConvConvRelu(
        name=f'{prefix}_s2',
        sFilter="28",
        nFilter=channels[1],
        offset="(2.5,0,0)",
        to=f"({prefix}_s1-east)",
        width=widths[1],
        height=11,
        depth=11,
        caption="Stage 2"
    ))
    layers.append(ToConnection(f"{prefix}_s1", f"{prefix}_s2"))

    # Stage 3: 14x14
    layers.append(ToConvConvRelu(
        name=f'{prefix}_s3',
        sFilter="14",
        nFilter=channels[2],
        offset="(2.5,0,0)",
        to=f"({prefix}_s2-east)",
        width=widths[2],
        height=8,
        depth=8,
        caption="Stage 3"
    ))
    layers.append(ToConnection(f"{prefix}_s2", f"{prefix}_s3"))

    # Stage 4: 7x7
    layers.append(ToConvConvRelu(
        name=f'{prefix}_s4',
        sFilter="7",
        nFilter=channels[3],
        offset="(2.5,0,0)",
        to=f"({prefix}_s3-east)",
        width=widths[3],
        height=5,
        depth=5,
        caption="Stage 4"
    ))
    layers.append(ToConnection(f"{prefix}_s3", f"{prefix}_s4"))

    # GAP (Global Average Pooling)
    layers.append(ToPool(
        name=f'{prefix}_gap',
        offset="(1.5,0,0)",
        to=f"({prefix}_s4-east)",
        width=1,
        height=3,
        depth=3,
        opacity=0.8,
        caption="GAP"
    ))
    layers.append(ToConnection(f"{prefix}_s4", f"{prefix}_gap"))

    return layers

# Flatten helper
def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(item)
        else:
            result.append(item)
    return result

arch = flatten([
    # Header
    ToHead(PLOTNN_PATH),
    ToCor(),
    CUSTOM_COLORS,
    ToBegin(),

    # ============ ROI ENCODER (1x) - TOP - ConvNeXtV2-Large ============
    convnext_encoder('roi', y_offset=0, is_large=True, input_caption="ROI (1x)"),

    # ============ CONTEXT ENCODER (3x) - ConvNeXtV2-Base ============
    convnext_encoder('ctx3', y_offset=-9, is_large=False, input_caption="Context (3x)"),

    # ============ CONTEXT ENCODER (5x) - ConvNeXtV2-Base ============
    convnext_encoder('ctx5', y_offset=-18, is_large=False, input_caption="Context (5x)"),

    # ============ FULL IMAGE ENCODER - ConvNeXtV2-Base ============
    convnext_encoder('full', y_offset=-27, is_large=False, input_caption="Full Image"),

    # ============ FEATURE FUSION ============
    ToSum(name="fusion", offset="(3,-13.5,0)", to="(roi_gap-east)", radius=2.5, opacity=0.7),

    # Connections from all encoders to fusion
    r"\draw [connection]  (roi_gap-east) -- node {\midarrow} (fusion-west);",
    r"\draw [connection]  (ctx3_gap-east) -- node {\midarrow} (fusion-west);",
    r"\draw [connection]  (ctx5_gap-east) -- node {\midarrow} (fusion-west);",
    r"\draw [connection]  (full_gap-east) -- node {\midarrow} (fusion-west);",

    # Fused features
    ToFullyConnected(
        name="fused",
        sFilter="4608",
        offset="(2.5,0,0)",
        to="(fusion-east)",
        width=3,
        height=4,
        depth=30,
        caption="Fused\\\\Features"
    ),

    ToConnection("fusion", "fused"),

    # ============ HIERARCHICAL HEADS ============
    ToSoftMax(name="kingdom", sFilter="2", offset="(4,9,0)", to="(fused-east)",
              width=1, height=2, depth=2, caption="Kingdom"),

    ToSoftMax(name="phylum", sFilter="8", offset="(4,6,0)", to="(fused-east)",
              width=1, height=2, depth=4, caption="Phylum"),

    ToSoftMax(name="class_head", sFilter="22", offset="(4,3,0)", to="(fused-east)",
              width=1, height=2, depth=6, caption="Class"),

    ToSoftMax(name="order", sFilter="43", offset="(4,0,0)", to="(fused-east)",
              width=1, height=2, depth=8, caption="Order"),

    ToSoftMax(name="family", sFilter="66", offset="(4,-3,0)", to="(fused-east)",
              width=1, height=2, depth=10, caption="Family"),

    ToSoftMax(name="genus", sFilter="75", offset="(4,-6,0)", to="(fused-east)",
              width=1, height=2, depth=12, caption="Genus"),

    ToSoftMax(name="species", sFilter="79", offset="(4,-9,0)", to="(fused-east)",
              width=1, height=2, depth=14, caption="Species"),

    # Connections from fused to all heads
    ToConnection("fused", "kingdom"),
    ToConnection("fused", "phylum"),
    ToConnection("fused", "class_head"),
    ToConnection("fused", "order"),
    ToConnection("fused", "family"),
    ToConnection("fused", "genus"),
    ToConnection("fused", "species"),

    # Hierarchical skip connections
    ToSkip(of="kingdom", to="phylum", pos=1.25),
    ToSkip(of="phylum", to="class_head", pos=1.25),
    ToSkip(of="class_head", to="order", pos=1.25),
    ToSkip(of="order", to="family", pos=1.25),
    ToSkip(of="family", to="genus", pos=1.25),
    ToSkip(of="genus", to="species", pos=1.25),

    # ============ TAXONOMIC DISTANCE LOSS (annotation) ============
    r"""
\node[draw, rounded corners, fill=red!20, minimum width=3cm, minimum height=1.5cm, align=center]
    (taxloss) at ([xshift=4cm, yshift=-6cm]species-east) {
    \textbf{Taxonomic Loss}\\[2pt]
    $\mathcal{L} = (1-\alpha)\cdot\text{CE} + \alpha\cdot\sum_i p_i \cdot d(i,y)$\\[2pt]
    $\alpha = 0.3$
};
""",

    # Arrow from species to loss
    r"\draw [->, thick, dashed, red!60] (species-south) |- (taxloss.west);",

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
\node[align=center, font=\Large\bfseries] at (10, 6) {
    4-Scale Multi-Scale Architecture with Taxonomic Distance Loss\\[4pt]
    \normalfont\normalsize Total Parameters: 461M
};
""",

    ToEnd()
])

def main():
    output_path = "/mnt/beegfs/home/dzimmerman2021/Documents/fathomnet/legacy/multiscale_architecture.tex"
    ToGenerate(arch, output_path)

    # Fix the shift syntax
    import subprocess
    subprocess.run(['sed', '-i', 's/shift=(\\([^)]*\\))/shift={(\\1)}/g', output_path])

    print(f"Generated {output_path}")
    print(f"Run: cd /mnt/beegfs/home/dzimmerman2021/Documents/fathomnet/legacy && tectonic multiscale_architecture.tex")

if __name__ == '__main__':
    main()
