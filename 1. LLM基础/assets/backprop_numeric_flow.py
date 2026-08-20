#!/usr/bin/env python3
"""生成 7.1 节的数值反向传播计算图。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FONT_CANDIDATES = [
    "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC",
    "WenQuanYi Micro Hei", "Source Han Sans SC", "Arial Unicode MS",
]
available = {font.name for font in fm.fontManager.ttflist}
font = next((name for name in FONT_CANDIDATES if name in available), "DejaVu Sans")
plt.rcParams.update({"font.family": font, "axes.unicode_minus": False})

x = np.array([0.8, 0.4, -0.1, 0.6, 0.1, 0.7])
pre1 = np.array([1.0038, 1.5705, -0.6179, -2.5639, -0.3149])
h = np.array([1.0038, 1.5705, 0.0, 0.0, 0.0])
logits = np.array([2.3050, -0.8975, 1.0104, -1.4460])
p = np.array([0.7473, 0.0304, 0.2048, 0.0176])
logits_grad = np.array([0.7473, 0.0304, 0.2048, -0.9824])
h_grad = np.array([0.3717, 1.8782, -1.1319, -1.2300, 2.0500])
relu_mask = np.array([1, 1, 0, 0, 0])
pre1_grad = np.array([0.3717, 1.8782, 0.0, 0.0, 0.0])
x_grad = np.array([3.1506, 1.3900, -0.6410, 1.5851, -0.9574, -0.9617])
W2_grad = np.array([
    [0.7501, 1.1736, 0.0, 0.0, 0.0],
    [0.0305, 0.0477, 0.0, 0.0, 0.0],
    [0.2055, 0.3216, 0.0, 0.0, 0.0],
    [-0.9862, -1.5429, 0.0, 0.0, 0.0],
])
W1_grad = np.outer(pre1_grad, x)
embedding_grad = np.array([
    [0.0, 0.0, 0.0],
    [3.1506, 1.3900, -0.6410],
    [1.5851, -0.9574, -0.9617],
    [0.0, 0.0, 0.0],
])

fig = plt.figure(figsize=(16, 10), dpi=170, constrained_layout=True)
grid = fig.add_gridspec(2, 3, height_ratios=[1.05, 1.0])
ax = fig.add_subplot(grid[0, :])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")

nodes = [
    (0.08, "x\n6 dims", x, x_grad),
    (0.25, "pre1\n5 dims", pre1, pre1_grad),
    (0.42, "h = ReLU(pre1)\n5 dims", h, h_grad),
    (0.59, "logits\n4 dims", logits, logits_grad),
    (0.76, "p = softmax\n4 dims", p, None),
    (0.92, "Loss\nscalar", np.array([4.0423]), np.array([1.0])),
]

def compact(values):
    return np.array2string(values, precision=2, suppress_small=True, separator=", ")

for position, (x_pos, title, forward, backward) in enumerate(nodes):
    width = 0.145 if position < 5 else 0.12
    box = FancyBboxPatch(
        (x_pos - width / 2, 0.58), width, 0.27,
        boxstyle="round,pad=0.012", facecolor="#E8F1FA", edgecolor="#356A9A", linewidth=1.6,
    )
    ax.add_patch(box)
    ax.text(x_pos, 0.78, title, ha="center", va="center", fontsize=10.5, weight="bold")
    ax.text(x_pos, 0.655, compact(forward), ha="center", va="center", fontsize=8.2)
    if backward is not None:
        grad_color = "#B33A3A" if np.any(backward != 0) else "#777777"
        ax.text(x_pos, 0.39, f"gradient\n{compact(backward)}", ha="center", va="center",
                fontsize=8.1, color=grad_color)

for left, right in zip(nodes[:-1], nodes[1:]):
    x_left, x_right = left[0], right[0]
    ax.add_patch(FancyArrowPatch((x_left + 0.075, 0.72), (x_right - 0.075, 0.72),
                                 arrowstyle="-|>", mutation_scale=14, color="#356A9A", linewidth=1.7))
    ax.add_patch(FancyArrowPatch((x_right - 0.075, 0.45), (x_left + 0.075, 0.45),
                                 arrowstyle="-|>", mutation_scale=14, color="#B33A3A", linewidth=1.7))

ax.text(0.02, 0.90, "Forward: values flow from input to Loss", color="#356A9A", fontsize=12, weight="bold")
ax.text(0.02, 0.27, "Backward: gradients flow from Loss to input", color="#B33A3A", fontsize=12, weight="bold")
ax.text(0.335, 0.29, "ReLU mask = [1, 1, 0, 0, 0]\nlast three paths are blocked",
        ha="center", va="center", fontsize=10, color="#555555",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F2F2F2", "edgecolor": "#999999"})

cmap = "RdBu_r"
norm = TwoSlopeNorm(vmin=-max(abs(W2_grad).max(), abs(W1_grad).max(), abs(embedding_grad).max()),
                    vcenter=0.0,
                    vmax=max(abs(W2_grad).max(), abs(W1_grad).max(), abs(embedding_grad).max()))

def heatmap(axis, data, title, rows, cols):
    image = axis.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    axis.set_title(title, fontsize=11, weight="bold")
    axis.set_xticks(range(len(cols)), cols)
    axis.set_yticks(range(len(rows)), rows)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            axis.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=7.7,
                      color="white" if abs(value) > 1.25 else "black")
    return image

ax_w2 = fig.add_subplot(grid[1, 0])
heatmap(ax_w2, W2_grad, "W2_grad = logits_grad outer h", ["I", "love", "AI", "today"], [f"h{i}" for i in range(5)])
ax_w1 = fig.add_subplot(grid[1, 1])
heatmap(ax_w1, W1_grad, "W1_grad = pre1_grad outer x", [f"pre{i}" for i in range(5)], [f"x{i}" for i in range(6)])
ax_emb = fig.add_subplot(grid[1, 2])
image = heatmap(ax_emb, embedding_grad, "x_grad split back to Embedding rows", ["I", "love", "AI", "today"], ["d0", "d1", "d2"])
fig.colorbar(image, ax=[ax_w2, ax_w1, ax_emb], shrink=0.82, label="gradient value")

fig.suptitle("One sample, one complete backward pass: [love, AI] -> today", fontsize=16, weight="bold")
output = Path(__file__).resolve().parent / "backprop_numeric_flow.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
