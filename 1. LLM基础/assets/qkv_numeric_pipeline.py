#!/usr/bin/env python3
"""生成 10.5 节的 Q/K/V 数值流水线图。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

FONT_CANDIDATES = [
    "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC",
    "WenQuanYi Micro Hei", "Source Han Sans SC", "Arial Unicode MS",
]
available = {font.name for font in fm.fontManager.ttflist}
font = next((name for name in FONT_CANDIDATES if name in available), "DejaVu Sans")
plt.rcParams.update({"font.family": font, "axes.unicode_minus": False})

TOKENS = ["The", "cat", "is", "sleeping"]
X = np.array([[0.1, -0.2], [1.0, 0.5], [-0.1, 0.1], [0.9, 0.6]])
WQ = np.array([[1.0, -0.1], [0.3, 0.8]])
WK = np.array([[0.8, 0.2], [-0.2, 1.0]])
WV = np.array([[0.5, 0.5], [-0.5, 1.0]])
Q, K, V = X @ WQ, X @ WK, X @ WV
query_index = 1
scores = Q[query_index] @ K.T / np.sqrt(2)
weights = np.exp(scores - scores.max())
weights /= weights.sum()
output_vector = weights @ V

fig = plt.figure(figsize=(16, 6.6), dpi=170, constrained_layout=True)
grid = fig.add_gridspec(2, 6, height_ratios=[1.25, 0.85], hspace=0.12)


def draw_matrix(axis, matrix, title, row_labels=None, col_labels=None, cmap="RdBu_r", limits=None):
    if limits is None:
        bound = max(abs(matrix).max(), 1e-9)
        limits = (-bound, bound)
    image = axis.imshow(matrix, cmap=cmap, vmin=limits[0], vmax=limits[1], aspect="auto")
    axis.set_title(title, fontsize=10.5, weight="bold")
    axis.set_xticks(range(matrix.shape[1]), col_labels or [f"d{i}" for i in range(matrix.shape[1])])
    axis.set_yticks(range(matrix.shape[0]), row_labels or range(matrix.shape[0]))
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    return image

ax_x = fig.add_subplot(grid[0, 0])
draw_matrix(ax_x, X, "Input X\n[4 tokens, 2 dims]", TOKENS)
ax_q = fig.add_subplot(grid[0, 1])
draw_matrix(ax_q, Q, "Q = X WQ", TOKENS)
ax_k = fig.add_subplot(grid[0, 2])
draw_matrix(ax_k, K, "K = X WK", TOKENS)
ax_v = fig.add_subplot(grid[0, 3])
draw_matrix(ax_v, V, "V = X WV", TOKENS)

ax_scores = fig.add_subplot(grid[0, 4])
draw_matrix(ax_scores, scores[None, :], "cat scores\nq_cat K^T / sqrt(2)", ["cat query"], TOKENS)
ax_weights = fig.add_subplot(grid[0, 5])
draw_matrix(ax_weights, weights[None, :], "Softmax weights", ["cat query"], TOKENS, cmap="Blues", limits=(0, weights.max()))

for axis, label in [(ax_x, "four input hidden states"), (ax_q, "query role"), (ax_k, "matching role"), (ax_v, "content role")]:
    axis.text(0.5, -0.20, label, transform=axis.transAxes, ha="center", va="top",
              fontsize=8.5, color="#555555", clip_on=False)

ax_formula = fig.add_subplot(grid[1, :4])
ax_formula.axis("off")
formula = (
    "Three independent 2 x 2 projections:  "
    f"WQ={WQ.tolist()}   WK={WK.tolist()}   WV={WV.tolist()}\n"
    "q_cat x K^T / sqrt(d_k) -> scores -> softmax -> weights   ->   "
    f"weights x V -> C = [{output_vector[0]:.3f}, {output_vector[1]:.3f}]   (C @ W_O -> Output)\n"
    f"scores  = {np.array2string(scores, precision=3)}\n"
    f"weights = {np.array2string(weights, precision=3)}"
)
ax_formula.text(0.03, 0.95, formula, ha="left", va="top", fontsize=10.5,
                linespacing=1.3, transform=ax_formula.transAxes,
                bbox={"boxstyle": "round,pad=0.7", "facecolor": "#F3F6F8", "edgecolor": "#7393B3"})

ax_out = fig.add_subplot(grid[1, 4:])
ax_out.set_anchor("N")
ax_out.bar([0, 1], output_vector, color=["#4C78A8", "#F58518"], width=0.55)
ax_out.axhline(0, color="#777777", linewidth=0.8)
ax_out.set_xticks([0, 1], ["C 第 0 维", "C 第 1 维"])
ax_out.set_ylabel("weighted value")
ax_out.set_title(f"cat 的 C = [{output_vector[0]:.3f}, {output_vector[1]:.3f}]", weight="bold")
ax_out.grid(axis="y", alpha=0.2)

fig.suptitle("Q decides what to match, K is matched, V supplies the output", fontsize=13.5, weight="bold")
output = Path(__file__).resolve().parent / "qkv_numeric_pipeline.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
