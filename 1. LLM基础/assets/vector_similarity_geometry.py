#!/usr/bin/env python3
"""生成 10.2 节的向量相似度几何图。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

vectors = {
    "cat": np.array([1.0, 0.5]),
    "sleeping": np.array([0.9, 0.6]),
    "long vector": np.array([3.0, -1.0]),
    "The": np.array([0.1, -0.2]),
    "is": np.array([-0.1, 0.1]),
}
query = vectors["cat"]


def cosine(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

names = ["sleeping", "long vector", "The", "is"]
dots = [float(query @ vectors[name]) for name in names]
cosines = [cosine(query, vectors[name]) for name in names]
colors = ["#2E8B57", "#D95F02", "#4C78A8", "#8E6C8A"]

fig, (ax_vec, ax_score) = plt.subplots(1, 2, figsize=(13.5, 5.8), dpi=170, constrained_layout=True)

for name, color in zip(["cat", *names], ["#222222", *colors]):
    vec = vectors[name]
    ax_vec.arrow(0, 0, vec[0], vec[1], width=0.012, head_width=0.10, head_length=0.13,
                 length_includes_head=True, color=color, alpha=0.9)
    offset = np.array([0.06, 0.06 if vec[1] >= 0 else -0.12])
    ax_vec.text(*(vec + offset), name, color=color, fontsize=10, weight="bold")

ax_vec.axhline(0, color="#999999", linewidth=0.8)
ax_vec.axvline(0, color="#999999", linewidth=0.8)
ax_vec.grid(alpha=0.22)
ax_vec.set_xlim(-0.45, 3.55)
ax_vec.set_ylim(-1.35, 1.35)
ax_vec.set_aspect("equal")
ax_vec.set(xlabel="dimension 1", ylabel="dimension 2",
           title="Direction and length are different geometric properties")

positions = np.arange(len(names))
width = 0.36
bars_dot = ax_score.bar(positions - width / 2, dots, width, label="dot product", color=colors, alpha=0.88)
bars_cos = ax_score.bar(positions + width / 2, cosines, width, label="cosine similarity", color=colors, alpha=0.42,
                        edgecolor=colors, linewidth=1.4)
ax_score.axhline(0, color="#777777", linewidth=0.8)
ax_score.set_xticks(positions, names, rotation=12)
ax_score.set_ylabel("score")
ax_score.set_title("Same query: cat = [1.0, 0.5]")
ax_score.grid(axis="y", alpha=0.22)
ax_score.legend()
for bars in (bars_dot, bars_cos):
    ax_score.bar_label(bars, fmt="%.2f", padding=3, fontsize=8.5)

ax_score.text(0.5, 0.96,
              "sleeping: nearly same direction\nlong vector: larger dot product, lower cosine",
              transform=ax_score.transAxes, va="top", ha="center", fontsize=9.5,
              bbox={"boxstyle": "round,pad=0.4", "facecolor": "#F5F5F5", "edgecolor": "#999999"})
fig.suptitle("Dot product mixes direction with length; cosine removes the length factor", fontsize=15, weight="bold")
output = Path(__file__).resolve().parent / "vector_similarity_geometry.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
