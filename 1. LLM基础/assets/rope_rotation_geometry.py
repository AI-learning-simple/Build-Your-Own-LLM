#!/usr/bin/env python3
"""生成 12.1 节的 RoPE 二维旋转与相对位置图。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

q = np.array([1.0, 0.5, -0.3, 0.8])
k = np.array([0.2, -0.7, 1.1, 0.4])
frequencies = np.array([1.0, 0.01])
positions = [(2, 5, "shift 0", "#1F77B4"), (102, 105, "shift +100", "#D95F02")]


def rotate(pair, angle):
    matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    return matrix @ pair


def rope(vector, position):
    result = vector.copy()
    for pair_index, frequency in enumerate(frequencies):
        start = 2 * pair_index
        result[start:start + 2] = rotate(vector[start:start + 2], position * frequency)
    return result

q2, k5 = rope(q, 2), rope(k, 5)
q102, k105 = rope(q, 102), rope(k, 105)
dot_2_5 = q2 @ k5
dot_102_105 = q102 @ k105

fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.5), dpi=170)
fig.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.19, wspace=0.24)
for pair_index, axis in enumerate(axes):
    start = 2 * pair_index
    max_radius = max(np.linalg.norm(q[start:start + 2]), np.linalg.norm(k[start:start + 2])) * 1.35
    circle = plt.Circle((0, 0), max_radius / 1.35, fill=False, linestyle=":", color="#999999")
    axis.add_patch(circle)
    for q_pos, k_pos, label, color in positions:
        q_rot = rope(q, q_pos)[start:start + 2]
        k_rot = rope(k, k_pos)[start:start + 2]
        axis.arrow(0, 0, q_rot[0], q_rot[1], width=0.012, head_width=0.08, head_length=0.10,
                   length_includes_head=True, color=color, alpha=0.95)
        axis.arrow(0, 0, k_rot[0], k_rot[1], width=0.006, head_width=0.07, head_length=0.09,
                   length_includes_head=True, color=color, alpha=0.55, linestyle="--")
        axis.text(q_rot[0] * 1.08, q_rot[1] * 1.08, f"q@{q_pos}", color=color, fontsize=9, weight="bold")
        axis.text(k_rot[0] * 1.08, k_rot[1] * 1.08, f"k@{k_pos}", color=color, fontsize=9)
    axis.axhline(0, color="#AAAAAA", linewidth=0.8)
    axis.axvline(0, color="#AAAAAA", linewidth=0.8)
    axis.set_xlim(-max_radius, max_radius)
    axis.set_ylim(-max_radius, max_radius)
    axis.set_aspect("equal")
    axis.grid(alpha=0.18)
    axis.set_title(f"dimension pair ({start}, {start + 1}), frequency = {frequencies[pair_index]:.2f}", weight="bold")
    axis.set_xlabel(f"dimension {start}")
    axis.set_ylabel(f"dimension {start + 1}")

fig.text(0.5, 0.055,
         "solid arrows: q; dashed arrows: k.  Both groups have directed displacement k_pos - q_pos = +3.\n"
         f"full 4D dot products: (2, 5) = {dot_2_5:.6f}, (102, 105) = {dot_102_105:.6f}",
         ha="center", fontsize=11,
         bbox={"boxstyle": "round,pad=0.45", "facecolor": "#F4F4F4", "edgecolor": "#888888"})
fig.suptitle("RoPE rotates each dimension pair; the relative rotation depends on position difference", fontsize=14.5, weight="bold")
output = Path(__file__).resolve().parent / "rope_rotation_geometry.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
