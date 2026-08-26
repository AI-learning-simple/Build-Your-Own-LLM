#!/usr/bin/env python3
"""生成第9章 RNN Cell 单步计算的数值流程图：h_t = tanh(Wxh·x_t + Whh·h_{t-1} + b)。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

FONT_CANDIDATES = [
    "PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC",
    "WenQuanYi Micro Hei", "Source Han Sans SC", "Arial Unicode MS",
]
available = {font.name for font in fm.fontManager.ttflist}
font = next((name for name in FONT_CANDIDATES if name in available), "DejaVu Sans")
plt.rcParams.update({"font.family": font, "axes.unicode_minus": False})

# 一个 2 维的具体数值示例：处理第 3 个 Token "AI"
x_t = np.array([2.0, 1.5])
h_prev = np.array([0.80, -0.60])
Wxh = np.array([[0.9, -0.7], [0.6, 0.8]])
Whh = np.array([[0.7, 0.3], [-0.5, 0.9]])
b = np.array([0.10, -0.20])

z = Wxh @ x_t + Whh @ h_prev + b
h_t = np.tanh(z)

fig, (ax_top, ax_bar) = plt.subplots(2, 1, figsize=(12.5, 8.6), dpi=170,
                                      gridspec_kw={"height_ratios": [1.15, 1]},
                                      constrained_layout=True)

# ---------- 上半部分：数值流程图 ----------
ax_top.set_xlim(0, 1)
ax_top.set_ylim(0, 1)
ax_top.axis("off")

boxes = [
    (0.09, "x_t\n(当前 Token: AI)", x_t, "#E8F1FA", "#356A9A"),
    (0.09, "h_{t-1}\n(上一时刻 Hidden)", h_prev, "#FDECC8", "#B36B00"),
    (0.40, "z = Wxh·x_t + Whh·h_{t-1} + b\n(Linear，先不加非线性)", z, "#F0E6F6", "#6A3D9A"),
    (0.72, "h_t = tanh(z)\n(新的 Hidden State)", h_t, "#DCEAF7", "#1F5FA0"),
]

y_positions = [0.72, 0.30, 0.51, 0.51]
widths = [0.22, 0.22, 0.30, 0.24]

def fmt(vec):
    return "[" + ", ".join(f"{v:.2f}" for v in vec) + "]"

positions = {}
for (x_pos, title, vec, face, edge), y_pos, width in zip(boxes, y_positions, widths):
    box = FancyBboxPatch((x_pos - width / 2, y_pos - 0.10), width, 0.20,
                          boxstyle="round,pad=0.014", facecolor=face, edgecolor=edge, linewidth=1.7)
    ax_top.add_patch(box)
    ax_top.text(x_pos, y_pos + 0.055, title, ha="center", va="center", fontsize=10, weight="bold")
    ax_top.text(x_pos, y_pos - 0.03, fmt(vec), ha="center", va="center", fontsize=10.5, color=edge)
    positions[title.split("\n")[0]] = (x_pos, y_pos, width)

# 箭头：x_t, h_{t-1} -> z
for key in ["x_t", "h_{t-1}"]:
    x0, y0, w0 = positions[key]
    x1, y1, w1 = positions["z = Wxh·x_t + Whh·h_{t-1} + b"]
    ax_top.add_patch(FancyArrowPatch((x0 + w0 / 2, y0), (x1 - w1 / 2, y1),
                                      arrowstyle="-|>", mutation_scale=15,
                                      color="#888888", linewidth=1.6, connectionstyle="arc3,rad=0.0"))
# 箭头：z -> h_t
x0, y0, w0 = positions["z = Wxh·x_t + Whh·h_{t-1} + b"]
x1, y1, w1 = positions["h_t = tanh(z)"]
ax_top.add_patch(FancyArrowPatch((x0 + w0 / 2, y0), (x1 - w1 / 2, y1),
                                  arrowstyle="-|>", mutation_scale=15,
                                  color="#1F5FA0", linewidth=2.0))
ax_top.text((x0 + x1) / 2 + 0.01, y0 + 0.12, "过 tanh 压缩到\n(-1, 1)", ha="center", fontsize=9.3,
            color="#1F5FA0")

ax_top.text(0.09, 0.10,
            f"Wxh =\n{np.array2string(Wxh, precision=2)}\nWhh =\n{np.array2string(Whh, precision=2)}\nb = {fmt(b)}",
            fontsize=8.6, color="#555555", family="monospace", va="top")

ax_top.set_title("RNN Cell 单步计算：一个具体的数值例子（处理 Token \u201cAI\u201d）", fontsize=13.5, weight="bold")

# ---------- 下半部分：柱状图，直观看 tanh 的压缩效果 ----------
dims = ["dim 1", "dim 2"]
width = 0.25
idx = np.arange(len(dims))
ax_bar.bar(idx - width, h_prev, width, label="h_{t-1}（旧记忆）", color="#B36B00", alpha=0.85)
ax_bar.bar(idx, z, width, label="z（Linear 之后，未压缩）", color="#6A3D9A", alpha=0.85)
ax_bar.bar(idx + width, h_t, width, label="h_t = tanh(z)（新记忆）", color="#1F5FA0", alpha=0.85)
ax_bar.axhline(1, color="#999999", linewidth=0.8, linestyle="--")
ax_bar.axhline(-1, color="#999999", linewidth=0.8, linestyle="--")
ax_bar.axhline(0, color="#CCCCCC", linewidth=0.8)
ax_bar.set_xticks(idx, dims)
ax_bar.set_ylabel("数值", fontsize=10.5)
ax_bar.set_title("z 已经超出 (-1, 1)，tanh 把它压回状态该有的范围", fontsize=12, weight="bold")
ax_bar.legend(loc="upper right", fontsize=9.5)
ax_bar.grid(axis="y", alpha=0.25)

output = Path(__file__).resolve().parent / "rnn_cell_numeric_step.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
print("z =", z, "h_t =", h_t)
