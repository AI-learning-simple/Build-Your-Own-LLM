#!/usr/bin/env python3
"""生成第11章 Dropout 一节的数值图：
左图：输入 [1,1,1,1,1]、p=0.4，六次前向的输出——每次被关闭的位置都不同，存活者放大到 1/(1-p) ≈ 1.67；
右图：每个位置重复一万次取平均 ≈ 1.0——缩放保证的是"期望不变"，单次前向仍是随机制造缺陷。
"""
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

rng = np.random.default_rng(7)
p = 0.4
scale = 1 / (1 - p)  # ≈ 1.667

fig, (ax_mask, ax_avg) = plt.subplots(1, 2, figsize=(13, 5.3), dpi=170, constrained_layout=True)

# 左图：六次前向，每次的关闭位置都不同（热力图 + 数值标注）
trials, units = 6, 5
closed = rng.random((trials, units)) < p
out = np.where(closed, 0.0, scale)

im = ax_mask.imshow(out, cmap="Blues", aspect="auto", vmin=0, vmax=scale)
for i in range(trials):
    for j in range(units):
        v = out[i, j]
        ax_mask.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11,
                     weight="bold", color="white" if v > 0 else "#555555")
ax_mask.set_xticks(range(units), [f"神经元 {j + 1}" for j in range(units)], fontsize=9.3)
ax_mask.set_yticks(range(trials), [f"第 {i + 1} 次前向" for i in range(trials)], fontsize=9.3)
ax_mask.set_title("同一输入 [1,1,1,1,1]，p=0.4：\n每次关闭的位置都不同，存活者 = 1/(1-p) ≈ 1.67",
                  fontsize=12.3, weight="bold")

# 右图：每个位置重复一万次取平均，回到 ≈ 1.0（期望不变）
closed_big = rng.random((10000, units)) < p
avg = np.where(closed_big, 0.0, scale).mean(axis=0)

bars = ax_avg.bar([f"神经元 {j + 1}" for j in range(units)], avg,
                  color="#3A8F5A", alpha=0.85, width=0.55)
for bar, v in zip(bars, avg):
    ax_avg.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", va="bottom", fontsize=10.5, weight="bold")
ax_avg.axhline(1.0, color="#999999", linestyle="--", linewidth=1.2)
ax_avg.text(4.35, 1.03, "期望 = 1.0", fontsize=9.5, color="#666666", ha="right")
ax_avg.set_ylim(0, 1.25)
ax_avg.set_ylabel("一万次前向的平均输出", fontsize=10.5)
ax_avg.set_title("存活概率 0.6 × 放大 1.67 = 1.00：\n缩放保证的是期望不变", fontsize=12.3, weight="bold")
ax_avg.tick_params(axis="x", labelsize=9.3)
ax_avg.grid(axis="y", alpha=0.3)

fig.suptitle("Dropout：单次前向随机关闭，大量重复后期望保持不变", fontsize=14.5, weight="bold")
output = Path(__file__).resolve().parent / "dropout_random_mask.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
