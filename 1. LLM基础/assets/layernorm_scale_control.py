#!/usr/bin/env python3
"""生成第11章 LayerNorm 一节的数值图：
左图：残差逐层叠加增量时激活尺度持续放大（按正文 10→12→18→31→52 的增速 ×1.5 外推 20 层），
     Pre-LN 结构在每个子层入口归一化，送入 Attention/FFN 的尺度恒 ≈ 1；
右图：正文示例 [100, 120, 140] 归一化为 [-1.22, 0.00, 1.22]——大小顺序保留，尺度回到 ±1.22。
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

# 正交标注样式：先水平（angleA=0）出文字块，再垂直（angleB=90）落到被标注点，没有斜角
ORTHO_ARROW = dict(arrowstyle="->", color="#B33A3A", lw=1.1,
                   connectionstyle="angle3,angleA=0,angleB=90")

fig, (ax_scale, ax_bar) = plt.subplots(1, 2, figsize=(13, 5.3), dpi=170, constrained_layout=True)

# 左图：逐层激活尺度（对数坐标）
n = np.arange(0, 21)
growth = 10 * 1.5 ** n
ax_scale.plot(n, growth, color="#B33A3A", linewidth=2.2,
              label="无 LayerNorm：残差逐层叠加，尺度持续放大")
ax_scale.plot(n, np.ones_like(n, dtype=float), color="#3A8F5A", linewidth=2.4, linestyle="--",
              label="Pre-LN：送入每个子层的尺度 ≈ 1")

# 正文开头的数值例子：10→12→18→31→52（前 5 层的实际数字）
text_points = np.array([10, 12, 18, 31, 52], dtype=float)
ax_scale.scatter(np.arange(5), text_points, color="#7A3FA0", s=42, zorder=5,
                 label="正文例子：10→12→18→31→52")
for i, v in enumerate(text_points):
    ax_scale.annotate(f"{v:.0f}", xy=(i, v), xytext=(i, v * 1.5), ha="center",
                      fontsize=8.5, color="#7A3FA0", weight="bold")

ax_scale.set_yscale("log")
ax_scale.annotate("按同样增速外推：\n第 20 层 10×1.5²⁰ ≈ 3.3×10⁴",
                  xy=(20, 10 * 1.5 ** 20), xytext=(8.5, 2.0e2),
                  fontsize=9.5, color="#B33A3A", arrowprops=ORTHO_ARROW)
ax_scale.set_xlabel("层数", fontsize=10.3)
ax_scale.set_ylabel("激活尺度（对数坐标）", fontsize=10.5)
ax_scale.set_title("尺度漂移：越叠越大 vs 每层拉回", fontsize=12.3, weight="bold")
ax_scale.set_ylim(3, 2.0e5)
ax_scale.legend(loc="lower left", fontsize=9.0, framealpha=0.95)  # 左下角：增长曲线从左下往右上走，实测不压线
ax_scale.grid(alpha=0.3, which="both")

# 右图：正文示例 [100, 120, 140] 归一化前后
raw = np.array([100.0, 120.0, 140.0])
normalized = (raw - raw.mean()) / raw.std()  # 与 nn.LayerNorm 一致：总体标准差 → [-1.22, 0.00, 1.22]

labels = ["第 1 维\n原值 100", "第 2 维\n原值 120", "第 3 维\n原值 140"]
colors = ["#1F5FA0" if v >= 0 else "#B33A3A" for v in normalized]
bars = ax_bar.bar(labels, normalized, color=colors, alpha=0.85, width=0.55)
# 修复：中间那根柱子归一化后正好是 0.00，柱子高度为 0 时视觉上完全不可见（原图曾出现"漏了一根柱子"的问题）。
# 用一个实心圆点标出每根柱子的真实取值位置，即使高度为 0 也能看到一个点，避免看起来像缺失。
ax_bar.scatter(range(len(normalized)), normalized, color=colors, s=55, zorder=5, edgecolor="white", linewidth=1.0)
for bar, v in zip(bars, normalized):
    ax_bar.text(bar.get_x() + bar.get_width() / 2,
                v + (0.16 if v >= 0 else -0.16), f"{v:.2f}",
                ha="center", va="bottom" if v >= 0 else "top",
                fontsize=11, weight="bold")
ax_bar.axhline(0, color="#666666", linewidth=0.9)
ax_bar.set_ylim(-1.9, 1.9)
ax_bar.set_ylabel("归一化后的值", fontsize=10.5)
ax_bar.set_title("示例 [100, 120, 140] → [-1.22, 0.00, 1.22]：\n顺序保留，尺度回到 ±1.22", fontsize=12.3, weight="bold")
ax_bar.tick_params(axis="x", labelsize=9.3)
ax_bar.grid(axis="y", alpha=0.3)

fig.suptitle("Layer Normalization：控制逐层尺度，同时保留每个维度的大小关系",
             fontsize=14.5, weight="bold")
output = Path(__file__).resolve().parent / "layernorm_scale_control.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
