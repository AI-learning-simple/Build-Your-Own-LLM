#!/usr/bin/env python3
"""生成第9章 BPTT 梯度消失/爆炸一节的数值曲线图：
重复乘同一个 Jacobian（这里简化为标量系数 r）时，梯度大小随回传步数 T 呈指数变化。
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

T = np.arange(0, 21)
cases = [
    (0.5, "#B33A3A", "r = 0.5（|Jacobian| < 1）→ 梯度消失", "vanishing"),
    (1.0, "#3A8F5A", "r = 1.0（|Jacobian| = 1）→ 恰好稳定", "stable"),
    (1.3, "#1F5FA0", "r = 1.3（|Jacobian| > 1）→ 梯度爆炸", "exploding"),
]

fig, (ax_log, ax_bar) = plt.subplots(1, 2, figsize=(13, 5.3), dpi=170, constrained_layout=True)

for r, color, label, _ in cases:
    grad = r ** T
    ax_log.plot(T, grad, marker="o", markersize=4, color=color, linewidth=2.2, label=label)
ax_log.set_yscale("log")
ax_log.axhline(1, color="#999999", linestyle="--", linewidth=0.9)
ax_log.set_xlabel("反向回传的步数 T（∂h_T/∂h_0 要连乘 T 次 Jacobian）", fontsize=10.3)
ax_log.set_ylabel("梯度大小（对数坐标）", fontsize=10.5)
ax_log.set_title("同一个系数连乘 T 次：\n<1 指数衰减，>1 指数爆炸", fontsize=12.3, weight="bold")
ax_log.legend(loc="upper left", fontsize=9.3)
ax_log.grid(alpha=0.3, which="both")

# 右图：T=10 时三种情况的具体数值对比（线性柱状图，直观看差距有多夸张）
labels = [c[2].split("→")[1].strip() for c in cases]
values_at_10 = [r ** 10 for r, *_ in cases]
colors = [c[1] for c in cases]
bars = ax_bar.bar(labels, values_at_10, color=colors, alpha=0.85)
ax_bar.set_yscale("log")
for bar, v in zip(bars, values_at_10):
    ax_bar.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.4g}", ha="center",
                va="bottom" if v >= 1 else "top", fontsize=10, weight="bold")
ax_bar.set_ylabel("T = 10 步时的梯度大小", fontsize=10.5)
ax_bar.set_title("回传 10 步之后：\n0.5¹⁰ ≈ 0.001，1.3¹⁰ ≈ 13.8", fontsize=12.3, weight="bold")
ax_bar.tick_params(axis="x", labelsize=9)
ax_bar.grid(axis="y", alpha=0.3, which="both")

fig.suptitle("BPTT 的核心风险：反复乘同一个 Jacobian，指数级放大或缩小", fontsize=14.5, weight="bold")
output = Path(__file__).resolve().parent / "bptt_gradient_scaling.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
