#!/usr/bin/env python3
"""生成第9章 tanh 与 Sigmoid 的对比曲线图。"""
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

x = np.linspace(-6, 6, 600)
sigmoid = 1 / (1 + np.exp(-x))
tanh = np.tanh(x)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=170, constrained_layout=True)

# --- 左图：Sigmoid ---
ax = axes[0]
ax.axhspan(0, 1, color="#FDECC8", alpha=0.5, label="值域 (0, 1)")
ax.plot(x, sigmoid, color="#B36B00", linewidth=2.6, label=r"$\sigma(x)=\dfrac{1}{1+e^{-x}}$")
ax.axhline(0, color="#999999", linewidth=0.8)
ax.axhline(1, color="#999999", linewidth=0.8, linestyle="--")
ax.axhline(0.5, color="#999999", linewidth=0.8, linestyle=":")
for px in [-2, 0, 2]:
    py = 1 / (1 + np.exp(-px))
    ax.plot(px, py, "o", color="#B36B00", markersize=6)
    ax.annotate(f"({px}, {py:.2f})", (px, py), textcoords="offset points",
                xytext=(8, -14 if px == 0 else 8), fontsize=9.5, color="#7A4A00")
ax.set_title("Sigmoid：输出永远落在 (0, 1)\n——像一个「通过比例」旋钮", fontsize=12.5, weight="bold")
ax.set_xlabel("输入 x", fontsize=10.5)
ax.set_ylabel(r"$\sigma(x)$", fontsize=10.5)
ax.set_ylim(-0.15, 1.15)
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.25)

# --- 右图：tanh ---
ax = axes[1]
ax.axhspan(-1, 1, color="#DCEAF7", alpha=0.6, label="值域 (-1, 1)")
ax.plot(x, tanh, color="#1F5FA0", linewidth=2.6, label=r"$\tanh(x)=\dfrac{e^x-e^{-x}}{e^x+e^{-x}}$")
ax.axhline(0, color="#999999", linewidth=0.8)
ax.axhline(1, color="#999999", linewidth=0.8, linestyle="--")
ax.axhline(-1, color="#999999", linewidth=0.8, linestyle="--")
for px in [-2, 0, 2]:
    py = np.tanh(px)
    ax.plot(px, py, "o", color="#1F5FA0", markersize=6)
    ax.annotate(f"({px}, {py:.2f})", (px, py), textcoords="offset points",
                xytext=(8, -14 if px == 0 else 8), fontsize=9.5, color="#123A63")
ax.set_title("tanh：输出落在 (-1, 1)，以 0 为中心\n——像一个「可正可负的状态值」", fontsize=12.5, weight="bold")
ax.set_xlabel("输入 x", fontsize=10.5)
ax.set_ylabel(r"$\tanh(x)$", fontsize=10.5)
ax.set_ylim(-1.15, 1.15)
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.25)

fig.suptitle("Sigmoid vs tanh：两种 S 形激活函数，分工完全不同", fontsize=15, weight="bold")
output = Path(__file__).resolve().parent / "tanh_sigmoid_curves.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
