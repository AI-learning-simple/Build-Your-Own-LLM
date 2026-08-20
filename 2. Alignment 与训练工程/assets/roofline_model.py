#!/usr/bin/env python3
"""生成 GPU Roofline 模型示意图。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

FONT_CANDIDATES = ["PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Source Han Sans SC"]
FONT_PATHS = ["/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/STHeiti Light.ttc", "C:/Windows/Fonts/msyh.ttc", "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]
for font_path in FONT_PATHS:
    if Path(font_path).exists():
        fm.fontManager.addfont(font_path)
available = {font.name for font in fm.fontManager.ttflist}
font_name = next((name for name in FONT_CANDIDATES if name in available), None)
if font_name is None:
    installed_path = next((path for path in FONT_PATHS if Path(path).exists()), None)
    font_name = fm.FontProperties(fname=installed_path).get_name() if installed_path else "DejaVu Sans"
plt.rcParams["font.family"] = font_name
plt.rcParams["axes.unicode_minus"] = False

peak = 100.0  # TFLOPS
bandwidth = 1.0  # TB/s, numerically gives TFLOPS = AI × 1
intensity = np.logspace(-1, 4, 500)
roof = np.minimum(peak, bandwidth * intensity)

fig, ax = plt.subplots(figsize=(9.5, 6.3), dpi=170, constrained_layout=True)
ax.loglog(intensity, roof, color="#222222", linewidth=3, label="性能上限")
ax.loglog(intensity, bandwidth * intensity, "--", color="#4C78A8", linewidth=1.3, alpha=0.7, label="带宽上限")
ax.axhline(peak, color="#E45756", linestyle="--", linewidth=1.3, label="峰值算力")

points = {"Softmax": (1.0, 1.0), "LayerNorm": (2.0, 2.0), "大矩阵乘法": (300.0, 100.0)}
colors = ["#F58518", "#54A24B", "#B279A2"]
for (name, (x, y)), color in zip(points.items(), colors):
    ax.scatter([x], [y], s=110, color=color, edgecolor="black", zorder=4)
    ax.annotate(name, (x, y), xytext=(8, 8), textcoords="offset points", fontsize=10, weight="bold")

ridge = peak / bandwidth
ax.axvline(ridge, color="#777777", linestyle=":", linewidth=1.2)
ax.text(ridge * 1.08, 0.18, f"转折点 = {ridge:.0f} FLOP/Byte", rotation=90, va="bottom", color="#555555")
ax.text(1.0, 22, "Memory-Bound\n增加算力帮助有限", color="#4C78A8", fontsize=12, weight="bold")
ax.text(380, 24, "Compute-Bound\n受峰值算力限制", color="#E45756", fontsize=12, weight="bold", ha="center")
ax.set_xlabel("Arithmetic Intensity：每搬 1 Byte 数据完成多少次浮点运算（FLOP/Byte）")
ax.set_ylabel("可达到的性能（TFLOPS）")
ax.set_title("Roofline：性能上限由算力与带宽中更低的一项决定", weight="bold")
ax.grid(which="both", alpha=0.18)
ax.legend(loc="lower right")

output = Path(__file__).resolve().parent / "roofline_model.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
