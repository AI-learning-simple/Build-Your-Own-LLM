#!/usr/bin/env python3
"""生成针孔相机反投影的数值几何图。"""
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

fx = fy = 500.0
cx, cy = 320.0, 240.0
u, v, Z = 420.0, 240.0, 2.5
X = (u - cx) * Z / fx
Y = (v - cy) * Z / fy

fig = plt.figure(figsize=(13.5, 5.8), dpi=170, constrained_layout=True)
ax = fig.add_subplot(1, 2, 1, projection="3d")
ax_img = fig.add_subplot(1, 2, 2)

# Camera ray and point in X-Z plane, embedded in 3D.
ax.scatter([0], [0], [0], color="#222222", s=60, label="相机光心")
ax.plot([0, X], [0, Y], [0, Z], color="#E45756", linewidth=3, label="像素对应射线")
ax.scatter([X], [Y], [Z], marker="*", s=180, color="#FFD23F", edgecolor="black", label=f"三维点 ({X:.1f},{Y:.1f},{Z:.1f})m")
ax.plot([0, X], [0, 0], [Z, Z], "--", color="#4C78A8", linewidth=1.4)
ax.plot([X, X], [0, 0], [0, Z], "--", color="#54A24B", linewidth=1.4)
ax.text(X / 2, 0, Z + 0.08, f"X={X:.1f}m", color="#4C78A8")
ax.set(xlabel="X（右）/m", ylabel="Y（下）/m", zlabel="Z（前）/m", title="像素 + 深度确定一条射线上的三维点")
ax.view_init(elev=24, azim=-62)
ax.legend(fontsize=8)

ax_img.set_xlim(0, 640)
ax_img.set_ylim(480, 0)
ax_img.scatter([cx], [cy], color="#4C78A8", s=80, label=f"主点 ({cx:.0f},{cy:.0f})")
ax_img.scatter([u], [v], marker="*", color="#FFD23F", edgecolor="black", s=170, label=f"像素 ({u:.0f},{v:.0f})")
ax_img.plot([cx, u], [cy, v], "--", color="#E45756", linewidth=2)
ax_img.annotate(f"水平偏移 u-cx={u-cx:.0f}px", ((cx+u)/2, cy), xytext=(0, -35), textcoords="offset points", ha="center", arrowprops={"arrowstyle": "->"})
ax_img.set_xlabel("像素 u")
ax_img.set_ylabel("像素 v")
ax_img.set_title("图像平面：偏离主点 100 像素", weight="bold")
ax_img.grid(alpha=0.2)
ax_img.legend()
ax_img.text(0.5, 0.06, "X=(420-320)×2.5/500=0.5m\nY=(240-240)×2.5/500=0m",
            transform=ax_img.transAxes, ha="center", fontsize=11, weight="bold",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#999999"})

fig.suptitle("反投影：像素偏移按深度和焦距换算为真实空间偏移", fontsize=15, weight="bold")
output = Path(__file__).resolve().parent / "camera_backprojection.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
