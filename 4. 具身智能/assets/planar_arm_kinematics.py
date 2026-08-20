#!/usr/bin/env python3
"""生成二连杆机械臂正逆运动学几何图。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle

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

l1 = l2 = 1.0

def points(theta1, theta2):
    joint = np.array([l1 * np.cos(theta1), l1 * np.sin(theta1)])
    end = joint + np.array([l2 * np.cos(theta1 + theta2), l2 * np.sin(theta1 + theta2)])
    return joint, end

fig, (ax_fk, ax_ik) = plt.subplots(1, 2, figsize=(13.5, 6.2), dpi=170, constrained_layout=True)

theta1, theta2 = np.deg2rad(0), np.deg2rad(90)
joint, end = points(theta1, theta2)
ax_fk.plot([0, joint[0], end[0]], [0, joint[1], end[1]], "-o", color="#1F77B4", linewidth=4, markersize=8)
ax_fk.plot([joint[0], end[0]], [joint[1], joint[1]], "--", color="#F58518", linewidth=1.5)
ax_fk.plot([end[0], end[0]], [joint[1], end[1]], "--", color="#54A24B", linewidth=1.5)
ax_fk.add_patch(Arc(joint, 0.45, 0.45, theta1=0, theta2=90, color="#D95F02", linewidth=2))
ax_fk.text(joint[0] + 0.18, joint[1] + 0.14, "θ₂=90°", color="#D95F02", fontsize=10)
ax_fk.text(0.45, -0.13, "l₁ cos θ₁ = 1", ha="center", color="#F58518")
ax_fk.text(1.08, 0.5, "l₂ sin(θ₁+θ₂) = 1", rotation=90, va="center", color="#54A24B")
ax_fk.text(end[0] + 0.08, end[1] + 0.08, "end=(1,1)", weight="bold")
ax_fk.set_title("正运动学：各连杆的 x/y 投影相加", weight="bold")

# Same target, two inverse-kinematics solutions.
target = np.array([1.0, 1.0])
cos_t2 = (target @ target - l1**2 - l2**2) / (2 * l1 * l2)
for sign, color, label in [(1, "#4C78A8", "肘下解"), (-1, "#E45756", "肘上解")]:
    t2 = sign * np.arccos(cos_t2)
    t1 = np.arctan2(target[1], target[0]) - np.arctan2(l2 * np.sin(t2), l1 + l2 * np.cos(t2))
    j, e = points(t1, t2)
    ax_ik.plot([0, j[0], e[0]], [0, j[1], e[1]], "-o", color=color, linewidth=3, markersize=7, label=f"{label}: θ₁={np.rad2deg(t1):.0f}°, θ₂={np.rad2deg(t2):.0f}°")
workspace = Circle((0, 0), l1 + l2, fill=False, linestyle="--", color="#999999", linewidth=1.2)
ax_ik.add_patch(workspace)
ax_ik.scatter(*target, marker="*", s=180, color="#FFD23F", edgecolor="black", zorder=5)
ax_ik.text(target[0] + 0.08, target[1] + 0.08, "同一目标 (1,1)", weight="bold")
ax_ik.set_title("逆运动学：同一目标可能有多组关节角", weight="bold")
ax_ik.legend(loc="lower left", fontsize=9)

for ax in (ax_fk, ax_ik):
    ax.axhline(0, color="#AAAAAA", linewidth=0.8)
    ax.axvline(0, color="#AAAAAA", linewidth=0.8)
    ax.set_aspect("equal")
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.set_xlabel("世界坐标 x")
    ax.set_ylabel("世界坐标 y")
    ax.grid(alpha=0.18)

fig.suptitle("Joint Space 与 Task Space 由运动学连接", fontsize=15, weight="bold")
output = Path(__file__).resolve().parent / "planar_arm_kinematics.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
