#!/usr/bin/env python3
"""生成累计折扣回报与 Q-Learning 单步更新示意图。"""
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

rewards = np.array([1.0, 0.0, 10.0])
gamma = 0.9
weights = gamma ** np.arange(len(rewards))
contributions = rewards * weights

fig, (ax_timeline, ax_update) = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=170, constrained_layout=True)

x = np.arange(3)
ax_timeline.axhline(0, color="#777777", linewidth=0.8)
ax_timeline.plot(x, np.zeros_like(x), color="#777777", linewidth=1.4)
ax_timeline.scatter(x, np.zeros_like(x), s=100, color="#4C78A8", zorder=3)
for index, (reward, weight, contribution) in enumerate(zip(rewards, weights, contributions)):
    ax_timeline.annotate(
        f"R{index}={reward:g}\nweight={weight:.2f}\ncontribution={contribution:.2f}",
        (index, 0), xytext=(0, 38 if index % 2 == 0 else -70), textcoords="offset points",
        ha="center", va="center", fontsize=10,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#F3F6F8", "edgecolor": "#7393B3"},
        arrowprops={"arrowstyle": "->", "color": "#7393B3"},
    )
ax_timeline.set_xlim(-0.5, 2.5)
ax_timeline.set_ylim(-1.1, 1.1)
ax_timeline.set_yticks([])
ax_timeline.set_xticks(x, ["t", "t+1", "t+2"])
ax_timeline.set_title("累计回报：越远的奖励权重越小", weight="bold")
ax_timeline.text(0.5, 0.05, "G_t = 1 + 0.9×0 + 0.9²×10 = 9.1", transform=ax_timeline.transAxes,
                 ha="center", fontsize=11, weight="bold")

labels = ["即时奖励 r", "折扣后未来价值\nγ max Q(s',a')", "TD Target", "旧 Q(s,a)", "TD Error", "更新后 Q(s,a)"]
values = [-1.0, 4.5, 3.5, 2.0, 1.5, 2.15]
colors = ["#E45756", "#54A24B", "#4C78A8", "#9D755D", "#B279A2", "#F58518"]
bars = ax_update.bar(np.arange(len(labels)), values, color=colors, width=0.72)
ax_update.axhline(0, color="#777777", linewidth=0.8)
ax_update.set_xticks(np.arange(len(labels)), labels, rotation=18, ha="right")
ax_update.set_ylabel("数值")
ax_update.set_title("一次 Q-Learning 更新：α=0.1，γ=0.9", weight="bold")
ax_update.grid(axis="y", alpha=0.2)
ax_update.bar_label(bars, fmt="%.2f", padding=3, fontsize=9)
ax_update.text(0.5, 0.96, "Target = -1 + 0.9×5 = 3.5\nError = 3.5 - 2 = 1.5\nQ_new = 2 + 0.1×1.5 = 2.15",
               transform=ax_update.transAxes, ha="center", va="top", fontsize=10,
               bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#999999"})

fig.suptitle("强化学习公式最终都在回答：未来奖励如何折算到当前决策", fontsize=15, weight="bold")
output = Path(__file__).resolve().parent / "rl_return_timeline.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
