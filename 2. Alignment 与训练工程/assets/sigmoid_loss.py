"""绘制 Sigmoid 与成对损失曲线，用于第 5 章说明 0.693 的来源。

运行：python "assets/sigmoid_loss.py"
输出：assets/sigmoid_loss.png
"""

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

z = np.linspace(-6, 6, 800)
sigmoid = 1 / (1 + np.exp(-z))
loss = np.log1p(np.exp(-z))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=150)

# ---- 左：Sigmoid ----
ax1.plot(z, sigmoid, color="#1f77b4", lw=2.4)
ax1.axhline(0.5, color="gray", lw=1, ls="--", alpha=0.6)
ax1.scatter([0], [0.5], s=95, color="#d62728", zorder=5)
ax1.annotate(
    r"$\Delta=0\ \Rightarrow\ \sigma(0)=0.5$",
    xy=(0, 0.5), xytext=(-5.6, 0.70),
    fontsize=11, color="#d62728",
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
)
for z0, c in [(1.5, "#2ca02c"), (-1.5, "#ff7f0e")]:
    y0 = 1 / (1 + np.exp(-z0))
    ax1.scatter([z0], [y0], s=55, color=c, zorder=5)
    ax1.annotate(
        f"{y0:.3f}", xy=(z0, y0), xytext=(z0 + 0.25, y0 - 0.09),
        fontsize=9, color=c,
    )
ax1.set_title(r"Sigmoid：$\sigma(\Delta)=\dfrac{1}{1+e^{-\Delta}}$", fontsize=12)
ax1.set_xlabel(r"分数差 $\Delta=r(x,y_w)-r(x,y_l)$", fontsize=10.5)
ax1.set_ylabel(r"$\sigma(\Delta)$（chosen 更好的概率）", fontsize=10.5)
ax1.set_xlim(-6, 6)
ax1.set_ylim(-0.05, 1.12)
ax1.set_yticks([0, 0.182, 0.5, 0.818, 1])
ax1.grid(alpha=0.25)

# ---- 右：成对损失 ----
ax2.plot(z, loss, color="#9467bd", lw=2.4)
ax2.axhline(np.log(2), color="gray", lw=1, ls="--", alpha=0.6)
ax2.text(5.8, np.log(2) + 0.12, r"$\log 2\approx 0.693$", ha="right", fontsize=10, color="dimgray")
ax2.scatter([0], [np.log(2)], s=95, color="#d62728", zorder=5)
ax2.annotate(
    r"$\Delta=0\ \Rightarrow\ -\log 0.5\approx 0.693$",
    xy=(0, np.log(2)), xytext=(-5.7, 2.55),
    fontsize=11, color="#d62728",
    arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
)
for z0, c in [(1.5, "#2ca02c"), (-1.5, "#ff7f0e")]:
    y0 = np.log1p(np.exp(-z0))
    ax2.scatter([z0], [y0], s=55, color=c, zorder=5)
    ax2.annotate(
        f"{y0:.3f}", xy=(z0, y0), xytext=(z0 + 0.3, y0 + 0.12),
        fontsize=9, color=c,
    )
ax2.set_title(r"成对损失：$L=-\log\sigma(\Delta)=\log(1+e^{-\Delta})$", fontsize=12)
ax2.set_xlabel(r"分数差 $\Delta=r(x,y_w)-r(x,y_l)$", fontsize=10.5)
ax2.set_ylabel(r"$L$", fontsize=10.5)
ax2.set_xlim(-6, 6)
ax2.set_ylim(-0.08, 3.2)
ax2.set_yticks([0, 0.693, 1.0, 2.0, 3.0])
ax2.grid(alpha=0.25)

fig.tight_layout()
fig.savefig("assets/sigmoid_loss.png")
print("assets/sigmoid_loss.png")
