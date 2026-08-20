#!/usr/bin/env python3
"""生成 8.1 节的优化器轨迹与学习率调度图。"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np

FONT_CANDIDATES = ["PingFang SC", "Microsoft YaHei", "Noto Sans CJK SC", "WenQuanYi Micro Hei", "Source Han Sans SC"]
available = {font.name for font in fm.fontManager.ttflist}
plt.rcParams["font.family"] = next((name for name in FONT_CANDIDATES if name in available), "DejaVu Sans")
plt.rcParams["axes.unicode_minus"] = False


def loss_fn(w):
    return 10 * w[..., 0] ** 2 + 0.1 * w[..., 1] ** 2


def grad_fn(w):
    return np.array([20 * w[0], 0.2 * w[1]])


def run_gd(steps=30, lr=0.05):
    w = np.array([1.0, 1.0])
    path = [w.copy()]
    for _ in range(steps):
        w -= lr * grad_fn(w)
        path.append(w.copy())
    return np.array(path)


def run_momentum(steps=30, lr=0.05, momentum=0.9):
    w = np.array([1.0, 1.0])
    velocity = np.zeros_like(w)
    path = [w.copy()]
    for _ in range(steps):
        velocity = momentum * velocity - lr * grad_fn(w)
        w += velocity
        path.append(w.copy())
    return np.array(path)


def run_adam(steps=30, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    w = np.array([1.0, 1.0])
    m = np.zeros_like(w)
    v = np.zeros_like(w)
    path = [w.copy()]
    for t in range(1, steps + 1):
        grad = grad_fn(w)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad ** 2
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        w -= lr * m_hat / (np.sqrt(v_hat) + eps)
        path.append(w.copy())
    return np.array(path)


def lr_schedule(step, total_steps=1000, warmup_steps=100, peak_lr=1e-3):
    if step < warmup_steps:
        return peak_lr * step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return peak_lr * 0.5 * (1 + np.cos(np.pi * progress))

paths = {
    "GD (lr=0.05)": (run_gd(), "#1F77B4"),
    "Momentum (lr=0.05)": (run_momentum(), "#D95F02"),
    "Adam (lr=0.01)": (run_adam(), "#2E8B57"),
}

fig, (ax_path, ax_lr) = plt.subplots(1, 2, figsize=(14, 5.8), dpi=170, constrained_layout=True)
axis = np.linspace(-1.1, 1.1, 500)
W0, W1 = np.meshgrid(axis, axis)
loss = 10 * W0 ** 2 + 0.1 * W1 ** 2
levels = np.geomspace(0.002, 12, 18)
contours = ax_path.contour(W0, W1, loss, levels=levels, cmap="Greys", linewidths=0.8)
ax_path.clabel(contours, levels=levels[::3], inline=True, fontsize=7, fmt="%.2g")

for name, (path, color) in paths.items():
    ax_path.plot(path[:, 0], path[:, 1], "-o", color=color, linewidth=2.1, markersize=3.2, label=name)
    final = path[-1]
    ax_path.annotate(f"({final[0]:.3f}, {final[1]:.3f})", final, xytext=(7, 6),
                     textcoords="offset points", fontsize=8, color=color)

ax_path.scatter([0], [0], marker="*", s=170, color="#FFD23F", edgecolor="black", zorder=5, label="minimum")
ax_path.set(xlabel="w0 (steep direction)", ylabel="w1 (flat direction)",
            title="30 update steps on L = 10 w0^2 + 0.1 w1^2")
ax_path.set_aspect("equal")
ax_path.grid(alpha=0.2)
ax_path.legend(fontsize=8.5, loc="lower left")

steps = np.arange(1000)
rates = np.array([lr_schedule(step) for step in steps])
ax_lr.plot(steps, rates, color="#6A3D9A", linewidth=2.5)
ax_lr.fill_between(steps[:101], rates[:101], alpha=0.22, color="#4C78A8", label="linear warmup")
ax_lr.fill_between(steps[100:], rates[100:], alpha=0.18, color="#F58518", label="cosine decay")
ax_lr.axvline(100, color="#555555", linestyle="--", linewidth=1)
ax_lr.scatter([0, 50, 100, 300, 600, 999], [lr_schedule(x) for x in [0, 50, 100, 300, 600, 999]],
              color="#6A3D9A", zorder=3)
ax_lr.ticklabel_format(axis="y", style="sci", scilimits=(-3, -3))
ax_lr.set(xlabel="training step", ylabel="learning rate",
          title="Warmup + Cosine Decay (peak_lr = 1e-3)")
ax_lr.grid(alpha=0.25)
ax_lr.legend()

fig.suptitle("Optimization has two scales: parameter-space path and step-size schedule", fontsize=15, weight="bold")
output = Path(__file__).resolve().parent / "optimizer_trajectories.png"
fig.savefig(output, bbox_inches="tight")
print(f"saved to {output}")
