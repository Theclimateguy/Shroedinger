#!/usr/bin/env python3
"""Illustrated scheme for article 3: the inheritance law and the principle of
regionalisation, drawn "on the fingers" (eddies, clouds, wind). Draft for the
author to redraw. Output: manuscript_v2/article3/figures/ru_fig1_scheme_draw.png"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent.parent / "manuscript_v2" / "article3" / "figures" / "ru_fig1_scheme_draw.png"
BLUE, DBLUE, RED, GREY, SAND, SEA = "#3b6fb6", "#1f3f73", "#c0392b", "#7f8c8d", "#e9dcc3", "#d7e8f5"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def spiral(ax, x, y, r, color, lw=1.4, turns=1.6, sense=1):
    t = np.linspace(0, 2 * np.pi * turns, 200)
    rr = r * (0.15 + 0.85 * t / t[-1])
    ax.plot(x + rr * np.cos(sense * t), y + rr * np.sin(sense * t), color=color, lw=lw, solid_capstyle="round")


def cloud(ax, x, y, s=1.0, color="white", ec=GREY, tall=False):
    parts = [(0, 0, 0.55), (-0.45, -0.1, 0.4), (0.45, -0.1, 0.4), (-0.15, 0.3, 0.42), (0.25, 0.32, 0.38)]
    if tall:
        parts += [(0.0, 0.75, 0.45), (0.15, 1.15, 0.4), (-0.1, 1.5, 0.35)]
    for dx, dy, r in parts:
        ax.add_patch(Circle((x + dx * s, y + dy * s), r * s, fc=color, ec=ec, lw=0.8, zorder=3))
    ax.add_patch(Rectangle((x - 0.85 * s, y - 0.5 * s), 1.7 * s, 0.45 * s, fc=color, ec="none", zorder=3))


def wind(ax, x0, y0, x1, y1, color=DBLUE, lw=1.2, rad=0.25):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), connectionstyle=f"arc3,rad={rad}",
                                 arrowstyle="-|>", mutation_scale=9, color=color, lw=lw, zorder=4))


fig = plt.figure(figsize=(9.6, 9.2))
gs = fig.add_gridspec(3, 2, height_ratios=[1.35, 1.0, 0.95], hspace=0.45, wspace=0.12)

# ---------------------------------------------------------------- (а) storm track
ax = fig.add_subplot(gs[0, 0]); ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
ax.add_patch(Rectangle((0, 0), 10, 7, fc=SEA, ec="none"))
ax.set_title("а. Над путями циклонов:\nмелкие вихри повторяют рисунок крупных", loc="left", fontsize=9, fontweight="bold")
spiral(ax, 5, 3.6, 3.0, DBLUE, lw=2.2, turns=1.4)
for cx, cy in [(3.2, 4.8), (6.9, 4.6), (4.2, 2.1), (6.4, 2.5)]:
    spiral(ax, cx, cy, 0.9, BLUE, lw=1.4)
    for k in range(3):
        ang = 2 * np.pi * k / 3 + 0.4
        spiral(ax, cx + 0.62 * np.cos(ang), cy + 0.62 * np.sin(ang), 0.22, RED, lw=1.0, turns=1.3)
cloud(ax, 1.4, 6.0, 0.55); cloud(ax, 8.6, 6.1, 0.5); cloud(ax, 8.3, 1.0, 0.45)
wind(ax, 0.4, 3.0, 2.2, 4.4); wind(ax, 7.8, 2.4, 9.6, 3.9)
ax.text(5, 0.35, "циклон (крупный уровень) $\\to$ вихри внутри него\n(средний) $\\to$ вихри внутри них (мелкий)",
        ha="center", va="bottom", fontsize=7.5, color=DBLUE)
ax.text(9.7, 6.6, "наследование\nсильное, $P$ высокая", ha="right", va="top", fontsize=8, color=DBLUE, fontweight="bold")

# ---------------------------------------------------------------- (б) deep convection
ax = fig.add_subplot(gs[0, 1]); ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.axis("off")
ax.add_patch(Rectangle((0, 0), 10, 7, fc=SAND, ec="none"))
ax.set_title("б. Над очагами глубокой конвекции:\nрисунок мелких вихрей слабо связан с крупными", loc="left", fontsize=9, fontweight="bold")
spiral(ax, 5, 3.6, 3.0, DBLUE, lw=1.4, turns=1.2)
rng = np.random.default_rng(3)
for _ in range(14):
    cx, cy = rng.uniform(0.7, 9.3), rng.uniform(0.9, 6.0)
    spiral(ax, cx, cy, 0.22, RED, lw=1.0, turns=1.3, sense=rng.choice([-1, 1]))
for cx, cy in [(1.6, 5.4), (4.6, 5.2), (7.8, 5.5), (3.0, 1.4), (6.6, 1.3)]:
    cloud(ax, cx, cy, 0.42, tall=True)
ax.text(5, 0.3, "крупный уровень есть, но мелкие вихри\nстоят где угодно, а не внутри крупных", ha="center", va="bottom", fontsize=7.5, color=DBLUE)
ax.text(9.7, 6.6, "наследование\nслабое, $P$ низкая", ha="right", va="top", fontsize=8, color=RED, fontweight="bold")

# ---------------------------------------------------------------- (в) the law
ax = fig.add_subplot(gs[1, :]); ax.set_xlim(0, 20); ax.set_ylim(0, 6); ax.axis("off")
ax.set_title("в. Закон: согласованность двух уровней зависит только от более крупного из них",
             loc="left", fontsize=9, fontweight="bold")
# three nested pictures, same coarse level 400 km
for k, (x0, fine, txt) in enumerate([(3.3, 50, "50 км внутри 400 км"), (10.0, 100, "100 км внутри 400 км"),
                                     (16.7, 200, "200 км внутри 400 км")]):
    ax.add_patch(Ellipse((x0, 3.2), 5.4, 3.6, fc="white", ec=DBLUE, lw=2.0))
    ax.text(x0, 5.25, "400 км", ha="center", fontsize=8, color=DBLUE)
    n = {50: 12, 100: 6, 200: 3}[fine]
    r = {50: 0.22, 100: 0.38, 200: 0.65}[fine]
    for m in range(n):
        ang = 2 * np.pi * m / n
        rr = 1.4 if fine < 200 else 1.0
        spiral(ax, x0 + rr * 1.6 * np.cos(ang), 3.2 + rr * 0.9 * np.sin(ang), r, RED if fine == 50 else BLUE, lw=1.0, turns=1.3)
    ax.text(x0, 0.9, txt, ha="center", fontsize=8)
ax.text(6.65, 3.2, "=", ha="center", va="center", fontsize=20, fontweight="bold")
ax.text(13.35, 3.2, "=", ha="center", va="center", fontsize=20, fontweight="bold")
ax.text(10, 0.15, "связь с уровнем 400 км одна и та же у вихрей 50, 100 и 200 км: она задана «общим предком», а не расстоянием между уровнями",
        ha="center", fontsize=8, color=DBLUE)

# ---------------------------------------------------------------- (г) regionalisation
ax = fig.add_subplot(gs[2, :]); ax.set_xlim(0, 20); ax.set_ylim(0, 6); ax.axis("off")
ax.set_title("г. Принцип районирования: устройство иерархии везде одно,\nразличается сила наследования",
             loc="left", fontsize=9, fontweight="bold")
# a schematic "planet strip": latitude bands
bands = [(5.0, "Южный океан: пути циклонов", 0.32, DBLUE), (4.0, "субтропики, океан", 0.24, BLUE),
         (3.0, "тропическая суша: очаги конвекции", 0.13, RED), (2.0, "субтропики, муссонная суша", 0.18, "#e07b39"),
         (1.0, "умеренные широты: пути циклонов над океаном, континенты", 0.28, BLUE)]
for y, name, P, col in bands:
    ax.add_patch(Rectangle((0.3, y - 0.42), 12.5 * P / 0.32, 0.84, fc=col, ec="none", alpha=0.85))
    ax.text(0.4, y, name, va="center", fontsize=7.8, color="white" if P > 0.2 else "black")
    ax.text(12.9, y, f"$P \\approx$ {P:.2f}", va="center", fontsize=8)
ax.text(17.0, 3.0, "карта силы наследования $P$\n= карта того, насколько\nмелкие уровни циркуляции\nповторяют крупные;\n"
                   "по ней территории\nделятся на районы\nтак же, как по осадкам\nили температуре",
        ha="center", va="center", fontsize=8, bbox=dict(boxstyle="round", fc="white", ec=GREY))
ax.text(6.5, 0.15, "длина полосы -- сила наследования (значения из статьи 2, округленно)", ha="center", fontsize=7.5, color=GREY)

fig.savefig(OUT, dpi=300, bbox_inches="tight")
print(OUT)
