#!/usr/bin/env python3
"""Phase 36: residual after the spectral null; adapted from the Phase-34 map of the co-location regions (k = 4), the stationary template share and
the co-location profile of steps 4-5, 66 windows, ERA5 850 hPa, mean of 2023-2024.

Output: clean_experiments/results/experiment_B34_colocation_geography/fig_regions_map_ru.png
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402

os.chdir(Path(__file__).resolve().parent.parent)
R = Path("clean_experiments/results/experiment_B36_residual_regions")
SURF, INK, INK2, LAND = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d3"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # validated categorical order
NAMES = {c: f"Тип {c}" for c in (1, 2, 3, 4)}

s = json.loads((R / "summary.json").read_text())
z = np.load(R / "window_fields.npz")
lab = np.array(s["H36_2"]["k"]["4"]["labels_pooled"])
share = 0.5 * (z["P45_null_2023"] + z["P45_null_2024"])
p45 = 0.5 * (z["P45_res_2023"] + z["P45_res_2024"])
z34 = np.load("clean_experiments/results/experiment_B34_colocation_geography/window_fields.npz")
W = [(la - 10, la + 10, lo - w / 2, lo + w / 2) for la, lo, w in
     zip(z34["lat_c"], z34["lon_c"], np.where(np.abs(z34["lat_c"]) == 50, 40, 30))]

# land background on a 0.5-degree grid
from global_land_mask import globe  # noqa: E402
glat = np.arange(-69.75, 70, 0.5); glon = np.arange(0.25, 360, 0.5)
LA, LO = np.meshgrid(glat, np.where(glon > 180, glon - 360, glon), indexing="ij")
land = globe.is_land(LA, LO).astype(float)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": INK2,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
fig, axes = plt.subplots(3, 1, figsize=(11.5, 13.8), facecolor=SURF,
                         gridspec_kw={"hspace": 0.52})


def base(ax, title):
    ax.set_facecolor(SURF)
    ax.contourf(glon, glat, land, levels=[0.5, 1.5], colors=[LAND], zorder=0)
    ax.set_xlim(0, 360); ax.set_ylim(-60.3, 60.3)
    ax.set_xticks(range(0, 361, 60))
    ax.set_xticklabels(["0°", "60° в.д.", "120° в.д.", "180°", "120° з.д.", "60° з.д.", "0°"])
    ax.set_yticks([-60, -40, -20, 0, 20, 40, 60])
    ax.set_yticklabels(["60° ю.ш.", "40°", "20°", "0°", "20°", "40°", "60° с.ш."])
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.tick_params(length=0)


def cells(ax, colors, alpha, labels=None):
    for k, (la0, la1, lo0, lo1) in enumerate(W):
        ax.add_patch(Rectangle((lo0 + 0.6, la0 + 0.6), lo1 - lo0 - 1.2, la1 - la0 - 1.2,
                               facecolor=colors[k], alpha=alpha, edgecolor=SURF, linewidth=2, zorder=2))
        ax.add_patch(Rectangle((lo0 + 0.6, la0 + 0.6), lo1 - lo0 - 1.2, la1 - la0 - 1.2,
                               facecolor="none", edgecolor=SURF, linewidth=2, zorder=3))
        if labels is not None:
            ax.text(0.5 * (lo0 + lo1), 0.5 * (la0 + la1), labels[k], ha="center", va="center",
                    fontsize=10, color=INK, fontweight="bold", zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.2, foreground=SURF)])
    ax.contour(glon, glat, land, levels=[0.5], colors=[INK2], linewidths=0.6, zorder=5)


# (a) regions
ax = axes[0]
base(ax, "а) Районы по остатку сверх спектрального нуля (k = 4; разбиение неустойчиво между 2023 и 2024 гг.)")
cells(ax, [CAT[c - 1] for c in lab], 0.78, [str(c) for c in lab])
ax.legend(handles=[Patch(facecolor=CAT[c - 1], alpha=0.78, edgecolor="none", label=NAMES[c]) for c in (1, 2, 3, 4)],
          loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4, frameon=False, fontsize=9.5,
          labelcolor=INK, handlelength=1.4, columnspacing=1.8)


def seq(ax, vals, cmap, title, cbar_label):
    base(ax, title)
    norm = plt.Normalize(np.min(vals), np.max(vals))
    cm = plt.get_cmap(cmap)
    cells(ax, [cm(0.15 + 0.8 * norm(v)) for v in vals], 0.9)
    sm = plt.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(cm(np.linspace(0.15, 0.95, 256))), norm=norm)
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.05, pad=0.13, aspect=45)
    cb.set_label(cbar_label, color=INK2); cb.outline.set_visible(False); cb.ax.tick_params(length=0)


seq(axes[1], share, "Blues", "б) Спектральный нуль: сопряжённость ступеней 4–5 у случайного поля с тем же спектром",
    "корреляция лог-огибающих соседних полос, зеркальные суррогаты")
seq(axes[2], p45, "Oranges", "в) Остаток: сопряжённость сверх спектрального нуля, ступени 4–5",
    "реальная минус нулевая; воспроизводимость 2023 ↔ 2024: 0,89, свежий 2022: 0,82, свободная модель: 0,81")

fig.text(0.01, 0.005, "ERA5, 850 гПа, V12, январь–март и июль–сентябрь, среднее 2023–2024 гг.; 66 окон ≈ 2200 × 3000 км, "
         "азимутально-эквидистантная плоскость. Остаток = реальное − зеркальный суррогат (стационарное поле со спектром окна).\n"
         "Спектр объясняет R² = 0,17 межоконной изменчивости сопряжённости. Районы на остатке: ARI 2023↔2024 = 0,25 (порог 0,40), "
         "свежий 2022 к пулу 0,50. Phase 36.",
         fontsize=8, color=INK2, ha="left", va="bottom")
out = R / "fig_residual_map_ru.png"
fig.savefig(out, dpi=200, facecolor=SURF, bbox_inches="tight")
print("saved", out)
