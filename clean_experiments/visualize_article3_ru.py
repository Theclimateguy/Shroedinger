#!/usr/bin/env python3
"""Figures of article 3 (journal version, Russian labels, 300 dpi) from the
Phase 27-30 results. Output: manuscript_v2/article3/figures/ru_fig{1..5}_*.png

fig1 — scheme: hierarchy of levels, common ancestor (drawn, no data)
fig2 — matrix of anchored covariances C_ij on the 12 regions (mean): the
       coarser-level-only structure
fig3 — decay G(a) by region (ERA5 units) and in the free-running model
fig4 — tiles: |C_03 - C_13| vs |C_02 - C_13| (same coarse vs same separation)
fig5 — P vs the normalised amplitude h on the 902 tiles
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent))
from clean_experiments.experiment_B27_mechanism import _load, WINDOWS  # noqa: E402
from clean_experiments.experiment_B29_structure_law import MODEL_WINDOWS  # noqa: E402

RES = HERE / "results"
FIG = HERE.parent / "manuscript_v2" / "article3" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
ORDER = ["R1_WPWP", "R2_NATL", "R3_AMAZ", "R4_CASIA", "R5_SPCZ", "R6_SATL",
         "R7_CONGO", "R8_AUS", "R9_NPAC", "R10_INDO", "R11_EURO", "R12_SAM"]
NAME = {"R1_WPWP": "Тёплый бассейн Тихого ок.", "R2_NATL": "Северная Атлантика", "R3_AMAZ": "Амазония",
        "R4_CASIA": "Центральная Азия", "R5_SPCZ": "Южно-Тихоокеанская ЗК", "R6_SATL": "Южная Атлантика",
        "R7_CONGO": "Бассейн Конго", "R8_AUS": "Австралия", "R9_NPAC": "Север Тихого ок.",
        "R10_INDO": "Индонезия", "R11_EURO": "Европа", "R12_SAM": "Юг Южной Америки"}
NAME = {k: v.replace("ё", "е") for k, v in NAME.items()}
ELLS = [50, 100, 200, 400, 800, 1600]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def units(kind="units"):
    return {r: [_load(RES / "experiment_B27_mechanism" / kind / f"{r}__{w}.json") for w in WINDOWS] for r in ORDER}


def model_units():
    return {r: [_load(RES / "experiment_B29_structure_law" / "model" / f"{r}__{w}.json") for w in MODEL_WINDOWS]
            for r in ORDER}


def cov_matrix(shards):
    M = np.full((6, 6), np.nan)
    for i in range(6):
        for j in range(i + 1, 6):
            M[i, j] = np.mean([d["anch"][f"cov_{i}_{j}"][0] for d in shards])
    return M


# ---------------------------------------------------------------- fig 1 scheme
def fig1():
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")
    levels = [(5.0, 5.2, 4.6, "Крупный уровень (800--1600 км):\nциклоны и антициклоны"),
              (2.6, 3.4, 2.2, "Средний уровень (200--400 км)"), (7.4, 3.4, 2.2, ""),
              (1.4, 1.6, 1.0, "Мелкий уровень\n(50--100 км)"), (3.8, 1.6, 1.0, ""), (6.2, 1.6, 1.0, ""), (8.6, 1.6, 1.0, "")]
    for x, y, w, t in levels:
        ax.add_patch(FancyBboxPatch((x - w / 2, y - 0.38), w, 0.76, boxstyle="round,pad=0.02", fc="#e8eef7", ec="#3b5b8c"))
        if t:
            ax.text(x, y, t.replace("--", "–"), ha="center", va="center", fontsize=8.5)
    for (x0, y0), (x1, y1) in [((5, 4.8), (2.6, 3.8)), ((5, 4.8), (7.4, 3.8)),
                               ((2.6, 3.0), (1.4, 2.0)), ((2.6, 3.0), (3.8, 2.0)),
                               ((7.4, 3.0), (6.2, 2.0)), ((7.4, 3.0), (8.6, 2.0))]:
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color="#3b5b8c", lw=1.2))
    ax.text(5, 5.85, "Каждый уровень получает свою «текстуру» от уровня выше и передает ее вниз",
            ha="center", fontsize=9, style="italic")
    ax.text(5, 0.35, "Согласованность двух уровней зависит только от того, где они встречаются, —\n"
                     "от общего для них крупного уровня (общего предка)", ha="center", fontsize=8.5)
    fig.savefig(FIG / "ru_fig1_scheme.png", dpi=300, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- fig 2 matrix
def fig2():
    U = units()
    M = cov_matrix([d for r in ORDER for d in U[r]])
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=np.nanmax(M[:, 2:]))
    labels = [f"{ELLS[i]}–{ELLS[i+1] if i < 5 else 3200}" for i in range(6)]
    labels = ["<50", "50–100", "100–200", "200–400", "400–800", "800–1600"]
    ax.set_xticks(range(6)); ax.set_yticks(range(6))
    ax.set_xticklabels(labels, rotation=35, ha="right"); ax.set_yticklabels(labels)
    ax.set_xlabel("Более крупный уровень, км"); ax.set_ylabel("Более мелкий уровень, км")
    for i in range(6):
        for j in range(i + 1, 6):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.15 else "black")
    for j in range(2, 6):
        ax.add_patch(plt.Rectangle((j - 0.5, -0.5), 1, j - 1, fill=False, ec="#c0392b", lw=1.5))
    plt.colorbar(im, ax=ax, fraction=0.046, label="Сверхспектральная ковариация огибающих")
    fig.savefig(FIG / "ru_fig2_matrix.png", dpi=300, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- fig 3 decay
def fig3():
    U, Mo = units(), model_units()
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), sharey=True)
    cm = plt.get_cmap("tab20")
    for ax, src, title in ((axes[0], U, "а"), (axes[1], Mo, "б")):
        for k, r in enumerate(ORDER):
            M = cov_matrix(src[r])
            g = [np.nanmean(M[:j - 1, j]) for j in range(2, 6)]
            ax.plot(ELLS[2:6], g, "o-", color=cm(k), ms=3.5, lw=1.2, label=NAME[r])
        ax.set_xscale("log"); ax.set_xticks(ELLS[2:6]); ax.set_xticklabels([str(e) for e in ELLS[2:6]]); ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel("Крупный уровень $a$, км"); ax.axhline(0, color="k", lw=0.6)
        ax.set_title(title, loc="left", fontweight="bold")
    axes[0].set_ylabel("Унаследованная текстура $G(a)$")
    axes[1].legend(fontsize=6.5, ncol=2, frameon=False)
    fig.savefig(FIG / "ru_fig3_decay.png", dpi=300, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- fig 4 tiles
def fig4():
    sh = {s: {p.stem: _load(p) for p in sorted((RES / "experiment_B27_mechanism" / "tiles" / s).glob("t*.json"))}
          for s in ("JFM", "JAS")}
    tids = np.load(RES / "experiment_B29_structure_law" / "tile_amplitude.npz")["tid"]
    def c(i, j):
        return np.asarray([np.mean([sh[s][t]["anch"][f"cov_{i}_{j}"][0] for s in sh]) for t in tids])
    same_sep = np.abs(c(0, 2) - c(1, 3)); same_coarse = np.abs(c(0, 3) - c(1, 3))
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(same_sep, same_coarse, s=6, alpha=0.5, color="#3b5b8c")
    m = max(same_sep.max(), 0.05)
    ax.plot([0, m], [0, m], "k--", lw=0.8, label="равенство")
    ax.set_xlim(0, m); ax.set_ylim(0, m)
    ax.set_xlabel("Разность при одинаковом расстоянии между уровнями\n|C(50, 200) – C(100, 400)|")
    ax.set_ylabel("Разность при одинаковом крупном уровне\n|C(50, 400) – C(100, 400)|")
    ax.text(0.03, 0.93, f"902 ячейки; медиана отношения {np.median(same_coarse / same_sep):.2f}",
            transform=ax.transAxes, fontsize=8.5)
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(FIG / "ru_fig4_tiles.png", dpi=300, bbox_inches="tight"); plt.close(fig)


# ---------------------------------------------------------------- fig 5 P vs h
def fig5():
    f = np.load(RES / "experiment_B30_unification" / "tile_fields.npz")
    tids = f["tid"]
    sh = {s: {p.stem: _load(p) for p in sorted((RES / "experiment_B27_mechanism" / "tiles" / s).glob("t*.json"))}
          for s in ("JFM", "JAS")}
    h = 0.5 * (f["h1"] + f["h2"])
    P = np.asarray([np.mean([np.mean([sh[s][t]["anch"][f"rho_{i}_{i+1}"][0] for i in (1, 2)]) for s in sh]) for t in tids])
    from scipy.stats import spearmanr
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.scatter(h, P, s=6, alpha=0.5, color="#3b5b8c")
    ax.set_xlabel("Нормированная амплитуда наследования $h$")
    ax.set_ylabel("Сопряженность соседних уровней $P$")
    ax.text(0.03, 0.93, f"902 ячейки; ранговая корреляция {spearmanr(h, P).statistic:.2f}",
            transform=ax.transAxes, fontsize=8.5)
    fig.savefig(FIG / "ru_fig5_P_vs_h.png", dpi=300, bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig1(); fig2(); fig3(); fig4(); fig5()
    print("figures written to", FIG)
