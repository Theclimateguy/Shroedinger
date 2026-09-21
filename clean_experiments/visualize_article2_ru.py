#!/usr/bin/env python3
"""Paper-2 journal figures with Russian labels, 300 dpi (no recomputation).

Redraws three figures whose committed versions carry English labels:
  ru_fig1_global_map.png  - global map of surrogate-corrected fine P (Arm B)
  ru_fig3_channels.png    - covariate loadings on P and on the spectral slope
  ru_fig5_model.png       - free-running model map vs ERA5 map (B22, P2 arm)
Inputs are the committed result files of B20 Arm B and B22; the original
English figures (hash-listed in reproducibility/article2_README.md) are left
untouched.  Output: manuscript_v2/article2/figures/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Rectangle
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiment_B20_armB_global_map import (  # noqa: E402
    OROG_EXCL_M, SEASONS, OUT as ARMB, tile_covariates, tile_grid,
)
from experiment_B18_equal_km_regions import REGIONS, REGIONS_ALL  # noqa: E402

B22 = _HERE / "results" / "experiment_B22_qt_round2"
FIGS = _HERE.parent / "manuscript_v2" / "article2" / "figures"
INV = _HERE.parent / "data" / "b4cov" / "era5_invariants_global.nc"

plt.rcParams.update({"font.family": ["Times New Roman", "DejaVu Serif"], "font.size": 11})


def fig_global_map() -> None:
    grid = tile_grid()
    by_tid = {t["tid"]: t for t in grid}
    seas = {s: {r["tid"]: r for r in json.loads(
        (ARMB / f"tiles_{s}2023.json").read_text(encoding="utf-8"))["tiles"]} for s in SEASONS}
    common = sorted(set(seas["JFM"]) & set(seas["JAS"]))
    cov = tile_covariates([by_tid[t] for t in common])
    vals = {t: 0.5 * (seas["JFM"][t]["P_fine_anch_mean"] + seas["JAS"][t]["P_fine_anch_mean"])
            for t in common}
    excl = {t for t in common if cov[t]["orog_mean"] > OROG_EXCL_M}

    inv = xr.open_dataset(INV)
    lsm = inv["lsm"].squeeze().values
    glat = inv["latitude"].values
    glon = inv["longitude"].values
    shift = np.argmin(np.abs(glon - 180.0))
    lsm_r = np.roll(lsm, -shift, axis=1)
    glon_r = np.sort(((glon + 180.0) % 360.0) - 180.0)
    inv.close()

    finite = [v for t, v in vals.items() if np.isfinite(v) and t not in excl]
    vmin, vmax = float(np.quantile(finite, 0.02)), float(np.quantile(finite, 0.98))
    cm = plt.get_cmap("viridis")
    fig, ax = plt.subplots(figsize=(13, 5.8))
    for tid, v in vals.items():
        t = by_tid[tid]
        lo0 = ((t["lon0"] + 180.0) % 360.0) - 180.0
        w = t["dlon"]
        spans = [(lo0, 180.0 - lo0), (-180.0, lo0 + w - 180.0)] if lo0 + w > 180.0 else [(lo0, w)]
        fc = "0.8" if (tid in excl or not np.isfinite(v)) else cm((v - vmin) / (vmax - vmin + 1e-12))
        for x0, wid in spans:
            ax.add_patch(Rectangle((x0, t["lat0"]), wid, 6.0, facecolor=fc, edgecolor="none"))
    ax.contour(glon_r, glat, lsm_r, levels=[0.5], colors="k", linewidths=0.5)
    for r in REGIONS:
        n, w, so, e = REGIONS_ALL[r]
        ax.add_patch(Rectangle((w, so), e - w, n - so, fill=False, edgecolor="tab:red", lw=1.0))
    ax.set_xlim(-180, 180)
    ax.set_ylim(-62, 62)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-60, 61, 30))
    ax.set_xlabel("долгота, °")
    ax.set_ylabel("широта, °")
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    plt.colorbar(sm, ax=ax, shrink=0.8, pad=0.01, label="сопряженность уровней $P$")
    fig.tight_layout()
    fig.savefig(FIGS / "ru_fig1_global_map.png", dpi=300)
    plt.close(fig)


def fig_channels() -> None:
    labels = {
        "eke_syn": "вихревая энергия *", "abs_lat": "модуль широты",
        "sst_grad": "градиент температуры\nповерхности океана", "cape_mean": "энергия неустойчивости *",
        "land_frac": "доля суши", "orog_std": "расчлененность рельефа",
        "orog_mean": "средняя высота рельефа", "coast_var": "изрезанность берега",
    }
    s = json.loads((ARMB / "summary.json").read_text(encoding="utf-8"))
    lp, ls = s["H_B4"]["loadings_anchP"], s["H_B4"]["loadings_slope"]
    order = sorted(labels, key=lambda k: -abs(lp[k]))
    y = np.arange(len(order))[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, load, title, colour in (
        (axes[0], lp, "а) сопряженность уровней $P$", "#1f5fa9"),
        (axes[1], ls, "б) наклон пространственного спектра", "#a33b1f"),
    ):
        v = [load[k] for k in order]
        ax.axvspan(-0.3, 0.3, color="0.92", zorder=0)
        ax.barh(y, v, height=0.62, color=colour, alpha=0.85, edgecolor="black", linewidth=0.4)
        ax.axvline(0.0, color="black", linewidth=0.8)
        ax.set_xlim(-0.95, 0.95)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("ранговая корреляция (902 ячейки)")
        ax.grid(axis="x", linewidth=0.3, alpha=0.5)
        ax.set_axisbelow(True)
        for yi, val in zip(y, v):
            txt = f"{val:+.2f}".replace("-", "–")
            ax.text(val + (0.03 if val >= 0 else -0.03), yi, txt, va="center",
                    ha="left" if val >= 0 else "right", fontsize=9)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([labels[k] for k in order])
    axes[0].set_ylim(-0.7, len(order) - 0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "ru_fig3_channels.png", dpi=300)
    plt.close(fig)


def fig_model() -> None:
    def mean_map(pattern: str) -> dict[str, float]:
        per: dict[str, list[float]] = {}
        for season in SEASONS:
            d = json.loads((B22 / pattern.format(season)).read_text(encoding="utf-8"))
            for r in d["tiles"]:
                if np.isfinite(r["P_fine_anch_mean"]):
                    per.setdefault(r["tid"], []).append(r["P_fine_anch_mean"])
        return {t: float(np.mean(v)) for t, v in per.items() if len(v) == len(SEASONS)}

    model, era = mean_map("tiles_MODEL_{}.json"), mean_map("tiles_E05_{}.json")
    grid = {t["tid"]: t for t in tile_grid()}
    common = sorted(set(model) & set(era))
    cov = tile_covariates([grid[t] for t in common])
    keep = [t for t in common if cov[t]["orog_mean"] <= OROG_EXCL_M]
    x = np.array([era[t] for t in keep])
    yv = np.array([model[t] for t in keep])
    rho = float(spearmanr(x, yv).statistic)
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.scatter(x, yv, s=9, alpha=0.5, color="#1f5fa9", edgecolors="none")
    lo, hi = float(min(x.min(), yv.min())), float(max(x.max(), yv.max()))
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.9)
    ax.set_xlabel("$P$ по реанализу ERA5, 2023 г. (сетка 0.5°)")
    ax.set_ylabel("$P$ по свободному расчету ECMWF-IFS-HR, 2014 г.")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "ru_fig5_model.png", dpi=300)
    plt.close(fig)
    print(f"model vs ERA5: n={len(keep)} rho={rho:.3f}")


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_global_map()
    fig_channels()
    fig_model()
    print("wrote", FIGS)
