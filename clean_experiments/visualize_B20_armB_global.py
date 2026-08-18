#!/usr/bin/env python3
"""Phase-20 Arm B figures: THE global map of anchored fine-P."""

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

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from experiment_B20_armB_global_map import (  # noqa: E402
    ALL_COV,
    OROG_EXCL_M,
    SEASONS,
    OUT,
    tile_grid,
    tile_covariates,
)
from experiment_B18_equal_km_regions import REGIONS, REGIONS_ALL  # noqa: E402

s = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
grid = tile_grid()
by_tid = {t["tid"]: t for t in grid}
seasons = {}
for name in SEASONS:
    d = json.loads((OUT / f"tiles_{name}2023.json").read_text(encoding="utf-8"))
    seasons[name] = {r["tid"]: r for r in d["tiles"]}
common = sorted(set(seasons["JFM"]) & set(seasons["JAS"]))
cov = tile_covariates([by_tid[t] for t in common])

vals = {t: 0.5 * (seasons["JFM"][t]["P_fine_anch_mean"]
                  + seasons["JAS"][t]["P_fine_anch_mean"]) for t in common}
excl = {t for t in common if cov[t]["orog_mean"] > OROG_EXCL_M}

inv = xr.open_dataset(_HERE.parent / "data/b4cov/era5_invariants_global.nc")
lsm = inv["lsm"].squeeze().values
glat = inv["latitude"].values
glon = inv["longitude"].values
shift = np.argmin(np.abs(glon - 180.0))
lsm_r = np.roll(lsm, -shift, axis=1)
glon_r = np.sort(((glon + 180.0) % 360.0) - 180.0)


def draw_map(ax, values: dict, title: str, cmap="viridis", vmin=None,
             vmax=None, label=""):
    finite = [v for v in values.values() if np.isfinite(v)]
    if vmin is None:
        vmin = float(np.quantile(finite, 0.02))
    if vmax is None:
        vmax = float(np.quantile(finite, 0.98))
    cm = plt.get_cmap(cmap)
    for tid, v in values.items():
        t = by_tid[tid]
        lo0 = ((t["lon0"] + 180.0) % 360.0) - 180.0
        w = t["dlon"]
        if lo0 + w > 180.0:          # split wrap tiles at the dateline
            spans = [(lo0, 180.0 - lo0), (-180.0, lo0 + w - 180.0)]
        else:
            spans = [(lo0, w)]
        if tid in excl or not np.isfinite(v):
            fc, a = ("0.75", 0.6)
        else:
            fc = cm((v - vmin) / (vmax - vmin + 1e-12))
            a = 1.0
        for x0, wid in spans:
            ax.add_patch(Rectangle((x0, t["lat0"]), wid, 6.0,
                                   facecolor=fc, edgecolor="none", alpha=a))
    ax.contour(glon_r, glat, lsm_r, levels=[0.5], colors="k", linewidths=0.5)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-62, 62)
    ax.set_title(title, fontsize=11)
    sm = plt.cm.ScalarMappable(cmap=cm,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    plt.colorbar(sm, ax=ax, shrink=0.75, pad=0.01, label=label)


# ---- fig1: THE map -------------------------------------------------------
fig, ax = plt.subplots(figsize=(17, 7.5))
draw_map(ax, vals, "The global map of anchored fine-P (50-400 km scale-"
         "coupling texture), mean of JFM+JAS 2023\n"
         f"grey = excluded (orography > {OROG_EXCL_M:.0f} m); "
         f"verdict: {s['ARM_B_VERDICT']}", label="anchored fine-P")
for r in REGIONS:
    n, w, so, e = REGIONS_ALL[r]
    ax.add_patch(Rectangle((w, so), e - w, n - so, fill=False,
                           edgecolor="tab:red", lw=1.0))
fig.tight_layout()
fig.savefig(OUT / "fig1_global_map.png", dpi=150)
plt.close(fig)

# ---- fig2: seasons + covariate story ------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(17, 9))
for ax, name in zip(axes[0], SEASONS):
    draw_map(ax, {t: seasons[name][t]["P_fine_anch_mean"] for t in common},
             f"anchored fine-P, {name} 2023", label="anchored fine-P")
d_season = {t: seasons["JFM"][t]["P_fine_anch_mean"]
            - seasons["JAS"][t]["P_fine_anch_mean"] for t in common}
draw_map(axes[1][0], d_season, "season contrast JFM - JAS",
         cmap="RdBu_r", vmin=-0.12, vmax=0.12, label="delta anchored P")
ax = axes[1][1]
h1, h2, h3 = s["H_B1"], s["H_B2"], s["H_B3"]
la = s["H_B4"]["loadings_anchP"]
names = list(ALL_COV)
xs = np.arange(len(names))
ax.bar(xs, [la[n] for n in names], color=["tab:blue"] * 6 + ["tab:red"] * 2)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(xs)
ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Spearman vs anchored P")
ax.set_title(
    f"H-B1: LOSO R2={h1['loso_r2']:.3f} p={h1['p_rot']:.3f} | "
    f"tier-1 R2={h2['tier1_r2']:.3f}, tier-2 gain={h2['tier2_increment']:+.3f}\n"
    f"forward selection: " + " -> ".join(
        f"{c['added']}({c['loso_r2']:.2f})" for c in h3["forward_curve"][:4])
    + f" | k80={h3['k80']}", fontsize=9)
fig.tight_layout()
fig.savefig(OUT / "fig2_seasons_attribution.png", dpi=150)
plt.close(fig)
print("figures written")
