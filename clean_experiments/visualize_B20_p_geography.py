#!/usr/bin/env python3
"""Phase-20 figures: tile maps of anchored fine-P and attribution panels."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

_HERE = Path(__file__).resolve().parent
RES = _HERE / "results" / "experiment_B20_p_geography"
COV = _HERE.parent / "data" / "b4cov"

import sys

sys.path.insert(0, str(_HERE))
from experiment_B20_p_geography import (  # noqa: E402
    COVARS,
    TILE_COLS,
    TILE_ROWS,
    static_covariates,
)
from experiment_B18_equal_km_regions import REGIONS  # noqa: E402

s = json.loads((RES / "summary.json").read_text(encoding="utf-8"))

# assemble tile medians
per_tile: dict[tuple, dict] = {}
for f in sorted(RES.glob("tiles_*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    for tid, t in d["tiles"].items():
        key = (d["region"], tid)
        e = per_tile.setdefault(key, {"anch": [], "lat_bounds": t["lat_bounds"],
                                      "lon_bounds": t["lon_bounds"]})
        e["anch"].append(t["P_fine_anch_mean"])
keys = sorted(per_tile.keys())
med = {k: float(np.median(v["anch"])) for k, v in per_tile.items()}

inv = xr.open_dataset(COV / "era5_invariants_global.nc")
lsm = inv["lsm"].squeeze()
lat_g = inv["latitude"].values
lon_g = inv["longitude"].values

# ---- fig1: 12 tile maps --------------------------------------------------
fig, axes = plt.subplots(3, 4, figsize=(17, 9))
vmax = max(abs(v) for v in med.values())
for ax, reg in zip(axes.ravel(), REGIONS):
    grid = np.full((TILE_ROWS, TILE_COLS), np.nan)
    lat_lo = min(per_tile[(reg, t)]["lat_bounds"][0] for r2, t in keys if r2 == reg)
    lat_hi = max(per_tile[(reg, t)]["lat_bounds"][1] for r2, t in keys if r2 == reg)
    lon_lo = min(per_tile[(reg, t)]["lon_bounds"][0] for r2, t in keys if r2 == reg)
    lon_hi = max(per_tile[(reg, t)]["lon_bounds"][1] for r2, t in keys if r2 == reg)
    for ri in range(TILE_ROWS):
        for ci in range(TILE_COLS):
            grid[ri, ci] = med.get((reg, f"r{ri}c{ci}"), np.nan)
    # native lat is descending in the source files: row 0 = north
    im = ax.imshow(grid, cmap="viridis", vmin=0, vmax=vmax, aspect="auto",
                   extent=(lon_lo, lon_hi, lat_lo, lat_hi), origin="upper")
    # coastline overlay from lsm 0.5 contour
    mlat = (lat_g >= lat_lo) & (lat_g <= lat_hi)
    lon_mod = ((lon_g + 180) % 360) - 180
    mlon = (lon_mod >= lon_lo) & (lon_mod <= lon_hi)
    if mlat.any() and mlon.any():
        sub = lsm.values[np.ix_(mlat, mlon)]
        order = np.argsort(lon_mod[mlon])
        ax.contour(np.sort(lon_mod[mlon]), lat_g[mlat], sub[:, order],
                   levels=[0.5], colors="w", linewidths=1.2)
    ax.set_title(reg, fontsize=10)
fig.colorbar(im, ax=axes, shrink=0.6, label="anchored fine-P (tile median)")
fig.suptitle("Phase 20: the internal geography of anchored fine-P "
             "(3x4 tiles per equal-km box; white = coastline)")
fig.savefig(RES / "fig1_tile_maps.png", dpi=150, bbox_inches="tight")
plt.close(fig)

# ---- fig2: attribution panels -------------------------------------------
stat = static_covariates({k: per_tile[k] for k in keys})
regions = np.asarray([k[0] for k in keys])
y = np.asarray([med[k] for k in keys])
Xr = {
    "orog_std": np.asarray([stat[k]["orog_std"] for k in keys]),
    "land_frac": np.asarray([stat[k]["land_frac"] for k in keys]),
    "coast_var": np.asarray([stat[k]["coast_var"] for k in keys]),
    "cape_mean": np.asarray([stat[k]["cape_mean"] for k in keys]),
    "eke_syn": None,
}
eke = []
for f in sorted(RES.glob("tiles_*.json")):
    pass
eke_map: dict[tuple, list] = {}
for f in sorted(RES.glob("tiles_*.json")):
    d = json.loads(f.read_text(encoding="utf-8"))
    for tid, t in d["tiles"].items():
        eke_map.setdefault((d["region"], tid), []).append(t["eke_syn"])
Xr["eke_syn"] = np.asarray([float(np.median(eke_map[k])) for k in keys])

cmap = plt.get_cmap("tab20")
reg_order = sorted(set(regions))
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
for ax, name in zip(axes.ravel(), COVARS):
    xv = Xr[name]
    for i, reg in enumerate(reg_order):
        m = regions == reg
        ax.scatter(xv[m], y[m], s=14, color=cmap(i % 20), alpha=0.8,
                   label=reg.split("_")[0] if name == COVARS[0] else None)
    h = s["H20b"][name]
    ax.set_title(f"{name}: rho={h['spearman']:+.2f} p_block={h['p_block']:.3f}"
                 f"{' *' if h['significant'] else ''}", fontsize=10)
    ax.set_xlabel(name)
    ax.set_ylabel("anchored fine-P")
    if name in ("orog_std", "cape_mean", "eke_syn"):
        ax.set_xscale("symlog", linthresh=10 if name != "eke_syn" else 1)
ax = axes.ravel()[-1]
h20c = s["H20c"]
mat = np.full((len(COVARS), len(reg_order)), np.nan)
for i, name in enumerate(COVARS):
    for j, reg in enumerate(reg_order):
        mat[i, j] = h20c[name]["per_region_rho"].get(reg, np.nan)
im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
ax.set_yticks(range(len(COVARS)))
ax.set_yticklabels(COVARS, fontsize=8)
ax.set_xticks(range(len(reg_order)))
ax.set_xticklabels([r.split("_")[0] for r in reg_order], fontsize=7,
                   rotation=45)
for i, name in enumerate(COVARS):
    c = h20c[name]
    ax.annotate(f"{c['n_pos']}/{c['n_regions']}+ p={c['p_within_perm']:.3f}",
                (len(reg_order) - 0.4, i), fontsize=7, va="center")
ax.set_xlim(-0.5, len(reg_order) + 2.5)
fig.colorbar(im, ax=ax, shrink=0.7, label="within-region rho")
ax.set_title("H20c: within-region tile correlations", fontsize=10)
h = s["H20a"]
fig.suptitle(f"Phase 20 attribution: H20a LOO-region R2={h['loo_region_r2']:.3f} "
             f"p={h['p']:.3f} -> {s['PHASE20_VERDICT']}")
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(RES / "fig2_attribution.png", dpi=150)
plt.close(fig)
print("figures written")
