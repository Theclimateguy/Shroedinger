#!/usr/bin/env python3
"""Phase-20 Arm B: two-channel figure.

Covariate loadings on the two targets of the global tile map -- anchored
fine-P and the spatial spectral slope -- drawn side by side.  The point of
the figure is that the two targets are predicted about equally well
(R^2 = 0.536 vs 0.570 under leave-one-sector-out) by *disjoint* sets of
covariates: P is read by circulation-regime variables, the slope by
boundary-condition variables.

Input:  results/experiment_B20_armB_global_map/summary.json
Output: results/experiment_B20_armB_global_map/fig3_two_channel.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

OUT = _HERE / "results" / "experiment_B20_armB_global_map"

LABELS = {
    "eke_syn": "synoptic eddy energy",
    "abs_lat": "|latitude|",
    "sst_grad": "SST gradient",
    "cape_mean": "CAPE climatology",
    "land_frac": "land fraction",
    "orog_std": "orography (s.d.)",
    "orog_mean": "orography (mean)",
    "coast_var": "land-sea contrast",
}
TIER2 = {"eke_syn", "cape_mean"}

s = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
lp = s["H_B4"]["loadings_anchP"]
ls = s["H_B4"]["loadings_slope"]
r2_p = s["H_B1"]["loso_r2"]
r2_s = s["H_B4"]["slope_loso_r2"]

order = sorted(LABELS, key=lambda k: -abs(lp[k]))
y = np.arange(len(order))[::-1]

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)

for ax, load, title, colour in (
    (axes[0], lp, f"target: anchored fine-$P$   ($R^2_{{LOSO}}$ = {r2_p:.3f})", "#1f5fa9"),
    (axes[1], ls, f"control target: spectral slope   ($R^2_{{LOSO}}$ = {r2_s:.3f})", "#a33b1f"),
):
    v = [load[k] for k in order]
    ax.barh(y, v, height=0.62, color=colour, alpha=0.85,
            edgecolor="black", linewidth=0.4)
    ax.axvline(0.0, color="black", linewidth=0.8)
    for lo, hi, c in ((-0.3, 0.3, "0.92"),):
        ax.axvspan(lo, hi, color=c, zorder=0)
    ax.set_xlim(-0.9, 0.9)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("rank correlation with target (902 tiles)", fontsize=9)
    ax.tick_params(labelsize=9)
    ax.grid(axis="x", linewidth=0.3, alpha=0.5)
    ax.set_axisbelow(True)
    for yi, val in zip(y, v):
        ax.text(val + (0.03 if val >= 0 else -0.03), yi, f"{val:+.2f}",
                va="center", ha="left" if val >= 0 else "right", fontsize=8)

labels = [LABELS[k] + (" *" if k in TIER2 else "") for k in order]
axes[0].set_yticks(y)
axes[0].set_yticklabels(labels, fontsize=9)
axes[0].set_ylim(-0.7, len(order) - 0.3)

fig.text(0.012, 0.02,
         "* tier-2 (co-evolving state); all others tier-1 (exogenous boundary "
         "conditions).  Shaded band |rho| < 0.3.",
         fontsize=7.5, color="0.3")
fig.suptitle("Two targets, comparable skill, disjoint loadings", fontsize=11.5)
fig.tight_layout(rect=(0, 0.04, 1, 0.95))
fig.savefig(OUT / "fig3_two_channel.png", dpi=200)
print("wrote", OUT / "fig3_two_channel.png")
