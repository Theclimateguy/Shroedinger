#!/usr/bin/env python3
"""Phase-19 figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = Path(__file__).resolve().parent
RES = _HERE / "results" / "experiment_B19_km_dynamics"
B14 = _HERE / "results" / "experiment_B14_p_relaxation"

s = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
b14s = json.loads((B14 / "summary.json").read_text(encoding="utf-8"))
gate = json.loads((RES / "gate_b.json").read_text(encoding="utf-8"))

# ---- fig1: C19-1 + sign split -------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
vals = list(s["C19_1"]["per_rw"].values())
ax.hist(vals, bins=16, color="tab:blue", edgecolor="k", alpha=0.8)
ax.axvline(s["C19_1"]["median"], color="tab:red", lw=2,
           label=f"median={s['C19_1']['median']:.3f}")
ax.axvline(0.5, color="k", ls="--", lw=1)
ax.axvline(0.995, color="k", ls="--", lw=1, label="bounds [0.5, 0.995]")
ax.set_xlabel("corr(km P(t), degree P(t)) per region-window")
ax.set_ylabel("count")
ax.set_title("C19-1: territory change is real, measurement stable", fontsize=10)
ax.legend(fontsize=9)

ax = axes[1]
deg_alphas = {t: v["alpha"] for t, v in b14s["H14c"]["per_rw"].items()}
xs = np.arange(4)
w = 0.2
for i, (reg, color) in enumerate((("R7_CONGO", "tab:green"),
                                  ("R5_SPCZ", "tab:purple"))):
    dg = [deg_alphas[t] for t in sorted(deg_alphas) if t.startswith(reg)]
    km = s["sign_split_km"][reg]["alphas"]
    ax.bar(xs + (2 * i - 1.5) * w, dg, w, color=color, alpha=0.45,
           label=f"{reg} degree")
    ax.bar(xs + (2 * i - 0.5) * w, km, w, color=color,
           label=f"{reg} km")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(xs)
ax.set_xticklabels(["W5", "W6", "W7", "W8"])
ax.set_ylabel("CAPE coefficient alpha")
ax.set_title("H19A: the sign split survives the territory change\n"
             "(R7 +4/4, R5 -4/4 on BOTH grids)", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(RES / "fig1_armA.png", dpi=150)
plt.close(fig)

# ---- fig2: tau_b(ell) ----------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
ax = axes[0]
ells = np.asarray(s["H19B"]["step_ell_km"])
rows = s["ARM_B_per_rw"]
regions = sorted({r["region"] for r in rows})
cmap = plt.get_cmap("tab20")
for i, reg in enumerate(regions):
    mats = [r["taus_h"] for r in rows if r["region"] == reg]
    med = [np.median([m[b] for m in mats if m[b] is not None])
           for b in range(5)]
    ax.plot(ells, med, "-o", ms=3, lw=1, color=cmap(i % 20), alpha=0.7,
            label=reg.split("_")[0])
med_all = s["H19B"]["median_taus_h"]
ax.plot(ells, med_all, "-o", color="k", lw=2.5, ms=5, label="median")
for slope, ls, lab in ((2 / 3, "--", "turnover 2/3"), (1.0, ":", "sweeping 1")):
    ref = med_all[0] * (ells / ells[0]) ** slope
    ax.plot(ells, ref, ls, color="grey", lw=1.5, label=lab)
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("band step scale ell (km)")
ax.set_ylabel("tau_b (hours)")
ax.set_title(f"H19B: flat timescale hierarchy — median alpha="
             f"{s['H19B']['median_alpha']:.3f}, "
             f"CI [{s['H19B']['ci95_cluster_bootstrap'][0]:.2f}, "
             f"{s['H19B']['ci95_cluster_bootstrap'][1]:.2f}]", fontsize=10)
ax.legend(fontsize=7, ncol=3)

ax = axes[1]
synth = [r["alpha"] for r in gate["rows"] if r["alpha"] is not None]
real = [r["alpha"] for r in rows if r["alpha"] is not None]
ax.hist(real, bins=24, color="tab:blue", alpha=0.7, density=True,
        label="real km rw (n=80)")
ax.hist(synth, bins=12, color="tab:orange", alpha=0.7, density=True,
        label="single-timescale synthetic (gate, n=20)")
ax.axvline(0, color="k", lw=0.8)
ax.axvline(2 / 3, color="grey", ls="--", lw=1, label="2/3")
ax.axvline(gate["median_abs_alpha"], color="tab:red", ls=":", lw=1.5)
ax.axvline(-gate["median_abs_alpha"], color="tab:red", ls=":", lw=1.5,
           label=f"gate median|alpha|={gate['median_abs_alpha']:.2f} (>0.15 FAIL)")
ax.set_xlabel("alpha (slope of ln tau vs ln ell)")
ax.set_ylabel("density")
ax.set_title("Gate failed (spurious |alpha|~0.18), but real alpha ~ 0:\n"
             "no scaling either way", fontsize=10)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(RES / "fig2_armB.png", dpi=150)
plt.close(fig)
print("figures written")
