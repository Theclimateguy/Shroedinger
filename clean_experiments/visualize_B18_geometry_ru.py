#!/usr/bin/env python3
"""Figure 2 of article 1 (journal version, Russian labels): surrogate-anchored
profiles P - P_surr on degree boxes (B3 results) vs equal-km boxes (B18, km_1.0),
12 regions x 4 windows 2023-2024 (W9-W12). Output: manuscript_v2/figures/article1/fig3_geometry.png"""
from __future__ import annotations
import glob, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
OUT = HERE.parent / "manuscript_v2" / "figures" / "article1" / "fig3_geometry.png"
ORDER = ["R1_WPWP", "R2_NATL", "R3_AMAZ", "R4_CASIA", "R5_SPCZ", "R6_SATL",
         "R7_CONGO", "R8_AUS", "R9_NPAC", "R10_INDO", "R11_EURO", "R12_SAM"]
STEPS = ["50–100", "100–200", "200–400", "400–800", "800–1600"]

deg, km = {r: [] for r in ORDER}, {r: [] for r in ORDER}
for f in sorted(glob.glob(str(RES / "experiment_B18_equal_km_regions" / "R*__*.json"))):
    rec = json.loads(Path(f).read_text(encoding="utf-8"))
    twin = RES / "experiment_B3_scattering_benchmark" / Path(f).name
    if not twin.exists():
        continue
    e = json.loads(twin.read_text(encoding="utf-8"))
    deg[rec["region"]].append(np.array(e["P_real"]) - np.array(e["P_surrogate_median"]))
    km[rec["region"]].append(np.array(rec["km_1.0"]["P_anchored"]))
rho = json.loads((RES / "experiment_B18_equal_km_regions" / "summary.json").read_text())["H18c_geography"]["per_region_rho"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
fig, axes = plt.subplots(3, 4, figsize=(12, 7.2), sharex=True, sharey=True)
x = np.arange(1, 6)
for ax, r in zip(axes.flat, ORDER):
    for arr, col, lab in ((np.array(deg[r]), "tab:blue", "градусные области"),
                          (np.array(km[r]), "tab:red", "равновеликие области")):
        ax.fill_between(x, arr.min(0), arr.max(0), color=col, alpha=0.18, linewidth=0)
        ax.plot(x, arr.mean(0), "-o", color=col, ms=3, lw=1.3, label=lab)
    ax.set_title(f"{r}   ρ = {rho[r]:.2f}", fontsize=8.5)
    ax.grid(alpha=0.3)
    ax.set_xticks(x); ax.set_xticklabels(STEPS, rotation=45, fontsize=7)
for ax in axes[:, 0]:
    ax.set_ylabel("P − P$_{surr}$")
for ax in axes[-1]:
    ax.set_xlabel("полосы ступеней, км")
axes[0, 0].legend(loc="lower right", fontsize=7, frameon=True)
n = sum(len(v) for v in deg.values())
fig.suptitle(f"Профили с суррогатной поправкой: градусные и равновеликие области ({n} основных регион-окон, 2023–2024 гг.)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97))
OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=300)
print("wrote", OUT, "windows:", n)
