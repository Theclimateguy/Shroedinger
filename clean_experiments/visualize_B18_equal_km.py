#!/usr/bin/env python3
"""Phase-18 figures: equal-km regionalization design control.

Reads clean_experiments/results/experiment_B18_equal_km_regions/ and the
Phase-3 degree-box results; writes PNGs into the same results directory.
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
sys.path.insert(0, str(_HERE))

from experiment_B18_equal_km_regions import (  # noqa: E402
    A_BANDS,
    EXTENT_FACTORS,
    PRIMARY_WINDOWS,
    REGIONS,
    REGIONS_ALL,
    km_half_extents_deg,
    region_center,
)

RES = _HERE / "results" / "experiment_B18_equal_km_regions"
B3 = _HERE / "results" / "experiment_B3_scattering_benchmark"
ROOT = _HERE.parent

STEP_LABELS = ["50-100", "100-200", "200-400", "400-800", "800-1600"]


def load_results() -> tuple[list[dict], list[dict], dict, dict]:
    summary = json.loads((RES / "summary.json").read_text(encoding="utf-8"))
    table = json.loads((RES / "region_table.json").read_text(encoding="utf-8"))
    primary, heldout = [], []
    for f in sorted(RES.glob("R*__*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        (primary if r["window"] in PRIMARY_WINDOWS else heldout).append(r)
    return primary, heldout, summary, table


def deg_anchored(tag: str) -> np.ndarray:
    d = json.loads((B3 / f"{tag}.json").read_text(encoding="utf-8"))
    return np.asarray(d["P_real"]) - np.asarray(d["P_surrogate_median"])


def fig1_map(table: dict) -> None:
    inv = xr.open_dataset(ROOT / "data/b4cov/era5_invariants_global.nc")
    lsm = inv["lsm"].squeeze().values
    lat = inv["latitude"].values
    lon = inv["longitude"].values
    shift = np.argmin(np.abs(lon - 180.0))
    lsm = np.roll(lsm, -shift, axis=1)
    lon_p = ((lon + 180.0) % 360.0) - 180.0
    lon_p = np.sort(lon_p)

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.pcolormesh(lon_p, lat, lsm, cmap="Greys", vmin=-0.4, vmax=2.2,
                  shading="auto", rasterized=True)
    for r in REGIONS:
        n, w, s, e = REGIONS_ALL[r]
        ax.add_patch(Rectangle((w, s), e - w, n - s, fill=False,
                               edgecolor="tab:blue", lw=1.6))
        lat_c, lon_c = region_center(r)
        hla, hlo = km_half_extents_deg(lat_c, 1.0)
        ax.add_patch(Rectangle((lon_c - hlo, lat_c - hla), 2 * hlo, 2 * hla,
                               fill=False, edgecolor="tab:red", lw=1.6, ls="--"))
        ax.annotate(r.split("_")[0], (lon_c, n + 1.5), ha="center", fontsize=9,
                    color="k", fontweight="bold")
    ax.plot([], [], color="tab:blue", lw=1.6, label="degree boxes 20x40 deg")
    ax.plot([], [], color="tab:red", lw=1.6, ls="--",
            label="equal-km boxes 2000x2800 km")
    ax.legend(loc="lower left", fontsize=10)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-70, 75)
    ax.set_xlabel("lon")
    ax.set_ylabel("lat")
    ax.set_title("Phase 18: degree-box vs equal-km regionalization "
                 "(same centres, native grid, index masks only)")
    fig.tight_layout()
    fig.savefig(RES / "fig1_region_map.png", dpi=150)
    plt.close(fig)


def fig2_profiles(primary: list[dict]) -> None:
    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharex=True, sharey=True)
    x = np.arange(5)
    for ax, reg in zip(axes.ravel(), REGIONS):
        km = np.asarray([r["km_1.0"]["P_anchored"] for r in primary
                         if r["region"] == reg])
        dg = np.asarray([deg_anchored(r["tag"]) for r in primary
                         if r["region"] == reg])
        for mat, color, lab in ((dg, "tab:blue", "degree"), (km, "tab:red", "km")):
            med = np.median(mat, axis=0)
            ax.plot(x, med, "-o", color=color, ms=4, label=lab)
            ax.fill_between(x, mat.min(axis=0), mat.max(axis=0),
                            color=color, alpha=0.15)
        rho = spearmanr(np.median(dg, axis=0), np.median(km, axis=0)).statistic
        ax.set_title(f"{reg}  rho={rho:.2f}", fontsize=10)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=9)
    for ax in axes[-1]:
        ax.set_xticks(x)
        ax.set_xticklabels(STEP_LABELS, rotation=45, fontsize=8)
        ax.set_xlabel("band step (km)")
    for ax in axes[:, 0]:
        ax.set_ylabel("anchored P")
    fig.suptitle("Anchored envelope-coupling profile P - P_surrogate: "
                 "degree boxes vs equal-km boxes (48 primary region-windows)")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(RES / "fig2_anchored_profiles.png", dpi=150)
    plt.close(fig)


def fig3_clustering(summary: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2),
                             gridspec_kw={"width_ratios": [1, 1, 1.4]})

    for ax, key, npy, title in (
        (axes[0], "H18b_anchored_P_km", "h18b_perm.npy",
         "H18b primary: anchored P clustering (km, 48 rw)"),
        (axes[1], "H18d_heldout", "h18d_perm.npy",
         "H18d held-out 2021-22 (km, 32 rw)"),
    ):
        perm = np.load(RES / npy)
        h = summary[key]
        ax.hist(perm, bins=40, color="lightgrey", edgecolor="grey")
        ax.axvline(h["diff"], color="tab:red", lw=2,
                   label=f"observed diff={h['diff']:+.3f}\np={h['p']:.3f}")
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("between - within median distance")
        ax.legend(fontsize=9)

    ax = axes[2]
    items = [
        ("anchored P\n(degree)", summary["H18b_anchored_P_deg_same_windows"]),
        ("anchored P\n(km)", summary["H18b_anchored_P_km"]),
        ("spectral placebo\n(km)", summary["C18_3_spectral_placebo"]["spec_clustering"]),
        ("beyond-spectrum\nanch. (degree)", summary["H18b_secondary_beyond_spectrum_deg_anchored"]),
        ("beyond-spectrum\nanch. (km)", summary["H18b_secondary_beyond_spectrum_km_anchored"]),
        ("beyond-spectrum\nraw (km)", summary["H18b_secondary_beyond_spectrum_km_raw"]),
        ("dynamic-scale\nanch. (km)", summary["P18_3_dynamic_scale_arm"]["anchored_P_clustering"]),
    ]
    xs = np.arange(len(items))
    diffs = [it[1]["diff"] for it in items]
    ps = [it[1]["p"] for it in items]
    colors = ["tab:green" if p < 0.05 else "tab:orange" for p in ps]
    ax.bar(xs, diffs, color=colors)
    for i, (d, p) in enumerate(zip(diffs, ps)):
        ax.annotate(f"p={p:.3f}", (i, d), ha="center",
                    va="bottom" if d >= 0 else "top", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([it[0].replace("\n", " ") for it in items],
                       fontsize=8, rotation=25, ha="right")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("clustering diff (between - within)")
    ax.set_title("Signature strengths on the same 48 windows", fontsize=10)
    fig.tight_layout()
    fig.savefig(RES / "fig3_clustering.png", dpi=150)
    plt.close(fig)


def fig4_A_control(summary: dict) -> None:
    h = summary["H18a_negative_control_A"]
    a_deg = np.asarray([h["regional_A_deg"][r] for r in REGIONS])
    a_km = np.asarray([h["regional_A_km"][r] for r in REGIONS])
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(a_deg, a_km, color="tab:red", zorder=3)
    for r, x, y in zip(REGIONS, a_deg, a_km):
        ax.annotate(r.split("_")[0], (x, y), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    lo = min(a_deg.min(), a_km.min()) * 0.9
    hi = max(a_deg.max(), a_km.max()) * 1.05
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_xlabel("regional A, degree boxes (Fnorm bands 200-800 km)")
    ax.set_ylabel("regional A, equal-km boxes")
    sig = h["A_km_signature"]
    ax.set_title(f"H18a negative control: Spearman rho={h['rho_regional_A_deg_vs_km']:.3f} "
                 f"(bar 0.9)\nkm signature diff={sig['diff']:+.3f} p={sig['p']:.3f}",
                 fontsize=10)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(RES / "fig4_A_control.png", dpi=150)
    plt.close(fig)


def fig5_congo_amazon(summary: dict) -> None:
    h = summary["H18e_congo_amazon"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=False)
    for ax, form in zip(axes, ("anchored", "raw")):
        d = h[form]
        xs = np.arange(2)
        wd = 0.35
        ax.bar(xs - wd / 2, [d["congo_deg"], d["amaz_deg"]], wd,
               color="tab:blue", label="degree")
        ax.bar(xs + wd / 2, [d["congo_km"], d["amaz_km"]], wd,
               color="tab:red", label="km")
        ax.set_xticks(xs)
        ax.set_xticklabels(["R7_CONGO", "R3_AMAZ"])
        ax.set_title(f"{form} rho_5: contrast deg={d['contrast_deg']:+.3f} "
                     f"km={d['contrast_km']:+.3f}", fontsize=10)
        ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("rho_5 (800-1600 km step)")
    axes[0].legend()
    fig.suptitle(f"H18e Congo-Amazon contrast: pass={h['pass']}")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(RES / "fig5_congo_amazon.png", dpi=150)
    plt.close(fig)


def fig6_sensitivity(summary: dict, primary: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    ax = axes[0]
    c = summary["C18_4_extent_sensitivity"]
    xs = np.arange(len(REGIONS))
    wd = 0.35
    for off, f, color in ((-wd / 2, "km_0.9", "tab:purple"),
                          (wd / 2, "km_1.1", "tab:cyan")):
        vals = [c[f]["per_region_rho"][r] for r in REGIONS]
        ax.bar(xs + off, vals, wd, color=color, label=f.replace("km_", "") + "x")
    ax.axhline(0.9, color="k", ls="--", lw=1, label="bar 0.9")
    ax.set_xticks(xs)
    ax.set_xticklabels([r.split("_")[0] for r in REGIONS], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Spearman rho vs 1.0x descriptors")
    ax.set_title("C18-4 boundary-extent sensitivity (logged, not scored)", fontsize=10)
    ax.legend(fontsize=9)

    ax = axes[1]
    lr = np.asarray([
        [r["km_1.0"]["F_spec"][f"logvar_{i}"] -
         json.loads((B3 / f"{r['tag']}.json").read_text())["F_spec"][f"logvar_{i}"]
         for i in range(6)]
        for r in primary])
    ax.boxplot([lr[:, i] for i in range(6)], tick_labels=[
        "<50", "50-100", "100-200", "200-400", "400-800", "800-1600"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("band (km)")
    ax.set_ylabel("log sigma_b^2 (km) - log sigma_b^2 (degree)")
    ax.set_title("C18-1: band-variance shift under re-boxing (48 rw)", fontsize=10)
    fig.tight_layout()
    fig.savefig(RES / "fig6_sensitivity.png", dpi=150)
    plt.close(fig)


def main() -> None:
    primary, heldout, summary, table = load_results()
    print(f"{len(primary)} primary, {len(heldout)} heldout")
    fig1_map(table)
    fig2_profiles(primary)
    fig3_clustering(summary)
    fig4_A_control(summary)
    fig5_congo_amazon(summary)
    fig6_sensitivity(summary, primary)
    print("figures written to", RES)


if __name__ == "__main__":
    main()
