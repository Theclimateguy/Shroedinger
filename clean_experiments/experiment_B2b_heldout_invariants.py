#!/usr/bin/env python3
"""Phase-2b: held-out test of the scale-irreversibility profile.

Preregistered protocol: docs/PROTOCOL_PHASE2B_HELDOUT_INVARIANTS.md
(frozen 2026-08-11). The per-region-window pipeline is imported UNCHANGED
from experiment_B2_scale_irreversibility; this script only supplies the
held-out file list and the corrected H1/H2 consolidation.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from clean_experiments.experiment_B2_scale_irreversibility import (
        _loo_nearest_centroid,
        run_region_window,
    )
except ImportError:
    from experiment_B2_scale_irreversibility import (  # type: ignore
        _loo_nearest_centroid,
        run_region_window,
    )

SEED = 20260811
REGIONS = ["R5_SPCZ", "R6_SATL", "R7_CONGO", "R8_AUS",
           "R9_NPAC", "R10_INDO", "R11_EURO", "R12_SAM"]
WINDOWS = ["W5_2021JFM", "W6_2021JAS", "W7_2022JFM", "W8_2022JAS"]
REGION_WINDOWS = [f"era5_wind850_{r}__{w}.nc" for r in REGIONS for w in WINDOWS]


def _zscore_cols(x: np.ndarray) -> np.ndarray:
    mu = x.mean(axis=0)
    sd = x.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (x - mu) / sd


def _within_between(dist_ok: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    n = len(labels)
    w, b = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (w if labels[i] == labels[j] else b).append(dist_ok[i, j])
    return float(np.median(w)), float(np.median(b))


def _pairwise_dist(x: np.ndarray) -> np.ndarray:
    n = x.shape[0]
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d[i, j] = d[j, i] = float(np.linalg.norm(x[i] - x[j]))
    return d


def _perm_test(x_z: np.ndarray, labels: np.ndarray, rng: np.random.Generator) -> dict:
    dist = _pairwise_dist(x_z)
    w_obs, b_obs = _within_between(dist, labels)
    diff_obs = b_obs - w_obs
    perm = []
    for _ in range(999):
        labs = labels[rng.permutation(len(labels))]
        w_p, b_p = _within_between(dist, labs)
        perm.append(b_p - w_p)
    p = float((1 + np.sum(np.asarray(perm) >= diff_obs)) / 1000)
    return {"median_within": w_obs, "median_between": b_obs,
            "diff": diff_obs, "p": p, "pass": bool(diff_obs > 0 and p < 0.05)}


def _loo_residuals(p_mat: np.ndarray, spec_mat: np.ndarray) -> np.ndarray:
    n = p_mat.shape[0]
    x_full = np.hstack([np.ones((n, 1)), spec_mat])
    res = np.zeros_like(p_mat)
    for k in range(n):
        tr = np.arange(n) != k
        coef, *_ = np.linalg.lstsq(x_full[tr], p_mat[tr], rcond=None)
        res[k] = p_mat[k] - x_full[k] @ coef
    return res


def consolidate(results: list[dict], out_dir: Path) -> None:
    rng = np.random.default_rng(SEED)
    labels = np.asarray([r["region"] for r in results])
    n = len(results)

    p_mat = np.asarray([r["P_real"] for r in results])
    dq_mat = np.asarray([r["dQ_real"] for r in results])
    spec_keys = sorted(results[0]["F_spec"].keys())
    spec_mat = np.asarray([[r["F_spec"][k] for k in spec_keys] for r in results])

    c1_count = sum(1 for r in results if r["C1_pass"])
    c1_ok = c1_count >= 26

    h1 = _perm_test(_zscore_cols(p_mat), labels, rng)
    res_mat = _loo_residuals(p_mat, spec_mat)
    h2 = _perm_test(_zscore_cols(res_mat), labels, rng)

    verdict = "POSITIVE" if (c1_ok and h1["pass"] and h2["pass"]) else "NEGATIVE"
    if not c1_ok:
        verdict = "HALT_C1_CONTROL_FAILED"

    desc = {
        "acc_region_spec": _loo_nearest_centroid(spec_mat, labels),
        "acc_region_P": _loo_nearest_centroid(p_mat, labels),
        "acc_region_res": _loo_nearest_centroid(res_mat, labels),
        "acc_region_irr": _loo_nearest_centroid(np.hstack([p_mat, dq_mat]), labels),
        "corr_P_negdQ_median": float(np.nanmedian([
            r["corr_P_vs_negdQ"] for r in results if r["corr_P_vs_negdQ"] is not None
        ])),
        "mean_profiles": {
            reg: [float(x) for x in p_mat[labels == reg].mean(axis=0)] for reg in REGIONS
        },
    }

    summary = {
        "C1_pass_count": c1_count, "C1_ok": c1_ok,
        "H1_profile_signature": h1,
        "H2_beyond_spectrum_residual": h2,
        "PHASE2B_VERDICT": verdict,
        "n_region_windows": n,
        "descriptive": desc,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["# Experiment B2b: held-out test of the irreversibility profile", ""]
    lines.append(f"Region-windows: {n}/32. C1: {c1_count}/{n} (need >=26/32).")
    lines.append("")
    lines.append(f"H1 (profile signature): within={h1['median_within']:.3f} "
                 f"between={h1['median_between']:.3f} diff={h1['diff']:+.3f} "
                 f"p={h1['p']:.3f} pass={h1['pass']}")
    lines.append(f"H2 (spectrum-residualized): within={h2['median_within']:.3f} "
                 f"between={h2['median_between']:.3f} diff={h2['diff']:+.3f} "
                 f"p={h2['p']:.3f} pass={h2['pass']}")
    lines.append("")
    lines.append(f"## PHASE2B_VERDICT: {verdict}")
    lines.append("")
    lines.append("## Descriptive")
    lines.append(f"- region classification acc: spec={desc['acc_region_spec']:.3f} "
                 f"P={desc['acc_region_P']:.3f} residual-P={desc['acc_region_res']:.3f} "
                 f"P+dQ={desc['acc_region_irr']:.3f}")
    lines.append(f"- median corr(P,-dQ) = {desc['corr_P_negdQ_median']:.3f}")
    lines.append("- mean profiles (rho_1..rho_5):")
    for reg in REGIONS:
        prof = ", ".join(f"{x:.3f}" for x in desc["mean_profiles"][reg])
        lines.append(f"  - {reg}: [{prof}]")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/b2b"))
    parser.add_argument("--out-dir", type=Path,
                        default=_HERE / "results" / "experiment_B2b_heldout_invariants")
    parser.add_argument("--n-surrogates", type=int, default=199)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()

    files = [args.data_dir / f for f in REGION_WINDOWS if (args.data_dir / f).exists()]
    if args.only:
        files = [f for f in files if args.only in f.name]
    if not files:
        raise SystemExit(f"No input files in {args.data_dir}")
    print(f"{len(files)} region-window files found")

    results = []
    for f in files:
        tag = f.stem.replace("era5_wind850_", "")
        cached = args.out_dir / f"{tag}.json"
        if cached.exists():
            print(f"[{tag}] cached, skipping", flush=True)
            results.append(json.loads(cached.read_text(encoding="utf-8")))
            continue
        results.append(run_region_window(f, args.out_dir, args.n_surrogates, args.workers))
    if len(results) >= 8 and args.only is None:
        consolidate(results, args.out_dir)


if __name__ == "__main__":
    main()
