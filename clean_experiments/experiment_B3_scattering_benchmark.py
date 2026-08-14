#!/usr/bin/env python3
"""Phase-3: position the irreversibility profile P against scattering-type
statistics on new held-out years.

Preregistered protocol: docs/PROTOCOL_PHASE3_SCATTERING_BENCHMARK.md
(frozen 2026-08-12). P/C1 pipeline imported unchanged from Phase 2;
this script adds the scattering features and the H3a/H3b consolidation.
Resume-capable: completed region-window JSONs are reused.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from clean_experiments.experiment_B1_true_flux_baselines import (
        _grid_geometry,
        _interior_mask,
    )
    from clean_experiments.experiment_B2_scale_irreversibility import (
        ELLS_KM,
        EPS,
        N_STEPS,
        _loo_nearest_centroid,
        compute_vorticity,
        gaussian_bar_grouped,
        run_region_window,
    )
    from clean_experiments.experiment_B2b_heldout_invariants import (
        _pairwise_dist,
        _perm_test,
        _within_between,
        _zscore_cols,
    )
    from clean_experiments.experiment_scale_gravity_einstein_box_era import (
        _load_vector_fields,
    )
except ImportError:
    from experiment_B1_true_flux_baselines import (  # type: ignore
        _grid_geometry,
        _interior_mask,
    )
    from experiment_B2_scale_irreversibility import (  # type: ignore
        ELLS_KM,
        EPS,
        N_STEPS,
        _loo_nearest_centroid,
        compute_vorticity,
        gaussian_bar_grouped,
        run_region_window,
    )
    from experiment_B2b_heldout_invariants import (  # type: ignore
        _pairwise_dist,
        _perm_test,
        _within_between,
        _zscore_cols,
    )
    from experiment_scale_gravity_einstein_box_era import (  # type: ignore
        _load_vector_fields,
    )

SEED = 20260811
REGIONS = ["R1_WPWP", "R2_NATL", "R3_AMAZ", "R4_CASIA",
           "R5_SPCZ", "R6_SATL", "R7_CONGO", "R8_AUS",
           "R9_NPAC", "R10_INDO", "R11_EURO", "R12_SAM"]
WINDOWS = ["W9_2023JFM", "W10_2023JAS", "W11_2024JFM", "W12_2024JAS"]
REGION_WINDOWS = [f"era5_wind850_{r}__{w}.nc" for r in REGIONS for w in WINDOWS]

N_PCA_H3A = 10
N_PCA_H3B = 5


def _envelopes(omega: np.ndarray, dx_km: np.ndarray, dy_km: float) -> list[np.ndarray]:
    """Six octave envelope cubes E_0..E_5, as in the Phase-2 pipeline."""
    w = np.nan_to_num(omega).astype(np.float32)
    bars = [gaussian_bar_grouped(w, ell / np.sqrt(12.0), dx_km, dy_km) for ell in ELLS_KM]
    envs = [gaussian_bar_grouped(np.abs(w - bars[0]), ELLS_KM[0] / np.sqrt(12.0), dx_km, dy_km)]
    for i in range(N_STEPS):
        envs.append(gaussian_bar_grouped(np.abs(bars[i] - bars[i + 1]),
                                         ELLS_KM[i + 1] / np.sqrt(12.0), dx_km, dy_km))
    return envs


def compute_scattering(nc_path: Path) -> dict[str, float]:
    """S1 (6) + S2 (15) features per protocol."""
    _, lat, lon, u, v, _, _ = _load_vector_fields(
        input_path=nc_path, field_set="wind", u_var=None, v_var=None,
        time_stride=1, lat_stride=1, lon_stride=1, time_start=0,
        max_time=None, crop_ny=None, crop_nx=None,
    )
    dx_km, dy_km = _grid_geometry(lat, lon)
    mask = _interior_mask(u.shape[1], u.shape[2], ELLS_KM[-1], dx_km, dy_km)
    omega = compute_vorticity(u, v, lat, lon)
    envs = _envelopes(omega, dx_km, dy_km)

    out: dict[str, float] = {}
    means = []
    for j, e in enumerate(envs):
        m_t = e[:, mask].mean(axis=1)                      # (nt,)
        means.append(m_t)
        out[f"S1_{j}"] = float(np.log(np.median(m_t) + EPS))

    for j1 in range(5):
        e1 = envs[j1]
        bars = [gaussian_bar_grouped(e1, ell / np.sqrt(12.0), dx_km, dy_km) for ell in ELLS_KM]
        for j2 in range(j1 + 1, 6):
            band = bars[j2 - 1] - bars[j2]                 # octave j2 of E_{j1}
            num = np.abs(band)[:, mask].mean(axis=1)
            ratio = num / np.maximum(means[j1], EPS)
            out[f"S2_{j1}_{j2}"] = float(np.log(np.median(ratio) + EPS))
    return out


# ---------------------------------------------------------------------------
# LOO PCA-residualization
# ---------------------------------------------------------------------------

def _fold_zscore(train: np.ndarray, test_row: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu, sd = train.mean(axis=0), train.std(axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (train - mu) / sd, (test_row - mu) / sd


def _pca_project(train_z: np.ndarray, test_z: np.ndarray, n_comp: int) -> tuple[np.ndarray, np.ndarray]:
    _, _, vt = np.linalg.svd(train_z, full_matrices=False)
    basis = vt[:n_comp].T
    return train_z @ basis, test_z @ basis


def loo_pca_residuals(x_feat: np.ndarray, y_mat: np.ndarray, n_comp: int) -> np.ndarray:
    """Residuals of y after LOO OLS on PCA(n_comp) of x_feat.

    If y_mat is None-like via callable, y per fold may itself be a fold-defined
    projection; here y_mat is fixed (P case). Returns (n, y_dim).
    """
    n = x_feat.shape[0]
    res = np.zeros_like(y_mat, dtype=float)
    for k in range(n):
        tr = np.arange(n) != k
        xtr_z, xte_z = _fold_zscore(x_feat[tr], x_feat[k])
        pc_tr, pc_te = _pca_project(xtr_z, xte_z[None, :], n_comp)
        a_tr = np.hstack([np.ones((pc_tr.shape[0], 1)), pc_tr])
        a_te = np.hstack([np.ones((1, 1)), pc_te])
        coef, *_ = np.linalg.lstsq(a_tr, y_mat[tr], rcond=None)
        res[k] = y_mat[k] - (a_te @ coef)[0]
    return res


def loo_scat_target_residuals(x_p: np.ndarray, scat: np.ndarray, n_comp: int) -> np.ndarray:
    """H3b: per fold, target = PCA(n_comp) projection of scat (basis from the
    training fold); predict it from P via OLS; residual for the held-out row."""
    n = scat.shape[0]
    res = np.zeros((n, n_comp))
    for k in range(n):
        tr = np.arange(n) != k
        str_z, ste_z = _fold_zscore(scat[tr], scat[k])
        t_tr, t_te = _pca_project(str_z, ste_z[None, :], n_comp)
        a_tr = np.hstack([np.ones((t_tr.shape[0], 1)), x_p[tr]])
        a_te = np.hstack([np.ones((1, 1)), x_p[k][None, :]])
        coef, *_ = np.linalg.lstsq(a_tr, t_tr, rcond=None)
        res[k] = t_te[0] - (a_te @ coef)[0]
    return res


def consolidate(results: list[dict], out_dir: Path) -> None:
    rng = np.random.default_rng(SEED)
    labels = np.asarray([r["region"] for r in results])
    n = len(results)

    p_mat = np.asarray([r["P_real"] for r in results])
    scat_keys = sorted(results[0]["F_scat"].keys())
    scat = np.asarray([[r["F_scat"][k] for k in scat_keys] for r in results])

    c1_count = sum(1 for r in results if r["C1_pass"])
    c1_ok = c1_count >= int(np.ceil(0.8 * n))

    res_a = loo_pca_residuals(scat, p_mat, N_PCA_H3A)
    h3a = _perm_test(_zscore_cols(res_a), labels, rng)
    res_b = loo_scat_target_residuals(p_mat, scat, N_PCA_H3B)
    h3b = _perm_test(_zscore_cols(res_b), labels, rng)

    if not c1_ok:
        verdict = "HALT_C1_CONTROL_FAILED"
    elif h3a["pass"]:
        verdict = "NOVEL"
    elif h3b["pass"]:
        verdict = "SUBSET"
    else:
        verdict = "EQUIVALENT"

    # Descriptive: signature strength P vs scat-PCA5 (global PCA, descriptive only)
    scat_z = _zscore_cols(scat)
    _, _, vt = np.linalg.svd(scat_z, full_matrices=False)
    scat5 = scat_z @ vt[:5].T
    sig_p = _perm_test(_zscore_cols(p_mat), labels, rng)
    sig_s = _perm_test(_zscore_cols(scat5), labels, rng)
    desc = {
        "signature_P": sig_p,
        "signature_scatPCA5": sig_s,
        "acc_region_P": _loo_nearest_centroid(p_mat, labels),
        "acc_region_scatPCA5": _loo_nearest_centroid(scat5, labels),
        "acc_region_scat_full": _loo_nearest_centroid(scat, labels),
        "mean_profiles": {
            reg: [float(x) for x in p_mat[labels == reg].mean(axis=0)] for reg in REGIONS
        },
    }

    summary = {
        "C1_pass_count": c1_count, "C1_ok": c1_ok, "n_region_windows": n,
        "H3a_P_beyond_scattering": h3a,
        "H3b_scattering_beyond_P": h3b,
        "PHASE3_VERDICT": verdict,
        "descriptive": desc,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["# Experiment B3: scattering benchmark", ""]
    lines.append(f"Region-windows: {n}/48. C1: {c1_count}/{n}.")
    lines.append(f"H3a (P beyond scattering): diff={h3a['diff']:+.3f} p={h3a['p']:.3f} pass={h3a['pass']}")
    lines.append(f"H3b (scattering beyond P): diff={h3b['diff']:+.3f} p={h3b['p']:.3f} pass={h3b['pass']}")
    lines.append("")
    lines.append(f"## PHASE3_VERDICT: {verdict}")
    lines.append("")
    lines.append(f"- signature strength: P diff={sig_p['diff']:+.3f} (p={sig_p['p']:.3f}); "
                 f"scat-PCA5 diff={sig_s['diff']:+.3f} (p={sig_s['p']:.3f})")
    lines.append(f"- region classification: P={desc['acc_region_P']:.3f} "
                 f"scatPCA5={desc['acc_region_scatPCA5']:.3f} scatFull={desc['acc_region_scat_full']:.3f}")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/b3"))
    parser.add_argument("--out-dir", type=Path,
                        default=_HERE / "results" / "experiment_B3_scattering_benchmark")
    parser.add_argument("--n-surrogates", type=int, default=199)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    parser.add_argument("--only", type=str, default=None)
    args = parser.parse_args()

    files = [args.data_dir / f for f in REGION_WINDOWS if (args.data_dir / f).exists()]
    if args.only:
        files = [f for f in files if args.only in f.name]
    if not files:
        raise SystemExit(f"No input files in {args.data_dir}")
    print(f"{len(files)} region-window files found", flush=True)

    results = []
    for f in files:
        tag = f.stem.replace("era5_wind850_", "")
        cached = args.out_dir / f"{tag}.json"
        if cached.exists():
            r = json.loads(cached.read_text(encoding="utf-8"))
            if "F_scat" in r:
                print(f"[{tag}] cached, skipping", flush=True)
                results.append(r)
                continue
        else:
            r = run_region_window(f, args.out_dir, args.n_surrogates, args.workers)
        if "F_scat" not in r:
            print(f"[{tag}] scattering features", flush=True)
            r["F_scat"] = compute_scattering(f)
            (args.out_dir / f"{tag}.json").write_text(json.dumps(r, indent=2), encoding="utf-8")
        results.append(r)

    if len(results) >= 12 and args.only is None:
        consolidate(results, args.out_dir)


if __name__ == "__main__":
    main()
