#!/usr/bin/env python3
"""Phase-2 B2: is the scale-irreversibility profile a geographic invariant
beyond the power spectrum?

Preregistered protocol: docs/PROTOCOL_PHASE2_SCALE_IRREVERSIBILITY.md
(frozen 2026-08-11).

Per region-window:
  - omega = dv/dx - du/dy (metric-correct, Phase-1 gradient code).
  - Gaussian ladder ell in {50,100,200,400,800,1600} km, bands
    D_i = bar_i - bar_{i+1}, envelopes E_i = |D_i| smoothed at ell_{i+1}.
  - Primary profile P: rho_i = median_t spatial Spearman of
    log E_i vs log E_{i+1} over the ell=1600 interior mask.
  - Secondary profile Q: normalized von Neumann entropies S_b of the A15
    band density matrices (6 modes/var, W=20, ridge 1e-6, shrink 0.05),
    dQ_b = S_{b+1} - S_b, first 19 steps dropped.
  - 199 phase-randomized surrogates (shared phase field on u,v per snapshot),
    both pipelines rerun per surrogate.

Criteria: C1 positive control (rho beyond surrogates), C2 nearest-centroid
LOO region classification gain from adding irreversibility features to
spectral features, C3 within-region vs between-region profile similarity.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from clean_experiments.experiment_B1_true_flux_baselines import (
        _grid_geometry,
        _interior_mask,
        phase_randomize,
    )
    from clean_experiments.experiment_M_cosmo_flow import (
        _build_band_masks,
        _select_mode_indices,
        _xy_coordinates_m,
    )
    from clean_experiments.experiment_scale_gravity_einstein_box_era import (
        _build_coefficients,
        _load_vector_fields,
    )
except ImportError:
    from experiment_B1_true_flux_baselines import (  # type: ignore
        _grid_geometry,
        _interior_mask,
        phase_randomize,
    )
    from experiment_M_cosmo_flow import (  # type: ignore
        _build_band_masks,
        _select_mode_indices,
        _xy_coordinates_m,
    )
    from experiment_scale_gravity_einstein_box_era import (  # type: ignore
        _build_coefficients,
        _load_vector_fields,
    )

ELLS_KM = [50.0, 100.0, 200.0, 400.0, 800.0, 1600.0]
N_STEPS = len(ELLS_KM) - 1          # 5 band steps
SCALE_EDGES_KM = [50.0, 100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0]
N_MODES_PER_VAR = 6
WINDOW = 20
RIDGE = 1e-6
COV_SHRINKAGE = 0.05
WARMUP = WINDOW - 1
EPS = 1e-12
SEED = 20260811

REGIONS = ["R1_WPWP", "R2_NATL", "R3_AMAZ", "R4_CASIA"]
WINDOWS = ["W1_2017JFM", "W2_2017JAS", "W3_2018JFM", "W4_2019JAS"]
REGION_WINDOWS = [f"era5_wind850_{r}__{w}.nc" for r in REGIONS for w in WINDOWS]


# ---------------------------------------------------------------------------
# Field construction and grouped metric-aware filtering
# ---------------------------------------------------------------------------

def gaussian_bar_grouped(field: np.ndarray, sigma_km: float, dx_km: np.ndarray, dy_km: float) -> np.ndarray:
    """B1 Gaussian filter with rows batched by pixel sigma rounded to 0.01.

    Accepts a 2D (ny, nx) field or a 3D (nt, ny, nx) cube; the y/x axes are
    the last two either way.
    """
    y_ax, x_ax = field.ndim - 2, field.ndim - 1
    out = gaussian_filter1d(field, sigma=sigma_km / dy_km, axis=y_ax, mode="nearest")
    keys = np.round(sigma_km / dx_km, 2)
    for k in np.unique(keys):
        rows = np.where(keys == k)[0]
        if field.ndim == 2:
            out[rows] = gaussian_filter1d(out[rows], sigma=float(k), axis=1, mode="nearest")
        else:
            out[:, rows, :] = gaussian_filter1d(out[:, rows, :], sigma=float(k), axis=2, mode="nearest")
    return out


def compute_vorticity(u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """omega = dv/dx - du/dy in 1/s with km-based, sign-correct gradients."""
    dx_km, dy_km = _grid_geometry(lat, lon)
    lat_sign = 1.0 if lat[1] > lat[0] else -1.0
    uk = np.nan_to_num(u).astype(np.float32) / 1000.0
    vk = np.nan_to_num(v).astype(np.float32) / 1000.0
    dvdx = np.gradient(vk, axis=2) / dx_km[None, :, None]
    dudy = np.gradient(uk, axis=1) / dy_km * lat_sign
    return (dvdx - dudy).astype(np.float32)


def envelope_rho_profile(
    omega: np.ndarray, dx_km: np.ndarray, dy_km: float, mask: np.ndarray
) -> np.ndarray:
    """rho_i(t) matrix (nt, 5): spatial Spearman of adjacent log envelopes."""
    w = np.nan_to_num(omega).astype(np.float32)
    bars = [gaussian_bar_grouped(w, ell / np.sqrt(12.0), dx_km, dy_km) for ell in ELLS_KM]
    # Six octave envelopes: E_0 = |sub-50km residual|, E_1..E_5 = |D_i|,
    # each smoothed at the coarse edge of its band (Deviations entry 1).
    envs = [np.log(gaussian_bar_grouped(np.abs(w - bars[0]), ELLS_KM[0] / np.sqrt(12.0), dx_km, dy_km) + EPS)]
    for i in range(N_STEPS):
        d = bars[i] - bars[i + 1]
        envs.append(np.log(gaussian_bar_grouped(np.abs(d), ELLS_KM[i + 1] / np.sqrt(12.0), dx_km, dy_km) + EPS))
    del bars
    # Rank-based rowwise correlation (Spearman without tie correction; the
    # fields are continuous so ties have measure zero).
    ranked = []
    for e in envs:
        m = e[:, mask]                                   # (nt, npix)
        ranks = np.argsort(np.argsort(m, axis=1), axis=1).astype(np.float32)
        ranked.append(ranks)
    del envs
    rho = np.zeros((w.shape[0], N_STEPS))
    for i in range(N_STEPS):
        a = ranked[i] - ranked[i].mean(axis=1, keepdims=True)
        b = ranked[i + 1] - ranked[i + 1].mean(axis=1, keepdims=True)
        denom = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
        denom = np.where(denom < 1e-12, 1.0, denom)
        rho[:, i] = (a * b).sum(axis=1) / denom
    return rho


def entropy_profile(u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Normalized von Neumann entropy S_b(t) for the 6 A15 bands, shape (nt, 6)."""
    x_m, y_m = _xy_coordinates_m(lat, lon)
    dx_km = float(np.median(np.abs(np.diff(x_m / 1000.0))))
    dy_km = float(np.median(np.abs(np.diff(y_m / 1000.0))))
    masks, _, _, _, _, wavelength_km = _build_band_masks(
        ny=u.shape[1], nx=u.shape[2], dy_km=dy_km, dx_km=dx_km, scale_edges_km=SCALE_EDGES_KM
    )
    mode_fields = {"u": u, "v": v}
    selected, _ = _select_mode_indices(
        mode_fields=mode_fields,
        masks=masks,
        n_modes_per_var=N_MODES_PER_VAR,
        wavelength_km=wavelength_km,
    )
    coeff = _build_coefficients(mode_fields=mode_fields, selected=selected)

    nt = coeff[0].shape[0]
    n_bands = len(coeff)
    s_prof = np.full((nt, n_bands), np.nan)
    for b in range(n_bands):
        arr = np.asarray(coeff[b], dtype=np.complex128)
        m = arr.shape[1]
        if m == 0:
            continue
        rms = np.sqrt(np.mean(np.abs(arr) ** 2, axis=0))
        rms = np.where(rms < 1e-12, 1.0, rms)
        arr = arr / rms[None, :]
        eye = np.eye(m, dtype=np.complex128)
        for t in range(nt):
            i0 = max(0, t - WINDOW + 1)
            a = arr[i0 : t + 1]
            c = (a.conj().T @ a) / float(len(a))
            c = 0.5 * (c + c.conj().T)
            if COV_SHRINKAGE > 0.0:
                c = (1.0 - COV_SHRINKAGE) * c + COV_SHRINKAGE * np.diag(np.diag(c))
            c = c + RIDGE * eye
            tr = float(np.real(np.trace(c)))
            rho = eye / m if tr < 1e-14 else c / tr
            vals = np.clip(np.real(np.linalg.eigvalsh(rho)), 1e-15, None)
            vals = vals / vals.sum()
            s_prof[t, b] = float(-(vals * np.log(vals)).sum() / np.log(m))
    return s_prof


# ---------------------------------------------------------------------------
# Surrogate worker
# ---------------------------------------------------------------------------

_G: dict[str, object] = {}


def _worker_init(u, v, lat, lon, dx_km, dy_km, mask) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    _G.update(u=u, v=v, lat=lat, lon=lon, dx=dx_km, dy=dy_km, mask=mask)


def _worker(seed: int) -> tuple[list[float], list[float]]:
    rng = np.random.default_rng(seed)
    us, vs = phase_randomize(_G["u"], _G["v"], rng)  # type: ignore[arg-type]
    omega_s = compute_vorticity(us, vs, _G["lat"], _G["lon"])  # type: ignore[arg-type]
    rho = envelope_rho_profile(omega_s, _G["dx"], _G["dy"], _G["mask"])  # type: ignore[arg-type]
    s_prof = entropy_profile(us, vs, _G["lat"], _G["lon"])  # type: ignore[arg-type]
    p = np.median(rho, axis=0)
    q = np.nanmedian(s_prof[WARMUP:], axis=0)
    return [float(x) for x in p], [float(x) for x in q]


# ---------------------------------------------------------------------------
# Spectral features (F_spec)
# ---------------------------------------------------------------------------

def spectral_features(omega: np.ndarray, dx_km: np.ndarray, dy_km: float, mask: np.ndarray) -> dict[str, float]:
    """Log band variances (6) + isotropic spectral slope (1)."""
    nt, ny, nx = omega.shape
    # Band variances over time and interior: [<50km residual, D_1..D_5]
    w = np.nan_to_num(omega).astype(np.float32)
    bars = [gaussian_bar_grouped(w, ell / np.sqrt(12.0), dx_km, dy_km) for ell in ELLS_KM]
    var_acc = np.zeros(6)
    var_acc[0] = float(np.mean(np.var((w - bars[0])[:, mask], axis=1)))
    for i in range(N_STEPS):
        var_acc[i + 1] = float(np.mean(np.var((bars[i] - bars[i + 1])[:, mask], axis=1)))
    del bars

    # Isotropic spectrum slope over wavelengths 100..1600 km (mean dx).
    dxm = float(np.mean(dx_km))
    ky = np.fft.fftfreq(ny, d=dy_km)
    kx = np.fft.rfftfreq(nx, d=dxm)
    kmag = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    psd = np.zeros_like(kmag)
    for t in range(nt):
        w = np.nan_to_num(omega[t])
        psd += np.abs(np.fft.rfft2(w - w.mean())) ** 2
    psd /= nt
    sel = (kmag > 1.0 / 1600.0) & (kmag < 1.0 / 100.0)
    k_bins = np.geomspace(1.0 / 1600.0, 1.0 / 100.0, 13)
    sh_k, sh_e = [], []
    for lo, hi in zip(k_bins[:-1], k_bins[1:]):
        m = sel & (kmag >= lo) & (kmag < hi)
        if m.sum() > 0:
            sh_k.append(np.sqrt(lo * hi))
            sh_e.append(float(psd[m].mean()))
    slope = float(np.polyfit(np.log(sh_k), np.log(np.asarray(sh_e) + 1e-300), 1)[0])

    out = {f"logvar_{i}": float(np.log(var_acc[i] + 1e-300)) for i in range(6)}
    out["slope"] = slope
    return out


# ---------------------------------------------------------------------------
# Per region-window driver
# ---------------------------------------------------------------------------

def run_region_window(nc_path: Path, out_dir: Path, n_surrogates: int, workers: int) -> dict:
    tag = nc_path.stem.replace("era5_wind850_", "")
    print(f"[{tag}] loading", flush=True)
    _, lat, lon, u, v, _, _ = _load_vector_fields(
        input_path=nc_path, field_set="wind", u_var=None, v_var=None,
        time_stride=1, lat_stride=1, lon_stride=1, time_start=0,
        max_time=None, crop_ny=None, crop_nx=None,
    )
    dx_km, dy_km = _grid_geometry(lat, lon)
    mask = _interior_mask(u.shape[1], u.shape[2], ELLS_KM[-1], dx_km, dy_km)

    print(f"[{tag}] real pipelines", flush=True)
    omega = compute_vorticity(u, v, lat, lon)
    rho = envelope_rho_profile(omega, dx_km, dy_km, mask)
    s_prof = entropy_profile(u, v, lat, lon)
    p_real = np.median(rho, axis=0)
    q_real = np.nanmedian(s_prof[WARMUP:], axis=0)
    dq_real = np.diff(q_real)
    f_spec = spectral_features(omega, dx_km, dy_km, mask)

    print(f"[{tag}] surrogates n={n_surrogates}", flush=True)
    tag_code = zlib.crc32(tag.encode()) % 100000
    seeds = [int(s) for s in np.random.SeedSequence(SEED + tag_code).generate_state(n_surrogates)]
    if workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers, _worker_init, (u, v, lat, lon, dx_km, dy_km, mask)) as pool:
            sur = pool.map(_worker, seeds)
    else:
        _worker_init(u, v, lat, lon, dx_km, dy_km, mask)
        sur = [_worker(s) for s in seeds]
    sur_p = np.asarray([s[0] for s in sur])      # (n_sur, 5)
    sur_q = np.asarray([s[1] for s in sur])      # (n_sur, 6)

    p95 = np.percentile(sur_p, 95, axis=0)
    c1_bands = int(np.sum(p_real > p95))
    corr_pq = spearmanr(p_real, -dq_real[: len(p_real)]).statistic if len(dq_real) >= len(p_real) else np.nan

    result = {
        "tag": tag,
        "region": tag.split("__")[0],
        "window": tag.split("__")[1],
        "P_real": [float(x) for x in p_real],
        "P_surrogate_p95": [float(x) for x in p95],
        "P_surrogate_median": [float(x) for x in np.median(sur_p, axis=0)],
        "C1_bands_beyond": c1_bands,
        "C1_pass": bool(c1_bands >= 3),
        "Q_real": [float(x) for x in q_real],
        "dQ_real": [float(x) for x in dq_real],
        "Q_surrogate_median": [float(x) for x in np.median(sur_q, axis=0)],
        "corr_P_vs_negdQ": float(corr_pq) if np.isfinite(corr_pq) else None,
        "F_spec": f_spec,
        "n_surrogates": n_surrogates,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tag}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[{tag}] done C1_bands={c1_bands}", flush=True)
    return result


# ---------------------------------------------------------------------------
# C2/C3 consolidation
# ---------------------------------------------------------------------------

def _loo_nearest_centroid(x: np.ndarray, labels: np.ndarray) -> float:
    n = len(labels)
    correct = 0
    for i in range(n):
        tr = np.arange(n) != i
        mu = x[tr].mean(axis=0)
        sd = x[tr].std(axis=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        xtr = (x[tr] - mu) / sd
        xte = (x[i] - mu) / sd
        cents = {}
        for lab in np.unique(labels[tr]):
            cents[lab] = xtr[labels[tr] == lab].mean(axis=0)
        pred = min(cents, key=lambda lab: float(np.sum((xte - cents[lab]) ** 2)))
        correct += int(pred == labels[i])
    return correct / n


def consolidate(results: list[dict], out_dir: Path) -> None:
    rng = np.random.default_rng(SEED)
    labels = np.asarray([r["region"] for r in results])
    n = len(results)

    spec_keys = sorted(results[0]["F_spec"].keys())
    x_spec = np.asarray([[r["F_spec"][k] for k in spec_keys] for r in results])
    x_irr = np.asarray([r["P_real"] + r["dQ_real"] for r in results])
    x_both = np.hstack([x_spec, x_irr])

    acc_spec = _loo_nearest_centroid(x_spec, labels)
    acc_both = _loo_nearest_centroid(x_both, labels)
    gain_obs = acc_both - acc_spec

    perm_gains = []
    for _ in range(999):
        perm = rng.permutation(n)
        xb = np.hstack([x_spec, x_irr[perm]])
        perm_gains.append(_loo_nearest_centroid(xb, labels) - acc_spec)
    p_c2 = float((1 + np.sum(np.asarray(perm_gains) >= gain_obs)) / 1000)
    c2 = bool(gain_obs >= 2.0 / n and p_c2 < 0.05)

    profs = np.asarray([r["P_real"] for r in results])
    sims = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            sims[i, j] = spearmanr(profs[i], profs[j]).statistic

    def within_between(labs: np.ndarray) -> tuple[float, float]:
        w, b = [], []
        for i in range(n):
            for j in range(i + 1, n):
                (w if labs[i] == labs[j] else b).append(sims[i, j])
        return float(np.median(w)), float(np.median(b))

    w_obs, b_obs = within_between(labels)
    diff_obs = w_obs - b_obs
    perm_diffs = []
    for _ in range(999):
        labs = labels[rng.permutation(n)]
        w_p, b_p = within_between(labs)
        perm_diffs.append(w_p - b_p)
    p_c3 = float((1 + np.sum(np.asarray(perm_diffs) >= diff_obs)) / 1000)
    c3 = bool(diff_obs > 0 and p_c3 < 0.05)

    c1_count = sum(1 for r in results if r["C1_pass"])
    c1_ok = c1_count >= 12

    verdict = "POSITIVE" if (c1_ok and c2 and c3) else "NEGATIVE"
    if not c1_ok:
        verdict = "HALT_C1_CONTROL_FAILED"

    summary = {
        "C1_pass_count": c1_count,
        "C1_ok": c1_ok,
        "C2": {"acc_spec": acc_spec, "acc_spec_plus_irr": acc_both,
               "gain": gain_obs, "p": p_c2, "pass": c2},
        "C3": {"median_within": w_obs, "median_between": b_obs,
               "diff": diff_obs, "p": p_c3, "pass": c3},
        "PHASE2_VERDICT": verdict,
        "n_region_windows": n,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = ["# Experiment B2: scale-irreversibility profile as geographic invariant", ""]
    lines.append(f"Region-windows: {n}/16. C1 positive control: {c1_count}/{n} pass (need >=12/16).")
    lines.append("")
    lines.append("| tag | P (rho_1..rho_5) | C1 bands>surr95 | corr(P,-dQ) |")
    lines.append("|---|---|---|---|")
    for r in results:
        p_str = ", ".join(f"{x:.3f}" for x in r["P_real"])
        lines.append(
            f"| {r['tag']} | {p_str} | {r['C1_bands_beyond']}/5 | "
            f"{r['corr_P_vs_negdQ'] if r['corr_P_vs_negdQ'] is not None else float('nan'):.3f} |"
        )
    lines.append("")
    lines.append(f"C2: acc(spec)={acc_spec:.3f}, acc(spec+irr)={acc_both:.3f}, "
                 f"gain={gain_obs:+.3f}, p={p_c2:.3f}, pass={c2}")
    lines.append(f"C3: within={w_obs:.3f}, between={b_obs:.3f}, diff={diff_obs:+.3f}, "
                 f"p={p_c3:.3f}, pass={c3}")
    lines.append("")
    lines.append(f"## PHASE2_VERDICT: {verdict}")
    (out_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/b1"))
    parser.add_argument("--out-dir", type=Path,
                        default=_HERE / "results" / "experiment_B2_scale_irreversibility")
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

    results = [run_region_window(f, args.out_dir, args.n_surrogates, args.workers) for f in files]
    if len(results) >= 4:
        consolidate(results, args.out_dir)


if __name__ == "__main__":
    main()
