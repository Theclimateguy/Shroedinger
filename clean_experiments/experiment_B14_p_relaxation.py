#!/usr/bin/env python3
"""Phase-14: regime-dependent relaxation dynamics of the profile P.

Preregistered protocol: docs/PROTOCOL_PHASE14_P_RELAXATION.md
(frozen 2026-08-16).

Model per region-window (composite of resolved band pairs 3, 4):
  DeltaP(t) = a + alpha*E(t) - gamma*P(t) + eta(t)
the Euler form of dP/dt = -gamma (P - P^eq(regime)) + eta with
P^eq = (a + alpha*E)/gamma. P is instantaneous (no estimator window).

  --stage series    : P_b(t), V(t), E(t), R(t) on data/b2b (+cape+precip)
  --stage series-b3 : P_b(t) on data/b3 for the descriptive ENSO contrast
  --stage synthetic : H14a estimator power + persistence specificity gate
  --stage tests     : H14b-H14f, descriptives, verdict
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import chi2

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from clean_experiments.experiment_B1_true_flux_baselines import (
        _grid_geometry, _interior_mask,
    )
    from clean_experiments.experiment_B2_scale_irreversibility import (
        ELLS_KM, EPS, N_STEPS, compute_vorticity, envelope_rho_profile,
        gaussian_bar_grouped,
    )
    from clean_experiments.experiment_scale_gravity_einstein_box_era import (
        _load_vector_fields,
    )
except ImportError:  # pragma: no cover
    from experiment_B1_true_flux_baselines import _grid_geometry, _interior_mask  # type: ignore
    from experiment_B2_scale_irreversibility import (  # type: ignore
        ELLS_KM, EPS, N_STEPS, compute_vorticity, envelope_rho_profile,
        gaussian_bar_grouped,
    )
    from experiment_scale_gravity_einstein_box_era import _load_vector_fields  # type: ignore

SEED = 20260816
OUT = _HERE / "results" / "experiment_B14_p_relaxation"
WIND_DIR = Path("data/b2b")
CAPE_DIR = Path("data/b6cape")
PRECIP_DIR = Path("data/b5precip")
WIND_B3_DIR = Path("data/b3")

FIT_REGIONS = ("R5_SPCZ", "R6_SATL", "R7_CONGO", "R8_AUS")
VAL_REGIONS = ("R9_NPAC", "R10_INDO", "R11_EURO", "R12_SAM")
PAIRS_RESOLVED = (3, 4)

N_NULL = 500           # H14c circular-shift nulls
N_NULL_INNER = 200     # inner nulls for H14a
N_SURR = 200           # phase-randomized surrogates per fit rw (H14a-ii)
SHIFT_MIN_D, SHIFT_MAX_D = 20, 70   # whole-day shifts, both signs
STEPS_PER_DAY = 4


# --------------------------------------------------------------------------
# series
# --------------------------------------------------------------------------

def _interior_mean(path: Path, var: str, mask: np.ndarray,
                   transform) -> np.ndarray:
    import xarray as xr
    ds = xr.open_dataset(path)
    da = ds[var].squeeze()
    tdim = next(d for d in da.dims if "time" in d or d == "valid_time")
    arr = np.nan_to_num(np.asarray(da.transpose(tdim, ...).values, dtype=float))
    ds.close()
    return transform(arr)[:, mask].mean(axis=1)


def band_amplitude(omega: np.ndarray, dx_km: np.ndarray, dy_km: float,
                   mask: np.ndarray) -> np.ndarray:
    """V_i(t) = log interior mean of D_i(t)^2 for the resolved pairs 3, 4."""
    w = np.nan_to_num(omega).astype(np.float32)
    bars = [gaussian_bar_grouped(w, ell / np.sqrt(12.0), dx_km, dy_km)
            for ell in ELLS_KM]
    out = np.zeros((w.shape[0], N_STEPS))
    for i in range(N_STEPS):
        d = bars[i] - bars[i + 1]
        out[:, i] = np.log(np.mean(d[:, mask] ** 2, axis=1) + EPS)
    return out


def _series_one(args_tuple) -> str:
    wind, cape, precip = args_tuple
    os.environ["OMP_NUM_THREADS"] = "1"
    tag = wind.stem.replace("era5_wind850_", "")
    cached = OUT / f"series_{tag}.npz"
    if cached.exists():
        return tag
    import xarray as xr
    _, lat, lon, u, v, _, _ = _load_vector_fields(
        input_path=wind, field_set="wind", u_var=None, v_var=None,
        time_stride=1, lat_stride=1, lon_stride=1, time_start=0,
        max_time=None, crop_ny=None, crop_nx=None,
    )
    ds = xr.open_dataset(wind)
    tname = "valid_time" if "valid_time" in ds else "time"
    hod = np.asarray(ds[tname].dt.hour.values, dtype=int)
    ds.close()
    dx_km, dy_km = _grid_geometry(lat, lon)
    mask = _interior_mask(u.shape[1], u.shape[2], ELLS_KM[-1], dx_km, dy_km)
    omega = compute_vorticity(u, v, lat, lon)
    P = envelope_rho_profile(omega, dx_km, dy_km, mask)          # (nt, 5)
    V = band_amplitude(omega, dx_km, dy_km, mask)                # (nt, 5)
    payload = {"P": P.astype(np.float32), "V": V.astype(np.float32),
               "hod": hod[: P.shape[0]]}
    if cape is not None:
        payload["E"] = _interior_mean(
            cape, "cape", mask,
            lambda a: np.log1p(np.maximum(a, 0.0))).astype(np.float32)
        payload["R"] = _interior_mean(
            precip, "tp", mask,
            lambda a: np.log(np.maximum(a, 0.0) + 1e-6)).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cached, **payload)
    print(f"[{tag}] series cached nt={P.shape[0]}", flush=True)
    return tag


def stage_series(workers: int, b3: bool = False) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    if not b3:
        for wf in sorted(WIND_DIR.glob("era5_wind850_*.nc")):
            tag = wf.stem.replace("era5_wind850_", "")
            cf = CAPE_DIR / f"era5_cape6h_{tag}.nc"
            pf = PRECIP_DIR / f"era5_precip_{tag}.nc"
            if cf.exists() and pf.exists():
                jobs.append((wf, cf, pf))
    else:
        keep = FIT_REGIONS + VAL_REGIONS
        for wf in sorted(WIND_B3_DIR.glob("era5_wind850_*.nc")):
            tag = wf.stem.replace("era5_wind850_", "")
            if tag.split("__")[0] in keep:
                jobs.append((wf, None, None))
    print(f"{len(jobs)} region-windows to process", flush=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        pool.map(_series_one, jobs)


# --------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------

def deseason(x: np.ndarray, hod: np.ndarray) -> np.ndarray:
    """Remove hour-of-day means, then a linear trend (protocol, fixed)."""
    y = np.asarray(x, dtype=float).copy()
    for h in np.unique(hod):
        m = hod == h
        y[m] -= y[m].mean()
    t = np.arange(len(y), dtype=float)
    t -= t.mean()
    y -= (t @ y) / (t @ t) * t
    return y


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / (x.std() + 1e-12)


def load_rw(tag: str, raw: bool = False) -> dict:
    z = np.load(OUT / f"series_{tag}.npz")
    P5 = np.asarray(z["P"], dtype=float)
    hod = np.asarray(z["hod"], dtype=int)
    comp = P5[:, list(PAIRS_RESOLVED)].mean(axis=1)
    Vcomp = np.asarray(z["V"], dtype=float)[:, list(PAIRS_RESOLVED)].mean(axis=1)
    out = {"tag": tag, "hod": hod, "P5_raw": P5}
    if raw:
        out["P"] = comp - comp.mean()
        out["V"] = _zscore(Vcomp)
    else:
        out["P"] = deseason(comp, hod)
        out["V"] = _zscore(deseason(Vcomp, hod))
        out["P5"] = np.column_stack([deseason(P5[:, b], hod) for b in range(N_STEPS)])
    if "E" in z:
        E = np.asarray(z["E"], dtype=float)
        R = np.asarray(z["R"], dtype=float)
        out["E"] = _zscore(E - E.mean() if raw else deseason(E, hod))
        out["E_raw"] = E
        out["R"] = _zscore(R - R.mean() if raw else deseason(R, hod))
    return out


# --------------------------------------------------------------------------
# fitting machinery
# --------------------------------------------------------------------------

def ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(X, y, rcond=None)[0]


def design(P: np.ndarray, cov: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    y = P[1:] - P[:-1]
    cols = [np.ones_like(y)]
    if cov is not None:
        cols.append(cov[:-1])
    cols.append(P[:-1])
    return np.column_stack(cols), y


def alpha_gamma(P: np.ndarray, cov: np.ndarray | None) -> tuple[float, float, np.ndarray]:
    X, y = design(P, cov)
    c = ols(X, y)
    resid = y - X @ c
    if cov is None:
        return np.nan, float(-c[1]), resid
    return float(c[1]), float(-c[2]), resid


def shift_offsets(rng: np.random.Generator, n: int, nt: int) -> np.ndarray:
    days = np.arange(SHIFT_MIN_D, SHIFT_MAX_D + 1)
    steps = np.concatenate([days, -days]) * STEPS_PER_DAY
    steps = steps[np.abs(steps) < nt - 8]
    return rng.choice(steps, size=n, replace=True)


def null_alphas(P: np.ndarray, E: np.ndarray, n_null: int,
                rng: np.random.Generator) -> np.ndarray:
    offs = shift_offsets(rng, n_null, len(P))
    out = np.empty(n_null)
    for j, s in enumerate(offs):
        out[j] = alpha_gamma(P, np.roll(E, int(s)))[0]
    return out


def coupling_test(P: np.ndarray, E: np.ndarray, n_null: int,
                  rng: np.random.Generator) -> dict:
    a, g, _ = alpha_gamma(P, E)
    nulls = null_alphas(P, E, n_null, rng)
    q95 = float(np.quantile(np.abs(nulls), 0.95))
    return {"alpha": a, "gamma": g, "null_q95_abs": q95,
            "pass": bool(abs(a) > q95),
            "p_perm": float((np.sum(np.abs(nulls) >= abs(a)) + 1) / (n_null + 1))}


def gamma_step_scan(P: np.ndarray) -> dict:
    gs = {}
    for k in (1, 2, 4):
        X = np.column_stack([np.ones(len(P) - k), P[:-k]])
        c = ols(X, P[k:])
        gs[k] = float(-np.log(c[1]) / k) if c[1] > 0 else np.nan
    acf = [float(np.corrcoef(P[:-k], P[k:])[0, 1]) for k in range(1, 9)]
    pos = [(k, r) for k, r in zip(range(1, 9), acf) if r > 0]
    if len(pos) >= 4:
        kk = np.array([p[0] for p in pos], dtype=float)
        lr = np.log([p[1] for p in pos])
        A = np.column_stack([np.ones_like(kk), kk])
        c = ols(A, lr)
        ss_res = float(np.sum((lr - A @ c) ** 2))
        ss_tot = float(np.sum((lr - lr.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    else:
        r2 = np.nan
    vals = [v for v in gs.values() if np.isfinite(v) and v > 0]
    ratio = max(vals) / min(vals) if len(vals) == 3 else np.nan
    return {"gamma_by_step": gs, "ratio": ratio, "acf": acf, "log_acf_r2": r2}


def ljung_box(resid: np.ndarray, nlags: int = 10) -> float:
    r = resid - resid.mean()
    n = len(r)
    denom = float(r @ r)
    q = 0.0
    for k in range(1, nlags + 1):
        rk = float(r[:-k] @ r[k:]) / denom
        q += rk * rk / (n - k)
    q *= n * (n + 2)
    return float(chi2.sf(q, nlags))


def phase_randomize(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(x)
    f = np.fft.rfft(x - x.mean())
    ph = rng.uniform(0, 2 * np.pi, len(f))
    ph[0] = 0.0
    if n % 2 == 0:
        ph[-1] = 0.0
    return np.fft.irfft(np.abs(f) * np.exp(1j * ph), n=n) + x.mean()


# --------------------------------------------------------------------------
# H14a synthetic
# --------------------------------------------------------------------------

def simulate_ou(E: np.ndarray, gamma: float, d: float, sd_target: float,
                rng: np.random.Generator, coupled: bool) -> np.ndarray:
    nt = len(E)
    a = 1.0 - gamma
    sig = sd_target * np.sqrt(max(1.0 - a * a, 1e-6))
    p = np.zeros(nt)
    eps = rng.standard_normal(nt)
    for t in range(nt - 1):
        eq = d * E[t] if coupled else 0.0
        p[t + 1] = p[t] + gamma * (eq - p[t]) + sig * eps[t]
    return p + 0.25 * sd_target * rng.standard_normal(nt)


def stage_synthetic() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "synthetic_h14a.json"
    if cache.exists():
        print("cached", flush=True)
        return
    rng = np.random.default_rng(SEED)
    fit_tags = [f.stem.replace("series_", "") for f in sorted(OUT.glob("series_*.npz"))
                if f.stem.replace("series_", "").split("__")[0] in FIT_REGIONS]
    power_rows, fp_flags = [], []
    for tag in fit_tags:
        rw = load_rw(tag)
        E, P_real = rw["E"], rw["P"]
        sd = float(P_real.std())
        # (i) power / recovery
        for gt in (0.05, 0.2, 0.5):
            for rep in range(20):
                ps = simulate_ou(E, gt, 0.5 * sd, sd, rng, coupled=True)
                al, gh, _ = alpha_gamma(ps, E)
                det = coupling_test(ps, E, N_NULL_INNER, rng)["pass"] if gt == 0.2 else None
                power_rows.append({"tag": tag, "gamma_true": gt, "rep": rep,
                                   "gamma_hat": gh, "alpha_hat": al,
                                   "detected": det})
        # (ii) specificity on phase-randomized surrogates of the REAL P
        for s in range(N_SURR):
            ps = phase_randomize(P_real, rng)
            fp_flags.append(bool(coupling_test(ps, E, N_NULL_INNER, rng)["pass"]))
        print(f"[{tag}] synthetic done", flush=True)
    res = {"power_rows": power_rows, "fp_flags": fp_flags}
    cache.write_text(json.dumps(res), encoding="utf-8")


def score_h14a() -> dict:
    d = json.loads((OUT / "synthetic_h14a.json").read_text())
    rows = d["power_rows"]
    out: dict = {}
    for gt in (0.05, 0.2, 0.5):
        gh = np.array([r["gamma_hat"] for r in rows if r["gamma_true"] == gt])
        gh = gh[np.isfinite(gh) & (gh > 0)]
        out[f"recovery_gamma_{gt}"] = {
            "median_gamma_hat": float(np.median(gh)),
            "median_abs_log_ratio": float(np.median(np.abs(np.log(gh / gt)))),
        }
    det = [r["detected"] for r in rows if r["gamma_true"] == 0.2]
    out["power_at_gamma_0.2"] = float(np.mean([bool(x) for x in det]))
    fp = np.asarray(d["fp_flags"], dtype=bool)
    out["fp_rate"] = float(fp.mean())
    out["n_surrogates"] = int(fp.size)
    rec_ok = all(out[f"recovery_gamma_{g}"]["median_abs_log_ratio"] <= np.log(1.5)
                 for g in (0.2, 0.5))
    out["H14a_power_pass"] = bool(rec_ok and out["power_at_gamma_0.2"] >= 0.80)
    out["H14a_specificity_pass"] = bool(out["fp_rate"] <= 0.10)
    return out


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def _pooled_fit(rws: list[dict], cov_key: str | None) -> np.ndarray:
    Xs, ys = [], []
    for rw in rws:
        X, y = design(rw["P"], rw[cov_key] if cov_key else None)
        Xs.append(X)
        ys.append(y)
    return ols(np.vstack(Xs), np.concatenate(ys))


def _val_mse(rw: dict, coef: np.ndarray, cov_key: str | None) -> float:
    X, y = design(rw["P"], rw[cov_key] if cov_key else None)
    return float(np.mean((y - X @ coef) ** 2))


def stage_tests() -> dict:
    rng = np.random.default_rng(SEED)
    tags = sorted(f.stem.replace("series_", "") for f in OUT.glob("series_*.npz")
                  if "__W" in f.stem and any(
                      f.stem.replace("series_", "").split("__")[1].startswith(w)
                      for w in ("W5", "W6", "W7", "W8")))
    fit_tags = [t for t in tags if t.split("__")[0] in FIT_REGIONS]
    val_tags = [t for t in tags if t.split("__")[0] in VAL_REGIONS]
    fit = {t: load_rw(t) for t in fit_tags}
    val = {t: load_rw(t) for t in val_tags}
    allrw = {**fit, **val}
    res: dict = {"seed": SEED, "protocol": "docs/PROTOCOL_PHASE14_P_RELAXATION.md",
                 "n_fit": len(fit_tags), "n_val": len(val_tags)}

    # ---- H14a gate -------------------------------------------------------
    res["H14a"] = score_h14a()
    if not res["H14a"]["H14a_specificity_pass"]:
        res["PHASE14_VERDICT"] = "ESTIMATOR_INVALID"
        return res

    # ---- H14b step invariance -------------------------------------------
    scans = {t: gamma_step_scan(rw["P"]) for t, rw in allrw.items()}
    ratios = np.array([s["ratio"] for s in scans.values()])
    r2s = np.array([s["log_acf_r2"] for s in scans.values()])
    res["H14b"] = {
        "median_gamma_step_ratio": float(np.nanmedian(ratios)),
        "median_log_acf_r2": float(np.nanmedian(r2s)),
        "per_rw": {t: {"gamma_by_step": s["gamma_by_step"], "ratio": s["ratio"],
                       "log_acf_r2": s["log_acf_r2"]} for t, s in scans.items()},
    }
    res["H14b_pass"] = bool(res["H14b"]["median_gamma_step_ratio"] <= 2.0
                            and res["H14b"]["median_log_acf_r2"] >= 0.90)

    # ---- H14c coupling vs coherent null ---------------------------------
    per = {}
    for t, rw in fit.items():
        per[t] = coupling_test(rw["P"], rw["E"], N_NULL, rng)
    n_pass = sum(1 for v in per.values() if v["pass"])
    signs = [np.sign(v["alpha"]) for v in per.values() if v["pass"]]
    n_sign = int(max((signs.count(s) for s in (-1.0, 1.0)), default=0))
    # pooled statistic against pooled null
    coef_pool = _pooled_fit(list(fit.values()), "E")
    alpha_pool = float(coef_pool[1])
    pool_null = np.empty(N_NULL)
    for j in range(N_NULL):
        Xs, ys = [], []
        for rw in fit.values():
            s = int(shift_offsets(rng, 1, len(rw["P"]))[0])
            X, y = design(rw["P"], np.roll(rw["E"], s))
            Xs.append(X)
            ys.append(y)
        pool_null[j] = ols(np.vstack(Xs), np.concatenate(ys))[1]
    q95_pool = float(np.quantile(np.abs(pool_null), 0.95))
    res["H14c"] = {
        "per_rw": per, "n_pass": n_pass, "n_consistent_sign": n_sign,
        "alpha_pooled": alpha_pool, "pooled_null_q95_abs": q95_pool,
        "pooled_pass": bool(abs(alpha_pool) > q95_pool),
        "p_perm_pooled": float((np.sum(np.abs(pool_null) >= abs(alpha_pool)) + 1)
                               / (N_NULL + 1)),
    }
    res["H14c_pass"] = bool(n_pass >= 12 and n_sign >= 12
                            and res["H14c"]["pooled_pass"])

    # ---- H14d held-out transfer -----------------------------------------
    det_d = {}
    n_win = 0
    for t, rw in val.items():
        mse_full = _val_mse(rw, coef_pool, "E")
        X, y = design(rw["P"], None)
        c_ar = ols(X, y)
        mse_ar = float(np.mean((y - X @ c_ar) ** 2))
        win = mse_full < mse_ar
        n_win += int(win)
        det_d[t] = {"mse_full_frozen": mse_full, "mse_ar1_local": mse_ar,
                    "win": bool(win)}
    res["H14d"] = {"per_rw": det_d, "n_win": n_win}
    res["H14d_pass"] = bool(n_win >= 12)

    # ---- H14e amplitude placebo -----------------------------------------
    coef_plc = _pooled_fit(list(fit.values()), "V")
    det_e = {}
    n_beat = 0
    for t, rw in val.items():
        mse_cape = _val_mse(rw, coef_pool, "E")
        mse_plc = _val_mse(rw, coef_plc, "V")
        beat = mse_cape < mse_plc
        n_beat += int(beat)
        det_e[t] = {"mse_cape": mse_cape, "mse_placebo": mse_plc,
                    "cape_beats": bool(beat)}
    res["H14e"] = {"per_rw": det_e, "n_cape_beats_placebo": n_beat,
                   "placebo_coef": float(coef_plc[1])}
    res["H14e_pass"] = bool(res["H14d_pass"] and n_beat >= 12)

    # ---- H14f residual whiteness at daily subsampling -------------------
    lb = {}
    for t, rw in allrw.items():
        _, _, resid = alpha_gamma(rw["P"][::4], rw["E"][::4])
        lb[t] = ljung_box(resid, 10)
    n_white = sum(1 for p in lb.values() if p > 0.01)
    res["H14f"] = {"per_rw": lb, "n_white": n_white}
    res["H14f_pass"] = bool(n_white >= 16)

    # ---- descriptives ----------------------------------------------------
    res["descriptive"] = _descriptives(allrw, fit, val, rng)

    # ---- verdict ---------------------------------------------------------
    if not res["H14b_pass"]:
        v = "FORM_REJECTED"
    elif not res["H14c_pass"]:
        v = "NO_REGIME_COUPLING"
    elif not (res["H14d_pass"] and res["H14e_pass"]):
        v = "COUPLING_NOT_TRANSFERABLE"
    else:
        v = "P_DYNAMICS_ESTABLISHED" if res["H14f_pass"] else "P_DYNAMICS_PARTIAL"
    res["PHASE14_VERDICT"] = v
    return res


def _descriptives(allrw: dict, fit: dict, val: dict,
                  rng: np.random.Generator) -> dict:
    out: dict = {}
    # CAPE-tercile equilibrium monotonicity
    mono = []
    terc = {}
    for t, rw in allrw.items():
        q = np.quantile(rw["E"], [1 / 3, 2 / 3])
        lo = rw["P"][rw["E"] <= q[0]].mean()
        mid = rw["P"][(rw["E"] > q[0]) & (rw["E"] <= q[1])].mean()
        hi = rw["P"][rw["E"] > q[1]].mean()
        terc[t] = [float(lo), float(mid), float(hi)]
        mono.append(np.sign(hi - lo) if (mid - lo) * (hi - mid) > 0 else 0.0)
    out["tercile_means"] = terc
    out["n_monotone_up"] = int(sum(1 for m in mono if m > 0))
    out["n_monotone_down"] = int(sum(1 for m in mono if m < 0))
    # precip covariate sign counts (full model + R)
    signs_r = []
    for t, rw in allrw.items():
        X, y = design(rw["P"], rw["E"])
        X = np.column_stack([X[:, 0], X[:, 1], rw["R"][:-1], X[:, 2]])
        c = ols(X, y)
        signs_r.append(float(np.sign(c[2])))
    out["precip_coef_positive"] = int(sum(1 for s in signs_r if s > 0))
    out["precip_coef_total"] = len(signs_r)
    # per-band H14c/H14d
    per_band = {}
    for b in range(N_STEPS):
        pb = {}
        for t, rw in fit.items():
            pb[t] = coupling_test(rw["P5"][:, b], rw["E"], 200, rng)
        per_band[f"band_{b}"] = {
            "n_pass": sum(1 for v in pb.values() if v["pass"]),
            "alphas": {t: v["alpha"] for t, v in pb.items()},
        }
    out["per_band_coupling_fit_set"] = per_band
    # raw (non-deseasonalized) composite coupling, for the record
    per_raw = {}
    for t in fit:
        rw = load_rw(t, raw=True)
        per_raw[t] = coupling_test(rw["P"], rw["E"], 200, rng)["pass"]
    out["raw_series_coupling_n_pass"] = int(sum(per_raw.values()))
    # ENSO window contrast (needs series-b3)
    b3tags = [f.stem.replace("series_", "") for f in OUT.glob("series_*.npz")
              if any(f.stem.split("__")[-1].startswith(w)
                     for w in ("W9", "W10", "W11", "W12"))]
    med = {}
    for t in list(allrw) + b3tags:
        z = np.load(OUT / f"series_{t}.npz")
        P5 = np.asarray(z["P"], dtype=float)
        med[t] = float(np.median(P5[:, list(PAIRS_RESOLVED)].mean(axis=1)))
    def _w(tag: str) -> str:
        return tag.split("__")[1].split("_")[0]
    regions = sorted({t.split("__")[0] for t in med})
    contrasts = {}
    for label, la, el in (("JFM_LaNina_vs_ElNino", ("W5", "W7"), ("W11",)),
                          ("JAS_LaNina_vs_after", ("W6", "W8"), ("W10", "W12"))):
        diffs = []
        for r in regions:
            a = [med[t] for t in med if t.startswith(r + "__") and _w(t) in la]
            b = [med[t] for t in med if t.startswith(r + "__") and _w(t) in el]
            if a and b:
                diffs.append(float(np.mean(a) - np.mean(b)))
        contrasts[label] = {"n_regions": len(diffs),
                            "n_lanina_higher": int(sum(1 for d in diffs if d > 0)),
                            "median_diff": float(np.median(diffs)) if diffs else None}
    out["enso_window_contrast"] = contrasts
    return out


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["series", "series-b3", "synthetic", "tests"],
                    default="series")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    if args.stage == "series":
        stage_series(args.workers, b3=False)
    elif args.stage == "series-b3":
        stage_series(args.workers, b3=True)
    elif args.stage == "synthetic":
        stage_synthetic()
    else:
        res = stage_tests()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "summary.json").write_text(json.dumps(res, indent=2),
                                          encoding="utf-8")
        brief = {k: res[k] for k in res
                 if k in ("n_fit", "n_val", "H14a", "H14b_pass", "H14c_pass",
                          "H14d_pass", "H14e_pass", "H14f_pass",
                          "PHASE14_VERDICT")}
        brief["H14c_n_pass"] = res.get("H14c", {}).get("n_pass")
        brief["H14d_n_win"] = res.get("H14d", {}).get("n_win")
        print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
