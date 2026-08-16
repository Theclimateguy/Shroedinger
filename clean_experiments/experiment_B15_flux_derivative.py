#!/usr/bin/env python3
"""Phase-15: flux-derivative dynamics.

Preregistered protocol: docs/PROTOCOL_PHASE15_FLUX_DERIVATIVE.md
(frozen 2026-08-16). Author's postulation corrected per section 9 there.

K_b(t) = 6-h difference of the net Aluie cross-scale KE transfer into band
b (deseasonalized), summarized as full-window RMS. Tested against the
frozen Phase-9 predictability metrics (lambda, mu, T50).

  --stage series : Pi_ell, band fluxes, band KE, boundary inflow (80 rws)
  --stage sanity : C15-1 regression test against frozen B1 code
  --stage gate   : C15-2 persistence gate on Phase-12 synthetic ladders
  --stage tests  : H15a-H15f, placebos, verdict
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kstest, spearmanr

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import clean_experiments.download_b1_era5_wind as dl_b1
    from clean_experiments.download_b2_era5_wind import REGIONS_B2
    from clean_experiments.download_b2b_era5_wind import REGIONS_B2B
    from clean_experiments.experiment_B1_true_flux_baselines import (
        _gaussian_bar, _gradients, _grid_geometry, _interior_mask,
        flux_and_gridded_baselines,
    )
    from clean_experiments.experiment_B12_estimator_validity import synthetic_pair
    from clean_experiments.experiment_B14_p_relaxation import deseason
    from clean_experiments.experiment_scale_gravity_einstein_box_era import (
        _load_vector_fields,
    )
except ImportError:  # pragma: no cover
    import download_b1_era5_wind as dl_b1  # type: ignore
    from download_b2_era5_wind import REGIONS_B2  # type: ignore
    from download_b2b_era5_wind import REGIONS_B2B  # type: ignore
    from experiment_B1_true_flux_baselines import (  # type: ignore
        _gaussian_bar, _gradients, _grid_geometry, _interior_mask,
        flux_and_gridded_baselines,
    )
    from experiment_B12_estimator_validity import synthetic_pair  # type: ignore
    from experiment_B14_p_relaxation import deseason  # type: ignore
    from experiment_scale_gravity_einstein_box_era import _load_vector_fields  # type: ignore

SEED = 20260816
OUT = _HERE / "results" / "experiment_B15_flux_derivative"
B9 = _HERE / "results" / "experiment_B9_predictability"
B3R = _HERE / "results" / "experiment_B3_scattering_benchmark"
B2BR = _HERE / "results" / "experiment_B2b_heldout_invariants"

EDGES_KM = [200.0, 400.0, 800.0, 1600.0]     # band edges; bands 2,3,4
BAND_LABELS = (2, 3, 4)                       # matches Phase-9 bands
PRIMARY = (0, 1)                              # indices of bands 2,3 in arrays
N_PERM = 999
REGIONS_ALL = {**dl_b1.REGIONS, **REGIONS_B2, **REGIONS_B2B}

WINDOWS_FIT = ("W9", "W10", "W11", "W12")
WINDOWS_OOS = ("W5", "W6", "W7", "W8")


# --------------------------------------------------------------------------
# series
# --------------------------------------------------------------------------

def flux_series(u: np.ndarray, v: np.ndarray, lat: np.ndarray,
                lon: np.ndarray) -> dict[str, np.ndarray]:
    """Pi_ell at the band edges, band KE, boundary band-KE inflow.

    Pi formula identical to the frozen B1 `flux_and_gridded_baselines`
    (C15-1 checks the agreement numerically).
    """
    nt, ny, nx = u.shape
    dx_km, dy_km = _grid_geometry(lat, lon)
    uk, vk = u / 1000.0, v / 1000.0
    n_e = len(EDGES_KM)
    masks = [_interior_mask(ny, nx, ell, dx_km, dy_km) for ell in EDGES_KM]
    m16 = masks[-1]
    lat_step_sign = 1.0 if lat[1] > lat[0] else -1.0

    pi = np.zeros((nt, n_e))
    e_band = np.zeros((nt, n_e - 1))
    fbound = np.zeros((nt, n_e - 1))

    # bounding rectangle of the 1600-km interior mask for the line integral
    rows = np.where(m16.any(axis=1))[0]
    cols = np.where(m16.any(axis=0))[0]
    i0, i1, j0, j1 = rows[0], rows[-1], cols[0], cols[-1]
    dxm = np.asarray(dx_km) if np.ndim(dx_km) else np.full(ny, float(dx_km))
    row0_is_north = lat[0] > lat[-1]

    for t in range(nt):
        ut = np.nan_to_num(uk[t])
        vt = np.nan_to_num(vk[t])
        uu, uv, vv = ut * ut, ut * vt, vt * vt
        ubs, vbs = [], []
        for j, ell in enumerate(EDGES_KM):
            sigma_km = ell / np.sqrt(12.0)
            ub = _gaussian_bar(ut, sigma_km, dx_km, dy_km)
            vb = _gaussian_bar(vt, sigma_km, dx_km, dy_km)
            tau_xx = _gaussian_bar(uu, sigma_km, dx_km, dy_km) - ub * ub
            tau_xy = _gaussian_bar(uv, sigma_km, dx_km, dy_km) - ub * vb
            tau_yy = _gaussian_bar(vv, sigma_km, dx_km, dy_km) - vb * vb
            dudx, dudy = _gradients(ub, dx_km, dy_km)
            dvdx, dvdy = _gradients(vb, dx_km, dy_km)
            dudy *= lat_step_sign
            dvdy *= lat_step_sign
            flux_map = -(tau_xx * dudx + tau_xy * (dudy + dvdx) + tau_yy * dvdy)
            pi[t, j] = float(np.mean(flux_map[masks[j]]))
            ubs.append(ub)
            vbs.append(vb)
        for jb in range(n_e - 1):
            ub_band = ubs[jb] - ubs[jb + 1]
            vb_band = vbs[jb] - vbs[jb + 1]
            eb = 0.5 * (ub_band ** 2 + vb_band ** 2)
            e_band[t, jb] = float(np.mean(eb[m16]))
            # net inward band-KE flux through the bounding rectangle;
            # v > 0 is northward, so at the north edge inward = -v.
            sgn_top = -1.0 if row0_is_north else 1.0
            top = sgn_top * np.sum(eb[i0, j0:j1 + 1] * vb_band[i0, j0:j1 + 1]
                                   * dxm[i0])
            bot = -sgn_top * np.sum(eb[i1, j0:j1 + 1] * vb_band[i1, j0:j1 + 1]
                                    * dxm[i1])
            west = np.sum(eb[i0:i1 + 1, j0] * ub_band[i0:i1 + 1, j0] * dy_km)
            east = -np.sum(eb[i0:i1 + 1, j1] * ub_band[i0:i1 + 1, j1] * dy_km)
            fbound[t, jb] = top + bot + west + east
    return {"pi": pi, "e_band": e_band, "fbound": fbound}


def _series_one(wind: Path) -> str:
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
    res = flux_series(u, v, lat, lon)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cached, pi=res["pi"].astype(np.float64),
                        e_band=res["e_band"].astype(np.float64),
                        fbound=res["fbound"].astype(np.float64),
                        hod=hod[: res["pi"].shape[0]])
    print(f"[{tag}] series cached nt={res['pi'].shape[0]}", flush=True)
    return tag


def stage_series(workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = sorted(Path("data/b3").glob("era5_wind850_*.nc"))
    jobs += sorted(Path("data/b2b").glob("era5_wind850_*.nc"))
    print(f"{len(jobs)} region-windows to process", flush=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        pool.map(_series_one, jobs)


# --------------------------------------------------------------------------
# summaries
# --------------------------------------------------------------------------

def piband_from_pi(pi: np.ndarray) -> np.ndarray:
    """Net transfer into band [edge_j, edge_{j+1}] = Pi_coarse - Pi_fine."""
    return pi[:, 1:] - pi[:, :-1]


def _acf(x: np.ndarray, k: int) -> float:
    return float(np.corrcoef(x[:-k], x[k:])[0, 1])


def rw_summary(tag: str) -> dict:
    z = np.load(OUT / f"series_{tag}.npz")
    pi = np.asarray(z["pi"], dtype=float)
    hod = np.asarray(z["hod"], dtype=int)
    pib = piband_from_pi(pi)                      # (nt, 3) bands 2,3,4
    eb = np.asarray(z["e_band"], dtype=float)
    fb = np.asarray(z["fbound"], dtype=float)
    out: dict = {"tag": tag, "region": tag.split("__")[0],
                 "window": tag.split("__")[1]}
    Kb, K24, sd_pi, mean_pi, acf12, acf_all = [], [], [], [], [], []
    for jb in range(pib.shape[1]):
        s = deseason(pib[:, jb], hod)
        Kb.append(float(np.sqrt(np.mean(np.diff(s) ** 2))))
        K24.append(float(np.sqrt(np.mean((s[4:] - s[:-4]) ** 2))))
        sd_pi.append(float(s.std()))
        mean_pi.append(float(pib[:, jb].mean()))
        acf12.append(_acf(s, 2))
        acf_all.append([_acf(s, k) for k in (1, 2, 3, 4)])
    out.update({"K": Kb, "K24": K24, "sd_pi": sd_pi, "mean_pi": mean_pi,
                "acf12_pi": acf12, "acf_pi": acf_all})
    # placebos from band KE (composite of bands 2,3)
    ecomp = eb[:, list(PRIMARY)].mean(axis=1)
    es = deseason(np.log(ecomp + 1e-300), hod)
    out["plc_log_e"] = float(np.median(np.log(ecomp + 1e-300)))
    out["plc_dKE_rms"] = float(np.sqrt(np.mean(np.diff(es) ** 2)))
    out["plc_persist"] = _acf(es, 2)
    # boundary companion, band 3 (index 1)
    fs = deseason(fb[:, 1], hod)
    out["K_R"] = float(np.sqrt(np.mean(np.diff(fs) ** 2)))
    return out


def load_metrics(tag: str) -> dict | None:
    f = B9 / f"metrics_{tag}.json"
    if not f.exists():
        return None
    d = json.loads(f.read_text())

    def comp(key):
        v = d.get(key)
        return float(np.mean(np.asarray(v, dtype=float)[:2])) if v else np.nan
    return {"lambda": comp("lambda"), "lambda_era5": comp("lambda_era5"),
            "mu": comp("mu"), "T50": comp("T50"),
            "log_sigma": float(np.log(np.mean(
                np.asarray(d["sigma_band_era5"], dtype=float)[:2])))}


def load_P(tag: str) -> float | None:
    for base in (B3R, B2BR):
        f = base / f"{tag}.json"
        if f.exists():
            p = np.asarray(json.loads(f.read_text())["P_real"], dtype=float)
            return float(np.mean(p[[3, 4]]))
    return None


def load_slope(tag: str) -> float | None:
    for base in (B3R, B2BR):
        f = base / f"{tag}.json"
        if f.exists():
            return float(json.loads(f.read_text())["F_spec"]["slope"])
    return None


# --------------------------------------------------------------------------
# C15-1 sanity
# --------------------------------------------------------------------------

def stage_sanity() -> None:
    ref = Path("data/b3/era5_wind850_R10_INDO__W9_2023JFM.nc")
    _, lat, lon, u, v, _, _ = _load_vector_fields(
        input_path=ref, field_set="wind", u_var=None, v_var=None,
        time_stride=1, lat_stride=1, lon_stride=1, time_start=0,
        max_time=40, crop_ny=None, crop_nx=None,
    )
    mine = flux_series(u, v, lat, lon)["pi"]
    frozen = flux_and_gridded_baselines(u, v, lat, lon, EDGES_KM)["pi"]
    rel = float(np.max(np.abs(mine - frozen) / (np.abs(frozen) + 1e-300)))
    finite = True
    for f in sorted(OUT.glob("series_*.npz")):
        z = np.load(f)
        for k in ("pi", "e_band", "fbound"):
            if not np.all(np.isfinite(z[k])):
                finite = False
    res = {"max_rel_diff_vs_B1": rel, "all_series_finite": bool(finite),
           "C15_1_pass": bool(rel < 1e-6 and finite)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "sanity.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


# --------------------------------------------------------------------------
# C15-2 persistence gate
# --------------------------------------------------------------------------

def _gate_one(args_tuple) -> dict:
    kind, level, rep, seed = args_tuple
    os.environ["OMP_NUM_THREADS"] = "1"
    rng = np.random.default_rng(seed)
    ny, nx, nt = 81, 161, 60
    lat = np.arange(10.0, -10.01, -0.25)
    lon = np.arange(130.0, 170.01, 0.25)
    dxm, dym = 27.7, 27.8
    sig_coarse = 550.0 / dxm / np.sqrt(12.0)
    sig_fine = 280.0 / dxm / np.sqrt(12.0)
    if kind == "irrec":
        u, v = synthetic_pair(rng, float(level), ny, nx, nt,
                              sig_coarse, sig_fine, dxm, dym, rho_t=0.8)
    else:
        u, v = synthetic_pair(rng, 0.5, ny, nx, nt,
                              sig_coarse, sig_fine, dxm, dym,
                              rho_t=float(level))
    pib = piband_from_pi(flux_series(u, v, lat, lon)["pi"])
    hod = np.zeros(nt, dtype=int)
    ks = []
    for jb in PRIMARY:
        s = deseason(pib[:, jb], hod)
        ks.append(np.sqrt(np.mean(np.diff(s) ** 2)))
    return {"kind": kind, "level": float(level), "rep": rep,
            "K": float(np.mean(ks))}


def stage_gate(workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "gate_c15_2.json"
    if cache.exists():
        print("cached", flush=True)
        return
    ss = np.random.SeedSequence(SEED)
    jobs = []
    irr = np.linspace(0.0, 1.0, 9)
    per = np.linspace(0.5, 0.95, 6)
    seeds = iter(int(s) for s in ss.generate_state(9 * 12 + 6 * 12))
    for lv in irr:
        for rep in range(12):
            jobs.append(("irrec", lv, rep, next(seeds)))
    for lv in per:
        for rep in range(12):
            jobs.append(("persist", lv, rep, next(seeds)))
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        rows = pool.map(_gate_one, jobs)
    cache.write_text(json.dumps(rows), encoding="utf-8")
    print(f"gate rows: {len(rows)}", flush=True)


def score_gate() -> dict:
    rows = json.loads((OUT / "gate_c15_2.json").read_text())

    def rung_stats(kind):
        sel = [r for r in rows if r["kind"] == kind]
        levels = sorted({r["level"] for r in sel})
        means = [float(np.mean([r["K"] for r in sel if r["level"] == lv]))
                 for lv in levels]
        rho = float(spearmanr([r["level"] for r in sel],
                              [r["K"] for r in sel]).statistic)
        return levels, means, rho

    _, m_irr, rho_irr = rung_stats("irrec")
    _, m_per, rho_per = rung_stats("persist")
    spread_irr = float(np.std(m_irr))
    spread_per = float(np.std(m_per))
    res = {"rho_K_vs_irrecoverability": rho_irr,
           "rho_K_vs_persistence": rho_per,
           "across_rung_spread_irrec": spread_irr,
           "across_rung_spread_persist": spread_per,
           "rung_means_irrec": m_irr, "rung_means_persist": m_per,
           "C15_2_pass": bool(abs(rho_per) <= 0.5 and spread_per < spread_irr)}
    return res


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def perm_p(x: np.ndarray, y: np.ndarray, rng: np.random.Generator,
           alternative: str = "greater") -> dict:
    rho = float(spearmanr(x, y).statistic)
    null = np.empty(N_PERM)
    for i in range(N_PERM):
        null[i] = spearmanr(x, rng.permutation(y)).statistic
    if alternative == "greater":
        p = float((np.sum(null >= rho) + 1) / (N_PERM + 1))
    elif alternative == "less":
        p = float((np.sum(null <= rho) + 1) / (N_PERM + 1))
    else:
        p = float((np.sum(np.abs(null) >= abs(rho)) + 1) / (N_PERM + 1))
    return {"rho": rho, "p_perm": p, "_null": null}


def _loo_lin_residuals(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(y)
    out = np.empty(n)
    A = np.column_stack([np.ones(n), X])
    for i in range(n):
        tr = np.arange(n) != i
        coef = np.linalg.pinv(A[tr]) @ y[tr]
        out[i] = y[i] - A[i] @ coef
    return out


def _fe_design(K: np.ndarray, regions: list[str], extra: list[np.ndarray] | None = None
               ) -> tuple[np.ndarray, list[str]]:
    regs = sorted(set(regions))
    dums = np.zeros((len(K), len(regs)))
    for i, r in enumerate(regions):
        dums[i, regs.index(r)] = 1.0
    cols = [K[:, None]] + ([e[:, None] for e in extra] if extra else []) + [dums]
    return np.hstack(cols), regs


def stage_tests() -> dict:
    rng = np.random.default_rng(SEED)
    res: dict = {"seed": SEED,
                 "protocol": "docs/PROTOCOL_PHASE15_FLUX_DERIVATIVE.md"}

    sanity = json.loads((OUT / "sanity.json").read_text())
    res["C15_1"] = sanity
    if not sanity["C15_1_pass"]:
        res["PHASE15_VERDICT"] = "PIPELINE_FAULT"
        return res
    res["C15_2"] = score_gate()
    gate_ok = res["C15_2"]["C15_2_pass"]

    # ---- assemble rw table ----------------------------------------------
    rows = []
    for f in sorted(OUT.glob("series_*.npz")):
        tag = f.stem.replace("series_", "")
        met = load_metrics(tag)
        if met is None:
            continue
        s = rw_summary(tag)
        s.update(met)
        p = load_P(tag)
        sl = load_slope(tag)
        s["P"] = p if p is not None else np.nan
        s["slope"] = sl if sl is not None else np.nan
        s["Kc"] = float(np.mean([s["K"][j] for j in PRIMARY]))
        s["K24c"] = float(np.mean([s["K24"][j] for j in PRIMARY]))
        s["sd_pic"] = float(np.mean([s["sd_pi"][j] for j in PRIMARY]))
        s["mean_pic"] = float(np.mean([s["mean_pi"][j] for j in PRIMARY]))
        s["acf12c"] = float(np.mean([s["acf12_pi"][j] for j in PRIMARY]))
        rows.append(s)
    fit_rows = [r for r in rows if r["window"].split("_")[0] in WINDOWS_FIT]
    oos_rows = [r for r in rows if r["window"].split("_")[0] in WINDOWS_OOS]
    res["n_fit_rws"] = len(fit_rows)
    res["n_oos_rws"] = len(oos_rows)
    res["acf_pi_median_lags_6_12_18_24h"] = [
        float(np.median([np.mean([r["acf_pi"][j][k] for j in PRIMARY])
                         for r in fit_rows])) for k in range(4)]
    if not gate_ok:
        # Mandatory descriptive report only (protocol 3.2); no hypothesis
        # may be scored once the persistence gate has failed.
        res["PHASE15_VERDICT"] = "ESTIMATOR_INVALID"
        return res

    # ---- region aggregation (between-region arm) ------------------------
    regions = sorted({r["region"] for r in fit_rows})

    def rmed(key):
        return np.array([np.nanmedian([r[key] for r in fit_rows
                                       if r["region"] == reg]) for reg in regions])

    Kreg = rmed("Kc")
    res["region_values"] = {reg: float(k) for reg, k in zip(regions, Kreg)}

    # ---- H15a -----------------------------------------------------------
    h15a = {}
    for key, alt in (("lambda", "greater"), ("mu", "greater"), ("T50", "less")):
        t = perm_p(Kreg, rmed(key), rng, alt)
        h15a[key] = {"rho": t["rho"], "p_perm": t["p_perm"]}
        if key == "lambda":
            null_l = t["_null"]
    res["H15a"] = h15a
    res["H15a_pass_lambda"] = bool(h15a["lambda"]["rho"] > 0
                                   and h15a["lambda"]["p_perm"] < 0.05)
    res["H15a_pass_mu"] = bool(h15a["mu"]["rho"] > 0
                               and h15a["mu"]["p_perm"] < 0.05)

    # ---- C15-3 null calibration -----------------------------------------
    ranks = np.array([(np.sum(null_l >= v)) / (N_PERM - 1) for v in null_l])
    res["C15_3"] = {"ks_p": float(kstest(ranks, "uniform").pvalue)}
    res["C15_3_pass"] = bool(res["C15_3"]["ks_p"] > 0.05)

    # ---- H15b beyond controls -------------------------------------------
    try:
        from clean_experiments.experiment_B4_curvature_invariant import region_covariates
    except ImportError:  # pragma: no cover
        from experiment_B4_curvature_invariant import region_covariates  # type: ignore
    cov = region_covariates(Path("data/b4cov"))
    lat_c = np.array([abs((REGIONS_ALL[r][0] + REGIONS_ALL[r][2]) / 2.0)
                      for r in regions])
    land = np.array([cov[r].get("land_frac", cov[r].get("land", np.nan))
                     for r in regions])
    X = np.column_stack([lat_c, land, rmed("log_sigma"), rmed("slope"),
                         rmed("acf12c"), rmed("mean_pic"), rmed("sd_pic")])
    rK = _loo_lin_residuals(X, Kreg)
    rL = _loo_lin_residuals(X, rmed("lambda"))
    t = perm_p(rK, rL, rng, "greater")
    res["H15b"] = {"rho": t["rho"], "p_perm": t["p_perm"],
                   "controls": ["abs_lat", "land", "log_sigma", "slope",
                                "acf12_pi", "mean_pi", "sd_pi"]}
    res["H15b_pass"] = bool(t["rho"] > 0 and t["p_perm"] < 0.05)

    # ---- C15-4 placebo ladder -------------------------------------------
    try:
        from clean_experiments.experiment_B11_learned_refinement import descriptors
    except ImportError:  # pragma: no cover
        from experiment_B11_learned_refinement import descriptors  # type: ignore
    desc = descriptors()
    lam = rmed("lambda")
    plc = {
        "i_log_band_KE": rmed("plc_log_e"),
        "ii_dKE_rms": rmed("plc_dKE_rms"),
        "iii_persistence_KE": rmed("plc_persist"),
        "iv_A": np.array([desc.get(r, {}).get("A", np.nan) for r in regions]),
        "v_P": rmed("P"),
        "vi_sd_pi": rmed("sd_pic"),
    }
    rho_K = abs(h15a["lambda"]["rho"])
    c154 = {}
    for name, vals in plc.items():
        m = np.isfinite(vals)
        c154[name] = float(spearmanr(vals[m], lam[m]).statistic)
    res["C15_4"] = {"rho_K_abs": rho_K,
                    "placebo_rhos": c154,
                    "beats_all": bool(all(rho_K > abs(v) for v in c154.values())),
                    "decisive_ok": bool(rho_K > abs(c154["ii_dKE_rms"])
                                        and rho_K > abs(c154["iii_persistence_KE"])
                                        and rho_K > abs(c154["vi_sd_pi"]))}

    # ---- H15c modulation ------------------------------------------------
    rows48 = [r for r in fit_rows if np.isfinite(r["P"])]
    K48 = np.array([r["Kc"] for r in rows48])
    P48 = np.array([r["P"] for r in rows48])
    L48 = np.array([r["lambda"] for r in rows48])
    reg48 = [r["region"] for r in rows48]
    Kz = (K48 - K48.mean()) / (K48.std() + 1e-12)
    Pz = (P48 - P48.mean()) / (P48.std() + 1e-12)

    def loo_r2(extra: list[np.ndarray]) -> float:
        sse, sst = 0.0, 0.0
        for hold in sorted(set(reg48)):
            tr = np.array([g != hold for g in reg48])
            te = ~tr
            Xtr, regs = _fe_design(Kz[tr], [g for g, m in zip(reg48, tr) if m],
                                   [e[tr] for e in extra])
            coef = np.linalg.pinv(Xtr) @ L48[tr]
            # held-out region has no FE estimate: use mean intercept
            n_fe = len(regs)
            fe_mean = float(np.mean(coef[-n_fe:]))
            Xte_cols = [Kz[te][:, None]] + [e[te][:, None] for e in extra]
            pred = np.hstack(Xte_cols) @ coef[:-n_fe] + fe_mean
            sse += float(np.sum((L48[te] - pred) ** 2))
            sst += float(np.sum((L48[te] - L48[tr].mean()) ** 2))
        return 1.0 - sse / sst

    r2_k = loo_r2([])
    r2_full = loo_r2([Pz, Kz * Pz])
    gain = r2_full - r2_k
    null_gain = np.empty(N_PERM)
    regs_sorted = sorted(set(reg48))
    for i in range(N_PERM):
        pm = {r: p for r, p in zip(regs_sorted,
                                   rng.permutation(rmed("P")))}
        Pp = np.array([pm[g] for g in reg48])
        Pp = (Pp - Pp.mean()) / (Pp.std() + 1e-12)
        null_gain[i] = loo_r2([Pp, Kz * Pp]) - r2_k
    res["H15c"] = {"r2_K_only": r2_k, "r2_with_KP": r2_full, "gain": gain,
                   "p_perm": float((np.sum(null_gain >= gain) + 1) / (N_PERM + 1))}
    res["H15c_pass"] = bool(gain > 0 and res["H15c"]["p_perm"] < 0.05)

    # ---- H15d within-region ---------------------------------------------
    Xfe, regs = _fe_design(Kz, reg48)
    coef = np.linalg.pinv(Xfe) @ L48
    slope = float(coef[0])
    null_s = np.empty(N_PERM)
    for i in range(N_PERM):
        Kp = Kz.copy()
        for reg in regs:
            m = np.array([g == reg for g in reg48])
            Kp[m] = rng.permutation(Kp[m])
        null_s[i] = (np.linalg.pinv(_fe_design(Kp, reg48)[0]) @ L48)[0]
    res["H15d"] = {"slope": slope,
                   "p_perm": float((np.sum(null_s >= slope) + 1) / (N_PERM + 1))}
    res["H15d_pass"] = bool(slope > 0 and res["H15d"]["p_perm"] < 0.05)

    # ---- H15e out-of-sample ---------------------------------------------
    fe = {reg: float(c) for reg, c in zip(regs, coef[1:])}
    mu_k, sd_k = K48.mean(), K48.std() + 1e-12
    oos = [r for r in oos_rows if r["region"] in fe]
    pred = np.array([fe[r["region"]] + slope * (r["Kc"] - mu_k) / sd_k
                     for r in oos])
    obs = np.array([r["lambda"] for r in oos])
    t = perm_p(pred, obs, rng, "greater")
    r2_oos = 1.0 - float(np.sum((obs - pred) ** 2)) / float(
        np.sum((obs - L48.mean()) ** 2))
    res["H15e"] = {"n_oos": len(oos), "rho": t["rho"], "p_perm": t["p_perm"],
                   "r2_oos": r2_oos}
    res["H15e_pass"] = bool(t["rho"] > 0 and t["p_perm"] < 0.05 and r2_oos > 0)

    # ---- H15f boundary companion ----------------------------------------
    KR = rmed("K_R")
    t = perm_p(KR, lam, rng, "greater")
    res["H15f"] = {"between_region": {"rho": t["rho"], "p_perm": t["p_perm"]}}

    # ---- C15-5 verification independence --------------------------------
    rho_l = h15a["lambda"]["rho"]
    rho_le = float(spearmanr(Kreg, rmed("lambda_era5")).statistic)
    rho_m = h15a["mu"]["rho"]
    res["C15_5"] = {"rho_lambda": rho_l, "rho_lambda_era5": rho_le,
                    "rho_mu": rho_m,
                    "sign_agree": bool(np.sign(rho_l) == np.sign(rho_le)
                                       == np.sign(rho_m))}

    # ---- verdict ---------------------------------------------------------
    confirmed = (res["H15a_pass_lambda"] and res["H15a_pass_mu"]
                 and res["H15b_pass"] and res["H15d_pass"] and res["H15e_pass"]
                 and res["C15_4"]["beats_all"] and res["C15_5"]["sign_agree"])
    negative = ((not res["H15a_pass_lambda"] and not res["H15a_pass_mu"])
                or not res["C15_5"]["sign_agree"]
                or not res["C15_4"]["decisive_ok"])
    partial = ((res["H15a_pass_lambda"] or res["H15a_pass_mu"])
               and res["H15b_pass"])
    if confirmed:
        res["PHASE15_VERDICT"] = "CONFIRMED"
    elif negative:
        res["PHASE15_VERDICT"] = "NEGATIVE"
    elif partial:
        res["PHASE15_VERDICT"] = "PARTIAL"
    else:
        res["PHASE15_VERDICT"] = "NEGATIVE"
    return res


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["series", "sanity", "gate", "tests"],
                    default="series")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    if args.stage == "series":
        stage_series(args.workers)
    elif args.stage == "sanity":
        stage_sanity()
    elif args.stage == "gate":
        stage_gate(args.workers)
    else:
        res = stage_tests()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "summary.json").write_text(json.dumps(res, indent=2),
                                          encoding="utf-8")
        brief = {k: res.get(k) for k in
                 ("C15_2", "n_fit_rws", "n_oos_rws",
                  "acf_pi_median_lags_6_12_18_24h", "H15a", "H15b", "H15c",
                  "H15d", "H15e", "C15_4", "C15_5", "PHASE15_VERDICT")}
        print(json.dumps(brief, indent=2, default=str))


if __name__ == "__main__":
    main()
