#!/usr/bin/env python3
"""Phase-18: equal-km regionalization as a DESIGN control for the box-geometry
confound (protocol frozen 2026-08-17 as "Phase 16" in the draft numbering;
renumbered 18 because 16/17 were executed as ENSO / long-record phases).

Protocol: docs/PROTOCOL_PHASE18_EQUAL_KM_REGIONS.md (deviations logged there).

Regions keep their frozen centres; boundaries are chosen so the great-circle
width at the mean latitude and the meridional height equal the nominal km
extents (Variant B: index masks on the native 0.25 deg grid, no reprojection).
Primary sample: the 48 region-windows of Phase 3 (data/b3, 2023-2024).
Held-out: R5-R12 x W5-W8 (data/b2b, 2021-2022), consulted only in H18d.

Descriptors, frozen Phase-2/3/4 machinery unchanged:
  - P: envelope_rho_profile; primary form = anchored excess
    P_real - median(P over 199 phase surrogates regenerated on the km window).
  - A: curvature_profiles Fnorm, resolved bands b=2,3 (200-800 km).
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
from scipy.stats import kstest, spearmanr

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import clean_experiments.download_b1_era5_wind as dl_b1
    from clean_experiments.download_b2_era5_wind import REGIONS_B2
    from clean_experiments.download_b2b_era5_wind import REGIONS_B2B
    from clean_experiments.experiment_B1_true_flux_baselines import (
        _grid_geometry,
        _interior_mask,
        phase_randomize,
    )
    from clean_experiments.experiment_B2_scale_irreversibility import (
        ELLS_KM,
        EPS,
        compute_vorticity,
        gaussian_bar_grouped,
        spectral_features,
    )
    from clean_experiments.experiment_B2b_heldout_invariants import (
        _loo_residuals,
        _pairwise_dist,
        _within_between,
        _zscore_cols,
    )
    from clean_experiments.experiment_B4_curvature_invariant import (
        curvature_profiles,
    )
    from clean_experiments.experiment_scale_gravity_einstein_box_era import (
        _load_vector_fields,
    )
except ImportError:
    import download_b1_era5_wind as dl_b1  # type: ignore
    from download_b2_era5_wind import REGIONS_B2  # type: ignore
    from download_b2b_era5_wind import REGIONS_B2B  # type: ignore
    from experiment_B1_true_flux_baselines import (  # type: ignore
        _grid_geometry,
        _interior_mask,
        phase_randomize,
    )
    from experiment_B2_scale_irreversibility import (  # type: ignore
        ELLS_KM,
        EPS,
        compute_vorticity,
        gaussian_bar_grouped,
        spectral_features,
    )
    from experiment_B2b_heldout_invariants import (  # type: ignore
        _loo_residuals,
        _pairwise_dist,
        _within_between,
        _zscore_cols,
    )
    from experiment_B4_curvature_invariant import (  # type: ignore
        curvature_profiles,
    )
    from experiment_scale_gravity_einstein_box_era import (  # type: ignore
        _load_vector_fields,
    )

SEED_PERM = 20260817          # frozen in the protocol
SEED_SUR = 20260811           # surrogate streams inherit the Phase-2/3 scheme
N_SUR_PRIMARY = 199
N_SUR_AUX = 49                # 0.9x/1.1x and dynamic-scale arms (not scored)

EARTH_R_KM = 6371.0
KM_PER_DEG = EARTH_R_KM * np.pi / 180.0
NOMINAL_KM = (2000.0, 2800.0)          # (meridional, zonal); deviation: 4000 -> 2800
EXTENT_FACTORS = (1.0, 0.9, 1.1)

OMEGA_E = 7.2921e-5
C_GRAVE = 25.0                          # m/s, first-baroclinic proxy speed
S_CLIP = (0.5, 2.0)

REGIONS_ALL = {**dl_b1.REGIONS, **REGIONS_B2, **REGIONS_B2B}
REGIONS = list(REGIONS_ALL.keys())      # R1..R12 in frozen order
PRIMARY_WINDOWS = ["W9_2023JFM", "W10_2023JAS", "W11_2024JFM", "W12_2024JAS"]
HELDOUT_REGIONS = ["R5_SPCZ", "R6_SATL", "R7_CONGO", "R8_AUS",
                   "R9_NPAC", "R10_INDO", "R11_EURO", "R12_SAM"]
HELDOUT_WINDOWS = ["W5_2021JFM", "W6_2021JAS", "W7_2022JFM", "W8_2022JAS"]

A_BANDS = (2, 3)                        # resolved bands, 200-800 km
N_PERM = 999


# ---------------------------------------------------------------------------
# Equal-km region table
# ---------------------------------------------------------------------------

def region_center(region: str) -> tuple[float, float]:
    n, w, s, e = REGIONS_ALL[region]
    return 0.5 * (n + s), 0.5 * (w + e)

def km_half_extents_deg(lat_c: float, factor: float) -> tuple[float, float]:
    h_km, w_km = NOMINAL_KM[0] * factor, NOMINAL_KM[1] * factor
    half_lat = 0.5 * h_km / KM_PER_DEG
    half_lon = 0.5 * w_km / (KM_PER_DEG * np.cos(np.deg2rad(lat_c)))
    return half_lat, half_lon

def km_crop(lat: np.ndarray, lon: np.ndarray, u: np.ndarray, v: np.ndarray,
            region: str, factor: float):
    lat_c, lon_c = region_center(region)
    hla, hlo = km_half_extents_deg(lat_c, factor)
    iy = np.where((lat >= lat_c - hla - 1e-9) & (lat <= lat_c + hla + 1e-9))[0]
    ix = np.where((lon >= lon_c - hlo - 1e-9) & (lon <= lon_c + hlo + 1e-9))[0]
    if len(iy) < 8 or len(ix) < 8:
        raise ValueError(f"km box for {region} x{factor} does not fit the file")
    return lat[iy], lon[ix], u[:, iy][:, :, ix], v[:, iy][:, :, ix]

def deformation_scale_factors() -> dict[str, float]:
    ld = {}
    for r in REGIONS:
        lat_c, _ = region_center(r)
        phi = np.deg2rad(lat_c)
        f = 2.0 * OMEGA_E * abs(np.sin(phi))
        beta = 2.0 * OMEGA_E * np.cos(phi) / (EARTH_R_KM * 1000.0)
        ld_eq = np.sqrt(C_GRAVE / beta)
        ld[r] = min(C_GRAVE / f if f > 0 else np.inf, ld_eq) / 1000.0  # km
    ref = float(np.median(list(ld.values())))
    return {r: float(np.clip(ld[r] / ref, *S_CLIP)) for r in REGIONS}, ld, ref  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# P ladder with parameterized band edges (frozen Phase-2 form; the ells become
# an argument only so the dynamic-scale arm can scale them)
# ---------------------------------------------------------------------------

def envelope_rho_profile_ells(omega, dx_km, dy_km, mask, ells_km) -> np.ndarray:
    n_steps = len(ells_km) - 1
    w = np.nan_to_num(omega).astype(np.float32)
    bars = [gaussian_bar_grouped(w, ell / np.sqrt(12.0), dx_km, dy_km) for ell in ells_km]
    envs = [np.log(gaussian_bar_grouped(np.abs(w - bars[0]), ells_km[0] / np.sqrt(12.0),
                                        dx_km, dy_km) + EPS)]
    for i in range(n_steps):
        d = bars[i] - bars[i + 1]
        envs.append(np.log(gaussian_bar_grouped(np.abs(d), ells_km[i + 1] / np.sqrt(12.0),
                                                dx_km, dy_km) + EPS))
    del bars
    ranked = []
    for e in envs:
        m = e[:, mask]
        ranks = np.argsort(np.argsort(m, axis=1), axis=1).astype(np.float32)
        ranked.append(ranks)
    del envs
    rho = np.zeros((w.shape[0], n_steps))
    for i in range(n_steps):
        a = ranked[i] - ranked[i].mean(axis=1, keepdims=True)
        b = ranked[i + 1] - ranked[i + 1].mean(axis=1, keepdims=True)
        denom = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
        denom = np.where(denom < 1e-12, 1.0, denom)
        rho[:, i] = (a * b).sum(axis=1) / denom
    return rho


# ---------------------------------------------------------------------------
# Surrogate workers (P only; entropy pipeline not needed for anchoring)
# ---------------------------------------------------------------------------

_G: dict[str, object] = {}

def _worker_init(u, v, lat, lon, dx_km, dy_km, mask, ells) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    _G.update(u=u, v=v, lat=lat, lon=lon, dx=dx_km, dy=dy_km, mask=mask, ells=ells)

def _worker(seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    us, vs = phase_randomize(_G["u"], _G["v"], rng)  # type: ignore[arg-type]
    om = compute_vorticity(us, vs, _G["lat"], _G["lon"])  # type: ignore[arg-type]
    rho = envelope_rho_profile_ells(om, _G["dx"], _G["dy"], _G["mask"], _G["ells"])  # type: ignore[arg-type]
    return [float(x) for x in np.median(rho, axis=0)]

def surrogate_p_median(u, v, lat, lon, dx_km, dy_km, mask, ells, tag: str,
                       n_sur: int, workers: int, salt: str = "") -> np.ndarray:
    tag_code = zlib.crc32((tag + salt).encode()) % 100000
    seeds = [int(s) for s in np.random.SeedSequence(SEED_SUR + tag_code).generate_state(n_sur)]
    if workers > 1:
        ctx = mp.get_context("fork")
        with ctx.Pool(workers, _worker_init, (u, v, lat, lon, dx_km, dy_km, mask, ells)) as pool:
            sur = pool.map(_worker, seeds)
    else:
        _worker_init(u, v, lat, lon, dx_km, dy_km, mask, ells)
        sur = [_worker(s) for s in seeds]
    return np.median(np.asarray(sur), axis=0)


# ---------------------------------------------------------------------------
# Per region-window computation
# ---------------------------------------------------------------------------

def compute_arm(u, v, lat, lon, tag: str, ells, n_sur: int, workers: int,
                salt: str = "", with_curvature: bool = True,
                curv_scale: float = 1.0) -> dict:
    dx_km, dy_km = _grid_geometry(lat, lon)
    mask = _interior_mask(u.shape[1], u.shape[2], ells[-1], dx_km, dy_km)
    cov = float(np.mean(np.isfinite(u) & np.isfinite(v)))
    omega = compute_vorticity(u, v, lat, lon)
    rho = envelope_rho_profile_ells(omega, dx_km, dy_km, mask, ells)
    p_real = np.median(rho, axis=0)
    sur_med = surrogate_p_median(u, v, lat, lon, dx_km, dy_km, mask, ells,
                                 tag, n_sur, workers, salt)
    out = {
        "P_real": [float(x) for x in p_real],
        "P_surrogate_median": [float(x) for x in sur_med],
        "P_anchored": [float(x) for x in (p_real - sur_med)],
        "coverage": cov,
        "n_cells": [int(u.shape[1]), int(u.shape[2])],
        "n_surrogates": n_sur,
    }
    try:
        out["F_spec"] = spectral_features(omega, dx_km, dy_km, mask)
    except Exception as exc:  # dynamic arm may push bands outside the fit range
        out["F_spec_error"] = str(exc)
    if with_curvature:
        # curvature_profiles reads SCALE_EDGES_KM from its module; the
        # dynamic-scale arm scales those edges via a scoped override.
        b4mod = sys.modules[curvature_profiles.__module__]
        frozen_edges = list(b4mod.SCALE_EDGES_KM)
        try:
            if curv_scale != 1.0:
                b4mod.SCALE_EDGES_KM = [e * curv_scale for e in frozen_edges]
            out.update(curvature_profiles(u, v, lat, lon))
            out["A"] = float(np.mean([out["Fnorm_profile"][b] for b in A_BANDS]))
        except Exception as exc:
            out["curvature_error"] = str(exc)
        finally:
            b4mod.SCALE_EDGES_KM = frozen_edges
    return out


def run_region_window(nc_path: Path, out_dir: Path, workers: int,
                      primary: bool, s_factors: dict[str, float]) -> dict:
    tag = nc_path.stem.replace("era5_wind850_", "")
    cached = out_dir / f"{tag}.json"
    if cached.exists():
        r = json.loads(cached.read_text(encoding="utf-8"))
        print(f"[{tag}] cached", flush=True)
        return r
    region = tag.split("__")[0]
    print(f"[{tag}] loading", flush=True)
    _, lat, lon, u, v, _, _ = _load_vector_fields(
        input_path=nc_path, field_set="wind", u_var=None, v_var=None,
        time_stride=1, lat_stride=1, lon_stride=1, time_start=0,
        max_time=None, crop_ny=None, crop_nx=None,
    )
    result: dict = {"tag": tag, "region": region, "window": tag.split("__")[1]}

    la, lo, uu, vv = km_crop(lat, lon, u, v, region, 1.0)
    print(f"[{tag}] km 1.0x grid {uu.shape[1]}x{uu.shape[2]}", flush=True)
    result["km_1.0"] = compute_arm(uu, vv, la, lo, tag, ELLS_KM,
                                   N_SUR_PRIMARY, workers)

    if primary:
        for f in EXTENT_FACTORS[1:]:
            la, lo, uu, vv = km_crop(lat, lon, u, v, region, f)
            key = f"km_{f:.1f}"
            print(f"[{tag}] {key} grid {uu.shape[1]}x{uu.shape[2]}", flush=True)
            result[key] = compute_arm(uu, vv, la, lo, tag, ELLS_KM,
                                      N_SUR_AUX, workers, salt=f"_{f:.1f}")
        s = s_factors[region]
        ells_dyn = [e * s for e in ELLS_KM]
        la, lo, uu, vv = km_crop(lat, lon, u, v, region, 1.0)
        print(f"[{tag}] dynamic s={s:.2f}", flush=True)
        result["km_dyn"] = compute_arm(uu, vv, la, lo, tag, ells_dyn,
                                       N_SUR_AUX, workers, salt="_dyn",
                                       curv_scale=s)
        result["dyn_scale_factor"] = s

    out_dir.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[{tag}] done", flush=True)
    return result


# ---------------------------------------------------------------------------
# Permutation machinery (B2b math, perm distribution returned for C18-2)
# ---------------------------------------------------------------------------

def perm_test_full(x_z: np.ndarray, labels: np.ndarray, rng) -> dict:
    dist = _pairwise_dist(x_z)
    w_obs, b_obs = _within_between(dist, labels)
    diff_obs = b_obs - w_obs
    perm = []
    for _ in range(N_PERM):
        labs = labels[rng.permutation(len(labels))]
        w_p, b_p = _within_between(dist, labs)
        perm.append(b_p - w_p)
    perm = np.asarray(perm)
    p = float((1 + np.sum(perm >= diff_obs)) / (N_PERM + 1))
    return {"median_within": w_obs, "median_between": b_obs, "diff": diff_obs,
            "p": p, "pass": bool(diff_obs > 0 and p < 0.05),
            "perm": perm}


def regional_median(values: dict[str, list[float]], regions: list[str]) -> np.ndarray:
    return np.asarray([np.median(values[r], axis=0) for r in regions])


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------

def consolidate(results: list[dict], heldout: list[dict], out_dir: Path,
                b3_dir: Path, b4_dir: Path) -> dict:
    rng = np.random.default_rng(SEED_PERM)
    labels = np.asarray([r["region"] for r in results])
    tags = [r["tag"] for r in results]

    deg = {}
    for t in tags:
        f = b3_dir / f"{t}.json"
        deg[t] = json.loads(f.read_text(encoding="utf-8"))

    anch_km = np.asarray([r["km_1.0"]["P_anchored"] for r in results])
    raw_km = np.asarray([r["km_1.0"]["P_real"] for r in results])
    anch_deg = np.asarray([np.asarray(deg[t]["P_real"]) -
                           np.asarray(deg[t]["P_surrogate_median"]) for t in tags])
    raw_deg = np.asarray([deg[t]["P_real"] for t in tags])

    fnorm_km = np.asarray([r["km_1.0"]["Fnorm_profile"] for r in results])
    a_km = np.asarray([r["km_1.0"]["A"] for r in results])
    a_deg = np.asarray([float(np.mean([deg[t]["E2_curvature"]["Fnorm_profile"][b]
                                       for b in A_BANDS])) for t in tags])

    spec_keys = sorted(results[0]["km_1.0"]["F_spec"].keys())
    spec_km = np.asarray([[r["km_1.0"]["F_spec"][k] for k in spec_keys] for r in results])
    spec_deg = np.asarray([[deg[t]["F_spec"][k] for k in spec_keys] for t in tags])

    # ---------------- C18-1 pipeline sanity ----------------
    coverage = np.asarray([r["km_1.0"]["coverage"] for r in results] +
                          [r["km_1.0"]["coverage"] for r in heldout])
    cov_ok = bool(np.all(coverage >= 0.95))
    logvar_cols = [k for k in spec_keys if k.startswith("logvar")]
    lv_idx = [spec_keys.index(k) for k in logvar_cols]
    sigma_logratio = spec_km[:, lv_idx] - spec_deg[:, lv_idx]
    c18_1 = {
        "coverage_min": float(coverage.min()),
        "coverage_ok": cov_ok,
        "sigma_b_logratio_median": [float(x) for x in np.median(sigma_logratio, axis=0)],
        "sigma_b_logratio_maxabs": [float(x) for x in np.max(np.abs(sigma_logratio), axis=0)],
        "pass": cov_ok,
    }
    if not cov_ok:
        summary = {"C18_1": c18_1, "VERDICT": "ABORT_PIPELINE_FAULT"}
        (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    # ---------------- H18a negative control (A) ----------------
    reg_a_km, reg_a_deg = {}, {}
    for t, r in zip(tags, results):
        reg = r["region"]
        reg_a_km.setdefault(reg, []).append(float(r["km_1.0"]["A"]))
        reg_a_deg.setdefault(reg, []).append(
            float(np.mean([deg[t]["E2_curvature"]["Fnorm_profile"][b] for b in A_BANDS])))
    a_km_12 = np.asarray([np.median(reg_a_km[r]) for r in REGIONS])
    a_deg_12 = np.asarray([np.median(reg_a_deg[r]) for r in REGIONS])
    rho_a = float(spearmanr(a_deg_12, a_km_12).statistic)
    sig_a_km = perm_test_full(_zscore_cols(fnorm_km), labels, rng)
    h18a = {
        "rho_regional_A_deg_vs_km": rho_a,
        "A_km_signature": {k: v for k, v in sig_a_km.items() if k != "perm"},
        "pass": bool(rho_a >= 0.9 and sig_a_km["pass"]),
        "regional_A_deg": {r: float(x) for r, x in zip(REGIONS, a_deg_12)},
        "regional_A_km": {r: float(x) for r, x in zip(REGIONS, a_km_12)},
    }

    # ---------------- H18b primary (anchored P clustering, km) ----------------
    h18b_full = perm_test_full(_zscore_cols(anch_km), labels, rng)
    h18b = {k: v for k, v in h18b_full.items() if k != "perm"}
    # comparable degree-box number on the same 48 windows
    rng_cmp = np.random.default_rng(SEED_PERM + 1)
    h18b_deg = {k: v for k, v in perm_test_full(
        _zscore_cols(anch_deg), labels, rng_cmp).items() if k != "perm"}
    # logged secondary: beyond-spectrum residual clustering (B2b H2 form)
    rng_res = np.random.default_rng(SEED_PERM + 2)
    res_km = _loo_residuals(anch_km, spec_km)
    beyond_km = {k: v for k, v in perm_test_full(
        _zscore_cols(res_km), labels, rng_res).items() if k != "perm"}
    rng_res2 = np.random.default_rng(SEED_PERM + 3)
    res_deg = _loo_residuals(anch_deg, spec_deg)
    beyond_deg = {k: v for k, v in perm_test_full(
        _zscore_cols(res_deg), labels, rng_res2).items() if k != "perm"}
    rng_res3 = np.random.default_rng(SEED_PERM + 4)
    res_km_raw = _loo_residuals(raw_km, spec_km)
    beyond_km_raw = {k: v for k, v in perm_test_full(
        _zscore_cols(res_km_raw), labels, rng_res3).items() if k != "perm"}

    # ---------------- H18c geography preservation ----------------
    per_region_rho = {}
    reg_anch_km, reg_anch_deg = {}, {}
    for t, r in zip(tags, results):
        reg = r["region"]
        reg_anch_km.setdefault(reg, []).append(r["km_1.0"]["P_anchored"])
        reg_anch_deg.setdefault(reg, []).append(
            (np.asarray(deg[t]["P_real"]) - np.asarray(deg[t]["P_surrogate_median"])).tolist())
    prof_km_12 = regional_median(reg_anch_km, REGIONS)
    prof_deg_12 = regional_median(reg_anch_deg, REGIONS)
    for i, reg in enumerate(REGIONS):
        per_region_rho[reg] = float(spearmanr(prof_deg_12[i], prof_km_12[i]).statistic)
    pooled_rho = float(spearmanr(prof_deg_12.ravel(), prof_km_12.ravel()).statistic)
    h18c = {"per_region_rho": per_region_rho, "pooled_rho": pooled_rho}

    # ---------------- H18d held-out ----------------
    ho_labels = np.asarray([r["region"] for r in heldout])
    ho_anch = np.asarray([r["km_1.0"]["P_anchored"] for r in heldout])
    rng_ho = np.random.default_rng(SEED_PERM + 5)
    h18d_full = perm_test_full(_zscore_cols(ho_anch), ho_labels, rng_ho)
    h18d = {k: v for k, v in h18d_full.items() if k != "perm"}
    h18d["sign_agrees"] = bool(np.sign(h18d["diff"]) == np.sign(h18b["diff"]))
    h18d["pass"] = bool(h18d["sign_agrees"] and h18d["p"] < 0.05 and h18d["diff"] > 0)

    # ---------------- H18e Congo-Amazon rho_5 contrast ----------------
    def contrast(mat_by_reg, band=-1):
        c = float(np.median([v[band] for v in mat_by_reg["R7_CONGO"]]))
        a = float(np.median([v[band] for v in mat_by_reg["R3_AMAZ"]]))
        return c, a, c - a

    reg_raw_km, reg_raw_deg = {}, {}
    for t, r in zip(tags, results):
        reg = r["region"]
        reg_raw_km.setdefault(reg, []).append(r["km_1.0"]["P_real"])
        reg_raw_deg.setdefault(reg, []).append(deg[t]["P_real"])
    c_km, a_km_v, d_km = contrast(reg_anch_km)
    c_dg, a_dg, d_dg = contrast(reg_anch_deg)
    rc_km, ra_km, rd_km = contrast(reg_raw_km)
    rc_dg, ra_dg, rd_dg = contrast(reg_raw_deg)
    h18e = {
        "anchored": {"congo_km": c_km, "amaz_km": a_km_v, "contrast_km": d_km,
                     "congo_deg": c_dg, "amaz_deg": a_dg, "contrast_deg": d_dg},
        "raw": {"congo_km": rc_km, "amaz_km": ra_km, "contrast_km": rd_km,
                "congo_deg": rc_dg, "amaz_deg": ra_dg, "contrast_deg": rd_dg},
        "pass": bool(np.sign(d_km) == np.sign(d_dg) and abs(d_km) >= 0.5 * abs(d_dg)),
    }

    # ---------------- C18-2 null calibration ----------------
    perm = h18b_full["perm"]
    ranks = np.array([(1 + np.sum(perm >= x)) / (len(perm) + 1) for x in perm])
    ks = kstest(ranks, "uniform")
    c18_2 = {"ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
             "pass": bool(ks.pvalue > 0.05)}

    # ---------------- C18-3 spectral placebo ----------------
    rng_pl = np.random.default_rng(SEED_PERM + 6)
    placebo = {k: v for k, v in perm_test_full(
        _zscore_cols(spec_km), labels, rng_pl).items() if k != "perm"}
    c18_3 = {"spec_clustering": placebo,
             "placebo_as_strong_as_P": bool(placebo["diff"] >= h18b["diff"])}

    # ---------------- C18-4 extent sensitivity ----------------
    c18_4 = {}
    for f in EXTENT_FACTORS[1:]:
        key = f"km_{f:.1f}"
        per_reg = {}
        for reg in REGIONS:
            base, alt = [], []
            for r in results:
                if r["region"] != reg or key not in r:
                    continue
                base.extend(list(r["km_1.0"]["P_anchored"]) + [r["km_1.0"]["A"]])
                alt.extend(list(r[key]["P_anchored"]) + [r[key]["A"]])
            per_reg[reg] = float(spearmanr(base, alt).statistic) if base else None
        vals = [v for v in per_reg.values() if v is not None]
        c18_4[key] = {"per_region_rho": per_reg,
                      "min_rho": float(min(vals)) if vals else None,
                      "all_ge_0.9": bool(all(v >= 0.9 for v in vals)) if vals else None}

    # ---------------- dynamic-scale arm (descriptive) ----------------
    dyn_anch = np.asarray([r["km_dyn"]["P_anchored"] for r in results if "km_dyn" in r])
    dyn_labels = np.asarray([r["region"] for r in results if "km_dyn" in r])
    rng_dyn = np.random.default_rng(SEED_PERM + 7)
    dyn_sig = {k: v for k, v in perm_test_full(
        _zscore_cols(dyn_anch), dyn_labels, rng_dyn).items() if k != "perm"}
    dyn_a = [r["km_dyn"].get("A") for r in results if "km_dyn" in r]
    dyn = {"anchored_P_clustering": dyn_sig,
           "scale_factors": {r: results[[x["region"] for x in results].index(r)]
                             .get("dyn_scale_factor") for r in REGIONS},
           "A_available": int(sum(1 for x in dyn_a if x is not None))}

    # ---------------- verdict ----------------
    if not h18a["pass"]:
        verdict = "UNSTABLE_MEASUREMENT"
    elif h18b["pass"] and h18d["pass"] and h18e["pass"]:
        verdict = "CONFIRMED_PHYSICAL"
    else:
        verdict = "CONFIRMED_ARTEFACT"

    summary = {
        "n_primary": len(results), "n_heldout": len(heldout),
        "C18_1_pipeline_sanity": c18_1,
        "H18a_negative_control_A": h18a,
        "H18b_anchored_P_km": h18b,
        "H18b_anchored_P_deg_same_windows": h18b_deg,
        "H18b_secondary_beyond_spectrum_km_anchored": beyond_km,
        "H18b_secondary_beyond_spectrum_deg_anchored": beyond_deg,
        "H18b_secondary_beyond_spectrum_km_raw": beyond_km_raw,
        "H18c_geography": h18c,
        "H18d_heldout": h18d,
        "H18e_congo_amazon": h18e,
        "C18_2_null_calibration": c18_2,
        "C18_3_spectral_placebo": c18_3,
        "C18_4_extent_sensitivity": c18_4,
        "P18_3_dynamic_scale_arm": dyn,
        "VERDICT": verdict,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.save(out_dir / "h18b_perm.npy", perm)
    np.save(out_dir / "h18d_perm.npy", h18d_full["perm"])
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b3-data", type=Path, default=Path("data/b3"))
    parser.add_argument("--b2b-data", type=Path, default=Path("data/b2b"))
    parser.add_argument("--out-dir", type=Path,
                        default=_HERE / "results" / "experiment_B18_equal_km_regions")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 3))
    parser.add_argument("--only", type=str, default=None)
    parser.add_argument("--skip-consolidate", action="store_true")
    args = parser.parse_args()

    s_factors, ld_km, ld_ref = deformation_scale_factors()
    (args.out_dir / "region_table.json").parent.mkdir(parents=True, exist_ok=True)
    table = {}
    for r in REGIONS:
        lat_c, lon_c = region_center(r)
        hla, hlo = km_half_extents_deg(lat_c, 1.0)
        table[r] = {"center": [lat_c, lon_c], "deg_box_NWSE": REGIONS_ALL[r],
                    "km_half_lat_deg": hla, "km_half_lon_deg": hlo,
                    "Ld_km": ld_km[r], "dyn_scale": s_factors[r]}
    table["_Ld_ref_km"] = ld_ref
    table["_nominal_km"] = list(NOMINAL_KM)
    (args.out_dir / "region_table.json").write_text(json.dumps(table, indent=2),
                                                    encoding="utf-8")

    primary_files = [args.b3_data / f"era5_wind850_{r}__{w}.nc"
                     for r in REGIONS for w in PRIMARY_WINDOWS]
    heldout_files = [args.b2b_data / f"era5_wind850_{r}__{w}.nc"
                     for r in HELDOUT_REGIONS for w in HELDOUT_WINDOWS]
    if args.only:
        primary_files = [f for f in primary_files if args.only in f.name]
        heldout_files = [f for f in heldout_files if args.only in f.name]
    missing = [f for f in primary_files + heldout_files if not f.exists()]
    if missing:
        raise SystemExit(f"Missing inputs: {missing[:5]} ({len(missing)} total)")

    results = [run_region_window(f, args.out_dir, args.workers, True, s_factors)
               for f in primary_files]
    heldout = [run_region_window(f, args.out_dir, args.workers, False, s_factors)
               for f in heldout_files]

    if not args.skip_consolidate and len(results) == 48 and len(heldout) == 32:
        summary = consolidate(results, heldout, args.out_dir,
                              _HERE / "results" / "experiment_B3_scattering_benchmark",
                              _HERE / "results" / "experiment_B4_curvature_invariant")
        print(json.dumps({k: v for k, v in summary.items()
                          if not k.startswith("H18b_secondary")}, indent=2, default=str))


if __name__ == "__main__":
    main()
