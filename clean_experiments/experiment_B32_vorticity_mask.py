#!/usr/bin/env python3
"""Phase 32, checks A1-A2: spherical curl term and the 850 hPa below-surface mask.

Protocol: docs/PROTOCOL_PHASE32_REVIEW_CHECKS.md (frozen 2026-10-08).

Variants of the profile pipeline (Phase 2 conventions otherwise unchanged):
  V0   archived: omega = dv/dx - du/dy (local Cartesian, per-row dx), interior mask
  V1   spherical: omega_s = omega + u tan(phi) / a
  V2a  mask-exclude: V0 field, correlation restricted to interior & ~below_surface
  V2b  mask-fill: below-surface points set to NaN, every Gaussian smoothing done as
       a normalised convolution (smooth(f*valid)/smooth(valid)), then as V2a
  V12  spherical + mask-fill (candidate definition for the revised manuscript)

Stages:
  era5        48 ERA5 windows 2023-2024 (data/b3): all variants, no surrogates
  merra       48 MERRA-2 windows (data/b8merra2): V0 (NaN->0, archived), V2a, V2b
  surrogates  199 phase surrogates for V0 and V12 on selected regions (anchored P)
  consolidate clustering / geography statistics against the archived numbers
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter1d
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
for p in (str(_HERE), str(_HERE.parent)):
    if p not in sys.path:
        sys.path.insert(0, p)

import experiment_B2_scale_irreversibility as b2  # noqa: E402
from experiment_B1_true_flux_baselines import _grid_geometry, _interior_mask, phase_randomize  # noqa: E402
from experiment_B32_review_checks import delta, loo_residuals, pdist, perm_test, zscore  # noqa: E402
from experiment_scale_gravity_einstein_box_era import _load_vector_fields  # noqa: E402

OUT = _HERE / "results" / "experiment_B32_review_checks"
ORO = _HERE.parent / "data" / "b32" / "era5_orography.nc"
B3 = _HERE.parent / "data" / "b3"
MERRA = _HERE.parent / "data" / "b8merra2"
SEED = 20261008
EARTH_M = 6.371e6
Z_850_M = 1457.0                      # ICAO standard atmosphere height of 850 hPa
ELLS = b2.ELLS_KM
MERRA_ELLS = [200.0, 400.0, 800.0, 1600.0]
EPS = b2.EPS
REGIONS = ["R1_WPWP", "R2_NATL", "R3_AMAZ", "R4_CASIA", "R5_SPCZ", "R6_SATL", "R7_CONGO",
           "R8_AUS", "R9_NPAC", "R10_INDO", "R11_EURO", "R12_SAM"]
WINDOWS = ["W9_2023JFM", "W10_2023JAS", "W11_2024JFM", "W12_2024JAS"]
SUR_REGIONS = ["R2_NATL", "R6_SATL", "R12_SAM", "R1_WPWP"]


# ---------------------------------------------------------------------------
# fields
# ---------------------------------------------------------------------------

def below_surface_mask(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """True where the ERA5 surface height exceeds the 850 hPa standard height."""
    ds = xr.open_dataset(ORO)
    z = ds["z"].squeeze()
    sel = z.sel(latitude=xr.DataArray(lat, dims="y"),
                longitude=xr.DataArray(np.mod(lon, 360.0), dims="x"), method="nearest")
    ds.close()
    return (np.asarray(sel.values, float) / 9.80665) > Z_850_M


def vorticity(u: np.ndarray, v: np.ndarray, lat: np.ndarray, lon: np.ndarray, spherical: bool) -> np.ndarray:
    om = b2.compute_vorticity(u, v, lat, lon)
    if spherical:
        term = (np.nan_to_num(u) * np.tan(np.deg2rad(lat))[None, :, None] / EARTH_M).astype(np.float32)
        om = om + term
    return om


def bar_nan(field: np.ndarray, sigma_km: float, dx_km: np.ndarray, dy_km: float,
            valid: np.ndarray | None) -> np.ndarray:
    """Gaussian smoothing; with `valid` (2-D bool) a normalised convolution."""
    if valid is None:
        return b2.gaussian_bar_grouped(field, sigma_km, dx_km, dy_km)
    w = valid.astype(np.float32)
    num = b2.gaussian_bar_grouped(np.nan_to_num(field) * w[None, :, :], sigma_km, dx_km, dy_km)
    den = b2.gaussian_bar_grouped(w, sigma_km, dx_km, dy_km)
    den = np.where(den < 1e-6, np.nan, den)
    return num / den[None, :, :]


def rho_profile(omega: np.ndarray, dx_km: np.ndarray, dy_km: float, corr_mask: np.ndarray,
                valid: np.ndarray | None, ells: list[float]) -> np.ndarray:
    """envelope_rho_profile of Phase 2 with optional NaN-aware smoothing."""
    w = omega.astype(np.float32)
    if valid is not None:
        w = np.where(valid[None, :, :], w, np.nan)
    bars = [bar_nan(w, ell / np.sqrt(12.0), dx_km, dy_km, valid) for ell in ells]
    envs = [np.log(bar_nan(np.abs(np.nan_to_num(w) - np.nan_to_num(bars[0])), ells[0] / np.sqrt(12.0), dx_km, dy_km, valid) + EPS)]
    for i in range(len(ells) - 1):
        d = bars[i] - bars[i + 1]
        envs.append(np.log(bar_nan(np.abs(np.nan_to_num(d)), ells[i + 1] / np.sqrt(12.0), dx_km, dy_km, valid) + EPS))
    del bars
    ranked = []
    for e in envs:
        m = e[:, corr_mask]
        ranked.append(np.argsort(np.argsort(m, axis=1), axis=1).astype(np.float32))
    del envs
    n_steps = len(ells) - 1
    rho = np.zeros((w.shape[0], n_steps))
    for i in range(n_steps):
        a = ranked[i] - ranked[i].mean(axis=1, keepdims=True)
        b = ranked[i + 1] - ranked[i + 1].mean(axis=1, keepdims=True)
        den = np.sqrt((a * a).sum(axis=1) * (b * b).sum(axis=1))
        rho[:, i] = (a * b).sum(axis=1) / np.where(den < 1e-12, 1.0, den)
    return rho


def variants(u, v, lat, lon, ells, interior, below, which=("V0", "V1", "V2a", "V2b", "V12")) -> dict:
    dx_km, dy_km = _grid_geometry(lat, lon)
    corr_excl = interior & ~below
    om_c = vorticity(u, v, lat, lon, spherical=False)
    om_s = vorticity(u, v, lat, lon, spherical=True)
    out = {}
    if "V0" in which:
        out["V0"] = np.median(rho_profile(om_c, dx_km, dy_km, interior, None, ells), axis=0)
    if "V1" in which:
        out["V1"] = np.median(rho_profile(om_s, dx_km, dy_km, interior, None, ells), axis=0)
    if "V2a" in which:
        out["V2a"] = np.median(rho_profile(om_c, dx_km, dy_km, corr_excl, None, ells), axis=0)
    if "V2b" in which:
        out["V2b"] = np.median(rho_profile(om_c, dx_km, dy_km, corr_excl, ~below, ells), axis=0)
    if "V12" in which:
        out["V12"] = np.median(rho_profile(om_s, dx_km, dy_km, corr_excl, ~below, ells), axis=0)
    return {k: [float(x) for x in val] for k, val in out.items()}


# ---------------------------------------------------------------------------
# stage: era5
# ---------------------------------------------------------------------------

def load_era5(path: Path):
    _, lat, lon, u, v, _, _ = _load_vector_fields(
        input_path=path, field_set="wind", u_var=None, v_var=None,
        time_stride=1, lat_stride=1, lon_stride=1, time_start=0,
        max_time=None, crop_ny=None, crop_nx=None)
    return lat, lon, u, v


def stage_era5(out_dir: Path, regions: list[str]) -> None:
    for reg in regions:
        for win in WINDOWS:
            tag = f"{reg}__{win}"
            target = out_dir / "era5" / f"{tag}.json"
            if target.exists():
                continue
            t0 = time.time()
            lat, lon, u, v = load_era5(B3 / f"era5_wind850_{tag}.nc")
            dx_km, dy_km = _grid_geometry(lat, lon)
            interior = _interior_mask(u.shape[1], u.shape[2], ELLS[-1], dx_km, dy_km)
            below = below_surface_mask(lat, lon)
            res = variants(u, v, lat, lon, ELLS, interior, below)
            res.update({"tag": tag, "region": reg, "window": win,
                        "below_frac_box": float(below.mean()),
                        "below_frac_interior": float(below[interior].mean()),
                        "mean_abs_u_ms": float(np.nanmean(np.abs(u))),
                        "lat_centre": float(np.mean(lat))})
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(res, indent=2), encoding="utf-8")
            print(f"[{tag}] era5 variants done in {time.time() - t0:.0f}s  below={res['below_frac_interior']:.3f}", flush=True)


# ---------------------------------------------------------------------------
# stage: merra
# ---------------------------------------------------------------------------

def stage_merra(out_dir: Path) -> None:
    for reg in REGIONS:
        for win in WINDOWS:
            tag = f"{reg}__{win}"
            target = out_dir / "merra" / f"{tag}.json"
            if target.exists():
                continue
            t0 = time.time()
            ds = xr.open_dataset(MERRA / f"merra2_wind850_{tag}.nc")
            tdim = next(d for d in ds["u"].dims if "time" in d)
            lat = np.asarray(ds["lat"].values, float)
            lon = np.asarray(ds["lon"].values, float)
            u = np.asarray(ds["u"].transpose(tdim, "lat", "lon").values, np.float32)
            v = np.asarray(ds["v"].transpose(tdim, "lat", "lon").values, np.float32)
            ds.close()
            nan_frac_t = np.isnan(u).mean(axis=(1, 2))
            below = np.isnan(u).mean(axis=0) > 0.5          # static mask: below surface most of the time
            dx_km, dy_km = _grid_geometry(lat, lon)
            interior = _interior_mask(u.shape[1], u.shape[2], MERRA_ELLS[-1], dx_km, dy_km)
            u0, v0 = np.nan_to_num(u), np.nan_to_num(v)      # archived treatment
            res = variants(u0, v0, lat, lon, MERRA_ELLS, interior, below, which=("V0", "V2a", "V2b", "V12"))
            res.update({"tag": tag, "region": reg, "window": win,
                        "nan_frac_mean": float(nan_frac_t.mean()),
                        "below_frac_interior": float(below[interior].mean())})
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(res, indent=2), encoding="utf-8")
            print(f"[{tag}] merra variants done in {time.time() - t0:.0f}s  nan={res['nan_frac_mean']:.3f}", flush=True)


# ---------------------------------------------------------------------------
# stage: surrogates
# ---------------------------------------------------------------------------

_G: dict = {}


def _sur_init(u, v, lat, lon, interior, below) -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    _G.update(u=u, v=v, lat=lat, lon=lon, interior=interior, below=below)


def _sur_worker(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    us, vs = phase_randomize(_G["u"], _G["v"], rng)
    return variants(us, vs, _G["lat"], _G["lon"], ELLS, _G["interior"], _G["below"], which=("V0", "V12"))


def stage_surrogates(out_dir: Path, regions: list[str], n_sur: int, workers: int) -> None:
    for reg in regions:
        for win in WINDOWS:
            tag = f"{reg}__{win}"
            target = out_dir / "surrogates" / f"{tag}.json"
            if target.exists():
                continue
            t0 = time.time()
            lat, lon, u, v = load_era5(B3 / f"era5_wind850_{tag}.nc")
            dx_km, dy_km = _grid_geometry(lat, lon)
            interior = _interior_mask(u.shape[1], u.shape[2], ELLS[-1], dx_km, dy_km)
            below = below_surface_mask(lat, lon)
            real = variants(u, v, lat, lon, ELLS, interior, below, which=("V0", "V12"))
            tag_code = zlib.crc32(tag.encode()) % 100000
            seeds = [int(s) for s in np.random.SeedSequence(SEED + tag_code).generate_state(n_sur)]
            ctx = mp.get_context("fork")
            with ctx.Pool(workers, _sur_init, (u, v, lat, lon, interior, below)) as pool:
                sur = pool.map(_sur_worker, seeds)
            res = {"tag": tag, "region": reg, "window": win, "n_surrogates": n_sur}
            for key in ("V0", "V12"):
                s = np.asarray([x[key] for x in sur])
                res[key] = {"P_real": real[key], "P_surrogate_median": [float(x) for x in np.median(s, axis=0)],
                            "P_anchored": [float(x) for x in (np.asarray(real[key]) - np.median(s, axis=0))]}
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps(res, indent=2), encoding="utf-8")
            print(f"[{tag}] surrogates done in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# stage: consolidate
# ---------------------------------------------------------------------------

def _archived_b3() -> dict[str, dict]:
    d = {}
    for f in (_HERE / "results" / "experiment_B3_scattering_benchmark").glob("R*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        d[r["tag"]] = r
    return d


def _archived_b18() -> dict[str, dict]:
    d = {}
    for f in (_HERE / "results" / "experiment_B18_equal_km_regions").glob("R*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        d[r["tag"]] = r["km_1.0"]
    return d


def _archived_merra() -> dict[str, dict]:
    d = {}
    for f in (_HERE / "results" / "experiment_B8_cross_reanalysis").glob("R*.json"):
        r = json.loads(f.read_text(encoding="utf-8"))
        d[r["tag"]] = r
    return d


def stage_consolidate(out_dir: Path) -> dict:
    rng = np.random.default_rng(SEED)
    res: dict = {}
    arch = _archived_b3()
    era = {}
    for f in sorted((out_dir / "era5").glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        era[r["tag"]] = r
    if era:
        tags = sorted(era)
        labels = np.asarray([era[t]["region"] for t in tags])
        p_arch = np.asarray([arch[t]["P_real"] for t in tags])
        p_v0 = np.asarray([era[t]["V0"] for t in tags])
        a1: dict = {"n_windows": len(tags),
                    "V0_reproduces_archived_max_abs_diff": float(np.abs(p_v0 - p_arch).max())}
        per_region: dict = {}
        for key in ("V1", "V2a", "V2b", "V12"):
            pk = np.asarray([era[t][key] for t in tags])
            diff = pk - p_v0
            sp = [float(spearmanr(pk[i], p_v0[i]).statistic) for i in range(len(tags))]
            by_reg = {}
            for reg in REGIONS:
                m = labels == reg
                by_reg[reg] = {"max_abs_diff_steps2_5": round(float(np.abs(diff[m][:, 1:]).max()), 4),
                               "max_abs_diff_step1": round(float(np.abs(diff[m][:, 0]).max()), 4),
                               "mean_diff_by_step": [round(float(x), 4) for x in diff[m].mean(axis=0)]}
            a1[key] = {"max_abs_diff_all_steps": round(float(np.abs(diff).max()), 4),
                       "max_abs_diff_steps2_5": round(float(np.abs(diff[:, 1:]).max()), 4),
                       "min_spearman_profile": round(min(sp), 4),
                       "mean_diff_by_step": [round(float(x), 4) for x in diff.mean(axis=0)],
                       "by_region": by_reg,
                       "clustering_48_full": perm_test(pk, labels, rng),
                       "clustering_48_steps2_5": perm_test(pk[:, 1:], labels, rng),
                       "clustering_48_steps4_5": perm_test(pk[:, 3:], labels, rng)}
            per_region[key] = by_reg
        a1["clustering_48_archived_V0_full"] = perm_test(p_v0, labels, rng)
        a1["clustering_48_archived_V0_steps2_5"] = perm_test(p_v0[:, 1:], labels, rng)
        a1["below_surface_fraction_interior_by_region"] = {
            reg: round(float(np.mean([era[t]["below_frac_interior"] for t in tags if era[t]["region"] == reg])), 4)
            for reg in REGIONS}
        a1["H32-1_pass"] = bool(a1["V1"]["min_spearman_profile"] >= 0.99 and a1["V1"]["max_abs_diff_all_steps"] <= 0.01
                                and abs(a1["V1"]["clustering_48_full"]["diff"] - a1["clustering_48_archived_V0_full"]["diff"]) <= 0.05)
        a1["H32-2_pass"] = bool(all(a1[k]["max_abs_diff_steps2_5"] <= 0.03 for k in ("V2a", "V2b"))
                                and all(abs(a1[k]["clustering_48_full"]["diff"] - a1["clustering_48_archived_V0_full"]["diff"]) <= 0.1
                                        for k in ("V2a", "V2b")))
        res["A1_A2_era5"] = a1

    sur = {}
    for f in sorted((out_dir / "surrogates").glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        sur[r["tag"]] = r
    if sur:
        b18 = _archived_b18()
        tags = sorted(sur)
        a_v0 = np.asarray([sur[t]["V0"]["P_anchored"] for t in tags])
        a_v12 = np.asarray([sur[t]["V12"]["P_anchored"] for t in tags])
        a_arch = np.asarray([arch[t]["P_real"] for t in tags]) - np.asarray([arch[t]["P_surrogate_median"] for t in tags])
        res["A1_A2_surrogates"] = {
            "n_windows": len(tags), "regions": sorted({sur[t]["region"] for t in tags}),
            "V0_anchored_vs_archived_degree_max_abs_diff": round(float(np.abs(a_v0 - a_arch).max()), 4),
            "V12_minus_V0_anchored_max_abs_diff": round(float(np.abs(a_v12 - a_v0).max()), 4),
            "V12_minus_V0_anchored_max_abs_diff_steps2_5": round(float(np.abs(a_v12 - a_v0)[:, 1:].max()), 4),
            "V12_minus_V0_anchored_mean_by_step": [round(float(x), 4) for x in (a_v12 - a_v0).mean(axis=0)],
            "per_window": {t: {"V0": sur[t]["V0"]["P_anchored"], "V12": sur[t]["V12"]["P_anchored"]} for t in tags},
            "H32-1_surrogate_pass": bool(np.abs(a_v12 - a_v0).max() <= 0.02),
            "note": "B18 km-box anchored profiles are not comparable (different box geometry); "
                    "degree-box archived anchored profile is P_real - P_surrogate_median of Phase 3.",
        }
        _ = b18

    mer = {}
    for f in sorted((out_dir / "merra").glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        mer[r["tag"]] = r
    if mer and era:
        amer = _archived_merra()
        tags = sorted(mer)
        labels = np.asarray([mer[t]["region"] for t in tags])
        m_arch = np.asarray([amer[t]["P_merra"] for t in tags])
        m_v0 = np.asarray([mer[t]["V0"] for t in tags])
        a2m: dict = {"n_windows": len(tags),
                     "V0_reproduces_archived_max_abs_diff": round(float(np.abs(m_v0 - m_arch).max()), 4),
                     "nan_fraction_by_region": {reg: round(float(np.mean([mer[t]["nan_frac_mean"] for t in tags if mer[t]["region"] == reg])), 4)
                                                for reg in REGIONS}}
        # ERA5 side: region medians of mean(steps 4-5) for V0 and V12
        def era_geo(key):
            per = {}
            for t, r in era.items():
                per.setdefault(r["region"], []).append(np.mean(r[key][3:5]))
            return np.asarray([np.median(per[reg]) for reg in REGIONS])
        for key in ("V0", "V2a", "V2b", "V12"):
            if key not in mer[tags[0]]:
                continue
            mk = np.asarray([mer[t][key] for t in tags])
            per = {}
            for t, row in zip(tags, mk):
                per.setdefault(mer[t]["region"], []).append(np.mean(row[1:3]))
            m_geo = np.asarray([np.median(per[reg]) for reg in REGIONS])
            block = {"signature_48": perm_test(mk, labels, rng),
                     "max_abs_diff_vs_V0": round(float(np.abs(mk - m_v0).max()), 4),
                     "by_region_max_abs_diff_vs_V0": {reg: round(float(np.abs(mk - m_v0)[labels == reg].max()), 4) for reg in REGIONS}}
            for ekey in ("V0", "V12"):
                e_geo = era_geo(ekey)
                block[f"geography_rho_vs_era5_{ekey}"] = round(float(spearmanr(e_geo, m_geo).statistic), 4)
            block["merra_region_median_steps2_3"] = {reg: round(float(m_geo[i]), 4) for i, reg in enumerate(REGIONS)}
            a2m[key] = block
        a2m["era5_region_median_steps4_5_V0"] = {reg: round(float(x), 4) for reg, x in zip(REGIONS, era_geo("V0"))}
        a2m["era5_region_median_steps4_5_V12"] = {reg: round(float(x), 4) for reg, x in zip(REGIONS, era_geo("V12"))}
        a2m["H32-2m_pass"] = bool(a2m["V2b"]["geography_rho_vs_era5_V12"] >= 0.8
                                  and a2m["V2b"]["signature_48"]["diff"] > 0.8 and a2m["V2b"]["signature_48"]["p"] <= 0.01)
        res["A2_merra2"] = a2m

    (out_dir / "summary_A1_A2.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if not isinstance(vv, dict) or len(json.dumps(vv)) < 400}
                      for k, v in res.items()}, indent=1, ensure_ascii=False)[:6000])
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["era5", "merra", "surrogates", "consolidate"], required=True)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--regions", type=str, default=None, help="comma-separated region codes")
    parser.add_argument("--n-surrogates", type=int, default=199)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    regions = args.regions.split(",") if args.regions else None
    if args.stage == "era5":
        stage_era5(args.out, regions or REGIONS)
    elif args.stage == "merra":
        stage_merra(args.out)
    elif args.stage == "surrogates":
        stage_surrogates(args.out, regions or SUR_REGIONS, args.n_surrogates, args.workers)
    else:
        stage_consolidate(args.out)


if __name__ == "__main__":
    main()
