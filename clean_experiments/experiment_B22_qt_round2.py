#!/usr/bin/env python3
"""Phase-22: theory-candidate tests, round 2 (QT-P2; QT-P1 seasonal).

Preregistered protocol: docs/PROTOCOL_PHASE22_QT_ROUND2.md
(frozen 2026-08-18; beta_T calibration amendment logged pre-computation).

Stages: --stage model-tiles | era5-05-tiles | p1s-series | tests
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
from scipy.stats import spearmanr, theilslopes

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from clean_experiments.experiment_B18_equal_km_regions import (
        REGIONS,
        envelope_rho_profile_ells,
        km_crop,
    )
    from clean_experiments.experiment_B1_true_flux_baselines import (
        _grid_geometry,
        _interior_mask,
        phase_randomize,
    )
    from clean_experiments.experiment_B20_armB_global_map import (
        OROG_EXCL_M,
        SEASONS,
        tile_covariates,
        tile_grid,
    )
    from clean_experiments.experiment_B20_p_geography import (
        COVARS,
        ELLS_FINE,
        N_FINE,
        N_SUR,
        RUNMEAN_STEPS,
        TILE_COLS,
        TILE_ROWS,
        fine_features,
        static_covariates,
        tile_slices,
    )
    from clean_experiments.experiment_B2_scale_irreversibility import (
        compute_vorticity,
    )
except ImportError:
    from experiment_B18_equal_km_regions import (  # type: ignore
        REGIONS,
        envelope_rho_profile_ells,
        km_crop,
    )
    from experiment_B1_true_flux_baselines import (  # type: ignore
        _grid_geometry,
        _interior_mask,
        phase_randomize,
    )
    from experiment_B20_armB_global_map import (  # type: ignore
        OROG_EXCL_M,
        SEASONS,
        tile_covariates,
        tile_grid,
    )
    from experiment_B20_p_geography import (  # type: ignore
        COVARS,
        ELLS_FINE,
        N_FINE,
        N_SUR,
        RUNMEAN_STEPS,
        TILE_COLS,
        TILE_ROWS,
        fine_features,
        static_covariates,
        tile_slices,
    )
    from experiment_B2_scale_irreversibility import (  # type: ignore
        compute_vorticity,
    )

SEED = 20260818
SEED_SUR = 20260811
OUT = _HERE / "results" / "experiment_B22_qt_round2"
ARMB_RES = _HERE / "results" / "experiment_B20_armB_global_map"
ARMA_RES = _HERE / "results" / "experiment_B20_p_geography"
MODEL_DIR = Path("data/b13hrmip/ECMWF-IFS-HR")
GLOB = Path("data/b20global")
B17_DATA = Path("data/b17daily")
B21_DATA = Path("data/b21")

MODEL_MONTHS = {"JFM": ["201401", "201402", "201403"],
                "JAS": ["201407", "201408", "201409"]}
SEASON_MONTHS = {"JFM": (1, 2, 3), "JAS": (7, 8, 9)}
MIN_SEASON_DAYS = 60
N_PERM = 999
N_BOOT = 9999
RHO_ABS_BAR = 0.5
RHO_CEIL_FRAC = 0.6
RHO_KILL = 0.3


# --------------------------------------------------------------------------
# generic global tile worker (arbitrary regular grid)
# --------------------------------------------------------------------------

_G: dict[str, object] = {}


def _tile_worker(t: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    from scipy.ndimage import uniform_filter1d
    u, v = _G["u"], _G["v"]
    lat, lon = _G["lat"], _G["lon"]
    salt = _G["salt"]
    dlon_g = float(lon[1] - lon[0])
    iy = np.where((lat >= t["lat0"] - 1e-9) & (lat <= t["lat1"] + 1e-9))[0]
    nx = len(lon)
    i0 = int(np.searchsorted(lon, t["lon0"] - 1e-9))
    n_cols = max(4, int(round((t["lon1"] - t["lon0"]) / dlon_g)))
    ix = (np.arange(i0, i0 + n_cols)) % nx
    la = lat[iy]
    lo = t["lon0"] + dlon_g * np.arange(n_cols)
    uu = u[:, iy][:, :, ix].astype(np.float32)
    vv = v[:, iy][:, :, ix].astype(np.float32)
    dx_km, dy_km = _grid_geometry(la, lo)
    mask = _interior_mask(uu.shape[1], uu.shape[2], ELLS_FINE[-1],
                          dx_km, dy_km)
    omega = compute_vorticity(uu, vv, la, lo)
    rho = envelope_rho_profile_ells(omega, dx_km, dy_km, mask, ELLS_FINE)
    p_real = np.median(rho, axis=0)
    tcode = zlib.crc32(f"{salt}_{t['tid']}".encode()) % 100000
    seeds = np.random.SeedSequence(SEED_SUR + tcode).generate_state(N_SUR)
    sur = np.empty((N_SUR, N_FINE))
    for j, s in enumerate(seeds):
        rng = np.random.default_rng(int(s))
        us, vs = phase_randomize(uu, vv, rng)
        om_s = compute_vorticity(us, vs, la, lo)
        sur[j] = np.median(envelope_rho_profile_ells(
            om_s, dx_km, dy_km, mask, ELLS_FINE), axis=0)
    up = uu - uniform_filter1d(uu, RUNMEAN_STEPS, axis=0, mode="nearest")
    vp = vv - uniform_filter1d(vv, RUNMEAN_STEPS, axis=0, mode="nearest")
    eke = float(np.mean(0.5 * (np.var(up, axis=0) + np.var(vp, axis=0))[mask]))
    return {"tid": t["tid"],
            "P_fine_anch_mean": float(np.mean(p_real - np.median(sur, axis=0))),
            "eke_syn": eke}


def _run_tiles(u, v, lat, lon, salt: str, cached: Path, workers: int) -> None:
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        u = u[:, ::-1, :]
        v = v[:, ::-1, :]
    _G.update(u=np.ascontiguousarray(u), v=np.ascontiguousarray(v),
              lat=lat, lon=lon, salt=salt)
    tiles = tile_grid()
    ctx = mp.get_context("fork")
    results = []
    with ctx.Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_tile_worker, tiles,
                                                  chunksize=4)):
            results.append(r)
            if (i + 1) % 200 == 0:
                print(f"[{salt}: {i+1}/{len(tiles)}]", flush=True)
    cached.write_text(json.dumps({"salt": salt, "tiles": results}, indent=1),
                      encoding="utf-8")
    print(f"{salt}: {len(results)} tiles done", flush=True)


def stage_model_tiles(workers: int) -> None:
    import xarray as xr
    OUT.mkdir(parents=True, exist_ok=True)
    for season, months in MODEL_MONTHS.items():
        cached = OUT / f"tiles_MODEL_{season}.json"
        if cached.exists():
            print("cached", season, flush=True)
            continue
        parts_u, parts_v = [], []
        for ym in months:
            fu = next(MODEL_DIR.glob(f"ua_*_{ym}01*-*.nc"))
            fv = next(MODEL_DIR.glob(f"va_*_{ym}01*-*.nc"))
            du = xr.open_dataset(fu)["ua"].sel(plev=85000.0)
            dv = xr.open_dataset(fv)["va"].sel(plev=85000.0)
            parts_u.append(du)
            parts_v.append(dv)
        ua = xr.concat(parts_u, dim="time")
        va = xr.concat(parts_v, dim="time")
        lat = np.asarray(ua["lat"].values, dtype=float)
        lon = np.asarray(ua["lon"].values, dtype=float)
        u = np.asarray(ua.values, dtype=np.float32)
        v = np.asarray(va.values, dtype=np.float32)
        print(f"MODEL {season}: cube {u.shape}", flush=True)
        _run_tiles(u, v, lat, lon, f"model_{season}", cached, workers)


def stage_era5_05_tiles(workers: int) -> None:
    import xarray as xr
    OUT.mkdir(parents=True, exist_ok=True)
    for season, months in SEASONS.items():
        cached = OUT / f"tiles_E05_{season}.json"
        if cached.exists():
            print("cached", season, flush=True)
            continue
        parts = [xr.open_dataset(GLOB / f"era5_wind850_global_{ym}.nc")
                 for ym in months]
        ds = xr.concat(parts, dim="valid_time")
        lat = np.asarray(ds["latitude"].values, dtype=float)[::2]
        lon = np.asarray(ds["longitude"].values, dtype=float)[::2]
        u = np.asarray(ds["u"].squeeze().values, dtype=np.float32)[:, ::2, ::2]
        v = np.asarray(ds["v"].squeeze().values, dtype=np.float32)[:, ::2, ::2]
        for p in parts:
            p.close()
        print(f"E05 {season}: cube {u.shape}", flush=True)
        _run_tiles(u, v, lat, lon, f"e05_{season}", cached, workers)


# --------------------------------------------------------------------------
# Arm P1s series
# --------------------------------------------------------------------------

def _p1s_one(path: Path) -> str:
    os.environ["OMP_NUM_THREADS"] = "1"
    import xarray as xr
    tag = path.stem.replace("era5_wind850daily_", "")
    cached = OUT / f"p1s_{tag}.json"
    if cached.exists():
        return tag
    region, year = tag.rsplit("_", 1)
    ds = xr.open_dataset(path)
    tname = "valid_time" if "valid_time" in ds else "time"
    lat = np.asarray(ds["latitude"].values, dtype=float)
    lon = np.asarray(ds["longitude"].values, dtype=float)
    u = np.asarray(ds["u"].squeeze().values, dtype=np.float32)
    v = np.asarray(ds["v"].squeeze().values, dtype=np.float32)
    months = np.asarray(ds[tname].dt.month.values, dtype=int)
    ds.close()
    if u.shape[0] < 360:
        return f"SHORT {tag}"
    lat, lon, u, v = km_crop(lat, lon, u, v, region, 1.0)
    rows = tile_slices(u.shape[1], TILE_ROWS)
    cols = tile_slices(u.shape[2], TILE_COLS)
    out_tiles: dict = {}
    for ri, rs in enumerate(rows):
        for ci, cs in enumerate(cols):
            la, lo = lat[rs], lon[cs]
            uu, vv = u[:, rs, cs], v[:, rs, cs]
            dx_km, dy_km = _grid_geometry(la, lo)
            mask = _interior_mask(uu.shape[1], uu.shape[2], ELLS_FINE[-1],
                                  dx_km, dy_km)
            om = compute_vorticity(uu, vv, la, lo)
            rho = envelope_rho_profile_ells(om, dx_km, dy_km, mask, ELLS_FINE)
            comp = rho.mean(axis=1)
            entry = {"lat_bounds": [float(la.min()), float(la.max())],
                     "lon_bounds": [float(lo.min()), float(lo.max())]}
            for sname, mos in SEASON_MONTHS.items():
                sel = np.isin(months[: len(comp)], mos)
                if sel.sum() >= MIN_SEASON_DAYS:
                    entry[sname] = {
                        "P_fine_raw": float(np.median(comp[sel])),
                        "F_fine": fine_features(om[sel], dx_km, dy_km, mask)}
            out_tiles[f"r{ri}c{ci}"] = entry
    OUT.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps({"tag": tag, "region": region,
                                  "year": int(year), "tiles": out_tiles},
                                 indent=1), encoding="utf-8")
    return f"DONE {tag}"


def stage_p1s(workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(B17_DATA.glob("era5_wind850daily_*.nc"))
    todo = [f for f in files
            if not (OUT / f"p1s_{f.stem.replace('era5_wind850daily_', '')}.json"
                    ).exists()]
    print(f"{len(files)} files, {len(todo)} to process", flush=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for i, msg in enumerate(pool.imap_unordered(_p1s_one, todo)):
            if not msg.startswith("DONE") or (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(todo)}] {msg}", flush=True)


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def _map_from(fname_fmt: str) -> dict[str, float]:
    per: dict[str, list] = {}
    for season in ("JFM", "JAS"):
        d = json.loads((OUT / fname_fmt.format(season)).read_text(
            encoding="utf-8"))
        for r in d["tiles"]:
            per.setdefault(r["tid"], []).append(r["P_fine_anch_mean"])
    return {tid: float(np.mean(v)) for tid, v in per.items() if len(v) == 2}


def tests_p2() -> dict:
    grid = tile_grid()
    by_tid = {t["tid"]: t for t in grid}
    cov = tile_covariates(grid)
    keep = {t for t in cov if cov[t]["orog_mean"] <= OROG_EXCL_M}

    model = _map_from("tiles_MODEL_{}.json")
    e05 = _map_from("tiles_E05_{}.json")
    e025: dict[str, list] = {}
    for season in ("JFM", "JAS"):
        d = json.loads((ARMB_RES / f"tiles_{season}2023.json").read_text(
            encoding="utf-8"))
        for r in d["tiles"]:
            e025.setdefault(r["tid"], []).append(r["P_fine_anch_mean"])
    e025 = {tid: float(np.mean(v)) for tid, v in e025.items() if len(v) == 2}

    tids = sorted(keep & set(model) & set(e05) & set(e025))
    m = np.asarray([model[t] for t in tids])
    a5 = np.asarray([e05[t] for t in tids])
    a25 = np.asarray([e025[t] for t in tids])
    rho_ceil = float(spearmanr(a25, a5).statistic)
    rho_mod = float(spearmanr(m, a5).statistic)
    rho_mod_025 = float(spearmanr(m, a25).statistic)

    if rho_mod >= RHO_ABS_BAR and rho_mod >= RHO_CEIL_FRAC * rho_ceil:
        v = "P2_SUPPORTED"
    elif rho_mod < RHO_KILL:
        v = "P2_FALSIFIED"
    else:
        v = "P2_PARTIAL"

    # descriptive: loadings on the model map
    keys = tids
    eke_model: dict[str, list] = {}
    for season in ("JFM", "JAS"):
        d = json.loads((OUT / f"tiles_MODEL_{season}.json").read_text(
            encoding="utf-8"))
        for r in d["tiles"]:
            eke_model.setdefault(r["tid"], []).append(r["eke_syn"])
    loads = {}
    for name in ("orog_std", "land_frac", "abs_lat", "sst_grad", "cape_mean"):
        x = np.asarray([cov[t][name] for t in keys])
        loads[name] = float(spearmanr(x, m).statistic)
    loads["eke_syn_model"] = float(spearmanr(
        [np.mean(eke_model[t]) for t in keys], m).statistic)

    return {"n_tiles": len(tids), "rho_ceiling_e025_vs_e05": rho_ceil,
            "rho_model_vs_e05": rho_mod, "rho_model_vs_e025": rho_mod_025,
            "bars": {"abs": RHO_ABS_BAR, "ceil_frac": RHO_CEIL_FRAC,
                     "kill": RHO_KILL},
            "model_map_loadings": loads, "VERDICT_P2": v}


def tests_p1s(rng: np.random.Generator) -> dict:
    import xarray as xr
    series: dict[tuple, dict] = {}
    for f in sorted(OUT.glob("p1s_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for tid, t in d["tiles"].items():
            for sname in SEASON_MONTHS:
                if sname not in t:
                    continue
                key = (d["region"], tid, sname)
                e = series.setdefault(key, {"years": [], "P": [], "F": [],
                                            "lat_bounds": t["lat_bounds"],
                                            "lon_bounds": t["lon_bounds"]})
                e["years"].append(d["year"])
                e["P"].append(t[sname]["P_fine_raw"])
                e["F"].append([t[sname]["F_fine"][k] for k in
                               ("logvar_0", "logvar_1", "logvar_2",
                                "logvar_3", "slope")])

    # seasonal CAPE per tile
    cape_series: dict[tuple, dict[int, float]] = {}
    for region in REGIONS:
        ds = xr.open_dataset(
            B21_DATA / f"era5_cape_monthlyrecord_{region}.nc"
            if (B21_DATA / f"era5_cape_monthlyrecord_{region}.nc").exists()
            else B21_DATA / f"era5_cape_monthly_longrecord_{region}.nc")
        da = ds["cape"].squeeze()
        tdim = next(dd for dd in da.dims if "time" in dd or dd == "valid_time")
        yrs = np.asarray(ds[tdim].dt.year.values, dtype=int)
        mos = np.asarray(ds[tdim].dt.month.values, dtype=int)
        cl = np.asarray(ds["latitude"].values, dtype=float)
        cn = np.asarray(ds["longitude"].values, dtype=float)
        arr = np.asarray(da.transpose(tdim, ...).values, dtype=float)
        ds.close()
        done = set()
        for (reg, tid, sname), e in series.items():
            if reg != region or (tid, sname) in done:
                continue
            done.add((tid, sname))
            (a0, a1), (o0, o1) = e["lat_bounds"], e["lon_bounds"]
            iy = np.where((cl >= a0 - 1e-9) & (cl <= a1 + 1e-9))[0]
            ix = np.where((cn >= o0 - 1e-9) & (cn <= o1 + 1e-9))[0]
            sub = arr[:, iy][:, :, ix].mean(axis=(1, 2))
            sel_m = np.isin(mos, SEASON_MONTHS[sname])
            out = {}
            for y in sorted(set(int(x) for x in yrs)):
                s = sub[(yrs == y) & sel_m]
                if len(s) == 3:
                    out[y] = float(s.mean())
            cape_series[(region, tid, sname)] = out

    p_trend, c_trend = {}, {}
    beta_t_num, beta_t_den = [], []
    for key, e in series.items():
        years = np.asarray(e["years"])
        order = np.argsort(years)
        yrs_s = years[order]
        yv = np.asarray(e["P"])[order]
        Fm = np.asarray(e["F"])[order]
        A = np.column_stack([np.ones(len(yv)), Fm])
        coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
        resid = yv - A @ coef
        cs = cape_series.get(key, {})
        cy = np.asarray([cs.get(int(y), np.nan) for y in yrs_s])
        ok = np.isfinite(cy)
        if ok.sum() < 30:
            continue
        p_trend[key] = float(theilslopes(resid[ok], yrs_s[ok])[0])
        c_trend[key] = float(theilslopes(cy[ok], yrs_s[ok])[0])
        # beta_T: detrended covariation (shares nothing with the trends)
        t = yrs_s[ok].astype(float)
        t -= t.mean()

        def detr(x):
            return x - (t @ x) / (t @ t) * t - x.mean()
        rp, rc = detr(resid[ok]), detr(cy[ok])
        beta_t_num.append(float(rc @ rp))
        beta_t_den.append(float(rc @ rc))

    keys = sorted(p_trend)
    pt = np.asarray([p_trend[k] for k in keys])
    ct = np.asarray([c_trend[k] for k in keys])
    regions = np.asarray([k[0] for k in keys])
    beta_T = float(np.sum(beta_t_num) / np.sum(beta_t_den))

    # beta_CS from Arm A (as Phase 21)
    arma = {}
    for f in sorted(ARMA_RES.glob("tiles_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for tid, t in d["tiles"].items():
            e = arma.setdefault((d["region"], tid), {"anch": [], "eke": [],
                                                     "lat_bounds": t["lat_bounds"],
                                                     "lon_bounds": t["lon_bounds"]})
            e["anch"].append(t["P_fine_anch_mean"])
            e["eke"].append(t["eke_syn"])
    akeys = sorted(arma)
    stat = static_covariates({k: arma[k] for k in akeys})
    ya = np.asarray([np.median(arma[k]["anch"]) for k in akeys])
    Xr = np.column_stack([
        [stat[k]["orog_std"] for k in akeys],
        [stat[k]["land_frac"] for k in akeys],
        [stat[k]["coast_var"] for k in akeys],
        [stat[k]["cape_mean"] for k in akeys],
        [np.median(arma[k]["eke"]) for k in akeys]])
    mu, sd = Xr.mean(0), Xr.std(0)
    Xz = (Xr - mu) / np.where(sd < 1e-12, 1.0, sd)
    A = np.column_stack([np.ones(len(akeys)), Xz])
    coef, *_ = np.linalg.lstsq(A, ya, rcond=None)
    beta_cs = float(coef[1 + COVARS.index("cape_mean")]) \
        / float(sd[COVARS.index("cape_mean")])

    thresh = np.quantile(ct, 2.0 / 3.0)
    top = ct >= thresh
    d_obs = float(np.mean(pt[top]))
    d_pred = beta_cs * float(np.mean(ct[top]))
    d_pred_t = beta_T * float(np.mean(ct[top]))

    perm = np.empty(N_PERM)
    for i in range(N_PERM):
        pp = pt.copy()
        for reg in np.unique(regions):
            msk = regions == reg
            pp[msk] = pp[msk][rng.permutation(msk.sum())]
        perm[i] = float(np.mean(pp[top]))
    p_a = float((1 + np.sum(perm <= d_obs)) / (N_PERM + 1))

    reg_list = sorted(set(regions))
    idx_by_reg = {g: np.where((regions == g) & top)[0] for g in reg_list}
    idx_by_reg = {g: v for g, v in idx_by_reg.items() if len(v)}
    gl = sorted(idx_by_reg)
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.choice(gl, size=len(gl), replace=True)
        boots[i] = float(np.mean(pt[np.concatenate([idx_by_reg[g] for g in pick])]))
    ci = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))

    p1a = bool(d_obs < 0 and p_a < 0.05)
    covers_pred = bool(ci[0] <= d_pred <= ci[1])
    covers_zero = bool(ci[0] <= 0.0 <= ci[1])
    excl_half = bool(ci[0] > 0.5 * d_pred)
    if p1a and covers_pred:
        v = "P1_SUPPORTED"
    elif (not covers_pred) and excl_half and covers_zero:
        v = "P1_FALSIFIED"
    elif covers_zero and covers_pred:
        v = "P1_UNDERPOWERED"
    else:
        v = "P1_MIXED"
    return {
        "n_tile_season_types": len(keys),
        "beta_CS_per_Jkg": beta_cs, "beta_T_per_Jkg": beta_T,
        "beta_ratio_T_over_CS": beta_T / beta_cs if beta_cs != 0 else None,
        "cape_trend_top_tercile_mean": float(np.mean(ct[top])),
        "D_pred_CS_per_yr": d_pred, "D_pred_T_per_yr": d_pred_t,
        "H22_P1a": {"D_obs_per_yr": d_obs, "p": p_a, "pass": p1a},
        "H22_P1b": {"ci95": list(ci), "covers_pred_CS": covers_pred,
                    "covers_pred_T": bool(ci[0] <= d_pred_t <= ci[1]),
                    "covers_zero": covers_zero},
        "VERDICT_P1s": v,
    }


def stage_tests() -> dict:
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED, "protocol": "docs/PROTOCOL_PHASE22_QT_ROUND2.md"}
    res["ARM_P2"] = tests_p2()
    res["ARM_P1s"] = tests_p1s(rng)
    res["PHASE22_VERDICT"] = [res["ARM_P2"]["VERDICT_P2"],
                              res["ARM_P1s"]["VERDICT_P1s"]]
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["model-tiles", "era5-05-tiles",
                                        "p1s-series", "tests"],
                    default="tests")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 4) - 3))
    args = ap.parse_args()
    if args.stage == "model-tiles":
        stage_model_tiles(args.workers)
    elif args.stage == "era5-05-tiles":
        stage_era5_05_tiles(args.workers)
    elif args.stage == "p1s-series":
        stage_p1s(args.workers)
    else:
        res = stage_tests()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "summary.json").write_text(json.dumps(res, indent=2),
                                          encoding="utf-8")
        print(json.dumps(res, indent=2, default=str))


if __name__ == "__main__":
    main()
