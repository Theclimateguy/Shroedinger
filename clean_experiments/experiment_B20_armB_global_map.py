#!/usr/bin/env python3
"""Phase-20 Arm B: the global tile map of anchored fine-P.

Frozen spec: docs/PROTOCOL_PHASE20_P_GEOGRAPHY.md, "Arm B execution
spec" (2026-08-18). Global 6-deg-row / 700-km tile grid, 60S-60N, two
seasons (JFM/JAS 2023); anchored fine-P per tile exactly as Arm A;
two-tier covariates; longitude-rotation nulls; LOSO (60-deg sector) CV.

Stages: --stage tiles --season JFM|JAS | --stage tests
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
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from clean_experiments.experiment_B18_equal_km_regions import (
        envelope_rho_profile_ells,
    )
    from clean_experiments.experiment_B1_true_flux_baselines import (
        _grid_geometry,
        _interior_mask,
        phase_randomize,
    )
    from clean_experiments.experiment_B20_p_geography import (
        ELLS_FINE,
        N_FINE,
        N_SUR,
        RUNMEAN_STEPS,
        fine_features,
    )
    from clean_experiments.experiment_B2_scale_irreversibility import (
        compute_vorticity,
    )
except ImportError:
    from experiment_B18_equal_km_regions import (  # type: ignore
        envelope_rho_profile_ells,
    )
    from experiment_B1_true_flux_baselines import (  # type: ignore
        _grid_geometry,
        _interior_mask,
        phase_randomize,
    )
    from experiment_B20_p_geography import (  # type: ignore
        ELLS_FINE,
        N_FINE,
        N_SUR,
        RUNMEAN_STEPS,
        fine_features,
    )
    from experiment_B2_scale_irreversibility import (  # type: ignore
        compute_vorticity,
    )

SEED = 20260818
SEED_SUR = 20260811
OUT = _HERE / "results" / "experiment_B20_armB_global_map"
ARMA = _HERE / "results" / "experiment_B20_p_geography"
GLOB = Path("data/b20global")
INV = Path("data/b4cov/era5_invariants_global.nc")
COVARS_NC = GLOB / "era5_covars_monthly_global.nc"

SEASONS = {"JFM": ["202301", "202302", "202303"],
           "JAS": ["202307", "202308", "202309"]}
LAT_STEP = 6.0
TILE_W_KM = 700.0
KM_PER_DEG = 6371.0 * np.pi / 180.0
LAT_MIN, LAT_MAX = -60.0, 60.0
OROG_EXCL_M = 1200.0
G0 = 9.80665
N_ROT = 999
ROT_MIN_DEG = 30.0
SECTOR_DEG = 60.0
TIER1 = ("orog_mean", "orog_std", "land_frac", "coast_var", "abs_lat",
         "sst_grad")
TIER2 = ("cape_mean", "eke_syn")
ALL_COV = TIER1 + TIER2


# --------------------------------------------------------------------------
# tile grid
# --------------------------------------------------------------------------

def tile_grid() -> list[dict]:
    tiles = []
    lat_edges = np.arange(LAT_MIN, LAT_MAX + 1e-9, LAT_STEP)
    for ri in range(len(lat_edges) - 1):
        la0, la1 = float(lat_edges[ri]), float(lat_edges[ri + 1])
        lat_c = 0.5 * (la0 + la1)
        dlon_km = TILE_W_KM / (KM_PER_DEG * np.cos(np.deg2rad(lat_c)))
        n_lon = int(np.floor(360.0 / dlon_km))
        dlon = 360.0 / n_lon
        for ci in range(n_lon):
            tiles.append({"tid": f"t{ri:02d}_{ci:03d}", "row": ri, "col": ci,
                          "lat0": la0, "lat1": la1, "lat_c": lat_c,
                          "lon0": ci * dlon, "lon1": (ci + 1) * dlon,
                          "lon_c": (ci + 0.5) * dlon, "dlon": dlon,
                          "n_lon_row": n_lon})
    return tiles


# --------------------------------------------------------------------------
# tiles stage
# --------------------------------------------------------------------------

_G: dict[str, object] = {}


def _tile_worker(t: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    from scipy.ndimage import uniform_filter1d
    u, v = _G["u"], _G["v"]
    lat, lon = _G["lat"], _G["lon"]
    season = _G["season"]
    iy = np.where((lat >= t["lat0"] - 1e-9) & (lat <= t["lat1"] + 1e-9))[0]
    nx = len(lon)
    i0 = int(np.searchsorted(lon, t["lon0"] - 1e-9))
    n_cols = int(round((t["lon1"] - t["lon0"]) / 0.25))
    ix = (np.arange(i0, i0 + n_cols)) % nx
    la = lat[iy]
    lo_raw = lon[ix]
    lo = t["lon0"] + 0.25 * np.arange(n_cols)      # monotone lon for geometry
    uu = u[:, iy][:, :, ix].astype(np.float32)
    vv = v[:, iy][:, :, ix].astype(np.float32)
    del lo_raw
    dx_km, dy_km = _grid_geometry(la, lo)
    mask = _interior_mask(uu.shape[1], uu.shape[2], ELLS_FINE[-1],
                          dx_km, dy_km)
    omega = compute_vorticity(uu, vv, la, lo)
    rho = envelope_rho_profile_ells(omega, dx_km, dy_km, mask, ELLS_FINE)
    p_real = np.median(rho, axis=0)
    tcode = zlib.crc32(f"glob_{t['tid']}_{season}".encode()) % 100000
    seeds = np.random.SeedSequence(SEED_SUR + tcode).generate_state(N_SUR)
    sur = np.empty((N_SUR, N_FINE))
    for j, s in enumerate(seeds):
        rng = np.random.default_rng(int(s))
        us, vs = phase_randomize(uu, vv, rng)
        om_s = compute_vorticity(us, vs, la, lo)
        sur[j] = np.median(envelope_rho_profile_ells(
            om_s, dx_km, dy_km, mask, ELLS_FINE), axis=0)
    sur_med = np.median(sur, axis=0)
    up = uu - uniform_filter1d(uu, RUNMEAN_STEPS, axis=0, mode="nearest")
    vp = vv - uniform_filter1d(vv, RUNMEAN_STEPS, axis=0, mode="nearest")
    eke = float(np.mean(0.5 * (np.var(up, axis=0) + np.var(vp, axis=0))[mask]))
    return {"tid": t["tid"],
            "P_fine_real": [float(x) for x in p_real],
            "P_fine_anch_mean": float(np.mean(p_real - sur_med)),
            "F_fine": fine_features(omega, dx_km, dy_km, mask),
            "eke_syn": eke}


def stage_tiles(season: str, workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cached = OUT / f"tiles_{season}2023.json"
    if cached.exists():
        print("cached", flush=True)
        return
    import xarray as xr
    parts = []
    for ym in SEASONS[season]:
        ds = xr.open_dataset(GLOB / f"era5_wind850_global_{ym}.nc")
        parts.append(ds)
    ds = xr.concat(parts, dim="valid_time")
    lat = np.asarray(ds["latitude"].values, dtype=float)
    lon = np.asarray(ds["longitude"].values, dtype=float)
    print("loading u, v ...", flush=True)
    u = np.asarray(ds["u"].squeeze().values, dtype=np.float32)
    v = np.asarray(ds["v"].squeeze().values, dtype=np.float32)
    for p in parts:
        p.close()
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        u = u[:, ::-1, :]
        v = v[:, ::-1, :]
    print(f"{season}: cube {u.shape}", flush=True)
    tiles = tile_grid()
    _G.update(u=u, v=v, lat=lat, lon=lon, season=season)
    ctx = mp.get_context("fork")
    results = []
    with ctx.Pool(workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_tile_worker, tiles, chunksize=4)):
            results.append(r)
            if (i + 1) % 100 == 0:
                print(f"[{i+1}/{len(tiles)}]", flush=True)
    cached.write_text(json.dumps({"season": season, "tiles": results},
                                 indent=1), encoding="utf-8")
    print(f"{season}: {len(results)} tiles done", flush=True)


# --------------------------------------------------------------------------
# covariates
# --------------------------------------------------------------------------

def tile_covariates(tiles: list[dict]) -> dict[str, dict[str, float]]:
    import xarray as xr
    inv = xr.open_dataset(INV)
    z = inv["z"].squeeze().values / G0
    lsm = inv["lsm"].squeeze().values
    ilat = inv["latitude"].values
    ilon = inv["longitude"].values
    if ilat[0] > ilat[-1]:
        ilat = ilat[::-1]
        z = z[::-1]
        lsm = lsm[::-1]

    cv = xr.open_dataset(COVARS_NC)
    def clim(name):
        da = cv[name]
        tdim = next(d for d in da.dims if "time" in d or d == "valid_time")
        m = da.mean(dim=tdim).squeeze().values
        clat = cv["latitude"].values
        if clat[0] > clat[-1]:
            m = m[::-1]
        return m
    sst = clim("sst")
    cape = clim("cape")
    lai = clim("lai_lv") + clim("lai_hv")
    # SST gradient magnitude (K/km) on the native grid
    dy = KM_PER_DEG * 0.25
    dx_row = KM_PER_DEG * 0.25 * np.cos(np.deg2rad(ilat))
    gy, gx = np.gradient(np.nan_to_num(sst, nan=np.nan))
    with np.errstate(invalid="ignore"):
        gmag = np.sqrt((gy / dy) ** 2 + (gx / dx_row[:, None]) ** 2)
    gmag[~np.isfinite(sst)] = np.nan

    out = {}
    nxg = len(ilon)
    for t in tiles:
        iy = np.where((ilat >= t["lat0"] - 1e-9) & (ilat <= t["lat1"] + 1e-9))[0]
        i0 = int(np.searchsorted(ilon, t["lon0"] - 1e-9))
        n_cols = int(round((t["lon1"] - t["lon0"]) / 0.25))
        ix = (np.arange(i0, i0 + n_cols)) % nxg
        zt = z[np.ix_(iy, ix)]
        lt = lsm[np.ix_(iy, ix)]
        st = sst[np.ix_(iy, ix)]
        gt = gmag[np.ix_(iy, ix)]
        ct = cape[np.ix_(iy, ix)]
        lai_t = lai[np.ix_(iy, ix)]
        ocean_frac = float(np.mean(np.isfinite(st)))
        out[t["tid"]] = {
            "orog_mean": float(np.mean(zt)),
            "orog_std": float(np.std(zt)),
            "land_frac": float(np.mean(lt)),
            "coast_var": float(np.std(lt)),
            "abs_lat": abs(t["lat_c"]),
            "sst_grad": (float(np.nanmean(gt)) if ocean_frac >= 0.3 else 0.0),
            "cape_mean": float(np.nanmean(ct)),
            "lai_mean": float(np.nanmean(lai_t)),
        }
    inv.close()
    cv.close()
    return out


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def loso_r2(X: np.ndarray, y: np.ndarray, sector: np.ndarray) -> float:
    pred = np.empty_like(y)
    for s in np.unique(sector):
        te = sector == s
        tr = ~te
        A = np.column_stack([np.ones(tr.sum()), X[tr]])
        coef, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pred[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ coef
    ss = float(np.sum((y - pred) ** 2))
    st = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss / st


def rotate_cov(X: np.ndarray, tiles: list[dict], offset_deg: float,
               tid_index: dict[str, int]) -> np.ndarray:
    Xr = np.empty_like(X)
    for t in tiles:
        shift = int(round(offset_deg / t["dlon"]))
        src_col = (t["col"] + shift) % t["n_lon_row"]
        src_tid = f"t{t['row']:02d}_{src_col:03d}"
        Xr[tid_index[t["tid"]]] = X[tid_index[src_tid]]
    return Xr


def stage_tests() -> dict:
    rng = np.random.default_rng(SEED)
    res: dict = {"seed": SEED,
                 "protocol": "docs/PROTOCOL_PHASE20_P_GEOGRAPHY.md (Arm B spec)"}
    grid = tile_grid()
    by_tid = {t["tid"]: t for t in grid}

    seasons = {}
    for s in SEASONS:
        d = json.loads((OUT / f"tiles_{s}2023.json").read_text(encoding="utf-8"))
        seasons[s] = {r["tid"]: r for r in d["tiles"]}
    common = sorted(set(seasons["JFM"]) & set(seasons["JAS"]))
    cov = tile_covariates([by_tid[t] for t in common])

    keep = [t for t in common if cov[t]["orog_mean"] <= OROG_EXCL_M]
    res["n_tiles_total"] = len(common)
    res["n_tiles_excluded_orog"] = len(common) - len(keep)
    tiles = [by_tid[t] for t in keep]
    tid_index = {t: i for i, t in enumerate(keep)}

    y = np.asarray([0.5 * (seasons["JFM"][t]["P_fine_anch_mean"]
                           + seasons["JAS"][t]["P_fine_anch_mean"])
                    for t in keep])
    slope = np.asarray([np.nanmean([seasons[s][t]["F_fine"]["slope"]
                                    for s in SEASONS]) for t in keep])
    eke = np.asarray([np.mean([seasons[s][t]["eke_syn"] for s in SEASONS])
                      for t in keep])
    Xr_cols = {n: np.asarray([cov[t][n] for t in keep]) for n in
               ("orog_mean", "orog_std", "land_frac", "coast_var",
                "abs_lat", "sst_grad", "cape_mean", "lai_mean")}
    Xr_cols["eke_syn"] = eke

    def z(a):
        sd = a.std()
        return (a - a.mean()) / (sd if sd > 1e-12 else 1.0)
    Xz = {n: z(v) for n, v in Xr_cols.items()}
    X_all = np.column_stack([Xz[n] for n in ALL_COV])
    X_t1 = np.column_stack([Xz[n] for n in TIER1])
    sector = np.asarray([int(by_tid[t]["lon_c"] // SECTOR_DEG) for t in keep])

    def rot_null(Xm: np.ndarray, target: np.ndarray, n=N_ROT) -> np.ndarray:
        vals = np.empty(n)
        for i in range(n):
            off = float(rng.uniform(ROT_MIN_DEG, 360.0 - ROT_MIN_DEG))
            vals[i] = loso_r2(rotate_cov(Xm, tiles, off, tid_index),
                              target, sector)
        return vals

    # ---- H-B0 sanity vs Arm A -------------------------------------------
    arma_tiles = {}
    for f in sorted(ARMA.glob("tiles_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for tid, t in d["tiles"].items():
            arma_tiles.setdefault((d["region"], tid), {"v": [], **t})[
                "v"].append(t["P_fine_anch_mean"])
    pairs = []
    for (reg, tid), t in arma_tiles.items():
        (a0, a1), (o0, o1) = t["lat_bounds"], t["lon_bounds"]
        la_c, lo_c = 0.5 * (a0 + a1), (0.5 * (o0 + o1)) % 360.0
        for g in keep:
            gt = by_tid[g]
            if gt["lat0"] <= la_c <= gt["lat1"] and gt["lon0"] <= lo_c <= gt["lon1"]:
                pairs.append((float(np.median(t["v"])), y[tid_index[g]]))
                break
    pa = np.asarray(pairs)
    rho_b0 = float(spearmanr(pa[:, 0], pa[:, 1]).statistic)
    res["H_B0"] = {"n_pairs": len(pairs), "rho": rho_b0,
                   "pass": bool(rho_b0 >= 0.7)}
    if not res["H_B0"]["pass"]:
        res["ARM_B_VERDICT"] = "MAP_INCONSISTENT"
        return res

    # ---- H-B1 primary ----------------------------------------------------
    r2_full = loso_r2(X_all, y, sector)
    null_full = rot_null(X_all, y)
    p_full = float((1 + np.sum(null_full >= r2_full)) / (N_ROT + 1))
    res["H_B1"] = {"loso_r2": r2_full, "p_rot": p_full,
                   "null_q95": float(np.quantile(null_full, 0.95)),
                   "pass": bool(r2_full > 0 and p_full < 0.05)}

    # ---- H-B2 tier separation -------------------------------------------
    r2_t1 = loso_r2(X_t1, y, sector)
    null_t1 = rot_null(X_t1, y, 499)
    res["H_B2"] = {"tier1_r2": r2_t1,
                   "tier1_p": float((1 + np.sum(null_t1 >= r2_t1)) / 500),
                   "tier2_increment": r2_full - r2_t1}

    # ---- H-B3 effective dimension ---------------------------------------
    chosen: list[str] = []
    curve = []
    remaining = list(ALL_COV)
    while remaining:
        best, best_r2 = None, -np.inf
        for n in remaining:
            cols = [Xz[m] for m in chosen + [n]]
            r2 = loso_r2(np.column_stack(cols), y, sector)
            if r2 > best_r2:
                best, best_r2 = n, r2
        chosen.append(best)
        remaining.remove(best)
        curve.append({"k": len(chosen), "added": best, "loso_r2": best_r2})
    k80 = next((c["k"] for c in curve if c["loso_r2"] >= 0.8 * r2_full),
               len(ALL_COV))
    res["H_B3"] = {"forward_curve": curve, "k80": k80,
                   "pass_theory_P3": bool(k80 <= 3)}

    # ---- H-B4 spectral placebo ------------------------------------------
    ok = np.isfinite(slope)
    r2_sl = loso_r2(X_all[ok], slope[ok], sector[ok])
    res["H_B4"] = {"slope_loso_r2": float(r2_sl),
                   "loadings_slope": {n: float(spearmanr(Xz[n][ok], slope[ok]).statistic)
                                      for n in ALL_COV},
                   "loadings_anchP": {n: float(spearmanr(Xz[n], y).statistic)
                                      for n in ALL_COV}}

    # ---- H-B5 LAI negative control --------------------------------------
    base = np.column_stack([Xz["land_frac"], Xz["cape_mean"], Xz["eke_syn"]])
    with_lai = np.column_stack([base, Xz["lai_mean"]])
    gain = loso_r2(with_lai, y, sector) - loso_r2(base, y, sector)
    null_gain = np.empty(499)
    for i in range(499):
        off = float(rng.uniform(ROT_MIN_DEG, 360.0 - ROT_MIN_DEG))
        lai_rot = rotate_cov(Xz["lai_mean"][:, None], tiles, off, tid_index)
        null_gain[i] = (loso_r2(np.column_stack([base, lai_rot]), y, sector)
                        - loso_r2(base, y, sector))
    p_lai = float((1 + np.sum(null_gain >= gain)) / 500)
    res["H_B5"] = {"lai_gain": float(gain), "p_rot": p_lai,
                   "confound_flag": bool(p_lai < 0.05 and gain > 0)}

    # ---- descriptive: season contrast -----------------------------------
    d_season = np.asarray([seasons["JFM"][t]["P_fine_anch_mean"]
                           - seasons["JAS"][t]["P_fine_anch_mean"]
                           for t in keep])
    res["season_contrast"] = {"median_abs": float(np.median(np.abs(d_season))),
                              "q90_abs": float(np.quantile(np.abs(d_season), 0.9))}

    # ---- verdict ---------------------------------------------------------
    v = ("GLOBAL_MAP_ATTRIBUTED" if res["H_B1"]["pass"]
         else "GLOBAL_MAP_UNATTRIBUTED")
    if res["H_B1"]["pass"]:
        v += f"_DIM{k80}"
    if res["H_B5"]["confound_flag"]:
        v += "_CONFOUND_FLAG"
    res["ARM_B_VERDICT"] = v
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["tiles", "tests"], default="tiles")
    ap.add_argument("--season", choices=list(SEASONS), default="JFM")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 4) - 3))
    args = ap.parse_args()
    if args.stage == "tiles":
        stage_tiles(args.season, args.workers)
    else:
        res = stage_tests()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "summary.json").write_text(json.dumps(res, indent=2),
                                          encoding="utf-8")
        brief = {k: res[k] for k in res if k in
                 ("n_tiles_total", "n_tiles_excluded_orog", "H_B0", "H_B1",
                  "H_B2", "H_B5", "ARM_B_VERDICT")}
        brief["H_B3_k80"] = res.get("H_B3", {}).get("k80")
        print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
