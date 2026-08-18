#!/usr/bin/env python3
"""Phase-20 Arm A: the geography of P — tile-level attribution.

Preregistered protocol: docs/PROTOCOL_PHASE20_P_GEOGRAPHY.md
(frozen 2026-08-18).

Each equal-km box (Phase 18) is split 3x4 into ~667x700 km tiles; the
frozen envelope machinery restricted to ELLS_FINE = [50,100,200,400] km
gives an anchored fine-P per tile (99 phase surrogates, Phase-2/3 seed
scheme salted with the tile id). Attribution against on-disk covariates
[orog_std, land_frac, coast_var, cape_mean, eke_syn] with
leave-one-region-out R^2 and region-block permutation nulls.

Stages: --stage tiles | tests
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
        HELDOUT_REGIONS,
        HELDOUT_WINDOWS,
        PRIMARY_WINDOWS,
        REGIONS,
        envelope_rho_profile_ells,
        km_crop,
    )
    from clean_experiments.experiment_B1_true_flux_baselines import (
        _grid_geometry,
        _interior_mask,
        phase_randomize,
    )
    from clean_experiments.experiment_B2_scale_irreversibility import (
        EPS,
        compute_vorticity,
    )
    from clean_experiments.experiment_scale_gravity_einstein_box_era import (
        _load_vector_fields,
    )
except ImportError:
    from experiment_B18_equal_km_regions import (  # type: ignore
        HELDOUT_REGIONS,
        HELDOUT_WINDOWS,
        PRIMARY_WINDOWS,
        REGIONS,
        envelope_rho_profile_ells,
        km_crop,
    )
    from experiment_B1_true_flux_baselines import (  # type: ignore
        _grid_geometry,
        _interior_mask,
        phase_randomize,
    )
    from experiment_B2_scale_irreversibility import (  # type: ignore
        EPS,
        compute_vorticity,
    )
    from experiment_scale_gravity_einstein_box_era import (  # type: ignore
        _load_vector_fields,
    )

SEED = 20260818
SEED_SUR = 20260811                     # Phase-2/3 surrogate scheme
OUT = _HERE / "results" / "experiment_B20_p_geography"
B18_RES = _HERE / "results" / "experiment_B18_equal_km_regions"
COV_DIR = Path("data/b4cov")
WIND_B3 = Path("data/b3")
WIND_B2B = Path("data/b2b")

ELLS_FINE = [50.0, 100.0, 200.0, 400.0]
N_FINE = len(ELLS_FINE) - 1             # 3 band steps
N_SUR = 99
TILE_ROWS, TILE_COLS = 3, 4
RUNMEAN_STEPS = 20                      # 5 days at 6 h
N_PERM = 999
G0 = 9.80665
COVARS = ("orog_std", "land_frac", "coast_var", "cape_mean", "eke_syn")


# --------------------------------------------------------------------------
# tiles stage
# --------------------------------------------------------------------------

def tile_slices(n: int, parts: int) -> list[slice]:
    edges = np.linspace(0, n, parts + 1).astype(int)
    return [slice(edges[i], edges[i + 1]) for i in range(parts)]


def fine_features(omega: np.ndarray, dx_km: np.ndarray, dy_km: float,
                  mask: np.ndarray) -> dict:
    """Fine-range log band variances (4) + isotropic slope over 50-400 km."""
    try:
        from clean_experiments.experiment_B2_scale_irreversibility import (
            gaussian_bar_grouped,
        )
    except ImportError:
        from experiment_B2_scale_irreversibility import gaussian_bar_grouped  # type: ignore
    nt, ny, nx = omega.shape
    w = np.nan_to_num(omega).astype(np.float32)
    bars = [gaussian_bar_grouped(w, ell / np.sqrt(12.0), dx_km, dy_km)
            for ell in ELLS_FINE]
    out = {}
    out["logvar_0"] = float(np.log(np.mean(np.var((w - bars[0])[:, mask], axis=1)) + 1e-300))
    for i in range(N_FINE):
        v = np.mean(np.var((bars[i] - bars[i + 1])[:, mask], axis=1))
        out[f"logvar_{i + 1}"] = float(np.log(v + 1e-300))
    dxm = float(np.mean(dx_km))
    ky = np.fft.fftfreq(ny, d=dy_km)
    kx = np.fft.rfftfreq(nx, d=dxm)
    kmag = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    psd = np.zeros_like(kmag)
    for t in range(nt):
        f = np.nan_to_num(omega[t])
        psd += np.abs(np.fft.rfft2(f - f.mean())) ** 2
    psd /= nt
    k_bins = np.geomspace(1.0 / 400.0, 1.0 / 50.0, 7)
    sh_k, sh_e = [], []
    for lo, hi in zip(k_bins[:-1], k_bins[1:]):
        m = (kmag >= lo) & (kmag < hi)
        if m.sum() > 0:
            sh_k.append(np.sqrt(lo * hi))
            sh_e.append(float(psd[m].mean()))
    out["slope"] = (float(np.polyfit(np.log(sh_k),
                                     np.log(np.asarray(sh_e) + 1e-300), 1)[0])
                    if len(sh_k) >= 3 else float("nan"))
    return out


def _tiles_one(nc_path: Path) -> str:
    os.environ["OMP_NUM_THREADS"] = "1"
    tag = nc_path.stem.replace("era5_wind850_", "")
    cached = OUT / f"tiles_{tag}.json"
    if cached.exists():
        return tag
    from scipy.ndimage import uniform_filter1d
    region = tag.split("__")[0]
    _, lat, lon, u, v, _, _ = _load_vector_fields(
        input_path=nc_path, field_set="wind", u_var=None, v_var=None,
        time_stride=1, lat_stride=1, lon_stride=1, time_start=0,
        max_time=None, crop_ny=None, crop_nx=None,
    )
    lat, lon, u, v = km_crop(lat, lon, u, v, region, 1.0)
    up = u - uniform_filter1d(u, RUNMEAN_STEPS, axis=0, mode="nearest")
    vp = v - uniform_filter1d(v, RUNMEAN_STEPS, axis=0, mode="nearest")

    rows = tile_slices(u.shape[1], TILE_ROWS)
    cols = tile_slices(u.shape[2], TILE_COLS)
    tiles = {}
    for ri, rs in enumerate(rows):
        for ci, cs in enumerate(cols):
            tid = f"r{ri}c{ci}"
            la, lo = lat[rs], lon[cs]
            uu, vv = u[:, rs, cs], v[:, rs, cs]
            dx_km, dy_km = _grid_geometry(la, lo)
            mask = _interior_mask(uu.shape[1], uu.shape[2], ELLS_FINE[-1],
                                  dx_km, dy_km)
            omega = compute_vorticity(uu, vv, la, lo)
            rho = envelope_rho_profile_ells(omega, dx_km, dy_km, mask, ELLS_FINE)
            p_real = np.median(rho, axis=0)
            tcode = zlib.crc32((tag + "_" + tid).encode()) % 100000
            seeds = np.random.SeedSequence(SEED_SUR + tcode).generate_state(N_SUR)
            sur = np.empty((N_SUR, N_FINE))
            for j, s in enumerate(seeds):
                rng = np.random.default_rng(int(s))
                us, vs = phase_randomize(uu, vv, rng)
                om_s = compute_vorticity(us, vs, la, lo)
                sur[j] = np.median(envelope_rho_profile_ells(
                    om_s, dx_km, dy_km, mask, ELLS_FINE), axis=0)
            sur_med = np.median(sur, axis=0)
            eke = float(np.mean(0.5 * (up[:, rs, cs][:, mask] ** 2
                                       + vp[:, rs, cs][:, mask] ** 2)))
            tiles[tid] = {
                "lat_bounds": [float(la.min()), float(la.max())],
                "lon_bounds": [float(lo.min()), float(lo.max())],
                "P_fine_real": [float(x) for x in p_real],
                "P_fine_sur_med": [float(x) for x in sur_med],
                "P_fine_anch_mean": float(np.mean(p_real - sur_med)),
                "F_fine": fine_features(omega, dx_km, dy_km, mask),
                "eke_syn": eke,
                "n_px": [int(uu.shape[1]), int(uu.shape[2])],
            }
    OUT.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(
        {"tag": tag, "region": region, "tiles": tiles}, indent=1),
        encoding="utf-8")
    print(f"[{tag}] 12 tiles done", flush=True)
    return tag


def stage_tiles(workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = [WIND_B3 / f"era5_wind850_{r}__{w}.nc"
             for r in REGIONS for w in PRIMARY_WINDOWS]
    files += [WIND_B2B / f"era5_wind850_{r}__{w}.nc"
              for r in HELDOUT_REGIONS for w in HELDOUT_WINDOWS]
    files = [f for f in files if f.exists()]
    print(f"{len(files)} region-windows", flush=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        pool.map(_tiles_one, files)


# --------------------------------------------------------------------------
# static covariates per tile
# --------------------------------------------------------------------------

def static_covariates(tile_meta: dict) -> dict:
    import xarray as xr
    inv = xr.open_dataset(COV_DIR / "era5_invariants_global.nc")
    z = inv["z"].squeeze()
    lsm = inv["lsm"].squeeze()
    lat_n = "latitude" if "latitude" in z.dims else "lat"
    lon_n = "longitude" if "longitude" in z.dims else "lon"
    lon_all = np.asarray(z[lon_n].values, dtype=float)

    cape_cache: dict[str, object] = {}
    out = {}
    for key, meta in tile_meta.items():
        region = key[0]
        (la0, la1), (lo0, lo1) = meta["lat_bounds"], meta["lon_bounds"]
        zz = z.sel({lat_n: slice(la1, la0)})
        ll = lsm.sel({lat_n: slice(la1, la0)})
        lom = (lon_all + 360.0) % 360.0
        sel = (lom >= (lo0 + 360) % 360) & (lom <= (lo1 + 360) % 360)
        if not sel.any():
            sel = (lom >= (lo0 + 360) % 360) | (lom <= (lo1 + 360) % 360)
        idx = np.where(sel)[0]
        zt = zz.isel({lon_n: idx}).values / G0
        lt = ll.isel({lon_n: idx}).values
        if region not in cape_cache:
            cape_cache[region] = xr.open_dataset(
                COV_DIR / f"era5_cape_monthly_{region}.nc")["cape"]
        cp = cape_cache[region]
        cpt = cp.sel(latitude=slice(la1, la0), longitude=slice(lo0, lo1))
        out[key] = {
            "orog_std": float(np.std(zt)),
            "land_frac": float(np.mean(lt)),
            "coast_var": float(np.std(lt)),
            "cape_mean": float(cpt.mean()),
        }
    inv.close()
    return out


# --------------------------------------------------------------------------
# tests stage
# --------------------------------------------------------------------------

def _loo_region_r2(X: np.ndarray, y: np.ndarray, regions: np.ndarray) -> float:
    pred = np.empty_like(y)
    for reg in np.unique(regions):
        te = regions == reg
        tr = ~te
        A = np.column_stack([np.ones(tr.sum()), X[tr]])
        coef, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pred[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot


def _block_perm_X(X: np.ndarray, regions: np.ndarray, reg_order: list[str],
                  rng: np.random.Generator, cols: list[int] | None = None) -> np.ndarray:
    """Rotate covariate blocks between regions (tile index preserved)."""
    perm = rng.permutation(len(reg_order))
    Xp = X.copy()
    idx_by_reg = {r: np.where(regions == r)[0] for r in reg_order}
    for i, r in enumerate(reg_order):
        src = reg_order[perm[i]]
        a, b = idx_by_reg[r], idx_by_reg[src]
        n = min(len(a), len(b))
        if cols is None:
            Xp[a[:n]] = X[b[:n]]
        else:
            for c in cols:
                Xp[a[:n], c] = X[b[:n], c]
    return Xp


def stage_tests() -> dict:
    rng = np.random.default_rng(SEED)
    res: dict = {"seed": SEED,
                 "protocol": "docs/PROTOCOL_PHASE20_P_GEOGRAPHY.md"}

    # assemble tile table: key = (region, tid)
    per_tile: dict[tuple, dict] = {}
    for f in sorted(OUT.glob("tiles_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for tid, t in d["tiles"].items():
            key = (d["region"], tid)
            e = per_tile.setdefault(key, {
                "anch": [], "eke": [], "slope": [], "raw": [],
                "lat_bounds": t["lat_bounds"], "lon_bounds": t["lon_bounds"],
                "logvars": []})
            e["anch"].append(t["P_fine_anch_mean"])
            e["eke"].append(t["eke_syn"])
            e["raw"].append(float(np.mean(t["P_fine_real"])))
            e["slope"].append(t["F_fine"]["slope"])
            e["logvars"].append([t["F_fine"][f"logvar_{i}"] for i in range(4)])

    keys = sorted(per_tile.keys())
    stat = static_covariates({k: per_tile[k] for k in keys})
    regions = np.asarray([k[0] for k in keys])
    reg_order = sorted(set(regions))
    y = np.asarray([float(np.median(per_tile[k]["anch"])) for k in keys])
    y_raw = np.asarray([float(np.median(per_tile[k]["raw"])) for k in keys])
    y_slope = np.asarray([float(np.nanmedian(per_tile[k]["slope"])) for k in keys])
    lv = np.asarray([np.median(np.asarray(per_tile[k]["logvars"]), axis=0)
                     for k in keys])
    X_raw_cov = np.column_stack([
        [stat[k]["orog_std"] for k in keys],
        [stat[k]["land_frac"] for k in keys],
        [stat[k]["coast_var"] for k in keys],
        [stat[k]["cape_mean"] for k in keys],
        [float(np.median(per_tile[k]["eke"])) for k in keys],
    ])
    mu, sd = X_raw_cov.mean(0), X_raw_cov.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    X = (X_raw_cov - mu) / sd
    res["n_tiles"] = len(keys)

    # ---- C20-1 sanity ----------------------------------------------------
    box_ref = {}
    for f in sorted(B18_RES.glob("R*__*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        box_ref.setdefault(d["region"], []).append(
            float(np.mean(d["km_1.0"]["P_anchored"][:3])))
    ref12 = np.asarray([np.median(box_ref[r]) for r in reg_order])
    tile12 = np.asarray([np.median(y[regions == r]) for r in reg_order])
    rho_c1 = float(spearmanr(ref12, tile12).statistic)
    res["C20_1"] = {"rho": rho_c1, "pass": bool(rho_c1 >= 0.7),
                    "box_ref": dict(zip(reg_order, ref12.tolist())),
                    "tile_med": dict(zip(reg_order, tile12.tolist()))}
    if not res["C20_1"]["pass"]:
        res["PHASE20_VERDICT"] = "TILE_SIGNAL_ABSENT"
        return res

    # ---- H20a primary ----------------------------------------------------
    r2_obs = _loo_region_r2(X, y, regions)
    perm = np.empty(N_PERM)
    for i in range(N_PERM):
        perm[i] = _loo_region_r2(_block_perm_X(X, regions, reg_order, rng),
                                 y, regions)
    p_a = float((1 + np.sum(perm >= r2_obs)) / (N_PERM + 1))
    res["H20a"] = {"loo_region_r2": r2_obs, "p": p_a,
                   "pass": bool(r2_obs > 0 and p_a < 0.05),
                   "perm_q95": float(np.quantile(perm, 0.95))}

    # ---- H20b per-covariate ---------------------------------------------
    h20b = {}
    for ci, name in enumerate(COVARS):
        keep = [j for j in range(len(COVARS)) if j != ci]
        drop = r2_obs - _loo_region_r2(X[:, keep], y, regions)
        rho_s = float(spearmanr(X[:, ci], y).statistic)
        perm_s = np.empty(N_PERM)
        rng_c = np.random.default_rng(SEED + 100 + ci)
        for i in range(N_PERM):
            Xp = _block_perm_X(X, regions, reg_order, rng_c, cols=[ci])
            perm_s[i] = spearmanr(Xp[:, ci], y).statistic
        p_s = float((1 + np.sum(np.abs(perm_s) >= abs(rho_s))) / (N_PERM + 1))
        h20b[name] = {"ablation_drop_r2": float(drop), "spearman": rho_s,
                      "p_block": p_s,
                      "significant": bool(p_s < 0.05 and drop > 0)}
    res["H20b"] = h20b

    # ---- H20c within-region ---------------------------------------------
    h20c = {}
    for ci, name in enumerate(COVARS):
        rhos = {}
        for r in reg_order:
            m = regions == r
            xv = X_raw_cov[m, ci]
            if np.std(xv) < 1e-12:
                continue
            rhos[r] = float(spearmanr(xv, y[m]).statistic)
        vals = np.asarray(list(rhos.values()))
        n_pos = int(np.sum(vals > 0))
        n_tot = len(vals)
        obs = float(np.mean(vals)) if n_tot else np.nan
        rng_c = np.random.default_rng(SEED + 200 + ci)
        perm_m = np.empty(N_PERM)
        for i in range(N_PERM):
            acc = []
            for r in rhos:
                m = regions == r
                acc.append(spearmanr(X_raw_cov[m, ci],
                                     y[m][rng_c.permutation(m.sum())]).statistic)
            perm_m[i] = np.mean(acc)
        p_c = (float((1 + np.sum(np.abs(perm_m) >= abs(obs))) / (N_PERM + 1))
               if n_tot else None)
        h20c[name] = {"per_region_rho": rhos, "mean_rho": obs,
                      "n_pos": n_pos, "n_regions": n_tot,
                      "sign_consistent": bool(max(n_pos, n_tot - n_pos) >= 10),
                      "p_within_perm": p_c}
    res["H20c"] = h20c

    # ---- C20-2 spectral placebo -----------------------------------------
    placebo = {}
    ok = np.isfinite(y_slope)
    rng_p = np.random.default_rng(SEED + 300)
    r2_sl = _loo_region_r2(X[ok], y_slope[ok], regions[ok])
    perm_sl = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_sl[i] = _loo_region_r2(
            _block_perm_X(X[ok], regions[ok],
                          sorted(set(regions[ok])), rng_p), y_slope[ok],
            regions[ok])
    placebo["slope_target"] = {
        "loo_region_r2": float(r2_sl),
        "p": float((1 + np.sum(perm_sl >= r2_sl)) / (N_PERM + 1))}
    # spectrum-residualized raw P target
    A = np.column_stack([np.ones(len(keys)), lv,
                         np.nan_to_num(y_slope, nan=float(np.nanmedian(y_slope)))])
    coef, *_ = np.linalg.lstsq(A, y_raw, rcond=None)
    y_res = y_raw - A @ coef
    rng_p2 = np.random.default_rng(SEED + 301)
    r2_res = _loo_region_r2(X, y_res, regions)
    perm_res = np.empty(N_PERM)
    for i in range(N_PERM):
        perm_res[i] = _loo_region_r2(
            _block_perm_X(X, regions, reg_order, rng_p2), y_res, regions)
    placebo["raw_P_spectrum_residualized"] = {
        "loo_region_r2": float(r2_res),
        "p": float((1 + np.sum(perm_res >= r2_res)) / (N_PERM + 1))}
    res["C20_2"] = placebo

    # ---- verdict ---------------------------------------------------------
    any_cov = any(v["significant"] for v in h20b.values())
    if res["H20a"]["pass"]:
        res["PHASE20_VERDICT"] = ("DRIVERS_IDENTIFIED" if any_cov
                                  else "WEAK_ATTRIBUTION")
    else:
        res["PHASE20_VERDICT"] = "NO_TILE_ATTRIBUTION"
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["tiles", "tests"], default="tiles")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 4) - 3))
    args = ap.parse_args()
    if args.stage == "tiles":
        stage_tiles(args.workers)
    else:
        res = stage_tests()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "summary.json").write_text(json.dumps(res, indent=2),
                                          encoding="utf-8")
        brief = {"n_tiles": res.get("n_tiles"),
                 "C20_1": res.get("C20_1", {}).get("rho"),
                 "H20a": res.get("H20a"),
                 "H20b": {k: {kk: v[kk] for kk in
                              ("spearman", "p_block", "significant")}
                          for k, v in res.get("H20b", {}).items()},
                 "PHASE20_VERDICT": res["PHASE20_VERDICT"]}
        print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
