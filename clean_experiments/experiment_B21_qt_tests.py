#!/usr/bin/env python3
"""Phase-21: theory-candidate tests, round 1 (QT-P5, QT-P1, QT-P4).

Preregistered protocol: docs/PROTOCOL_PHASE21_QT_TESTS.md
(frozen 2026-08-18).

Stages: --stage p5-series | p1-series | tests
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, theilslopes

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import clean_experiments.experiment_B14_p_relaxation as b14
    import clean_experiments.experiment_B19_armC_km_longrecord as armc
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
    )
    from clean_experiments.experiment_B20_p_geography import (
        COVARS,
        ELLS_FINE,
        TILE_COLS,
        TILE_ROWS,
        fine_features,
        static_covariates,
        tile_slices,
    )
    from clean_experiments.experiment_B2_scale_irreversibility import (
        compute_vorticity,
    )
    from clean_experiments.experiment_scale_gravity_einstein_box_era import (
        _load_vector_fields,
    )
except ImportError:
    import experiment_B14_p_relaxation as b14  # type: ignore
    import experiment_B19_armC_km_longrecord as armc  # type: ignore
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
    )
    from experiment_B20_p_geography import (  # type: ignore
        COVARS,
        ELLS_FINE,
        TILE_COLS,
        TILE_ROWS,
        fine_features,
        static_covariates,
        tile_slices,
    )
    from experiment_B2_scale_irreversibility import (  # type: ignore
        compute_vorticity,
    )
    from experiment_scale_gravity_einstein_box_era import (  # type: ignore
        _load_vector_fields,
    )

SEED = 20260818
OUT = _HERE / "results" / "experiment_B21_qt_tests"
ARMA_RES = _HERE / "results" / "experiment_B20_p_geography"
B21_DATA = Path("data/b21")
B17_DATA = Path("data/b17daily")
WIND_B3 = Path("data/b3")
WIND_B2B = Path("data/b2b")

SIZE_FACTORS = (0.35, 0.5, 0.7, 1.0)
ZONAL_KM = 2800.0
DT_H = 6.0
N_PERM = 999
N_BOOT = 9999
CONF_YEARS = range(1979, 2017)
EXCESS_REGIONS = ("R1_WPWP", "R3_AMAZ", "R5_SPCZ")
INDEX_NAMES = ("pdo", "amo", "dmi", "ipo_tpi")


# --------------------------------------------------------------------------
# Arm P5 series
# --------------------------------------------------------------------------

def _p5_one(nc_path: Path) -> str:
    os.environ["OMP_NUM_THREADS"] = "1"
    import xarray as xr
    tag = nc_path.stem.replace("era5_wind850_", "")
    cached = OUT / f"p5_{tag}.json"
    if cached.exists():
        return tag
    region = tag.split("__")[0]
    _, lat, lon, u, v, _, _ = _load_vector_fields(
        input_path=nc_path, field_set="wind", u_var=None, v_var=None,
        time_stride=1, lat_stride=1, lon_stride=1, time_start=0,
        max_time=None, crop_ny=None, crop_nx=None,
    )
    ds = xr.open_dataset(nc_path)
    tname = "valid_time" if "valid_time" in ds else "time"
    hod = np.asarray(ds[tname].dt.hour.values, dtype=int)
    ds.close()
    out = {"tag": tag, "region": region, "sizes": {}}
    for f in SIZE_FACTORS:
        la, lo, uu, vv = km_crop(lat, lon, u, v, region, f)
        dx_km, dy_km = _grid_geometry(la, lo)
        mask = _interior_mask(uu.shape[1], uu.shape[2], ELLS_FINE[-1],
                              dx_km, dy_km)
        om = compute_vorticity(uu, vv, la, lo)
        rho = envelope_rho_profile_ells(om, dx_km, dy_km, mask, ELLS_FINE)
        comp = rho.mean(axis=1)
        x = b14.deseason(comp, hod[: len(comp)])
        r1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        tau = float(-DT_H / np.log(r1)) if 0.0 < r1 < 0.99 else None
        entry = {"tau_h": tau, "r1": r1}
        if f == 1.0:
            spd = np.sqrt(uu.astype(np.float64) ** 2 + vv.astype(np.float64) ** 2)
            entry["U_mean_ms"] = float(spd[:, mask].mean())
        out["sizes"][f"{f:.2f}"] = entry
    OUT.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[{tag}] p5 done", flush=True)
    return tag


def stage_p5(workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = [WIND_B3 / f"era5_wind850_{r}__{w}.nc"
             for r in REGIONS for w in PRIMARY_WINDOWS]
    files += [WIND_B2B / f"era5_wind850_{r}__{w}.nc"
              for r in HELDOUT_REGIONS for w in HELDOUT_WINDOWS]
    files = [f for f in files if f.exists()]
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        pool.map(_p5_one, files)


# --------------------------------------------------------------------------
# Arm P1 series
# --------------------------------------------------------------------------

def _p1_one(path: Path) -> str:
    os.environ["OMP_NUM_THREADS"] = "1"
    import xarray as xr
    tag = path.stem.replace("era5_wind850daily_", "")
    cached = OUT / f"p1_{tag}.json"
    if cached.exists():
        return tag
    region, year = tag.rsplit("_", 1)
    ds = xr.open_dataset(path)
    lat = np.asarray(ds["latitude"].values, dtype=float)
    lon = np.asarray(ds["longitude"].values, dtype=float)
    u = np.asarray(ds["u"].squeeze().values, dtype=np.float32)
    v = np.asarray(ds["v"].squeeze().values, dtype=np.float32)
    ds.close()
    if u.shape[0] < 360:
        return f"SHORT {tag}"
    lat, lon, u, v = km_crop(lat, lon, u, v, region, 1.0)
    rows = tile_slices(u.shape[1], TILE_ROWS)
    cols = tile_slices(u.shape[2], TILE_COLS)
    tiles = {}
    for ri, rs in enumerate(rows):
        for ci, cs in enumerate(cols):
            la, lo = lat[rs], lon[cs]
            uu, vv = u[:, rs, cs], v[:, rs, cs]
            dx_km, dy_km = _grid_geometry(la, lo)
            mask = _interior_mask(uu.shape[1], uu.shape[2], ELLS_FINE[-1],
                                  dx_km, dy_km)
            om = compute_vorticity(uu, vv, la, lo)
            rho = envelope_rho_profile_ells(om, dx_km, dy_km, mask, ELLS_FINE)
            tiles[f"r{ri}c{ci}"] = {
                "P_fine_raw": float(np.median(rho.mean(axis=1))),
                "F_fine": fine_features(om, dx_km, dy_km, mask),
                "lat_bounds": [float(la.min()), float(la.max())],
                "lon_bounds": [float(lo.min()), float(lo.max())],
            }
    OUT.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps({"tag": tag, "region": region,
                                  "year": int(year), "tiles": tiles},
                                 indent=1), encoding="utf-8")
    return f"DONE {tag}"


def stage_p1(workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(B17_DATA.glob("era5_wind850daily_*.nc"))
    todo = [f for f in files
            if not (OUT / f"p1_{f.stem.replace('era5_wind850daily_', '')}.json"
                    ).exists()]
    print(f"{len(files)} files, {len(todo)} to process", flush=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for i, msg in enumerate(pool.imap_unordered(_p1_one, todo)):
            if not msg.startswith("DONE") or (i + 1) % 50 == 0:
                print(f"[{i+1}/{len(todo)}] {msg}", flush=True)


# --------------------------------------------------------------------------
# index parsing (NOAA PSL .data format)
# --------------------------------------------------------------------------

def load_index_monthly(name: str) -> dict[tuple[int, int], float]:
    out = {}
    for line in (B21_DATA / f"index_{name}.txt").read_text().splitlines():
        parts = line.split()
        if len(parts) == 13 and parts[0].lstrip("-").isdigit():
            y = int(parts[0])
            if not (1800 < y < 2100):
                continue
            for m, val in enumerate(parts[1:], start=1):
                try:
                    x = float(val)
                except ValueError:
                    continue
                if x > -8.0:
                    out[(y, m)] = x
    return out


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def tests_p5(rng: np.random.Generator) -> dict:
    rows = []
    for f in sorted(OUT.glob("p5_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        u = d["sizes"]["1.00"]["U_mean_ms"]
        for sf, e in d["sizes"].items():
            if e["tau_h"] is None:
                continue
            t_cross = ZONAL_KM * float(sf) * 1000.0 / u / 3600.0
            rows.append({"region": d["region"], "tag": d["tag"],
                         "f": float(sf), "tau": e["tau_h"],
                         "t_cross": t_cross})
    tau = np.asarray([r["tau"] for r in rows])
    tc = np.asarray([r["t_cross"] for r in rows])
    regions = np.asarray([r["region"] for r in rows])
    rho_obs = float(spearmanr(np.log(tau), np.log(tc)).statistic)

    # null: within-region permutation of tau across that region's pairs
    # (region-label rotation would preserve the size structure and have
    # no power; logged as a protocol deviation clarification)
    perm = np.empty(N_PERM)
    for i in range(N_PERM):
        tp = tau.copy()
        for reg in np.unique(regions):
            m = regions == reg
            tp[m] = tp[m][rng.permutation(m.sum())]
        perm[i] = spearmanr(np.log(tp), np.log(tc)).statistic
    p = float((1 + np.sum(perm >= rho_obs)) / (N_PERM + 1))

    by_tag: dict[str, list] = {}
    for r in rows:
        by_tag.setdefault(r["tag"], []).append(r)
    signs = []
    for tag, rr in by_tag.items():
        if len(rr) >= 3:
            s = spearmanr([x["f"] for x in rr], [x["tau"] for x in rr]).statistic
            signs.append(s > 0)
    frac_pos = float(np.mean(signs))

    # log-log slope, cluster bootstrap over regions
    reg_list = sorted(set(regions))
    def slope_of(idx):
        A = np.column_stack([np.ones(len(idx)), np.log(tc[idx])])
        return float(np.linalg.lstsq(A, np.log(tau[idx]), rcond=None)[0][1])
    boots = np.empty(N_BOOT)
    idx_by_reg = {g: np.where(regions == g)[0] for g in reg_list}
    for i in range(N_BOOT):
        pick = rng.choice(reg_list, size=len(reg_list), replace=True)
        idx = np.concatenate([idx_by_reg[g] for g in pick])
        boots[i] = slope_of(idx)
    res = {
        "n_pairs": len(rows),
        "H21_P5a": {"rho": rho_obs, "p": p, "pass": bool(rho_obs > 0 and p < 0.05)},
        "H21_P5b": {"frac_within_rw_positive": frac_pos, "n_rw": len(signs),
                    "pass": bool(frac_pos >= 2.0 / 3.0)},
        "slope_loglog": {"point": slope_of(np.arange(len(rows))),
                         "ci95": [float(np.quantile(boots, 0.025)),
                                  float(np.quantile(boots, 0.975))]},
    }
    if not res["H21_P5a"]["pass"]:
        v = "P5_FALSIFIED"
    elif res["H21_P5b"]["pass"]:
        v = "P5_SUPPORTED"
    else:
        v = "P5_MIXED"
    res["VERDICT_P5"] = v
    return res


def tests_p1(rng: np.random.Generator) -> dict:
    import xarray as xr
    # assemble per-tile yearly series
    series: dict[tuple, dict] = {}
    for f in sorted(OUT.glob("p1_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for tid, t in d["tiles"].items():
            key = (d["region"], tid)
            e = series.setdefault(key, {"years": [], "P": [], "F": [],
                                        "lat_bounds": t["lat_bounds"],
                                        "lon_bounds": t["lon_bounds"]})
            e["years"].append(d["year"])
            e["P"].append(t["P_fine_raw"])
            e["F"].append([t["F_fine"][k] for k in
                           ("logvar_0", "logvar_1", "logvar_2", "logvar_3",
                            "slope")])
    # CAPE yearly tile means
    cape_trend, p_trend = {}, {}
    for region in REGIONS:
        nc = B21_DATA / f"era5_cape_monthly_longrecord_{region}.nc"
        ds = xr.open_dataset(nc)
        da = ds["cape"].squeeze()
        tdim = next(dd for dd in da.dims if "time" in dd or dd == "valid_time")
        yrs = np.asarray(ds[tdim].dt.year.values, dtype=int)
        cl = np.asarray(ds["latitude"].values, dtype=float)
        cn = np.asarray(ds["longitude"].values, dtype=float)
        arr = np.asarray(da.transpose(tdim, ...).values, dtype=float)
        ds.close()
        for (reg, tid), e in series.items():
            if reg != region:
                continue
            (a0, a1), (o0, o1) = e["lat_bounds"], e["lon_bounds"]
            iy = np.where((cl >= a0 - 1e-9) & (cl <= a1 + 1e-9))[0]
            ix = np.where((cn >= o0 - 1e-9) & (cn <= o1 + 1e-9))[0]
            sub = arr[:, iy][:, :, ix].mean(axis=(1, 2))
            years_c = sorted(set(int(y) for y in yrs))
            cy = np.asarray([sub[yrs == y].mean() for y in years_c])
            sl_c = theilslopes(cy, np.asarray(years_c))[0]
            # spectrum-fixed P trend
            years_p = np.asarray(e["years"])
            order = np.argsort(years_p)
            yv = np.asarray(e["P"])[order]
            Fm = np.asarray(e["F"])[order]
            A = np.column_stack([np.ones(len(yv)), Fm])
            coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
            resid = yv - A @ coef
            sl_p = theilslopes(resid, years_p[order])[0]
            cape_trend[(reg, tid)] = float(sl_c)
            p_trend[(reg, tid)] = float(sl_p)

    keys = sorted(cape_trend)
    ct = np.asarray([cape_trend[k] for k in keys])
    pt = np.asarray([p_trend[k] for k in keys])
    regions = np.asarray([k[0] for k in keys])

    # beta_CAPE physical from Arm A
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
    beta_z = float(coef[1 + COVARS.index("cape_mean")])
    beta_phys = beta_z / float(sd[COVARS.index("cape_mean")])   # dP per (J/kg)

    thresh = np.quantile(ct, 2.0 / 3.0)
    top = ct >= thresh
    d_obs = float(np.mean(pt[top]))
    d_pred = beta_phys * float(np.mean(ct[top]))

    # null: within-region shuffle of P trends across tiles
    perm = np.empty(N_PERM)
    for i in range(N_PERM):
        pp = pt.copy()
        for reg in np.unique(regions):
            m = regions == reg
            pp[m] = pp[m][rng.permutation(m.sum())]
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

    p1a_pass = bool(d_obs < 0 and p_a < 0.05)
    ci_covers_pred = bool(ci[0] <= d_pred <= ci[1])
    ci_covers_zero = bool(ci[0] <= 0.0 <= ci[1])
    excl_half_pred = bool(ci[0] > 0.5 * d_pred)   # d_pred<0: CI above half-pred
    if p1a_pass and ci_covers_pred:
        v = "P1_SUPPORTED"
    elif (not ci_covers_pred) and excl_half_pred and ci_covers_zero:
        v = "P1_FALSIFIED"
    elif ci_covers_zero and ci_covers_pred:
        v = "P1_UNDERPOWERED"
    else:
        v = "P1_MIXED"
    return {
        "n_tiles": len(keys),
        "beta_cape_z": beta_z, "beta_cape_phys_per_Jkg": beta_phys,
        "cape_trend_top_tercile_mean_Jkg_per_yr": float(np.mean(ct[top])),
        "D_pred_per_yr": d_pred,
        "H21_P1a": {"D_obs_per_yr": d_obs, "p": p_a, "pass": p1a_pass},
        "H21_P1b": {"ci95": list(ci), "covers_pred": ci_covers_pred,
                    "covers_zero": ci_covers_zero},
        "VERDICT_P1": v,
    }


def tests_p4(rng: np.random.Generator) -> dict:
    armc_out_orig = armc.OUT
    table = armc.monthly_table_km()
    anomP = {r: armc.b17.anomalies(s, "P") for r, s in table.items()
             if r in EXCESS_REGIONS}
    idx_monthly = {n: load_index_monthly(n) for n in INDEX_NAMES}

    def annual(series: dict, years) -> dict[int, float]:
        out = {}
        for y in years:
            vals = [series[(y, m)] for m in range(1, 13) if (y, m) in series]
            if len(vals) >= 10:
                out[y] = float(np.mean(vals))
        return out

    def smooth5(d: dict[int, float]) -> dict[int, float]:
        ys = sorted(d)
        out = {}
        for y in ys:
            w = [d[t] for t in range(y - 2, y + 3) if t in d]
            if len(w) == 5:
                out[y] = float(np.mean(w))
        return out

    def t_stat(p_ann: dict[int, float], idx_ann_all: dict) -> tuple[float, str]:
        best, best_name = 0.0, ""
        for name, ia in idx_ann_all.items():
            ys = sorted(set(p_ann) & set(ia))
            if len(ys) < 20:
                continue
            r = abs(spearmanr([p_ann[y] for y in ys],
                              [ia[y] for y in ys]).statistic)
            if r > best:
                best, best_name = r, name
        return best, best_name

    res = {"named_indices": list(INDEX_NAMES)}
    per_region = {}
    for reg in EXCESS_REGIONS:
        p_ann = annual(anomP[reg], CONF_YEARS)
        idx_ann = {}
        for n in INDEX_NAMES:
            ia = annual(idx_monthly[n], CONF_YEARS)
            idx_ann[n + "_annual"] = ia
            idx_ann[n + "_5yr"] = smooth5(ia)
        t_obs, best = t_stat(p_ann, idx_ann)
        null = np.empty(N_PERM)
        for i in range(N_PERM):
            s = int(rng.integers(20, 71)) * int(rng.choice([-1, 1]))
            idx_sh = {}
            for n in INDEX_NAMES:
                mo = idx_monthly[n]
                keys = sorted(mo)
                vals = [mo[k] for k in keys]
                shifted = {k: vals[(j + s) % len(vals)]
                           for j, k in enumerate(keys)}
                ia = annual(shifted, CONF_YEARS)
                idx_sh[n + "_annual"] = ia
                idx_sh[n + "_5yr"] = smooth5(ia)
            null[i] = t_stat(p_ann, idx_sh)[0]
        p = float((1 + np.sum(null >= t_obs)) / (N_PERM + 1))
        per_region[reg] = {"T": t_obs, "best_index": best, "p": p,
                           "null_q95": float(np.quantile(null, 0.95)),
                           "significant": bool(p < 0.05)}
    res["per_region"] = per_region
    sig = [r for r, v in per_region.items() if v["significant"]]
    best_idx = {per_region[r]["best_index"].rsplit("_", 1)[0] for r in sig}
    if len(sig) >= 2 and len(best_idx) == 1:
        v = "P4_SUPPORTED"
    elif len(sig) == 0:
        v = "P4_FALSIFIED"
    else:
        v = "P4_INCONCLUSIVE"
    res["VERDICT_P4"] = v
    armc.OUT = armc_out_orig
    return res


def stage_tests() -> dict:
    rng = np.random.default_rng(SEED)
    res = {"seed": SEED, "protocol": "docs/PROTOCOL_PHASE21_QT_TESTS.md"}
    res["ARM_P5"] = tests_p5(rng)
    res["ARM_P1"] = tests_p1(rng)
    res["ARM_P4"] = tests_p4(rng)
    res["PHASE21_VERDICT"] = [res["ARM_P5"]["VERDICT_P5"],
                              res["ARM_P1"]["VERDICT_P1"],
                              res["ARM_P4"]["VERDICT_P4"]]
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["p5-series", "p1-series", "tests"],
                    default="tests")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 4) - 3))
    args = ap.parse_args()
    if args.stage == "p5-series":
        stage_p5(args.workers)
    elif args.stage == "p1-series":
        stage_p1(args.workers)
    else:
        res = stage_tests()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "summary.json").write_text(json.dumps(res, indent=2),
                                          encoding="utf-8")
        brief = {
            "P5": {k: res["ARM_P5"][k] for k in ("H21_P5a", "H21_P5b",
                                                 "slope_loglog", "VERDICT_P5")},
            "P1": {k: res["ARM_P1"][k] for k in
                   ("D_pred_per_yr", "H21_P1a", "H21_P1b", "VERDICT_P1")},
            "P4": res["ARM_P4"],
            "PHASE21_VERDICT": res["PHASE21_VERDICT"],
        }
        print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
