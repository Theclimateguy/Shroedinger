#!/usr/bin/env python3
"""Phase-23: drift physicality (H23a/b) and the stratigraphic hypothesis
(H23c/d).

Preregistered protocol: docs/PROTOCOL_PHASE23_DRIFT_STRATIGRAPHY.md
(frozen 2026-08-18).

Stages: --stage esyn-series | tests-ondisk | model-series | tests-model
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import theilslopes

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
    )
    from clean_experiments.experiment_B20_p_geography import (
        ELLS_FINE,
        TILE_COLS,
        TILE_ROWS,
        fine_features,
        tile_slices,
    )
    from clean_experiments.experiment_B22_qt_round2 import (
        SEASON_MONTHS,
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
    )
    from experiment_B20_p_geography import (  # type: ignore
        ELLS_FINE,
        TILE_COLS,
        TILE_ROWS,
        fine_features,
        tile_slices,
    )
    from experiment_B22_qt_round2 import (  # type: ignore
        SEASON_MONTHS,
    )
    from experiment_B2_scale_irreversibility import (  # type: ignore
        compute_vorticity,
    )

SEED = 20260818
OUT = _HERE / "results" / "experiment_B23_drift_stratigraphy"
B22_RES = _HERE / "results" / "experiment_B22_qt_round2"
B17_DATA = Path("data/b17daily")
B21_DATA = Path("data/b21")
B23_DATA = Path("data/b23hrmip")

EXCESS_REGIONS = ("R1_WPWP", "R3_AMAZ", "R5_SPCZ")
EPOCH_A = range(1989, 1999)          # pre-transition decade
EPOCH_B = range(1999, 2009)          # post-transition decade
MODEL_EPOCH_A = range(1979, 1987)
MODEL_EPOCH_B = range(2007, 2015)
N_PERM = 999
N_BOOT_SLOPE = 199
N_BOOT = 9999
MIN_SEASON_DAYS = 60


# --------------------------------------------------------------------------
# E_syn tile series from b17daily (daily-00Z within-season wind variance)
# --------------------------------------------------------------------------

def _esyn_one(path: Path) -> str:
    os.environ["OMP_NUM_THREADS"] = "1"
    import xarray as xr
    tag = path.stem.replace("era5_wind850daily_", "")
    cached = OUT / f"esyn_{tag}.json"
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
    tiles: dict = {}
    for ri, rs in enumerate(rows):
        for ci, cs in enumerate(cols):
            entry = {}
            for sname, mos in SEASON_MONTHS.items():
                sel = np.isin(months[: u.shape[0]], mos)
                if sel.sum() >= MIN_SEASON_DAYS:
                    uu = u[sel][:, rs, cs]
                    vv = v[sel][:, rs, cs]
                    entry[sname] = float(
                        (np.var(uu, axis=0) + np.var(vv, axis=0)).mean())
            tiles[f"r{ri}c{ci}"] = entry
    OUT.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps({"tag": tag, "region": region,
                                  "year": int(year), "tiles": tiles}),
                      encoding="utf-8")
    return f"DONE {tag}"


def stage_esyn(workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    files = sorted(B17_DATA.glob("era5_wind850daily_*.nc"))
    todo = [f for f in files
            if not (OUT / f"esyn_{f.stem.replace('era5_wind850daily_', '')}.json"
                    ).exists()]
    print(f"{len(todo)} to process", flush=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        for i, msg in enumerate(pool.imap_unordered(_esyn_one, todo)):
            if not msg.startswith("DONE") or (i + 1) % 100 == 0:
                print(f"[{i+1}/{len(todo)}] {msg}", flush=True)


# --------------------------------------------------------------------------
# assembly helpers (ERA5 side, from B22/B21/B23 caches)
# --------------------------------------------------------------------------

def load_series() -> dict[tuple, dict]:
    """key = (region, tid, season) -> {years, P_fix, cape, esyn, slope}."""
    import xarray as xr
    series: dict[tuple, dict] = {}
    for f in sorted(B22_RES.glob("p1s_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for tid, t in d["tiles"].items():
            for sname in SEASON_MONTHS:
                if sname not in t:
                    continue
                key = (d["region"], tid, sname)
                e = series.setdefault(key, {"P": {}, "F": {}, "esyn": {},
                                            "lat_bounds": t["lat_bounds"],
                                            "lon_bounds": t["lon_bounds"]})
                e["P"][d["year"]] = t[sname]["P_fine_raw"]
                e["F"][d["year"]] = [t[sname]["F_fine"][k] for k in
                                     ("logvar_0", "logvar_1", "logvar_2",
                                      "logvar_3", "slope")]
    for f in sorted(OUT.glob("esyn_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        for tid, t in d["tiles"].items():
            for sname, val in t.items():
                key = (d["region"], tid, sname)
                if key in series:
                    series[key]["esyn"][d["year"]] = val
    # seasonal CAPE
    for region in REGIONS:
        nc = B21_DATA / f"era5_cape_monthly_longrecord_{region}.nc"
        ds = xr.open_dataset(nc)
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
            cape = {}
            for y in sorted(set(int(x) for x in yrs)):
                s = sub[(yrs == y) & sel_m]
                if len(s) == 3:
                    cape[y] = float(s.mean())
            e["cape"] = cape
    # spectrum-fixed P
    for key, e in series.items():
        years = sorted(set(e["P"]) & set(e["F"]))
        yv = np.asarray([e["P"][y] for y in years])
        Fm = np.asarray([e["F"][y] for y in years])
        A = np.column_stack([np.ones(len(yv)), Fm])
        coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
        resid = yv - A @ coef
        e["P_fix"] = dict(zip(years, resid.tolist()))
        e["slope"] = {y: e["F"][y][4] for y in years}
    return series


def step_stat(vals: dict[int, float], ea, eb) -> float | None:
    a = np.asarray([vals[y] for y in ea if y in vals])
    b = np.asarray([vals[y] for y in eb if y in vals])
    if len(a) < 7 or len(b) < 7:
        return None
    resid = np.concatenate([a - a.mean(), b - b.mean()])
    sd = float(np.std(resid, ddof=2))
    if sd < 1e-15:
        return None
    return float(abs(b.mean() - a.mean()) / sd)


def snr_stat(vals: dict[int, float], rng) -> float | None:
    years = np.asarray(sorted(vals))
    if len(years) < 30:
        return None
    x = np.asarray([vals[y] for y in years])
    sl = theilslopes(x, years)[0]
    boots = np.empty(N_BOOT_SLOPE)
    n = len(years)
    for i in range(N_BOOT_SLOPE):
        idx = np.sort(rng.integers(0, n, n))
        yy, xx = years[idx], x[idx]
        if len(set(yy.tolist())) < 5:
            boots[i] = np.nan
            continue
        boots[i] = theilslopes(xx, yy)[0]
    se = float(np.nanstd(boots))
    if se < 1e-18:
        return None
    return float(abs(sl) / se)


def tests_ondisk() -> dict:
    rng = np.random.default_rng(SEED)
    series = load_series()
    res: dict = {}

    # ---- H23c: stratigraphic step at the 1998/99 IPO transition ---------
    fam = [k for k in series if k[0] in EXCESS_REGIONS]
    wins = {"CAPE": 0, "Esyn": 0, "slope": 0}
    n_used = 0
    details = []
    for k in sorted(fam):
        e = series[k]
        s_p = step_stat(e["P_fix"], EPOCH_A, EPOCH_B)
        s_c = step_stat(e["cape"], EPOCH_A, EPOCH_B)
        s_e = step_stat(e["esyn"], EPOCH_A, EPOCH_B)
        s_s = step_stat(e["slope"], EPOCH_A, EPOCH_B)
        if None in (s_p, s_c, s_e, s_s):
            continue
        n_used += 1
        wins["CAPE"] += int(s_p > s_c)
        wins["Esyn"] += int(s_p > s_e)
        wins["slope"] += int(s_p > s_s)
        details.append({"key": list(k), "S_P": s_p, "S_CAPE": s_c,
                        "S_Esyn": s_e, "S_slope": s_s})
    fracs = {x: wins[x] / n_used for x in wins}
    null_q95 = {}
    for x in wins:
        perm = np.empty(N_PERM)
        for i in range(N_PERM):
            perm[i] = np.mean(rng.random(n_used) < 0.5)
        null_q95[x] = float(np.quantile(perm, 0.95))
    n_beat = sum(1 for x in wins if fracs[x] > null_q95[x])
    verdict = ("STRATIGRAPHY_SUPPORTED" if n_beat == 3
               else "STRATIGRAPHY_FAILED" if n_beat == 0
               else "STRATIGRAPHY_MIXED")
    res["H23c"] = {"n_tile_season_types": n_used, "win_fracs": fracs,
                   "null_q95": null_q95, "n_beat_of_3": n_beat,
                   "VERDICT": verdict, "details": details}

    # ---- H23d: trend-sensitivity ranking --------------------------------
    snr = {"P": [], "CAPE": [], "Esyn": [], "slope": []}
    for k in sorted(series):
        e = series[k]
        vals = {"P": snr_stat(e["P_fix"], rng), "CAPE": snr_stat(e["cape"], rng),
                "Esyn": snr_stat(e["esyn"], rng),
                "slope": snr_stat(e["slope"], rng)}
        if None in vals.values():
            continue
        for x, v in vals.items():
            snr[x].append(v)
    med = {x: float(np.median(v)) for x, v in snr.items()}
    comp = {}
    for x in ("CAPE", "Esyn", "slope"):
        diffs = np.asarray(snr["P"]) - np.asarray(snr[x])
        perm = np.empty(N_PERM)
        for i in range(N_PERM):
            signs = rng.choice([-1.0, 1.0], len(diffs))
            perm[i] = np.median(diffs * signs)
        obs = float(np.median(diffs))
        comp[x] = {"median_diff": obs,
                   "p": float((1 + np.sum(perm >= obs)) / (N_PERM + 1))}
    first = med["P"] == max(med.values())
    all_sig = all(c["p"] < 0.05 and c["median_diff"] > 0 for c in comp.values())
    res["H23d"] = {"n_units": len(snr["P"]), "median_snr": med,
                   "pairwise_vs_P": comp,
                   "VERDICT": ("SENSITIVITY_SUPPORTED" if first and all_sig
                               else "SENSITIVITY_RANKED_ONLY")}
    return res


# --------------------------------------------------------------------------
# model side (H23a)
# --------------------------------------------------------------------------

def _model_series_one(args_t) -> str:
    year, = args_t
    os.environ["OMP_NUM_THREADS"] = "1"
    import xarray as xr
    cached = OUT / f"model_{year}.json"
    if cached.exists():
        return str(year)
    fu = B23_DATA / f"ua850_{year}.nc"
    fv = B23_DATA / f"va850_{year}.nc"
    du = xr.open_dataset(fu)
    dv = xr.open_dataset(fv)
    lat = np.asarray(du["lat"].values, dtype=float)
    lon = np.asarray(du["lon"].values, dtype=float)
    u = np.asarray(du["ua"].squeeze().values, dtype=np.float32)
    v = np.asarray(dv["va"].squeeze().values, dtype=np.float32)
    tname = "time"
    months = np.asarray(du[tname].dt.month.values, dtype=int)
    du.close()
    dv.close()
    out: dict = {"year": int(year), "regions": {}}
    for region in REGIONS:
        la, lo, uu, vv = km_crop(lat, lon, u, v, region, 1.0)
        rows = tile_slices(uu.shape[1], TILE_ROWS)
        cols = tile_slices(uu.shape[2], TILE_COLS)
        tiles = {}
        for ri, rs in enumerate(rows):
            for ci, cs in enumerate(cols):
                la2, lo2 = la[rs], lo[cs]
                u2, v2 = uu[:, rs, cs], vv[:, rs, cs]
                dx_km, dy_km = _grid_geometry(la2, lo2)
                mask = _interior_mask(u2.shape[1], u2.shape[2],
                                      ELLS_FINE[-1], dx_km, dy_km)
                om = compute_vorticity(u2, v2, la2, lo2)
                rho = envelope_rho_profile_ells(om, dx_km, dy_km, mask,
                                                ELLS_FINE)
                comp = rho.mean(axis=1)
                entry = {}
                for sname, mos in SEASON_MONTHS.items():
                    sel = np.isin(months[: len(comp)], mos)
                    if sel.sum() >= MIN_SEASON_DAYS:
                        entry[sname] = {
                            "P_fine_raw": float(np.median(comp[sel])),
                            "F_fine": fine_features(om[sel], dx_km, dy_km,
                                                    mask)}
                tiles[f"r{ri}c{ci}"] = entry
        out["regions"][region] = tiles
    OUT.mkdir(parents=True, exist_ok=True)
    cached.write_text(json.dumps(out), encoding="utf-8")
    return f"DONE {year}"


def stage_model_series(workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    years = [y for y in list(MODEL_EPOCH_A) + list(MODEL_EPOCH_B)
             if (B23_DATA / f"ua850_{y}.nc").exists()
             and (B23_DATA / f"va850_{y}.nc").exists()]
    todo = [(y,) for y in years if not (OUT / f"model_{y}.json").exists()]
    print(f"{len(todo)} model years to process", flush=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(min(workers, 4)) as pool:
        for msg in pool.imap_unordered(_model_series_one, todo):
            print(msg, flush=True)


def tests_model() -> dict:
    rng = np.random.default_rng(SEED + 1)
    b22 = json.loads((B22_RES / "summary.json").read_text(encoding="utf-8"))

    # rebuild the frozen B22 selection (top CAPE-trend tercile keys)
    series = load_series()
    keys = []
    ct = []
    for k in sorted(series):
        e = series[k]
        cy = e["cape"]
        yrs = np.asarray(sorted(cy))
        if len(yrs) < 30:
            continue
        keys.append(k)
        ct.append(float(theilslopes(np.asarray([cy[y] for y in yrs]),
                                    yrs)[0]))
    ct = np.asarray(ct)
    thresh = np.quantile(ct, 2.0 / 3.0)
    sel_keys = [k for k, c in zip(keys, ct) if c >= thresh]

    # model spectrum-fixed seasonal series on those keys
    model: dict[tuple, dict[int, float]] = {}
    feats: dict[tuple, dict[int, list]] = {}
    for f in sorted(OUT.glob("model_*.json")):
        d = json.loads(f.read_text(encoding="utf-8"))
        y = d["year"]
        for region, tiles in d["regions"].items():
            for tid, t in tiles.items():
                for sname, e in t.items():
                    key = (region, tid, sname)
                    model.setdefault(key, {})[y] = e["P_fine_raw"]
                    feats.setdefault(key, {})[y] = [
                        e["F_fine"][kk] for kk in
                        ("logvar_0", "logvar_1", "logvar_2", "logvar_3",
                         "slope")]

    contrasts = []
    regions_used = []
    for k in sel_keys:
        if k not in model:
            continue
        years = sorted(model[k])
        yv = np.asarray([model[k][y] for y in years])
        Fm = np.asarray([feats[k][y] for y in years])
        A = np.column_stack([np.ones(len(yv)), Fm])
        coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
        resid = dict(zip(years, (yv - A @ coef).tolist()))
        a = [resid[y] for y in MODEL_EPOCH_A if y in resid]
        b = [resid[y] for y in MODEL_EPOCH_B if y in resid]
        if len(a) < 6 or len(b) < 6:
            continue
        span = float(np.mean(list(MODEL_EPOCH_B)) - np.mean(list(MODEL_EPOCH_A)))
        contrasts.append((np.mean(b) - np.mean(a)) / span)   # per yr
        regions_used.append(k[0])
    contrasts = np.asarray(contrasts)
    regions_used = np.asarray(regions_used)
    e_model = float(np.mean(contrasts))

    perm = np.empty(N_PERM)
    for i in range(N_PERM):
        cc = contrasts.copy()
        for reg in np.unique(regions_used):
            m = regions_used == reg
            cc[m] = cc[m] * rng.choice([-1.0, 1.0], m.sum())
        perm[i] = float(np.mean(cc))
    p = float((1 + np.sum(perm <= e_model)) / (N_PERM + 1))

    reg_list = sorted(set(regions_used))
    idx = {g: np.where(regions_used == g)[0] for g in reg_list}
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.choice(reg_list, size=len(reg_list), replace=True)
        boots[i] = float(np.mean(contrasts[np.concatenate([idx[g] for g in pick])]))
    ci = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))

    d_era5 = b22["ARM_P1s"]["H22_P1a"]["D_obs_per_yr"]
    ci_era5 = b22["ARM_P1s"]["H22_P1b"]["ci95"]
    overlap = not (ci[1] < ci_era5[0] or ci[0] > ci_era5[1])
    sig_neg = bool(e_model < 0 and p < 0.05)
    if sig_neg and overlap:
        v = "DRIFT_PHYSICAL"
    elif sig_neg:
        v = "DRIFT_PART_PHYSICAL"
    else:
        v = "DRIFT_NOT_CONFIRMED_IN_MODEL"
    return {"n_selected_units": len(contrasts),
            "E_model_per_yr": e_model, "p": p, "ci95": list(ci),
            "E_era5_per_yr": d_era5, "ci95_era5": ci_era5,
            "ci_overlap_with_era5": overlap,
            "physical_share_model_over_era5": e_model / d_era5,
            "VERDICT_H23a": v}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["esyn-series", "tests-ondisk",
                                        "model-series", "tests-model"],
                    default="tests-ondisk")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 4) - 3))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.stage == "esyn-series":
        stage_esyn(args.workers)
    elif args.stage == "model-series":
        stage_model_series(args.workers)
    elif args.stage == "tests-ondisk":
        res = tests_ondisk()
        (OUT / "summary_ondisk.json").write_text(json.dumps(res, indent=2),
                                                 encoding="utf-8")
        brief = {"H23c": {k: res["H23c"][k] for k in
                          ("n_tile_season_types", "win_fracs", "null_q95",
                           "VERDICT")},
                 "H23d": {k: res["H23d"][k] for k in
                          ("median_snr", "pairwise_vs_P", "VERDICT")}}
        print(json.dumps(brief, indent=2))
    else:
        res = tests_model()
        (OUT / "summary_model.json").write_text(json.dumps(res, indent=2),
                                                encoding="utf-8")
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
