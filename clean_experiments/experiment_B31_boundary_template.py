#!/usr/bin/env python3
"""Phase 31: the boundary template of inter-level coupling.

Frozen protocol: docs/PROTOCOL_PHASE31_BOUNDARY_TEMPLATE.md (2026-09-30).
Stages: --stage tiles --year 2023|2024 --season JFM|JAS | --stage tests
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from experiment_B1_true_flux_baselines import _grid_geometry, _interior_mask  # noqa: E402
from experiment_B20_armB_global_map import tile_grid  # noqa: E402
from experiment_B20_p_geography import ELLS_FINE  # noqa: E402
from experiment_B2_scale_irreversibility import (  # noqa: E402
    compute_vorticity, gaussian_bar_grouped,
)

PROTOCOL = Path("docs/PROTOCOL_PHASE31_BOUNDARY_TEMPLATE.md")
OUT = _HERE / "results" / "experiment_B31_boundary_template"
GLOB = Path("data/b20global")
SP_NC = Path("data/xp_physics/_x_era5_monthly_surface_pbl/data_stream-moda_stepType-avgua.nc")
MONTHS = {"JFM": ("01", "02", "03"), "JAS": ("07", "08", "09")}
BLOCK_DAYS = 5
EPS = 1e-12
PAIRS = {"b1|b2": (0, 1), "b2|b3": (1, 2), "b1|b3": (0, 2)}
N_ROT = 999
SEED = 20260930
_G: dict[str, object] = {}


def envelopes(w, dx, dy):
    """Log-envelopes of bands 50-100, 100-200, 200-400 km."""
    bars = [gaussian_bar_grouped(w, ell / np.sqrt(12.0), dx, dy) for ell in ELLS_FINE]
    return [np.log(gaussian_bar_grouped(np.abs(bars[i] - bars[i + 1]),
                                        ELLS_FINE[i + 1] / np.sqrt(12.0), dx, dy) + EPS)
            for i in range(3)]


def _tile_worker(t):
    os.environ["OMP_NUM_THREADS"] = "1"
    u, v, lat, lon, block = (_G[k] for k in ("u", "v", "lat", "lon", "block"))
    dlon = float(lon[1] - lon[0])
    iy = np.where((lat >= t["lat0"] - 1e-9) & (lat <= t["lat1"] + 1e-9))[0]
    i0 = int(np.searchsorted(lon, t["lon0"] - 1e-9))
    n_cols = max(4, int(round((t["lon1"] - t["lon0"]) / dlon)))
    ix = np.arange(i0, i0 + n_cols) % len(lon)
    la = lat[iy]
    step = dlon / np.cos(np.deg2rad(t["lat_c"]))            # isotropic grid
    xo = np.arange(n_cols) * dlon
    xn = np.arange(int(np.floor((n_cols - 1) * dlon / step + 1e-9)) + 1) * step
    j0 = np.clip(np.searchsorted(xo, xn, side="right") - 1, 0, n_cols - 2)
    wg = ((xn - xo[j0]) / dlon).astype(np.float32)

    def cut(a):
        b = a[:, iy][:, :, ix].astype(np.float32)
        return b[:, :, j0] * (1 - wg) + b[:, :, j0 + 1] * wg
    uu, vv = cut(u), cut(v)
    lo = t["lon0"] + xn
    dx, dy = _grid_geometry(la, lo)
    mask = _interior_mask(uu.shape[1], uu.shape[2], ELLS_FINE[-1], dx, dy)
    om = np.nan_to_num(compute_vorticity(uu, vv, la, lo))
    e = [x[:, mask].astype(np.float64) for x in envelopes(om, dx, dy)]
    e = [x - x.mean(axis=1, keepdims=True) for x in e]
    tot = [[float(np.mean(np.mean(e[i] * e[j], axis=1))) for j in range(3)] for i in range(3)]
    ma = [x[block == 0].mean(axis=0) for x in e]
    mb = [x[block == 1].mean(axis=0) for x in e]
    tpl = [[float(0.5 * (np.mean(ma[i] * mb[j]) + np.mean(mb[i] * ma[j])))
            for j in range(3)] for i in range(3)]
    return {"tid": t["tid"], "tot": tot, "tpl": tpl}


def stage_tiles(year, season, workers):
    import xarray as xr
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / f"tiles_{year}_{season}.json"
    if out.exists():
        print("cached", out.name)
        return
    parts = [xr.open_dataset(GLOB / f"era5_wind850_global_{year}{m}.nc") for m in MONTHS[season]]
    ds = xr.concat(parts, dim="valid_time")
    lat = np.asarray(ds["latitude"].values, float)
    lon = np.asarray(ds["longitude"].values, float)
    u = np.asarray(ds["u"].squeeze().values, np.float32)
    v = np.asarray(ds["v"].squeeze().values, np.float32)
    tm = ds["valid_time"].values.astype("datetime64[D]")
    block = (((tm - tm[0]).astype(int) // BLOCK_DAYS) % 2).astype(int)
    if lat[0] > lat[-1]:
        lat, u, v = lat[::-1], u[:, ::-1], v[:, ::-1]
    _G.update(u=np.ascontiguousarray(u), v=np.ascontiguousarray(v), lat=lat, lon=lon, block=block)
    with mp.get_context("fork").Pool(workers) as pool:
        res = pool.map(_tile_worker, tile_grid(), chunksize=8)
    out.write_text(json.dumps({"year": year, "season": season, "n_steps": int(u.shape[0]),
                               "tiles": res}), encoding="utf-8")
    print(f"B31_TILES_{year}_{season}_DONE", len(res), flush=True)


def stage_tests():
    import xarray as xr
    import explore_physics_analysis as A
    from experiment_B25_residual_candidates import tile_mean  # noqa: F401
    fz = A.Frozen()
    la, oc = fz.is_land, fz.is_ocean

    def load(year):
        tot, tpl = [], []
        for s in MONTHS:
            d = {r["tid"]: r for r in json.loads((OUT / f"tiles_{year}_{s}.json").read_text())["tiles"]}
            tot.append(np.array([d[t]["tot"] for t in fz.keep]))
            tpl.append(np.array([d[t]["tpl"] for t in fz.keep]))
        return tot, tpl

    def corr(C, i, j):
        return C[:, i, j] / np.sqrt(np.maximum(C[:, i, i], 1e-30) * np.maximum(C[:, j, j], 1e-30))

    Y = {}
    for year in (2023, 2024):
        tot, tpl = load(year)
        share = np.mean([np.mean([tp[:, i, i] / to[:, i, i] for i in range(3)], axis=0)
                         for to, tp in zip(tot, tpl)], axis=0)
        r = {}
        for pn, (i, j) in PAIRS.items():
            r_tpl = []
            for to, tp in zip(tot, tpl):
                x = corr(tp, i, j)
                ok = (tp[:, i, i] > 0.02 * to[:, i, i]) & (tp[:, j, j] > 0.02 * to[:, j, j])
                r_tpl.append(np.where(ok, x, np.nan))
            r[pn] = {"tot": np.mean([corr(to, i, j) for to in tot], axis=0),
                     "mov": np.mean([corr(to - tp, i, j) for to, tp in zip(tot, tpl)], axis=0),
                     "tpl": np.nanmean(r_tpl, axis=0)}
        Y[year] = {"s": share, "r": r}
    res = {"protocol": str(PROTOCOL),
           "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
           "n_tiles": fz.n, "n_land": int(la.sum()), "n_ocean": int(oc.sum())}

    # H31-1
    s23, s24 = Y[2023]["s"], Y[2024]["s"]
    rho_all, p_all = fz.spearman_rot(s23, s24, n_rot=N_ROT, seed=SEED)
    rho_land = float(spearmanr(s23[la], s24[la]).statistic)
    res["H31_1"] = {"rho_land": rho_land, "rho_all": rho_all, "p_rot_all": p_all,
                    "rho_ocean": float(spearmanr(s23[oc], s24[oc]).statistic),
                    "median_share": {str(y): {"ocean": float(np.median(Y[y]["s"][oc])),
                                              "land": float(np.median(Y[y]["s"][la]))}
                                     for y in (2023, 2024)},
                    "pass": bool(rho_land >= 0.70 and rho_all >= 0.60 and p_all < 0.01)}
    # H31-2
    cls = {"low": la & (s23 < 0.03), "mid": la & (s23 >= 0.03) & (s23 <= 0.15),
           "high": la & (s23 > 0.15)}
    h2 = {"class_n": {k: int(m.sum()) for k, m in cls.items()}, "pairs": {}}
    for pn in PAIRS:
        rt, rm = Y[2024]["r"][pn]["tot"], Y[2024]["r"][pn]["mov"]
        h2["pairs"][pn] = {
            "r_tot": {k: round(float(rt[m].mean()), 4) for k, m in {**cls, "ocean": oc}.items()},
            "r_mov": {k: round(float(rm[m].mean()), 4) for k, m in {**cls, "ocean": oc}.items()},
            "rho_tot_share_land": float(spearmanr(rt[la], s23[la]).statistic),
            "rho_mov_share_land": float(spearmanr(rm[la], s23[la]).statistic)}
    p = h2["pairs"]["b1|b3"]
    a = abs(p["r_tot"]["high"] - p["r_tot"]["low"]) <= 0.02
    b = (p["r_mov"]["low"] - p["r_mov"]["high"]) >= 0.04
    c = (abs(p["rho_tot_share_land"]) <= 0.2) and (p["rho_mov_share_land"] <= -0.30)
    h2.update({"a": bool(a), "b": bool(b), "c": bool(c), "pass": bool(a and b and c)})
    res["H31_2"] = h2
    # H31-3
    t23, t24 = Y[2023]["r"]["b1|b3"]["tot"], Y[2024]["r"]["b1|b3"]["tot"]
    ro, rl = float(spearmanr(t23[oc], t24[oc]).statistic), float(spearmanr(t23[la], t24[la]).statistic)
    res["H31_3"] = {"rho_ocean": ro, "rho_land": rl,
                    "other_pairs": {pn: {"ocean": float(spearmanr(Y[2023]["r"][pn]["tot"][oc], Y[2024]["r"][pn]["tot"][oc]).statistic),
                                         "land": float(spearmanr(Y[2023]["r"][pn]["tot"][la], Y[2024]["r"][pn]["tot"][la]).statistic)}
                                    for pn in ("b1|b2", "b2|b3")},
                    "pass": bool(ro < 0.20 and rl > 0.40)}
    # H31-4
    med = {pn: float(np.nanmedian(Y[2024]["r"][pn]["tpl"][la])) for pn in PAIRS}
    res["H31_4"] = {"median_r_tpl_land": med,
                    "n_defined_land": {pn: int(np.isfinite(Y[2024]["r"][pn]["tpl"][la]).sum()) for pn in PAIRS},
                    "pass": bool(med["b1|b2"] >= 0.70 and med["b2|b3"] >= 0.70 and med["b1|b3"] >= 0.50)}
    # H31-5
    ds = xr.open_dataset(SP_NC)
    sp = ds["sp"].mean("valid_time").values
    glat, glon = ds["latitude"].values, ds["longitude"].values
    ds.close()
    spmin = np.empty(fz.n)
    for k, t in enumerate(fz.tiles):
        a_ = (glat >= t["lat0"]) & (glat <= t["lat1"])
        lo0, lo1 = t["lon0"] % 360.0, t["lon1"] % 360.0
        b_ = (glon >= lo0) & (glon <= lo1) if lo0 <= lo1 else (glon >= lo0) | (glon <= lo1)
        spmin[k] = float(sp[np.ix_(a_, b_)].min())
    clean = la & (spmin >= 85000.0)
    ratio = float(np.median(s24[clean]) / np.median(s24[oc]))
    rho_or = float(spearmanr(s24[clean], fz.raw["orog_std"][clean]).statistic)
    res["H31_5"] = {"n_clean_land": int(clean.sum()), "median_share_clean_land": float(np.median(s24[clean])),
                    "median_share_ocean": float(np.median(s24[oc])), "ratio": ratio,
                    "rho_share_orog_std": rho_or, "pass": bool(ratio >= 3.0 and rho_or >= 0.30)}
    v = ("NEGATIVE" if not res["H31_1"]["pass"] else
         "BOUNDARY_TEMPLATE_CONFIRMED" if (res["H31_2"]["pass"] and res["H31_3"]["pass"])
         else "TEMPLATE_INVARIANT_ONLY" if not res["H31_2"]["pass"] else "PARTIAL(H31-1,H31-2)")
    res["VERDICT"] = v
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez(OUT / "tile_fields.npz", tid=np.array(fz.keep), lat=fz.lat, lon=fz.lon,
             land=fz.land, share_2023=s23, share_2024=s24, sp_min=spmin,
             **{f"r_{k}_{pn.replace('|', '_')}_{y}": Y[y]["r"][pn][k]
                for y in (2023, 2024) for pn in PAIRS for k in ("tot", "mov", "tpl")})
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["tiles", "tests"], default="tiles")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--season", choices=list(MONTHS), default="JFM")
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args()
    if a.stage == "tiles":
        stage_tiles(a.year, a.season, a.workers)
    else:
        stage_tests()
