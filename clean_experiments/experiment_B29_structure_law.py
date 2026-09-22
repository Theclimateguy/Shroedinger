#!/usr/bin/env python3
"""Phase 29: the structure law — tile replicate, free-running model, amplitude
vs intermittency block, estimator reliability.

Frozen spec: docs/PROTOCOL_PHASE29_STRUCTURE_LAW.md (2026-09-22).
Stages: --stage model (new envelope computation on IFS-HR) | tests
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
from scipy.stats import f as f_dist, spearmanr, wilcoxon

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from clean_experiments.experiment_B18_equal_km_regions import REGIONS, km_crop  # noqa: E402
from clean_experiments.experiment_B2_scale_irreversibility import ELLS_KM  # noqa: E402
from clean_experiments.experiment_B20_p_geography import ELLS_FINE  # noqa: E402
from clean_experiments.experiment_B20_armB_global_map import (  # noqa: E402
    ALL_COV, OROG_EXCL_M, SEASONS, SECTOR_DEG, loso_r2, rotate_cov,
    tile_covariates, tile_grid,
)
from clean_experiments.experiment_B27_mechanism import (  # noqa: E402
    ARMB, ELL_C_UNITS, OUT as B27, WINDOWS, _load, _write_atomic, anchored_payload,
)
from clean_experiments.experiment_B28_inheritance_depth import (  # noqa: E402
    categorical_r2, nonadj_pairs, unit_cov,
)

CONFIG_VERSION = "b29_v1"
PROTOCOL = Path("docs/PROTOCOL_PHASE29_STRUCTURE_LAW.md")
SEED = 20260922
OUT = _HERE / "results" / "experiment_B29_structure_law"
A2 = _HERE / "results" / "experiment_A2_intermittency" / "tiles"
B13 = _HERE / "results" / "experiment_B13_free_running"
MODEL_DIR = Path("data/b13hrmip/ECMWF-IFS-HR")
MODEL_WINDOWS = {"W15_2014JFM": ("201401", "201402", "201403"),
                 "W16_2014JAS": ("201407", "201408", "201409")}
LEVEL_PA = 85000.0
INT_COLS = ("P_cascade", "INT_sig2", "INT_flat", "INT_mu")
N_ROT = 999
ROT_MIN_DEG = 30.0
N_BOOT = 1999
BLOCK_DEG = 30.0


# --------------------------------------------------------------------------
# model stage
# --------------------------------------------------------------------------

_G: dict[str, object] = {}


def _model_worker(job: tuple[str, str]) -> str:
    os.environ["OMP_NUM_THREADS"] = "1"
    region, window = job
    tag = f"{region}__{window}"
    path = OUT / "model" / f"{tag}.json"
    if path.exists():
        return "cached"
    U, V, lat, lon = _G["U"], _G["V"], _G["lat"], _G["lon"]
    # global 0..360 grid -> longitudes re-centred on the region so that the
    # signed region centre used by km_crop is inside the array
    from clean_experiments.experiment_B18_equal_km_regions import region_center
    _, lon_c = region_center(region)
    lon_rc = ((lon - lon_c + 180.0) % 360.0) + lon_c - 180.0
    order = np.argsort(lon_rc)
    la, lo, uu, vv = km_crop(lat, lon_rc[order], U[:, :, order], V[:, :, order], region, 1.0)
    payload = anchored_payload(uu, vv, la, lo, ELLS_KM, ELL_C_UNITS, f"b29_model_{tag}")
    payload.update(tag=tag, region=region, window=window, source="IFS-HR-freerun")
    _write_atomic(path, payload)
    return "done"


def stage_model(workers: int) -> None:
    import xarray as xr
    for window, months in MODEL_WINDOWS.items():
        todo = [(r, window) for r in REGIONS if not (OUT / "model" / f"{r}__{window}.json").exists()]
        if not todo:
            continue
        cubes = {}
        for var in ("ua", "va"):
            parts = []
            for m in months:
                hit = sorted(MODEL_DIR.glob(f"{var}_*_{m}01*.nc"))[0]
                ds = xr.open_dataset(hit)
                sel = ds[var].sel(plev=LEVEL_PA)
                parts.append((np.asarray(sel.values, dtype=np.float32),
                              np.asarray(ds["lat"].values, dtype=float),
                              np.asarray(ds["lon"].values, dtype=float)))
                ds.close()
            cubes[var] = (np.concatenate([p[0] for p in parts], axis=0), parts[0][1], parts[0][2])
        U, lat, lon = cubes["ua"]
        V = cubes["va"][0]
        lon = (lon + 360.0) % 360.0
        order = np.argsort(lon)
        lon, U, V = lon[order], U[:, :, order], V[:, :, order]
        if lat[0] > lat[-1]:
            lat, U, V = lat[::-1], U[:, ::-1, :], V[:, ::-1, :]
        print(f"{window}: model cube {U.shape}", flush=True)
        _G.update(U=U, V=V, lat=lat, lon=lon)
        ctx = mp.get_context("fork")
        with ctx.Pool(workers) as pool:
            for i, _ in enumerate(pool.imap_unordered(_model_worker, todo)):
                print(f"[{window} {i+1}/{len(todo)}]", flush=True)


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def _dR2(d: dict, n_env: int) -> float:
    c = unit_cov(d, n_env)
    return categorical_r2(c, lambda p: p[1]) - categorical_r2(c, lambda p: p[1] - p[0])


def _kappa(d: dict) -> float:
    c = [d["anch"][f"cov_0_{j}"][0] for j in range(2, 6)]
    if min(c) <= 0:
        return np.nan
    return float(np.polyfit(np.log(ELLS_KM[2:6]), np.log(c), 1)[0])


def _perm_p(x, y, rng, n=9999) -> float:
    obs = spearmanr(x, y).statistic
    return float((1 + sum(spearmanr(x, rng.permutation(y)).statistic >= obs for _ in range(n))) / (n + 1))


def _bw_F(groups: list[np.ndarray]) -> dict:
    groups = [g[np.isfinite(g)] for g in groups]
    groups = [g for g in groups if len(g) > 1]
    k, n = len(groups), sum(len(g) for g in groups)
    gm = np.mean(np.concatenate(groups))
    ssb = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
    ssw = sum(float(np.sum((g - g.mean()) ** 2)) for g in groups)
    F = (ssb / (k - 1)) / (ssw / (n - k))
    return {"F": float(F), "p": float(f_dist.sf(F, k - 1, n - k))}


def tests_units_model(rng) -> dict:
    n_env = len(ELLS_KM)
    res: dict = {}
    units = {(r, w): _load(B27 / "units" / f"{r}__{w}.json") for r in REGIONS for w in WINDOWS}
    small = {(r, w): _load(B27 / "units_x0.7" / f"{r}__{w}.json") for r in REGIONS for w in WINDOWS}
    A_reg = np.asarray([np.mean([units[(r, w)]["anch"]["cov_0_2"][0] for w in WINDOWS]) for r in REGIONS])
    A_small = np.asarray([np.mean([small[(r, w)]["anch"]["cov_0_2"][0] for w in WINDOWS]) for r in REGIONS])
    res["A_regions_era5"] = dict(zip(REGIONS, map(float, A_reg)))
    res["A_small_over_big"] = float(np.median(A_small / A_reg))

    # H29-5 kappa
    kap = {k: _kappa(d) for k, d in units.items()}
    kap_reg = np.asarray([np.nanmean([kap[(r, w)] for w in WINDOWS]) for r in REGIONS])
    jfm = [w for w in WINDOWS if "JFM" in w]
    jas = [w for w in WINDOWS if "JAS" in w]
    rhos = []
    for a in jfm:
        for b in jas:
            xa = np.asarray([kap[(r, a)] for r in REGIONS]); xb = np.asarray([kap[(r, b)] for r in REGIONS])
            ok = np.isfinite(xa) & np.isfinite(xb)
            rhos.append(spearmanr(xa[ok], xb[ok]).statistic)
    P_reg = []
    for r in REGIONS:
        vals = []
        for w in WINDOWS:
            f = _HERE / "results" / "experiment_A3_robustness" / "units" / f"{r}__{w}.json"
            if f.exists():
                vals.append(json.loads(f.read_text())["P_anch_mean"])
        P_reg.append(np.mean(vals))
    P_reg = np.asarray(P_reg)
    cov_reg = {}
    covmeta = json.loads((ARMB.parent / "experiment_B20_p_geography" / "summary.json").read_text()) \
        if (ARMB.parent / "experiment_B20_p_geography" / "summary.json").exists() else {}
    res["H29_5_kappa"] = {"region_kappa": dict(zip(REGIONS, map(float, kap_reg))),
                          "n_undefined_units": int(sum(not np.isfinite(v) for v in kap.values())),
                          "cross_season_rho": float(np.mean(rhos)),
                          "between_within": _bw_F([np.asarray([kap[(r, w)] for w in WINDOWS]) for r in REGIONS]),
                          "rho_kappa_A": float(spearmanr(kap_reg, A_reg).statistic),
                          "rho_kappa_P": float(spearmanr(kap_reg, P_reg).statistic),
                          "rho_A_P": float(spearmanr(A_reg, P_reg).statistic)}

    # model
    model = {(r, w): _load(OUT / "model" / f"{r}__{w}.json")
             for r in REGIONS for w in MODEL_WINDOWS if (OUT / "model" / f"{r}__{w}.json").exists()}
    res["n_model_units"] = len(model)
    if model:
        mw = list(MODEL_WINDOWS)
        dr2 = np.asarray([np.mean([_dR2(model[(r, w)], n_env) for w in mw if (r, w) in model]) for r in REGIONS])
        p = float(wilcoxon(dr2 - 0.0, alternative="greater").pvalue)
        A_mod = np.asarray([np.mean([model[(r, w)]["anch"]["cov_0_2"][0] for w in mw if (r, w) in model]) for r in REGIONS])
        kap_mod = np.asarray([np.nanmean([_kappa(model[(r, w)]) for w in mw if (r, w) in model]) for r in REGIONS])
        # C29-1
        adj = [f"rho_{i}_{i+1}" for i in range(n_env - 1)]
        P_mod = [np.mean([np.mean([model[(r, w)]["anch"][n][0] for n in adj]) for w in mw if (r, w) in model]) for r in REGIONS]
        P_b13 = []
        for r in REGIONS:
            vals = []
            for w in mw:
                f = B13 / f"model_{r}__{w}.json"
                if f.exists():
                    rec = json.loads(f.read_text())
                    vals.append(rec["P"])
            P_b13.append(np.mean(vals) if vals else np.nan)
        okb = np.isfinite(P_b13)
        c1 = float(spearmanr(np.asarray(P_mod)[okb], np.asarray(P_b13)[okb]).statistic) if okb.sum() >= 5 else np.nan
        res["C29_1"] = {"rho": c1, "n": int(okb.sum()), "pass": bool(np.isfinite(c1) and c1 >= 0.8)}
        res["H29_2"] = {"region_dR2": dict(zip(REGIONS, map(float, dr2))),
                        "median_dR2": float(np.median(dr2)), "p_wilcoxon_gt": p,
                        "region_A_model": dict(zip(REGIONS, map(float, A_mod))),
                        "rho_A_model_vs_era5": float(spearmanr(A_mod, A_reg).statistic),
                        "rho_kappa_model_vs_era5": float(spearmanr(kap_mod, kap_reg, nan_policy="omit").statistic),
                        "median_kappa_model": float(np.nanmedian(kap_mod)),
                        "pass": bool(np.median(dr2) > 0.10 and p < 0.05 and res["C29_1"]["pass"])}
    else:
        res["H29_2"] = {"pass": False, "note": "no model shards"}
    return res, A_reg


def tests_tiles(rng, A_reg_units) -> dict:
    n_env = len(ELLS_FINE)
    grid = tile_grid()
    by_tid = {t["tid"]: t for t in grid}
    sh = {s: {p.stem: _load(p) for p in sorted((B27 / "tiles" / s).glob("t*.json"))} for s in SEASONS}
    a2 = {s: {p.stem: json.loads(p.read_text()) for p in sorted((A2 / s).glob("t*.json"))} for s in SEASONS}
    armb = {s: {r["tid"]: r for r in json.loads((ARMB / f"tiles_{s}2023.json").read_text())["tiles"]} for s in SEASONS}
    common = sorted(set.intersection(*[set(sh[s]) for s in SEASONS], *[set(a2[s]) for s in SEASONS],
                                     *[set(armb[s]) for s in SEASONS]))
    cov = tile_covariates([by_tid[t] for t in common])
    keep = [t for t in common if cov[t]["orog_mean"] <= OROG_EXCL_M]
    tiles = [by_tid[t] for t in keep]
    tid_index = {t: i for i, t in enumerate(keep)}
    res: dict = {"n_kept": len(keep)}

    def sm(fn):
        return np.asarray([np.mean([fn(s, t) for s in SEASONS]) for t in keep])

    C = {p: sm(lambda s, t, p=p: sh[s][t]["anch"][f"cov_{p[0]}_{p[1]}"][0]) for p in nonadj_pairs(n_env)}
    block = np.asarray([int(by_tid[t]["lon_c"] // BLOCK_DEG) * 10 + int((by_tid[t]["lat_c"] + 60.0) // BLOCK_DEG) for t in keep])
    ub = np.unique(block)
    members = [np.where(block == b)[0] for b in ub]

    def boot_median(x):
        m = np.empty(N_BOOT)
        for k in range(N_BOOT):
            pick = rng.integers(0, len(ub), size=len(ub))
            m[k] = np.median(x[np.concatenate([members[i] for i in pick])])
        return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]

    # H29-1
    same_coarse = np.abs(C[(0, 3)] - C[(1, 3)])
    same_sep = np.abs(C[(0, 2)] - C[(1, 3)])
    d = same_sep - same_coarse
    ratio = same_coarse / np.maximum(same_sep, 1e-12)
    ci_d = boot_median(d)
    ci_r = boot_median(ratio)
    floor03 = sm(lambda s, t: sh[s][t]["floor"]["cov_0_3"])
    eke = sm(lambda s, t: armb[s][t]["eke_syn"])
    cape = np.asarray([cov[t]["cape_mean"] for t in keep])
    res["H29_1"] = {"median_d": float(np.median(d)), "d_ci95": ci_d,
                    "frac_d_positive": float(np.mean(d > 0)),
                    "median_ratio": float(np.median(ratio)), "ratio_ci95": ci_r,
                    "median_C02": float(np.median(C[(0, 2)])), "median_C03": float(np.median(C[(0, 3)])),
                    "median_C13": float(np.median(C[(1, 3)])),
                    "rho_ratio_cape": float(spearmanr(ratio, cape).statistic),
                    "rho_ratio_eke": float(spearmanr(ratio, eke).statistic),
                    "pass": bool(ci_d[0] > 0 and np.median(ratio) <= 0.5)}
    res["C29_2"] = {"frac_same_coarse_below_floor": float(np.mean(same_coarse < floor03)),
                    "floor_limited": bool(np.mean(same_coarse < floor03) > 0.5)}

    # H29-3
    A = C[(0, 2)]
    P = sm(lambda s, t: armb[s][t]["P_fine_anch_mean"])

    def z(a):
        a = np.asarray(a, float); s = a.std()
        return (a - a.mean()) / (s if s > 1e-12 else 1.0)

    cols = {n: np.asarray([cov[t][n] for t in keep]) for n in ALL_COV if n != "eke_syn"}
    cols["eke_syn"] = eke
    X_cov = np.column_stack([z(cols[n]) for n in ALL_COV])
    X_int = np.column_stack([z(sm(lambda s, t, k=k: a2[s][t][f"{k}_anch_mean"])) for k in INT_COLS])
    X_A = z(A).reshape(-1, 1)
    sector = np.asarray([int(by_tid[t]["lon_c"] // SECTOR_DEG) for t in keep])
    base = np.column_stack([X_cov, X_int])
    r2_base = loso_r2(base, P, sector)
    r2_full = loso_r2(np.column_stack([base, X_A]), P, sector)
    inc_A = r2_full - r2_base
    null = np.empty(N_ROT)
    for k in range(N_ROT):
        off = float(rng.uniform(ROT_MIN_DEG, 360.0 - ROT_MIN_DEG))
        null[k] = loso_r2(np.column_stack([base, rotate_cov(X_A, tiles, off, tid_index)]), P, sector) - r2_base
    p_inc = float((1 + np.sum(null >= inc_A)) / (N_ROT + 1))
    r2_covA = loso_r2(np.column_stack([X_cov, X_A]), P, sector)
    inc_block = r2_full - r2_covA
    res["H29_3"] = {"r2_base_cov_plus_block": float(r2_base), "r2_plus_A": float(r2_full),
                    "inc_A": float(inc_A), "p_rot_inc_A": p_inc, "null_q95": float(np.quantile(null, 0.95)),
                    "r2_cov": float(loso_r2(X_cov, P, sector)),
                    "r2_cov_plus_A": float(r2_covA), "inc_block_over_cov_A": float(inc_block),
                    "r2_A_alone": float(loso_r2(X_A, P, sector)),
                    "r2_block_alone": float(loso_r2(X_int, P, sector)),
                    "rho_A_INTsig2": float(spearmanr(A, X_int[:, 1]).statistic),
                    "rho_A_P": float(spearmanr(A, P).statistic),
                    "pass": bool(inc_A >= 0.03 and p_inc < 0.05)}

    # H29-4 reliability
    def sb(x1, x2):
        r = spearmanr(x1, x2).statistic
        return 2 * r / (1 + r)
    adj = [f"rho_{i}_{i+1}" for i in range(n_env - 1)]
    rel = {}
    for name, fn in (("A", lambda s, t, h: sh[s][t]["anch"]["cov_0_2"][h]),
                     ("P", lambda s, t, h: np.mean([sh[s][t]["anch"][n][h] for n in adj]))):
        per = []
        for s in SEASONS:
            x1 = np.asarray([fn(s, t, 1) for t in keep]); x2 = np.asarray([fn(s, t, 2) for t in keep])
            per.append(sb(x1, x2))
        xs = {s: np.asarray([fn(s, t, 0) for t in keep]) for s in SEASONS}
        rel[name] = {"split_half_SB_by_season": [float(v) for v in per], "split_half_SB_mean": float(np.mean(per)),
                     "cross_season_rho": float(spearmanr(xs["JFM"], xs["JAS"]).statistic)}
    from clean_experiments.experiment_B18_equal_km_regions import region_center
    A_tile_reg = []
    for r in REGIONS:
        la_c, lo_c = region_center(r); lo_c %= 360.0
        vals = [A[k] for k, t in enumerate(keep) if abs(by_tid[t]["lat_c"] - la_c) <= 9.0
                and min(abs(by_tid[t]["lon_c"] - lo_c), 360 - abs(by_tid[t]["lon_c"] - lo_c)) <= 12.0]
        A_tile_reg.append(np.mean(vals) if vals else np.nan)
    ok = np.isfinite(A_tile_reg)
    res["H29_4"] = {**rel, "rho_A_units_vs_tiles_12regions": float(spearmanr(np.asarray(A_reg_units)[ok], np.asarray(A_tile_reg)[ok]).statistic),
                    "pass": bool(rel["A"]["split_half_SB_mean"] >= rel["P"]["split_half_SB_mean"] - 0.05)}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "tile_amplitude.npz", tid=np.asarray(keep), A=A, P=P, ratio=ratio,
             lat=np.asarray([by_tid[t]["lat_c"] for t in keep]), lon=np.asarray([by_tid[t]["lon_c"] for t in keep]))
    return res


def stage_tests() -> dict:
    rng = np.random.default_rng(SEED)
    res: dict = {"protocol": str(PROTOCOL), "version": CONFIG_VERSION, "seed": SEED,
                 "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()}
    res["units_model"], A_reg = tests_units_model(rng)
    res["tiles"] = tests_tiles(rng, A_reg)
    p = {"H29_1": res["tiles"]["H29_1"]["pass"], "H29_2": res["units_model"]["H29_2"]["pass"],
         "H29_3": res["tiles"]["H29_3"]["pass"], "H29_4": res["tiles"]["H29_4"]["pass"]}
    res["primary_pass"] = p
    if p["H29_1"] and p["H29_2"]:
        v = "STRUCTURE_LAW_ROBUST+" + ("AMPLITUDE_NEW_AXIS" if p["H29_3"] else "AMPLITUDE_IS_INTERMITTENCY")
    elif any(p.values()):
        v = "PARTIAL(" + ",".join(k for k, ok in p.items() if ok) + ")"
    else:
        v = "NOT_SUPPORTED"
    res["VERDICT"] = v
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"VERDICT": v, "primary_pass": p}, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["model", "tests"])
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    if a.stage == "model":
        stage_model(a.workers)
    else:
        stage_tests()


if __name__ == "__main__":
    main()
