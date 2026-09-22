#!/usr/bin/env python3
"""Phase 28: the inheritance law C_ij = lambda2 (ln L - ln a_j) and its depth L.

Frozen spec: docs/PROTOCOL_PHASE28_INHERITANCE_DEPTH.md (2026-09-22). Inputs:
Phase-27 shards (units, units_x0.7, tiles) — no new envelope computation here.

Stage: --stage tests
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import f as f_dist, spearmanr, wilcoxon

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from clean_experiments.experiment_B18_equal_km_regions import REGIONS
    from clean_experiments.experiment_B2_scale_irreversibility import ELLS_KM
    from clean_experiments.experiment_B20_p_geography import ELLS_FINE
    from clean_experiments.experiment_B20_armB_global_map import (
        ALL_COV, OROG_EXCL_M, SEASONS, SECTOR_DEG, loso_r2, rotate_cov,
        tile_covariates, tile_grid,
    )
    from clean_experiments.experiment_B27_mechanism import (
        A3, ARMB, OUT as B27, WINDOWS, _load,
    )
except ImportError:
    from experiment_B18_equal_km_regions import REGIONS  # type: ignore
    from experiment_B2_scale_irreversibility import ELLS_KM  # type: ignore
    from experiment_B20_p_geography import ELLS_FINE  # type: ignore
    from experiment_B20_armB_global_map import (  # type: ignore
        ALL_COV, OROG_EXCL_M, SEASONS, SECTOR_DEG, loso_r2, rotate_cov,
        tile_covariates, tile_grid,
    )
    from experiment_B27_mechanism import (  # type: ignore
        A3, ARMB, OUT as B27, WINDOWS, _load,
    )

CONFIG_VERSION = "b28_v1"
PROTOCOL = Path("docs/PROTOCOL_PHASE28_INHERITANCE_DEPTH.md")
SEED = 20260922
OUT = _HERE / "results" / "experiment_B28_inheritance_depth"
SMALL = 0.7
N_PERM = 9999
N_ROT = 999
ROT_MIN_DEG = 30.0


# --------------------------------------------------------------------------
# the law
# --------------------------------------------------------------------------

def nonadj_pairs(n_env: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_env) for j in range(i + 2, n_env)]


def fit_law(cov: dict[tuple[int, int], float], ells: list[float]) -> dict:
    """lambda2, ln L and R2 of C_ij = lambda2 (ln L - ln a_j) on non-adjacent pairs."""
    pairs = [p for p in nonadj_pairs(len(ells)) if p in cov]
    x = np.asarray([-np.log(ells[j]) for _, j in pairs])
    y = np.asarray([cov[p] for p in pairs])
    A = np.column_stack([x, np.ones_like(x)])
    (lam, b0), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ np.array([lam, b0])
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    lnL = b0 / lam if lam > 0 else np.nan
    return {"lambda2": float(lam), "lnL": float(lnL), "R2_lin": float(r2),
            "n_pairs": len(pairs)}


def categorical_r2(cov: dict[tuple[int, int], float], key) -> float:
    pairs = list(cov)
    y = np.asarray([cov[p] for p in pairs])
    groups = {}
    for p, v in zip(pairs, y):
        groups.setdefault(key(p), []).append(v)
    ss_res = sum(float(np.sum((np.asarray(g) - np.mean(g)) ** 2)) for g in groups.values())
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan


def unit_cov(d: dict, n_env: int, h: int = 0) -> dict[tuple[int, int], float]:
    return {(i, j): d["anch"][f"cov_{i}_{j}"][h] for i, j in nonadj_pairs(n_env)}


def unit_floor(d: dict, n_env: int) -> dict[tuple[int, int], float]:
    return {(i, j): d["floor"][f"cov_{i}_{j}"] for i, j in nonadj_pairs(n_env)}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def perm_p_spearman(x, y, rng, n=N_PERM) -> float:
    obs = spearmanr(x, y).statistic
    cnt = 0
    for _ in range(n):
        if spearmanr(x, rng.permutation(y)).statistic >= obs:
            cnt += 1
    return float((1 + cnt) / (n + 1))


def between_within_F(values: dict[str, list[float]]) -> dict:
    groups = [np.asarray(v) for v in values.values() if len(v) > 1]
    k = len(groups)
    n = sum(len(g) for g in groups)
    gm = np.mean(np.concatenate(groups))
    ssb = sum(len(g) * (g.mean() - gm) ** 2 for g in groups)
    ssw = sum(float(np.sum((g - g.mean()) ** 2)) for g in groups)
    F = (ssb / (k - 1)) / (ssw / (n - k))
    return {"F": float(F), "p": float(f_dist.sf(F, k - 1, n - k)),
            "var_ratio_between_within": float((ssb / (k - 1)) / (ssw / (n - k)))}


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def tests_units(rng) -> dict:
    n_env = len(ELLS_KM)
    res: dict = {}
    units = {(r, w): _load(B27 / "units" / f"{r}__{w}.json")
             for r in REGIONS for w in WINDOWS}
    small_dir = B27 / f"units_x{SMALL:g}"
    small = {(r, w): _load(small_dir / f"{r}__{w}.json")
             for r in REGIONS for w in WINDOWS
             if (small_dir / f"{r}__{w}.json").exists()}
    res["n_units"] = len(units)
    res["n_units_small"] = len(small)

    # per-unit fits
    fits, fits_h1, fits_h2, dr2, r2j, r2sep = {}, {}, {}, {}, {}, {}
    for key, d in units.items():
        c = unit_cov(d, n_env)
        fits[key] = fit_law(c, ELLS_KM)
        fits_h1[key] = fit_law(unit_cov(d, n_env, 1), ELLS_KM)
        fits_h2[key] = fit_law(unit_cov(d, n_env, 2), ELLS_KM)
        r2j[key] = categorical_r2(c, lambda p: p[1])
        r2sep[key] = categorical_r2(c, lambda p: p[1] - p[0])
        dr2[key] = r2j[key] - r2sep[key]

    def region_mean(dd):
        return np.asarray([np.mean([dd[(r, w)] for w in WINDOWS]) for r in REGIONS])

    def region_median(dd):
        return np.asarray([np.nanmedian([dd[(r, w)] for w in WINDOWS]) for r in REGIONS])

    # C28-1 floor
    above = []
    for key, d in units.items():
        c, fl = unit_cov(d, n_env), unit_floor(d, n_env)
        sep3 = [(i, j) for i, j in c if j - i >= 3]
        above.append(np.mean([c[p] > fl[p] for p in sep3]))
        floor_ratio = np.median([c[p] / max(fl[p], 1e-12) for p in c])
    res["C28_1"] = {"frac_units_sep3_above_floor_majority": float(np.mean(np.asarray(above) > 0.5)),
                    "median_anchored_over_floor_ratio": float(floor_ratio),
                    "restricted_fit_needed": bool(np.mean(np.asarray(above) > 0.5) < 0.5)}

    # H28-1
    x = region_mean(dr2)
    p1 = float(wilcoxon(x - 0.0, alternative="greater").pvalue)
    neg = [f"{r}__{w}" for (r, w), f in fits.items() if f["lambda2"] <= 0]
    res["H28_1"] = {"region_dR2": dict(zip(REGIONS, map(float, x))),
                    "median_dR2": float(np.median(x)), "p_wilcoxon_gt": p1,
                    "median_R2_j": float(np.median(region_mean(r2j))),
                    "median_R2_sep": float(np.median(region_mean(r2sep))),
                    "median_R2_linear_law": float(np.nanmedian(region_mean({k: f["R2_lin"] for k, f in fits.items()}))),
                    "units_with_nonpositive_lambda2": neg,
                    "pass": bool(np.median(x) > 0.10 and p1 < 0.05)}

    # H28-2
    lnL = {k: f["lnL"] for k, f in fits.items()}
    lam = {k: f["lambda2"] for k, f in fits.items()}
    regL = region_median(lnL)
    regLam = region_median(lam)

    def cross_season_rho(dd):
        jfm = [w for w in WINDOWS if "JFM" in w]
        jas = [w for w in WINDOWS if "JAS" in w]
        rhos, ps = [], []
        for a in jfm:
            for b in jas:
                xa = np.asarray([dd[(r, a)] for r in REGIONS])
                xb = np.asarray([dd[(r, b)] for r in REGIONS])
                ok = np.isfinite(xa) & np.isfinite(xb)
                rhos.append(spearmanr(xa[ok], xb[ok]).statistic)
                ps.append(perm_p_spearman(xa[ok], xb[ok], rng, 1999))
        return float(np.mean(rhos)), rhos, float(np.median(ps))

    rho_L, rhos_L, p_L = cross_season_rho(lnL)
    rho_lam, rhos_lam, p_lam = cross_season_rho(lam)
    bw_L = between_within_F({r: [lnL[(r, w)] for w in WINDOWS if np.isfinite(lnL[(r, w)])]
                             for r in REGIONS})
    bw_lam = between_within_F({r: [lam[(r, w)] for w in WINDOWS] for r in REGIONS})
    h1 = np.asarray([fits_h1[k]["lnL"] for k in units])
    h2 = np.asarray([fits_h2[k]["lnL"] for k in units])
    ok = np.isfinite(h1) & np.isfinite(h2)
    res["H28_2"] = {"region_lnL": dict(zip(REGIONS, map(float, regL))),
                    "region_L_km": dict(zip(REGIONS, [float(np.exp(v)) for v in regL])),
                    "region_lambda2": dict(zip(REGIONS, map(float, regLam))),
                    "cross_season_rho_lnL": rho_L, "cross_season_rhos_lnL": [float(v) for v in rhos_L],
                    "perm_p_lnL": p_L, "between_within_lnL": bw_L,
                    "cross_season_rho_lambda2": rho_lam, "perm_p_lambda2": p_lam,
                    "between_within_lambda2": bw_lam,
                    "splithalf_rho_lnL_units": float(spearmanr(h1[ok], h2[ok]).statistic),
                    "splithalf_rho_lambda2_units": float(spearmanr(
                        [fits_h1[k]["lambda2"] for k in units], [fits_h2[k]["lambda2"] for k in units]).statistic),
                    "pass": bool(rho_L >= 0.60 and p_L < 0.05 and bw_L["p"] < 0.05)}

    # H28-3 small boxes
    if small:
        fits_s = {k: fit_law(unit_cov(d, n_env), ELLS_KM) for k, d in small.items()}
        ratio = {}
        for r in REGIONS:
            Lb = [fits[(r, w)]["lnL"] for w in WINDOWS if (r, w) in fits_s]
            Ls = [fits_s[(r, w)]["lnL"] for w in WINDOWS if (r, w) in fits_s]
            ratio[r] = float(np.exp(np.nanmedian(Ls) - np.nanmedian(Lb)))
        q = float(np.nanmedian(list(ratio.values())))
        lam_ratio = float(np.nanmedian([
            np.median([fits_s[(r, w)]["lambda2"] for w in WINDOWS if (r, w) in fits_s])
            / np.median([fits[(r, w)]["lambda2"] for w in WINDOWS]) for r in REGIONS]))
        # C28-2
        adj = [f"rho_{i}_{i+1}" for i in range(n_env - 1)]
        pb = [np.mean([np.mean([units[(r, w)]["anch"][n][0] for n in adj]) for w in WINDOWS]) for r in REGIONS]
        ps = [np.mean([np.mean([small[(r, w)]["anch"][n][0] for n in adj]) for w in WINDOWS if (r, w) in small])
              for r in REGIONS]
        c2 = float(spearmanr(pb, ps).statistic)
        cov_small = {p: float(np.median([unit_cov(d, n_env)[p] for d in small.values()])) for p in nonadj_pairs(n_env)}
        cov_big = {p: float(np.median([unit_cov(d, n_env)[p] for d in units.values()])) for p in nonadj_pairs(n_env)}
        res["H28_3"] = {"region_L_ratio_small_over_big": ratio, "q": q,
                        "lambda2_ratio": lam_ratio,
                        "C28_2_rho_adjacentP": c2, "C28_2_pass": bool(c2 >= 0.8),
                        "median_anchored_cov_big": {f"{i}_{j}": v for (i, j), v in cov_big.items()},
                        "median_anchored_cov_small": {f"{i}_{j}": v for (i, j), v in cov_small.items()},
                        "pass": bool(q >= 0.85 and c2 >= 0.8)}
    else:
        res["H28_3"] = {"pass": False, "note": "small-box shards missing"}

    # region-level P vs (lambda2, lnL) — reported (underpowered)
    regP = []
    for r in REGIONS:
        vals = []
        for w in WINDOWS:
            f = A3 / "units" / f"{r}__{w}.json"
            if f.exists():
                vals.append(json.loads(f.read_text())["P_anch_mean"])
        regP.append(np.mean(vals))
    regP = np.asarray(regP)
    res["units_P_vs_law_reported"] = {
        "rho_P_lnL": float(spearmanr(regP, regL).statistic),
        "rho_P_lambda2": float(spearmanr(regP, regLam).statistic),
        "rho_lnL_lambda2": float(spearmanr(regL, regLam).statistic)}
    res["unit_fits"] = {f"{r}__{w}": fits[(r, w)] for r in REGIONS for w in WINDOWS}
    return res, regL, regLam


def tests_tiles(rng, regL_units, regLam_units) -> dict:
    n_env = len(ELLS_FINE)
    grid = tile_grid()
    by_tid = {t["tid"]: t for t in grid}
    shards = {s: {p.stem: _load(p) for p in sorted((B27 / "tiles" / s).glob("t*.json"))}
              for s in SEASONS}
    armb = {}
    for s in SEASONS:
        d = json.loads((ARMB / f"tiles_{s}2023.json").read_text(encoding="utf-8"))
        armb[s] = {r["tid"]: r for r in d["tiles"]}
    common = sorted(set.intersection(*[set(shards[s]) for s in SEASONS],
                                     *[set(armb[s]) for s in SEASONS]))
    cov = tile_covariates([by_tid[t] for t in common])
    keep = [t for t in common if cov[t]["orog_mean"] <= OROG_EXCL_M]
    tiles = [by_tid[t] for t in keep]
    tid_index = {t: i for i, t in enumerate(keep)}
    res: dict = {"n_kept": len(keep)}

    lam = np.empty(len(keep)); lnL = np.empty(len(keep)); r2 = np.empty(len(keep))
    for k, t in enumerate(keep):
        c = {p: np.mean([unit_cov(shards[s][t], n_env)[p] for s in SEASONS]) for p in nonadj_pairs(n_env)}
        f = fit_law(c, ELLS_FINE)
        lam[k], lnL[k], r2[k] = f["lambda2"], f["lnL"], f["R2_lin"]
    okL = np.isfinite(lnL)
    res["tile_fit"] = {"frac_lambda2_positive": float(np.mean(lam > 0)),
                       "median_lambda2": float(np.median(lam)),
                       "median_L_km": float(np.exp(np.nanmedian(lnL))),
                       "L_km_q10_q90": [float(np.exp(np.nanquantile(lnL, q))) for q in (0.1, 0.9)]}

    P = np.asarray([np.mean([armb[s][t]["P_fine_anch_mean"] for s in SEASONS]) for t in keep])
    eke = np.asarray([np.mean([armb[s][t]["eke_syn"] for s in SEASONS]) for t in keep])
    cape = np.asarray([cov[t]["cape_mean"] for t in keep])
    sector = np.asarray([int(by_tid[t]["lon_c"] // SECTOR_DEG) for t in keep])

    def z(a):
        a = np.asarray(a, float)
        return (a - a.mean()) / (a.std() if a.std() > 1e-12 else 1.0)

    # undefined lnL: replace by the tile lnL median (declared handling: lambda2 <= 0 -> L undefined)
    lnL_f = np.where(okL, lnL, np.nanmedian(lnL))
    X_law = np.column_stack([z(lam), z(lnL_f)])
    cols = {n: np.asarray([cov[t][n] for t in keep]) for n in ALL_COV if n != "eke_syn"}
    cols["eke_syn"] = eke
    X_cov = np.column_stack([z(cols[n]) for n in ALL_COV])

    def rot_null(Xm, target, n=N_ROT):
        vals = np.empty(n)
        for i in range(n):
            off = float(rng.uniform(ROT_MIN_DEG, 360.0 - ROT_MIN_DEG))
            vals[i] = loso_r2(rotate_cov(Xm, tiles, off, tid_index), target, sector)
        return vals

    r2_law = loso_r2(X_law, P, sector)
    null_law = rot_null(X_law, P)
    p_law = float((1 + np.sum(null_law >= r2_law)) / (N_ROT + 1))
    r2_cov = loso_r2(X_cov, P, sector)
    X_both = np.column_stack([X_cov, X_law])
    r2_both = loso_r2(X_both, P, sector)
    null_inc = rot_null(X_law, P) * 0.0
    # increment null: rotate only the law columns, keep covariates fixed
    for i in range(N_ROT):
        off = float(rng.uniform(ROT_MIN_DEG, 360.0 - ROT_MIN_DEG))
        Xr = np.column_stack([X_cov, rotate_cov(X_law, tiles, off, tid_index)])
        null_inc[i] = loso_r2(Xr, P, sector) - r2_cov
    inc = r2_both - r2_cov
    p_inc = float((1 + np.sum(null_inc >= inc)) / (N_ROT + 1))
    res["H28_4"] = {"loso_r2_law": float(r2_law), "p_rot": p_law,
                    "null_q95": float(np.quantile(null_law, 0.95)),
                    "loso_r2_lambda2_only": float(loso_r2(X_law[:, :1], P, sector)),
                    "loso_r2_lnL_only": float(loso_r2(X_law[:, 1:], P, sector)),
                    "loso_r2_covariates": float(r2_cov),
                    "increment_over_covariates": float(inc), "p_rot_increment": p_inc,
                    "increment_ladder": ("+0.05/p<=0.01" if inc >= 0.05 and p_inc <= 0.01 else
                                         "+0.03/p<0.05" if inc >= 0.03 and p_inc < 0.05 else "none"),
                    "pass": bool(r2_law >= 0.30 and p_law < 0.05)}

    # H28-5 geography (reported)
    def rot_p_rho(x, y):
        obs = spearmanr(x, y).statistic
        xx = x.reshape(-1, 1)
        cnt = 0
        for _ in range(N_ROT):
            off = float(rng.uniform(ROT_MIN_DEG, 360.0 - ROT_MIN_DEG))
            if abs(spearmanr(rotate_cov(xx, tiles, off, tid_index)[:, 0], y).statistic) >= abs(obs):
                cnt += 1
        return float(obs), float((1 + cnt) / (N_ROT + 1))

    geo = {}
    for name, x in (("lnL", lnL_f), ("lambda2", lam)):
        for cn, cv in (("cape", cape), ("eke", eke), ("P", P)):
            geo[f"rho_{name}_{cn}"], geo[f"p_{name}_{cn}"] = rot_p_rho(x, cv)
    # unit-vs-tile co-location over the 12 regions
    from clean_experiments.experiment_B18_equal_km_regions import region_center  # type: ignore
    tileL_reg, tileLam_reg = [], []
    for r in REGIONS:
        la_c, lo_c = region_center(r)
        lo_c %= 360.0
        vals = [(lnL_f[k], lam[k]) for k, t in enumerate(keep)
                if abs(by_tid[t]["lat_c"] - la_c) <= 9.0
                and min(abs(by_tid[t]["lon_c"] - lo_c), 360 - abs(by_tid[t]["lon_c"] - lo_c)) <= 12.0]
        tileL_reg.append(np.mean([v[0] for v in vals]) if vals else np.nan)
        tileLam_reg.append(np.mean([v[1] for v in vals]) if vals else np.nan)
    ok = np.isfinite(tileL_reg)
    geo["rho_units_vs_tiles_lnL_12regions"] = float(spearmanr(np.asarray(regL_units)[ok], np.asarray(tileL_reg)[ok]).statistic)
    geo["rho_units_vs_tiles_lambda2_12regions"] = float(spearmanr(np.asarray(regLam_units)[ok], np.asarray(tileLam_reg)[ok]).statistic)
    geo["n_regions_matched"] = int(ok.sum())
    res["H28_5_geography"] = geo

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "tile_law.npz", tid=np.asarray(keep), lambda2=lam, lnL=lnL, R2=r2, P=P,
             lat=np.asarray([by_tid[t]["lat_c"] for t in keep]),
             lon=np.asarray([by_tid[t]["lon_c"] for t in keep]))
    return res


def stage_tests() -> dict:
    rng = np.random.default_rng(SEED)
    res: dict = {"protocol": str(PROTOCOL), "version": CONFIG_VERSION, "seed": SEED,
                 "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()}
    res["units"], regL, regLam = tests_units(rng)
    res["tiles"] = tests_tiles(rng, regL, regLam)
    passed = {"H28_1": res["units"]["H28_1"]["pass"], "H28_2": res["units"]["H28_2"]["pass"],
              "H28_3": res["units"]["H28_3"]["pass"], "H28_4": res["tiles"]["H28_4"]["pass"]}
    res["primary_pass"] = passed
    if not passed["H28_1"]:
        v = "NOT_SUPPORTED"
    elif all(passed.values()):
        v = "LAW_CONFIRMED"
    elif passed["H28_2"] and passed["H28_4"] and not passed["H28_3"]:
        v = "LAW_DOMAIN_LIMITED"
    else:
        v = "PARTIAL(" + ",".join(k for k, ok in passed.items() if ok) + ")"
    res["VERDICT"] = v
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"VERDICT": v, "primary_pass": passed}, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="tests", choices=["tests"])
    ap.parse_args()
    stage_tests()


if __name__ == "__main__":
    main()
