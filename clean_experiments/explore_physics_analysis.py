#!/usr/bin/env python3
"""EXPLORATORY (not frozen) analysis of the physics probes.

Register: docs/EXPLORATION_PHYSICS_2026-09-29.md
Inputs: results/explore_physics/tiles_*.json (explore_physics_tiles.py) and the
frozen Phase-20 / AUDIT-2b / AUDIT-4 objects (target map, covariates,
intermittency block, residual).

Usage: python clean_experiments/explore_physics_analysis.py --test t0|t1|t2|t3|t4|t5|all
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiment_B20_armB_global_map import (  # noqa: E402
    ALL_COV, OROG_EXCL_M, SEASONS, loso_r2, rotate_cov, tile_covariates,
    tile_grid,
)

XP = _HERE / "results" / "explore_physics"
A2B = _HERE / "results" / "experiment_A2b_splithalf"
ARMB = _HERE / "results" / "experiment_B20_armB_global_map"
INT_COLS = ("P_cascade", "INT_sig2", "INT_flat", "INT_mu")
HALVES = ("odd", "even")
SECTOR_DEG = 60.0
ROT_MIN_DEG = 30.0
SEED = 20260929
N_ROT = 499
STD_HEIGHT_M = {925: 762.0, 850: 1457.0, 700: 3012.0, 600: 4206.0,
                500: 5574.0, 250: 10363.0, 50: 20576.0}


def z(x):
    x = np.asarray(x, float)
    sd = np.nanstd(x)
    return (x - np.nanmean(x)) / (sd if sd > 1e-12 else 1.0)


def ols_predict(X_fit, y_fit, X_pred):
    A = np.column_stack([np.ones(len(X_fit)), X_fit])
    coef, *_ = np.linalg.lstsq(A, y_fit, rcond=None)
    return np.column_stack([np.ones(len(X_pred)), X_pred]) @ coef


class Frozen:
    """The committed objects of Phases 20 / AUDIT-2b / AUDIT-4."""

    def __init__(self) -> None:
        self.grid = {t["tid"]: t for t in tile_grid()}
        sh = {(s, h): {r["tid"]: r for r in json.loads(
            (A2B / f"tiles_{s}2023_{h}.json").read_text(encoding="utf-8"))["tiles"]}
            for s in SEASONS for h in HALVES}
        armb = {s: {r["tid"]: r for r in json.loads(
            (ARMB / f"tiles_{s}2023.json").read_text(encoding="utf-8"))["tiles"]}
            for s in SEASONS}
        common = sorted(set.intersection(*[set(v) for v in sh.values()]))
        cov = tile_covariates([self.grid[t] for t in common])
        self.keep = [t for t in common if cov[t]["orog_mean"] <= OROG_EXCL_M]
        k = self.keep
        self.tiles = [self.grid[t] for t in k]
        self.tid_index = {t: i for i, t in enumerate(k)}
        self.n = len(k)
        self.raw = {n: np.array([cov[t][n] for t in k])
                    for n in cov[k[0]].keys()}
        self.raw["eke_syn"] = np.array(
            [np.mean([armb[s][t]["eke_syn"] for s in SEASONS]) for t in k])
        self.Xz = {n: z(self.raw[n]) for n in ALL_COV}
        self.X_cov = np.column_stack([self.Xz[n] for n in ALL_COV])
        self.lat = np.array([self.grid[t]["lat_c"] for t in k])
        self.lon = np.array([self.grid[t]["lon_c"] for t in k])
        self.sector = np.array([int(self.grid[t]["lon_c"] // SECTOR_DEG)
                                for t in k])
        self.P_season = {s: np.array([armb[s][t]["P_fine_anch_mean"] for t in k])
                         for s in SEASONS}
        self.P = 0.5 * (self.P_season["JFM"] + self.P_season["JAS"])
        self.slope = np.array([np.nanmean([armb[s][t]["F_fine"]["slope"]
                                           for s in SEASONS]) for t in k])

        def Ph(s, h):
            return np.array([sh[(s, h)][t]["P_anch_mean"] for t in k])

        def INT(s, h):
            return np.column_stack(
                [z([sh[(s, h)][t][f"{c}_anch_mean"] for t in k])
                 for c in INT_COLS])

        self.INT = np.mean([INT(s, h) for s in SEASONS for h in HALVES], axis=0)
        self.int_sig2 = np.mean(
            [[sh[(s, h)][t]["INT_sig2_anch_mean"] for t in k]
             for s in SEASONS for h in HALVES], axis=0)
        rc_all, rf_all = [], []
        for s in SEASONS:
            rc, rf = {}, {}
            for h, ho in (("odd", "even"), ("even", "odd")):
                y, yo = Ph(s, h), Ph(s, ho)
                rc[h] = y - ols_predict(self.X_cov, yo, self.X_cov)
                Xf = np.column_stack([self.X_cov, INT(s, ho)])
                rf[h] = y - ols_predict(Xf, yo, Xf)
            rc_all.append(0.5 * (rc["odd"] + rc["even"]))
            rf_all.append(0.5 * (rf["odd"] + rf["even"]))
        self.R_cov = np.mean(rc_all, axis=0)      # after the 8 covariates
        self.R_full = np.mean(rf_all, axis=0)     # after covariates + intermittency
        self.R_cov_season = dict(zip(SEASONS, rc_all))
        self.X_base = np.column_stack([self.X_cov, self.INT])
        self.land = self.raw["land_frac"]
        self.is_land = self.land >= 0.7
        self.is_ocean = self.land <= 0.05

    # -- nulls ------------------------------------------------------------
    def rot(self, x, rng):
        off = float(rng.uniform(ROT_MIN_DEG, 360.0 - ROT_MIN_DEG))
        x2 = np.asarray(x, float)
        if x2.ndim == 1:
            x2 = x2[:, None]
        return rotate_cov(x2, self.tiles, off, self.tid_index)

    def spearman_rot(self, x, y, n_rot=N_ROT, seed=SEED):
        ok = np.isfinite(x) & np.isfinite(y)
        rho = float(spearmanr(x[ok], y[ok]).statistic)
        rng = np.random.default_rng(seed)
        xx = np.where(np.isfinite(x), x, np.nanmedian(x))
        null = np.empty(n_rot)
        for i in range(n_rot):
            xr = self.rot(xx, rng)[:, 0]
            null[i] = spearmanr(xr[ok], y[ok]).statistic
        p = float((1 + np.sum(np.abs(null) >= abs(rho))) / (n_rot + 1))
        return rho, p

    def loso_gain(self, cand, y, base=None, n_rot=N_ROT, seed=SEED):
        """LOSO-sector R2 increment of candidate column(s) over a base."""
        base = self.X_cov if base is None else base
        c = np.asarray(cand, float)
        if c.ndim == 1:
            c = c[:, None]
        c = np.column_stack([z(np.where(np.isfinite(c[:, j]), c[:, j],
                                        np.nanmedian(c[:, j])))
                             for j in range(c.shape[1])])
        r0 = loso_r2(base, y, self.sector)
        r1 = loso_r2(np.column_stack([base, c]), y, self.sector)
        rng = np.random.default_rng(seed)
        null = np.empty(n_rot)
        for i in range(n_rot):
            null[i] = loso_r2(np.column_stack([base, self.rot(c, rng)]), y,
                              self.sector) - r0
        gain = r1 - r0
        p = float((1 + np.sum(null >= gain)) / (n_rot + 1))
        return {"base_r2": round(r0, 4), "gain": round(gain, 4), "p_rot": p,
                "null_q95": round(float(np.quantile(null, 0.95)), 4)}


def load_xp(tag: str, fz: Frozen) -> dict[str, np.ndarray] | None:
    f = XP / f"tiles_{tag}.json"
    if not f.exists():
        return None
    d = {r["tid"]: r for r in json.loads(f.read_text(encoding="utf-8"))["tiles"]}
    out: dict[str, np.ndarray] = {}
    keys = [k for k in d[fz.keep[0]].keys() if k != "tid"]
    for k in keys:
        out[k] = np.array([d[t][k] for t in fz.keep], dtype=float)
    out["P_anch"] = np.mean(out["P_real_all"] - out["P_sur_all"], axis=1)
    out["P_anch_h"] = np.mean(out["P_real_h"] - out["P_sur_h"], axis=2)  # (n,4)
    return out


def season_mean(tag_fmt: str, fz: Frozen, key: str):
    parts = []
    for s in SEASONS:
        x = load_xp(tag_fmt.format(s=s), fz)
        if x is None:
            return None
        parts.append(x[key])
    return np.mean(parts, axis=0)


def belts(fz: Frozen):
    a = np.abs(fz.lat)
    return {"tropics(<20)": a < 20, "subtropics(20-35)": (a >= 20) & (a < 35),
            "midlat(>=35)": a >= 35}


def fmt(x, nd=3):
    return f"{x:+.{nd}f}"


# --------------------------------------------------------------------------
def t0(fz: Frozen) -> dict:
    """Control: reduced-surrogate base run reproduces the committed map."""
    res = {}
    for s in SEASONS:
        x = load_xp(f"era5850_vort_full_{s}", fz)
        if x is None:
            continue
        res[s] = {"spearman_vs_committed": float(
            spearmanr(x["P_anch"], fz.P_season[s]).statistic),
            "max_abs_diff": float(np.max(np.abs(x["P_anch"] - fz.P_season[s]))),
            "mean_diff": float(np.mean(x["P_anch"] - fz.P_season[s]))}
    return res


def t1(fz: Frozen) -> dict:
    """Diurnal cycle of anchored P by local solar time."""
    res = {}
    utc = np.array([0.0, 6.0, 12.0, 18.0])
    groups = {"land": fz.is_land, "ocean": fz.is_ocean}
    for name, b in belts(fz).items():
        groups[f"land_{name}"] = fz.is_land & b
        groups[f"ocean_{name}"] = fz.is_ocean & b
    for s in SEASONS:
        x = load_xp(f"era5850_vort_full_{s}", fz)
        if x is None:
            continue
        ph = x["P_anch_h"]                                   # (n, 4)
        raw_h = np.mean(x["P_real_h"], axis=2)
        sur_h = np.mean(x["P_sur_h"], axis=2)
        anom = ph - ph.mean(axis=1, keepdims=True)
        c1 = 0.5 * np.sum(ph * np.exp(-2j * np.pi * utc / 24.0)[None, :], axis=1)
        amp = np.abs(c1)
        phase_lst = (np.angle(c1) * 24.0 / (2 * np.pi) + fz.lon / 15.0) % 24.0
        lst = (utc[None, :] + fz.lon[:, None] / 15.0) % 24.0
        out = {}
        for g, m in groups.items():
            if m.sum() < 8:
                continue
            bins = np.arange(0, 25, 3.0)
            comp, comp_raw, comp_sur, se = [], [], [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                sel = (lst[m] >= lo) & (lst[m] < hi)
                a = anom[m][sel]
                comp.append(float(a.mean()) if a.size else np.nan)
                se.append(float(a.std() / np.sqrt(max(a.size, 1))) if a.size else np.nan)
                ar = (raw_h - raw_h.mean(axis=1, keepdims=True))[m][sel]
                asr = (sur_h - sur_h.mean(axis=1, keepdims=True))[m][sel]
                comp_raw.append(float(ar.mean()) if ar.size else np.nan)
                comp_sur.append(float(asr.mean()) if asr.size else np.nan)
            # vector-mean of the first harmonic in LST frame
            cl = c1[m] * np.exp(-2j * np.pi * (fz.lon[m] / 15.0) / 24.0)
            vm = np.mean(cl)
            rng = np.random.default_rng(SEED)
            boot = np.array([np.abs(np.mean(rng.choice(cl, cl.size)))
                             for _ in range(999)])
            # null: random phases (no LST locking)
            null = np.array([np.abs(np.mean(np.abs(cl) * np.exp(
                2j * np.pi * rng.uniform(size=cl.size)))) for _ in range(999)])
            out[g] = {
                "n": int(m.sum()),
                "median_tile_amplitude": float(np.median(amp[m])),
                "locked_amplitude": float(np.abs(vm)),
                "locked_amp_ci95": [float(np.quantile(boot, 0.025)),
                                    float(np.quantile(boot, 0.975))],
                "p_locking": float((1 + np.sum(null >= np.abs(vm))) / 1000),
                "lst_of_max_h": float((np.angle(vm) * 24 / (2 * np.pi)) % 24),
                "composite_anom_by_LST_3h": [None if not np.isfinite(c) else round(c, 4) for c in comp],
                "composite_se": [None if not np.isfinite(c) else round(c, 4) for c in se],
                "composite_raw": [None if not np.isfinite(c) else round(c, 4) for c in comp_raw],
                "composite_sur": [None if not np.isfinite(c) else round(c, 4) for c in comp_sur],
                "mean_P": float(ph[m].mean()),
            }
        res[s] = out
    return res


def _sfrac(x, field="vort", bands=(0, 1, 2, 3)):
    v = x[f"{field}_var"][:, bands]
    st = np.sum(x[f"{field}_S_stat"][:, bands] * v, axis=1) / np.sum(v, axis=1)
    di = np.sum(x[f"{field}_S_diur"][:, bands] * v, axis=1) / np.sum(v, axis=1)
    return st, di


def t2(fz: Frozen) -> dict:
    """Quenched forcing: stationary/diurnal variance fractions; transient P."""
    res = {}
    xs = {s: load_xp(f"era5850_vort_full_{s}", fz) for s in SEASONS}
    if any(v is None for v in xs.values()):
        return {"status": "base runs missing"}
    st = {s: _sfrac(xs[s])[0] for s in SEASONS}
    di = {s: _sfrac(xs[s])[1] for s in SEASONS}
    S_stat = 0.5 * (st["JFM"] + st["JAS"])
    S_diur = 0.5 * (di["JFM"] + di["JAS"])
    S_tot = S_stat + S_diur
    per_band = {}
    for b in range(4):
        per_band[f"band{b}"] = {
            "S_stat_median": float(np.median(np.mean(
                [xs[s]["vort_S_stat"][:, b] for s in SEASONS], axis=0))),
            "S_diur_median": float(np.median(np.mean(
                [xs[s]["vort_S_diur"][:, b] for s in SEASONS], axis=0)))}
    res["per_band"] = per_band
    res["S_by_group"] = {
        g: {"S_stat": float(np.median(S_stat[m])), "S_diur": float(np.median(S_diur[m])),
            "n": int(m.sum())}
        for g, m in {"land": fz.is_land, "ocean": fz.is_ocean,
                     "mixed": ~fz.is_land & ~fz.is_ocean}.items()}
    res["season_reliability_S_tot"] = float(spearmanr(
        st["JFM"] + di["JFM"], st["JAS"] + di["JAS"]).statistic)
    res["loadings_S_tot"] = {n: float(spearmanr(S_tot, fz.raw[n]).statistic)
                             for n in ALL_COV}
    for nm, c in (("S_stat", S_stat), ("S_diur", S_diur), ("S_tot", S_tot)):
        rho, p = fz.spearman_rot(c, fz.P)
        rr, pr = fz.spearman_rot(c, fz.R_cov)
        rf, pf = fz.spearman_rot(c, fz.R_full)
        res[nm] = {"rho_P": rho, "p_P": p, "rho_Rcov": rr, "p_Rcov": pr,
                   "rho_Rfull": rf, "p_Rfull": pf,
                   "gain_over_cov": fz.loso_gain(c, fz.P),
                   "gain_over_cov_int": fz.loso_gain(c, fz.P, base=fz.X_base)}
    # cross-season (decorrelated sampling noise)
    res["cross_season_rho_P"] = {
        "S_JFM_vs_P_JAS": float(spearmanr(st["JFM"] + di["JFM"], fz.P_season["JAS"]).statistic),
        "S_JAS_vs_P_JFM": float(spearmanr(st["JAS"] + di["JAS"], fz.P_season["JFM"]).statistic)}
    # transient variant
    xt = {s: load_xp(f"era5850_vort_trans_{s}", fz) for s in SEASONS}
    if all(v is not None for v in xt.values()):
        Pf = np.mean([xs[s]["P_anch"] for s in SEASONS], axis=0)
        Pt = np.mean([xt[s]["P_anch"] for s in SEASONS], axis=0)
        d = Pt - Pf
        raw_f = np.mean([xs[s]["P_real_all"].mean(axis=1) for s in SEASONS], axis=0)
        raw_t = np.mean([xt[s]["P_real_all"].mean(axis=1) for s in SEASONS], axis=0)
        sur_f = np.mean([xs[s]["P_sur_all"].mean(axis=1) for s in SEASONS], axis=0)
        sur_t = np.mean([xt[s]["P_sur_all"].mean(axis=1) for s in SEASONS], axis=0)
        tr = {"mean_delta": float(d.mean()), "median_delta": float(np.median(d)),
              "q05_q95_delta": [float(np.quantile(d, 0.05)), float(np.quantile(d, 0.95))],
              "delta_raw_mean": float((raw_t - raw_f).mean()),
              "delta_sur_mean": float((sur_t - sur_f).mean()),
              "rho_maps": float(spearmanr(Pf, Pt).statistic),
              "rho_delta_S_tot": float(spearmanr(d, S_tot).statistic),
              "delta_by_group": {g: float(d[m].mean()) for g, m in
                                 {"land": fz.is_land, "ocean": fz.is_ocean}.items()},
              "land_ocean_contrast": {}}
        for name, b in belts(fz).items():
            lo, oc = fz.is_land & b, fz.is_ocean & b
            if lo.sum() > 5 and oc.sum() > 5:
                tr["land_ocean_contrast"][name] = {
                    "full": float(Pf[oc].mean() - Pf[lo].mean()),
                    "transient": float(Pt[oc].mean() - Pt[lo].mean())}
        tr["loso_r2_cov_full"] = float(loso_r2(fz.X_cov, Pf, fz.sector))
        tr["loso_r2_cov_trans"] = float(loso_r2(fz.X_cov, Pt, fz.sector))
        tr["var_P_full"] = float(np.var(Pf))
        tr["var_P_trans"] = float(np.var(Pt))
        # does the change project on the frozen residual?
        tr["rho_delta_Rcov"] = float(spearmanr(d, fz.R_cov).statistic)
        tr["rho_delta_Rfull"] = float(spearmanr(d, fz.R_full).statistic)
        res["transient"] = tr
    return res


def _rrot(x, bands=(1, 2, 3)):
    v = np.sum(x["vort_var"][:, bands], axis=1)
    d = np.sum(x["div_var"][:, bands], axis=1)
    return v / (v + d)


def t3(fz: Frozen) -> dict:
    """Balance: rotational share; divergence-field coupling."""
    res = {}
    xs = {s: load_xp(f"era5850_vort_full_{s}", fz) for s in SEASONS}
    if any(v is None for v in xs.values()):
        return {"status": "base runs missing"}
    rr = {s: _rrot(xs[s]) for s in SEASONS}
    R = 0.5 * (rr["JFM"] + rr["JAS"])
    res["R_rot"] = {"median": float(np.median(R)),
                    "q05_q95": [float(np.quantile(R, 0.05)), float(np.quantile(R, 0.95))],
                    "by_band_median": [float(np.median(np.mean(
                        [xs[s]["vort_var"][:, b] / (xs[s]["vort_var"][:, b] + xs[s]["div_var"][:, b])
                         for s in SEASONS], axis=0))) for b in range(4)],
                    "season_reliability": float(spearmanr(rr["JFM"], rr["JAS"]).statistic),
                    "by_group": {g: float(np.median(R[m])) for g, m in
                                 {"land": fz.is_land, "ocean": fz.is_ocean}.items()},
                    "loadings": {n: float(spearmanr(R, fz.raw[n]).statistic) for n in ALL_COV}}
    rho, p = fz.spearman_rot(R, fz.P)
    r1, p1 = fz.spearman_rot(R, fz.R_cov)
    r2, p2 = fz.spearman_rot(R, fz.R_full)
    res["R_rot_vs_P"] = {"rho_P": rho, "p": p, "rho_Rcov": r1, "p_Rcov": p1,
                         "rho_Rfull": r2, "p_Rfull": p2,
                         "cross_season": [float(spearmanr(rr["JFM"], fz.P_season["JAS"]).statistic),
                                          float(spearmanr(rr["JAS"], fz.P_season["JFM"]).statistic)],
                         "alone_loso_r2": float(loso_r2(z(R)[:, None], fz.P, fz.sector)),
                         "gain_over_cov": fz.loso_gain(R, fz.P),
                         "gain_over_cov_int": fz.loso_gain(R, fz.P, base=fz.X_base)}
    # mediation of the CAPE / EKE / land loadings
    med = {}
    for n in ("cape_mean", "eke_syn", "land_frac", "abs_lat"):
        c = fz.Xz[n]
        raw = float(spearmanr(c, fz.P).statistic)
        e_p = fz.P - ols_predict(z(R)[:, None], fz.P, z(R)[:, None])
        e_c = c - ols_predict(z(R)[:, None], c, z(R)[:, None])
        med[n] = {"rho_raw": raw, "rho_given_R_rot": float(spearmanr(e_c, e_p).statistic)}
    res["mediation"] = med
    # within-group (does it survive inside ocean / inside land?)
    res["within"] = {g: float(spearmanr(R[m], fz.P[m]).statistic) for g, m in
                     {"land": fz.is_land, "ocean": fz.is_ocean}.items()}
    for name, b in belts(fz).items():
        res["within"][name] = float(spearmanr(R[b], fz.P[b]).statistic)
    xd = {s: load_xp(f"era5850_div_full_{s}", fz) for s in SEASONS}
    if all(v is not None for v in xd.values()):
        Pd = np.mean([xd[s]["P_anch"] for s in SEASONS], axis=0)
        Xvd = np.mean([np.mean(xd[s]["X_real_all"] - xd[s]["X_sur_all"], axis=1)
                       for s in SEASONS], axis=0)
        Xvd_raw = np.mean([np.mean(xd[s]["X_real_all"], axis=1) for s in SEASONS], axis=0)
        res["P_div"] = {"mean": float(Pd.mean()), "mean_P_vort": float(fz.P.mean()),
                        "rho_maps": float(spearmanr(Pd, fz.P).statistic),
                        "season_reliability": float(spearmanr(
                            xd["JFM"]["P_anch"], xd["JAS"]["P_anch"]).statistic),
                        "by_group": {g: float(Pd[m].mean()) for g, m in
                                     {"land": fz.is_land, "ocean": fz.is_ocean}.items()},
                        "loadings": {n: float(spearmanr(Pd, fz.raw[n]).statistic) for n in ALL_COV},
                        "loso_r2_cov": float(loso_r2(fz.X_cov, Pd, fz.sector))}
        rho, p = fz.spearman_rot(Xvd, fz.P)
        res["X_vort_div"] = {"mean_anch": float(Xvd.mean()), "mean_raw": float(Xvd_raw.mean()),
                             "rho_P": rho, "p": p,
                             "rho_Rfull": float(spearmanr(Xvd, fz.R_full).statistic),
                             "loadings": {n: float(spearmanr(Xvd, fz.raw[n]).statistic) for n in ALL_COV},
                             "gain_over_cov": fz.loso_gain(Xvd, fz.P),
                             "gain_over_cov_int": fz.loso_gain(Xvd, fz.P, base=fz.X_base)}
    return res


def below_ground_frac(fz: Frozen, level: int) -> np.ndarray:
    import xarray as xr
    inv = xr.open_dataset("data/b4cov/era5_invariants_global.nc")
    zz = inv["z"].squeeze().values / 9.80665
    ilat = inv["latitude"].values
    ilon = inv["longitude"].values
    if ilat[0] > ilat[-1]:
        ilat = ilat[::-1]
        zz = zz[::-1]
    out = np.empty(fz.n)
    for i, t in enumerate(fz.tiles):
        iy = np.where((ilat >= t["lat0"] - 1e-9) & (ilat <= t["lat1"] + 1e-9))[0]
        i0 = int(np.searchsorted(ilon, t["lon0"] - 1e-9))
        n_cols = int(round((t["lon1"] - t["lon0"]) / 0.25))
        ix = (np.arange(i0, i0 + n_cols)) % len(ilon)
        out[i] = float(np.mean(zz[np.ix_(iy, ix)] > STD_HEIGHT_M[level]))
    inv.close()
    return out


def t4(fz: Frozen) -> dict:
    """Vertical structure of the map in the free-running model."""
    res = {}
    levels = [925, 850, 700, 600, 500, 250, 50]
    maps, feats = {}, {}
    for L in levels:
        xs = {s: load_xp(f"model{L}_vort_full_{s}", fz) for s in SEASONS}
        if any(v is None for v in xs.values()):
            continue
        maps[L] = np.mean([xs[s]["P_anch"] for s in SEASONS], axis=0)
        feats[L] = {"R_rot": np.mean([_rrot(xs[s]) for s in SEASONS], axis=0),
                    "eke": np.mean([xs[s]["eke_syn"] for s in SEASONS], axis=0),
                    "raw": np.mean([xs[s]["P_real_all"].mean(axis=1) for s in SEASONS], axis=0),
                    "sur": np.mean([xs[s]["P_sur_all"].mean(axis=1) for s in SEASONS], axis=0),
                    "S_tot": np.mean([np.sum(_sfrac(xs[s]), axis=0) for s in SEASONS], axis=0),
                    "season_rel": float(spearmanr(xs["JFM"]["P_anch"], xs["JAS"]["P_anch"]).statistic)}
    if 850 not in maps:
        return {"status": "model 850 missing"}
    bg925 = below_ground_frac(fz, 925)
    clean = bg925 <= 0.05                      # tiles free of terrain at all levels
    res["n_clean_tiles"] = int(clean.sum())
    res["n_clean_land"] = int((clean & fz.is_land).sum())
    res["control_model850_vs_era5"] = float(spearmanr(maps[850], fz.P).statistic)
    for L in maps:
        m = clean if L == 925 else np.ones(fz.n, bool)
        P = maps[L]
        r = {"mean_P": float(P[m].mean()), "sd_P": float(P[m].std()),
             "mean_raw": float(feats[L]["raw"][m].mean()),
             "mean_sur": float(feats[L]["sur"][m].mean()),
             "season_reliability": feats[L]["season_rel"],
             "rho_vs_850": float(spearmanr(P[m], maps[850][m]).statistic),
             "rho_vs_850_clean": float(spearmanr(P[clean], maps[850][clean]).statistic),
             "rho_vs_era5_850": float(spearmanr(P[m], fz.P[m]).statistic),
             "median_R_rot": float(np.median(feats[L]["R_rot"][m])),
             "median_S_tot": float(np.median(feats[L]["S_tot"][m])),
             "rho_P_Rrot": float(spearmanr(P[m], feats[L]["R_rot"][m]).statistic),
             "zonal_share": None, "contrast": {}, "loadings": {}}
        # zonal share of variance
        rows = np.array([t["row"] for t in fz.tiles])
        zm = np.array([P[m & (rows == r_)].mean() if (m & (rows == r_)).any() else np.nan
                       for r_ in rows])
        r["zonal_share"] = float(1.0 - np.nanvar((P - zm)[m]) / np.var(P[m]))
        for name, b in belts(fz).items():
            for tag, mm in (("all", m), ("clean", clean)):
                lo, oc = fz.is_land & b & mm, fz.is_ocean & b & mm
                if lo.sum() > 5 and oc.sum() > 5:
                    r["contrast"][f"{name}|{tag}"] = {
                        "ocean_minus_land": float(P[oc].mean() - P[lo].mean()),
                        "n_land": int(lo.sum()), "n_ocean": int(oc.sum())}
        oc = fz.is_ocean
        a = np.abs(fz.lat)
        r["ocean_midlat_minus_tropics"] = float(P[oc & (a >= 40)].mean() - P[oc & (a < 15)].mean())
        r["ocean_by_abs_lat"] = {f"{int(lo)}-{int(lo)+12}": float(P[oc & (a >= lo) & (a < lo + 12)].mean())
                                 for lo in (0, 12, 24, 36, 48)}
        r["land_by_abs_lat"] = {f"{int(lo)}-{int(lo)+12}": float(P[fz.is_land & m & (a >= lo) & (a < lo + 12)].mean())
                                for lo in (0, 12, 24, 36, 48)
                                if (fz.is_land & m & (a >= lo) & (a < lo + 12)).sum() > 3}
        for n in ALL_COV:
            r["loadings"][n] = float(spearmanr(P[m], fz.raw[n][m]).statistic)
        r["loadings"]["eke_level"] = float(spearmanr(P[m], feats[L]["eke"][m]).statistic)
        r["loso_r2_cov"] = float(loso_r2(fz.X_cov[m], P[m], fz.sector[m]))
        res[str(L)] = r
    Ls = sorted(maps, reverse=True)
    res["level_correlation_matrix"] = {
        str(a): {str(b): round(float(spearmanr(maps[a], maps[b]).statistic), 3)
                 for b in Ls} for a in Ls}
    return res


def t5(fz: Frozen) -> dict:
    """Vortex stretching: cyclonic skewness, conditional coupling."""
    res = {}
    xs = {s: load_xp(f"era5850_vort_full_{s}", fz) for s in SEASONS}
    if any(v is None for v in xs.values()):
        return {"status": "base runs missing"}
    sk = np.mean([xs[s]["vort_skew_cyc"] for s in SEASONS], axis=0)   # (n, 4)
    ku = np.mean([xs[s]["vort_kurt"] for s in SEASONS], axis=0)
    dsk = np.mean([xs[s]["div_skew"] for s in SEASONS], axis=0)
    pc = np.mean([xs[s]["P_cyc"] for s in SEASONS], axis=0)
    pa = np.mean([xs[s]["P_anti"] for s in SEASONS], axis=0)
    d = np.nanmean(pc - pa, axis=1)
    res["skew_cyc_median_by_band"] = [float(np.median(sk[:, b])) for b in range(4)]
    res["kurt_median_by_band"] = [float(np.median(ku[:, b])) for b in range(4)]
    res["div_skew_median_by_band"] = [float(np.median(dsk[:, b])) for b in range(4)]
    skm = sk[:, 1:].mean(axis=1)
    rho, p = fz.spearman_rot(skm, fz.P)
    res["skew_vs_P"] = {"rho": rho, "p": p,
                        "cross_season": [
                            float(spearmanr(np.mean(xs["JFM"]["vort_skew_cyc"][:, 1:], axis=1), fz.P_season["JAS"]).statistic),
                            float(spearmanr(np.mean(xs["JAS"]["vort_skew_cyc"][:, 1:], axis=1), fz.P_season["JFM"]).statistic)],
                        "rho_Rcov": float(spearmanr(skm, fz.R_cov).statistic),
                        "rho_Rfull": float(spearmanr(skm, fz.R_full).statistic),
                        "gain_over_cov": fz.loso_gain(skm, fz.P),
                        "gain_over_cov_int": fz.loso_gain(skm, fz.P, base=fz.X_base),
                        "loadings": {n: float(spearmanr(skm, fz.raw[n]).statistic) for n in ALL_COV}}
    kum = np.log(np.maximum(ku[:, 1:].mean(axis=1) + 3.0, 1e-3))
    rho, p = fz.spearman_rot(kum, fz.P)
    res["logflatness_vs_P"] = {"rho": rho, "p": p,
                               "rho_Rfull": float(spearmanr(kum, fz.R_full).statistic)}
    dskm = dsk[:, 1:].mean(axis=1)
    rho, p = fz.spearman_rot(dskm, fz.P)
    res["div_skew_vs_P"] = {"rho": rho, "p": p,
                            "rho_Rfull": float(spearmanr(dskm, fz.R_full).statistic),
                            "gain_over_cov": fz.loso_gain(dskm, fz.P)}
    ok = np.isfinite(d)
    res["cyc_minus_anti"] = {"n": int(ok.sum()), "mean": float(np.nanmean(d)),
                             "se": float(np.nanstd(d) / np.sqrt(ok.sum())),
                             "frac_positive": float(np.mean(d[ok] > 0)),
                             "by_band": [float(np.nanmean((pc - pa)[:, i])) for i in range(3)],
                             "by_belt": {}, "rho_with_P": float(spearmanr(d[ok], fz.P[ok]).statistic),
                             "rho_with_abs_lat": float(spearmanr(d[ok], np.abs(fz.lat[ok])).statistic)}
    for name, b in belts(fz).items():
        for g, m in (("land", fz.is_land), ("ocean", fz.is_ocean)):
            mm = b & m & ok
            if mm.sum() > 5:
                res["cyc_minus_anti"]["by_belt"][f"{g}_{name}"] = {
                    "mean": float(d[mm].mean()),
                    "se": float(d[mm].std() / np.sqrt(mm.sum())), "n": int(mm.sum())}
    # f-dependence over low-CAPE ocean
    lowc = fz.is_ocean & (fz.raw["cape_mean"] < np.quantile(fz.raw["cape_mean"], 0.5))
    sinl = np.abs(np.sin(np.deg2rad(fz.lat)))
    res["f_dependence_lowCAPE_ocean"] = {
        "n": int(lowc.sum()),
        "rho_P_sinlat": float(spearmanr(fz.P[lowc], sinl[lowc]).statistic),
        "rho_P_eke": float(spearmanr(fz.P[lowc], fz.raw["eke_syn"][lowc]).statistic)}
    e_p = fz.P[lowc] - ols_predict(fz.Xz["eke_syn"][lowc][:, None], fz.P[lowc], fz.Xz["eke_syn"][lowc][:, None])
    e_s = sinl[lowc] - ols_predict(fz.Xz["eke_syn"][lowc][:, None], sinl[lowc], fz.Xz["eke_syn"][lowc][:, None])
    res["f_dependence_lowCAPE_ocean"]["rho_P_sinlat_given_eke"] = float(spearmanr(e_p, e_s).statistic)
    return res


def t4b(fz: Frozen) -> dict:
    """ERA5 500 vs 850 hPa on the 12 regional boxes (reanalysis leg of T4)."""
    import xarray as xr
    f = XP / "regional_levels_500_850.json"
    if not f.exists():
        return {"status": "regional run missing"}
    rows = json.loads(f.read_text(encoding="utf-8"))
    inv = xr.open_dataset("data/b4cov/era5_invariants_global.nc")
    lsm = inv["lsm"].squeeze().values
    zz = inv["z"].squeeze().values / 9.80665
    ilat = inv["latitude"].values
    ilon = inv["longitude"].values
    tiles: dict = {}
    for r in rows:
        key = (r["region"], r["tile"])
        tiles.setdefault(key, {"lat": r["lat_c"], "lon": r["lon_c"] % 360.0,
                               850: [], 500: [], "rr850": [], "rr500": []})
        tiles[key][r["level"]].append(r["P_anch"])
        v = sum(r["vort_var"][1:])
        d = sum(r["div_var"][1:])
        tiles[key][f"rr{r['level']}"].append(v / (v + d))
    keys = sorted(tiles)
    p850 = np.array([np.mean(tiles[k][850]) for k in keys])
    p500 = np.array([np.mean(tiles[k][500]) for k in keys])
    r850 = np.array([np.mean(tiles[k]["rr850"]) for k in keys])
    r500 = np.array([np.mean(tiles[k]["rr500"]) for k in keys])
    lat = np.array([tiles[k]["lat"] for k in keys])
    lon = np.array([tiles[k]["lon"] for k in keys])
    land = np.empty(len(keys))
    orog = np.empty(len(keys))
    for i in range(len(keys)):
        hw = 350.0 / (111.2 * np.cos(np.deg2rad(lat[i])))
        iy = np.where(np.abs(ilat - lat[i]) <= 3.0)[0]
        dl = (ilon - lon[i] + 180.0) % 360.0 - 180.0
        ix = np.where(np.abs(dl) <= hw)[0]
        land[i] = float(np.mean(lsm[np.ix_(iy, ix)]))
        orog[i] = float(np.mean(zz[np.ix_(iy, ix)]))
    inv.close()
    ok = orog <= OROG_EXCL_M
    is_land = (land >= 0.7) & ok
    is_ocean = (land <= 0.05) & ok
    res = {"n_tiles": int(ok.sum()), "n_land": int(is_land.sum()),
           "n_ocean": int(is_ocean.sum()),
           "mean_P": {"850": float(p850[ok].mean()), "500": float(p500[ok].mean())},
           "rho_850_500": float(spearmanr(p850[ok], p500[ok]).statistic),
           "ocean_minus_land": {"850": float(p850[is_ocean].mean() - p850[is_land].mean()),
                                "500": float(p500[is_ocean].mean() - p500[is_land].mean())},
           "median_R_rot": {"850": float(np.median(r850[ok])), "500": float(np.median(r500[ok]))},
           "rho_P_Rrot": {"850": float(spearmanr(p850[ok], r850[ok]).statistic),
                          "500": float(spearmanr(p500[ok], r500[ok]).statistic)},
           "by_region": {}}
    regs = sorted({k[0] for k in keys})
    m850, m500 = [], []
    for rg in regs:
        m = np.array([k[0] == rg for k in keys]) & ok
        if m.sum() == 0:
            continue
        res["by_region"][rg] = {"P850": round(float(p850[m].mean()), 4),
                                "P500": round(float(p500[m].mean()), 4),
                                "land": round(float(land[m].mean()), 2),
                                "n": int(m.sum())}
        m850.append(p850[m].mean())
        m500.append(p500[m].mean())
    res["rho_region_means_850_500"] = float(spearmanr(m850, m500).statistic)
    return res


def t9(fz: Frozen) -> dict:
    """Composition of the active objects (12 regions, Arm-A tiles)."""
    from collections import defaultdict
    from scipy.stats import wilcoxon
    f = XP / "objects_T9.json"
    if not f.exists():
        return {"status": "objects run missing"}
    rows = json.loads(f.read_text(encoding="utf-8"))
    acc = defaultdict(list)
    for r in rows:
        acc[(r["region"], r["tile"])].append(r)
    keys = sorted(acc)

    def col(name):
        return np.array([np.nanmean([x[name] for x in acc[k]]) for k in keys])

    P = col("P")
    eke = col("eke_syn")
    reg = np.array([k[0] for k in keys])
    regs = sorted(set(reg))
    names = {"c_O1": "precipitation", "c_O2": "moisture_front",
             "c_O3": "vapour_transport", "c_O4": "divergence_envelope"}
    res = {"n_tiles": len(keys), "note": "c_R2 is not reported: in-sample rank R2 "
           "on ~10 effective degrees of freedom is inflated",
           "by_region": {}, "tests": {}}
    for rg in sorted(regs, key=lambda r_: -np.mean(P[reg == r_])):
        m = reg == rg
        res["by_region"][rg] = {"P": round(float(P[m].mean()), 3),
                                **{v: round(float(np.nanmean(col(k)[m])), 3)
                                   for k, v in names.items()},
                                "wet_frac": round(float(col("wet_frac")[m].mean()), 3)}

    def rankz(a):
        r_ = np.argsort(np.argsort(a)).astype(float)
        return (r_ - r_.mean()) / r_.std()

    for k, v in names.items():
        x = col(k)
        wr = np.array([spearmanr(x[reg == rg], P[reg == rg]).statistic for rg in regs])
        rx, rp, re = rankz(x), rankz(P), rankz(eke)
        ex = rx - np.polyval(np.polyfit(re, rx, 1), re)
        ep = rp - np.polyval(np.polyfit(re, rp, 1), re)
        rm_x = [float(np.nanmean(x[reg == rg])) for rg in regs]
        rm_p = [float(P[reg == rg].mean()) for rg in regs]
        res["tests"][v] = {
            "rho_all_tiles": float(spearmanr(x, P).statistic),
            "rho_region_means_n12": float(spearmanr(rm_x, rm_p).statistic),
            "within_region_rho": [round(float(w), 2) for w in wr],
            "within_region_median": float(np.median(wr)),
            "within_region_n_positive": int((wr > 0).sum()),
            "within_region_wilcoxon_p": float(wilcoxon(wr).pvalue),
            "rho_given_eke": float(np.corrcoef(ex, ep)[0, 1])}
    return res


def t8(fz: Frozen) -> dict:
    """Toy turbulence: cascade versus independent sources."""
    res = {}
    for f in sorted((XP / "toy2d").glob("toy_*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        C = np.array(r["C_anch_mean"])
        res[r["tag"]] = {
            "share": r["args"]["share"], "model": r["args"].get("model", "2d"),
            "P_anch": round(r["P_anch_mean"], 4),
            "P_anch_sd_tiles": round(r["P_anch_sd_tiles"], 4),
            "P_anch_se": round(r["P_anch_sd_tiles"] / np.sqrt(r["n_tiles"]), 4),
            "energy_slope_k8_32": round(r["energy_slope_k8_32"], 2),
            "vorticity_flatness": round(r["w_flatness"], 2),
            "C_coarse2_col": [round(float(C[0, 2]), 4), round(float(C[1, 2]), 4)],
            "C_coarse3_col": [round(float(C[0, 3]), 4), round(float(C[1, 3]), 4),
                              round(float(C[2, 3]), 4)]}
    return res


def t10(fz: Frozen) -> dict:
    f = XP / "modulator" / "modulator_cases.json"
    if not f.exists():
        return {"status": "modulator run missing"}
    rows = json.loads(f.read_text(encoding="utf-8"))
    return {f"beta{r['beta']}_s2{r['s2']}": {
        "R2_coarse_only": round(r["R2_coarse_only"], 3),
        "R2_separation": round(r["R2_separation"], 3),
        "G_by_coarse_km": {k: round(v, 4) for k, v in r["G_by_coarse_km"].items()},
        "G_exponent": None if not np.isfinite(r["G_exponent"]) else round(r["G_exponent"], 2),
        "P_adjacent": [round(x, 3) for x in r["P_adjacent_anchored"]]}
        for r in rows}


def _rows(fz):
    return np.array([t["row"] for t in fz.tiles])


def nonzonal(fz, x):
    rows = _rows(fz)
    zm = np.array([np.nanmean(x[rows == r]) for r in rows])
    return x - zm


def zonal_share(fz, x):
    return float(1.0 - np.nanvar(nonzonal(fz, x)) / np.nanvar(x))


def corrected(fz, tag="era5850_vort_full_iso_{s}"):
    """Corrected target: band pairs 50-100|100-200 and 100-200|200-400 km on
    the isotropic grid. Returns per-season and season-mean arrays."""
    out = {}
    for s in SEASONS:
        x = load_xp(tag.format(s=s), fz)
        if x is None:
            return None
        a = x["P_real_all"] - x["P_sur_all"]
        out[s] = {"x": x, "pairs": a, "P3": a.mean(axis=1), "Pc": a[:, 1:].mean(axis=1)}
    out["Pc"] = 0.5 * (out["JFM"]["Pc"] + out["JAS"]["Pc"])
    out["P3"] = 0.5 * (out["JFM"]["P3"] + out["JAS"]["P3"])
    out["pairs"] = 0.5 * (out["JFM"]["pairs"] + out["JAS"]["pairs"])
    return out


def tc(fz: Frozen) -> dict:
    """T11 and the corrected map: what survives the grid-anisotropy control."""
    iso = corrected(fz)
    nat = corrected(fz, "era5850_vort_full_{s}")
    if iso is None or nat is None:
        return {"status": "iso or base runs missing"}
    res = {}
    a = np.round(np.abs(fz.lat), 1)
    oc, la = fz.is_ocean, fz.is_land
    tab = []
    for r in sorted(set(a)):
        m = oc & (a == r)
        ml = la & (a == r)
        sur_n = np.mean([nat[s]["x"]["P_sur_all"][m] for s in SEASONS], axis=(0, 1))
        sur_i = np.mean([iso[s]["x"]["P_sur_all"][m] for s in SEASONS], axis=(0, 1))
        raw_n = np.mean([nat[s]["x"]["P_real_all"][m] for s in SEASONS], axis=(0, 1))
        tab.append({"abs_lat": float(r), "n_ocean": int(m.sum()),
                    "P_native_3pairs": round(float(nat["P3"][m].mean()), 4),
                    "P_iso_3pairs": round(float(iso["P3"][m].mean()), 4),
                    "P_native_pairs12": round(float(nat["Pc"][m].mean()), 4),
                    "P_c_iso_pairs12": round(float(iso["Pc"][m].mean()), 4),
                    "P_c_land": (round(float(iso["Pc"][ml].mean()), 4)
                                 if ml.sum() > 2 else None),
                    "anch_by_pair_native": [round(float(v), 3) for v in nat["pairs"][m].mean(axis=0)],
                    "anch_by_pair_iso": [round(float(v), 3) for v in iso["pairs"][m].mean(axis=0)],
                    "raw_by_pair_native": [round(float(v), 3) for v in raw_n],
                    "sur_by_pair_native": [round(float(v), 3) for v in sur_n],
                    "sur_by_pair_iso": [round(float(v), 3) for v in sur_i]})
    res["ocean_by_latitude"] = tab
    al = np.abs(fz.lat)

    def contrast(P):
        return float(P[oc & (al >= 40)].mean() - P[oc & (al < 15)].mean())
    res["ocean_midlat_minus_tropics"] = {
        "native_3pairs": contrast(nat["P3"]), "iso_3pairs": contrast(iso["P3"]),
        "native_pairs12": contrast(nat["Pc"]), "iso_pairs12": contrast(iso["Pc"]),
        "retained_3pairs": contrast(iso["P3"]) / contrast(nat["P3"]),
        "retained_Pc_vs_native3": contrast(iso["Pc"]) / contrast(nat["P3"])}
    res["zonal_share"] = {
        "native_3pairs": zonal_share(fz, nat["P3"]),
        "native_by_pair": [zonal_share(fz, nat["pairs"][:, k]) for k in range(3)],
        "iso_3pairs": zonal_share(fz, iso["P3"]),
        "iso_by_pair": [zonal_share(fz, iso["pairs"][:, k]) for k in range(3)],
        "P_c": zonal_share(fz, iso["Pc"])}
    res["sd"] = {"native_3pairs": float(nat["P3"].std()), "iso_3pairs": float(iso["P3"].std()),
                 "P_c": float(iso["Pc"].std()),
                 "nonzonal_native": float(np.std(nonzonal(fz, nat["P3"]))),
                 "nonzonal_P_c": float(np.std(nonzonal(fz, iso["Pc"])))}
    res["agreement"] = {
        "rho_native3_vs_iso3": float(spearmanr(nat["P3"], iso["P3"]).statistic),
        "rho_native3_vs_Pc": float(spearmanr(nat["P3"], iso["Pc"]).statistic),
        "rho_frozen_vs_Pc": float(spearmanr(fz.P, iso["Pc"]).statistic),
        "rho_nonzonal_native3_vs_iso3": float(spearmanr(
            nonzonal(fz, nat["P3"]), nonzonal(fz, iso["P3"])).statistic),
        "rho_nonzonal_frozen_vs_Pc": float(spearmanr(
            nonzonal(fz, fz.P), nonzonal(fz, iso["Pc"])).statistic)}

    def rel(key, src):
        r = float(spearmanr(src["JFM"][key], src["JAS"][key]).statistic)
        rn = float(spearmanr(nonzonal(fz, src["JFM"][key]),
                             nonzonal(fz, src["JAS"][key])).statistic)
        return {"cross_season_rho": r, "cross_season_rho_nonzonal": rn}
    res["reliability"] = {"native_3pairs": rel("P3", nat), "iso_3pairs": rel("P3", iso),
                          "P_c": rel("Pc", iso)}
    Pc = iso["Pc"]
    res["P_c_groups"] = {}
    for name, b in {**belts(fz), "all": np.ones(fz.n, bool)}.items():
        lo_, oc_ = la & b, oc & b
        res["P_c_groups"][name] = {
            "ocean": float(Pc[oc_].mean()), "land": float(Pc[lo_].mean()),
            "ocean_minus_land": float(Pc[oc_].mean() - Pc[lo_].mean()),
            "se": float(np.sqrt(Pc[oc_].var() / oc_.sum() + Pc[lo_].var() / lo_.sum())),
            "native3_ocean_minus_land": float(nat["P3"][oc_].mean() - nat["P3"][lo_].mean())}
    bg = below_ground_frac(fz, 850)
    res["below_ground_850"] = {
        "rho_Pc_bg": float(spearmanr(Pc, bg).statistic),
        "groups": {k: {"n": int(m.sum()), "mean": float(Pc[m].mean()), "sd": float(Pc[m].std())}
                   for k, m in {"ocean": oc, "land_bg0": la & (bg == 0),
                                "land_bg_0_10": la & (bg > 0) & (bg <= 0.1),
                                "land_bg_gt10": la & (bg > 0.1),
                                "mixed_bg0": ~oc & ~la & (bg == 0),
                                "mixed_bg_gt0": ~oc & ~la & (bg > 0)}.items()}}
    res["loadings"] = {n: {"rho": float(spearmanr(Pc, fz.raw[n]).statistic),
                           "rho_nonzonal": (None if n == "abs_lat" else float(spearmanr(
                               nonzonal(fz, Pc), nonzonal(fz, fz.raw[n])).statistic)),
                           "rho_native3": float(spearmanr(nat["P3"], fz.raw[n]).statistic)}
                       for n in ALL_COV}
    res["loso_r2"] = {"P_c_on_8cov": float(loso_r2(fz.X_cov, Pc, fz.sector)),
                      "native3_on_8cov": float(loso_r2(fz.X_cov, nat["P3"], fz.sector)),
                      "iso3_on_8cov": float(loso_r2(fz.X_cov, iso["P3"], fz.sector))}
    # candidate physical features, computed on the isotropic grid
    feats = {}
    for s in SEASONS:
        x = iso[s]["x"]
        feats[s] = {"R_rot": _rrot(x), "S_stat": _sfrac(x)[0], "S_diur": _sfrac(x)[1],
                    "skew_cyc": x["vort_skew_cyc"][:, 1:].mean(axis=1),
                    "log_flatness": np.log(np.maximum(x["vort_kurt"][:, 1:].mean(axis=1) + 3, 1e-3)),
                    "log_envelope_variance": x["env_var"][:, 1:].mean(axis=1),
                    "cyc_minus_anti": np.nanmean(x["P_cyc"] - x["P_anti"], axis=1),
                    "eke_syn": x["eke_syn"], "mean_speed": x["speed_mean"]}
    res["features"] = {}
    for k in feats["JFM"]:
        f_ = 0.5 * (feats["JFM"][k] + feats["JAS"][k])
        ok = np.isfinite(f_)
        rho, p_ = fz.spearman_rot(np.where(ok, f_, np.nanmedian(f_)), Pc, n_rot=199)
        res["features"][k] = {
            "rho": rho, "p_rot": p_,
            "rho_nonzonal": float(spearmanr(nonzonal(fz, f_)[ok], nonzonal(fz, Pc)[ok]).statistic),
            "rho_ocean": float(spearmanr(f_[ok & oc], Pc[ok & oc]).statistic),
            "rho_land": float(spearmanr(f_[ok & la], Pc[ok & la]).statistic),
            "cross_season": [float(spearmanr(feats["JFM"][k][ok], iso["JAS"]["Pc"][ok]).statistic),
                             float(spearmanr(feats["JAS"][k][ok], iso["JFM"]["Pc"][ok]).statistic)],
            "loso_gain_over_cov": fz.loso_gain(np.where(ok, f_, np.nanmedian(f_)), Pc, n_rot=199)}
    o = np.argsort(Pc)
    def row(i):
        return {"lat": float(fz.lat[i]), "lon": round(float(fz.lon[i]), 1),
                "land": round(float(fz.land[i]), 2), "P_c": round(float(Pc[i]), 3),
                "P_frozen": round(float(fz.P[i]), 3), "bg850": round(float(bg[i]), 2)}
    res["lowest"] = [row(i) for i in o[:15]]
    res["highest"] = [row(i) for i in o[-15:][::-1]]
    np.savez(XP / "corrected_map.npz", tid=np.array(fz.keep), lat=fz.lat, lon=fz.lon,
             P_c=Pc, P_iso3=iso["P3"], P_native3=nat["P3"], P_frozen=fz.P,
             P_c_JFM=iso["JFM"]["Pc"], P_c_JAS=iso["JAS"]["Pc"], land=fz.land, bg850=bg)
    return res


TESTS = {"t0": t0, "t1": t1, "t2": t2, "t3": t3, "t4": t4, "t4b": t4b,
         "t5": t5, "t8": t8, "t9": t9, "t10": t10, "tc": tc}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test", default="all")
    args = ap.parse_args()
    fz = Frozen()
    names = list(TESTS) if args.test == "all" else args.test.split(",")
    out = {}
    for n in names:
        out[n] = TESTS[n](fz)
        (XP / f"analysis_{n}.json").write_text(json.dumps(out[n], indent=1),
                                               encoding="utf-8")
        print(f"===== {n}")
        print(json.dumps(out[n], indent=1))


if __name__ == "__main__":
    main()
