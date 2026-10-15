#!/usr/bin/env python3
"""Phase 33: inheritance fidelity of forecast systems (B33).

Protocol: docs/PROTOCOL_PHASE33_MODEL_FIDELITY.md (frozen 2026-10-10).

For every (system, lead, domain) the anchored V12 profile of the forecast fields
is compared with the anchored profile of the ERA5 fields at the paired valid
times.  For the conditional-mean systems two constructions of the truth field are
added: S1 (uniform spectral match to the model) and S2 (selective damping with the
same spectrum).

Usage:
    python clean_experiments/experiment_B33_model_fidelity.py compute --arm A --workers 10
    python clean_experiments/experiment_B33_model_fidelity.py tests --arm A
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import multiprocessing as mp
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.chdir(_HERE.parent)

import experiment_B2_scale_irreversibility as b2  # noqa: E402
import experiment_B32_vorticity_mask as vm  # noqa: E402
from experiment_B1_true_flux_baselines import _grid_geometry, _interior_mask, phase_randomize  # noqa: E402
from download_b11_wb2 import DOMAINS, LEADS  # noqa: E402

SEED = 20261010
N_SUR = 99
ELLS = list(b2.ELLS_KM)                     # 50 .. 1600 km
N_STEPS = len(ELLS) - 1
STEP_DEG = 0.25
KAPPA_S2 = 1.0
N_GAIN_ITER = 6                             # Deviation 1 (protocol): 6 iterations
DELTA = 0.010                               # protocol threshold, anchored units
HEADLINE_STEPS = (3, 4)                     # Deviation D5: steps 4-5 = bands 200-1600 km only
FROZEN_STEPS = (2, 3, 4)                    # steps 3-5 as frozen 2026-10-10, reported for the record
SECONDARY_STEPS = (1, 2, 3, 4)              # steps 2-5, reported
MISMATCH_COMPONENTS = slice(3, 6)           # Deviation D5: components 200-400, 400-800, 800-1600 km
PRIMARY_LEAD = 120

ROLES = {   # scored conditional-mean systems, negative controls, exploratory (not scored)
    "A": (("graphcast", "pangu", "gencast_mean"), ("hres", "gencast_member"), ()),
    "B": (("graphcast", "pangu", "gencast_mean"), ("hres", "gencast_member"), ()),
    "C": (("graphcast", "pangu", "ifs_ens_mean"), ("hres", "ifs_ens_member"), ("aurora",)),   # Arm C addendum
}
CM_SYSTEMS, NEG_SYSTEMS, EXPLORE_SYSTEMS = ROLES["A"]
ALL_SYSTEMS = CM_SYSTEMS + NEG_SYSTEMS + EXPLORE_SYSTEMS
S_SYSTEMS = CM_SYSTEMS + EXPLORE_SYSTEMS          # systems that get the S1/S2 constructions


def set_roles(arm: str) -> None:
    global CM_SYSTEMS, NEG_SYSTEMS, EXPLORE_SYSTEMS, ALL_SYSTEMS, S_SYSTEMS
    CM_SYSTEMS, NEG_SYSTEMS, EXPLORE_SYSTEMS = ROLES[arm]
    ALL_SYSTEMS = CM_SYSTEMS + NEG_SYSTEMS + EXPLORE_SYSTEMS
    S_SYSTEMS = CM_SYSTEMS + EXPLORE_SYSTEMS

ARMS = {
    "A": {"data": Path("data/b11wb2"),
          "windows": {"W13_2020JFM": ("2020-01-01", "2020-03-31"),
                      "W14_2020JAS": ("2020-07-01", "2020-09-30")}},
    "B": {"data": Path("data/b33wb2"), "stride": 5,                  # Deviation D2
          "windows": {"W15_2020AMJ": ("2020-04-01", "2020-06-30"),
                      "W16_2020OND": ("2020-10-01", "2020-12-31")}},
    "C": {"data": Path("data/b33c_wb2"), "stride": 5,                # Arm C addendum (frozen 2026-10-15)
          "windows": {"W17_2022JFM": ("2022-01-01", "2022-03-31"),
                      "W18_2022AMJ": ("2022-04-01", "2022-06-30"),
                      "W19_2022JAS": ("2022-07-01", "2022-09-30"),
                      "W20_2022OND": ("2022-10-01", "2022-12-31")}},
}
OUT_ROOT = Path("clean_experiments/results/experiment_B33_model_fidelity")
INIT_STRIDE_DAYS = 10


# ---------------------------------------------------------------------------
# geometry and data
# ---------------------------------------------------------------------------

def init_dates(arm: str) -> list[dt.datetime]:
    out = []
    for _w, (a, b) in ARMS[arm]["windows"].items():
        d0, d1 = dt.date.fromisoformat(a), dt.date.fromisoformat(b)
        d = d0
        while d <= d1:
            out.append(dt.datetime(d.year, d.month, d.day))
            d += dt.timedelta(days=ARMS[arm].get("stride", INIT_STRIDE_DAYS))
    return out


def domain_grid(name: str) -> tuple[np.ndarray, np.ndarray]:
    n, w, s, e = DOMAINS[name]
    return (np.arange(n, s - 1e-9, -STEP_DEG), np.arange(w, e + 1e-9, STEP_DEG))


def load_pairs(arm: str, system: str, lead: int, domain: str):
    """Stacked (nt, ny, nx) model u, v and paired ERA5 u, v; list of inits used."""
    data = ARMS[arm]["data"]
    um, vmod, ue, ve, used = [], [], [], [], []
    for d0 in init_dates(arm):
        fm = data / f"{system}_{d0:%Y%m%d%H}_f{lead:03d}.npz"
        fe = data / f"era5_{d0 + dt.timedelta(hours=lead):%Y%m%d%H}.npz"
        if not fm.exists() or not fe.exists():
            continue
        with np.load(fm) as zm, np.load(fe) as ze:
            if domain not in zm or domain not in ze:
                continue
            am, ae = zm[domain], ze[domain]
        if not (np.all(np.isfinite(am)) and np.all(np.isfinite(ae))):
            continue
        um.append(am[0]); vmod.append(am[1]); ue.append(ae[0]); ve.append(ae[1])
        used.append(f"{d0:%Y%m%d%H}")
    if not used:
        return None
    st = lambda x: np.stack(x).astype(np.float32)  # noqa: E731
    return st(um), st(vmod), st(ue), st(ve), used


# ---------------------------------------------------------------------------
# estimator pieces (V12)
# ---------------------------------------------------------------------------

class Geo:
    def __init__(self, lat: np.ndarray, lon: np.ndarray):
        self.lat, self.lon = lat, lon
        self.dx, self.dy = _grid_geometry(lat, lon)
        ny, nx = len(lat), len(lon)
        self.interior = _interior_mask(ny, nx, ELLS[-1], self.dx, self.dy)
        self.below = vm.below_surface_mask(lat, lon)
        self.valid = ~self.below
        self.corr = self.interior & self.valid


def vort(u, v, g: Geo) -> np.ndarray:
    return vm.vorticity(u, v, g.lat, g.lon, spherical=True)


def profile(om: np.ndarray, g: Geo) -> np.ndarray:
    """Median over fields of the V12 rho profile (5 steps)."""
    return np.median(vm.rho_profile(om, g.dx, g.dy, g.corr, g.valid, ELLS), axis=0)


def components(f: np.ndarray, g: Geo, valid: np.ndarray | None) -> list[np.ndarray]:
    """Gaussian-bar ladder components c_0..c_6 of a (nt, ny, nx) field."""
    bars = [vm.bar_nan(f, ell / np.sqrt(12.0), g.dx, g.dy, valid) for ell in ELLS]
    comps = [np.nan_to_num(f) - np.nan_to_num(bars[0])]
    for k in range(1, len(ELLS)):
        comps.append(np.nan_to_num(bars[k - 1]) - np.nan_to_num(bars[k]))
    comps.append(np.nan_to_num(bars[-1]))
    return comps


def component_variances(om: np.ndarray, g: Geo) -> np.ndarray:
    w = np.where(g.valid[None, :, :], om, np.nan).astype(np.float32)
    comps = components(w, g, g.valid)
    return np.asarray([float(np.mean(c[:, g.corr] ** 2)) for c in comps])


def anchored(u, v, g: Geo, tag: str, n_sur: int) -> dict:
    om = vort(u, v, g)
    p_real = profile(om, g)
    cvar = component_variances(om, g)
    seeds = [int(s) for s in np.random.SeedSequence(SEED + zlib.crc32(tag.encode()) % 100000).generate_state(n_sur)]
    sur = []
    for s in seeds:
        rng = np.random.default_rng(s)
        us, vs = phase_randomize(u, v, rng)
        sur.append(profile(vort(us, vs, g), g))
    sur_med = np.median(np.asarray(sur), axis=0)
    return {"P_real": p_real.tolist(), "P_surrogate_median": sur_med.tolist(),
            "P_anchored": (p_real - sur_med).tolist(), "comp_var": cvar.tolist()}


# ---------------------------------------------------------------------------
# constructions S1 / S2
# ---------------------------------------------------------------------------

def construct(ue, ve, target_var: np.ndarray, g: Geo, mode: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Truth fields re-weighted per band to the model's component variances.

    mode 'S1': constant gains only.  mode 'S2': components 0..4 first multiplied by
    exp(kappa * z_{k+1}), z = standardised log envelope of the next-coarser component
    of the truth vorticity, then the gains are fitted.  Returns (u', v', mismatch).
    """
    cu = components(ue, g, None)
    cv = components(ve, g, None)
    if mode == "S2":
        om = vort(ue, ve, g)
        co = components(np.where(g.valid[None], om, np.nan).astype(np.float32), g, g.valid)
        for k in range(0, 5):
            env = np.log(vm.bar_nan(np.abs(co[k + 1]), ELLS[k + 1] / np.sqrt(12.0), g.dx, g.dy, g.valid) + b2.EPS)
            env = np.nan_to_num(env, nan=np.nanmean(env))
            z = (env - env.mean(axis=(1, 2), keepdims=True)) / (env.std(axis=(1, 2), keepdims=True) + 1e-12)
            m = np.exp(KAPPA_S2 * z).astype(np.float32)
            cu[k] = cu[k] * m
            cv[k] = cv[k] * m
    gains = np.ones(len(cu))
    mismatch = np.nan
    for _it in range(N_GAIN_ITER):
        up = sum(gk * c for gk, c in zip(gains, cu)).astype(np.float32)
        vp = sum(gk * c for gk, c in zip(gains, cv)).astype(np.float32)
        cur = component_variances(vort(up, vp, g), g)
        ratio = target_var / np.where(cur < 1e-30, 1e-30, cur)
        gains = gains * np.sqrt(ratio)
    up = sum(gk * c for gk, c in zip(gains, cu)).astype(np.float32)
    vp = sum(gk * c for gk, c in zip(gains, cv)).astype(np.float32)
    cur = component_variances(vort(up, vp, g), g)
    achieved = cur / np.where(target_var < 1e-30, 1e-30, target_var)
    # Deviation 1: the criterion covers components 2-5 (the bands of steps 3-5);
    # component 1 (50-100 km) is leakage-limited and reported separately.
    mismatch = float(np.max(np.abs(np.log(achieved[2:6]))))
    return up, vp, mismatch, achieved


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------

def task(args) -> str:
    arm, system, lead, domain, n_sur, out_dir = args
    os.environ["OMP_NUM_THREADS"] = "1"
    target = out_dir / "sets" / f"{system}_f{lead:03d}_{domain}.json"
    if target.exists():
        return f"cached {target.name}"
    t0 = time.time()
    pairs = load_pairs(arm, system, lead, domain)
    if pairs is None:
        return f"no data {system} f{lead} {domain}"
    um, vmod, ue, ve, used = pairs
    lat, lon = domain_grid(domain)
    g = Geo(lat, lon)
    tag = f"{arm}_{system}_f{lead}_{domain}"
    res = {"arm": arm, "system": system, "lead_h": lead, "domain": domain,
           "n_fields": len(used), "inits": used, "n_surrogates": n_sur,
           "model": anchored(um, vmod, g, tag + "_model", n_sur),
           "truth": anchored(ue, ve, g, tag + "_truth", n_sur)}
    if system in S_SYSTEMS:
        tv = np.asarray(res["model"]["comp_var"])
        for mode in ("S1", "S2"):
            up, vp, mis, achieved = construct(ue, ve, tv, g, mode)
            r = anchored(up, vp, g, tag + "_" + mode, n_sur)
            r["spectral_mismatch"] = mis
            r["comp_var_ratio_to_model"] = [float(x) for x in achieved]
            res[mode] = r
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(res, indent=1), encoding="utf-8")
    return f"ok {target.name} ({len(used)} fields, {time.time() - t0:.0f}s)"


def stage_compute(arm: str, systems: list[str], workers: int, n_sur: int, domains: list[str] | None) -> None:
    out_dir = OUT_ROOT / f"arm{arm}"
    doms = domains or list(DOMAINS)
    jobs = [(arm, s, lead, d, n_sur, out_dir) for s in systems for lead in LEADS for d in doms]
    print(f"arm {arm}: {len(jobs)} tasks, {workers} workers, {n_sur} surrogates", flush=True)
    ctx = mp.get_context("fork")
    done = 0
    with ctx.Pool(workers) as pool:
        for msg in pool.imap_unordered(task, jobs):
            done += 1
            if msg.startswith(("no data", "cached")) and done % 50 != 0:
                continue
            print(f"[{done}/{len(jobs)}] {msg}", flush=True)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def sign_flip_p(x: np.ndarray, n: int = 9999, seed: int = SEED) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    obs = x.mean()
    flips = rng.choice([-1.0, 1.0], size=(n, x.size))
    null = (flips * x[None, :]).mean(axis=1)
    return float((np.sum(null >= obs) + 1) / (n + 1))


def loso_r2(y: np.ndarray, X: np.ndarray) -> float:
    n = len(y)
    pred = np.empty(n)
    Xd = np.column_stack([np.ones(n), X])
    for i in range(n):
        keep = np.arange(n) != i
        beta, *_ = np.linalg.lstsq(Xd[keep], y[keep], rcond=None)
        pred[i] = Xd[i] @ beta
    ss = np.sum((y - y.mean()) ** 2)
    return float(1.0 - np.sum((y - pred) ** 2) / ss) if ss > 0 else float("nan")


def head(p: list[float], steps=None) -> float:
    steps = HEADLINE_STEPS if steps is None else steps
    return float(np.mean([p[i] for i in steps]))


def stage_tests(arm: str, suffix: str = "") -> dict:
    out_dir = OUT_ROOT / f"arm{arm}"
    sets = {}
    for f in sorted((out_dir / "sets").glob("*.json")):
        r = json.loads(f.read_text(encoding="utf-8"))
        sets[(r["system"], r["lead_h"], r["domain"])] = r
    domains = sorted({k[2] for k in sets})
    summary: dict = {"arm": arm, "protocol": "docs/PROTOCOL_PHASE33_MODEL_FIDELITY.md",
                     "seed": SEED, "delta": DELTA, "n_domains": len(domains), "systems": {}}

    def vec(system, lead, key, fn):
        return np.asarray([fn(sets[(system, lead, d)][key]) if (system, lead, d) in sets and key in sets[(system, lead, d)]
                           else np.nan for d in domains])

    for system in ALL_SYSTEMS:
        for lead in LEADS:
            if not any(k[0] == system and k[1] == lead for k in sets):
                continue
            pm = vec(system, lead, "model", lambda r: head(r["P_anchored"]))
            pt = vec(system, lead, "truth", lambda r: head(r["P_anchored"]))
            D = pm - pt
            rec = {"n": int(np.sum(np.isfinite(D))),
                   "n_fields_median": float(np.nanmedian(vec(system, lead, "n_fields", lambda x: x))),
                   "P_model_median": float(np.nanmedian(pm)), "P_truth_median": float(np.nanmedian(pt)),
                   "ratio_of_medians": float(np.nanmedian(pm) / np.nanmedian(pt)) if np.nanmedian(pt) != 0 else float("nan"),
                   "D_median": float(np.nanmedian(D)), "D_mean": float(np.nanmean(D)),
                   "D_frac_positive": float(np.nanmean(D > 0)), "D_p_signflip": sign_flip_p(D),
                   "D_frozen35_median": float(np.nanmedian(vec(system, lead, "model", lambda r: head(r["P_anchored"], FROZEN_STEPS))
                                                           - vec(system, lead, "truth", lambda r: head(r["P_anchored"], FROZEN_STEPS)))),
                   "D_steps25_median": float(np.nanmedian(vec(system, lead, "model", lambda r: head(r["P_anchored"], SECONDARY_STEPS))
                                                          - vec(system, lead, "truth", lambda r: head(r["P_anchored"], SECONDARY_STEPS)))),
                   "D_per_step_median": [float(np.nanmedian(vec(system, lead, "model", lambda r, i=i: r["P_anchored"][i])
                                                            - vec(system, lead, "truth", lambda r, i=i: r["P_anchored"][i])))
                                         for i in range(N_STEPS)],
                   "D_raw_median": float(np.nanmedian(vec(system, lead, "model", lambda r: head(r["P_real"]))
                                                      - vec(system, lead, "truth", lambda r: head(r["P_real"])))),
                   "D_by_domain": {d: float(x) for d, x in zip(domains, D)}}
            # deficits on the ladder
            cvm = np.asarray([sets[(system, lead, d)]["model"]["comp_var"] if (system, lead, d) in sets else [np.nan] * 7 for d in domains])
            cvt = np.asarray([sets[(system, lead, d)]["truth"]["comp_var"] if (system, lead, d) in sets else [np.nan] * 7 for d in domains])
            deficit = 1.0 - cvm / cvt
            rec["deficit_median_per_component"] = [float(x) for x in np.nanmedian(deficit, axis=0)]
            d34 = np.nanmean(deficit[:, 3:5], axis=1)
            rec["deficit_34_median"] = float(np.nanmedian(d34))
            rec["deficit_34_frac_positive"] = float(np.nanmean(d34 > 0))
            ok = np.isfinite(D) & np.isfinite(d34)
            if ok.sum() >= 10:
                rec["rho_D_vs_deficit34"] = float(spearmanr(D[ok], d34[ok]).correlation)
                rec["loso_r2_D_on_deficit15"] = loso_r2(D[ok], deficit[ok][:, 1:6])
            for mode in ("S1", "S2"):
                if system in S_SYSTEMS and any(mode in sets[k] for k in sets if k[0] == system and k[1] == lead):
                    ps = vec(system, lead, mode, lambda r: head(r["P_anchored"]))
                    Ds = ps - pt
                    rec[mode] = {"D_median": float(np.nanmedian(Ds)), "D_mean": float(np.nanmean(Ds)),
                                 "D_p_signflip": sign_flip_p(Ds),
                                 "D_per_step_median": [float(np.nanmedian(vec(system, lead, mode, lambda r, i=i: r["P_anchored"][i])
                                                                          - vec(system, lead, "truth", lambda r, i=i: r["P_anchored"][i])))
                                                       for i in range(N_STEPS)],
                                 "spectral_mismatch_median": float(np.nanmedian(vec(system, lead, mode, lambda r: float(np.max(np.abs(np.log(np.asarray(r["comp_var_ratio_to_model"])[MISMATCH_COMPONENTS]))))))),
                                 "spectral_mismatch_median_D1_comp25": float(np.nanmedian(vec(system, lead, mode, lambda r: r["spectral_mismatch"]))),
                                 "frac_of_model_D": float(np.nanmedian(Ds) / np.nanmedian(D)) if np.nanmedian(D) != 0 else float("nan")}
            summary["systems"].setdefault(system, {})[f"f{lead:03d}"] = rec

    # sanity checks
    S = summary["systems"]
    chk = {}
    try:
        chk["C33_1"] = {"graphcast_frac_pos": S["graphcast"]["f120"]["deficit_34_frac_positive"],
                        "pangu_frac_pos": S["pangu"]["f120"]["deficit_34_frac_positive"],
                        "hres_abs_median": abs(S["hres"]["f120"]["deficit_34_median"]),
                        "graphcast_median": S["graphcast"]["f120"]["deficit_34_median"]}
        chk["C33_1"]["pass"] = bool(chk["C33_1"]["graphcast_frac_pos"] >= 0.9 and chk["C33_1"]["pangu_frac_pos"] >= 0.9
                                    and chk["C33_1"]["hres_abs_median"] < chk["C33_1"]["graphcast_median"])
    except KeyError:
        chk["C33_1"] = {"pass": None}
    # C33-2 truth reliability across valid-time sets (use graphcast's pairing: all inits)
    try:
        ref = "graphcast" if "graphcast" in S else list(S)[0]
        t24 = np.asarray([head(sets[(ref, 24, d)]["truth"]["P_anchored"]) if (ref, 24, d) in sets else np.nan for d in domains])
        t120 = np.asarray([head(sets[(ref, 120, d)]["truth"]["P_anchored"]) if (ref, 120, d) in sets else np.nan for d in domains])
        ok = np.isfinite(t24) & np.isfinite(t120)
        rho = float(spearmanr(t24[ok], t120[ok]).correlation)
        chk["C33_2"] = {"rho_truth_24_vs_120": rho, "pass": bool(rho >= 0.70)}
    except Exception:  # noqa: BLE001
        chk["C33_2"] = {"pass": None}
    mis = [S[s]["f120"][m]["spectral_mismatch_median"] for s in CM_SYSTEMS for m in ("S1", "S2")
           if s in S and "f120" in S[s] and m in S[s]["f120"]]
    chk["C33_3"] = {"mismatch_medians": mis, "pass": bool(mis and max(mis) <= 0.15)}
    if arm == "A":
        chk["C33_4"] = {s: S[s]["f120"]["D_median"] for s in CM_SYSTEMS if s in S and "f120" in S[s]}
        chk["C33_4"]["pass"] = bool(all(v > 0 for k, v in chk["C33_4"].items() if k != "pass"))
    summary["checks"] = chk

    # hypotheses (scored on arm B; computed on both for the record)
    H = {}
    h1 = {}
    for s in CM_SYSTEMS:
        r = S.get(s, {}).get("f120")
        h1[s] = {"D_median": r["D_median"], "p": r["D_p_signflip"],
                 "pass": bool(r["D_median"] >= DELTA and r["D_p_signflip"] <= 0.01)} if r else {"pass": None}
    H["H33_1"] = h1
    h2 = {}
    for s in NEG_SYSTEMS:
        r = S.get(s, {}).get("f120")
        if r:
            satisfies = r["D_median"] >= DELTA and r["D_p_signflip"] <= 0.01
            h2[s] = {"D_median": r["D_median"], "p": r["D_p_signflip"], "satisfies_H1_criterion": bool(satisfies), "pass": bool(not satisfies)}
    H["H33_2"] = h2
    h3 = {}
    for s in CM_SYSTEMS:
        r = S.get(s, {}).get("f120")
        if r and "S1" in r:
            h3[s] = {"D_S1_median": r["S1"]["D_median"], "D_model_median": r["D_median"], "frac": r["S1"]["frac_of_model_D"],
                     "pass_absolute_clause_dropped_D3": bool(r["S1"]["D_median"] < DELTA / 2),
                     "pass": bool(r["S1"]["D_median"] < 0.5 * r["D_median"])}                  # Deviation D3
    H["H33_3"] = h3
    h4 = {}
    for s in CM_SYSTEMS:
        r = S.get(s, {}).get("f120")
        if r and "S2" in r:
            h4[s] = {"D_S2_median": r["S2"]["D_median"], "p": r["S2"]["D_p_signflip"],
                     "pass": bool(r["S2"]["D_median"] > 0 and r["S2"]["D_p_signflip"] <= 0.01)}
    H["H33_4"] = h4
    H["H33_5_lead"] = {s: {"D24": S[s]["f024"]["D_median"], "D120": S[s]["f120"]["D_median"]}
                       for s in ALL_SYSTEMS if s in S and "f024" in S[s] and "f120" in S[s]}
    H["H33_6_geography"] = {s: {"rho_D_vs_deficit34": S[s]["f120"].get("rho_D_vs_deficit34"),
                                "loso_r2_D_on_deficit15": S[s]["f120"].get("loso_r2_D_on_deficit15")}
                            for s in CM_SYSTEMS if s in S and "f120" in S[s]}
    summary["hypotheses"] = H

    def allpass(d):
        return bool(d) and all(v.get("pass") for v in d.values())

    n1 = sum(1 for v in h1.values() if v.get("pass"))
    n3 = sum(1 for v in h3.values() if v.get("pass"))
    hres_ok = h2.get("hres", {}).get("pass", False)
    hres_sat = h2.get("hres", {}).get("satisfies_H1_criterion", False)
    checks_ok = all(chk[c].get("pass") for c in ("C33_1", "C33_2", "C33_3"))
    if arm == "A":
        verdict = "NOT_SCORED (arm A is hypothesis-generating)"
    elif n1 <= 1 or n3 <= 1 or hres_sat:
        verdict = "NEGATIVE"
    elif checks_ok and n1 == 3 and n3 == 3 and allpass(h2) and allpass(h4):
        verdict = "CONFIRMED"
    elif checks_ok and n1 >= 2 and n3 >= 2 and hres_ok:
        verdict = "PARTIAL"
    else:
        verdict = "UNDETERMINED (see checks)"
    summary["verdict"] = verdict
    summary["definition"] = "frozen: steps 3-5, C33-3 on 100-1600 km" if suffix else "D5: steps 4-5 (200-1600 km), C33-3 on 200-1600 km"
    (out_dir / f"summary{suffix}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(summary, out_dir / f"report{suffix}.md")
    print(f"arm {arm} [{summary['definition']}]: verdict {verdict}", flush=True)
    return summary


def write_report(s: dict, path: Path) -> None:
    S = s["systems"]
    L = [f"# Phase 33 — inheritance fidelity, arm {s['arm']}", "",
         f"Protocol: `{s['protocol']}`. Verdict: **{s['verdict']}**. Domains: {s['n_domains']}. δ = {s['delta']}.", "",
         "## D = P̂_model − P̂_truth (anchored V12, steps 4–5 = 200–1600 km, Deviation D5), 120 h", "",
         "| system | n | P̂_truth | P̂_model | ratio | median D | frac D>0 | p (sign-flip) | deficit 200–800 km | D_S1 (frac) | D_S2 | D steps 3–5 (as frozen) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for sys_ in ALL_SYSTEMS:
        r = S.get(sys_, {}).get("f120")
        if not r:
            continue
        s1 = f"{r['S1']['D_median']:+.4f} ({r['S1']['frac_of_model_D']:.2f})" if "S1" in r else "—"
        s2 = f"{r['S2']['D_median']:+.4f}" if "S2" in r else "—"
        L.append(f"| {sys_} | {r['n']} | {r['P_truth_median']:.4f} | {r['P_model_median']:.4f} | ×{r['ratio_of_medians']:.3f} | "
                 f"{r['D_median']:+.4f} | {r['D_frac_positive']:.2f} | {r['D_p_signflip']:.4f} | {r['deficit_34_median']:+.3f} | {s1} | {s2} | {r['D_frozen35_median']:+.4f} |")
    L += ["", "## Same at 24 h", "", "| system | P̂_truth | P̂_model | median D | p | deficit 200–800 km |", "|---|---|---|---|---|---|"]
    for sys_ in ALL_SYSTEMS:
        r = S.get(sys_, {}).get("f024")
        if r:
            L.append(f"| {sys_} | {r['P_truth_median']:.4f} | {r['P_model_median']:.4f} | {r['D_median']:+.4f} | {r['D_p_signflip']:.4f} | {r['deficit_34_median']:+.3f} |")
    L += ["", "## Per-step median D at 120 h (steps 1–5; step 1 diagnostic)", "", "| system | s1 | s2 | s3 | s4 | s5 |", "|---|---|---|---|---|---|"]
    for sys_ in ALL_SYSTEMS:
        r = S.get(sys_, {}).get("f120")
        if r:
            L.append(f"| {sys_} | " + " | ".join(f"{x:+.4f}" for x in r["D_per_step_median"]) + " |")
        for mode in ("S1", "S2"):
            if r and mode in r:
                L.append(f"| {sys_}:{mode} | " + " | ".join(f"{x:+.4f}" for x in r[mode]["D_per_step_median"]) + " |")
    L += ["", "## Checks and hypotheses", "", "```", json.dumps(s["checks"], indent=1), "```", "```",
          json.dumps(s["hypotheses"], indent=1), "```", ""]
    path.write_text("\n".join(L), encoding="utf-8")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["compute", "tests"])
    ap.add_argument("--arm", choices=["A", "B", "C"], required=True)
    ap.add_argument("--systems", nargs="*", default=None)
    ap.add_argument("--domains", nargs="*", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--n-sur", type=int, default=N_SUR)
    args = ap.parse_args()
    set_roles(args.arm)
    if args.stage == "compute":
        stage_compute(args.arm, args.systems or list(ALL_SYSTEMS), args.workers, args.n_sur, args.domains)
    else:
        stage_tests(args.arm)
        # verdict under the frozen definition, reported alongside (Deviation D5)
        global HEADLINE_STEPS, MISMATCH_COMPONENTS
        HEADLINE_STEPS, MISMATCH_COMPONENTS = FROZEN_STEPS, slice(2, 6)
        stage_tests(args.arm, suffix="_frozen35")


if __name__ == "__main__":
    main()
