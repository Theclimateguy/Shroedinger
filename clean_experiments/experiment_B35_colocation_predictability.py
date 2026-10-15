#!/usr/bin/env python3
"""Phase 35: co-location and predictability (article 3).

Frozen protocol: docs/PROTOCOL_PHASE35_COLOCATION_PREDICTABILITY.md (2026-10-15).

Stages (all on the 39 programme domains, 2022, WeatherBench 2):
    members   A/B  N-member means of IFS ENS (24, 120 h) and FGN (120 h)
    decouple  C    truth with bands 200-400 and 400-800 km phase-randomised (H35-3)
    synth     C    frequency-modulated and multiplicative-cascade synthetic fields (H35-4)
    leads     D    lead curve 24-240 h (GraphCast, IFS ENS mean, FGN mean, HRES)
    truth2    E    Arm C against the IFS analysis (hres_t0)
    variants  F    Pearson form, mirror surrogates, Portilla-Simoncelli and s21 forms (H35-8, H35-5)
    tests          hypotheses and verdict
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import multiprocessing as mp
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import experiment_B33_model_fidelity as m33  # noqa: E402  (chdir to repo root on import)
import experiment_B32_vorticity_mask as vm  # noqa: E402
from experiment_B1_true_flux_baselines import phase_randomize  # noqa: E402
from download_b11_wb2 import DOMAINS  # noqa: E402
import download_b33c_wb2_2022 as c22  # noqa: E402

PROTOCOL = Path("docs/PROTOCOL_PHASE35_COLOCATION_PREDICTABILITY.md")
OUT = Path("clean_experiments/results/experiment_B35_colocation_predictability")
D33C = Path("data/b33c_wb2")
D35 = Path("data/b35_wb2")
ARMC = Path("clean_experiments/results/experiment_B33_model_fidelity/armC/sets")
SEED = 20261016
N_SUR = 99
HEAD = (3, 4)                         # steps 4-5
DELTA = 0.010
ENS_N = {"ifs_ens": (1, 2, 4, 8, 16, 32, 50), "fgn": (1, 2, 4, 8, 16, 32, 64)}
LEADS = (24, 48, 72, 120, 168, 240)
CURVE_SYSTEMS = ("graphcast", "ifs_ens_mean", "fgn_mean", "hres")
ARMC_SYSTEMS = ("graphcast", "pangu", "ifs_ens_mean", "hres", "ifs_ens_member")
CM3 = ("graphcast", "pangu", "ifs_ens_mean")


def seeds(tag: str, n: int) -> list[int]:
    return [int(s) for s in np.random.SeedSequence(SEED + zlib.crc32(tag.encode()) % 100000).generate_state(n)]


def head(p) -> float:
    return float(np.mean([p[i] for i in HEAD]))


# ---------------------------------------------------------------------------
# estimator with variants
# ---------------------------------------------------------------------------

def rho_profile_pearson(omega, dx, dy, corr_mask, valid, ells):
    """vm.rho_profile with Pearson instead of rank correlation."""
    w = np.where(valid[None], omega.astype(np.float32), np.nan)
    bars = [vm.bar_nan(w, ell / np.sqrt(12.0), dx, dy, valid) for ell in ells]
    envs = [np.log(vm.bar_nan(np.abs(np.nan_to_num(w) - np.nan_to_num(bars[0])), ells[0] / np.sqrt(12.0), dx, dy, valid) + vm.EPS)]
    for i in range(len(ells) - 1):
        envs.append(np.log(vm.bar_nan(np.abs(np.nan_to_num(bars[i] - bars[i + 1])), ells[i + 1] / np.sqrt(12.0), dx, dy, valid) + vm.EPS))
    x = [e[:, corr_mask].astype(np.float64) for e in envs]
    rho = np.zeros((w.shape[0], len(ells) - 1))
    for i in range(len(ells) - 1):
        a = x[i] - x[i].mean(1, keepdims=True); b = x[i + 1] - x[i + 1].mean(1, keepdims=True)
        rho[:, i] = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1))
    return rho


def phase_randomize_mirror(u, v, rng):
    nt, ny, nx = u.shape
    us, vs = np.empty_like(u), np.empty_like(v)
    for t in range(nt):
        ext = lambda f: np.block([[f, f[:, ::-1]], [f[::-1], f[::-1, ::-1]]])  # noqa: E731
        eu, ev = ext(u[t]), ext(v[t])
        ph = np.exp(1j * np.angle(np.fft.rfft2(rng.standard_normal(eu.shape)))); ph[0, 0] = 1.0
        us[t] = np.fft.irfft2(np.fft.rfft2(eu) * ph, s=eu.shape)[:ny, :nx]
        vs[t] = np.fft.irfft2(np.fft.rfft2(ev) * ph, s=ev.shape)[:ny, :nx]
    return us, vs


def profile_v(om, g, pearson=False):
    if pearson:
        return np.median(rho_profile_pearson(om, g.dx, g.dy, g.corr, g.valid, m33.ELLS), axis=0)
    return m33.profile(om, g)


def anchored_v(u, v, g, tag, n_sur=N_SUR, pearson=False, mirror=False, with_var=False) -> dict:
    om = m33.vort(u, v, g)
    p_real = profile_v(om, g, pearson)
    sur = []
    for s in seeds(tag, n_sur):
        rng = np.random.default_rng(s)
        us, vs = (phase_randomize_mirror if mirror else phase_randomize)(u, v, rng)
        sur.append(profile_v(m33.vort(us, vs, g), g, pearson))
    med = np.median(np.asarray(sur), axis=0)
    out = {"P_real": p_real.tolist(), "P_surrogate_median": med.tolist(), "P_anchored": (p_real - med).tolist()}
    if with_var:
        out["comp_var"] = m33.component_variances(om, g).tolist()
    return out


# ---------------------------------------------------------------------------
# loaders
# ---------------------------------------------------------------------------

def member_inits():
    return c22.init_dates(member=True)


def truth_file(valid: dt.datetime) -> Path:
    for d in (D33C, D35):
        f = d / f"era5_{valid:%Y%m%d%H}.npz"
        if f.exists():
            return f
    raise FileNotFoundError(valid)


def stack(files_keys):
    us, vs = [], []
    for f, k in files_keys:
        with np.load(f) as z:
            a = z[k]
        us.append(a[0]); vs.append(a[1])
    return np.stack(us).astype(np.float32), np.stack(vs).astype(np.float32)


def system_file(system: str, init: dt.datetime, lead: int) -> Path:
    for d in (D33C, D35):
        f = d / f"{system}_{init:%Y%m%d%H}_f{lead:03d}.npz"
        if f.exists():
            return f
    raise FileNotFoundError(f"{system} {init} {lead}")


# ---------------------------------------------------------------------------
# task runner
# ---------------------------------------------------------------------------

def run_tasks(fn, tasks, workers, label):
    OUT.mkdir(parents=True, exist_ok=True)
    todo = [t for t in tasks if not (OUT / t["file"]).exists()]
    print(f"[{label}] {len(tasks)} tasks, {len(todo)} to do, {workers} workers", flush=True)
    with mp.get_context("fork").Pool(workers) as pool:
        for i, msg in enumerate(pool.imap_unordered(fn, todo), 1):
            if i % 10 == 0 or i == len(todo):
                print(f"[{label}] {i}/{len(todo)} {msg}", flush=True)


def _write(t, res):
    p = OUT / t["file"]; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res), encoding="utf-8")


# --- A/B: members --------------------------------------------------------------

def task_members(t):
    os.environ["OMP_NUM_THREADS"] = "1"
    t0 = time.time()
    sysn, lead, dom = t["system"], t["lead"], t["domain"]
    g = m33.Geo(*m33.domain_grid(dom))
    inits = member_inits()
    mem = []
    for d0 in inits:
        with np.load(D35 / f"{sysn}_members_{d0:%Y%m%d%H}_f{lead:03d}.npz") as z:
            mem.append(z[dom])                                         # (M, 2, ny, nx)
    mem = np.stack(mem)                                                # (nf, M, 2, ny, nx)
    ue, ve = stack([(truth_file(d0 + dt.timedelta(hours=lead)), dom) for d0 in inits])
    res = {**t, "n_fields": len(inits), "M": int(mem.shape[1]),
           "truth": anchored_v(ue, ve, g, f"m35_{sysn}_{lead}_{dom}_truth", with_var=True), "N": {}}
    for n in ENS_N[sysn]:
        mu = mem[:, :n].mean(1)
        res["N"][str(n)] = anchored_v(mu[:, 0], mu[:, 1], g, f"m35_{sysn}_{lead}_{dom}_N{n}", with_var=True)
    res["member_comp_var"] = [m33.component_variances(m33.vort(mem[:, k, 0], mem[:, k, 1], g), g).tolist()
                              for k in range(mem.shape[1])]
    _write(t, res)
    return f"{sysn} f{lead} {dom} {time.time() - t0:.0f}s"


# --- C: decoupled truth ------------------------------------------------------

def task_decouple(t):
    os.environ["OMP_NUM_THREADS"] = "1"
    dom = t["domain"]
    g = m33.Geo(*m33.domain_grid(dom))
    inits = c22.init_dates()
    ue, ve = stack([(truth_file(d0 + dt.timedelta(hours=120)), dom) for d0 in inits])
    cu, cv = m33.components(ue, g, None), m33.components(ve, g, None)
    rng = np.random.default_rng(seeds(f"dec_{dom}", 1)[0])
    for k in (3, 4):                                                   # 200-400 and 400-800 km
        cu[k], cv[k] = phase_randomize(cu[k].astype(np.float64), cv[k].astype(np.float64), rng)
    ud = sum(cu).astype(np.float32); vd = sum(cv).astype(np.float32)
    res = {**t, "truth": anchored_v(ue, ve, g, f"dec_{dom}_truth"),
           "decoupled": anchored_v(ud, vd, g, f"dec_{dom}_dec")}
    _write(t, res)
    return dom


# --- C: synthetic fields -------------------------------------------------------

SYN_DOMAIN = "R11_EURO"


def _uv_from_vort(z, g):
    """Doubly periodic inversion of a vorticity map to u, v on the domain grid (km metric at the centre)."""
    ny, nx = z.shape
    dx = float(np.mean(g.dx)) * 1e3; dy = g.dy * 1e3
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=dy)[:, None]; kx = 2 * np.pi * np.fft.fftfreq(nx, d=dx)[None, :]
    k2 = kx ** 2 + ky ** 2; k2[0, 0] = 1.0
    psi = np.fft.ifft2(-np.fft.fft2(z) / k2).real
    u = -np.gradient(psi, dy, axis=0); v = np.gradient(psi, dx, axis=1)
    return u.astype(np.float32), v.astype(np.float32)


def synth_fields(kind, g, n, rng):
    ny, nx = len(g.lat), len(g.lon)
    dxk, dyk = float(np.mean(g.dx)), g.dy
    ky = np.fft.fftfreq(ny, d=dyk)[:, None]; kx = np.fft.fftfreq(nx, d=dxk)[None, :]
    kk = np.hypot(kx, ky)                                              # cycles/km
    us, vs = [], []
    for _ in range(n):
        if kind == "fm":
            # uniform-amplitude field whose dominant wavelength varies smoothly from 250 to 900 km
            lam_x = np.linspace(250.0, 900.0, nx)[None, :] * np.ones((ny, 1))
            comps, lams = [], np.geomspace(250.0, 900.0, 8)
            for lam in lams:
                spec = np.exp(-0.5 * ((np.log(np.maximum(kk, 1e-9)) - np.log(1 / lam)) / 0.15) ** 2)
                f = np.fft.ifft2(np.fft.fft2(rng.standard_normal((ny, nx))) * spec).real
                comps.append(f / f.std())
            wts = np.stack([np.exp(-0.5 * ((np.log(lam_x) - np.log(l)) / 0.18) ** 2) for l in lams])
            wts /= np.sqrt((wts ** 2).sum(0, keepdims=True))
            z = (wts * np.stack(comps)).sum(0)
        else:
            # log-normal multiplicative cascade, five octave levels (1600 -> 100 km), log-amplitude variance 0.2 per level
            z = np.zeros((ny, nx)); logA = np.zeros((ny, nx))
            for lam in (1600.0, 800.0, 400.0, 200.0, 100.0):
                band = ((kk >= 1 / (2 * lam)) & (kk < 1 / lam)).astype(float)
                gfield = np.fft.ifft2(np.fft.fft2(rng.standard_normal((ny, nx))) * band).real
                logA = logA + np.sqrt(0.2) * gfield / max(gfield.std(), 1e-12)
                n_k = np.fft.ifft2(np.fft.fft2(rng.standard_normal((ny, nx))) * band).real
                z = z + np.exp(logA - logA.var() / 2) * n_k / max(n_k.std(), 1e-12)
        u, v = _uv_from_vort(z * 1e-5, g)
        us.append(u); vs.append(v)
    return np.stack(us), np.stack(vs)


def task_synth(t):
    os.environ["OMP_NUM_THREADS"] = "1"
    g = m33.Geo(*m33.domain_grid(SYN_DOMAIN))
    g.valid = np.ones_like(g.valid); g.corr = g.interior.copy()        # synthetic: no terrain mask
    rng = np.random.default_rng(seeds(f"syn_{t['kind']}", 1)[0])
    u, v = synth_fields(t["kind"], g, 100, rng)
    _write(t, {**t, "res": anchored_v(u, v, g, f"syn_{t['kind']}")})
    return t["kind"]


# --- D: lead curve -----------------------------------------------------------

def task_leads(t):
    os.environ["OMP_NUM_THREADS"] = "1"
    sysn, lead, dom = t["system"], t["lead"], t["domain"]
    g = m33.Geo(*m33.domain_grid(dom))
    inits = member_inits()
    um, vmm = stack([(system_file(sysn, d0, lead), dom) for d0 in inits])
    ue, ve = stack([(truth_file(d0 + dt.timedelta(hours=lead)), dom) for d0 in inits])
    _write(t, {**t, "model": anchored_v(um, vmm, g, f"ld_{sysn}_{lead}_{dom}_m"),
               "truth": anchored_v(ue, ve, g, f"ld_{lead}_{dom}_t")})
    return f"{sysn} f{lead} {dom}"


# --- E: second truth -------------------------------------------------------------

def task_truth2(t):
    os.environ["OMP_NUM_THREADS"] = "1"
    lead, dom, subset = t["lead"], t["domain"], t["subset"]
    g = m33.Geo(*m33.domain_grid(dom))
    inits = member_inits() if subset == "member" else c22.init_dates()
    files = [(D35 / f"hres_t0_{d0 + dt.timedelta(hours=lead):%Y%m%d%H}.npz", dom) for d0 in inits]
    files = [f for f in files if f[0].exists()]
    u, v = stack(files)
    _write(t, {**t, "n_fields": len(files), "truth_t0": anchored_v(u, v, g, f"t0_{subset}_{lead}_{dom}")})
    return f"t0 {subset} f{lead} {dom}"


# --- F: variants and nearest prior forms -------------------------------------------

def ps_and_s21(u, v, g):
    """Portilla-Simoncelli adjacent modulus correlation and scattering ratio s21 for steps 4-5 band pairs."""
    om = np.where(g.valid[None], m33.vort(u, v, g), np.nan).astype(np.float32)
    comps = m33.components(om, g, g.valid)
    out = {"ps": [], "s21": []}
    for k1, k2 in ((3, 4), (4, 5)):
        a, b = np.abs(comps[k1]), np.abs(comps[k2])
        x = a[:, g.corr].astype(np.float64); y = b[:, g.corr].astype(np.float64)
        xa = x - x.mean(1, keepdims=True); ya = y - y.mean(1, keepdims=True)
        out["ps"].append(float(np.median((xa * ya).sum(1) / np.sqrt((xa ** 2).sum(1) * (ya ** 2).sum(1)))))
        modc = m33.components(a, g, g.valid)[k2]                       # band k2 of the modulus of band k1
        out["s21"].append(float(np.median(np.mean(np.abs(modc[:, g.corr]), 1) / np.mean(a[:, g.corr], 1))))
    return out


def task_variants(t):
    os.environ["OMP_NUM_THREADS"] = "1"
    sysn, dom = t["system"], t["domain"]
    g = m33.Geo(*m33.domain_grid(dom))
    inits = member_inits() if sysn == "ifs_ens_member" else c22.init_dates()
    um, vmm = stack([(system_file(sysn, d0, 120), dom) for d0 in inits])
    ue, ve = stack([(truth_file(d0 + dt.timedelta(hours=120)), dom) for d0 in inits])
    res = {**t}
    for side, (a, b) in (("model", (um, vmm)), ("truth", (ue, ve))):
        res[side] = {"pearson": anchored_v(a, b, g, f"vp_{sysn}_{dom}_{side}", pearson=True),
                     "mirror": anchored_v(a, b, g, f"vm_{sysn}_{dom}_{side}", mirror=True),
                     "prior_forms": ps_and_s21(a, b, g)}
    _write(t, res)
    return f"{sysn} {dom}"


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def sfp(x, lower=False):
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    return m33.sign_flip_p(-x if lower else x)


def load(sub):
    return [json.loads(p.read_text()) for p in sorted((OUT / sub).glob("*.json"))]


def stage_tests():
    from scipy.stats import spearmanr
    sp = lambda a, b: float(spearmanr(a, b).statistic)  # noqa: E731
    res = {"protocol": str(PROTOCOL), "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()}

    # H35-1 and H35-2
    mem = load("members")
    h1, h2 = {}, {}
    for sysn in ENS_N:
        for lead in (24, 120):
            rs = [r for r in mem if r["system"] == sysn and r["lead"] == lead]
            if not rs:
                continue
            Ns = ENS_N[sysn]
            D = {n: np.array([head(r["N"][str(n)]["P_anchored"]) - head(r["truth"]["P_anchored"]) for r in rs]) for n in Ns}
            med = [float(np.median(D[n])) for n in Ns]
            rec = {"median_D": dict(zip(map(str, Ns), med)), "rho_N_D": sp(Ns, med),
                   "D1_p_upper": sfp(D[1]), "DM_p_upper": sfp(D[Ns[-1]]), "n_domains": len(rs)}
            rec["pass"] = bool(rec["rho_N_D"] >= 0.90 and rec["D1_p_upper"] > 0.05 and med[-1] >= DELTA and rec["DM_p_upper"] <= 0.01)
            h1[f"{sysn}_f{lead:03d}"] = rec
            # H35-2 mixture model on the raw profile
            errs = []
            for r in rs:
                M = r["M"]
                vmem = np.mean(np.asarray(r["member_comp_var"]), axis=0)
                vmean = np.asarray(r["N"][str(M)]["comp_var"])
                for s, fb in ((3, 3), (4, 4)):                          # step index, fine component index
                    q = (vmean[fb] - vmem[fb] / M) / (vmem[fb] * (1 - 1 / M))
                    q = float(np.clip(q, 1e-3, 1.0))
                    P1 = r["N"]["1"]["P_real"][s]; PM = r["N"][str(M)]["P_real"][s]
                    # P(N) = (q rp + (1-q) ru / N) / (q + (1-q)/N); solve rp, ru from N = 1 and N = M
                    A = np.array([[q, 1 - q], [q / (q + (1 - q) / M), (1 - q) / M / (q + (1 - q) / M)]])
                    rp, ru = np.linalg.solve(A, [P1, PM])
                    span = abs(PM - P1)
                    if span < 1e-4:
                        continue
                    for n in Ns[1:-1]:
                        pred = (q * rp + (1 - q) * ru / n) / (q + (1 - q) / n)
                        errs.append(abs(r["N"][str(n)]["P_real"][s] - pred) / span)
            h2[f"{sysn}_f{lead:03d}"] = {"median_rel_err": float(np.median(errs)) if errs else None, "n": len(errs)}
    res["H35_1"] = h1
    res["H35_1"]["pass"] = bool(h1.get("ifs_ens_f120", {}).get("pass") and h1.get("fgn_f120", {}).get("pass"))
    res["H35_1"]["n_pass_120"] = int(bool(h1.get("ifs_ens_f120", {}).get("pass"))) + int(bool(h1.get("fgn_f120", {}).get("pass")))
    res["H35_2"] = {**h2, "pass": bool(all(h2.get(k, {}).get("median_rel_err") is not None and h2[k]["median_rel_err"] <= 0.25 for k in ("ifs_ens_f120", "fgn_f120")))}

    # H35-3
    dec = load("decouple")
    if dec:
        Dd = np.array([head(r["decoupled"]["P_anchored"]) - head(r["truth"]["P_anchored"]) for r in dec])
        res["H35_3"] = {"median_D_decoupled": float(np.median(Dd)), "p_lower": sfp(Dd, lower=True), "n": len(dec),
                        "pass": bool(np.median(Dd) <= -0.05 and sfp(Dd, lower=True) <= 0.01)}
    # H35-4
    syn = {r["kind"]: r for r in load("synth")}
    if syn:
        fm, cas = head(syn["fm"]["res"]["P_anchored"]), head(syn["cascade"]["res"]["P_anchored"])
        res["H35_4"] = {"P_fm": fm, "P_cascade": cas, "pass": bool(abs(fm) <= 0.02 and cas >= 0.10)}
    # H35-6
    ld = load("leads")
    if ld:
        h6 = {}
        for sysn in CURVE_SYSTEMS:
            med = {}
            for lead in LEADS:
                rs = [r for r in ld if r["system"] == sysn and r["lead"] == lead]
                if rs:
                    med[lead] = float(np.median([head(r["model"]["P_anchored"]) - head(r["truth"]["P_anchored"]) for r in rs]))
            h6[sysn] = {"median_D": {str(k): v for k, v in med.items()},
                        "rho_lead_D": sp(list(med), list(med.values())) if len(med) > 2 else None}
        cm_ok = all((h6[s]["rho_lead_D"] or -1) >= 0.80 for s in ("graphcast", "ifs_ens_mean", "fgn_mean"))
        hr_ok = all(abs(v) < DELTA for v in h6["hres"]["median_D"].values())
        res["H35_6"] = {**h6, "pass": bool(cm_ok and hr_ok)}
    # H35-7
    t2 = load("truth2")
    if t2:
        h7 = {}
        T = {(r["lead"], r["domain"], r["subset"]): r for r in t2}
        for sysn in ARMC_SYSTEMS:
            subset = "member" if sysn == "ifs_ens_member" else "all"
            Ds = []
            for dom in DOMAINS:
                f = ARMC / f"{sysn}_f120_{dom}.json"
                if not f.exists() or (120, dom, subset) not in T:
                    continue
                mrec = json.loads(f.read_text())
                Ds.append(head(mrec["model"]["P_anchored"]) - head(T[(120, dom, subset)]["truth_t0"]["P_anchored"]))
            Ds = np.array(Ds)
            h7[sysn] = {"median_D": float(np.median(Ds)), "p_upper": sfp(Ds), "n": len(Ds)}
        ok = all(h7[s]["median_D"] >= DELTA and h7[s]["p_upper"] <= 0.01 for s in CM3) and \
            all(abs(h7[s]["median_D"]) < DELTA for s in ("hres", "ifs_ens_member"))
        res["H35_7"] = {**h7, "pass": bool(ok)}
    # H35-8 and H35-5
    va = load("variants")
    if va:
        h8, h5 = {}, {}
        for sysn in ARMC_SYSTEMS:
            rs = [r for r in va if r["system"] == sysn]
            for var in ("pearson", "mirror"):
                Ds = np.array([head(r["model"][var]["P_anchored"]) - head(r["truth"][var]["P_anchored"]) for r in rs])
                h8.setdefault(var, {})[sysn] = {"median_D": float(np.median(Ds)), "p_upper": sfp(Ds)}
            for form in ("ps", "s21"):
                Ds = np.array([np.mean(r["model"]["prior_forms"][form]) - np.mean(r["truth"]["prior_forms"][form]) for r in rs])
                q75, q25 = np.percentile(Ds, [75, 25])
                h5.setdefault(form, {})[sysn] = {"median_D": float(np.median(Ds)), "effect": float(np.median(Ds) / max(q75 - q25, 1e-9))}
        armc = {}
        for sysn in ARMC_SYSTEMS:
            Ds = []
            for dom in DOMAINS:
                f = ARMC / f"{sysn}_f120_{dom}.json"
                if f.exists():
                    r = json.loads(f.read_text()); Ds.append(head(r["model"]["P_anchored"]) - head(r["truth"]["P_anchored"]))
            Ds = np.array(Ds); q75, q25 = np.percentile(Ds, [75, 25])
            armc[sysn] = {"median_D": float(np.median(Ds)), "effect": float(np.median(Ds) / max(q75 - q25, 1e-9))}
        h5["anchored_log_rank (Arm C)"] = armc
        ok = all(all(h8[v][s]["median_D"] >= DELTA for s in CM3) and all(abs(h8[v][s]["median_D"]) < DELTA for s in ("hres", "ifs_ens_member"))
                 for v in ("pearson", "mirror"))
        res["H35_8"] = {**h8, "pass": bool(ok)}
        res["H35_5"] = h5

    get = lambda k: bool(res.get(k, {}).get("pass"))  # noqa: E731
    if res["H35_1"]["n_pass_120"] == 0 or (k4 := res.get("H35_4")) is not None and not k4["pass"]:
        v = "NEGATIVE"
    elif get("H35_1") and get("H35_3") and get("H35_4") and get("H35_6") and get("H35_7"):
        v = "CO-LOCATION_PREDICTABILITY_CONFIRMED"
    elif res["H35_1"]["n_pass_120"] >= 1 and get("H35_3") and get("H35_4"):
        v = "PARTIAL"
    else:
        v = "UNDETERMINED"
    res["VERDICT"] = v
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps({"VERDICT": v}, indent=1), flush=True)
    return res


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["members", "decouple", "synth", "leads", "truth2", "variants", "tests"])
    ap.add_argument("--workers", type=int, default=14)
    a = ap.parse_args()
    doms = list(DOMAINS)
    if a.stage == "members":
        tasks = [{"system": s, "lead": l, "domain": d, "file": f"members/{s}_f{l:03d}_{d}.json"}
                 for s, leads in (("ifs_ens", (120, 24)), ("fgn", (120,))) for l in leads for d in doms]
        run_tasks(task_members, tasks, a.workers, "members")
    elif a.stage == "decouple":
        run_tasks(task_decouple, [{"domain": d, "file": f"decouple/{d}.json"} for d in doms], a.workers, "decouple")
    elif a.stage == "synth":
        run_tasks(task_synth, [{"kind": k, "file": f"synth/{k}.json"} for k in ("fm", "cascade")], 2, "synth")
    elif a.stage == "leads":
        tasks = [{"system": s, "lead": l, "domain": d, "file": f"leads/{s}_f{l:03d}_{d}.json"}
                 for s in CURVE_SYSTEMS for l in LEADS for d in doms]
        run_tasks(task_leads, tasks, a.workers, "leads")
    elif a.stage == "truth2":
        tasks = [{"lead": l, "domain": d, "subset": sub, "file": f"truth2/{sub}_f{l:03d}_{d}.json"}
                 for l in (24, 120) for d in doms for sub in ("all", "member")]
        run_tasks(task_truth2, tasks, a.workers, "truth2")
    elif a.stage == "variants":
        tasks = [{"system": s, "domain": d, "file": f"variants/{s}_{d}.json"} for s in ARMC_SYSTEMS for d in doms]
        run_tasks(task_variants, tasks, a.workers, "variants")
    else:
        stage_tests()


if __name__ == "__main__":
    main()
