#!/usr/bin/env python3
"""EXPLORATORY (not frozen) toy model v2: bundle instrumentation on 2-D turbulence.

Register and expectations (written before the runs):
clean_experiments/results/explore_toy_v2/EXPECTATIONS.md
Theory note: docs/THEORY_NOTE_BUNDLE_REVISITED_2026-10-16.md (sections 3, 4.3, 6).

Dynamical core: `Flow` from explore_physics_toy2d.py (doubly periodic vorticity
equation, pseudo-spectral, 2/3 dealiasing, integrating-factor RK4). `JetFlow`
adds a fixed background shear flow U(y) = U0 sin y.

Per filter scale ell (Gaussian, standard deviation ell, applied spectrally) the
function `bundle_fields` returns the fibre variables of the note:
  coarse gradient A = grad(u_bar): sigma_n, sigma_s, omega, delta, OW, theta_S;
  subgrid stress tau (txx, txy, tyy), k, anisotropy c, director theta_T;
  Germano energy flux Pi, its isotropic part -k delta and deviatoric part;
  Clark (zero-memory) stress tau_C = ell^2 A A^T and flux Pi_C;
  enstrophy flux Z = -tau_w . grad(omega_bar) and its Clark term Z_C;
  band velocity u_band = u_bar(ell/2) - u_bar(ell), sweeping number
  Sw = |u_band . grad theta_S| / sigma, alignment of the fine vorticity gradient
  with the coarse strain axes.
`material_dtheta` gives D_t theta_S = d_t theta_S + u_bar . grad theta_S from
three consecutive snapshots (Eulerian evaluation of the Lagrangian derivative).

Stages:
  run  --U0rel X   simulate (X in units of the base run's u_rms; 0 = base)
  t21              identity and Clark-null checks
  t22              a-priori Clark test for the enstrophy flux
  t24              jet scan: efficiency against sweeping, r vs r_corr
Outputs under clean_experiments/results/explore_toy_v2/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import scipy.fft as sfft
from scipy.stats import rankdata, spearmanr

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_physics_toy2d import Flow  # noqa: E402

OUT = _HERE / "results" / "explore_toy_v2"
SNAPS = OUT / "snaps"
EPS = 1e-30
ELLS = {"l32": 2 * np.pi / 32, "l16": 2 * np.pi / 16, "l8": 2 * np.pi / 8}


# ----------------------------------------------------------------------------
# dynamics
# ----------------------------------------------------------------------------
class JetFlow(Flow):
    """Flow with a fixed background zonal shear U(y) = U0 sin y.

    Perturbation vorticity w' obeys
      d_t w' = -(U + u') d_x w' - v' d_y w' - v' d_y(w_bg),  w_bg = -U0 cos y,
    so the extra terms are U d_x w' + v' U0 sin y = U d_x w' + v' U.
    """

    def __init__(self, n, mu, nu_p, p, seed, U0: float = 0.0):
        super().__init__(n, mu, nu_p, p, seed, "2d")
        y = 2 * np.pi * np.arange(n) / n
        self.U0 = float(U0)
        self.U = (self.U0 * np.sin(y))[:, None] * np.ones((1, n))

    def nl(self, wh):
        u, v = self.velocity(wh)
        wx = sfft.irfft2(1j * self.kx * wh, s=(self.n, self.n))
        wy = sfft.irfft2(1j * self.ky * wh, s=(self.n, self.n))
        return -sfft.rfft2((u + self.U) * wx + v * wy + v * self.U) * self.dealias

    def cfl_dt(self, cfl=0.5, dt_max=0.02):
        u, v = self.velocity(self.wh)
        umax = max(float(np.max(np.abs(u + self.U))), float(np.max(np.abs(v))), 1e-6)
        return min(dt_max, cfl * (2 * np.pi / self.n) / umax)


def simulate(U0: float, seed: int, n: int = 256, mu: float = 0.05,
             spinup: float = 60.0, nsnap: int = 81, dsnap: float = 0.05):
    p = 4
    kmax = n / 3.0
    nu_p = 20.0 / kmax ** (2 * p)
    fl = JetFlow(n, mu, nu_p, p, seed, U0)
    shells = [(3.0, 5.0, 1.0)]
    fl.wh = 0.01 * fl.forcing([(3.0, 5.0, 1.0)])
    us, vs, ts = [], [], []
    next_snap = spinup
    t_end = spinup + (nsnap - 1) * dsnap + 1e-9
    step = 0
    while fl.t < t_end:
        dt = fl.cfl_dt()
        if fl.t + dt > next_snap and len(us) < nsnap:
            dt = max(next_snap - fl.t, 1e-6)
        fl.step(dt, shells)
        step += 1
        if fl.t >= next_snap - 1e-9 and len(us) < nsnap:
            u, v = fl.velocity(fl.wh)
            us.append((u + fl.U).astype(np.float32))
            vs.append(v.astype(np.float32))
            ts.append(fl.t)
            next_snap += dsnap
        if step % 2000 == 0:
            w = sfft.irfft2(fl.wh, s=(n, n))
            print(f"[U0={U0:.3f}] t={fl.t:.2f} step={step} dt={dt:.4f} "
                  f"w_rms={np.sqrt(np.mean(w * w)):.3f}", flush=True)
    u_s, v_s = np.stack(us), np.stack(vs)
    up = u_s - fl.U.astype(np.float32)[None]
    return {"u": u_s, "v": v_s, "t": np.array(ts), "U0": U0, "steps": step,
            "u_rms_pert": float(np.sqrt(np.mean(up ** 2 + v_s ** 2))),
            "dsnap": dsnap, "n": n}


# ----------------------------------------------------------------------------
# spectral helpers and fibre variables
# ----------------------------------------------------------------------------
class Spec:
    def __init__(self, n: int):
        self.n = n
        k = sfft.fftfreq(n, 1.0 / n)
        kr = sfft.rfftfreq(n, 1.0 / n)
        self.kx, self.ky = np.meshgrid(kr, k)
        self.k2 = self.kx ** 2 + self.ky ** 2

    def fft(self, f):
        return sfft.rfft2(np.asarray(f, dtype=np.float64))

    def ifft(self, fh):
        return sfft.irfft2(fh, s=(self.n, self.n))

    def gauss(self, f, ell):
        return self.ifft(self.fft(f) * np.exp(-0.5 * ell ** 2 * self.k2))

    def grad(self, f):
        fh = self.fft(f)
        return self.ifft(1j * self.kx * fh), self.ifft(1j * self.ky * fh)

    def curl(self, u, v):
        return self.ifft(1j * self.kx * self.fft(v) - 1j * self.ky * self.fft(u))


def _grad_periodic(f, dx):
    """Centred differences on the periodic grid (used for angle fields)."""
    fx = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dx)
    fy = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dx)
    return fx, fy


def grad_angle(theta, dx):
    """Gradient of a nematic angle (defined mod pi) via the doubled-angle vector."""
    c2, s2 = np.cos(2 * theta), np.sin(2 * theta)
    cx, cy = _grad_periodic(c2, dx)
    sx, sy = _grad_periodic(s2, dx)
    return 0.5 * (c2 * sx - s2 * cx), 0.5 * (c2 * sy - s2 * cy)


def wrap_pi(a):
    """Wrap to (-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def bundle_fields(u, v, ell, spec: Spec, ell_fine: float | None = None):
    n = spec.n
    dx = 2 * np.pi / n
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    w = spec.curl(u, v)
    g = lambda f: spec.gauss(f, ell)  # noqa: E731
    ub, vb, wb = g(u), g(v), g(w)
    dudx, dudy = spec.grad(ub)
    dvdx, dvdy = spec.grad(vb)
    sn, ss = dudx - dvdy, dudy + dvdx
    om, dl = dvdx - dudy, dudx + dvdy
    sig = np.hypot(sn, ss)
    thS = 0.5 * np.arctan2(ss, sn)
    # subgrid stress and its density-matrix form
    txx = g(u * u) - ub * ub
    txy = g(u * v) - ub * vb
    tyy = g(v * v) - vb * vb
    tr = txx + tyy
    k = 0.5 * tr
    tn, ts = txx - tyy, 2.0 * txy
    c = np.hypot(tn, ts) / (tr + EPS)
    thT = 0.5 * np.arctan2(ts, tn)
    dth = thT - thS
    Pi = -(txx * dudx + txy * (dudy + dvdx) + tyy * dvdy)
    Pi_iso = -k * dl
    Pi_id = -k * dl - k * c * sig * np.cos(2 * dth)
    # Clark (zero-memory) stress: ell^2 A A^T, A_ik = d_k u_i
    e2 = ell ** 2
    tC_xx = e2 * (dudx ** 2 + dudy ** 2)
    tC_xy = e2 * (dudx * dvdx + dudy * dvdy)
    tC_yy = e2 * (dvdx ** 2 + dvdy ** 2)
    Pi_C = -(tC_xx * dudx + tC_xy * (dudy + dvdx) + tC_yy * dvdy)
    Pi_C_formula = -0.25 * e2 * dl * (3 * sig ** 2 + om ** 2 + dl ** 2)
    thC = 0.5 * np.arctan2(2 * tC_xy, tC_xx - tC_yy)
    # enstrophy flux and its Clark term
    wx, wy = spec.grad(wb)
    tw_x = g(u * w) - ub * wb
    tw_y = g(v * w) - vb * wb
    Z = -(tw_x * wx + tw_y * wy)
    twC_x = e2 * (dudx * wx + dudy * wy)
    twC_y = e2 * (dvdx * wx + dvdy * wy)
    Z_C = -(twC_x * wx + twC_y * wy)
    Z_C_formula = -e2 * (0.5 * sn * (wx ** 2 - wy ** 2) + ss * wx * wy
                         + 0.5 * dl * (wx ** 2 + wy ** 2))
    gw = np.hypot(wx, wy)
    eta_Z = Z / (np.hypot(tw_x, tw_y) * gw + EPS)
    eta_Pi = Pi / (k * sig + EPS)              # = -c cos 2 dth when delta = 0
    thG_same = np.arctan2(wy, wx)
    a_same = -np.cos(2 * (thG_same - thS))     # +1: gradient along compressional axis
    thx, thy = grad_angle(thS, dx)
    out = dict(ub=ub, vb=vb, wb=wb, sn=sn, ss=ss, om=om, dl=dl, sig=sig, thS=thS,
               OW=sig ** 2 - om ** 2, txx=txx, txy=txy, tyy=tyy, k=k, c=c, thT=thT,
               dth=dth, Pi=Pi, Pi_iso=Pi_iso, Pi_dev=Pi - Pi_iso, Pi_id=Pi_id,
               Pi_C=Pi_C, Pi_C_formula=Pi_C_formula, thC=thC, Z=Z, Z_C=Z_C,
               Z_C_formula=Z_C_formula, eta_Z=eta_Z, eta_Pi=eta_Pi, a_same=a_same,
               thx=thx, thy=thy, gw=gw)
    if ell_fine is not None:
        uf, vf = spec.gauss(u, ell_fine), spec.gauss(v, ell_fine)
        ubx, vby = uf - ub, vf - vb
        sweep = ubx * thx + vby * thy          # u_band . grad theta_S (rad / time)
        wf = spec.gauss(w, ell_fine)
        wfx, wfy = spec.grad(wf)
        thG_fine = np.arctan2(wfy, wfx)
        out.update(u_band=ubx, v_band=vby, sweep=sweep,
                   Sw=np.abs(sweep) / (sig + EPS),
                   a_fine=-np.cos(2 * (thG_fine - thS)),
                   gw_fine=np.hypot(wfx, wfy))
    return out


def material_dtheta(th_prev, th_now, th_next, ub, vb, thx, thy, dt):
    """D_t theta_S along the coarse flow from three snapshots (nematic angle)."""
    d2 = wrap_pi(2 * th_next - 2 * th_prev) / (2 * dt)
    return 0.5 * d2 + ub * thx + vb * thy


# ----------------------------------------------------------------------------
# statistics helpers
# ----------------------------------------------------------------------------
def spear(x, y, sub=1):
    x = np.asarray(x).ravel()[::sub]
    y = np.asarray(y).ravel()[::sub]
    return float(spearmanr(x, y).correlation)


def pearson(x, y):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    return float(np.corrcoef(x, y)[0, 1])


def binned(x, y, nb=20):
    """Quantile bins of x: bin centres (median x), mean y, eta^2 = var(binmeans)/var(y)."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    q = np.quantile(x, np.linspace(0, 1, nb + 1))
    idx = np.clip(np.searchsorted(q, x, side="right") - 1, 0, nb - 1)
    cen, mean, cnt = [], [], []
    for b in range(nb):
        m = idx == b
        cen.append(float(np.median(x[m])) if m.any() else float("nan"))
        mean.append(float(np.mean(y[m])) if m.any() else float("nan"))
        cnt.append(int(m.sum()))
    mean_a = np.array(mean)
    cnt_a = np.array(cnt, dtype=float)
    ok = cnt_a > 0
    eta2 = float(np.sum(cnt_a[ok] * (mean_a[ok] - y.mean()) ** 2) / (y.size * y.var() + EPS))
    return {"x": cen, "y": mean, "n": cnt, "eta2": eta2}


def load_run(tag):
    z = np.load(SNAPS / f"run_{tag}.npz")
    return {k: z[k] for k in z.files}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


# ----------------------------------------------------------------------------
# stages
# ----------------------------------------------------------------------------
def stage_run(args):
    SNAPS.mkdir(parents=True, exist_ok=True)
    if args.U0rel == 0.0:
        U0, tag = 0.0, "U0rel0.0"
    else:
        base = load_run("U0rel0.0")
        U0 = args.U0rel * float(base["u_rms_pert"])
        tag = f"U0rel{args.U0rel:.1f}"
    f = SNAPS / f"run_{tag}.npz"
    if f.exists() and not args.force:
        print("cached", f)
        return
    res = simulate(U0, args.seed, n=args.N, spinup=args.spinup,
                   nsnap=args.nsnap, dsnap=args.dsnap)
    np.savez_compressed(f, **res)
    print(json.dumps({k: res[k] for k in ("U0", "steps", "u_rms_pert", "dsnap", "n")}))


def stage_t21(args):
    run = load_run("U0rel0.0")
    spec = Spec(int(run["n"]))
    rows = {}
    for name, ell in ELLS.items():
        err_id, ratio_C, errC, errZ, ang = [], [], [], [], []
        for it in range(0, run["u"].shape[0], 8):
            b = bundle_fields(run["u"][it], run["v"][it], ell, spec)
            err_id.append(np.max(np.abs(b["Pi"] - b["Pi_id"])) / np.max(np.abs(b["Pi"])))
            ratio_C.append(np.max(np.abs(b["Pi_C"])) / np.max(np.abs(b["Pi"])))
            errC.append(np.max(np.abs(b["Pi_C"] - b["Pi_C_formula"])) / (np.max(np.abs(b["Pi"]))))
            errZ.append(np.max(np.abs(b["Z_C"] - b["Z_C_formula"])) / np.max(np.abs(b["Z_C"])))
            a = np.rad2deg(np.abs(wrap_pi(2 * (b["thC"] - b["thS"]))))
            ang.append([float(np.median(a)), float(np.percentile(a, 5)), float(np.percentile(a, 95))])
        rows[name] = {"ell": ell, "n_snap": len(err_id),
                      "identity_max_rel_err": float(np.max(err_id)),
                      "PiC_over_Pi_max": float(np.max(ratio_C)),
                      "PiC_formula_err_rel_to_Pi": float(np.max(errC)),
                      "ZC_formula_max_rel_err": float(np.max(errZ)),
                      "clark_director_2dtheta_deg_med_p5_p95": np.median(np.array(ang), axis=0).tolist(),
                      "max_abs_div_over_sigma": float(np.max(np.abs(b["dl"])) / np.max(b["sig"]))}
    out = OUT / "t21"
    out.mkdir(parents=True, exist_ok=True)
    (out / "t21_identity.json").write_text(json.dumps(rows, indent=1))
    print(json.dumps(rows, indent=1))


def _collect(run, ell, ell_fine, spec, it_range):
    """Pooled pixel arrays for one run and scale (every 2nd pixel in each direction)."""
    dt = float(run["dsnap"])
    keys = ["Z", "Z_C", "Pi", "Pi_id", "eta_Z", "eta_Pi", "sig", "om", "c", "dth", "Sw",
            "sweep", "a_fine", "a_same", "k", "gw", "gw_fine"]
    pool = {k: [] for k in keys + ["r", "r_corr", "Dth", "gth"]}
    prev = None
    bs = {}
    for it in it_range:
        if it not in bs:
            bs[it] = bundle_fields(run["u"][it], run["v"][it], ell, spec, ell_fine)
        for j in (it - 1, it + 1):
            if j not in bs:
                bs[j] = bundle_fields(run["u"][j], run["v"][j], ell, spec, None)
        b = bs[it]
        Dth = material_dtheta(bs[it - 1]["thS"], b["thS"], bs[it + 1]["thS"],
                              b["ub"], b["vb"], b["thx"], b["thy"], dt)
        r = (b["om"] + 2 * Dth) / (b["sig"] + EPS)
        r_corr = (b["om"] + 2 * (Dth + b["sweep"])) / (b["sig"] + EPS)
        sl = (slice(None, None, 2), slice(None, None, 2))
        for k in keys:
            pool[k].append(b[k][sl].ravel())
        pool["r"].append(r[sl].ravel())
        pool["r_corr"].append(r_corr[sl].ravel())
        pool["Dth"].append(Dth[sl].ravel())
        pool["gth"].append(np.hypot(b["thx"], b["thy"])[sl].ravel())
        # keep memory bounded
        for j in list(bs):
            if j < it - 1:
                del bs[j]
        prev = it
    return {k: np.concatenate(v) for k, v in pool.items()}


def stage_t22(args):
    run = load_run("U0rel0.0")
    spec = Spec(int(run["n"]))
    nt = run["u"].shape[0]
    its = range(1, nt - 1, args.stride)
    rows, figs = {}, {}
    for name, ell in ELLS.items():
        P = _collect(run, ell, ell / 2, spec, its)
        # per-snapshot Pearson corr(Z, Z_C)
        per = []
        npx = (int(run["n"]) // 2) ** 2
        for s in range(len(its)):
            sl = slice(s * npx, (s + 1) * npx)
            per.append(pearson(P["Z"][sl], P["Z_C"][sl]))
        resid = np.abs(P["Z"] - P["Z_C"]) / (np.median(np.abs(P["Z"])) + EPS)
        absr = np.abs(P["r"])
        rows[name] = {
            "ell": ell,
            "corr_Z_ZC_pooled": pearson(P["Z"], P["Z_C"]),
            "corr_Z_ZC_snapshot_median": float(np.median(per)),
            "corr_Z_ZC_snapshot_p5_p95": [float(np.percentile(per, 5)), float(np.percentile(per, 95))],
            "spearman_Z_ZC": spear(P["Z"], P["Z_C"], 4),
            "corr_Pi_Piid": pearson(P["Pi"], P["Pi_id"]),
            "frac_Z_positive": float(np.mean(P["Z"] > 0)),
            "mean_Z_over_median_absZ": float(np.mean(P["Z"]) / (np.median(np.abs(P["Z"])) + EPS)),
            "mean_ZC_over_mean_Z": float(np.mean(P["Z_C"]) / (np.mean(P["Z"]) + EPS)),
            "frac_abs_r_lt_1": float(np.mean(absr < 1)),
            "spearman_resid_absr": spear(resid, absr, 4),
            "resid_binned_by_absr": binned(np.log(absr + 1e-6), resid, 12),
            "Z_binned_by_absr": binned(np.log(absr + 1e-6), P["Z"] / (np.median(np.abs(P["Z"])) + EPS), 12),
            "a_same_binned_by_absr": binned(np.log(absr + 1e-6), P["a_same"], 12),
            "spearman_a_same_absr": spear(P["a_same"], absr, 4),
            "median_eta_Z": float(np.median(P["eta_Z"])),
            "median_c": float(np.median(P["c"])),
        }
        figs[name] = (P["Z"][::20], P["Z_C"][::20])
        print(name, json.dumps({k: v for k, v in rows[name].items() if not isinstance(v, dict)}, indent=1))
    out = OUT / "t22"
    out.mkdir(parents=True, exist_ok=True)
    (out / "t22_clark_apriori.json").write_text(json.dumps(rows, indent=1))
    plt = _plt()
    fig, axs = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (name, (z, zc)) in zip(axs, figs.items()):
        s = np.median(np.abs(z)) + EPS
        ax.hexbin(zc / s, z / s, gridsize=60, bins="log", extent=(-8, 8, -8, 8))
        ax.plot([-8, 8], [-8, 8], "r--", lw=0.8)
        ax.set_title(f"{name}: corr {rows[name]['corr_Z_ZC_pooled']:.2f}")
        ax.set_xlabel("Z_C / median|Z|")
        ax.set_ylabel("Z / median|Z|")
    fig.tight_layout()
    fig.savefig(out / "fig_Z_vs_ZC.png", dpi=120)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    for name in ELLS:
        bb = rows[name]["resid_binned_by_absr"]
        axs[0].plot(bb["x"], bb["y"], "o-", label=name)
        bb = rows[name]["a_same_binned_by_absr"]
        axs[1].plot(bb["x"], bb["y"], "o-", label=name)
    axs[0].set_xlabel("log |r| (LKH)")
    axs[0].set_ylabel("|Z - Z_C| / median|Z|")
    axs[1].set_xlabel("log |r| (LKH)")
    axs[1].set_ylabel("alignment a (same scale)")
    axs[0].legend()
    fig.tight_layout()
    fig.savefig(out / "fig_resid_vs_r.png", dpi=120)
    print("written", out)


def stage_t24(args):
    tags = [t for t in ("U0rel0.0", "U0rel0.5", "U0rel1.0", "U0rel2.0")
            if (SNAPS / f"run_{t}.npz").exists()]
    rows = {}
    curves = {}
    for tag in tags:
        run = load_run(tag)
        spec = Spec(int(run["n"]))
        nt = run["u"].shape[0]
        its = range(1, nt - 1, args.stride)
        rows[tag] = {"U0": float(run["U0"]), "u_rms_pert": float(run["u_rms_pert"])}
        for name, ell in ELLS.items():
            P = _collect(run, ell, ell / 2, spec, its)
            hi = P["sig"] > np.median(P["sig"])
            absr, absrc = np.abs(P["r"]), np.abs(P["r_corr"])
            lr, lrc = np.log(absr + 1e-6), np.log(absrc + 1e-6)
            d = {
                "median_sigma": float(np.median(P["sig"])),
                "median_abs_omega": float(np.median(np.abs(P["om"]))),
                "median_grad_theta": float(np.median(P["gth"])),
                "median_Sw": float(np.median(P["Sw"])),
                "median_c": float(np.median(P["c"])),
                "median_cos2dth": float(np.median(np.cos(2 * P["dth"]))),
                "median_eta_Z": float(np.median(P["eta_Z"])),
                "median_eta_Pi": float(np.median(P["eta_Pi"])),
                "mean_Z_over_median_absZ": float(np.mean(P["Z"]) / (np.median(np.abs(P["Z"])) + EPS)),
                "spearman_etaZ_Sw": spear(P["eta_Z"], P["Sw"], 4),
                "spearman_etaZ_Sw_hi_sigma": spear(P["eta_Z"][hi], P["Sw"][hi], 4),
                "spearman_absetaPi_Sw": spear(np.abs(P["eta_Pi"]), P["Sw"], 4),
                "spearman_absetaPi_Sw_hi_sigma": spear(np.abs(P["eta_Pi"][hi]), P["Sw"][hi], 4),
                "spearman_c_Sw": spear(P["c"], P["Sw"], 4),
                "etaZ_binned_by_logSw": binned(np.log(P["Sw"] + 1e-6), P["eta_Z"], 12),
                "spearman_afine_absr": spear(P["a_fine"], absr, 4),
                "spearman_afine_absrcorr": spear(P["a_fine"], absrc, 4),
                "eta2_afine_r": binned(lr, P["a_fine"], 20)["eta2"],
                "eta2_afine_rcorr": binned(lrc, P["a_fine"], 20)["eta2"],
                "afine_binned_by_logabsr": binned(lr, P["a_fine"], 16),
                "afine_binned_by_logabsrcorr": binned(lrc, P["a_fine"], 16),
                "spearman_asame_absr": spear(P["a_same"], absr, 4),
                "spearman_asame_absrcorr": spear(P["a_same"], absrc, 4),
                "spearman_Z_absr": spear(P["Z"], absr, 4),
                "spearman_Z_absrcorr": spear(P["Z"], absrc, 4),
                "frac_abs_r_lt_1": float(np.mean(absr < 1)),
                "frac_abs_rcorr_lt_1": float(np.mean(absrc < 1)),
                "median_abs_sweep_over_abs_Dth": float(np.median(np.abs(P["sweep"]) / (np.abs(P["Dth"]) + EPS))),
            }
            rows[tag][name] = d
            curves[(tag, name)] = d
            print(tag, name, json.dumps({k: v for k, v in d.items() if not isinstance(v, dict)}, indent=1))
    out = OUT / "t24"
    out.mkdir(parents=True, exist_ok=True)
    (out / "t24_jet_scan.json").write_text(json.dumps(rows, indent=1))
    plt = _plt()
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, name in zip(axs, ELLS):
        for tag in tags:
            bb = curves[(tag, name)]["etaZ_binned_by_logSw"]
            ax.plot(bb["x"], bb["y"], "o-", label=f"{tag} ρ={curves[(tag, name)]['spearman_etaZ_Sw']:.2f}")
        ax.set_title(name)
        ax.set_xlabel("log Sw")
        ax.set_ylabel("η_Z")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_etaZ_vs_Sw.png", dpi=120)
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, name in zip(axs, ELLS):
        for tag in tags:
            d = curves[(tag, name)]
            b1, b2 = d["afine_binned_by_logabsr"], d["afine_binned_by_logabsrcorr"]
            ln, = ax.plot(b1["x"], b1["y"], "-", label=f"{tag} r (η²={d['eta2_afine_r']:.3f})")
            ax.plot(b2["x"], b2["y"], "--", color=ln.get_color(),
                    label=f"{tag} r_corr (η²={d['eta2_afine_rcorr']:.3f})")
        ax.set_title(name)
        ax.set_xlabel("log |r|")
        ax.set_ylabel("a_fine (fine gradient vs coarse compressional axis)")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / "fig_afine_vs_r.png", dpi=120)
    print("written", out)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["run", "t21", "t22", "t24"])
    ap.add_argument("--U0rel", type=float, default=0.0)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--spinup", type=float, default=60.0)
    ap.add_argument("--nsnap", type=int, default=81)
    ap.add_argument("--dsnap", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=20261016)
    ap.add_argument("--stride", type=int, default=2, help="snapshot stride in analysis stages")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"run": stage_run, "t21": stage_t21, "t22": stage_t22, "t24": stage_t24}[args.stage](args)


if __name__ == "__main__":
    main()
