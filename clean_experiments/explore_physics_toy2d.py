#!/usr/bin/env python3
"""EXPLORATORY (not frozen) toy model T8: cascade versus independent sources.

Register and expectations (written before the run):
docs/EXPLORATION_PHYSICS_2026-09-29.md

Doubly periodic 2-D vorticity equation, pseudo-spectral, 2/3 dealiasing,
integrating-factor RK4, hyperviscosity and linear drag. Two white-in-time
random forcings share a fixed total enstrophy injection:
  large scale  k in [3, 5]   -> forward enstrophy cascade through the bands;
  small scale  k in [37, 43] -> independent sources inside band 1.
The anchored coupling P is measured with the programme's own estimator on
25 x 25 tiles with the grid step read as 27.8 km.

Usage: python clean_experiments/explore_physics_toy2d.py --share 0.0 [--N 256]
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

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiment_B18_equal_km_regions import envelope_rho_profile_ells  # noqa: E402
from experiment_B1_true_flux_baselines import _interior_mask  # noqa: E402
from experiment_B20_p_geography import ELLS_FINE, N_FINE  # noqa: E402
from experiment_B2_scale_irreversibility import gaussian_bar_grouped  # noqa: E402

OUT = _HERE / "results" / "explore_physics" / "toy2d"
DX_KM = 27.8
TILE = 25
EPS = 1e-12


class Flow:
    def __init__(self, n: int, mu: float, nu_p: float, p: int, seed: int,
                 model: str = "2d"):
        self.n = n
        self.model = model
        k = sfft.fftfreq(n, 1.0 / n)
        kr = sfft.rfftfreq(n, 1.0 / n)
        self.kx, self.ky = np.meshgrid(kr, k)
        self.k2 = self.kx ** 2 + self.ky ** 2
        self.kmag = np.sqrt(self.k2)
        self.k2inv = np.where(self.k2 > 0, 1.0 / np.maximum(self.k2, 1e-30), 0.0)
        if model == "sqg":
            # surface quasi-geostrophy: the advected scalar is the surface
            # buoyancy, psi_hat = -theta_hat / |k|
            self.k2inv = np.where(self.k2 > 0,
                                  1.0 / np.maximum(self.kmag, 1e-30), 0.0)
        kc = n / 3.0
        self.dealias = (np.abs(self.kx) < kc) & (np.abs(self.ky) < kc)
        self.lin = -(mu + nu_p * self.k2 ** p)
        self.lin[0, 0] = 0.0
        self.rng = np.random.default_rng(seed)
        self.wh = np.zeros_like(self.k2, dtype=np.complex128)
        self.t = 0.0

    def velocity(self, wh):
        psih = -wh * self.k2inv
        u = sfft.irfft2(-1j * self.ky * psih, s=(self.n, self.n))
        v = sfft.irfft2(1j * self.kx * psih, s=(self.n, self.n))
        return u, v

    def nl(self, wh):
        u, v = self.velocity(wh)
        wx = sfft.irfft2(1j * self.kx * wh, s=(self.n, self.n))
        wy = sfft.irfft2(1j * self.ky * wh, s=(self.n, self.n))
        return -sfft.rfft2(u * wx + v * wy) * self.dealias

    def forcing(self, shells):
        """Band-limited random field with <f^2> = 2 * eta per shell."""
        fh = np.zeros_like(self.wh)
        for (k0, k1, eta) in shells:
            if eta <= 0:
                continue
            m = (self.kmag >= k0) & (self.kmag <= k1)
            g = np.zeros_like(self.wh)
            g[m] = (self.rng.standard_normal(m.sum())
                    + 1j * self.rng.standard_normal(m.sum()))
            f = sfft.irfft2(g, s=(self.n, self.n))
            f *= np.sqrt(2.0 * eta / np.mean(f * f))
            fh += sfft.rfft2(f)
        return fh

    def step(self, dt, shells):
        e = np.exp(self.lin * dt)
        e2 = np.exp(self.lin * dt / 2)
        w0 = self.wh
        k1 = self.nl(w0)
        k2 = self.nl(e2 * (w0 + 0.5 * dt * k1))
        k3 = self.nl(e2 * w0 + 0.5 * dt * k2)
        k4 = self.nl(e * w0 + dt * e2 * k3)
        self.wh = (e * w0 + dt / 6.0 * (e * k1 + 2 * e2 * (k2 + k3) + k4))
        self.wh += np.sqrt(dt) * self.forcing(shells)
        self.wh *= self.dealias
        self.t += dt

    def cfl_dt(self, cfl=0.5, dt_max=0.02):
        u, v = self.velocity(self.wh)
        umax = max(float(np.max(np.abs(u))), float(np.max(np.abs(v))), 1e-6)
        return min(dt_max, cfl * (2 * np.pi / self.n) / umax)

    def spectrum(self):
        """Isotropic enstrophy spectrum Z(k) = k^4 |psi|^2, integer shells."""
        w2 = np.abs(self.wh * self.k2inv * self.k2) ** 2 / self.n ** 4
        w2 = w2.copy()
        w2[:, 1:-1] *= 2.0
        kb = np.arange(1, self.n // 3)
        return kb, np.array([w2[(self.kmag >= k - 0.5) & (self.kmag < k + 0.5)].sum()
                             for k in kb])


def phase_randomize_scalar(w, rng):
    """Phase surrogate of a scalar cube, tile-wise, same rule as the programme."""
    nt, ny, nx = w.shape
    out = np.empty_like(w)
    for t in range(nt):
        g = rng.standard_normal((ny, nx))
        fac = np.exp(1j * np.angle(np.fft.rfft2(g)))
        fac[0, 0] = 1.0
        out[t] = np.fft.irfft2(np.fft.rfft2(w[t]) * fac, s=(ny, nx))
    return out


def phase_randomize_uv(u, v, rng):
    nt, ny, nx = u.shape
    us, vs = np.empty_like(u), np.empty_like(v)
    for t in range(nt):
        g = rng.standard_normal((ny, nx))
        fac = np.exp(1j * np.angle(np.fft.rfft2(g)))
        fac[0, 0] = 1.0
        us[t] = np.fft.irfft2(np.fft.rfft2(u[t]) * fac, s=(ny, nx))
        vs[t] = np.fft.irfft2(np.fft.rfft2(v[t]) * fac, s=(ny, nx))
    return us, vs


def vort_fd(u, v, dx_km):
    """Finite-difference vorticity exactly as the programme computes it."""
    uk = u.astype(np.float32) / 1000.0
    vk = v.astype(np.float32) / 1000.0
    return (np.gradient(vk, axis=2) / dx_km
            - np.gradient(uk, axis=1) / dx_km).astype(np.float32)


def log_env_cov(w, dx, dy, mask):
    """Covariance matrix of the four band log-envelopes (mean over steps)."""
    bars = [gaussian_bar_grouped(w, ell / np.sqrt(12.0), dx, dy)
            for ell in ELLS_FINE]
    bands = [w - bars[0]] + [bars[i] - bars[i + 1] for i in range(N_FINE)]
    ells = [ELLS_FINE[0]] + list(ELLS_FINE[1:])
    envs = [np.log(gaussian_bar_grouped(np.abs(b), e / np.sqrt(12.0), dx, dy)
                   + EPS)[:, mask] for b, e in zip(bands, ells)]
    nb = len(envs)
    c = np.zeros((nb, nb))
    for i in range(nb):
        for j in range(nb):
            a = envs[i] - envs[i].mean(axis=1, keepdims=True)
            b = envs[j] - envs[j].mean(axis=1, keepdims=True)
            c[i, j] = float(np.mean(np.mean(a * b, axis=1)))
    return c


def measure(u_snap, v_snap, n_sur, seed):
    nt, n, _ = u_snap.shape
    dx = np.full(TILE, DX_KM)
    dy = DX_KM
    mask = _interior_mask(TILE, TILE, ELLS_FINE[-1], dx, dy)
    n_t = n // TILE
    rng = np.random.default_rng(seed)
    rows = []
    for a in range(n_t):
        for b in range(n_t):
            sl = (slice(None), slice(a * TILE, (a + 1) * TILE),
                  slice(b * TILE, (b + 1) * TILE))
            uu, vv = u_snap[sl], v_snap[sl]
            om = vort_fd(uu, vv, DX_KM)
            rho = envelope_rho_profile_ells(om, dx, dy, mask, ELLS_FINE)
            p_real = np.median(rho, axis=0)
            sur = np.empty((n_sur, N_FINE))
            c_sur = np.zeros((4, 4))
            for j in range(n_sur):
                us, vs = phase_randomize_uv(uu, vv, rng)
                om_s = vort_fd(us, vs, DX_KM)
                sur[j] = np.median(envelope_rho_profile_ells(
                    om_s, dx, dy, mask, ELLS_FINE), axis=0)
                if j < 6:
                    c_sur += log_env_cov(om_s, dx, dy, mask) / 6.0
            c_real = log_env_cov(om, dx, dy, mask)
            rows.append({"P_real": p_real.tolist(),
                         "P_sur": np.median(sur, axis=0).tolist(),
                         "P_anch": float(np.mean(p_real - np.median(sur, axis=0))),
                         "C_anch": (c_real - c_sur).tolist(),
                         "kurt": float(np.mean(om ** 4) / np.mean(om ** 2) ** 2)})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--share", type=float, default=0.0,
                    help="small-scale share of the enstrophy injection")
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--ks", type=float, default=40.0)
    ap.add_argument("--mu", type=float, default=0.05)
    ap.add_argument("--spinup", type=float, default=60.0)
    ap.add_argument("--nsnap", type=int, default=60)
    ap.add_argument("--dsnap", type=float, default=2.0)
    ap.add_argument("--nsur", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--tag", default="")
    ap.add_argument("--model", choices=["2d", "sqg"], default="2d")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    tag = (f"{'' if args.model == '2d' else args.model + '_'}"
           f"N{args.N}_s{args.share:.2f}_ks{int(args.ks)}{args.tag}")
    out_f = OUT / f"toy_{tag}.json"
    if out_f.exists():
        print("cached", tag)
        return
    n = args.N
    p = 4
    kmax = n / 3.0
    nu_p = 20.0 / kmax ** (2 * p)          # e-folding rate 20 at the cutoff
    fl = Flow(n, args.mu, nu_p, p, args.seed, args.model)
    shells = [(3.0, 5.0, 1.0 - args.share),
              (args.ks - 3.0, args.ks + 3.0, args.share)]
    fl.wh = 0.01 * fl.forcing([(3.0, 5.0, 1.0)])
    snaps_u, snaps_v = [], []
    next_snap = args.spinup
    t_end = args.spinup + args.nsnap * args.dsnap
    step = 0
    while fl.t < t_end - 1e-9:
        dt = fl.cfl_dt()
        fl.step(dt, shells)
        step += 1
        if fl.t >= next_snap and len(snaps_u) < args.nsnap:
            u, v = fl.velocity(fl.wh)
            snaps_u.append(u.astype(np.float32))
            snaps_v.append(v.astype(np.float32))
            next_snap += args.dsnap
        if step % 2000 == 0:
            w = sfft.irfft2(fl.wh, s=(n, n))
            print(f"[{tag}] t={fl.t:.1f} step={step} dt={dt:.4f} "
                  f"w_rms={np.sqrt(np.mean(w * w)):.3f}", flush=True)
    kb, zk = fl.spectrum()
    w = sfft.irfft2(fl.wh, s=(n, n))
    # scale velocities so that the grid step is 27.8 km and U ~ 10 m/s
    u_s = np.stack(snaps_u)
    v_s = np.stack(snaps_v)
    scale = 10.0 / np.sqrt(np.mean(u_s ** 2 + v_s ** 2))
    rows = measure(u_s * scale, v_s * scale, args.nsur, args.seed + 1)
    sel = (kb >= 8) & (kb <= 32)
    slope = float(np.polyfit(np.log(kb[sel]), np.log(zk[sel] + 1e-300), 1)[0])
    res = {"tag": tag, "args": vars(args), "steps": step,
           "w_rms": float(np.sqrt(np.mean(w * w))),
           "w_flatness": float(np.mean(w ** 4) / np.mean(w ** 2) ** 2),
           "enstrophy_spectrum": {"k": kb.tolist(), "Z": zk.tolist()},
           "enstrophy_slope_k8_32": slope,
           "energy_slope_k8_32": slope - 2.0,
           "P_anch_mean": float(np.mean([r["P_anch"] for r in rows])),
           "P_anch_sd_tiles": float(np.std([r["P_anch"] for r in rows])),
           "P_real_mean": np.mean([r["P_real"] for r in rows], axis=0).tolist(),
           "P_sur_mean": np.mean([r["P_sur"] for r in rows], axis=0).tolist(),
           "C_anch_mean": np.mean([r["C_anch"] for r in rows], axis=0).tolist(),
           "n_tiles": len(rows), "tiles": rows}
    out_f.write_text(json.dumps(res), encoding="utf-8")
    print(json.dumps({k: res[k] for k in res if k not in ("tiles", "enstrophy_spectrum")},
                     indent=1))


if __name__ == "__main__":
    main()
