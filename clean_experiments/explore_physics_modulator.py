#!/usr/bin/env python3
"""EXPLORATORY (not frozen) synthetic test T10: does the "coarse-scale-only"
structure law need a cascade, or does any common activity modulator give it?

Register and expectations (written before the run):
docs/EXPLORATION_PHYSICS_2026-09-29.md, hypothesis H-K.

Field: vorticity w = exp(m) * g, g Gaussian with a power-law spectrum,
m Gaussian with 2-D spectral density k^-beta and variance s2 (the log of the
"activity map"). Winds are obtained by inverting w on a periodic domain; a
2000 x 2800 km box is cut out and passed to the programme's own Phase-27
estimator (six-band ladder, 99 phase surrogates).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from experiment_B27_mechanism import ELL_C_UNITS, anchored_payload  # noqa: E402
from experiment_B28_inheritance_depth import categorical_r2  # noqa: E402
from experiment_B2_scale_irreversibility import ELLS_KM  # noqa: E402

OUT = _HERE / "results" / "explore_physics" / "modulator"
N = 256
DX_KM = 27.8
NY, NX = 73, 101            # 0.25 deg box, 18 x 25 deg at the equator


def gaussian_field(rng, slope_2d: float, kmin: float = 0.0, kmax: float = 1e9):
    """Real Gaussian field with 2-D spectral density k^-slope_2d on N x N."""
    ky = np.fft.fftfreq(N, 1.0 / N)
    kx = np.fft.rfftfreq(N, 1.0 / N)
    kk = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    amp = np.where(kk > 0, np.maximum(kk, 1e-9) ** (-slope_2d / 2.0), 0.0)
    amp[(kk < kmin) | (kk > kmax)] = 0.0
    f = np.fft.irfft2(amp * (rng.standard_normal(kk.shape)
                             + 1j * rng.standard_normal(kk.shape)), s=(N, N))
    return (f - f.mean()) / f.std()


def winds_from_vorticity(w):
    ky = np.fft.fftfreq(N, 1.0 / N) * 2 * np.pi / (N * DX_KM * 1000.0)
    kx = np.fft.rfftfreq(N, 1.0 / N) * 2 * np.pi / (N * DX_KM * 1000.0)
    KX, KY = np.meshgrid(kx, ky)
    k2 = KX ** 2 + KY ** 2
    wh = np.fft.rfft2(w - w.mean())
    psih = np.where(k2 > 0, -wh / np.maximum(k2, 1e-30), 0.0)
    u = np.fft.irfft2(-1j * KY * psih, s=(N, N))
    v = np.fft.irfft2(1j * KX * psih, s=(N, N))
    return u, v


def one_case(beta: float, s2: float, slope_g: float, nt: int, seed: int,
             per_band: bool) -> dict:
    rng = np.random.default_rng(seed)
    uu = np.empty((nt, NY, NX), dtype=np.float32)
    vv = np.empty((nt, NY, NX), dtype=np.float32)
    for t in range(nt):
        g = gaussian_field(rng, slope_g, kmax=N / 2.2)
        if s2 > 0:
            m = np.sqrt(s2) * gaussian_field(rng, beta, kmax=N / 2.2)
            w = np.exp(m) * g
        else:
            w = g
        w = 1e-5 * w / w.std()
        u, v = winds_from_vorticity(w)
        uu[t], vv[t] = u[:NY, :NX], v[:NY, :NX]
    la = -9.0 + 0.25 * np.arange(NY)
    lo = 0.25 * np.arange(NX)
    pay = anchored_payload(uu, vv, la, lo, ELLS_KM, ELL_C_UNITS,
                           f"xpmod_b{beta}_s{s2}_{seed}")
    idx = {n: i for i, n in enumerate(pay["names"])}
    n_env = len(ELLS_KM)
    real = np.asarray(pay["real"])[:, 0]
    sur = np.asarray(pay["sur_med"])[:, 0]
    anch = real - sur
    cov = {(i, j): float(anch[idx[f"cov_{i}_{j}"]])
           for i in range(n_env) for j in range(i + 1, n_env)}
    nonadj = {p: c for p, c in cov.items() if p[1] - p[0] >= 2}
    r2_j = categorical_r2(nonadj, lambda p: p[1])
    r2_sep = categorical_r2(nonadj, lambda p: p[1] - p[0])
    G = {j: float(np.mean([c for (i, jj), c in nonadj.items() if jj == j]))
         for j in range(2, n_env)}
    js = [j for j in G if G[j] > 0]
    expo = (float(np.polyfit(np.log([ELLS_KM[j] for j in js]),
                             np.log([G[j] for j in js]), 1)[0])
            if len(js) >= 3 else float("nan"))
    rho_adj = [float(anch[idx[f"rho_{i}_{i+1}"]]) for i in range(n_env - 1)]
    return {"beta": beta, "s2": s2, "slope_g": slope_g, "nt": nt,
            "R2_coarse_only": r2_j, "R2_separation": r2_sep,
            "G_by_coarse_km": {str(int(ELLS_KM[j])): G[j] for j in G},
            "G_exponent": expo, "predicted_exponent": -(2.0 - beta),
            "P_adjacent_anchored": rho_adj,
            "cov_matrix_anchored": {f"{i}_{j}": c for (i, j), c in cov.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nt", type=int, default=80)
    ap.add_argument("--slope-g", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20260929)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cases = [(0.0, 0.0), (0.0, 0.5), (0.45, 0.5), (1.0, 0.5), (2.0, 0.5),
             (0.45, 0.2), (0.45, 1.0)]
    res = []
    for beta, s2 in cases:
        r = one_case(beta, s2, args.slope_g, args.nt, args.seed, False)
        res.append(r)
        print(json.dumps({k: r[k] for k in r if k != "cov_matrix_anchored"}),
              flush=True)
        (OUT / "modulator_cases.json").write_text(json.dumps(res, indent=1),
                                                  encoding="utf-8")


if __name__ == "__main__":
    main()
