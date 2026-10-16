#!/usr/bin/env python3
"""Phase 38: co-location below 1600 km, free of frequency modulation.

Frozen protocol: docs/PROTOCOL_PHASE38_SMALLSCALE_COLOCATION.md (2026-10-16).
Planes come from experiment_B37_frequency_modulation.py (stage `plane`, unchanged).

Stages:
    synth    fresh calibration C38-1..3
    tests    hypotheses and verdict
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import experiment_B34_colocation_geography as b34  # noqa: E402  (chdir to repo root)
import experiment_B37_frequency_modulation as b37  # noqa: E402

PROTOCOL = Path("docs/PROTOCOL_PHASE38_SMALLSCALE_COLOCATION.md")
OUT = Path("clean_experiments/results/experiment_B38_smallscale_colocation")
B36 = Path("clean_experiments/results/experiment_B36_residual_regions")
SEED = 20261019
N_PERM = 999


def fm_field(lam_map, rng, kk):
    ny, nx = lam_map.shape
    lams = np.geomspace(lam_map.min(), lam_map.max(), 10); comps = []
    for lam in lams:
        spec = np.exp(-0.5 * ((np.log(np.maximum(kk, 1e-9)) - np.log(1 / lam)) / 0.15) ** 2)
        f = np.fft.ifft2(np.fft.fft2(rng.standard_normal((ny, nx))) * spec).real
        comps.append(f / f.std())
    w = np.stack([np.exp(-0.5 * ((np.log(lam_map) - np.log(l)) / 0.18) ** 2) for l in lams])
    w /= np.sqrt((w ** 2).sum(0, keepdims=True))
    return (w * np.stack(comps)).sum(0)


def cascade_field(rng, kk, var):
    ny, nx = kk.shape
    z = np.zeros((ny, nx)); logA = np.zeros((ny, nx))
    for lam in (1600.0, 800.0, 400.0, 200.0, 100.0):
        band = ((kk >= 1 / (2 * lam)) & (kk < 1 / lam)).astype(float)
        g = np.fft.ifft2(np.fft.fft2(rng.standard_normal((ny, nx))) * band).real
        logA = logA + np.sqrt(var) * g / max(g.std(), 1e-12)
        nk = np.fft.ifft2(np.fft.fft2(rng.standard_normal((ny, nx))) * band).real
        z = z + np.exp(logA - logA.var() / 2) * nk / max(nk.std(), 1e-12)
    return z


def stage_synth():
    OUT.mkdir(parents=True, exist_ok=True)
    w = [x for x in b34.windows() if x["belt"] == 5][0]
    step = b34.EARTH_KM * np.deg2rad(0.25)
    _, _, inside = b34.aeqd_grid(w, step)
    ny, nx = inside.shape
    vm = np.ones_like(inside)
    pl = b37.Plane(vm, step)
    rng = np.random.default_rng(SEED)
    ky = np.fft.fftfreq(ny, d=step)[:, None]; kx = np.fft.fftfreq(nx, d=step)[None, :]
    kk = np.hypot(kx, ky)
    rows = np.where(inside.any(1))[0]; cols = np.where(inside.any(0))[0]
    fy = np.clip((np.arange(ny) - rows.min()) / max(rows.max() - rows.min(), 1), 0, 1)
    lam_y = (300.0 * (1200.0 / 300.0) ** fy)[:, None] * np.ones((1, nx))
    Y, X = np.mgrid[0:ny, 0:nx] * step
    cy, cx = rows.mean() * step, cols.mean() * step
    span = (cols.max() - cols.min()) * step
    d = np.minimum(np.hypot(Y - cy, X - (cx - span / 4)), np.hypot(Y - cy, X - (cx + span / 4)))
    sgm = 1 / (1 + np.exp(-(d - 600.0) / 150.0))
    lam_r = 250.0 * (900.0 / 250.0) ** sgm
    res = {}
    for name, gen in (("C38_1_fm_y", lambda: fm_field(lam_y, rng, kk)),
                      ("C38_2_fm_radial", lambda: fm_field(lam_r, rng, kk)),
                      ("C38_3_cascade", lambda: cascade_field(rng, kk, 0.3))):
        z = np.stack([gen() for _ in range(100)])
        sur = np.stack([b37.mirror_surrogate(z[i], vm, rng) for i in range(len(z))])
        r = b37.decompose(z, sur, pl, inside)
        res[name] = {k: 0.5 * (v["P4"] + v["P5"]) for k, v in r.items()}
        print(name, {k: round(v, 3) for k, v in res[name].items()}, flush=True)
    (OUT / "synth.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


def stage_tests():
    from scipy.stats import spearmanr
    sp = lambda a, b: float(spearmanr(a, b).statistic)  # noqa: E731
    rng = np.random.default_rng(SEED)
    W = b34.windows(); belts = np.array([w["belt"] for w in W])
    res = {"protocol": str(PROTOCOL), "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()}
    syn = json.loads((OUT / "synth.json").read_text())
    c1 = abs(syn["C38_1_fm_y"]["real"] - syn["C38_1_fm_y"]["fa"]) <= 0.10
    c2 = abs(syn["C38_2_fm_radial"]["real"] - syn["C38_2_fm_radial"]["fa"]) <= 0.10
    c3 = (syn["C38_3_cascade"]["real"] - syn["C38_3_cascade"]["fa"]) >= 0.05
    res["calibration"] = {**syn, "C38_1": bool(c1), "C38_2": bool(c2), "C38_3": bool(c3)}
    Y = {y: b37.load_year("era5", y) for y in (2022, 2023, 2024)}
    S = {y: Y[y]["real"] - Y[y]["fa"] for y in Y}
    Sm = 0.5 * (S[2023] + S[2024])
    p_sf = float(b34_sign_flip(Sm))
    res["H38_1"] = {"median_S": float(np.median(Sm)), "p_signflip": p_sf, "frac_positive": float(np.mean(Sm > 0)),
                    "pass": bool(np.median(Sm) >= 0.05 and p_sf < 0.01)}
    rho = sp(S[2023], S[2024])
    null = np.array([sp(S[2023], b34.belt_perm(S[2024], belts, rng, cyclic=True)) for _ in range(N_PERM)])
    p = float((np.sum(null >= rho) + 1) / (N_PERM + 1))
    rho22 = sp(S[2022], Sm)
    res["H38_2"] = {"rho_2023_2024": rho, "p_belt_shift": p, "rho_2022_vs_mean": rho22,
                    "pass": bool(rho >= 0.60 and p < 0.01 and rho22 >= 0.60)}
    mean = {k: 0.5 * (Y[2023][k] + Y[2024][k]) for k in b37.KINDS}
    resid = mean["real"] - mean["null"]
    h3 = {"median": {k: float(np.median(mean[k])) for k in b37.KINDS},
          "median_share_smooth": float(np.median((mean["fa"] - mean["null"]) / resid)),
          "median_share_small_scale": float(np.median(Sm / resid)),
          "rho_S_abslat": sp(Sm, np.abs([w["lat_c"] for w in W]))}
    if (B36 / "window_fields.npz").exists():
        z = np.load(B36 / "window_fields.npz")
        h3["rho_S_vs_phase36_residual"] = sp(Sm, 0.5 * (z["P45_res_2023"] + z["P45_res_2024"]))
    if (b37.OUT / "plane_ifshr_2014_JAS.json").exists():
        M = b37.load_year("ifshr", 2014)
        h3["rho_S_model_vs_era5"] = sp(M["real"] - M["fa"], Sm)
    res["H38_3"] = h3
    if not (c1 and c2 and c3):
        v = "CALIBRATION_FAILED"
    elif not res["H38_1"]["pass"]:
        v = "NO_SMALL_SCALE_COLOCATION"
    elif res["H38_2"]["pass"]:
        v = "SMALL_SCALE_COLOCATION_CONFIRMED"
    else:
        v = "SMALL_SCALE_COLOCATION_NOT_GEOGRAPHIC"
    res["VERDICT"] = v
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez(OUT / "window_fields.npz", wid=np.array([w["wid"] for w in W]),
             **{f"S_{y}": S[y] for y in S}, **{f"{k}_mean": mean[k] for k in mean})
    print(json.dumps({"VERDICT": v}, indent=1), flush=True)


def b34_sign_flip(x, n=9999):
    x = np.asarray(x, float); rng = np.random.default_rng(SEED)
    obs = x.mean(); null = (rng.choice([-1.0, 1.0], size=(n, x.size)) * x[None]).mean(1)
    return (np.sum(null >= obs) + 1) / (n + 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["synth", "tests"])
    a = ap.parse_args()
    stage_synth() if a.stage == "synth" else stage_tests()


if __name__ == "__main__":
    main()
