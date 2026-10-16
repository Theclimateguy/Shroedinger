#!/usr/bin/env python3
"""Phase 37: how much of the co-location residual is frequency modulation.

Frozen protocol: docs/PROTOCOL_PHASE37_FREQUENCY_MODULATION.md (2026-10-16).

Stages:
    synth                                      calibration C37-1, C37-2 on synthetic planes
    plane --source era5 --year Y --season S    real, spectral null and three modulated nulls
    plane --source ifshr                       free-running IFS-HR 2014
    tests                                      hypotheses and verdict
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
from scipy.ndimage import gaussian_filter, map_coordinates

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import experiment_B34_colocation_geography as b34  # noqa: E402  (chdir to repo root)

PROTOCOL = Path("docs/PROTOCOL_PHASE37_FREQUENCY_MODULATION.md")
OUT = Path("clean_experiments/results/experiment_B37_frequency_modulation")
B36 = Path("clean_experiments/results/experiment_B36_residual_regions")
SEED = 20261018
STRIDE = 2
SMOOTH_KM = 1600.0
N_PERM = 999
KINDS = ("real", "null", "fm", "a", "fa")
_G: dict = {}


# ---------------------------------------------------------------------------
# plane ladder
# ---------------------------------------------------------------------------

class Plane:
    def __init__(self, vmask: np.ndarray, step_km: float):
        self.wv = vmask.astype(np.float32)
        self.step = step_km
        self.den = {}

    def _g(self, x, ell):
        s = ell / np.sqrt(12.0) / self.step
        return gaussian_filter(x, sigma=(0, s, s), mode="nearest")

    def bar(self, x, ell):
        if ell not in self.den:
            self.den[ell] = self._g(self.wv[None], ell)
        num = self._g(np.nan_to_num(x) * self.wv[None], ell)
        den = self.den[ell]
        return np.where(den > 1e-3, num / np.maximum(den, 1e-6), np.nan)

    def ladder(self, f):
        bars = [self.bar(f, ell) for ell in b34.ELLS]
        fine = np.nan_to_num(f) - np.nan_to_num(bars[0])
        bands = [np.nan_to_num(bars[k] - bars[k + 1]) for k in range(3)]
        return fine, bands, np.nan_to_num(bars[3])

    def envelopes(self, bands):
        return [np.log(self.bar(np.abs(bands[k]), b34.ELLS[k + 1]) + b34.EPS) for k in range(3)]


def p45_steps(E, sel):
    x = [e[:, sel].astype(np.float64) for e in E]
    x = [xi - xi.mean(1, keepdims=True) for xi in x]
    c = lambda i, j: (x[i] * x[j]).mean(1)  # noqa: E731
    p4 = c(0, 1) / np.sqrt(c(0, 0) * c(1, 1))
    p5 = c(1, 2) / np.sqrt(c(1, 1) * c(2, 2))
    return float(np.median(p4)), float(np.median(p5))


def decompose(real, sur, pl: Plane, sel):
    """P4, P5 for the real field, its stationary null and the three modulated nulls."""
    _, rb, _ = pl.ladder(real)
    Er = pl.envelopes(rb)
    L = [pl.bar(e, SMOOTH_KM) for e in Er]
    A = np.mean(L, axis=0)
    sf, sb, sc = pl.ladder(sur)

    def norm(w):
        ms = np.sqrt(np.nanmean(np.where(sel[None], w ** 2, np.nan), axis=(1, 2), keepdims=True))
        return np.nan_to_num(w / ms)
    weights = {"fm": [norm(np.exp(L[k] - A)) for k in range(3)],
               "a": [norm(np.exp(A - np.nanmean(np.where(sel[None], A, np.nan), axis=(1, 2), keepdims=True))) for _ in range(3)],
               "fa": [norm(np.exp(L[k] - np.nanmean(np.where(sel[None], L[k], np.nan), axis=(1, 2), keepdims=True))) for k in range(3)]}
    out = {"real": p45_steps(Er, sel), "null": p45_steps(pl.envelopes(sb), sel)}
    for kind, w in weights.items():
        out[kind] = p45_steps(pl.envelopes([w[k] * sb[k] for k in range(3)]), sel)
    return {k: {"P4": v[0], "P5": v[1]} for k, v in out.items()}


def mirror_surrogate(f, vm, rng):
    ext = np.block([[f, f[:, ::-1]], [f[::-1, :], f[::-1, ::-1]]])
    ph = np.exp(1j * np.angle(np.fft.rfft2(rng.standard_normal(ext.shape)))); ph[0, 0] = 1.0
    s = np.fft.irfft2(np.fft.rfft2(ext) * ph, s=ext.shape)[: f.shape[0], : f.shape[1]]
    return np.where(vm, s, np.nan)


# ---------------------------------------------------------------------------
# synthetic calibration
# ---------------------------------------------------------------------------

def synth_plane(kind, n, rng, step, inside):
    ny, nx = inside.shape
    ky = np.fft.fftfreq(ny, d=step)[:, None]; kx = np.fft.fftfreq(nx, d=step)[None, :]
    kk = np.hypot(kx, ky)
    cols = np.where(inside.any(0))[0]
    xs = np.arange(nx)
    frac = np.clip((xs - cols.min()) / max(cols.max() - cols.min(), 1), 0, 1)
    lam_x = (250.0 * (900.0 / 250.0) ** frac)[None, :] * np.ones((ny, 1))
    out = []
    for _ in range(n):
        if kind == "fm":
            lams = np.geomspace(250.0, 900.0, 8); comps = []
            for lam in lams:
                spec = np.exp(-0.5 * ((np.log(np.maximum(kk, 1e-9)) - np.log(1 / lam)) / 0.15) ** 2)
                f = np.fft.ifft2(np.fft.fft2(rng.standard_normal((ny, nx))) * spec).real
                comps.append(f / f.std())
            w = np.stack([np.exp(-0.5 * ((np.log(lam_x) - np.log(l)) / 0.18) ** 2) for l in lams])
            w /= np.sqrt((w ** 2).sum(0, keepdims=True))
            z = (w * np.stack(comps)).sum(0)
        else:
            z = np.zeros((ny, nx)); logA = np.zeros((ny, nx))
            for lam in (1600.0, 800.0, 400.0, 200.0, 100.0):
                band = ((kk >= 1 / (2 * lam)) & (kk < 1 / lam)).astype(float)
                g = np.fft.ifft2(np.fft.fft2(rng.standard_normal((ny, nx))) * band).real
                logA = logA + np.sqrt(0.2) * g / max(g.std(), 1e-12)
                nk = np.fft.ifft2(np.fft.fft2(rng.standard_normal((ny, nx))) * band).real
                z = z + np.exp(logA - logA.var() / 2) * nk / max(nk.std(), 1e-12)
        out.append(z)
    return np.stack(out)


def stage_synth():
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / "synth.json"
    if target.exists():
        print("cached synth", flush=True); return
    w = [x for x in b34.windows() if x["belt"] == 5][0]
    step = b34.EARTH_KM * np.deg2rad(0.25)
    _, _, inside = b34.aeqd_grid(w, step)
    vm = np.ones_like(inside)
    pl = Plane(vm, step)
    rng = np.random.default_rng(SEED)
    res = {}
    for kind in ("fm", "cascade"):
        z = synth_plane(kind, 100, rng, step, inside)
        sur = np.stack([mirror_surrogate(z[i], vm, rng) for i in range(len(z))])
        res[kind] = decompose(z, sur, pl, inside)
        print(kind, {k: round(0.5 * (v["P4"] + v["P5"]), 3) for k, v in res[kind].items()}, flush=True)
    target.write_text(json.dumps(res, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------
# real fields
# ---------------------------------------------------------------------------

def _init(om, valid, lat, lon):
    os.environ["OMP_NUM_THREADS"] = "1"
    _G.update(om=om, valid=valid, lat=lat, lon=lon)


def _worker(args):
    wi, w, seed = args
    om, valid, lat, lon = (_G[k] for k in ("om", "valid", "lat", "lon"))
    step = b34.EARTH_KM * np.deg2rad(float(lat[1] - lat[0]))
    plat, plon, inside = b34.aeqd_grid(w, step)
    fy = np.interp(plat, lat, np.arange(len(lat))); fx = plon / float(lon[1] - lon[0])
    vm = map_coordinates(valid.astype(np.float32), [fy, fx], order=0, mode="grid-wrap") > 0.5
    rng = np.random.default_rng(seed)
    real, sur = [], []
    for t in range(om.shape[0]):
        f = map_coordinates(np.nan_to_num(om[t]), [fy, fx], order=1, mode="grid-wrap").astype(np.float64)
        f = np.where(vm, f - f[vm].mean(), 0.0)
        real.append(np.where(vm, f, np.nan)); sur.append(mirror_surrogate(f, vm, rng))
    pl = Plane(vm, step)
    sel = inside & vm
    return {"wid": w["wid"], **decompose(np.stack(real), np.stack(sur), pl, sel)}


def stage_plane(source, year, season, workers):
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"plane_{source}_{year}_{season}.json"
    if target.exists():
        print("cached", target.name, flush=True); return
    if source == "era5":
        lat, lon, u, v, t, valid = b34.load_era5(year, season)
    else:
        lat, lon, u, v, t, valid = b34.load_ifshr(season)
    idx = np.arange(0, u.shape[0], STRIDE)
    u, v = u[idx], v[idx]
    om = np.concatenate([b34.vorticity_sphere(u[i:i + 24], v[i:i + 24], lat, lon) for i in range(0, len(idx), 24)])
    om = np.where(valid[None], om, np.nan).astype(np.float32)
    del u, v
    W = b34.windows()
    seeds = np.random.SeedSequence(SEED + year * 10 + (0 if season == "JFM" else 1) + (5 if source == "ifshr" else 0)).generate_state(len(W))
    with mp.get_context("fork").Pool(workers, _init, (om, valid, lat, lon)) as pool:
        res = pool.map(_worker, [(i, w, int(s)) for i, (w, s) in enumerate(zip(W, seeds))])
    target.write_text(json.dumps({"source": source, "year": year, "season": season, "windows": res}), encoding="utf-8")
    print(f"B37_PLANE_{source}_{year}_{season}_DONE", flush=True)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def load_year(source, year):
    W = b34.windows()
    per = [{r["wid"]: r for r in json.loads((OUT / f"plane_{source}_{year}_{s}.json").read_text())["windows"]}
           for s in b34.MONTHS]
    return {k: np.mean([[0.5 * (p[w["wid"]][k]["P4"] + p[w["wid"]][k]["P5"]) for w in W] for p in per], axis=0)
            for k in KINDS}


def stage_tests():
    from scipy.stats import spearmanr
    sp = lambda a, b: float(spearmanr(a, b).statistic)  # noqa: E731
    rng = np.random.default_rng(SEED)
    W = b34.windows(); belts = np.array([w["belt"] for w in W])
    res = {"protocol": str(PROTOCOL), "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()}
    syn = json.loads((OUT / "synth.json").read_text())
    p45 = lambda d: 0.5 * (d["P4"] + d["P5"])  # noqa: E731
    fm, cas = syn["fm"], syn["cascade"]
    c1 = abs(p45(fm["real"]) - p45(fm["fm"])) <= 0.10 and abs(p45(fm["fm"]) - p45(fm["null"])) >= 0.10
    c2 = (p45(cas["real"]) - p45(cas["fm"])) >= 0.05
    res["C37"] = {"fm": {k: p45(v) for k, v in fm.items()}, "cascade": {k: p45(v) for k, v in cas.items()},
                  "fm_steps": fm, "C37_1": bool(c1), "C37_2": bool(c2)}
    Y = {y: load_year("era5", y) for y in (2022, 2023, 2024)}
    mean = {k: 0.5 * (Y[2023][k] + Y[2024][k]) for k in KINDS}
    resid = mean["real"] - mean["null"]
    f_fm = (mean["fm"] - mean["null"]) / np.where(np.abs(resid) < 1e-6, np.nan, resid)
    res["H37_1"] = {"median_f_FM": float(np.nanmedian(f_fm)), "median_FM_contrib": float(np.median(mean["fm"] - mean["null"])),
                    "median_residual": float(np.median(resid)), "pass": bool(np.nanmedian(f_fm) <= 0.25)}
    r23, r24 = Y[2023]["real"] - Y[2023]["fm"], Y[2024]["real"] - Y[2024]["fm"]
    rho = sp(r23, r24)
    null = np.array([sp(r23, b34.belt_perm(r24, belts, rng, cyclic=True)) for _ in range(N_PERM)])
    p = float((np.sum(null >= rho) + 1) / (N_PERM + 1))
    rho22 = sp(Y[2022]["real"] - Y[2022]["fm"], 0.5 * (r23 + r24))
    res["H37_2"] = {"rho_2023_2024": rho, "p_belt_shift": p, "rho_2022_vs_mean": rho22,
                    "pass": bool(rho >= 0.60 and p < 0.01 and rho22 >= 0.60)}
    h3 = {"median": {k: float(np.median(mean[k])) for k in KINDS},
          "median_contrib": {"fm": float(np.median(mean["fm"] - mean["null"])), "a": float(np.median(mean["a"] - mean["null"])),
                             "fa": float(np.median(mean["fa"] - mean["null"])), "small_scale": float(np.median(mean["real"] - mean["fa"]))},
          "median_share_small_scale": float(np.nanmedian((mean["real"] - mean["fa"]) / np.where(np.abs(resid) < 1e-6, np.nan, resid))),
          "rho_f_FM_abslat": sp(np.nan_to_num(f_fm), np.abs([w["lat_c"] for w in W]))}
    z36 = B36 / "window_fields.npz"
    if z36.exists():
        z = np.load(z36)
        r36 = 0.5 * (z["P45_res_2023"] + z["P45_res_2024"])
        h3["rho_noFM_vs_phase36_residual"] = sp(0.5 * (r23 + r24), r36)
    if (OUT / "plane_ifshr_2014_JAS.json").exists():
        M = load_year("ifshr", 2014)
        h3["rho_noFM_model_vs_era5"] = sp(M["real"] - M["fm"], 0.5 * (r23 + r24))
    res["H37_3"] = h3
    if not (c1 and c2):
        v = "CALIBRATION_FAILED"
    elif res["H37_1"]["pass"] and res["H37_2"]["pass"]:
        v = "FM_MINOR"
    elif res["H37_2"]["pass"]:
        v = "FM_SUBSTANTIAL_GEOGRAPHY_SURVIVES"
    else:
        v = "FM_DOMINANT"
    res["VERDICT"] = v
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez(OUT / "window_fields.npz", wid=np.array([w["wid"] for w in W]),
             **{f"{k}_{y}": Y[y][k] for y in Y for k in KINDS}, f_fm=f_fm)
    print(json.dumps({"VERDICT": v}, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["synth", "plane", "tests"])
    ap.add_argument("--source", choices=["era5", "ifshr"], default="era5")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--season", choices=list(b34.MONTHS), default="JFM")
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args()
    if a.stage == "synth":
        stage_synth()
    elif a.stage == "plane":
        if a.source == "ifshr":
            for s in b34.MONTHS:
                stage_plane("ifshr", 2014, s, a.workers)
        else:
            stage_plane("era5", a.year, a.season, a.workers)
    else:
        stage_tests()


if __name__ == "__main__":
    main()
