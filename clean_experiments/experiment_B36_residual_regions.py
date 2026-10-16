#!/usr/bin/env python3
"""Phase 36: regions on the residual after a spectral null.

Frozen protocol: docs/PROTOCOL_PHASE36_RESIDUAL_REGIONS.md (2026-10-16).

Stages:
    plane --source era5 --year Y --season S    real and mirror-surrogate features per window
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

PROTOCOL = Path("docs/PROTOCOL_PHASE36_RESIDUAL_REGIONS.md")
OUT = Path("clean_experiments/results/experiment_B36_residual_regions")
B34 = Path("clean_experiments/results/experiment_B34_colocation_geography")
SEED = 20261017
STRIDE = 2
N_PERM = 999
_G: dict = {}


def plane_envelopes(f: np.ndarray, vmask: np.ndarray, step_km: float) -> list[np.ndarray]:
    wv = vmask.astype(np.float32)

    def g(x, ell):
        s = ell / np.sqrt(12.0) / step_km
        return gaussian_filter(x, sigma=(0, s, s), mode="nearest")

    def bar(x, ell):
        num = g(np.nan_to_num(x) * wv[None], ell)
        den = g(wv[None], ell)
        return np.where(den > 1e-3, num / np.maximum(den, 1e-6), np.nan)
    bars = [bar(f, ell) for ell in b34.ELLS]
    return [np.log(bar(np.abs(np.nan_to_num(bars[k] - bars[k + 1])), b34.ELLS[k + 1]) + b34.EPS).astype(np.float32)
            for k in range(3)]


def features(E: list[np.ndarray], sel: np.ndarray, block: np.ndarray) -> dict:
    x = np.stack([e[:, sel] for e in E]).astype(np.float64)          # (3, nt, n)
    x = x - x.mean(-1, keepdims=True)
    C = np.array([[np.mean(np.mean(x[i] * x[j], axis=1)) for j in range(3)] for i in range(3)])
    c = lambda i, j: (x[i] * x[j]).mean(1)                            # noqa: E731  per-step covariance
    p4 = c(0, 1) / np.sqrt(c(0, 0) * c(1, 1))
    p5 = c(1, 2) / np.sqrt(c(1, 1) * c(2, 2))
    ma = [xi[block == 0].mean(0) for xi in x]; mb = [xi[block == 1].mean(0) for xi in x]
    ma = [m - m.mean() for m in ma]; mb = [m - m.mean() for m in mb]
    T = np.array([[0.5 * (np.mean(ma[i] * mb[j]) + np.mean(mb[i] * ma[j])) for j in range(3)] for i in range(3)])
    s = float(np.mean([T[i, i] / C[i, i] for i in range(3)]))
    M = C - T
    r_mov = float(M[0, 2] / np.sqrt(M[0, 0] * M[2, 2]))
    return {"P4": float(np.median(p4)), "P5": float(np.median(p5)), "s": s, "r_mov": r_mov}


def _init(om, valid, lat, lon, block):
    os.environ["OMP_NUM_THREADS"] = "1"
    _G.update(om=om, valid=valid, lat=lat, lon=lon, block=block)


def _worker(args):
    wi, w, seed = args
    om, valid, lat, lon, block = (_G[k] for k in ("om", "valid", "lat", "lon", "block"))
    step = b34.EARTH_KM * np.deg2rad(float(lat[1] - lat[0]))
    plat, plon, inside = b34.aeqd_grid(w, step)
    fy = np.interp(plat, lat, np.arange(len(lat)))
    fx = plon / float(lon[1] - lon[0])
    vm = map_coordinates(valid.astype(np.float32), [fy, fx], order=0, mode="grid-wrap") > 0.5
    rng = np.random.default_rng(seed)
    real, sur = [], []
    for t in range(om.shape[0]):
        f = map_coordinates(np.nan_to_num(om[t]), [fy, fx], order=1, mode="grid-wrap").astype(np.float64)
        f = np.where(vm, f - f[vm].mean(), 0.0)
        real.append(np.where(vm, f, np.nan))
        ext = np.block([[f, f[:, ::-1]], [f[::-1, :], f[::-1, ::-1]]])
        ph = np.exp(1j * np.angle(np.fft.rfft2(rng.standard_normal(ext.shape)))); ph[0, 0] = 1.0
        s = np.fft.irfft2(np.fft.rfft2(ext) * ph, s=ext.shape)[: f.shape[0], : f.shape[1]]
        sur.append(np.where(vm, s, np.nan))
    Er = plane_envelopes(np.stack(real), vm, step)
    Es = plane_envelopes(np.stack(sur), vm, step)
    fin = inside & vm & np.all([np.all(np.isfinite(e), axis=0) for e in Er + Es], axis=0)
    return {"wid": w["wid"], "real": features(Er, fin, block), "null": features(Es, fin, block),
            "n_steps": int(om.shape[0]), "n_pix": int(fin.sum())}


def stage_plane(source: str, year: int, season: str, workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"plane_{source}_{year}_{season}.json"
    if target.exists():
        print("cached", target.name, flush=True)
        return
    if source == "era5":
        lat, lon, u, v, t, valid = b34.load_era5(year, season)
    else:
        lat, lon, u, v, t, valid = b34.load_ifshr(season)
    idx = np.arange(0, u.shape[0], STRIDE)
    u, v = u[idx], v[idx]
    om = np.concatenate([b34.vorticity_sphere(u[i:i + 24], v[i:i + 24], lat, lon) for i in range(0, len(idx), 24)])
    om = np.where(valid[None], om, np.nan).astype(np.float32)
    del u, v
    tt = t[idx]
    block = (((tt - tt[0]).astype(int) // b34.BLOCK_DAYS) % 2).astype(int)
    W = b34.windows()
    seeds = np.random.SeedSequence(SEED + year * 10 + (0 if season == "JFM" else 1) + (5 if source == "ifshr" else 0)).generate_state(len(W))
    with mp.get_context("fork").Pool(workers, _init, (om, valid, lat, lon, block)) as pool:
        res = pool.map(_worker, [(i, w, int(s)) for i, (w, s) in enumerate(zip(W, seeds))])
    target.write_text(json.dumps({"source": source, "year": year, "season": season, "windows": res}), encoding="utf-8")
    print(f"B36_PLANE_{source}_{year}_{season}_DONE", flush=True)


# ---------------------------------------------------------------------------

def load_year(source: str, year: int) -> dict:
    W = b34.windows()
    per = [{r["wid"]: r for r in json.loads((OUT / f"plane_{source}_{year}_{s}.json").read_text())["windows"]}
           for s in b34.MONTHS]
    out = {}
    for kind in ("real", "null"):
        for f in ("P4", "P5", "s", "r_mov"):
            out[f"{f}_{kind}"] = np.mean([[p[w["wid"]][kind][f] for w in W] for p in per], axis=0)
    for f in ("P4", "P5", "s", "r_mov"):
        out[f"{f}_res"] = out[f"{f}_real"] - out[f"{f}_null"]
    for k in ("real", "null", "res"):
        out[f"P45_{k}"] = 0.5 * (out[f"P4_{k}"] + out[f"P5_{k}"])
    return out


def regions(Y, belts, rng, kind, k=4):
    from sklearn.metrics import adjusted_rand_score as ari
    feats = lambda y: np.column_stack([y[f"P4_{kind}"], y[f"P5_{kind}"], y[f"s_{kind}"], y[f"r_mov_{kind}"]])  # noqa: E731
    zs = lambda X, mu, sd: (X - mu) / sd  # noqa: E731
    X23, X24 = feats(Y[2023]), feats(Y[2024])
    l23 = b34.ward_labels(zs(X23, X23.mean(0), X23.std(0)), k)
    l24 = b34.ward_labels(zs(X24, X24.mean(0), X24.std(0)), k)
    a0 = float(ari(l23, l24))
    nul = np.array([ari(l23, b34.belt_perm(l24, belts, rng, cyclic=False)) for _ in range(N_PERM)])
    Xp = 0.5 * (X23 + X24); mu, sd = Xp.mean(0), Xp.std(0)
    lp = b34.ward_labels(zs(Xp, mu, sd), k)
    cen = np.array([zs(Xp, mu, sd)[lp == c].mean(0) for c in np.unique(lp)])
    X22 = zs(feats(Y[2022]), mu, sd)
    l22 = np.unique(lp)[np.argmin(((X22[:, None, :] - cen[None]) ** 2).sum(-1), axis=1)]
    a2 = float(ari(lp, l22))
    nul2 = np.array([ari(lp, b34.belt_perm(l22, belts, rng, cyclic=False)) for _ in range(N_PERM)])
    return {"ari_2023_2024": a0, "p_belt_perm": float((np.sum(nul >= a0) + 1) / (N_PERM + 1)),
            "ari_2022_vs_pooled": a2, "p_2022_belt_perm": float((np.sum(nul2 >= a2) + 1) / (N_PERM + 1)),
            "labels_pooled": lp.tolist()}


def stage_tests() -> dict:
    from scipy.stats import spearmanr
    from sklearn.metrics import adjusted_rand_score as ari
    sp = lambda a, b: float(spearmanr(a, b).statistic)  # noqa: E731
    rng = np.random.default_rng(SEED)
    W = b34.windows()
    belts = np.array([w["belt"] for w in W])
    Y = {y: load_year("era5", y) for y in (2022, 2023, 2024)}
    M = load_year("ifshr", 2014) if (OUT / "plane_ifshr_2014_JAS.json").exists() else None
    res = {"protocol": str(PROTOCOL), "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()}
    r23, r24 = Y[2023]["P45_res"], Y[2024]["P45_res"]
    rho = sp(r23, r24)
    null = np.array([sp(r23, b34.belt_perm(r24, belts, rng, cyclic=True)) for _ in range(N_PERM)])
    p = float((np.sum(null >= rho) + 1) / (N_PERM + 1))
    rmean = 0.5 * (r23 + r24)
    h1 = {"rho_2023_2024": rho, "p_belt_shift": p, "rho_2022_vs_mean": sp(Y[2022]["P45_res"], rmean),
          "median_R45": {str(y): float(np.median(Y[y]["P45_res"])) for y in Y},
          "median_P45_real": {str(y): float(np.median(Y[y]["P45_real"])) for y in Y},
          "median_P45_null": {str(y): float(np.median(Y[y]["P45_null"])) for y in Y},
          "reliability_other": {f: sp(Y[2023][f"{f}_res"], Y[2024][f"{f}_res"]) for f in ("P4", "P5", "s", "r_mov")}}
    h1["pass"] = bool(rho >= 0.60 and p < 0.01 and h1["rho_2022_vs_mean"] >= 0.60)
    res["H36_1"] = h1
    h2 = {"k": {str(k): regions(Y, belts, rng, "res", k) for k in (3, 4, 5)}}
    r4 = h2["k"]["4"]
    h2["pass"] = bool(r4["ari_2023_2024"] >= 0.40 and r4["p_belt_perm"] < 0.05 and
                      r4["ari_2022_vs_pooled"] >= 0.40 and r4["p_2022_belt_perm"] < 0.05)
    res["H36_2"] = h2
    raw4 = regions(Y, belts, rng, "real", 4)
    pr = 0.5 * (Y[2023]["P45_real"] + Y[2024]["P45_real"]); pn = 0.5 * (Y[2023]["P45_null"] + Y[2024]["P45_null"])
    r2 = float(np.corrcoef(pr, pn)[0, 1] ** 2)
    ph34 = np.array(json.loads((B34 / "summary.json").read_text())["H34_6"]["k"]["4"]["labels_pooled"])
    res["H36_3"] = {"R2_raw_P45_on_null": r2, "rho_raw_null_P45": sp(pr, pn),
                    "ari_residual_vs_rawplane": float(ari(r4["labels_pooled"], raw4["labels_pooled"])),
                    "ari_rawplane_vs_phase34": float(ari(raw4["labels_pooled"], ph34)),
                    "ari_residual_vs_phase34": float(ari(r4["labels_pooled"], ph34)),
                    "rawplane_regions": raw4,
                    "s_null_median": float(np.median(0.5 * (Y[2023]["s_null"] + Y[2024]["s_null"]))),
                    "s_real_median": float(np.median(0.5 * (Y[2023]["s_real"] + Y[2024]["s_real"])))}
    if M is not None:
        res["H36_4"] = {"rho_R45_model": sp(M["P45_res"], rmean), "pass": bool(sp(M["P45_res"], rmean) >= 0.50)}
    v = ("SPECTRAL_GEOGRAPHY" if not h1["pass"] else
         "RESIDUAL_REGIONS_CONFIRMED" if h2["pass"] else "RESIDUAL_WITHOUT_REGIONS")
    res["VERDICT"] = v
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez(OUT / "window_fields.npz", wid=np.array([w["wid"] for w in W]),
             **{f"{k}_{y}": Y[y][k] for y in Y for k in Y[y]},
             **({f"{k}_ifshr": M[k] for k in M} if M is not None else {}))
    print(json.dumps({"VERDICT": v}, indent=1), flush=True)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["plane", "tests"])
    ap.add_argument("--source", choices=["era5", "ifshr"], default="era5")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--season", choices=list(b34.MONTHS), default="JFM")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.stage == "plane":
        if a.source == "ifshr":
            for s in b34.MONTHS:
                stage_plane("ifshr", 2014, s, a.workers)
        else:
            stage_plane("era5", a.year, a.season, a.workers)
    else:
        stage_tests()


if __name__ == "__main__":
    main()
