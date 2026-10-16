#!/usr/bin/env python3
"""Phase 39: co-location and energy.

Frozen protocol: docs/PROTOCOL_PHASE39_COLOCATION_ENERGY.md (2026-10-16).

Stages:
    season --year Y --season S    flux, envelopes, band energy per window
    tests                         hypotheses and verdict
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

PROTOCOL = Path("docs/PROTOCOL_PHASE39_COLOCATION_ENERGY.md")
OUT = Path("clean_experiments/results/experiment_B39_colocation_energy")
B34 = Path("clean_experiments/results/experiment_B34_colocation_geography")
B36 = Path("clean_experiments/results/experiment_B36_residual_regions")
SEED = 20261020
STRIDE = 2
LAG = 10                          # retained steps (12 h) -> 5 days
CHUNK = 12
FLUX_ELLS = (400.0, 800.0)
PAIR_OF = {400.0: (0, 1), 800.0: (1, 2)}
A_M = b34.EARTH_M


def grad_sphere(f, lat, lon):
    phi = np.deg2rad(lat)[None, :, None]
    dlam = np.deg2rad(float(lon[1] - lon[0]))
    dfdx = (np.roll(f, -1, axis=-1) - np.roll(f, 1, axis=-1)) / (2 * dlam * A_M * np.cos(phi))
    dfdy = np.gradient(f, np.deg2rad(lat), axis=-2) / A_M
    return dfdx, dfdy


def flux(u, v, sm: b34.Smoother, lat, lon, ell):
    ub, vb = sm.bar(u, ell), sm.bar(v, ell)
    txx = sm.bar(u * u, ell) - ub * ub
    txy = sm.bar(u * v, ell) - ub * vb
    tyy = sm.bar(v * v, ell) - vb * vb
    dudx, dudy = grad_sphere(np.nan_to_num(ub), lat, lon)
    dvdx, dvdy = grad_sphere(np.nan_to_num(vb), lat, lon)
    return (-(txx * dudx + txy * (dudy + dvdx) + tyy * dvdy)).astype(np.float32)


def stage_season(year: int, season: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"season_{year}_{season}.json"
    if target.exists():
        print("cached", target.name, flush=True); return
    lat, lon, u, v, t, valid = b34.load_era5(year, season)
    idx = np.arange(0, u.shape[0], STRIDE)
    u, v = u[idx], v[idx]
    sm = b34.Smoother(lat, lon, valid)
    rows = np.abs(lat) <= b34.LAT_STAT + 1e-9
    nt = len(idx)
    E = [np.empty((nt, rows.sum(), len(lon)), np.float32) for _ in range(3)]
    PI = {ell: np.empty((nt, rows.sum(), len(lon)), np.float32) for ell in FLUX_ELLS}
    su = np.zeros((rows.sum(), len(lon))); sv = np.zeros_like(su); sk = np.zeros_like(su)
    for c0 in range(0, nt, CHUNK):
        sl = slice(c0, min(nt, c0 + CHUNK))
        uc = np.where(valid[None], u[sl], np.nan).astype(np.float32)
        vc = np.where(valid[None], v[sl], np.nan).astype(np.float32)
        env = b34.envelopes(b34.vorticity_sphere(u[sl], v[sl], lat, lon), sm)
        for k in range(3):
            E[k][sl] = env[k][:, rows]
        for ell in FLUX_ELLS:
            PI[ell][sl] = flux(uc, vc, sm, lat, lon, ell)[:, rows]
        ubd = sm.bar(uc, 200.0) - sm.bar(uc, 1600.0)
        vbd = sm.bar(vc, 200.0) - sm.bar(vc, 1600.0)
        su += np.nansum(ubd[:, rows], 0); sv += np.nansum(vbd[:, rows], 0)
        sk += np.nansum(0.5 * (ubd[:, rows] ** 2 + vbd[:, rows] ** 2), 0)
        print(f"[{year} {season}] {sl.stop}/{nt}", flush=True)
    del u, v
    latw = lat[rows]
    W = b34.windows()
    res = []
    for w in W:
        r_i, c_i = b34.window_index(latw, lon, w)
        vmask = valid[rows][np.ix_(r_i, c_i)]
        wt0 = (np.cos(np.deg2rad(latw[r_i]))[:, None] * np.ones(len(c_i))[None]) * vmask
        rec = {"wid": w["wid"]}
        for ell in FLUX_ELLS:
            k1, k2 = PAIR_OF[ell]
            P = PI[ell][:, r_i][:, :, c_i]
            A1 = E[k1][:, r_i][:, :, c_i]; A2 = E[k2][:, r_i][:, :, c_i]
            XtX = {"real": np.zeros((4, 4)), "lag": np.zeros((4, 4))}
            Xty = {"real": np.zeros(4), "lag": np.zeros(4)}
            joint = single = jw = swt = 0.0
            mean_pi = mean_abs = 0.0; nsum = 0.0

            def zstd(a, m):
                ww = wt0[m]; x = a[m]
                mu = (x * ww).sum() / ww.sum(); sd = np.sqrt(((x - mu) ** 2 * ww).sum() / ww.sum())
                return (x - mu) / max(sd, 1e-12), ww
            for ti in range(nt):
                for kind, tj in (("real", ti), ("lag", ti + LAG)):
                    if tj >= nt:
                        continue
                    pa = np.abs(P[ti])
                    m = (wt0 > 0) & np.isfinite(pa) & (pa > 0) & np.isfinite(A1[tj]) & np.isfinite(A2[tj])
                    if m.sum() < 50:
                        continue
                    y, ww = zstd(np.log(pa), m)
                    z1, _ = zstd(A1[tj], m); z2, _ = zstd(A2[tj], m)
                    X = np.column_stack([np.ones_like(z1), z1, z2, z1 * z2])
                    XtX[kind] += (X * ww[:, None]).T @ X
                    Xty[kind] += (X * ww[:, None]).T @ y
                    if kind == "real":
                        jm = (z1 > 0) & (z2 > 0); smk = (z1 > 0) ^ (z2 > 0)
                        joint += (pa[m][jm] * ww[jm]).sum(); jw += ww[jm].sum()
                        single += (pa[m][smk] * ww[smk]).sum(); swt += ww[smk].sum()
                        mean_pi += (P[ti][m] * ww).sum(); mean_abs += (pa[m] * ww).sum(); nsum += ww.sum()
            b_real = np.linalg.solve(XtX["real"], Xty["real"]); b_lag = np.linalg.solve(XtX["lag"], Xty["lag"])
            rec[f"L{int(ell)}"] = {"b_real": b_real.tolist(), "b_lag": b_lag.tolist(),
                                   "d": float(b_real[3] - b_lag[3]),
                                   "conc_ratio": float((joint / jw) / (single / swt)),
                                   "mean_pi": float(mean_pi / nsum), "mean_abs_pi": float(mean_abs / nsum)}
        mu_u = su[np.ix_(r_i, c_i)] / nt; mu_v = sv[np.ix_(r_i, c_i)] / nt
        kst = 0.5 * (mu_u ** 2 + mu_v ** 2); ktot = sk[np.ix_(r_i, c_i)] / nt
        rec["K_st"] = float((kst * wt0).sum() / wt0.sum()); rec["K"] = float((ktot * wt0).sum() / wt0.sum())
        rec["sigma"] = rec["K_st"] / rec["K"]
        res.append(rec)
    target.write_text(json.dumps({"year": year, "season": season, "n_steps": nt, "windows": res}), encoding="utf-8")
    print(f"B39_SEASON_{year}_{season}_DONE", flush=True)


# ---------------------------------------------------------------------------

def sign_flip(x, n=9999):
    x = np.asarray(x, float); rng = np.random.default_rng(SEED)
    null = (rng.choice([-1.0, 1.0], size=(n, x.size)) * x[None]).mean(1)
    return float((np.sum(null >= x.mean()) + 1) / (n + 1))


def partial_spearman(a, b, c):
    from scipy.stats import rankdata
    ra, rb, rc = rankdata(a), rankdata(b), rankdata(c)
    res = lambda y: y - np.polyval(np.polyfit(rc, y, 1), rc)  # noqa: E731
    return float(np.corrcoef(res(ra), res(rb))[0, 1])


def load_year(year):
    W = b34.windows()
    per = [{r["wid"]: r for r in json.loads((OUT / f"season_{year}_{s}.json").read_text())["windows"]} for s in b34.MONTHS]
    out = {}
    for ell in FLUX_ELLS:
        L = f"L{int(ell)}"
        out[f"d_{L}"] = np.mean([[p[w["wid"]][L]["d"] for w in W] for p in per], 0)
        out[f"b3_{L}"] = np.mean([[p[w["wid"]][L]["b_real"][3] for w in W] for p in per], 0)
        out[f"b3lag_{L}"] = np.mean([[p[w["wid"]][L]["b_lag"][3] for w in W] for p in per], 0)
        out[f"conc_{L}"] = np.mean([[p[w["wid"]][L]["conc_ratio"] for w in W] for p in per], 0)
        out[f"pi_{L}"] = np.mean([[p[w["wid"]][L]["mean_pi"] for w in W] for p in per], 0)
        out[f"abspi_{L}"] = np.mean([[p[w["wid"]][L]["mean_abs_pi"] for w in W] for p in per], 0)
    for k in ("K_st", "K", "sigma"):
        out[k] = np.mean([[p[w["wid"]][k] for w in W] for p in per], 0)
    return out


def stage_tests():
    from scipy.stats import spearmanr
    sp = lambda a, b: float(spearmanr(a, b).statistic)  # noqa: E731
    W = b34.windows(); abslat = np.abs([w["lat_c"] for w in W])
    Y = {y: load_year(y) for y in (2022, 2023, 2024)}
    res = {"protocol": str(PROTOCOL), "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()}
    h1 = {}
    ok = True
    for ell in FLUX_ELLS:
        L = f"L{int(ell)}"
        d = 0.5 * (Y[2023][f"d_{L}"] + Y[2024][f"d_{L}"])
        rec = {"median_d": float(np.median(d)), "p": sign_flip(d), "frac_pos": float(np.mean(d > 0)),
               "median_b3": float(np.median(0.5 * (Y[2023][f"b3_{L}"] + Y[2024][f"b3_{L}"]))),
               "median_b3_lag": float(np.median(0.5 * (Y[2023][f"b3lag_{L}"] + Y[2024][f"b3lag_{L}"]))),
               "fresh_median_d": float(np.median(Y[2022][f"d_{L}"])), "fresh_p": sign_flip(Y[2022][f"d_{L}"])}
        rec["pass"] = bool(rec["median_d"] > 0 and rec["p"] < 0.01 and rec["frac_pos"] >= 0.75 and
                           rec["fresh_median_d"] > 0 and rec["fresh_p"] < 0.01)
        ok &= rec["pass"]; h1[L] = rec
    h1["pass"] = bool(ok); res["H39_1"] = h1
    z34 = np.load(B34 / "window_fields.npz")
    s_mean = 0.5 * (z34["s_2023"] + z34["s_2024"]); sig = 0.5 * (Y[2023]["sigma"] + Y[2024]["sigma"])
    h2 = {"rho_s_sigma": sp(s_mean, sig), "partial_abslat": partial_spearman(s_mean, sig, abslat),
          "fresh_rho": sp(z34["s_2022"], Y[2022]["sigma"]), "median_sigma": float(np.median(sig)),
          "rho_sigma_abslat": sp(sig, abslat)}
    h2["pass"] = bool(h2["rho_s_sigma"] >= 0.50 and h2["partial_abslat"] >= 0.30 and h2["fresh_rho"] >= 0.50)
    res["H39_2"] = h2
    lab = np.array(json.loads((B34 / "summary.json").read_text())["H34_6"]["k"]["4"]["labels_pooled"])
    h3 = {"by_region": {}}
    for c in np.unique(lab):
        m = lab == c
        h3["by_region"][int(c)] = {"n": int(m.sum()), "sigma": float(sig[m].mean()),
                                   **{f"d_L{int(e)}": float((0.5 * (Y[2023][f"d_L{int(e)}"] + Y[2024][f"d_L{int(e)}"]))[m].mean()) for e in FLUX_ELLS},
                                   **{f"conc_L{int(e)}": float((0.5 * (Y[2023][f"conc_L{int(e)}"] + Y[2024][f"conc_L{int(e)}"]))[m].mean()) for e in FLUX_ELLS},
                                   **{f"pi_L{int(e)}": float((0.5 * (Y[2023][f"pi_L{int(e)}"] + Y[2024][f"pi_L{int(e)}"]))[m].mean()) for e in FLUX_ELLS}}
    for e in FLUX_ELLS:
        L = f"L{int(e)}"
        h3[f"median_conc_{L}"] = float(np.median(0.5 * (Y[2023][f"conc_{L}"] + Y[2024][f"conc_{L}"])))
        h3[f"frac_forward_{L}"] = float(np.mean(0.5 * (Y[2023][f"pi_{L}"] + Y[2024][f"pi_{L}"]) > 0))
    if (B36 / "window_fields.npz").exists():
        z36 = np.load(B36 / "window_fields.npz"); r45 = 0.5 * (z36["P45_res_2023"] + z36["P45_res_2024"])
        K = 0.5 * (Y[2023]["K"] + Y[2024]["K"])
        for e in FLUX_ELLS:
            L = f"L{int(e)}"
            eff = 0.5 * (Y[2023][f"abspi_{L}"] + Y[2024][f"abspi_{L}"]) / K
            h3[f"rho_R45_efficiency_{L}"] = sp(r45, eff)
            h3[f"rho_R45_signed_pi_{L}"] = sp(r45, 0.5 * (Y[2023][f"pi_{L}"] + Y[2024][f"pi_{L}"]))
            h3[f"rho_R45_d_{L}"] = sp(r45, 0.5 * (Y[2023][f"d_{L}"] + Y[2024][f"d_{L}"]))
    res["H39_3"] = h3
    v = ("ENERGY_LOCALITY_AND_STATIONARY_ENERGY" if h1["pass"] and h2["pass"] else
         "ENERGY_LOCALITY_ONLY" if h1["pass"] else "STATIONARY_ENERGY_ONLY" if h2["pass"] else "NEGATIVE")
    res["VERDICT"] = v
    (OUT / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez(OUT / "window_fields.npz", wid=np.array([w["wid"] for w in W]),
             **{f"{k}_{y}": Y[y][k] for y in Y for k in Y[y]})
    print(json.dumps({"VERDICT": v}, indent=1), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["season", "tests"])
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--season", choices=list(b34.MONTHS), default="JFM")
    a = ap.parse_args()
    stage_season(a.year, a.season) if a.stage == "season" else stage_tests()


if __name__ == "__main__":
    main()
