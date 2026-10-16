#!/usr/bin/env python3
"""Phase 40: the Clark term and strain alignment on the Phase-39 pipeline.

Frozen protocol: docs/PROTOCOL_PHASE40_ALIGNMENT.md (2026-10-16). The code reuses the Phase-34 smoother, envelopes and windows and the
Phase-39 flux and gradient operators.

Stages:
    season --year Y --season S    two passes: envelopes, then per-chunk fields with
                                  per-window accumulators (no full fields stored)
    tests                         hypotheses and verdict on the three years
    smoke                         synthetic sphere, both stages, no data needed

Run from the repository root.
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

import experiment_B34_colocation_geography as b34  # noqa: E402
import experiment_B39_colocation_energy as b39  # noqa: E402

PROTOCOL = Path("docs/PROTOCOL_PHASE40_ALIGNMENT.md")
OUT = Path("clean_experiments/results/experiment_B40_alignment")
B34 = Path("clean_experiments/results/experiment_B34_colocation_geography")
B36 = Path("clean_experiments/results/experiment_B36_residual_regions")
SEED = 20261030
STRIDE = 2
LAG = 10                           # retained steps (12 h) -> 5 days
CHUNK = 8                          # retained steps per chunk
DT_NEIGHBOUR_S = 6.0 * 3600.0      # 6 h between raw steps
FLUX_ELLS = (400.0, 800.0)
PAIR_OF = {400.0: (0, 1), 800.0: (1, 2)}
FINE_OF = {400.0: 200.0, 800.0: 400.0}
R_EDGES = np.linspace(-3.0, 3.0, 13)   # bins of log|r| for the LKH curve
EPS = 1e-30
grad_sphere = b39.grad_sphere


# ---------------------------------------------------------------------------
# fields on the sphere
# ---------------------------------------------------------------------------

def wrap_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def strain_axis(ub, vb, lat, lon):
    """Coarse gradient components and the extensional axis angle from filtered winds."""
    dudx, dudy = grad_sphere(np.nan_to_num(ub), lat, lon)
    dvdx, dvdy = grad_sphere(np.nan_to_num(vb), lat, lon)
    sn, ss = dudx - dvdy, dudy + dvdx
    return dict(dudx=dudx, dudy=dudy, dvdx=dvdx, dvdy=dvdy, sn=sn, ss=ss,
                om=dvdx - dudy, dl=dudx + dvdy, sig=np.hypot(sn, ss),
                thS=0.5 * np.arctan2(ss, sn))


def grad_angle_sphere(theta, lat, lon):
    c2, s2 = np.cos(2 * theta), np.sin(2 * theta)
    cx, cy = grad_sphere(c2, lat, lon)
    sx, sy = grad_sphere(s2, lat, lon)
    return 0.5 * (c2 * sx - s2 * cx), 0.5 * (c2 * sy - s2 * cy)


def bundle_sphere(u, v, om, sm, lat, lon, ell, th_prev, th_next):
    """All Phase-40 fields for one filter scale on a chunk (nt, ny, nx).

    u, v, om: raw fields at the retained steps (NaN outside the valid mask);
    th_prev/th_next: strain-axis angle at the 6 h neighbours of each retained step.
    """
    ub, vb = sm.bar(u, ell), sm.bar(v, ell)
    G = strain_axis(ub, vb, lat, lon)
    txx = sm.bar(u * u, ell) - ub * ub
    txy = sm.bar(u * v, ell) - ub * vb
    tyy = sm.bar(v * v, ell) - vb * vb
    tr = txx + tyy
    k = 0.5 * tr
    c = np.hypot(txx - tyy, 2 * txy) / np.where(tr > 0, tr, np.nan)
    thT = 0.5 * np.arctan2(2 * txy, txx - tyy)
    Pi = -(txx * G["dudx"] + txy * (G["dudy"] + G["dvdx"]) + tyy * G["dvdy"])
    Pi_iso = -k * G["dl"]
    e2 = (ell * 1000.0) ** 2 / 12.0
    tC_xx = e2 * (G["dudx"] ** 2 + G["dudy"] ** 2)
    tC_xy = e2 * (G["dudx"] * G["dvdx"] + G["dudy"] * G["dvdy"])
    tC_yy = e2 * (G["dvdx"] ** 2 + G["dvdy"] ** 2)
    Pi_C = -(tC_xx * G["dudx"] + tC_xy * (G["dudy"] + G["dvdx"]) + tC_yy * G["dvdy"])
    # enstrophy flux
    wb = sm.bar(om, ell)
    wx, wy = grad_sphere(np.nan_to_num(wb), lat, lon)
    tw_x = sm.bar(u * om, ell) - ub * wb
    tw_y = sm.bar(v * om, ell) - vb * wb
    Z = -(tw_x * wx + tw_y * wy)
    Z_C = -e2 * (0.5 * G["sn"] * (wx ** 2 - wy ** 2) + G["ss"] * wx * wy
                 + 0.5 * G["dl"] * (wx ** 2 + wy ** 2))
    a_same = -np.cos(2 * (np.arctan2(wy, wx) - G["thS"]))
    wf = sm.bar(om, FINE_OF[ell])
    wfx, wfy = grad_sphere(np.nan_to_num(wf), lat, lon)
    a_fine = -np.cos(2 * (np.arctan2(wfy, wfx) - G["thS"]))
    # LKH ratio
    thx, thy = grad_angle_sphere(G["thS"], lat, lon)
    dth_dt = wrap_pi(2 * th_next - 2 * th_prev) / (2 * DT_NEIGHBOUR_S) * 0.5
    Dth = dth_dt + np.nan_to_num(ub) * thx + np.nan_to_num(vb) * thy
    r = (G["om"] + 2 * Dth) / np.where(G["sig"] > 0, G["sig"], np.nan)
    f32 = lambda a: np.asarray(a, np.float32)  # noqa: E731
    return dict(Pi=f32(Pi), Pi_iso=f32(Pi_iso), Pi_C=f32(Pi_C), c=f32(c),
                cos2dth=f32(np.cos(2 * (thT - G["thS"]))), Z=f32(Z), Z_C=f32(Z_C),
                a_same=f32(a_same), a_fine=f32(a_fine), r=f32(r))


# ---------------------------------------------------------------------------
# per-window accumulators
# ---------------------------------------------------------------------------

class Acc:
    """Pooled weighted sums for one window and one scale."""

    def __init__(self):
        self.corr = {name: np.zeros(6) for name in ("ZZC", "PiPiC")}   # w, x, y, xx, yy, xy
        self.reg = {}                                                    # name -> [XtX, Xty, n]
        self.s = dict(w=0.0, lkh=0.0, a_fine=0.0, a_same=0.0, abs_iso=0.0, abs_pi=0.0,
                      z=0.0, abs_z=0.0, z_pos=0.0)
        self.curve = np.zeros((2, len(R_EDGES) - 1))                     # sum w*a_fine, sum w

    def add_corr(self, name, x, y, w):
        self.corr[name] += [w.sum(), (w * x).sum(), (w * y).sum(),
                            (w * x * x).sum(), (w * y * y).sum(), (w * x * y).sum()]

    def add_reg(self, name, X, y, w):
        if name not in self.reg:
            self.reg[name] = [np.zeros((X.shape[1], X.shape[1])), np.zeros(X.shape[1]), 0]
        Xw = X * w[:, None]
        self.reg[name][0] += Xw.T @ X
        self.reg[name][1] += Xw.T @ y
        self.reg[name][2] += 1

    def finish(self):
        out = {}
        for name, s in self.corr.items():
            w, sx, sy, sxx, syy, sxy = s
            if w <= 0:
                out[f"corr_{name}"] = float("nan"); continue
            mx, my = sx / w, sy / w
            vx, vy = sxx / w - mx * mx, syy / w - my * my
            out[f"corr_{name}"] = float((sxy / w - mx * my) / np.sqrt(max(vx * vy, 1e-300)))
        for name, (XtX, Xty, n) in self.reg.items():
            try:
                b = np.linalg.solve(XtX, Xty)
            except np.linalg.LinAlgError:
                b = np.full(len(Xty), np.nan)
            out[f"b_{name}"] = b.tolist()
            out[f"n_{name}"] = n
        w = max(self.s["w"], EPS)
        out.update(f_lkh=self.s["lkh"] / w, mean_a_fine=self.s["a_fine"] / w,
                   mean_a_same=self.s["a_same"] / w,
                   f_iso=self.s["abs_iso"] / max(self.s["abs_pi"], EPS),
                   mean_z=self.s["z"] / w, mean_abs_z=self.s["abs_z"] / w,
                   frac_z_pos=self.s["z_pos"] / w)
        out["lkh_curve"] = (self.curve[0] / np.maximum(self.curve[1], EPS)).tolist()
        out["lkh_curve_w"] = self.curve[1].tolist()
        return out


def zstd(x, w):
    mu = (x * w).sum() / w.sum()
    sd = np.sqrt(((x - mu) ** 2 * w).sum() / w.sum())
    return (x - mu) / max(sd, 1e-12)


def accumulate(acc: Acc, F: dict, E_now, E_lag, wt0):
    """One retained step of one window: F = fields on the window, E_* = (A1, A2) envelopes."""
    base = (wt0 > 0)
    # fixed unit scaling (correlation is scale-invariant; avoids 1e-32 variances)
    for name, x, y, scale in (("ZZC", F["Z"], F["Z_C"], 1e15), ("PiPiC", F["Pi"], F["Pi_C"], 1e5)):
        m = base & np.isfinite(x) & np.isfinite(y)
        if m.sum() >= 50:
            acc.add_corr(name, scale * x[m].astype(float), scale * y[m].astype(float), wt0[m])
    for carrier, key, extra in (("Z", "Z", ("a_same",)), ("Pi", "Pi", ("c", "cos2dth"))):
        ya = np.abs(F[key])
        for kind, E in (("real", E_now), ("lag", E_lag)):
            if E is None:
                continue
            A1, A2 = E
            m = base & np.isfinite(ya) & (ya > 0) & np.isfinite(A1) & np.isfinite(A2)
            for ex in extra:
                m &= np.isfinite(F[ex]) & (np.abs(F[ex]) > 0)
            if m.sum() < 50:
                continue
            w = wt0[m]
            y = zstd(np.log(ya[m].astype(float)), w)
            z1, z2 = zstd(A1[m].astype(float), w), zstd(A2[m].astype(float), w)
            X = np.column_stack([np.ones_like(z1), z1, z2, z1 * z2])
            acc.add_reg(f"{carrier}_{kind}", X, y, w)
            cols = [zstd(np.log(np.abs(F[ex][m].astype(float))), w) for ex in extra]
            acc.add_reg(f"{carrier}_{kind}_aug", np.column_stack([X] + cols), y, w)
    m = base & np.isfinite(F["r"]) & np.isfinite(F["a_fine"]) & np.isfinite(F["Pi"]) & np.isfinite(F["Z"])
    if m.sum() >= 50:
        w = wt0[m]
        r = np.abs(F["r"][m].astype(float)); af = F["a_fine"][m].astype(float)
        acc.s["w"] += w.sum(); acc.s["lkh"] += (w * (r < 1)).sum()
        acc.s["a_fine"] += (w * af).sum(); acc.s["a_same"] += (w * F["a_same"][m]).sum()
        acc.s["abs_iso"] += (w * np.abs(F["Pi_iso"][m])).sum(); acc.s["abs_pi"] += (w * np.abs(F["Pi"][m])).sum()
        z = F["Z"][m].astype(float)
        acc.s["z"] += (w * z).sum(); acc.s["abs_z"] += (w * np.abs(z)).sum(); acc.s["z_pos"] += (w * (z > 0)).sum()
        idx = np.clip(np.searchsorted(R_EDGES, np.log(r + 1e-12), side="right") - 1, 0, len(R_EDGES) - 2)
        acc.curve[0] += np.bincount(idx, weights=w * af, minlength=len(R_EDGES) - 1)
        acc.curve[1] += np.bincount(idx, weights=w, minlength=len(R_EDGES) - 1)


# ---------------------------------------------------------------------------
# season stage
# ---------------------------------------------------------------------------

def run_season(lat, lon, u, v, valid, tag: str, out_dir: Path) -> Path:
    """u, v raw 6-hourly (nt_raw, ny, nx); statistics on every STRIDE-th step."""
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"season_{tag}.json"
    if target.exists():
        print("cached", target.name, flush=True); return target
    sm = b34.Smoother(lat, lon, valid)
    rows = np.abs(lat) <= b34.LAT_STAT + 1e-9
    ridx = np.arange(0, u.shape[0], STRIDE)
    nt = len(ridx)
    # pass 1: envelopes at the retained steps (as Phase 39)
    E = [np.empty((nt, rows.sum(), len(lon)), np.float32) for _ in range(3)]
    for c0 in range(0, nt, CHUNK):
        sl = slice(c0, min(nt, c0 + CHUNK)); raw = ridx[sl]
        env = b34.envelopes(b34.vorticity_sphere(u[raw], v[raw], lat, lon), sm)
        for k in range(3):
            E[k][sl] = env[k][:, rows]
        print(f"[{tag}] envelopes {sl.stop}/{nt}", flush=True)
    W = b34.windows(); latw = lat[rows]
    accs = {w["wid"]: {ell: Acc() for ell in FLUX_ELLS} for w in W}
    widx = {w["wid"]: b34.window_index(latw, lon, w) for w in W}
    wt = {}
    for w in W:
        r_i, c_i = widx[w["wid"]]
        vmask = valid[rows][np.ix_(r_i, c_i)]
        wt[w["wid"]] = (np.cos(np.deg2rad(latw[r_i]))[:, None] * np.ones(len(c_i))[None]) * vmask
    # pass 2: fields per chunk, accumulate per window
    for c0 in range(0, nt, CHUNK):
        sl = slice(c0, min(nt, c0 + CHUNK)); raw = ridx[sl]
        prev = np.clip(raw - 1, 0, u.shape[0] - 1); nxt = np.clip(raw + 1, 0, u.shape[0] - 1)
        uc = np.where(valid[None], u[raw], np.nan).astype(np.float32)
        vc = np.where(valid[None], v[raw], np.nan).astype(np.float32)
        om = np.where(valid[None], b34.vorticity_sphere(u[raw], v[raw], lat, lon), np.nan).astype(np.float32)
        fields = {}
        for ell in FLUX_ELLS:
            thp = strain_axis(sm.bar(u[prev].astype(np.float32), ell), sm.bar(v[prev].astype(np.float32), ell), lat, lon)["thS"]
            thn = strain_axis(sm.bar(u[nxt].astype(np.float32), ell), sm.bar(v[nxt].astype(np.float32), ell), lat, lon)["thS"]
            F = bundle_sphere(uc, vc, om, sm, lat, lon, ell, thp, thn)
            fields[ell] = {k: a[:, rows] for k, a in F.items()}
        for w in W:
            r_i, c_i = widx[w["wid"]]; wt0 = wt[w["wid"]]
            for ell in FLUX_ELLS:
                k1, k2 = PAIR_OF[ell]
                for j, ti in enumerate(range(sl.start, sl.stop)):
                    Fw = {k: a[j][np.ix_(r_i, c_i)] for k, a in fields[ell].items()}
                    E_now = (E[k1][ti][np.ix_(r_i, c_i)], E[k2][ti][np.ix_(r_i, c_i)])
                    tl = ti + LAG
                    E_lag = (E[k1][tl][np.ix_(r_i, c_i)], E[k2][tl][np.ix_(r_i, c_i)]) if tl < nt else None
                    accumulate(accs[w["wid"]][ell], Fw, E_now, E_lag, wt0)
        print(f"[{tag}] fields {sl.stop}/{nt}", flush=True)
    res = [{"wid": w["wid"], **{f"L{int(ell)}": accs[w["wid"]][ell].finish() for ell in FLUX_ELLS}} for w in W]
    target.write_text(json.dumps({"tag": tag, "n_steps": nt, "windows": res}), encoding="utf-8")
    print(f"B40_SEASON_{tag}_DONE", flush=True)
    return target


def stage_season(year: int, season: str) -> None:
    lat, lon, u, v, t, valid = b34.load_era5(year, season)
    run_season(lat, lon, u, v, valid, f"{year}_{season}", OUT)


# ---------------------------------------------------------------------------
# tests stage
# ---------------------------------------------------------------------------

def _interaction(rec, carrier):
    """d = b3_real - b3_lag and its augmented counterpart; NaN when the lag fit is missing."""
    g = lambda k, i: (rec[f"b_{carrier}_{k}"][i] if f"b_{carrier}_{k}" in rec else np.nan)  # noqa: E731
    return g("real", 3) - g("lag", 3), g("real_aug", 3) - g("lag_aug", 3)


def load_year(files: dict, year) -> dict:
    W = b34.windows()
    per = [{r["wid"]: r for r in json.loads(Path(files[(year, s)]).read_text())["windows"]} for s in b34.MONTHS]
    out = {}
    for ell in FLUX_ELLS:
        L = f"L{int(ell)}"
        stat = lambda f: np.nanmean([[f(p[w["wid"]][L]) for w in W] for p in per], 0)  # noqa: E731
        out[f"corrZ_{L}"] = stat(lambda r: r["corr_ZZC"])
        out[f"corrPi_{L}"] = stat(lambda r: r["corr_PiPiC"])
        for car in ("Z", "Pi"):
            out[f"d{car}_{L}"] = stat(lambda r, c=car: _interaction(r, c)[0])
            out[f"d{car}aug_{L}"] = stat(lambda r, c=car: _interaction(r, c)[1])
            out[f"b3{car}_{L}"] = stat(lambda r, c=car: r[f"b_{c}_real"][3])
        for k in ("f_lkh", "mean_a_fine", "mean_a_same", "f_iso", "mean_z", "mean_abs_z", "frac_z_pos"):
            out[f"{k}_{L}"] = stat(lambda r, k=k: r[k])
        out[f"lkh_curve_{L}"] = stat(lambda r: np.array(r["lkh_curve"]))
    return out


def stage_tests(files: dict | None = None, smoke: bool = False) -> dict:
    from scipy.stats import spearmanr
    sp = lambda a, b: float(spearmanr(a, b, nan_policy="omit").statistic)  # noqa: E731
    years = (2022, 2023, 2024)
    if files is None:
        files = {(y, s): OUT / f"season_{y}_{s}.json" for y in years for s in b34.MONTHS}
    out_dir = OUT / ("smoke" if smoke else "")
    out_dir.mkdir(parents=True, exist_ok=True)
    Y = {y: load_year(files, y) for y in years}
    main = lambda k: 0.5 * (Y[2023][k] + Y[2024][k])  # noqa: E731
    res = {"protocol": str(PROTOCOL), "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest(),
           "status": "FROZEN 2026-10-16" if not smoke else "SMOKE on synthetic fields"}
    # H40-1
    h1 = {}; ok1 = True
    for ell in FLUX_ELLS:
        L = f"L{int(ell)}"; cz = main(f"corrZ_{L}")
        rec = {"median_corr_ZZC": float(np.nanmedian(cz)), "frac_ge_0.7": float(np.nanmean(cz >= 0.7)),
               "fresh_median": float(np.nanmedian(Y[2022][f"corrZ_{L}"])),
               "median_corr_PiPiC": float(np.nanmedian(main(f"corrPi_{L}")))}
        rec["pass"] = bool(rec["median_corr_ZZC"] >= 0.7 and rec["frac_ge_0.7"] >= 0.75 and rec["fresh_median"] >= 0.7)
        ok1 &= rec["pass"]; h1[L] = rec
    h1["pass"] = bool(ok1); res["H40_1"] = h1
    # H40-2
    h2 = {}; ok2 = True
    for ell in FLUX_ELLS:
        L = f"L{int(ell)}"
        for car in ("Z", "Pi"):
            d, da = main(f"d{car}_{L}"), main(f"d{car}aug_{L}")
            red = (d - da) / np.where(np.abs(d) > 1e-9, d, np.nan)
            rec = {"median_d": float(np.nanmedian(d)), "median_d_aug": float(np.nanmedian(da)),
                   "median_reduction": float(np.nanmedian(red)), "p_sign": b39.sign_flip(np.nan_to_num(d - da)),
                   "frac_d_pos": float(np.nanmean(d > 0)),
                   "fresh_median_reduction": float(np.nanmedian((Y[2022][f"d{car}_{L}"] - Y[2022][f"d{car}aug_{L}"]) /
                                                                np.where(np.abs(Y[2022][f"d{car}_{L}"]) > 1e-9, Y[2022][f"d{car}_{L}"], np.nan))),
                   "fresh_p_sign": b39.sign_flip(np.nan_to_num(Y[2022][f"d{car}_{L}"] - Y[2022][f"d{car}aug_{L}"]))}
            rec["pass"] = bool(rec["median_reduction"] >= 0.5 and rec["median_d"] > 0 and rec["p_sign"] < 0.01
                               and rec["fresh_median_reduction"] >= 0.5 and rec["fresh_p_sign"] < 0.01)
            if car == "Z":
                ok2 &= rec["pass"]
            h2[f"{car}_{L}"] = rec
    h2["pass"] = bool(ok2); res["H40_2"] = h2
    # H40-3
    h3 = {}; n_pass = {400.0: 0, 800.0: 0}
    try:
        z34 = np.load(B34 / "window_fields.npz"); z36 = np.load(B36 / "window_fields.npz")
        P45 = 0.5 * (z34["P45_2023"] + z34["P45_2024"]); P45f = z34["P45_2022"]
        P45res = 0.5 * (z36["P45_res_2023"] + z36["P45_res_2024"])
        for ell in FLUX_ELLS:
            L = f"L{int(ell)}"
            for key, sign in (("f_lkh", +1), ("mean_a_fine", +1), ("f_iso", -1)):
                x = main(f"{key}_{L}")
                rec = {"rho_P45": sp(x, P45), "rho_P45_res": sp(x, P45res), "fresh_rho_P45": sp(Y[2022][f"{key}_{L}"], P45f),
                       "expected_sign": sign}
                rec["pass"] = bool(sign * rec["rho_P45"] >= 0.40 and sign * rec["rho_P45_res"] >= 0.30
                                   and sign * rec["fresh_rho_P45"] >= 0.30)
                n_pass[ell] += rec["pass"]; h3[f"{key}_{L}"] = rec
        h3["pass"] = bool(all(n >= 2 for n in n_pass.values()))
    except FileNotFoundError as e:
        h3 = {"pass": False, "error": f"Phase-34/36 window fields missing: {e}"}
    res["H40_3"] = h3
    # H40-4 reported
    h4 = {}
    for ell in FLUX_ELLS:
        L = f"L{int(ell)}"
        curve = 0.5 * (np.stack(Y[2023][f"lkh_curve_{L}"]) + np.stack(Y[2024][f"lkh_curve_{L}"]))
        h4[L] = {"lkh_curve_mean_over_windows": np.nanmean(curve, 0).tolist(),
                 "r_edges": R_EDGES.tolist(),
                 "median_frac_z_pos": float(np.nanmedian(main(f"frac_z_pos_{L}"))),
                 "frac_windows_forward_z": float(np.nanmean(main(f"mean_z_{L}") > 0)),
                 "median_f_iso": float(np.nanmedian(main(f"f_iso_{L}"))),
                 "median_f_lkh": float(np.nanmedian(main(f"f_lkh_{L}")))}
    res["H40_4"] = h4
    v = ("ALIGNMENT_EXPLAINS" if h1["pass"] and h2["pass"] and h3["pass"] else
         "CLARK_AND_INTERACTION_ONLY" if h1["pass"] and h2["pass"] else
         "CLARK_ONLY" if h1["pass"] else "NEGATIVE")
    res["VERDICT"] = v
    (out_dir / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez(out_dir / "window_fields.npz", wid=np.array([w["wid"] for w in b34.windows()]),
             **{f"{k}_{y}": (np.stack(Y[y][k]) if k.startswith("lkh_curve") else Y[y][k]) for y in Y for k in Y[y]})
    print(json.dumps({"VERDICT": v, "H40_1": h1["pass"], "H40_2": h2["pass"], "H40_3": h3["pass"]}, indent=1), flush=True)
    return res


# ---------------------------------------------------------------------------
# smoke stage: synthetic sphere, no data
# ---------------------------------------------------------------------------

def synthetic_fields(nt_raw=2 * (LAG + 4), dlat=1.0, seed=SEED):
    """Smooth random winds on a 1-degree sphere, 70S-70N, nt_raw 6-hourly steps."""
    rng = np.random.default_rng(seed)
    lat = np.arange(-70, 70 + 1e-9, dlat); lon = np.arange(0, 360, dlat)
    ny, nx = len(lat), len(lon)
    ky = np.fft.fftfreq(ny) * ny; kx = np.fft.rfftfreq(nx) * nx
    K = np.sqrt(kx[None] ** 2 + ky[:, None] ** 2)
    amp = np.where(K > 0, K ** -1.7, 0.0) * (K <= 40)
    phase_u = rng.standard_normal((ny, nx // 2 + 1)) + 1j * rng.standard_normal((ny, nx // 2 + 1))
    phase_v = rng.standard_normal((ny, nx // 2 + 1)) + 1j * rng.standard_normal((ny, nx // 2 + 1))
    u = np.empty((nt_raw, ny, nx), np.float32); v = np.empty_like(u)
    for t in range(nt_raw):
        rot = np.exp(1j * 0.15 * t * K)                 # slow dispersion in time
        drift = np.exp(-2j * np.pi * kx[None] * 0.02 * t)  # eastward advection
        uf = np.fft.irfft2(amp * phase_u * rot * drift, s=(ny, nx))
        vf = np.fft.irfft2(amp * phase_v * rot * drift, s=(ny, nx))
        s = 10.0 / np.sqrt(np.mean(uf ** 2 + vf ** 2))
        u[t], v[t] = s * uf, s * vf
    valid = np.ones((ny, nx), bool)
    valid[(lat > 20)[:, None] & (lat < 30)[:, None] & (lon > 60)[None] & (lon < 80)[None]] = False
    return lat, lon, u, v, valid


def stage_smoke() -> None:
    out_dir = OUT / "smoke"
    lat, lon, u, v, valid = synthetic_fields()
    f = run_season(lat, lon, u, v, valid, "smoke", out_dir)
    files = {(y, s): f for y in (2022, 2023, 2024) for s in b34.MONTHS}
    res = stage_tests(files, smoke=True)
    rec = json.loads(f.read_text())["windows"][0]["L400"]
    print(json.dumps({k: rec[k] for k in ("corr_ZZC", "corr_PiPiC", "f_lkh", "mean_a_fine", "f_iso", "b_Z_real", "b_Z_real_aug")}, indent=1))
    print("SMOKE_OK", res["VERDICT"])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["season", "tests", "smoke"])
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--season", choices=list(b34.MONTHS), default="JFM")
    a = ap.parse_args()
    {"season": lambda: stage_season(a.year, a.season), "tests": lambda: stage_tests(), "smoke": stage_smoke}[a.stage]()


if __name__ == "__main__":
    main()
