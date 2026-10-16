#!/usr/bin/env python3
"""Phase 41: the enstrophy budget of the fine band against the geography of P.

Frozen protocol: docs/PROTOCOL_PHASE41_ENSTROPHY_BUDGET.md (2026-10-16).

Stages:
    season --year Y --season S    single pass, per-window weighted sums
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
import experiment_B40_alignment as b40  # noqa: E402

PROTOCOL = Path("docs/PROTOCOL_PHASE41_ENSTROPHY_BUDGET.md")
OUT = Path("clean_experiments/results/experiment_B41_enstrophy_budget")
B34 = Path("clean_experiments/results/experiment_B34_colocation_geography")
B36 = Path("clean_experiments/results/experiment_B36_residual_regions")
SEED = 20261031
STRIDE = 2
CHUNK = 8
ELLS = (400.0, 800.0)
FINE_OF = {400.0: 200.0, 800.0: 400.0}
OMEGA_E = 7.2921e-5
EPS = 1e-300
grad_sphere = b39.grad_sphere

KEYS = ("w", "V", "Z_in", "Z_out", "S_div", "A_div", "S_f", "A_f", "S_nl", "A_nl")


def divergence_sphere(u, v, lat, lon):
    """Horizontal divergence on the sphere, centred differences, periodic in longitude."""
    phi = np.deg2rad(lat)[None, :, None]
    dlam = np.deg2rad(float(lon[1] - lon[0]))
    du_dlam = (np.roll(u, -1, axis=-1) - np.roll(u, 1, axis=-1)) / (2.0 * dlam)
    dvcos_dphi = np.gradient(v * np.cos(phi), np.deg2rad(lat), axis=-2)
    return ((du_dlam + dvcos_dphi) / (b34.EARTH_M * np.cos(phi))).astype(np.float32)


def enstrophy_flux(u, v, om, sm, lat, lon, ell):
    ub, vb, wb = sm.bar(u, ell), sm.bar(v, ell), sm.bar(om, ell)
    wx, wy = grad_sphere(np.nan_to_num(wb), lat, lon)
    tw_x = sm.bar(u * om, ell) - ub * wb
    tw_y = sm.bar(v * om, ell) - vb * wb
    return -(tw_x * wx + tw_y * wy)


def budget_fields(u, v, om, dl, fcor, sm, lat, lon, ell):
    """Fine-band enstrophy budget terms for the band (ell/2, ell) on a chunk."""
    ellf = FINE_OF[ell]
    om_f = sm.bar(om, ellf) - sm.bar(om, ell)
    G_f = fcor[None] * dl                 # planetary stretching f delta
    G_nl = om * dl                        # nonlinear stretching zeta delta
    B = lambda g: sm.bar(g, ellf) - sm.bar(g, ell)  # noqa: E731
    bf, bnl = B(G_f), B(G_nl)
    return dict(V=om_f ** 2,
                Z_in=enstrophy_flux(u, v, om, sm, lat, lon, ell),
                Z_out=enstrophy_flux(u, v, om, sm, lat, lon, ellf),
                S_div=-om_f * (bf + bnl), A_div=np.abs(om_f * (bf + bnl)),
                S_f=-om_f * bf, A_f=np.abs(om_f * bf),
                S_nl=-om_f * bnl, A_nl=np.abs(om_f * bnl))


def run_season(lat, lon, u, v, valid, tag: str, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"season_{tag}.json"
    if target.exists():
        print("cached", target.name, flush=True); return target
    sm = b34.Smoother(lat, lon, valid)
    rows = np.abs(lat) <= b34.LAT_STAT + 1e-9
    ridx = np.arange(0, u.shape[0], STRIDE)
    nt = len(ridx)
    fcor = (2 * OMEGA_E * np.sin(np.deg2rad(lat)))[:, None] * np.ones((1, len(lon)))
    W = b34.windows(); latw = lat[rows]
    widx = {w["wid"]: b34.window_index(latw, lon, w) for w in W}
    wt = {}
    for w in W:
        r_i, c_i = widx[w["wid"]]
        vmask = valid[rows][np.ix_(r_i, c_i)]
        wt[w["wid"]] = (np.cos(np.deg2rad(latw[r_i]))[:, None] * np.ones(len(c_i))[None]) * vmask
    sums = {w["wid"]: {ell: {k: 0.0 for k in KEYS} for ell in ELLS} for w in W}
    for c0 in range(0, nt, CHUNK):
        sl = slice(c0, min(nt, c0 + CHUNK)); raw = ridx[sl]
        uc = np.where(valid[None], u[raw], np.nan).astype(np.float32)
        vc = np.where(valid[None], v[raw], np.nan).astype(np.float32)
        om = np.where(valid[None], b34.vorticity_sphere(u[raw], v[raw], lat, lon), np.nan).astype(np.float32)
        dl = np.where(valid[None], divergence_sphere(u[raw], v[raw], lat, lon), np.nan).astype(np.float32)
        for ell in ELLS:
            F = budget_fields(uc, vc, om, dl, fcor.astype(np.float32), sm, lat, lon, ell)
            F = {k: a[:, rows] for k, a in F.items()}
            for w in W:
                r_i, c_i = widx[w["wid"]]; wt0 = wt[w["wid"]]
                S = sums[w["wid"]][ell]
                for j in range(len(raw)):
                    Fw = {k: a[j][np.ix_(r_i, c_i)].astype(float) for k, a in F.items()}
                    m = wt0 > 0
                    for k in Fw:
                        m &= np.isfinite(Fw[k])
                    if m.sum() < 50:
                        continue
                    ww = wt0[m]
                    S["w"] += ww.sum()
                    for k in KEYS[1:]:
                        S[k] += (ww * Fw[k][m]).sum()
        print(f"[{tag}] {sl.stop}/{nt}", flush=True)
    res = []
    for w in W:
        rec = {"wid": w["wid"]}
        for ell in ELLS:
            S = sums[w["wid"]][ell]; ww = max(S["w"], EPS)
            means = {k: S[k] / ww for k in KEYS[1:]}
            V = max(means["V"], EPS)
            rec[f"L{int(ell)}"] = {**means,
                                   "r_casc": means["Z_in"] / V, "r_out": means["Z_out"] / V,
                                   "r_div": means["A_div"] / V, "r_f": means["A_f"] / V, "r_nl": means["A_nl"] / V,
                                   "share_div": means["A_div"] / V / (means["A_div"] / V + abs(means["Z_in"] / V) + EPS)}
        res.append(rec)
    target.write_text(json.dumps({"tag": tag, "n_steps": nt, "windows": res}), encoding="utf-8")
    print(f"B41_SEASON_{tag}_DONE", flush=True)
    return target


def stage_season(year: int, season: str) -> None:
    lat, lon, u, v, t, valid = b34.load_era5(year, season)
    run_season(lat, lon, u, v, valid, f"{year}_{season}", OUT)


# ---------------------------------------------------------------------------

STATS = ("r_casc", "r_out", "r_div", "r_f", "r_nl", "share_div", "S_div", "S_f", "S_nl", "V", "Z_in", "Z_out")


def load_year(files: dict, year) -> dict:
    W = b34.windows()
    per = [{r["wid"]: r for r in json.loads(Path(files[(year, s)]).read_text())["windows"]} for s in b34.MONTHS]
    out = {}
    for ell in ELLS:
        L = f"L{int(ell)}"
        for k in STATS:
            out[f"{k}_{L}"] = np.nanmean([[p[w["wid"]][L][k] for w in W] for p in per], 0)
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
    z34 = np.load(B34 / "window_fields.npz"); z36 = np.load(B36 / "window_fields.npz")
    P45 = 0.5 * (z34["P45_2023"] + z34["P45_2024"]); P45f = z34["P45_2022"]
    P45res = 0.5 * (z36["P45_res_2023"] + z36["P45_res_2024"])

    def geo(key, sign, bar_main=0.40, bar_res=0.30, bar_fresh=0.30):
        out = {}
        for ell in ELLS:
            L = f"L{int(ell)}"; x = main(f"{key}_{L}")
            rec = {"rho_P45": sp(x, P45), "rho_P45_res": sp(x, P45res), "fresh_rho_P45": sp(Y[2022][f"{key}_{L}"], P45f),
                   "expected_sign": sign}
            rec["pass"] = bool(sign * rec["rho_P45"] >= bar_main and sign * rec["rho_P45_res"] >= bar_res
                               and sign * rec["fresh_rho_P45"] >= bar_fresh)
            out[L] = rec
        out["pass"] = bool(all(out[f"L{int(e)}"]["pass"] for e in ELLS))
        return out

    res["H41_1"] = geo("r_casc", +1)
    res["H41_2"] = geo("r_div", -1)
    res["H41_2_reported"] = {k: geo(k, -1) for k in ("share_div", "r_f", "r_nl")}
    h3 = {}
    try:
        lab = np.array(json.loads((B34 / "summary.json").read_text())["H34_6"]["k"]["4"]["labels_pooled"])
        for ell in ELLS:
            L = f"L{int(ell)}"
            h3[L] = {"median_r_casc": float(np.nanmedian(main(f"r_casc_{L}"))),
                     "median_r_div": float(np.nanmedian(main(f"r_div_{L}"))),
                     "median_share_div": float(np.nanmedian(main(f"share_div_{L}"))),
                     "median_r_out_over_r_casc": float(np.nanmedian(main(f"r_out_{L}") / main(f"r_casc_{L}"))),
                     "frac_windows_S_div_pos": float(np.nanmean(main(f"S_div_{L}") > 0)),
                     "frac_windows_Z_in_pos": float(np.nanmean(main(f"Z_in_{L}") > 0)),
                     "median_r_f_over_r_nl": float(np.nanmedian(main(f"r_f_{L}") / main(f"r_nl_{L}"))),
                     "by_region": {int(c): {"n": int((lab == c).sum()),
                                            "r_casc": float(np.nanmean(main(f"r_casc_{L}")[lab == c])),
                                            "r_div": float(np.nanmean(main(f"r_div_{L}")[lab == c])),
                                            "share_div": float(np.nanmean(main(f"share_div_{L}")[lab == c]))}
                                   for c in np.unique(lab)}}
    except FileNotFoundError as e:
        h3 = {"error": str(e)}
    res["H41_3"] = h3
    v = ("BUDGET_EXPLAINS" if res["H41_1"]["pass"] and res["H41_2"]["pass"] else
         "CASCADE_ONLY" if res["H41_1"]["pass"] else "DIVERGENCE_ONLY" if res["H41_2"]["pass"] else "NEGATIVE")
    res["VERDICT"] = v
    (out_dir / "summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez(out_dir / "window_fields.npz", wid=np.array([w["wid"] for w in b34.windows()]),
             **{f"{k}_{y}": Y[y][k] for y in Y for k in Y[y]})
    print(json.dumps({"VERDICT": v, "H41_1": res["H41_1"]["pass"], "H41_2": res["H41_2"]["pass"]}, indent=1), flush=True)
    return res


def stage_smoke() -> None:
    out_dir = OUT / "smoke"
    lat, lon, u, v, valid = b40.synthetic_fields(nt_raw=6)
    f = run_season(lat, lon, u, v, valid, "smoke", out_dir)
    files = {(y, s): f for y in (2022, 2023, 2024) for s in b34.MONTHS}
    res = stage_tests(files, smoke=True)
    rec = json.loads(f.read_text())["windows"][0]["L400"]
    print(json.dumps({k: rec[k] for k in ("r_casc", "r_out", "r_div", "r_f", "r_nl", "share_div")}, indent=1))
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
