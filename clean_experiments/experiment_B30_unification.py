#!/usr/bin/env python3
"""Phase 30: unification — P as the normalised amplitude of the structure law,
the inherited share h and the own variance N.

Frozen spec: docs/PROTOCOL_PHASE30_UNIFICATION.md (2026-09-22). Inputs are the
Phase-27 shards (units, tiles) and the Phase-29 model shards; nothing new is
computed on the wind fields.

Stage: --stage tests
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from clean_experiments.experiment_B18_equal_km_regions import REGIONS  # noqa: E402
from clean_experiments.experiment_B2_scale_irreversibility import ELLS_KM  # noqa: E402
from clean_experiments.experiment_B20_p_geography import ELLS_FINE  # noqa: E402
from clean_experiments.experiment_B20_armB_global_map import (  # noqa: E402
    OROG_EXCL_M, SEASONS, rotate_cov, tile_covariates, tile_grid,
)
from clean_experiments.experiment_B27_mechanism import ARMB, OUT as B27, WINDOWS, _load  # noqa: E402
from clean_experiments.experiment_B29_structure_law import MODEL_WINDOWS, OUT as B29  # noqa: E402

CONFIG_VERSION = "b30_v1"
PROTOCOL = Path("docs/PROTOCOL_PHASE30_UNIFICATION.md")
SEED = 20260922
OUT = _HERE / "results" / "experiment_B30_unification"
N_ROT = 999
ROT_MIN_DEG = 30.0
N_BOOT = 1999
BLOCK_DEG = 30.0


# --------------------------------------------------------------------------
# per-shard quantities
# --------------------------------------------------------------------------

def quantities(d: dict, n_env: int, h: int = 0) -> dict:
    """G_hat, C_adj, V, V_s, O, h, P_obs, P_pred, N for one shard at half h."""
    idx = {k: i for i, k in enumerate(d["names"])}
    real = np.asarray(d["real"])[:, h]
    sur = np.asarray(d["sur_med"])[:, h]
    anch = {k: real[i] - sur[i] for k, i in idx.items()}
    q: dict = {"G": {}, "Cadj": {}, "V": {}, "Vs": {}, "O": {}, "h": {}, "Pobs": {},
               "Ppred": {}, "N": {}}
    for j in range(2, n_env):
        q["G"][j] = float(np.mean([anch[f"cov_{k}_{j}"] for k in range(0, j - 1)]))
    for i in range(n_env):
        q["V"][i] = float(real[idx[f"cov_{i}_{i}"]])
        q["Vs"][i] = float(sur[idx[f"cov_{i}_{i}"]])
    for i in range(n_env - 1):
        j = i + 1
        q["Cadj"][i] = float(anch[f"cov_{i}_{j}"])
        q["O"][i] = float(sur[idx[f"cov_{i}_{j}"]])
        q["Pobs"][i] = float(anch[f"rho_{i}_{j}"])
        if j in q["G"]:
            G = q["G"][j]
            q["h"][i] = G / np.sqrt(q["V"][i] * q["V"][j])
            q["Ppred"][i] = ((G + q["O"][i]) / np.sqrt(q["V"][i] * q["V"][j])
                             - q["O"][i] / np.sqrt(q["Vs"][i] * q["Vs"][j]))
    for j in q["G"]:
        q["N"][j] = (q["V"][j] - q["Vs"][j]) - q["G"][j]
    return q


def law_on_adjacent(shards: list[dict], n_env: int) -> dict:
    c, g = [], []
    for d in shards:
        q = quantities(d, n_env)
        for i in q["h"]:
            c.append(q["Cadj"][i]); g.append(q["G"][i + 1])
    c, g = np.asarray(c), np.asarray(g)
    return {"n": len(c), "pearson_r": float(pearsonr(c, g)[0]),
            "median_ratio": float(np.median(c / g)), "_ratio": c / g}


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def stage_tests() -> dict:
    rng = np.random.default_rng(SEED)
    res: dict = {"protocol": str(PROTOCOL), "version": CONFIG_VERSION, "seed": SEED,
                 "protocol_sha256": hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()}

    # ---- units (C30-1) and model (H30-1 model arm) --------------------------
    units = [_load(B27 / "units" / f"{r}__{w}.json") for r in REGIONS for w in WINDOWS]
    lu = law_on_adjacent(units, len(ELLS_KM))
    res["C30_1_units"] = {k: v for k, v in lu.items() if k != "_ratio"}
    res["C30_1_units"]["pass"] = bool(lu["pearson_r"] >= 0.99)
    model = [_load(B29 / "model" / f"{r}__{w}.json") for r in REGIONS for w in MODEL_WINDOWS]
    lm = law_on_adjacent(model, len(ELLS_KM))
    res["H30_1_model"] = {k: v for k, v in lm.items() if k != "_ratio"}
    res["H30_1_model"]["pass"] = bool(lm["pearson_r"] >= 0.90 and 0.80 <= lm["median_ratio"] <= 1.25)

    # ---- tiles ------------------------------------------------------------
    n_env = len(ELLS_FINE)
    grid = tile_grid()
    by_tid = {t["tid"]: t for t in grid}
    sh = {s: {p.stem: _load(p) for p in sorted((B27 / "tiles" / s).glob("t*.json"))} for s in SEASONS}
    armb = {s: {r["tid"]: r for r in json.loads((ARMB / f"tiles_{s}2023.json").read_text())["tiles"]}
            for s in SEASONS}
    common = sorted(set.intersection(*[set(sh[s]) for s in SEASONS], *[set(armb[s]) for s in SEASONS]))
    cov = tile_covariates([by_tid[t] for t in common])
    keep = [t for t in common if cov[t]["orog_mean"] <= OROG_EXCL_M]
    tiles = [by_tid[t] for t in keep]
    tid_index = {t: i for i, t in enumerate(keep)}
    res["n_kept"] = len(keep)
    Q = {(s, t, h): quantities(sh[s][t], n_env, h) for s in SEASONS for t in keep for h in (0, 1, 2)}

    def field(key, i, h=0):
        return np.asarray([np.mean([Q[(s, t, h)][key][i] for s in SEASONS]) for t in keep])

    steps = (1, 2)
    block = np.asarray([int(by_tid[t]["lon_c"] // BLOCK_DEG) * 10 + int((by_tid[t]["lat_c"] + 60.0) // BLOCK_DEG)
                        for t in keep])
    ub = np.unique(block)
    members = [np.where(block == b)[0] for b in ub]

    def boot_median_ci(x):
        m = np.empty(N_BOOT)
        for k in range(N_BOOT):
            pick = rng.integers(0, len(ub), size=len(ub))
            m[k] = np.median(x[np.concatenate([members[i] for i in pick])])
        return [float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))]

    # C30-2
    A29 = np.load(B29 / "tile_amplitude.npz")
    G2 = field("G", 2)
    a29 = dict(zip(A29["tid"], A29["A"]))
    res["C30_2"] = {"max_abs_diff": float(np.max(np.abs(G2 - np.asarray([a29[t] for t in keep])))),
                    "pass": bool(np.max(np.abs(G2 - np.asarray([a29[t] for t in keep]))) < 1e-9)}

    # H30-1 tiles: pooled steps 1,2 -> per tile-step ratio; bootstrap over tiles' blocks
    c = np.concatenate([field("Cadj", i) for i in steps])
    g = np.concatenate([field("G", i + 1) for i in steps])
    ratio_by_tile = np.mean([field("Cadj", i) / field("G", i + 1) for i in steps], axis=0)
    ci = boot_median_ci(ratio_by_tile)
    res["H30_1_tiles"] = {"pearson_r": float(pearsonr(c, g)[0]), "median_ratio": float(np.median(c / g)),
                          "median_ratio_ci95": ci,
                          "per_step_r": {i: float(pearsonr(field("Cadj", i), field("G", i + 1))[0]) for i in steps},
                          "per_step_ratio": {i: float(np.median(field("Cadj", i) / field("G", i + 1))) for i in steps}}
    res["H30_1_tiles"]["pass"] = bool(res["H30_1_tiles"]["pearson_r"] >= 0.90
                                      and 0.80 <= res["H30_1_tiles"]["median_ratio"] <= 1.25
                                      and 0.70 <= ci[0] and ci[1] <= 1.40)
    res["H30_1_pass"] = bool(res["H30_1_tiles"]["pass"] and res["H30_1_model"]["pass"])

    # H30-2
    def rot_p(x, y):
        obs = spearmanr(x, y).statistic
        xx = x.reshape(-1, 1)
        cnt = sum(spearmanr(rotate_cov(xx, tiles, float(rng.uniform(ROT_MIN_DEG, 360 - ROT_MIN_DEG)), tid_index)[:, 0], y).statistic >= obs
                  for _ in range(N_ROT))
        return float(obs), float((1 + cnt) / (N_ROT + 1))

    a_part = {}
    for i in steps:
        rho, p = rot_p(field("h", i), field("Pobs", i))
        a_part[i] = {"rho": rho, "p_rot": p, "median_h": float(np.median(field("h", i))),
                     "median_P": float(np.median(field("Pobs", i)))}
    pass_a = all(v["rho"] >= 0.60 and v["p_rot"] < 0.05 for v in a_part.values())
    Pp = np.concatenate([field("Ppred", i) for i in steps])
    Po = np.concatenate([field("Pobs", i) for i in steps])
    slope, intercept = np.polyfit(Pp, Po, 1)
    bias = np.concatenate([field("h", i) - field("Pobs", i) for i in steps])
    cause = np.concatenate([field("O", i) * (1 / np.sqrt(field("Vs", i) * field("Vs", i + 1))
                                             - 1 / np.sqrt(field("V", i) * field("V", i + 1))) for i in steps])
    b_part = {"median_abs_err": float(np.median(np.abs(Pp - Po))), "slope": float(slope),
              "intercept": float(intercept), "pearson_r": float(pearsonr(Pp, Po)[0]),
              "median_Ppred": float(np.median(Pp)), "median_Pobs": float(np.median(Po)),
              "rho_bias_vs_cause": float(spearmanr(bias, cause).statistic)}
    pass_b = bool(b_part["median_abs_err"] <= 0.05 and 0.80 <= slope <= 1.20)
    res["H30_2"] = {"a": a_part, "a_pass": bool(pass_a), "b": b_part, "b_pass": pass_b,
                    "pass": bool(pass_a and pass_b)}

    # H30-3 reliability of h vs P (mean over steps 1-2)
    def sb(x1, x2):
        r = spearmanr(x1, x2).statistic
        return 2 * r / (1 + r)

    def rel(key):
        per = []
        for s in SEASONS:
            x1 = np.asarray([np.mean([Q[(s, t, 1)][key][i] for i in steps]) for t in keep])
            x2 = np.asarray([np.mean([Q[(s, t, 2)][key][i] for i in steps]) for t in keep])
            per.append(sb(x1, x2))
        xs = {s: np.asarray([np.mean([Q[(s, t, 0)][key][i] for i in steps]) for t in keep]) for s in SEASONS}
        return {"split_half_SB_by_season": [float(v) for v in per], "split_half_SB_mean": float(np.mean(per)),
                "cross_season_rho": float(spearmanr(xs["JFM"], xs["JAS"]).statistic)}

    r_h, r_P = rel("h"), rel("Pobs")
    res["H30_3"] = {"h": r_h, "P": r_P, "pass": bool(r_h["split_half_SB_mean"] >= r_P["split_half_SB_mean"] - 0.05)}

    # H30-4 own variance N
    def Nfield(h=0, s=None):
        ss = SEASONS if s is None else [s]
        return np.asarray([np.mean([np.mean([Q[(sx, t, h)]["N"][j] for j in (2, 3)]) for sx in ss]) for t in keep])

    N = Nfield()
    P = np.asarray([np.mean([armb[s][t]["P_fine_anch_mean"] for s in SEASONS]) for t in keep])
    eke = np.asarray([np.mean([armb[s][t]["eke_syn"] for s in SEASONS]) for t in keep])
    cape = np.asarray([cov[t]["cape_mean"] for t in keep])
    per = [sb(Nfield(1, s), Nfield(2, s)) for s in SEASONS]
    geo = {}
    for nm, y in (("P", P), ("cape", cape), ("eke", eke), ("G2", G2)):
        geo[f"rho_N_{nm}"], geo[f"p_rot_N_{nm}"] = rot_p(N, y)
    res["H30_4"] = {"split_half_SB_by_season": [float(v) for v in per], "split_half_SB_mean": float(np.mean(per)),
                    "cross_season_rho": float(spearmanr(Nfield(0, "JFM"), Nfield(0, "JAS")).statistic),
                    "frac_N_positive": float(np.mean(N > 0)), "median_N": float(np.median(N)),
                    "median_V_anch_2": float(np.median(field("V", 2) - field("Vs", 2))),
                    "median_G_2": float(np.median(G2)), **geo,
                    "pass": bool(np.mean(per) >= 0.50 and abs(geo["rho_N_P"]) <= 0.90)}

    passed = {"H30_1": res["H30_1_pass"], "H30_2": res["H30_2"]["pass"],
              "H30_3": res["H30_3"]["pass"], "H30_4": res["H30_4"]["pass"]}
    res["primary_pass"] = passed
    if not passed["H30_1"]:
        v = "NOT_UNIFIED"
    elif passed["H30_2"] and passed["H30_4"]:
        v = "UNIFIED"
    else:
        v = "PARTIAL(" + ",".join(k for k, ok in passed.items() if ok) + ")"
    res["VERDICT"] = v
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(OUT / "tile_fields.npz", tid=np.asarray(keep), N=N, P=P, G2=G2,
             h1=field("h", 1), h2=field("h", 2),
             lat=np.asarray([by_tid[t]["lat_c"] for t in keep]), lon=np.asarray([by_tid[t]["lon_c"] for t in keep]))
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"VERDICT": v, "primary_pass": passed}, indent=1))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="tests", choices=["tests"])
    ap.parse_args()
    stage_tests()


if __name__ == "__main__":
    main()
