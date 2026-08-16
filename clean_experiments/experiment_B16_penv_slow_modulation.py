#!/usr/bin/env python3
"""Phase-16: slow (interannual/ENSO) modulation of the equilibrium profile P^eq.

Preregistered protocol: docs/PROTOCOL_PHASE16_PENV_SLOW_MODULATION.md
(frozen 2026-08-17). Inputs are the stored P_real / F_spec JSONs of
Phases 2/2b/3 and the frozen ONI table; no field recomputation.

  --stage gate  : C16-1 amplitude-specificity gate on the Phase-12 generator
  --stage tests : H16a (fresh arm R1-R4), C16-2 placebo, H16b replication,
                  descriptives, verdict
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from clean_experiments.experiment_B1_true_flux_baselines import (
        _grid_geometry, _interior_mask,
    )
    from clean_experiments.experiment_B2_scale_irreversibility import (
        ELLS_KM, compute_vorticity, envelope_rho_profile,
    )
    from clean_experiments.experiment_B12_estimator_validity import (
        _uv_from_vorticity,
    )
except ImportError:  # pragma: no cover
    from experiment_B1_true_flux_baselines import _grid_geometry, _interior_mask  # type: ignore
    from experiment_B2_scale_irreversibility import (  # type: ignore
        ELLS_KM, compute_vorticity, envelope_rho_profile,
    )
    from experiment_B12_estimator_validity import _uv_from_vorticity  # type: ignore

SEED = 20260817
OUT = _HERE / "results" / "experiment_B16_penv_slow_modulation"
N_PERM = 999
EPS = 1e-12

SRC = {
    "B2": _HERE / "results" / "experiment_B2_scale_irreversibility",
    "B2b": _HERE / "results" / "experiment_B2b_heldout_invariants",
    "B3": _HERE / "results" / "experiment_B3_scattering_benchmark",
}

# Frozen ONI (ERSST.v6), protocol section 2
ONI = {"W1": 0.1, "W2": -0.1, "W3": -0.7, "W4": 0.2,
       "W5": -0.9, "W6": -0.5, "W7": -0.7, "W8": -0.8,
       "W9": -0.3, "W10": 1.3, "W11": 1.5, "W12": 0.0}

FRESH = ("R1_WPWP", "R2_NATL", "R3_AMAZ", "R4_CASIA")
REPL = ("R5_SPCZ", "R6_SATL", "R7_CONGO", "R8_AUS",
        "R9_NPAC", "R10_INDO", "R11_EURO", "R12_SAM")


# --------------------------------------------------------------------------
# C16-1 amplitude gate
# --------------------------------------------------------------------------

def synthetic_pair_amp(rng: np.random.Generator, indep: float, amp: float,
                       ny: int, nx: int, nt: int, sig_coarse: float,
                       sig_fine: float, dx_km: float, dy_km: float,
                       rho_t: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Frozen B12 `synthetic_pair` formula with a fine-band amplitude knob.

    Identical to Phase-12 except the fine-band weight 0.8 is multiplied by
    `amp`; amp = 1 reproduces the original generator.
    """
    def ar1(sig: float) -> np.ndarray:
        out = np.empty((nt, ny, nx), dtype=np.float32)
        prev = gaussian_filter(rng.standard_normal((ny, nx)), sig, mode="wrap")
        for t in range(nt):
            new = gaussian_filter(rng.standard_normal((ny, nx)), sig, mode="wrap")
            prev = rho_t * prev + np.sqrt(1 - rho_t ** 2) * new
            out[t] = prev
        return out / (out.std() + EPS)

    coarse = ar1(sig_coarse)
    carrier = ar1(sig_fine)
    alt = np.abs(ar1(sig_coarse))
    cenv = np.abs(gaussian_filter(coarse, (0, sig_coarse, sig_coarse), mode="wrap"))
    cenv /= cenv.std() + EPS
    alt /= alt.std() + EPS
    modul = (1.0 - indep) * cenv + indep * alt
    fine = carrier * modul
    omega = coarse + amp * 0.8 * fine / (fine.std() + EPS)
    return _uv_from_vorticity(omega.astype(np.float32), dx_km, dy_km)


def _gate_one(args_tuple) -> dict:
    kind, level, rep, seed = args_tuple
    os.environ["OMP_NUM_THREADS"] = "1"
    rng = np.random.default_rng(seed)
    ny, nx, nt = 81, 161, 60
    lat = np.arange(10.0, -10.01, -0.25)
    lon = np.arange(130.0, 170.01, 0.25)
    dx_km, dy_km = _grid_geometry(lat, lon)
    dxm, dym = float(np.mean(dx_km)), float(dy_km)
    sig_coarse = 550.0 / dxm / np.sqrt(12.0)
    sig_fine = 280.0 / dxm / np.sqrt(12.0)
    if kind == "amp":
        u, v = synthetic_pair_amp(rng, 0.5, float(level), ny, nx, nt,
                                  sig_coarse, sig_fine, dxm, dym)
    else:
        u, v = synthetic_pair_amp(rng, float(level), 1.0, ny, nx, nt,
                                  sig_coarse, sig_fine, dxm, dym)
    mask = _interior_mask(ny, nx, ELLS_KM[-1], dx_km, dy_km)
    om = compute_vorticity(u, v, lat, lon)
    rho = envelope_rho_profile(om, dx_km, dy_km, mask)
    p = float(np.median(rho[:, [3, 4]].mean(axis=1)))
    return {"kind": kind, "level": float(level), "rep": rep, "P": p}


def stage_gate(workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "gate_c16_1.json"
    if cache.exists():
        print("cached", flush=True)
        return
    ss = np.random.SeedSequence(SEED)
    amps = [0.25, 0.5, 1.0, 2.0, 4.0]
    inds = np.linspace(0.0, 1.0, 9)
    jobs = []
    seeds = iter(int(s) for s in ss.generate_state(len(amps) * 12 + len(inds) * 12))
    for lv in amps:
        for rep in range(12):
            jobs.append(("amp", lv, rep, next(seeds)))
    for lv in inds:
        for rep in range(12):
            jobs.append(("indep", lv, rep, next(seeds)))
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        rows = pool.map(_gate_one, jobs)
    cache.write_text(json.dumps(rows), encoding="utf-8")
    print(f"gate rows: {len(rows)}", flush=True)


def score_gate() -> dict:
    rows = json.loads((OUT / "gate_c16_1.json").read_text())

    def stats(kind):
        sel = [r for r in rows if r["kind"] == kind]
        levels = sorted({r["level"] for r in sel})
        means = [float(np.mean([r["P"] for r in sel if r["level"] == lv]))
                 for lv in levels]
        rho = float(spearmanr([r["level"] for r in sel],
                              [r["P"] for r in sel]).statistic)
        return means, rho

    m_amp, rho_amp = stats("amp")
    m_ind, rho_ind = stats("indep")
    res = {"rho_P_vs_amplitude": rho_amp,
           "rho_P_vs_independence": rho_ind,
           "spread_amp": float(np.std(m_amp)),
           "spread_indep": float(np.std(m_ind)),
           "rung_means_amp": m_amp, "rung_means_indep": m_ind,
           "C16_1_pass": bool(abs(rho_amp) <= 0.5
                              and np.std(m_amp) < np.std(m_ind))}
    return res


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def load_table() -> list[dict]:
    rows = []
    for src in SRC.values():
        for f in sorted(src.glob("R*__W*.json")):
            d = json.loads(f.read_text())
            if "P_real" not in d:
                continue
            w = d["window"].split("_")[0]
            if w not in ONI:
                continue
            p = np.asarray(d["P_real"], dtype=float)
            fs = d.get("F_spec", {})
            lv = [fs.get("logvar_3"), fs.get("logvar_4")]
            rows.append({
                "tag": d["tag"], "region": d["region"], "w": w,
                "season": "JFM" if "JFM" in d["window"] else "JAS",
                "year": int(d["window"].split("_")[1][:4]),
                "oni": ONI[w],
                "Peq": float(np.mean(p[[3, 4]])),
                "P_bands": [float(x) for x in p],
                "logvar": float(np.mean([x for x in lv if x is not None]))
                if all(x is not None for x in lv) else np.nan,
            })
    # unique per tag (a tag may exist in only one source)
    seen = {}
    for r in rows:
        seen[r["tag"]] = r
    return list(seen.values())


def cells_of(rows: list[dict], regions: tuple[str, ...]) -> dict:
    cells: dict = {}
    for r in rows:
        if r["region"] in regions:
            cells.setdefault((r["region"], r["season"]), []).append(r)
    return {k: v for k, v in cells.items() if len(v) == 4}


def stat_S(cells: dict, key: str, rng: np.random.Generator | None = None
           ) -> float:
    vals = []
    for cell in cells.values():
        y = np.array([r[key] for r in cell])
        x = np.array([r["oni"] for r in cell])
        if rng is not None:
            x = rng.permutation(x)
        m = np.isfinite(y)
        if m.sum() == 4:
            vals.append(spearmanr(x, y).statistic)
    return float(np.mean(vals))


def perm_test(cells: dict, key: str, rng: np.random.Generator) -> dict:
    s = stat_S(cells, key)
    null = np.array([stat_S(cells, key, rng) for _ in range(N_PERM)])
    return {"S": s,
            "p_perm_pos": float((np.sum(null >= s) + 1) / (N_PERM + 1)),
            "null_sd": float(null.std())}


def stage_tests() -> dict:
    rng = np.random.default_rng(SEED)
    res: dict = {"seed": SEED,
                 "protocol": "docs/PROTOCOL_PHASE16_PENV_SLOW_MODULATION.md"}

    res["C16_1"] = score_gate()
    if not res["C16_1"]["C16_1_pass"]:
        res["PHASE16_VERDICT"] = "ESTIMATOR_INVALID"
        return res

    rows = load_table()
    res["n_region_windows"] = len(rows)
    fresh = cells_of(rows, FRESH)
    repl = cells_of(rows, REPL)
    res["n_cells_fresh"] = len(fresh)
    res["n_cells_repl"] = len(repl)

    # H16a fresh arm
    res["H16a"] = perm_test(fresh, "Peq", rng)
    res["H16a_pass"] = bool(res["H16a"]["S"] > 0
                            and res["H16a"]["p_perm_pos"] < 0.05)

    # C16-2 amplitude placebo on the fresh arm
    res["C16_2"] = perm_test(fresh, "logvar", rng)
    res["C16_2_pass"] = bool(abs(res["H16a"]["S"]) > abs(res["C16_2"]["S"]))

    # H16b replication arm
    res["H16b"] = perm_test(repl, "Peq", rng)
    res["H16b_pass"] = bool(res["H16b"]["S"] > 0
                            and res["H16b"]["p_perm_pos"] < 0.05)
    res["H16b_same_sign"] = bool(np.sign(res["H16b"]["S"])
                                 == np.sign(res["H16a"]["S"]))

    # ---- descriptives ----------------------------------------------------
    desc: dict = {}
    per_region = {}
    for (reg, sea), cell in {**fresh, **repl}.items():
        y = [r["Peq"] for r in cell]
        x = [r["oni"] for r in cell]
        per_region[f"{reg}_{sea}"] = float(spearmanr(x, y).statistic)
    desc["within_cell_rho"] = per_region
    allc = {**fresh, **repl}
    desc["S_full_pool"] = stat_S(allc, "Peq")
    # per-band S on the fresh arm
    for b in range(5):
        for cell in fresh.values():
            for r in cell:
                r[f"Pb{b}"] = r["P_bands"][b]
        desc[f"S_fresh_band_{b}"] = stat_S(fresh, f"Pb{b}")
    # year trend after removing cell means and pooled ONI slope
    ys, ps, onis, cid = [], [], [], []
    for k, cell in allc.items():
        for r in cell:
            ys.append(r["year"]); ps.append(r["Peq"])
            onis.append(r["oni"]); cid.append(k)
    ys, ps, onis = np.array(ys, float), np.array(ps), np.array(onis)
    for k in set(cid):
        m = np.array([c == k for c in cid])
        ps[m] -= ps[m].mean()
        onis[m] -= onis[m].mean()
        ys[m] -= ys[m].mean()
    beta_oni = float((onis @ ps) / (onis @ onis))
    resid = ps - beta_oni * onis
    desc["pooled_beta_P_per_oni_degC"] = beta_oni
    desc["year_trend_after_oni_per_decade"] = float(
        (ys @ resid) / (ys @ ys) * 10.0)
    # seasonal offset per region (JFM minus JAS mean Peq)
    soff = {}
    for reg in FRESH + REPL:
        jfm = [r["Peq"] for r in rows if r["region"] == reg
               and r["season"] == "JFM"]
        jas = [r["Peq"] for r in rows if r["region"] == reg
               and r["season"] == "JAS"]
        if jfm and jas:
            soff[reg] = float(np.mean(jfm) - np.mean(jas))
    desc["seasonal_offset_JFM_minus_JAS"] = soff
    res["descriptive"] = desc

    # ---- verdict ---------------------------------------------------------
    if res["H16a_pass"] and res["C16_2_pass"] and res["H16b_same_sign"]:
        v = "CONFIRMED"
    elif res["H16a_pass"]:
        v = "PARTIAL"
    else:
        v = "NEGATIVE"
    res["PHASE16_VERDICT"] = v
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["gate", "tests"], default="gate")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()
    if args.stage == "gate":
        stage_gate(args.workers)
    else:
        res = stage_tests()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "summary.json").write_text(json.dumps(res, indent=2),
                                          encoding="utf-8")
        print(json.dumps(res, indent=2)[:6000])


if __name__ == "__main__":
    main()
