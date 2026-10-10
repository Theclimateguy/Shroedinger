#!/usr/bin/env python3
"""Excel workbook with the source data of the graphs of article 1 (v8), as the
journal asks ("файл в формате Excel, содержащий исходные данные" for graphs
and diagrams). One sheet per figure; the map (Fig. 4) and the supplement
figures П1–П3 are included as well.

Run from the repository root:
    python manuscript_v2/article1/tools/build_article1_figure_data_xlsx.py
Output: manuscript_v2/article1/journal_package/article1_v8_figure_data.xlsx
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
RES = ROOT / "clean_experiments" / "results"
OUT = ROOT / "manuscript_v2" / "article1" / "journal_package" / "article1_v8_figure_data.xlsx"
sys.path.insert(0, str(ROOT / "clean_experiments"))
from download_b11_wb2 import DOMAINS  # noqa: E402

STEP_LABELS = ["<50 | 50-100", "50-100 | 100-200", "100-200 | 200-400", "200-400 | 400-800", "400-800 | 800-1600"]
REGIONS = ["R1_WPWP", "R2_NATL", "R3_AMAZ", "R4_CASIA", "R5_SPCZ", "R6_SATL", "R7_CONGO",
           "R8_AUS", "R9_NPAC", "R10_INDO", "R11_EURO", "R12_SAM"]
PERIOD = {"W1": "I", "W2": "I", "W3": "I", "W4": "I", "W5": "II", "W6": "II", "W7": "II", "W8": "II",
          "W9": "III", "W10": "III", "W11": "III", "W12": "III"}


def load(pattern: str) -> list[dict]:
    out = []
    for f in sorted(glob.glob(pattern)):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        if "region" in d and "window" in d:
            out.append(d)
    return out


def degree_windows() -> pd.DataFrame:
    recs = (load(str(RES / "experiment_B2_scale_irreversibility_v12" / "R*.json"))
            + load(str(RES / "experiment_B2b_heldout_invariants_v12" / "R*.json"))
            + load(str(RES / "experiment_B3_scattering_benchmark_v12" / "R*.json")))
    rows = []
    for r in recs:
        w = r["window"]
        rows.append({"region": r["region"], "window": w, "period": PERIOD[w.split("_")[0]],
                     "year": w.split("_")[1][:4], "season": w[-3:],
                     **{f"P_step{i + 1} ({STEP_LABELS[i]} km)": r["P_real"][i] for i in range(5)},
                     **{f"P_surr_step{i + 1}": r["P_surrogate_median"][i] for i in range(5)}})
    return pd.DataFrame(rows).sort_values(["region", "window"])


def km_windows() -> pd.DataFrame:
    rows = []
    for r in load(str(RES / "experiment_B18_equal_km_regions_v12" / "R*.json")):
        k = r["km_1.0"]
        rows.append({"region": r["region"], "window": r["window"], "period": PERIOD[r["window"].split("_")[0]],
                     **{f"P_step{i + 1}": k["P_real"][i] for i in range(5)},
                     **{f"P_surr_step{i + 1}": k["P_surrogate_median"][i] for i in range(5)},
                     **{f"P_anchored_step{i + 1}": k["P_anchored"][i] for i in range(5)}})
    return pd.DataFrame(rows).sort_values(["region", "window"])


def free_run() -> tuple[pd.DataFrame, pd.DataFrame]:
    s = json.loads((RES / "experiment_B13_free_running_v12" / "summary.json").read_text(encoding="utf-8"))
    dv = s["domain_values"]
    per = {}
    for f in sorted(glob.glob(str(RES / "experiment_B13_free_running_v12" / "era5_*.json"))):
        d = json.loads(Path(f).read_text(encoding="utf-8"))
        per.setdefault(d["domain"], []).append(d["P"])
    rows = []
    for name in sorted(dv):
        n, w, sth, e = DOMAINS[name]
        rows.append({"domain": name, "type": "main" if name.startswith("R") else "lattice",
                     "north": n, "west": w, "south": sth, "east": e,
                     "P_era5 (median over windows, mean of steps 4-5)": dv[name]["P_era5"],
                     "P_model (ECMWF-IFS-HR free run)": dv[name]["P_model"],
                     "P_era5_min_over_windows": min(per[name]), "P_era5_max_over_windows": max(per[name]),
                     "n_era5_windows": len(per[name]),
                     "A_era5": dv[name]["A_era5"], "A_model": dv[name]["A_model"],
                     "u_era5": dv[name]["u_era5"], "u_model": dv[name]["u_model"],
                     "persist_era5": dv[name]["persist_era5"], "persist_model": dv[name]["persist_model"]})
    stats = pd.DataFrame([{"quantity": q, "spearman_rho_era5_vs_model": s[k]["rho"], "p_perm": s[k]["p_perm"], "n": s[k]["n"]}
                          for q, k in (("P", "H13a"), ("A", "H13b"), ("u", "H13c"), ("persist", "H13d"))])
    return pd.DataFrame(rows), stats


def main() -> None:
    deg = degree_windows()
    km = km_windows()
    dom, stats = free_run()
    # Fig. 2a: median raw profile per region (8 windows), steps 2-5
    f2a = deg.groupby("region")[[c for c in deg.columns if c.startswith("P_step")]].median().loc[REGIONS]
    # Fig. 2b: median anchored profile on km boxes, primary windows (W9-W12)
    kmp = km[km["period"] == "III"]
    f2b = kmp.groupby("region")[[c for c in kmp.columns if c.startswith("P_anchored")]].median().loc[REGIONS]
    # Fig. A2: degree anchored vs km anchored, W9-W12
    degp = deg[deg["period"] == "III"].copy()
    for i in range(5):
        degp[f"P_anchored_deg_step{i + 1}"] = degp[f"P_step{i + 1} ({STEP_LABELS[i]} km)"] - degp[f"P_surr_step{i + 1}"]
    fa2 = degp[["region", "window"] + [f"P_anchored_deg_step{i + 1}" for i in range(5)]].merge(
        kmp[["region", "window"] + [f"P_anchored_step{i + 1}" for i in range(5)]].rename(
            columns={f"P_anchored_step{i + 1}": f"P_anchored_km_step{i + 1}" for i in range(5)}), on=["region", "window"])
    readme = pd.DataFrame({"sheet": ["Fig2a", "Fig2b", "Fig3_Fig4", "Fig3_stats", "FigA1_windows", "FigA2", "FigA3"],
                           "content": ["Fig. 2a: median raw profile P per region over its 8 windows (degree boxes), steps 1-5 (steps 2-5 plotted)",
                                       "Fig. 2b: median surrogate-corrected profile (P - P_surr) per region on equal-km boxes, windows 2023-2024",
                                       "Fig. 3 and Fig. 4: 39 domains, ERA5 vs free-running model, box corners, per-window min/max",
                                       "Spearman correlations ERA5 vs model (Fig. 3, Fig. A3)",
                                       "Fig. A1: all 96 region-windows, raw and surrogate profiles, steps 1-5",
                                       "Fig. A2: surrogate-corrected profiles on degree vs equal-km boxes, 48 windows 2023-2024",
                                       "Fig. A3: A, u, persistence per domain (same table as Fig3_Fig4)"],
                           "source": ["clean_experiments/results/experiment_B{2,2b,3}_*_v12/", "experiment_B18_equal_km_regions_v12/",
                                      "experiment_B13_free_running_v12/", "experiment_B13_free_running_v12/summary.json",
                                      "experiment_B{2,2b,3}_*_v12/", "experiment_B3_*_v12 + experiment_B18_*_v12", "experiment_B13_free_running_v12/"]})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        readme.to_excel(xw, sheet_name="README", index=False)
        f2a.round(4).to_excel(xw, sheet_name="Fig2a")
        f2b.round(4).to_excel(xw, sheet_name="Fig2b")
        dom.round(4).to_excel(xw, sheet_name="Fig3_Fig4", index=False)
        stats.round(4).to_excel(xw, sheet_name="Fig3_stats", index=False)
        deg.round(4).to_excel(xw, sheet_name="FigA1_windows", index=False)
        fa2.round(4).to_excel(xw, sheet_name="FigA2", index=False)
        dom[["domain", "type", "A_era5", "A_model", "u_era5", "u_model", "persist_era5", "persist_model"]].round(4).to_excel(
            xw, sheet_name="FigA3", index=False)
    print("wrote", OUT, "| sheets:", "README, Fig2a, Fig2b, Fig3_Fig4, Fig3_stats, FigA1_windows, FigA2, FigA3",
          "| rows:", len(deg), len(km), len(dom))


if __name__ == "__main__":
    main()
