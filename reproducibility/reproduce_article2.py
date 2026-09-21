#!/usr/bin/env python3
"""Reproduce paper 2 of the series end to end - the single entry point.

Paper: "Geography of the coupling between hierarchical levels of the
atmospheric circulation: a planetary map, its controls and a regionalisation"
(journal-format source manuscript_v2/article2/article2_v4_src.tex).
Per-result mapping (every number, table and figure -> script -> artifact):
reproducibility/article2_README.md.

Stages
------
  download    fetch the primary fields (ERA5 regional + global, covariates,
              CAPE + climate indices, daily ERA5, HighResMIP steady and
              transient integrations, monthly hydroclimatic fields)
  compute     run the analyses behind the paper, in dependency order:
              Phase-20 arms A/B (maps, factors), AUDIT-2 (intermittency
              block), AUDIT-2b (reliability ceiling), AUDIT-4 (unexplained
              part), Phase 17 (long record), Phases 21-23 (IPO, free-running
              model, regime shift), Phase 25 (residual candidates), and the
              descriptive readouts quoted in the text
  figures     descriptive layer (zonal profiles, typology / regionalisation
              map, map of the unexplained part, per-cell table, gazetteer of
              named places) and the Russian-label journal figures 1, 3, 5;
              refresh manuscript copies
  manuscript  build the journal-format manuscript (tex, pdf, docx) from
              article2_v4_src.tex; needs pandoc, latexmk + biber, python-docx;
              a missing tool is reported and skipped, not fatal
  check       verify that every expected artifact exists; print SHA-256
  all         download + compute + figures + manuscript + check (default)

Examples
--------
  python reproducibility/reproduce_article2.py --stage check
  python reproducibility/reproduce_article2.py --stage compute --only B20
  python reproducibility/reproduce_article2.py --stage figures
  python reproducibility/reproduce_article2.py --stage download --only era5

Credentials: ERA5/CAPE need ~/.cdsapirc (CDS API). HighResMIP (ESGF) and
the NOAA PSL index series need no account. Downloads are resumable;
tile computations cache per-shard and can be interrupted and resumed.
All derived tables are committed, so `--stage check` and `--stage figures`
work without any download.
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
CE = REPO / "clean_experiments"

# name, script, needs-credential, note
DOWNLOADS = [
    ("era5-b3", "download_b3_era5_wind.py", "cds",
     "ERA5 850 hPa wind: 12 regions, windows W9-W12 (2023-2024)"),
    ("era5-b2b", "download_b2b_era5_wind.py", "cds",
     "ERA5: R5-R12, windows W5-W8 (2021-2022) - part-A windows"),
    ("era5-global-b20", "download_b20_era5_global.py", "cds",
     "ERA5 global 850 hPa wind, JFM+JAS 2023 (936-tile map)"),
    ("covariates-b4", "download_b4_covariates.py", "cds",
     "ERA5 invariants: orography, land-sea mask, SST, vegetation (LAI)"),
    ("cape-indices-b21", "download_b21_cape_indices.py", "cds",
     "Monthly CAPE (CDS) + named climate indices (NOAA PSL, no account)"),
    ("era5-daily-b17", "download_b17_era5_daily.py", "cds",
     "Daily 00Z ERA5 850 hPa wind, 12 regions, 1979-2024 (long record)"),
    ("highresmip-b13", "download_b13_highresmip.py", None,
     "ECMWF-IFS-HR highresSST-present, global 850 hPa wind, 2014 (ESGF)"),
    ("hrmip-transient-b23", "download_b23_hrmip_transient.py", None,
     "ECMWF-IFS-HR transient epochs 1979-1986 / 2007-2014 (ESGF)"),
    ("hydro-b25", "download_b25_hydroclim.py", "cds",
     "ERA5 monthly hydroclimatic fields 0.5 deg 2021-2024 (Phase-25 candidates)"),
]

# name, script, extra argv, produces (relative to clean_experiments/results/)
EXPERIMENTS = [
    ("B20A-tiles", "experiment_B20_p_geography.py", ["--stage", "tiles"], None),
    ("B20A-tests", "experiment_B20_p_geography.py", ["--stage", "tests"],
     "experiment_B20_p_geography/summary.json"),
    ("B20B-tiles-JFM", "experiment_B20_armB_global_map.py",
     ["--stage", "tiles", "--season", "JFM"], None),
    ("B20B-tiles-JAS", "experiment_B20_armB_global_map.py",
     ["--stage", "tiles", "--season", "JAS"], None),
    ("B20B-tests", "experiment_B20_armB_global_map.py", ["--stage", "tests"],
     "experiment_B20_armB_global_map/summary.json"),
    ("A2-intermittency", "experiment_A2_intermittency_headtohead.py", ["--stage", "all"],
     "experiment_A2_intermittency/summary.json"),
    ("A2b-splithalf", "experiment_A2b_splithalf.py", [],
     "experiment_A2b_splithalf/summary_halves.json"),
    ("A4-residual", "experiment_A4_residual_structure.py", [],
     "experiment_A4_residual/summary.json"),
    ("B17-series", "experiment_B17_penv_long_record.py", ["--stage", "series"], None),
    ("B17-tests", "experiment_B17_penv_long_record.py", ["--stage", "tests"],
     "experiment_B17_penv_long_record/summary.json"),
    ("B21-p5", "experiment_B21_qt_tests.py", ["--stage", "p5-series"], None),
    ("B21-p1", "experiment_B21_qt_tests.py", ["--stage", "p1-series"], None),
    ("B21-tests", "experiment_B21_qt_tests.py", ["--stage", "tests"],
     "experiment_B21_qt_tests/summary.json"),
    ("B22-round2", "experiment_B22_qt_round2.py", [],
     "experiment_B22_qt_round2/summary.json"),
    ("B23-esyn", "experiment_B23_drift_stratigraphy.py",
     ["--stage", "esyn-series"], None),
    ("B23-ondisk", "experiment_B23_drift_stratigraphy.py",
     ["--stage", "tests-ondisk"],
     "experiment_B23_drift_stratigraphy/summary_ondisk.json"),
    ("B23-model-series", "experiment_B23_drift_stratigraphy.py",
     ["--stage", "model-series"], None),
    ("B23-model", "experiment_B23_drift_stratigraphy.py",
     ["--stage", "tests-model"],
     "experiment_B23_drift_stratigraphy/summary_model.json"),
    ("B25-candidates", "experiment_B25_residual_candidates.py", [],
     "experiment_B25_residual_candidates/summary.json"),
    ("descriptives", "verify_article2_descriptives.py", [],
     "verify_article2_descriptives/summary.json"),
]

FIGURES = [
    ("fig-armA", "visualize_B20_p_geography.py"),
    ("fig-armB", "visualize_B20_armB_global.py"),
    ("fig-channels", "visualize_B20_two_channel.py"),
    ("fig-describe", "describe_B20_geography.py"),   # descriptive layer: zonal profiles, typology, residual map
    ("fig-ru", "visualize_article2_ru.py"),          # Russian-label journal figures 1, 3, 5
]

# results copied into the manuscript as (source in results/, manuscript name)
MANUSCRIPT_FIGS = [
    ("describe_B20_geography/fig_zonal_profiles.png", "fig2_zonal_profiles.png"),
    ("describe_B20_geography/fig_regionalization.png", "fig2_regionalization.png"),
    ("describe_B20_geography/fig_residual_map.png", "fig2_residual_map.png"),
]
MANUSCRIPT_TOOL = REPO / "manuscript_v2" / "article2" / "tools" / "build_article2_journal.py"

EXPECTED = [e[3] for e in EXPERIMENTS if e[3]] + [
    "experiment_B20_armB_global_map/fig1_global_map.png",
    "describe_B20_geography/summary.json",
    "describe_B20_geography/provinces.json",
    "experiment_B20_p_geography/fig1_tile_maps.png",
]


def run(script: str, extra: list[str]) -> int:
    print(f"\n=== {script} {' '.join(extra)} ===", flush=True)
    return subprocess.run([sys.executable, str(CE / script)] + extra,
                          cwd=REPO).returncode


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage",
                    choices=["download", "compute", "figures", "manuscript", "check", "all"],
                    default="all")
    ap.add_argument("--only", type=str, default=None,
                    help="substring filter on step names (e.g. 'era5', 'B20')")
    ap.add_argument("--keep-going", action="store_true")
    args = ap.parse_args()

    failed: list[str] = []

    if args.stage in ("download", "all"):
        steps = [d for d in DOWNLOADS
                 if not args.only or args.only.lower() in d[0].lower()]
        if any(d[2] == "cds" for d in steps) and not (Path.home() / ".cdsapirc").exists():
            print("WARNING: ~/.cdsapirc not found - CDS downloads will fail "
                  "(register at cds.climate.copernicus.eu)")
        for name, script, _, note in steps:
            print(f"\n[{name}] {note}")
            if run(script, []) != 0:
                failed.append(name)
                if not args.keep_going:
                    sys.exit(f"FAILED at download step {name}")

    if args.stage in ("compute", "all"):
        for name, script, extra, _ in EXPERIMENTS:
            if args.only and args.only.lower() not in name.lower():
                continue
            if run(script, extra) != 0:
                failed.append(name)
                if not args.keep_going:
                    sys.exit(f"FAILED at experiment {name}")

    if args.stage in ("figures", "all"):
        for name, script in FIGURES:
            if args.only and args.only.lower() not in name.lower():
                continue
            if run(script, []) != 0:
                failed.append(name)
                if not args.keep_going:
                    sys.exit(f"FAILED at figure step {name}")
        import shutil
        figdir = REPO / "manuscript_v2" / "article2" / "figures"
        for src, dst in MANUSCRIPT_FIGS:
            p = CE / "results" / src
            if p.exists():
                shutil.copyfile(p, figdir / dst)
                print(f"  copied {dst}")

    if args.stage in ("manuscript", "all"):
        import shutil as _sh
        missing_tools = [x for x in ("pandoc", "latexmk", "biber") if not _sh.which(x)]
        if missing_tools:
            print(f"\nmanuscript: missing {', '.join(missing_tools)} - the corresponding "
                  "outputs are skipped")
        extra = (["--no-docx"] if "pandoc" in missing_tools else []) + \
                (["--no-pdf"] if ("latexmk" in missing_tools or "biber" in missing_tools) else [])
        print(f"\n=== {MANUSCRIPT_TOOL.name} {' '.join(extra)} ===", flush=True)
        rc = subprocess.run([sys.executable, str(MANUSCRIPT_TOOL)] + extra, cwd=REPO).returncode
        if rc != 0:
            print("manuscript build failed (non-fatal for the numerical route)")

    if args.stage in ("check", "all"):
        print("\n=== artifact check ===")
        missing = 0
        for rel in EXPECTED:
            p = CE / "results" / rel
            if p.exists():
                digest = hashlib.sha256(p.read_bytes()).hexdigest()
                print(f"  OK  {rel}  sha256={digest}")
            else:
                print(f"  MISSING  {rel}")
                missing += 1
        print(f"\n{len(EXPECTED) - missing}/{len(EXPECTED)} artifacts present."
              " Reference hashes: reproducibility/article2_README.md, sect. 5.")

    if failed:
        sys.exit("Failed steps: " + ", ".join(failed))


if __name__ == "__main__":
    main()
