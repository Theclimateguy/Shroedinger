#!/usr/bin/env python3
"""Phase-33 Arm B data: fresh WeatherBench-2 initialisations (2020 AMJ and OND).

Protocol: docs/PROTOCOL_PHASE33_MODEL_FIDELITY.md (frozen 2026-10-10).
Wraps the frozen Phase-11 downloader (same stores, variables, level, domains,
leads, stride); only the windows and the output directory differ.

Usage:
    python clean_experiments/download_b33_wb2_fresh.py --system graphcast
    python clean_experiments/download_b33_wb2_fresh.py --system gencast_member   # optional, 2 inits/window
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import download_b11_wb2 as dl  # noqa: E402

dl.WINDOWS = {"W15_2020AMJ": ("2020-04-01", "2020-06-30"),
              "W16_2020OND": ("2020-10-01", "2020-12-31")}
dl.INIT_STRIDE_DAYS = 5               # Deviation D2
dl.GENCAST_MEMBER_MAX_INITS = 6       # Deviation D4


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--system", choices=sorted(dl.SYSTEMS), required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/b33wb2"))
    ap.add_argument("--workers", type=int, default=3)
    args = ap.parse_args()
    print("inits:", [f"{d:%Y-%m-%d}" for d in dl.init_dates()], flush=True)
    dl.run(args.system, args.out_dir, args.workers)


if __name__ == "__main__":
    main()
