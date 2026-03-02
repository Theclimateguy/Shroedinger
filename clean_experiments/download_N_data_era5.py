#!/usr/bin/env python3
"""Plan or execute ERA5 requests for Experiment N follow-up data bundles."""

from __future__ import annotations

import argparse
import calendar
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterator

import pandas as pd


DEFAULT_AREA = "10,120,-10,280"  # north,west,south,east
DEFAULT_GRID = "0.25/0.25"
DEFAULT_MODEL_LEVELS = "1/to/137"
DEFAULT_INSTANT_PARAM = "130/131/132/133/135"
DEFAULT_PHYSICS_PARAM = "235006/235009/235010/235011/235012"
DEFAULT_ANALYSIS_TIMES = "00/03/06/09/12/15/18/21"
DEFAULT_FC_TIMES = "06/18"
DEFAULT_FC_STEPS = "3/6/9/12"
DEFAULT_SURFACE_VARS = (
    "total_precipitation",
    "evaporation",
    "convective_available_potential_energy",
    "convective_inhibition",
    "convective_precipitation",
    "large_scale_precipitation",
)


@dataclass(frozen=True)
class RequestItem:
    dataset: str
    request: dict
    target: Path
    tag: str


def _parse_date(s: str) -> date:
    return pd.Timestamp(s).date()


def _month_starts(start: date, end: date) -> Iterator[date]:
    cur = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while cur <= last:
        yield cur
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)


def _month_end(d: date) -> date:
    return date(d.year, d.month, calendar.monthrange(d.year, d.month)[1])


def _fmt_date_window(start_d: date, end_d: date) -> str:
    return f"{start_d:%Y-%m-%d}/to/{end_d:%Y-%m-%d}"


def _days_of_month(year: int, month: int) -> list[str]:
    n = calendar.monthrange(year, month)[1]
    return [f"{d:02d}" for d in range(1, n + 1)]


def _area_as_list(area: str) -> list[float]:
    vals = [float(x.strip()) for x in area.split(",")]
    if len(vals) != 4:
        raise ValueError(f"--area must have 4 comma-separated floats, got: {area}")
    return vals


def _build_requests(
    *,
    start_date: date,
    end_date: date,
    outdir: Path,
    area: str,
    grid: str,
    levelist: str,
    instant_param: str,
    analysis_times: str,
    physics_param: str,
    fc_times: str,
    fc_steps: str,
    surface_vars: tuple[str, ...],
    include_physics: bool,
    include_surface: bool,
) -> list[RequestItem]:
    reqs: list[RequestItem] = []
    request_dir = outdir / "requests"
    data_dir = outdir / "raw"
    request_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    area_list = _area_as_list(area)
    for m0 in _month_starts(start_date, end_date):
        m1 = _month_end(m0)
        d0 = max(m0, start_date)
        d1 = min(m1, end_date)
        if d0 > d1:
            continue
        ym = f"{m0:%Y%m}"
        date_window = _fmt_date_window(d0, d1)

        instant_req = {
            "class": "ea",
            "expver": "1",
            "stream": "oper",
            "type": "an",
            "levtype": "ml",
            "levelist": levelist,
            "param": instant_param,
            "date": date_window,
            "time": analysis_times,
            "grid": grid,
            "area": area,
            "format": "grib",
        }
        reqs.append(
            RequestItem(
                dataset="reanalysis-era5-complete",
                request=instant_req,
                target=data_dir / f"era5_complete_ml_instant_{ym}.grib",
                tag="ml_instant",
            )
        )

        if include_physics:
            phys_req = {
                "class": "ea",
                "expver": "1",
                "stream": "oper",
                "type": "fc",
                "levtype": "ml",
                "levelist": levelist,
                "param": physics_param,
                "date": date_window,
                "time": fc_times,
                "step": fc_steps,
                "grid": grid,
                "area": area,
                "format": "grib",
            }
            reqs.append(
                RequestItem(
                    dataset="reanalysis-era5-complete",
                    request=phys_req,
                    target=data_dir / f"era5_complete_ml_physics_{ym}.grib",
                    tag="ml_physics",
                )
            )

        if include_surface:
            surface_req = {
                "product_type": "reanalysis",
                "variable": list(surface_vars),
                "year": f"{m0.year:04d}",
                "month": f"{m0.month:02d}",
                "day": _days_of_month(m0.year, m0.month),
                "time": [f"{h:02d}:00" for h in range(0, 24, 3)],
                "area": area_list,
                "format": "netcdf",
            }
            reqs.append(
                RequestItem(
                    dataset="reanalysis-era5-single-levels",
                    request=surface_req,
                    target=data_dir / f"era5_singlelevel_surface_{ym}.nc",
                    tag="surface",
                )
            )

    # Persist JSON copies for reproducibility.
    for i, item in enumerate(reqs):
        meta = {
            "dataset": item.dataset,
            "tag": item.tag,
            "target": str(item.target),
            "request": item.request,
        }
        with (request_dir / f"{i:04d}_{item.tag}.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=True)

    return reqs


def _run_requests(reqs: list[RequestItem]) -> None:
    try:
        import cdsapi
    except ImportError as exc:
        raise RuntimeError("cdsapi is required for --download. Install with: pip install cdsapi") from exc

    client = cdsapi.Client()
    for i, item in enumerate(reqs):
        item.target.parent.mkdir(parents=True, exist_ok=True)
        print(f"[{i+1}/{len(reqs)}] download {item.dataset} -> {item.target}")
        client.retrieve(item.dataset, item.request, str(item.target))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2017-01-01")
    parser.add_argument("--end-date", default="2019-12-31")
    parser.add_argument("--outdir", type=Path, default=Path("data/raw/era5_n13"))
    parser.add_argument("--area", default=DEFAULT_AREA, help="north,west,south,east")
    parser.add_argument("--grid", default=DEFAULT_GRID, help="e.g. 0.25/0.25")
    parser.add_argument("--model-levels", default=DEFAULT_MODEL_LEVELS)
    parser.add_argument("--instant-param", default=DEFAULT_INSTANT_PARAM)
    parser.add_argument("--analysis-times", default=DEFAULT_ANALYSIS_TIMES)
    parser.add_argument("--physics-param", default=DEFAULT_PHYSICS_PARAM)
    parser.add_argument("--fc-times", default=DEFAULT_FC_TIMES, help="e.g. 06/18")
    parser.add_argument("--fc-steps", default=DEFAULT_FC_STEPS, help="e.g. 3/6/9/12")
    parser.add_argument("--surface-vars", nargs="+", default=list(DEFAULT_SURFACE_VARS))
    parser.add_argument("--skip-physics", action="store_true")
    parser.add_argument("--skip-surface", action="store_true")
    parser.add_argument("--download", action="store_true", help="Execute downloads via cdsapi")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    d0 = _parse_date(args.start_date)
    d1 = _parse_date(args.end_date)
    if d0 > d1:
        raise ValueError("--start-date must be <= --end-date")

    reqs = _build_requests(
        start_date=d0,
        end_date=d1,
        outdir=args.outdir,
        area=args.area,
        grid=args.grid,
        levelist=args.model_levels,
        instant_param=args.instant_param,
        analysis_times=args.analysis_times,
        physics_param=args.physics_param,
        fc_times=args.fc_times,
        fc_steps=args.fc_steps,
        surface_vars=tuple(args.surface_vars),
        include_physics=not args.skip_physics,
        include_surface=not args.skip_surface,
    )

    manifest_rows = [
        {
            "idx": i,
            "dataset": r.dataset,
            "tag": r.tag,
            "target": str(r.target),
            "date": r.request.get("date", f"{r.request.get('year')}-{r.request.get('month')}"),
        }
        for i, r in enumerate(reqs)
    ]
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.outdir / "retrieval_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"Prepared {len(reqs)} requests. Manifest: {manifest_path}")

    if args.download:
        _run_requests(reqs)
        print("Download completed.")
    else:
        print("Dry-run only. Use --download to execute CDS retrievals.")


if __name__ == "__main__":
    main()
