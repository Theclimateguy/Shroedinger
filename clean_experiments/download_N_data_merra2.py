#!/usr/bin/env python3
"""Search and optionally download MERRA-2 granules for Experiment N follow-up.

If earthaccess is unavailable, the script falls back to an offline granule-plan mode
that generates expected file names per day and collection.
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_SHORT_NAMES = ("M2I3NPASM", "M2T3NPQDT", "M2T3NPMST", "M2T3NPCLD")
DEFAULT_BBOX = "120,-10,280,10"  # west,south,east,north
SHORT_TO_COLLECTION = {
    "M2I3NPASM": "inst3_3d_asm_Np",
    "M2T3NPQDT": "tavg3_3d_qdt_Np",
    "M2T3NPMST": "tavg3_3d_mst_Np",
    "M2T3NPCLD": "tavg3_3d_cld_Np",
}


def _parse_bbox(s: str | None) -> tuple[float, float, float, float] | None:
    if s is None or len(s.strip()) == 0:
        return None
    vals = [float(x.strip()) for x in s.split(",")]
    if len(vals) != 4:
        raise ValueError(f"--bbox must be west,south,east,north (4 comma-separated floats), got: {s}")
    return vals[0], vals[1], vals[2], vals[3]


def _try_size_mb(granule: Any) -> float | None:
    # earthaccess granules usually expose size() in MB.
    try:
        size = granule.size()
        if size is None:
            return None
        return float(size)
    except Exception:
        pass
    return None


def _granule_id(granule: Any) -> str:
    for attr in ("concept_id", "native_id", "id"):
        try:
            v = getattr(granule, attr)
            if callable(v):
                out = v()
            else:
                out = v
            if out:
                return str(out)
        except Exception:
            continue
    return str(granule)


def _granule_links(granule: Any) -> list[str]:
    try:
        links = granule.data_links()
        if links:
            return [str(x) for x in links]
    except Exception:
        pass
    return []


def _parse_date(s: str) -> date:
    return pd.Timestamp(s).date()


def _stream_code(year: int) -> int:
    # MERRA-2 stream identifiers by period.
    if year <= 1991:
        return 100
    if year <= 2000:
        return 200
    if year <= 2010:
        return 300
    return 400


def _offline_plan_rows(
    *,
    short_names: list[str],
    start_date: str,
    end_date: str,
    limit_per_shortname: int,
) -> list[dict[str, object]]:
    d0 = _parse_date(start_date)
    d1 = _parse_date(end_date)
    if d0 > d1:
        raise ValueError("--start-date must be <= --end-date")

    rows: list[dict[str, object]] = []
    days = pd.date_range(d0, d1, freq="D")
    for short_name in short_names:
        coll = SHORT_TO_COLLECTION.get(short_name, "")
        if len(coll) == 0:
            print(f"[WARN] Unknown short name for offline filename mapping: {short_name}")
            continue
        count = 0
        for ts in days:
            y = int(ts.year)
            code = _stream_code(y)
            ymd = ts.strftime("%Y%m%d")
            fname = f"MERRA2_{code}.{coll}.{ymd}.nc4"
            rows.append(
                {
                    "short_name": short_name,
                    "granule_id": fname,
                    "size_mb": None,
                    "first_data_link": "",
                    "n_links": 0,
                    "mode": "offline_plan",
                }
            )
            count += 1
            if limit_per_shortname > 0 and count >= limit_per_shortname:
                break
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default="2017-01-01")
    parser.add_argument("--end-date", default="2019-12-31")
    parser.add_argument("--short-names", nargs="+", default=list(DEFAULT_SHORT_NAMES))
    parser.add_argument("--bbox", default=DEFAULT_BBOX, help="west,south,east,north")
    parser.add_argument("--outdir", type=Path, default=Path("data/raw/merra2_n13"))
    parser.add_argument("--limit-per-shortname", type=int, default=0, help="0 means no limit")
    parser.add_argument("--download", action="store_true", help="Download granules with earthaccess")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bbox = _parse_bbox(args.bbox)

    try:
        import earthaccess  # type: ignore
        has_earthaccess = True
    except ImportError:
        earthaccess = None  # type: ignore
        has_earthaccess = False

    # Search works without login, but authenticated login is needed for data downloads.
    if args.download and not has_earthaccess:
        raise RuntimeError("earthaccess is required for --download. Install with: pip install earthaccess")
    if args.download and has_earthaccess:
        earthaccess.login(strategy="interactive")

    args.outdir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    total_granules = 0
    total_mb = 0.0

    if not has_earthaccess:
        print("earthaccess not installed: using offline filename planning mode.")
        rows = _offline_plan_rows(
            short_names=list(args.short_names),
            start_date=args.start_date,
            end_date=args.end_date,
            limit_per_shortname=args.limit_per_shortname,
        )
        total_granules = len(rows)
    else:
        for short_name in args.short_names:
            kwargs: dict[str, object] = {
                "short_name": short_name,
                "temporal": (args.start_date, args.end_date),
            }
            if bbox is not None:
                kwargs["bounding_box"] = bbox
            granules = earthaccess.search_data(**kwargs)
            if args.limit_per_shortname > 0:
                granules = granules[: args.limit_per_shortname]

            print(f"[{short_name}] found granules: {len(granules)}")
            total_granules += len(granules)

            short_out = args.outdir / short_name
            if args.download:
                short_out.mkdir(parents=True, exist_ok=True)
                earthaccess.download(granules, local_path=str(short_out))

            for g in granules:
                size_mb = _try_size_mb(g)
                if size_mb is not None:
                    total_mb += size_mb
                links = _granule_links(g)
                rows.append(
                    {
                        "short_name": short_name,
                        "granule_id": _granule_id(g),
                        "size_mb": size_mb,
                        "first_data_link": links[0] if len(links) > 0 else "",
                        "n_links": len(links),
                        "mode": "earthaccess_search",
                    }
                )

                if len(links) > 0:
                    url_path = args.outdir / f"{short_name}_urls.txt"
                    with url_path.open("a", encoding="utf-8") as f:
                        for link in links:
                            f.write(f"{link}\n")

    manifest_path = args.outdir / "granule_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"Granule manifest written: {manifest_path}")
    print(f"Total granules: {total_granules}")
    if total_mb > 0.0:
        print(f"Approx total size (reported by CMR granules): {total_mb/1024.0:.2f} GB")
    if not args.download:
        print("Dry-run only. Use --download to fetch granules.")


if __name__ == "__main__":
    main()
