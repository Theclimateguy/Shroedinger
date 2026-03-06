#!/usr/bin/env python3
"""Build an aligned MRMS<->GOES nearest-scan catalog for pilot event analysis."""

from __future__ import annotations

import argparse
from datetime import timezone
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mrms-manifest", type=Path, default=Path("manifests/mrms_manifest.csv"))
    parser.add_argument("--goes-manifest", type=Path, default=Path("manifests/goes_manifest.csv"))
    parser.add_argument("--out", type=Path, default=Path("aligned_catalog.parquet"))
    parser.add_argument("--tolerance-minutes", type=int, default=10)
    parser.add_argument(
        "--goes-products",
        nargs="+",
        default=["ABI-L2-CMIPF", "GLM-L2-LCFA"],
        help="GOES products allowed for nearest mapping",
    )
    parser.add_argument(
        "--allowed-status",
        nargs="+",
        default=["downloaded", "exists"],
        help="Rows with these statuses are used for alignment",
    )
    return parser.parse_args()


def _load_manifest(path: Path, status_allow: set[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"].astype(str).isin(status_allow)].copy()
    if "obs_time_utc" not in df.columns:
        raise ValueError(f"Manifest missing obs_time_utc: {path}")
    if "event_id" not in df.columns:
        df["event_id"] = ""
    df["obs_dt"] = pd.to_datetime(df["obs_time_utc"], utc=True, errors="coerce")
    df = df.dropna(subset=["obs_dt"]).copy()
    return df


def _save_catalog(df: pd.DataFrame, out: Path) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".parquet":
        try:
            df.to_parquet(out, index=False)
            return out
        except Exception:
            fallback = out.with_suffix(".csv")
            df.to_csv(fallback, index=False)
            return fallback
    df.to_csv(out, index=False)
    return out


def main() -> None:
    args = parse_args()
    status_allow = {x.strip() for x in args.allowed_status if len(x.strip()) > 0}

    mrms = _load_manifest(args.mrms_manifest, status_allow=status_allow)
    goes = _load_manifest(args.goes_manifest, status_allow=status_allow)
    goes = goes[goes["product"].astype(str).isin(set(args.goes_products))].copy()

    if mrms.empty or goes.empty:
        empty = pd.DataFrame(
            columns=[
                "event_id",
                "mrms_obs_time_utc",
                "mrms_product",
                "mrms_key",
                "mrms_local_path",
                "mrms_file_size_bytes",
                "goes_obs_time_utc",
                "goes_satellite",
                "goes_product",
                "goes_abi_channel",
                "goes_key",
                "goes_local_path",
                "goes_file_size_bytes",
                "abs_time_delta_seconds",
                "match_within_tolerance",
            ]
        )
        saved = _save_catalog(empty, args.out)
        print(f"No alignable rows (mrms={len(mrms)}, goes={len(goes)}). Saved empty catalog: {saved}")
        return

    mrms = mrms.sort_values("obs_dt").reset_index(drop=True)
    goes = goes.sort_values("obs_dt").reset_index(drop=True)
    common_events = sorted(set(mrms["event_id"].astype(str)) & set(goes["event_id"].astype(str)))
    use_event_id = len(common_events) > 0 and (common_events != [""])

    if use_event_id:
        mrms = mrms[mrms["event_id"].astype(str).isin(common_events)].copy()
        goes = goes[goes["event_id"].astype(str).isin(common_events)].copy()
        # pandas merge_asof requires global monotonic order by the "on" key,
        # even when "by" is provided.
        mrms = mrms.sort_values(["obs_dt", "event_id"]).reset_index(drop=True)
        goes = goes.sort_values(["obs_dt", "event_id"]).reset_index(drop=True)
        aligned = pd.merge_asof(
            left=mrms,
            right=goes,
            by="event_id",
            on="obs_dt",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=args.tolerance_minutes),
            suffixes=("_mrms", "_goes"),
        )
    else:
        aligned = pd.merge_asof(
            left=mrms,
            right=goes,
            on="obs_dt",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=args.tolerance_minutes),
            suffixes=("_mrms", "_goes"),
        )
        if "event_id_mrms" in aligned.columns:
            aligned["event_id"] = aligned["event_id_mrms"].fillna("")
        elif "event_id" not in aligned.columns:
            aligned["event_id"] = ""

    aligned["mrms_obs_time_utc"] = aligned["obs_dt"].dt.tz_convert(timezone.utc).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    aligned["goes_obs_time_utc"] = pd.to_datetime(
        aligned["obs_time_utc_goes"], utc=True, errors="coerce"
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    goes_obs = pd.to_datetime(aligned["obs_time_utc_goes"], utc=True, errors="coerce")
    delta = (aligned["obs_dt"] - goes_obs).abs()
    aligned["abs_time_delta_seconds"] = delta.dt.total_seconds()
    aligned["match_within_tolerance"] = aligned["abs_time_delta_seconds"].notna()

    for col in [
        "file_size_bytes_mrms",
        "file_size_bytes_goes",
        "local_path_mrms",
        "local_path_goes",
        "key_mrms",
        "key_goes",
        "abi_channel",
        "satellite",
    ]:
        if col not in aligned.columns:
            aligned[col] = None

    out_cols = [
        "event_id",
        "mrms_obs_time_utc",
        "product_mrms",
        "key_mrms",
        "local_path_mrms",
        "file_size_bytes_mrms",
        "goes_obs_time_utc",
        "satellite",
        "product_goes",
        "abi_channel",
        "key_goes",
        "local_path_goes",
        "file_size_bytes_goes",
        "abs_time_delta_seconds",
        "match_within_tolerance",
    ]
    aligned = aligned.rename(
        columns={
            "product_mrms": "mrms_product",
            "key_mrms": "mrms_key",
            "local_path_mrms": "mrms_local_path",
            "file_size_bytes_mrms": "mrms_file_size_bytes",
            "satellite": "goes_satellite",
            "product_goes": "goes_product",
            "abi_channel": "goes_abi_channel",
            "key_goes": "goes_key",
            "local_path_goes": "goes_local_path",
            "file_size_bytes_goes": "goes_file_size_bytes",
        }
    )
    out_cols = [
        "event_id",
        "mrms_obs_time_utc",
        "mrms_product",
        "mrms_key",
        "mrms_local_path",
        "mrms_file_size_bytes",
        "goes_obs_time_utc",
        "goes_satellite",
        "goes_product",
        "goes_abi_channel",
        "goes_key",
        "goes_local_path",
        "goes_file_size_bytes",
        "abs_time_delta_seconds",
        "match_within_tolerance",
    ]
    aligned = aligned[out_cols]

    saved = _save_catalog(aligned, args.out)

    matched = int(aligned["match_within_tolerance"].sum())
    total = int(len(aligned))
    pct = 100.0 * matched / max(total, 1)
    print(f"Aligned catalog saved: {saved}")
    print(
        f"Alignment summary: total_mrms={total}, matched={matched}, "
        f"match_rate={pct:.2f}%, tolerance={args.tolerance_minutes} min"
    )


if __name__ == "__main__":
    main()
