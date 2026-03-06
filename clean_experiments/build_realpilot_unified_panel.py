#!/usr/bin/env python3
"""Build unified MRMS+ABI+GLM panel from two aligned-catalog CSV files.

Input A: ABI run aligned catalog (MRMS matched with ABI)
Input B: GLM run aligned catalog (MRMS matched with GLM)

Output columns follow M-realpilot v1 frozen panel schema.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ABI_REQUIRED = {
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
}

GLM_REQUIRED = {
    "event_id",
    "mrms_obs_time_utc",
    "mrms_key",
    "mrms_local_path",
    "mrms_file_size_bytes",
    "goes_obs_time_utc",
    "goes_satellite",
    "goes_product",
    "goes_key",
    "goes_local_path",
    "goes_file_size_bytes",
    "abs_time_delta_seconds",
    "match_within_tolerance",
}


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _require_cols(df: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{label} missing columns: {missing}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--abi-aligned-csv",
        type=Path,
        required=True,
        help="Aligned catalog CSV from ABI run (stage1).",
    )
    p.add_argument(
        "--glm-aligned-csv",
        type=Path,
        required=True,
        help="Aligned catalog CSV from GLM run (stage2).",
    )
    p.add_argument(
        "--out-csv",
        type=Path,
        required=True,
        help="Output unified panel CSV.",
    )
    p.add_argument(
        "--keep-unmatched",
        action="store_true",
        help="Keep rows where either source is outside tolerance; default drops them.",
    )
    p.add_argument(
        "--strict-existing-paths",
        action="store_true",
        help="Fail if any local data path in output is missing on disk.",
    )
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    abi = _load_csv(args.abi_aligned_csv)
    glm = _load_csv(args.glm_aligned_csv)

    _require_cols(abi, ABI_REQUIRED, "ABI aligned catalog")
    _require_cols(glm, GLM_REQUIRED, "GLM aligned catalog")

    if not args.keep_unmatched:
        abi = abi[abi["match_within_tolerance"].fillna(False)].copy()
        glm = glm[glm["match_within_tolerance"].fillna(False)].copy()

    abi = abi.rename(
        columns={
            "goes_obs_time_utc": "abi_obs_time_utc",
            "goes_satellite": "abi_satellite",
            "goes_product": "abi_product",
            "goes_abi_channel": "abi_channel",
            "goes_key": "abi_key",
            "goes_local_path": "abi_local_path",
            "goes_file_size_bytes": "abi_file_size_bytes",
            "abs_time_delta_seconds": "abi_abs_time_delta_seconds",
        }
    )

    glm = glm.rename(
        columns={
            "mrms_local_path": "mrms_local_path_glmrun",
            "mrms_file_size_bytes": "mrms_file_size_bytes_glmrun",
            "goes_obs_time_utc": "glm_obs_time_utc",
            "goes_satellite": "glm_satellite",
            "goes_product": "glm_product",
            "goes_key": "glm_key",
            "goes_local_path": "glm_local_path",
            "goes_file_size_bytes": "glm_file_size_bytes",
            "abs_time_delta_seconds": "glm_abs_time_delta_seconds",
        }
    )

    key_cols = ["event_id", "mrms_obs_time_utc", "mrms_key"]
    abi_key_uniq = abi[key_cols].drop_duplicates()
    glm_key_uniq = glm[key_cols].drop_duplicates()

    if len(abi_key_uniq) != len(abi):
        raise ValueError("ABI aligned catalog has duplicate (event_id, mrms_obs_time_utc, mrms_key) rows")
    if len(glm_key_uniq) != len(glm):
        raise ValueError("GLM aligned catalog has duplicate (event_id, mrms_obs_time_utc, mrms_key) rows")

    out = abi.merge(
        glm[
            [
                "event_id",
                "mrms_obs_time_utc",
                "mrms_key",
                "mrms_local_path_glmrun",
                "mrms_file_size_bytes_glmrun",
                "glm_obs_time_utc",
                "glm_satellite",
                "glm_product",
                "glm_key",
                "glm_local_path",
                "glm_file_size_bytes",
                "glm_abs_time_delta_seconds",
            ]
        ],
        on=key_cols,
        how="inner",
    )

    ordered_cols = [
        "event_id",
        "mrms_obs_time_utc",
        "mrms_product",
        "mrms_key",
        "mrms_local_path",
        "mrms_file_size_bytes",
        "abi_obs_time_utc",
        "abi_satellite",
        "abi_product",
        "abi_channel",
        "abi_key",
        "abi_local_path",
        "abi_file_size_bytes",
        "abi_abs_time_delta_seconds",
        "mrms_local_path_glmrun",
        "mrms_file_size_bytes_glmrun",
        "glm_obs_time_utc",
        "glm_satellite",
        "glm_product",
        "glm_key",
        "glm_local_path",
        "glm_file_size_bytes",
        "glm_abs_time_delta_seconds",
    ]

    out = out[ordered_cols].copy()
    out = out.sort_values(["event_id", "mrms_obs_time_utc"]).reset_index(drop=True)

    if args.strict_existing_paths:
        missing_rows = []
        for i, r in out.iterrows():
            for c in ["mrms_local_path", "abi_local_path", "glm_local_path"]:
                p = Path(str(r[c]))
                if not p.exists():
                    missing_rows.append((int(i), c, str(p)))
        if missing_rows:
            sample = "; ".join([f"row={i} col={c}" for i, c, _ in missing_rows[:6]])
            raise FileNotFoundError(f"Missing local files in unified panel ({len(missing_rows)} cells). Sample: {sample}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(f"Unified panel saved: {args.out_csv}")
    print(f"Rows={len(out)} events={out['event_id'].nunique()}")
    print(f"ABI rows input={len(abi)} GLM rows input={len(glm)}")


if __name__ == "__main__":
    run(parse_args())
