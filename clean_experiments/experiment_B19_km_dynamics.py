#!/usr/bin/env python3
"""Phase-19: dynamics re-asked on the equal-km territory.

Preregistered protocol: docs/PROTOCOL_PHASE19_KM_DYNAMICS.md
(frozen 2026-08-18).

Arm A: the FROZEN Phase-14 pipeline (code imported and run verbatim) on
km-cropped fields — the only change is the Phase-18 region geometry.
Arm B: tau_b(ell) scaling of the band-coupling fluctuation timescales,
with a single-timescale synthetic estimator gate.

Stages: --stage series | gate-b | tests
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import shutil
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (str(_HERE), str(_HERE.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import clean_experiments.experiment_B14_p_relaxation as b14
    from clean_experiments.experiment_B18_equal_km_regions import (
        HELDOUT_REGIONS,
        HELDOUT_WINDOWS,
        PRIMARY_WINDOWS,
        REGIONS,
        km_crop,
    )
    from clean_experiments.experiment_B1_true_flux_baselines import (
        _grid_geometry,
        _interior_mask,
        phase_randomize as phase_randomize_2d,
    )
    from clean_experiments.experiment_B2_scale_irreversibility import (
        ELLS_KM,
        N_STEPS,
        compute_vorticity,
        envelope_rho_profile,
    )
    from clean_experiments.experiment_scale_gravity_einstein_box_era import (
        _load_vector_fields,
    )
except ImportError:
    import experiment_B14_p_relaxation as b14  # type: ignore
    from experiment_B18_equal_km_regions import (  # type: ignore
        HELDOUT_REGIONS,
        HELDOUT_WINDOWS,
        PRIMARY_WINDOWS,
        REGIONS,
        km_crop,
    )
    from experiment_B1_true_flux_baselines import (  # type: ignore
        _grid_geometry,
        _interior_mask,
        phase_randomize as phase_randomize_2d,
    )
    from experiment_B2_scale_irreversibility import (  # type: ignore
        ELLS_KM,
        N_STEPS,
        compute_vorticity,
        envelope_rho_profile,
    )
    from experiment_scale_gravity_einstein_box_era import (  # type: ignore
        _load_vector_fields,
    )

SEED = 20260818
OUT = _HERE / "results" / "experiment_B19_km_dynamics"
B14_OUT = _HERE / "results" / "experiment_B14_p_relaxation"

WIND_B2B = Path("data/b2b")
WIND_B3 = Path("data/b3")
CAPE_DIR = Path("data/b6cape")
PRECIP_DIR = Path("data/b5precip")

STEP_ELL_KM = [float(np.sqrt(ELLS_KM[i] * ELLS_KM[i + 1])) for i in range(N_STEPS)]
DT_H = 6.0
TAU_R1_MAX = 0.99
MIN_VALID_BANDS = 4
GATE_REGIONS = ("R1_WPWP", "R2_NATL", "R7_CONGO", "R12_SAM")
GATE_REALIZATIONS = 5
GATE_A = float(np.exp(-DT_H / 12.0))
GATE_ALPHA_BAR = 0.15
N_BOOT = 9999
N_PERM = 999
C1_BOUNDS = (0.5, 0.995)


# --------------------------------------------------------------------------
# series (km-cropped; mirrors the frozen B14 series construction)
# --------------------------------------------------------------------------

def _cropped_interior_mean(path: Path, var: str, region: str,
                           mask: np.ndarray, transform) -> np.ndarray:
    import xarray as xr
    ds = xr.open_dataset(path)
    da = ds[var].squeeze()
    tdim = next(d for d in da.dims if "time" in d or d == "valid_time")
    ydim = "latitude" if "latitude" in da.dims else "lat"
    xdim = "longitude" if "longitude" in da.dims else "lon"
    arr = np.nan_to_num(np.asarray(
        da.transpose(tdim, ydim, xdim).values, dtype=float))
    lat = np.asarray(ds[ydim].values, dtype=float)
    lon = np.asarray(ds[xdim].values, dtype=float)
    ds.close()
    _, _, a2, _ = km_crop(lat, lon, arr, arr, region, 1.0)
    return transform(a2)[:, mask].mean(axis=1)


def _series_one(args_tuple) -> str:
    wind, cape, precip = args_tuple
    os.environ["OMP_NUM_THREADS"] = "1"
    tag = wind.stem.replace("era5_wind850_", "")
    cached = OUT / f"series_{tag}.npz"
    if cached.exists():
        return tag
    import xarray as xr
    region = tag.split("__")[0]
    _, lat, lon, u, v, _, _ = _load_vector_fields(
        input_path=wind, field_set="wind", u_var=None, v_var=None,
        time_stride=1, lat_stride=1, lon_stride=1, time_start=0,
        max_time=None, crop_ny=None, crop_nx=None,
    )
    ds = xr.open_dataset(wind)
    tname = "valid_time" if "valid_time" in ds else "time"
    hod = np.asarray(ds[tname].dt.hour.values, dtype=int)
    ds.close()
    lat, lon, u, v = km_crop(lat, lon, u, v, region, 1.0)
    dx_km, dy_km = _grid_geometry(lat, lon)
    mask = _interior_mask(u.shape[1], u.shape[2], ELLS_KM[-1], dx_km, dy_km)
    omega = compute_vorticity(u, v, lat, lon)
    P = envelope_rho_profile(omega, dx_km, dy_km, mask)
    V = b14.band_amplitude(omega, dx_km, dy_km, mask)
    payload = {"P": P.astype(np.float32), "V": V.astype(np.float32),
               "hod": hod[: P.shape[0]]}
    if cape is not None:
        payload["E"] = _cropped_interior_mean(
            cape, "cape", region, mask,
            lambda a: np.log1p(np.maximum(a, 0.0))).astype(np.float32)
        payload["R"] = _cropped_interior_mean(
            precip, "tp", region, mask,
            lambda a: np.log(np.maximum(a, 0.0) + 1e-6)).astype(np.float32)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cached, **payload)
    print(f"[{tag}] km series cached nt={P.shape[0]} "
          f"grid={u.shape[1]}x{u.shape[2]}", flush=True)
    return tag


def stage_series(workers: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = []
    for r in HELDOUT_REGIONS:
        for w in HELDOUT_WINDOWS:
            tag = f"{r}__{w}"
            wf = WIND_B2B / f"era5_wind850_{tag}.nc"
            cf = CAPE_DIR / f"era5_cape6h_{tag}.nc"
            pf = PRECIP_DIR / f"era5_precip_{tag}.nc"
            if wf.exists() and cf.exists() and pf.exists():
                jobs.append((wf, cf, pf))
    for r in REGIONS:
        for w in PRIMARY_WINDOWS:
            wf = WIND_B3 / f"era5_wind850_{r}__{w}.nc"
            if wf.exists():
                jobs.append((wf, None, None))
    print(f"{len(jobs)} region-windows to process", flush=True)
    ctx = mp.get_context("fork")
    with ctx.Pool(workers) as pool:
        pool.map(_series_one, jobs)


# --------------------------------------------------------------------------
# Arm B: tau_b(ell) machinery
# --------------------------------------------------------------------------

def band_taus(P5: np.ndarray, hod: np.ndarray) -> list[float | None]:
    taus: list[float | None] = []
    for b in range(N_STEPS):
        x = b14.deseason(P5[:, b], hod)
        r1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        taus.append(float(-DT_H / np.log(r1)) if 0.0 < r1 < TAU_R1_MAX else None)
    return taus


def alpha_from_taus(taus: list[float | None]) -> float | None:
    pts = [(np.log(STEP_ELL_KM[i]), np.log(t))
           for i, t in enumerate(taus) if t is not None]
    if len(pts) < MIN_VALID_BANDS:
        return None
    x = np.asarray([p[0] for p in pts])
    y = np.asarray([p[1] for p in pts])
    X = np.column_stack([np.ones_like(x), x])
    return float(np.linalg.lstsq(X, y, rcond=None)[0][1])


def stage_gate_b() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cache = OUT / "gate_b.json"
    if cache.exists():
        print("cached", flush=True)
        return
    rng = np.random.default_rng(SEED)
    rows = []
    for region in GATE_REGIONS:
        wf = WIND_B3 / f"era5_wind850_{region}__W9_2023JFM.nc"
        _, lat, lon, u, v, _, _ = _load_vector_fields(
            input_path=wf, field_set="wind", u_var=None, v_var=None,
            time_stride=1, lat_stride=1, lon_stride=1, time_start=0,
            max_time=None, crop_ny=None, crop_nx=None,
        )
        lat, lon, u, v = km_crop(lat, lon, u, v, region, 1.0)
        dx_km, dy_km = _grid_geometry(lat, lon)
        mask = _interior_mask(u.shape[1], u.shape[2], ELLS_KM[-1], dx_km, dy_km)
        nt = u.shape[0]
        hod = np.tile([0, 6, 12, 18], nt // 4 + 1)[:nt]
        for rep in range(GATE_REALIZATIONS):
            # innovations: temporally incoherent, spatial spectra preserved
            xi_u, xi_v = phase_randomize_2d(u, v, rng)
            su = np.empty_like(xi_u)
            sv = np.empty_like(xi_v)
            su[0], sv[0] = xi_u[0], xi_v[0]
            c = float(np.sqrt(1.0 - GATE_A * GATE_A))
            for t in range(1, nt):
                su[t] = GATE_A * su[t - 1] + c * xi_u[t]
                sv[t] = GATE_A * sv[t - 1] + c * xi_v[t]
            omega = compute_vorticity(su, sv, lat, lon)
            P5 = envelope_rho_profile(omega, dx_km, dy_km, mask)
            taus = band_taus(P5, hod)
            a = alpha_from_taus(taus)
            rows.append({"region": region, "rep": rep, "alpha": a,
                         "taus": taus})
            print(f"[gate {region} rep{rep}] alpha="
                  f"{a if a is None else round(a, 3)}", flush=True)
    alphas = [r["alpha"] for r in rows if r["alpha"] is not None]
    med = float(np.median(np.abs(alphas))) if alphas else None
    res = {"rows": rows, "median_abs_alpha": med,
           "n_valid": len(alphas),
           "pass": bool(med is not None and med <= GATE_ALPHA_BAR)}
    cache.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"GATE-B median|alpha|={med} pass={res['pass']}", flush=True)


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------

def _composite_deseason(npz_path: Path) -> np.ndarray:
    z = np.load(npz_path)
    P5 = np.asarray(z["P"], dtype=float)
    hod = np.asarray(z["hod"], dtype=int)
    comp = P5[:, list(b14.PAIRS_RESOLVED)].mean(axis=1)
    return b14.deseason(comp, hod)


def stage_tests() -> dict:
    rng = np.random.default_rng(SEED)
    res: dict = {"seed": SEED,
                 "protocol": "docs/PROTOCOL_PHASE19_KM_DYNAMICS.md"}

    # ---- C19-1 sanity ----------------------------------------------------
    corrs = {}
    for r in HELDOUT_REGIONS:
        for w in HELDOUT_WINDOWS:
            tag = f"{r}__{w}"
            km_f = OUT / f"series_{tag}.npz"
            dg_f = B14_OUT / f"series_{tag}.npz"
            if km_f.exists() and dg_f.exists():
                a = _composite_deseason(km_f)
                b = _composite_deseason(dg_f)
                n = min(len(a), len(b))
                corrs[tag] = float(np.corrcoef(a[:n], b[:n])[0, 1])
    med_corr = float(np.median(list(corrs.values())))
    c19_1 = {"per_rw": corrs, "median": med_corr,
             "bounds": list(C1_BOUNDS),
             "informative": bool(C1_BOUNDS[0] <= med_corr <= C1_BOUNDS[1]),
             "abort": bool(med_corr < C1_BOUNDS[0])}
    res["C19_1"] = c19_1
    if c19_1["abort"]:
        res["ARM_A_VERDICT"] = "ABORT_MEASUREMENT_INSTABILITY"

    # ---- Arm A: frozen B14 ladder on km series ---------------------------
    if not c19_1["abort"]:
        gate_src = B14_OUT / "synthetic_h14a.json"
        gate_dst = OUT / "synthetic_h14a.json"
        if not gate_dst.exists():
            shutil.copy(gate_src, gate_dst)
        b14_out_orig = b14.OUT
        try:
            b14.OUT = OUT
            res_a = b14.stage_tests()
        finally:
            b14.OUT = b14_out_orig
        res["ARM_A_B14_ladder"] = res_a
        ladder = res_a["PHASE14_VERDICT"]
        reopened = ladder in ("P_DYNAMICS_ESTABLISHED", "P_DYNAMICS_PARTIAL")
        res["ARM_A_VERDICT"] = ("REOPENED" if reopened
                                else "NEGATIVE_TERRITORY_ROBUST")
        if not c19_1["informative"]:
            res["ARM_A_VERDICT"] += "_UNINFORMATIVE"
        # explicit sign-split tracking (R7 vs R5, all windows, km territory)
        split = {}
        for reg in ("R7_CONGO", "R5_SPCZ"):
            alphas = []
            for w in HELDOUT_WINDOWS:
                tag = f"{reg}__{w}"
                f = OUT / f"series_{tag}.npz"
                if f.exists():
                    rw = _load_rw_km(tag)
                    alphas.append(float(b14.alpha_gamma(rw["P"], rw["E"])[0]))
            split[reg] = alphas
        res["sign_split_km"] = {
            k: {"alphas": v,
                "n_pos": int(sum(1 for a in v if a > 0)),
                "n_neg": int(sum(1 for a in v if a < 0))}
            for k, v in split.items()}

    # ---- Arm B -----------------------------------------------------------
    gate = json.loads((OUT / "gate_b.json").read_text(encoding="utf-8"))
    res["H19B_gate"] = {k: gate[k] for k in
                        ("median_abs_alpha", "n_valid", "pass")}

    rows = []
    for f in sorted(OUT.glob("series_*.npz")):
        tag = f.stem.replace("series_", "")
        z = np.load(f)
        P5 = np.asarray(z["P"], dtype=float)
        hod = np.asarray(z["hod"], dtype=int)
        taus = band_taus(P5, hod)
        a = alpha_from_taus(taus)
        rows.append({"tag": tag, "region": tag.split("__")[0],
                     "taus_h": taus, "alpha": a})
    res["ARM_B_per_rw"] = rows

    valid = [r for r in rows if r["alpha"] is not None]
    alphas = np.asarray([r["alpha"] for r in valid])
    regions = np.asarray([r["region"] for r in valid])
    med_alpha = float(np.median(alphas))

    # cluster bootstrap over regions
    reg_list = sorted(set(regions))
    by_reg = {g: alphas[regions == g] for g in reg_list}
    boots = np.empty(N_BOOT)
    for i in range(N_BOOT):
        pick = rng.choice(reg_list, size=len(reg_list), replace=True)
        boots[i] = np.median(np.concatenate([by_reg[g] for g in pick]))
    ci = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
    exists = bool(med_alpha > 0 and ci[0] > 0)

    if ci[0] <= 2.0 / 3.0 <= ci[1] and not (ci[0] <= 1.0 <= ci[1]):
        klass = "TURNOVER"
    elif ci[0] <= 1.0 <= ci[1] and not (ci[0] <= 2.0 / 3.0 <= ci[1]):
        klass = "SWEEPING"
    else:
        klass = "PRESENT_UNCLASSIFIED"

    # universality: between vs within region alpha spread
    reg_med = np.asarray([np.median(by_reg[g]) for g in reg_list])
    between_obs = float(np.std(reg_med))
    perm_b = np.empty(N_PERM)
    for i in range(N_PERM):
        lab = regions[rng.permutation(len(regions))]
        perm_b[i] = np.std([np.median(alphas[lab == g]) for g in reg_list])
    p_uni = float((1 + np.sum(perm_b >= between_obs)) / (N_PERM + 1))

    res["H19B"] = {
        "n_valid_rw": len(valid), "n_rw": len(rows),
        "median_alpha": med_alpha, "ci95_cluster_bootstrap": list(ci),
        "exists_pass": exists, "class": klass if exists else None,
        "universality": {"between_region_sd": between_obs, "p": p_uni,
                         "region_medians": {g: float(np.median(by_reg[g]))
                                            for g in reg_list}},
        "median_taus_h": [float(np.median([r["taus_h"][b] for r in valid
                                           if r["taus_h"][b] is not None]))
                          for b in range(N_STEPS)],
        "step_ell_km": STEP_ELL_KM,
    }
    if not gate["pass"]:
        res["ARM_B_VERDICT"] = "ESTIMATOR_INVALID"
    elif not exists:
        res["ARM_B_VERDICT"] = "NO_SCALING"
    else:
        res["ARM_B_VERDICT"] = f"TIMESCALE_LAW_{klass}"

    res["PHASE19_VERDICT"] = [res.get("ARM_A_VERDICT"), res["ARM_B_VERDICT"]]
    return res


def _load_rw_km(tag: str) -> dict:
    b14_out_orig = b14.OUT
    try:
        b14.OUT = OUT
        return b14.load_rw(tag)
    finally:
        b14.OUT = b14_out_orig


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=["series", "gate-b", "tests"],
                    default="series")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 4) - 3))
    args = ap.parse_args()
    if args.stage == "series":
        stage_series(args.workers)
    elif args.stage == "gate-b":
        stage_gate_b()
    else:
        res = stage_tests()
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "summary.json").write_text(json.dumps(res, indent=2),
                                          encoding="utf-8")
        brief = {
            "C19_1_median": res["C19_1"]["median"],
            "ARM_A_VERDICT": res.get("ARM_A_VERDICT"),
            "ARM_A_ladder": res.get("ARM_A_B14_ladder", {}).get("PHASE14_VERDICT"),
            "sign_split_km": res.get("sign_split_km"),
            "H19B_gate": res["H19B_gate"],
            "H19B_median_alpha": res["H19B"]["median_alpha"],
            "H19B_ci": res["H19B"]["ci95_cluster_bootstrap"],
            "ARM_B_VERDICT": res["ARM_B_VERDICT"],
            "PHASE19_VERDICT": res["PHASE19_VERDICT"],
        }
        print(json.dumps(brief, indent=2))


if __name__ == "__main__":
    main()
