#!/usr/bin/env python3
"""Data-driven figures for the revised manuscript (v8) of article 1.

Journal rules (Izvestiya RAN, Ser. Geogr.): width exactly 17.5 cm, height
4.7-17.5 cm, dpi 400, no title text on the figure field, readable in
black-and-white print, Russian labels, DejaVu Sans 7-8 pt.

Inputs (recomputed "V12" pipeline, all JSON under clean_experiments/results):
  experiment_B2_scale_irreversibility_v12   16 region-windows (R1-R4, W1-W4)
  experiment_B2b_heldout_invariants_v12     32 region-windows (R5-R12, W5-W8)
  experiment_B3_scattering_benchmark_v12    48 region-windows (R1-R12, W9-W12)
  experiment_B18_equal_km_regions_v12       equal-km boxes (km_1.0), summary.json
  experiment_B13_free_running_v12           free-running model vs ERA5, 39 domains
  download_b11_wb2.DOMAINS                  box corners (north, west, south, east)
  data/b4cov/era5_invariants_global.nc      land-sea mask for coastlines (no cartopy)

Outputs: manuscript_v2/article1/figures/v8/*.png and captions_draft_data_figures.txt

Usage (from the repo root):
  python3 clean_experiments/visualize_article1_v8_figures.py
  python3 clean_experiments/visualize_article1_v8_figures.py --only fig2,fig4
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RES = HERE / "results"
OUT = ROOT / "manuscript_v2" / "article1" / "figures" / "v8"
LSM_FILE = ROOT / "data" / "b4cov" / "era5_invariants_global.nc"

sys.path.insert(0, str(HERE))
from download_b11_wb2 import DOMAINS  # noqa: E402

CM = 1.0 / 2.54
WIDTH_CM = 17.5
DPI = 400

ORDER = ["R1_WPWP", "R2_NATL", "R3_AMAZ", "R4_CASIA", "R5_SPCZ", "R6_SATL",
         "R7_CONGO", "R8_AUS", "R9_NPAC", "R10_INDO", "R11_EURO", "R12_SAM"]
SHORT = {r: r.split("_")[0] for r in ORDER}
NAMES = {
    "R1_WPWP": "Теплый бассейн\nТихого океана", "R2_NATL": "Северная Атлантика",
    "R3_AMAZ": "Амазония", "R4_CASIA": "Центральная Азия", "R5_SPCZ": "ЮТЗК",
    "R6_SATL": "Южная Атлантика", "R7_CONGO": "Бассейн Конго", "R8_AUS": "Австралия",
    "R9_NPAC": "Север Тихого океана", "R10_INDO": "Индийский океан",
    "R11_EURO": "Европа", "R12_SAM": "Юг Южной Америки",
}
# Regimes: (label, colour, linestyle, member regions). Colours are Okabe-Ito
# (colour-blind safe); line styles make the regimes distinct in B/W print.
REGIMES = [
    ("тропический океан", "#0072B2", "-", ["R1_WPWP", "R5_SPCZ", "R10_INDO"]),
    ("штормовые пути", "#000000", "--", ["R2_NATL", "R6_SATL", "R9_NPAC"]),
    ("глубокая конвекция над сушей", "#009E73", "-.", ["R3_AMAZ", "R7_CONGO"]),
    ("континентальный умеренный", "#D55E00", ":", ["R4_CASIA", "R11_EURO"]),
    ("континентальный субтропический", "#CC79A7", (0, (5, 1.5, 1, 1.5)), ["R8_AUS", "R12_SAM"]),
]
REGION_STYLE = {}
for _lab, _col, _ls, _members in REGIMES:
    for _i, _r in enumerate(_members):
        REGION_STYLE[_r] = dict(color=_col, ls=_ls, marker=["o", "s", "^"][_i], regime=_lab)

STEP_LABELS = ["<50 |\n50–100", "50–100 |\n100–200", "100–200 |\n200–400",
               "200–400 |\n400–800", "400–800 |\n800–1600"]
STEP_LABELS_1LINE = [lab.replace("\n", " ") for lab in STEP_LABELS]
MAP_ASPECT = 1.0  # lat/lon stretch of the map; 1.0 = Plate Carrée with equal degrees
PERIODS = [("I (2017–2019)", range(1, 5), "#0072B2", "o", "-"),
           ("II (2021–2022)", range(5, 9), "#D55E00", "s", "--"),
           ("III (2023–2024)", range(9, 13), "#000000", "^", "-.")]
PRIMARY = {9, 10, 11, 12}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5,
    "axes.linewidth": 0.6, "axes.titlesize": 8, "axes.labelsize": 7.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.unicode_minus": False, "savefig.dpi": DPI,
})

PROBLEMS: list[str] = []
REPORT: dict[str, object] = {}


def _jload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_window(code: str) -> tuple[int, int, str]:
    m = re.match(r"W(\d+)_(\d{4})(JFM|JAS)", code)
    if not m:
        raise ValueError(f"bad window code {code}")
    return int(m.group(1)), int(m.group(2)), m.group(3)


def period_of(wnum: int) -> int:
    for i, (_, rng, *_rest) in enumerate(PERIODS):
        if wnum in rng:
            return i
    raise ValueError(wnum)


def check_nan(arr, where: str) -> np.ndarray:
    a = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(a)):
        PROBLEMS.append(f"non-finite values in {where}")
    return a


# --------------------------------------------------------------------------
# data loaders
# --------------------------------------------------------------------------

def load_degree_profiles() -> dict[str, list[dict]]:
    """96 region-windows on degree boxes: region -> list of records."""
    out = {r: [] for r in ORDER}
    n = 0
    for sub in ("experiment_B2_scale_irreversibility_v12",
                "experiment_B2b_heldout_invariants_v12",
                "experiment_B3_scattering_benchmark_v12"):
        files = sorted(glob.glob(str(RES / sub / "R*__*.json")))
        if not files:
            PROBLEMS.append(f"no R*.json files in {sub}")
        for f in files:
            rec = _jload(Path(f))
            wnum, year, season = parse_window(rec["window"])
            out[rec["region"]].append(dict(
                window=rec["window"], wnum=wnum, year=year, season=season,
                period=period_of(wnum),
                P=check_nan(rec["P_real"], f"{sub}/{Path(f).name} P_real"),
                Ps=check_nan(rec["P_surrogate_median"], f"{sub}/{Path(f).name} P_surrogate_median"),
            ))
            n += 1
    if n != 96:
        PROBLEMS.append(f"expected 96 degree-box region-windows, found {n}")
    for r in ORDER:
        if len(out[r]) != 8:
            PROBLEMS.append(f"{r}: {len(out[r])} degree-box windows (expected 8)")
    REPORT["n_degree_region_windows"] = n
    return out


def load_equal_km() -> tuple[dict[str, list[dict]], dict]:
    """Equal-km boxes (B18 v12, km_1.0): region -> list of records; summary."""
    out = {r: [] for r in ORDER}
    files = sorted(glob.glob(str(RES / "experiment_B18_equal_km_regions_v12" / "R*__*.json")))
    if not files:
        PROBLEMS.append("no R*.json files in experiment_B18_equal_km_regions_v12")
    for f in files:
        rec = _jload(Path(f))
        wnum, year, season = parse_window(rec["window"])
        km = rec["km_1.0"]
        out[rec["region"]].append(dict(
            window=rec["window"], wnum=wnum, year=year, season=season,
            P=check_nan(km["P_real"], f"B18 {Path(f).name} P_real"),
            Ps=check_nan(km["P_surrogate_median"], f"B18 {Path(f).name} P_surrogate_median"),
            Pa=check_nan(km["P_anchored"], f"B18 {Path(f).name} P_anchored"),
        ))
    summary = _jload(RES / "experiment_B18_equal_km_regions_v12" / "summary.json")
    REPORT["n_equal_km_region_windows"] = sum(len(v) for v in out.values())
    return out, summary


def load_free_running() -> tuple[dict, dict[str, list[float]]]:
    """B13 v12: domain_values (39 domains) and per-window ERA5 P per domain."""
    summary = _jload(RES / "experiment_B13_free_running_v12" / "summary.json")
    dv = summary["domain_values"]
    if len(dv) != 39:
        PROBLEMS.append(f"domain_values has {len(dv)} domains (expected 39)")
    per_window: dict[str, list[float]] = {}
    for dom in dv:
        files = sorted(glob.glob(str(RES / "experiment_B13_free_running_v12" / f"era5_{dom}__*.json")))
        vals = [float(_jload(Path(f))["P"]) for f in files]
        if not vals:
            PROBLEMS.append(f"no per-window ERA5 files for {dom}")
        if any(not np.isfinite(v) for v in vals):
            PROBLEMS.append(f"non-finite per-window P for {dom}")
        per_window[dom] = vals
        for k, v in dv[dom].items():
            if not np.isfinite(v):
                PROBLEMS.append(f"domain_values[{dom}][{k}] non-finite")
    REPORT["H13a_rho_from_summary"] = summary.get("H13a", {}).get("rho")
    REPORT["windows_per_domain"] = {d: len(v) for d, v in per_window.items()}
    return dv, per_window


def spearman(x, y) -> float:
    try:
        from scipy.stats import spearmanr
        return float(spearmanr(x, y).correlation)
    except Exception:  # pragma: no cover - fallback without scipy
        rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
        return float(np.corrcoef(rx, ry)[0, 1])


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def spread_labels(ys: np.ndarray, min_gap: float) -> np.ndarray:
    """Push label y-positions apart so that none are closer than min_gap."""
    ys = np.asarray(ys, dtype=float)
    order = np.argsort(ys)
    y = ys[order].copy()
    for i in range(1, len(y)):
        if y[i] - y[i - 1] < min_gap:
            y[i] = y[i - 1] + min_gap
    y -= y.mean() - ys.mean()
    out = np.empty_like(y)
    out[order] = y
    return out


def panel_letter(ax, letter: str) -> None:
    ax.set_title(letter, loc="left", fontsize=8, pad=3)


def save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def regime_legend_handles():
    return [Line2D([], [], color=c, ls=ls, lw=1.1, label=lab) for lab, c, ls, _ in REGIMES]


def draw_profiles_by_region(ax, values: dict[str, np.ndarray], steps: np.ndarray,
                            label_gap: float) -> None:
    """One line per region on the x-grid `steps`, styled by regime, code at the right end."""
    y_end = np.array([values[r][-1] for r in ORDER])
    y_lab = spread_labels(y_end, label_gap)
    for r, yl in zip(ORDER, y_lab):
        st = REGION_STYLE[r]
        ax.plot(steps, values[r], color=st["color"], ls=st["ls"], marker=st["marker"],
                ms=3, lw=1.0, mew=0.6, mfc="white" if st["marker"] != "o" else st["color"])
        ax.annotate(SHORT[r], xy=(steps[-1], values[r][-1]),
                    xytext=(steps[-1] + 0.12, yl), textcoords="data",
                    fontsize=6.5, va="center", ha="left", color=st["color"],
                    arrowprops=dict(arrowstyle="-", color="0.6", lw=0.4,
                                    shrinkA=0, shrinkB=1.5))
    ax.set_xticks(steps)
    ax.set_xlim(steps[0] - 0.25, steps[-1] + 0.75)
    ax.grid(alpha=0.3, lw=0.4)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def fig2(deg: dict, km: dict) -> Path:
    """Profiles, steps 2-5: (a) raw P, median over 8 windows; (b) P_anchored on equal-km
    boxes, primary windows W9-W12, median over 4 windows."""
    steps = np.arange(2, 6)
    raw = {r: np.median(np.array([w["P"] for w in deg[r]]), axis=0)[1:] for r in ORDER}
    anch = {}
    for r in ORDER:
        recs = [w["Pa"] for w in km[r] if w["wnum"] in PRIMARY]
        if len(recs) != 4:
            PROBLEMS.append(f"fig2: {r} has {len(recs)} primary equal-km windows (expected 4)")
        anch[r] = np.median(np.array(recs), axis=0)[1:]

    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_CM * CM, 8.6 * CM))
    for ax, vals, letter, ylab in ((axes[0], raw, "(а)", "сопряженность"),
                                   (axes[1], anch, "(б)", "превышение над суррогатами")):
        rng = max(v.max() for v in vals.values()) - min(v.min() for v in vals.values())
        draw_profiles_by_region(ax, vals, steps, label_gap=0.045 * rng)
        ax.set_xticklabels(STEP_LABELS[1:], fontsize=7)
        ax.set_xlabel("пары соседних полос, км")
        ax.set_ylabel(ylab)
        panel_letter(ax, letter)
    fig.legend(handles=regime_legend_handles(), loc="lower center", ncol=3, frameon=False,
               fontsize=7, handlelength=2.8, columnspacing=1.8, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.11, 1, 1), w_pad=1.5)
    return save(fig, "fig2_profiles_v8.png")


LABEL_OFFSETS = [(2.5, 2.5), (2.5, -7.5), (-3.0, 2.5), (-3.0, -7.5), (5.5, -2.5), (-6.0, -2.5),
                 (2.5, 7.5), (-3.0, 7.5), (2.5, -12.5), (-3.0, -12.5), (9.0, 4.0), (9.0, -9.0),
                 (-9.0, 4.0), (-9.0, -9.0), (12.0, 8.0), (-12.0, 8.0), (12.0, -13.0), (-12.0, -13.0),
                 (4.0, 14.0), (-4.0, 14.0), (4.0, -19.0), (-4.0, -19.0)]


def annotate_codes(ax, pts: dict[str, tuple[float, float]], fontsize: float, avoid: list | None = None):
    """Greedy placement of short code labels next to points: the first candidate offset whose
    estimated text box neither overlaps an already placed label nor covers another point."""
    fig = ax.figure
    px_per_pt = fig.dpi / 72.0
    disp = {k: ax.transData.transform(v) for k, v in pts.items()}
    others = [ax.transData.transform(v) for v in (avoid or [])]
    placed = []
    r_pt = 2.6  # marker radius in points to keep clear
    ax_box = ax.get_window_extent()
    for code, (x, y) in pts.items():
        w = 0.62 * fontsize * len(code) * px_per_pt
        h = 1.05 * fontsize * px_per_pt
        x0, y0 = disp[code]
        chosen = LABEL_OFFSETS[0]; chosen_idx = 0
        for idx, (dx, dy) in enumerate(LABEL_OFFSETS):
            lx = x0 + dx * px_per_pt - (w if dx < 0 else 0)
            ly = y0 + dy * px_per_pt - (h if dy < 0 else 0)
            box = (lx, ly, lx + w, ly + h)
            def hit(b):
                return not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3])
            pts_boxes = [(cx - r_pt * px_per_pt, cy - r_pt * px_per_pt, cx + r_pt * px_per_pt, cy + r_pt * px_per_pt)
                         for k, (cx, cy) in disp.items() if k != code] + \
                        [(cx - r_pt * px_per_pt, cy - r_pt * px_per_pt, cx + r_pt * px_per_pt, cy + r_pt * px_per_pt)
                         for (cx, cy) in others]
            inside_axes = (box[0] >= ax_box.x0 and box[2] <= ax_box.x1 and
                           box[1] >= ax_box.y0 and box[3] <= ax_box.y1)
            if inside_axes and not any(hit(b) for b in placed) and not any(hit(b) for b in pts_boxes):
                chosen = (dx, dy); chosen_idx = idx; placed.append(box); break
        else:
            dx, dy = chosen
            lx = x0 + dx * px_per_pt - (w if dx < 0 else 0); ly = y0 + dy * px_per_pt - (h if dy < 0 else 0)
            placed.append((lx, ly, lx + w, ly + h))
        dx, dy = chosen
        far = chosen_idx >= 1  # every non-default placement gets a thin leader line
        ax.annotate(code, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=fontsize,
                    ha="right" if dx < 0 else "left", va="bottom" if dy >= 0 else "top", zorder=4,
                    arrowprops=dict(arrowstyle="-", color="0.5", lw=0.4, shrinkA=0, shrinkB=2.5) if far else None)


def _scatter_domains(ax, dv: dict, xkey: str, ykey: str, annotate: bool, label_size: float = 6.5):
    lattice = [d for d in dv if d not in ORDER]
    xs_l = [dv[d][xkey] for d in lattice]; ys_l = [dv[d][ykey] for d in lattice]
    ax.scatter(xs_l, ys_l, s=14, facecolors="none", edgecolors="0.45", linewidths=0.6,
               zorder=2, label="контрольные области")
    xs_m = [dv[r][xkey] for r in ORDER]; ys_m = [dv[r][ykey] for r in ORDER]
    ax.scatter(xs_m, ys_m, s=16, c="black", marker="o", linewidths=0, zorder=3,
               label="основные области")
    allx = xs_l + xs_m; ally = ys_l + ys_m
    lo = min(min(allx), min(ally)); hi = max(max(allx), max(ally))
    pad = 0.06 * (hi - lo)
    lo -= pad; hi += pad
    ax.plot([lo, hi], [lo, hi], ls="--", color="0.5", lw=0.7, zorder=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3, lw=0.4)
    if annotate:
        annotate_codes(ax, {SHORT[r]: (dv[r][xkey], dv[r][ykey]) for r in ORDER}, label_size,
                       avoid=list(zip(xs_l, ys_l)))
    return spearman([dv[d][xkey] for d in dv], [dv[d][ykey] for d in dv])


def fig3(dv: dict) -> Path:
    fig, ax = plt.subplots(figsize=(WIDTH_CM * CM, 8.5 * CM))
    ax.set_xlabel("ERA5, среднее ступеней 4–5")
    ax.set_ylabel("свободный расчет ECMWF-IFS-HR")
    ax.set_box_aspect(1)
    fig.tight_layout()
    rho = _scatter_domains(ax, dv, "P_era5", "P_model", annotate=True)
    REPORT["fig3_rho_P_computed"] = rho
    ax.legend(loc="upper left", frameon=False, fontsize=7, handletextpad=0.4)
    return save(fig, "fig3_free_run_v8.png")


def figA3(dv: dict) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH_CM * CM, 6.5 * CM))
    specs = [("A_era5", "A_model", "(а)", "индекс невзаимности A"),
             ("u_era5", "u_model", "(б)", "невосстановимость u"),
             ("persist_era5", "persist_model", "(в)", "автокорреляция, лаг 6 ч")]
    for ax, (xk, yk, letter, lab) in zip(axes, specs):
        ax.set_title(f"{letter} {lab}", loc="left", fontsize=7.5, pad=3)
        ax.set_xlabel("ERA5", fontsize=7)
        ax.set_ylabel("свободный расчет", fontsize=7)
        ax.set_box_aspect(1)
    fig.tight_layout(w_pad=1.2)
    rhos = {}
    for ax, (xk, yk, letter, lab) in zip(axes, specs):
        rhos[xk.split("_")[0]] = _scatter_domains(ax, dv, xk, yk, annotate=True, label_size=5.5)
    REPORT["figA3_rho_computed"] = rhos
    return save(fig, "figA3_free_run_extra_v8.png")


def load_coastlines():
    """Return (lon, lat, lsm) rolled to -180..180, or None."""
    try:
        import cartopy  # noqa: F401
        REPORT["coastlines"] = "cartopy available but not used (plain axes); lsm contour used"
    except Exception:
        REPORT["coastlines"] = "cartopy unavailable"
    path = LSM_FILE
    if not path.exists():
        cands = sorted(glob.glob(str(ROOT / "data" / "**" / "*invariants*.nc"), recursive=True))
        path = Path(cands[0]) if cands else None
    if path is None:
        PROBLEMS.append("no land-sea mask found under data/; map drawn without coastlines")
        REPORT["coastlines"] += "; no lsm file"
        return None
    try:
        import xarray as xr
        ds = xr.open_dataset(path)
        lsm = ds["lsm"].squeeze().values.astype(np.float32)
        lat = ds["latitude"].values.astype(float)
        lon = ds["longitude"].values.astype(float)
    except Exception as exc:  # pragma: no cover
        PROBLEMS.append(f"could not read lsm from {path}: {exc}")
        return None
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon180)
    lon180 = lon180[order]; lsm = lsm[:, order]
    # subsample to 0.5 deg for speed
    lat = lat[::2]; lon180 = lon180[::2]; lsm = lsm[::2, ::2]
    REPORT["coastlines"] += f"; lsm contour from {path.relative_to(ROOT)}"
    return lon180, lat, lsm


LAT_MIN, LAT_MAX = -60.0, 60.0   # -60 so that the R6 box (to 55S) is shown whole


def _text_anchor(mask: np.ndarray, lons: np.ndarray, lats: np.ndarray) -> tuple[float, float, float]:
    """Point of the visible region farthest from its boundary (centre of the largest
    inscribed disc) and that distance in degrees."""
    from scipy.ndimage import distance_transform_edt
    if not mask.any():
        return float("nan"), float("nan"), 0.0
    step = float(abs(lons[1] - lons[0]))
    # text lines are ~2.4x wider than tall: shrink the lon axis so that lon clearance counts more
    dist = distance_transform_edt(np.pad(mask, 1), sampling=(step, step / 2.4))[1:-1, 1:-1]
    j, i = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return float(lons[i]), float(lats[j]), float(dist[j, i])


def fig4(dv: dict, per_window: dict) -> Path:
    coast = load_coastlines()
    vals = {d: dv[d]["P_era5"] for d in dv}
    vmin = min(vals.values()); vmax = max(vals.values())
    cmap = plt.get_cmap("cividis")
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    # geometry in cm: axes width fixed, map height follows from the lat/lon extent
    left_cm, right_cm = 1.35, 0.25
    map_w = WIDTH_CM - left_cm - right_cm
    map_h = map_w * (LAT_MAX - LAT_MIN) / 360.0 * MAP_ASPECT
    cb_h, cb_block, xt_block, top_cm = 0.28, 1.3, 0.55, 0.3
    H = cb_block + xt_block + map_h + top_cm
    fig = plt.figure(figsize=(WIDTH_CM * CM, H * CM))
    ax = fig.add_axes([left_cm / WIDTH_CM, (cb_block + xt_block) / H, map_w / WIDTH_CM, map_h / H])
    cax = fig.add_axes([0.30, (cb_block - cb_h) / H, 0.40, cb_h / H])
    if coast is not None:
        lon, lat, lsm = coast
        ax.contourf(lon, lat, lsm, levels=[0.5, 1.5], colors=["0.92"], zorder=0)
        ax.contour(lon, lat, lsm, levels=[0.5], colors=["0.55"], linewidths=0.35, zorder=1)
    missing = [d for d in dv if d not in DOMAINS]
    if missing:
        PROBLEMS.append(f"domains without box corners in DOMAINS: {missing}")

    # visibility grid (0.5 deg) to place text where a box is not covered by boxes drawn later
    glon = np.arange(-180 + 0.25, 180, 0.5); glat = np.arange(LAT_MIN + 0.25, LAT_MAX, 0.5)
    GLON, GLAT = np.meshgrid(glon, glat)

    def inside(d):
        n, w, s_, e = DOMAINS[d]
        return (GLON > w) & (GLON < e) & (GLAT > s_) & (GLAT < n)

    lattice = [d for d in dv if d not in ORDER and d in DOMAINS]
    draw_order = lattice + [r for r in ORDER if r in dv]   # lattice first, main boxes on top
    for k, d in enumerate(draw_order):
        n, w, s_, e = DOMAINS[d]
        main = d in ORDER
        v = vals[d]
        ax.add_patch(Rectangle((w, s_), e - w, n - s_, facecolor=cmap(norm(v)),
                               edgecolor="black" if main else "0.25",
                               linewidth=1.0 if main else 0.4, zorder=3 if main else 2))
        vis = inside(d)
        for later in draw_order[k + 1:]:
            vis &= ~inside(later)
        cx, cy, rad = _text_anchor(vis, glon, glat)
        if rad < 3.0:
            PROBLEMS.append(f"fig4: little visible area for {d} (anisotropic clearance {rad:.1f} deg)")
        tc = "white" if norm(v) < 0.55 else "black"
        pw = per_window.get(d, [])
        rng = f"{min(pw):.2f}–{max(pw):.2f}" if pw else "—"
        z = 5
        if main:
            ax.text(w + 1.0, n - 1.0, SHORT[d], fontsize=6, ha="left", va="top",
                    color=tc, zorder=z, fontweight="bold")
            ax.text(cx, cy + 0.5, f"{v:.2f}", fontsize=7, ha="center", va="center", color=tc, zorder=z)
            ax.text(cx, cy - 4.0, rng, fontsize=5.5, ha="center", va="center", color=tc, zorder=z)
        else:
            ax.text(cx, cy + 1.8, f"{v:.2f}", fontsize=7, ha="center", va="center", color=tc, zorder=z)
            ax.text(cx, cy - 2.7, rng, fontsize=5.5, ha="center", va="center", color=tc, zorder=z)
    ax.set_xlim(-180, 180); ax.set_ylim(LAT_MIN, LAT_MAX)
    ax.set_aspect(MAP_ASPECT)
    xt = np.arange(-180, 181, 60)
    ax.set_xticks(xt)
    ax.set_xticklabels([("%d° з.д." % -x) if -180 < x < 0 else ("%d° в.д." % x) if 0 < x < 180
                        else ("0°" if x == 0 else "180°") for x in xt], fontsize=7)
    yt = np.arange(-60, 61, 20)
    ax.set_yticks(yt)
    ax.set_yticklabels([("%d° ю.ш." % -y) if y < 0 else ("%d° с.ш." % y) if y > 0 else "0°"
                        for y in yt], fontsize=7)
    ax.tick_params(length=2, pad=2)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("сопряженность уровней, ступени 4–5", fontsize=7.5)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(0.6)
    REPORT["fig4_vmin_vmax"] = (vmin, vmax)
    REPORT["fig4_height_cm"] = round(H, 2)
    return save(fig, "fig4_map_v8.png")


def figA1(deg: dict) -> Path:
    steps = np.arange(1, 6)
    fig, axes = plt.subplots(3, 4, figsize=(WIDTH_CM * CM, 14.5 * CM), sharex=True, sharey=True)
    for ax, r in zip(axes.flat, ORDER):
        ax.axvspan(0.5, 1.5, color="0.88", lw=0, zorder=0)
        for w in deg[r]:
            ax.plot(steps, w["P"], color="0.6", lw=0.5, ls="-" if w["season"] == "JFM" else "--",
                    zorder=1)
        for pi, (lab, rng, col, mk, ls) in enumerate(PERIODS):
            recs = [w["P"] for w in deg[r] if w["period"] == pi]
            if not recs:
                continue
            ax.plot(steps, np.median(np.array(recs), axis=0), color=col, lw=1.3, ls=ls,
                    marker=mk, ms=3, mfc="white" if mk != "o" else col, mew=0.7, zorder=3)
        ax.set_title(f"{SHORT[r]}  {NAMES[r]}", fontsize=6.8, pad=3, loc="left")
        ax.grid(alpha=0.3, lw=0.4)
        ax.set_xticks(steps)
        ax.set_xlim(0.5, 5.5)
    for ax in axes[-1]:
        ax.set_xticklabels(STEP_LABELS_1LINE, fontsize=6, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_xlabel("пары соседних полос, км", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("сопряженность P", fontsize=7.5)
    handles = [Line2D([], [], color=c, ls=ls, marker=mk, ms=3, lw=1.3,
                      mfc="white" if mk != "o" else c, label=f"медиана, период {lab}")
               for lab, _, c, mk, ls in PERIODS]
    handles += [Line2D([], [], color="0.6", lw=0.6, ls="-", label="окно ЯФМ"),
                Line2D([], [], color="0.6", lw=0.6, ls="--", label="окно ИАС"),
                Patch(facecolor="0.88", label="ступень 1 (диагностическая)")]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7,
               bbox_to_anchor=(0.5, 0.0), handlelength=2.6, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0.075, 1, 1), h_pad=0.8, w_pad=0.6)
    return save(fig, "figA1_profiles_full_v8.png")


def figA2(deg: dict, km: dict, km_summary: dict) -> Path:
    steps = np.arange(1, 6)
    rho = km_summary["H18c_geography"]["per_region_rho"]
    REPORT["figA2_pooled_rho"] = km_summary["H18c_geography"].get("pooled_rho")
    fig, axes = plt.subplots(3, 4, figsize=(WIDTH_CM * CM, 11.0 * CM), sharex=True, sharey=True)
    n_pairs = 0
    for ax, r in zip(axes.flat, ORDER):
        ax.axvspan(0.5, 1.5, color="0.88", lw=0, zorder=0)
        d_arr = np.array([w["P"] - w["Ps"] for w in deg[r] if w["wnum"] in PRIMARY])
        k_arr = np.array([w["Pa"] for w in km[r] if w["wnum"] in PRIMARY])
        if len(d_arr) != 4 or len(k_arr) != 4:
            PROBLEMS.append(f"figA2: {r} has {len(d_arr)} degree / {len(k_arr)} equal-km primary windows")
        n_pairs += min(len(d_arr), len(k_arr))
        for arr, col, ls, mk, lab in ((d_arr, "#0072B2", "-", "o", "градусные области"),
                                      (k_arr, "#D55E00", "--", "s", "равновеликие области")):
            ax.fill_between(steps, arr.min(0), arr.max(0), color=col, alpha=0.18, lw=0, zorder=1)
            ax.plot(steps, arr.mean(0), color=col, ls=ls, marker=mk, ms=2.8, lw=1.0,
                    mfc="white" if mk != "o" else col, mew=0.7, label=lab, zorder=3)
        ax.set_title(f"{SHORT[r]}   ρ = {rho[r]:.2f}", fontsize=7.5, pad=3, loc="left")
        ax.grid(alpha=0.3, lw=0.4)
        ax.set_xticks(steps)
        ax.set_xlim(0.5, 5.5)
    for ax in axes[-1]:
        ax.set_xticklabels(STEP_LABELS_1LINE, fontsize=6, rotation=45, ha="right", rotation_mode="anchor")
        ax.set_xlabel("пары соседних полос, км", fontsize=7)
    for ax in axes[:, 0]:
        ax.set_ylabel("P − P$_{surr}$", fontsize=7.5)
    h, l = axes[0, 0].get_legend_handles_labels()
    h.append(Patch(facecolor="0.88", label="ступень 1 (диагностическая)")); l.append(h[-1].get_label())
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.955), h_pad=0.8, w_pad=0.6)
    REPORT["figA2_n_region_windows"] = n_pairs
    return save(fig, "figA2_geometry_v8.png")


# --------------------------------------------------------------------------
# captions
# --------------------------------------------------------------------------

CAPTIONS = """\
Черновые подписи к рисункам по данным (v8). Номера рисунков условные; цифры в тексте
подписей приводятся только там, где они читаются непосредственно с рисунка.

Рис. 2. Профили сопряженности соседних октавных уровней поля относительного вихря
на 850 гПа по 12 основным областям (ERA5, ступени 2–5; ступень 1 исключена как
диагностическая). (а) Исходный профиль P: медиана по восьми сезонным окнам каждой
области (периоды I и III для R1–R4, II и III для R5–R12). (б) Профиль с суррогатной
поправкой (превышение над медианой ансамбля фазовых суррогатов) на равновеликих
областях, медиана по четырем основным окнам 2023–2024 гг. Цвет и тип линии отвечают
циркуляционному режиму, маркер различает области внутри режима; коды областей
указаны у правого конца линий. По оси абсцисс — пары соседних полос, км.

Рис. 3. Сопоставление сопряженности уровней (среднее по разрешенным ступеням 4–5)
в реанализе ERA5 (ось абсцисс) и в свободном расчете модели ECMWF-IFS-HR без
усвоения атмосферных наблюдений (эксперимент highresSST-present проекта HighResMIP,
сезонные окна 2014 г.; ось ординат) по 39 областям. Закрашенные маркеры с кодами —
12 основных областей, полые серые кружки — 27 контрольных областей регулярной
решетки; штриховая линия — тождество. Значения ERA5 — медианы по сезонным окнам
2023–2024 гг. (для R5–R12 также 2021–2022 гг.).

Рис. 4. География сопряженности уровней (ERA5, среднее ступеней 4–5) по 39 областям:
12 основных областей (жирная рамка, код области) и 27 контрольных областей регулярной
решетки 20° × 40° (тонкая рамка). Заливка прямоугольника — значение для области
(шкала внизу); в каждом прямоугольнике приведено значение и под ним размах
(минимум–максимум) по отдельным сезонным окнам. Значения отнесены к областям целиком
и не интерполированы. Береговая линия — по маске суша–море ERA5.

Рис. A1 (приложение). Полные профили сопряженности P (ступени 1–5) по 12 основным
областям, 96 регион-окон ERA5. Тонкие серые линии — отдельные сезонные окна (сплошные —
январь–март, штриховые — июль–сентябрь); жирные линии с маркерами — медианы по
периодам I (2017–2019 гг.), II (2021–2022 гг.) и III (2023–2024 гг.); у каждой области
представлены два из трех периодов. Серая полоса выделяет ступень 1 (пара полос
мельче 50 и 50–100 км), которая приводится только как диагностическая из-за
зависимости от геометрии градусной сетки.

Рис. A2 (приложение). Профили с суррогатной поправкой (P − P_surr) на градусных
(сплошная линия, круглые маркеры) и равновеликих (штриховая линия, квадратные маркеры)
областях по 12 основным областям, основные окна 2023–2024 гг. (W9–W12). Линия — среднее
по четырем окнам, полоса — размах минимум–максимум. В заголовке панели — ранговая
корреляция ρ профилей двух нарезок для данной области (по ступеням). Серая полоса —
диагностическая ступень 1.

Рис. A3 (приложение). Сопоставление ERA5 (оси абсцисс) и свободного расчета
ECMWF-IFS-HR (оси ординат) по 39 областям для трех сопутствующих величин: (а) индекс
невзаимности A; (б) невосстановимость размещения u; (в) автокорреляция огибающей
разрешенных полос на лаге 6 ч. Обозначения как на рис. 3; штриховые линии — тождество.
"""


def write_captions() -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "captions_draft_data_figures.txt"
    path.write_text(CAPTIONS, encoding="utf-8")
    return path


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

STAGES = ["fig2", "fig3", "fig4", "figA1", "figA2", "figA3", "captions"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="all", help="comma-separated subset of: " + ",".join(STAGES))
    args = ap.parse_args(argv)
    stages = STAGES if args.only == "all" else [s.strip() for s in args.only.split(",") if s.strip()]
    bad = [s for s in stages if s not in STAGES]
    if bad:
        ap.error(f"unknown stage(s): {bad}; choose from {STAGES}")

    deg = km = km_summary = dv = per_window = None
    if any(s in stages for s in ("fig2", "figA1", "figA2")):
        deg = load_degree_profiles()
    if any(s in stages for s in ("fig2", "figA2")):
        km, km_summary = load_equal_km()
    if any(s in stages for s in ("fig3", "fig4", "figA3")):
        dv, per_window = load_free_running()

    written = []
    if "fig2" in stages:
        written.append(fig2(deg, km))
    if "fig3" in stages:
        written.append(fig3(dv))
    if "fig4" in stages:
        written.append(fig4(dv, per_window))
    if "figA1" in stages:
        written.append(figA1(deg))
    if "figA2" in stages:
        written.append(figA2(deg, km, km_summary))
    if "figA3" in stages:
        written.append(figA3(dv))
    if "captions" in stages:
        written.append(write_captions())

    print("\n=== written files ===")
    try:
        from PIL import Image
    except Exception:  # pragma: no cover
        Image = None
    for p in written:
        if p.suffix == ".png" and Image is not None:
            with Image.open(p) as im:
                w, h = im.size
                dpi = im.info.get("dpi", (DPI, DPI))
            print(f"{p.relative_to(ROOT)}: {w}x{h} px, dpi={dpi[0]:.0f}, "
                  f"{w / dpi[0] * 2.54:.2f} x {h / dpi[1] * 2.54:.2f} cm")
        else:
            print(f"{p.relative_to(ROOT)}")
    print("\n=== report ===")
    for k, v in REPORT.items():
        if k == "windows_per_domain":
            counts = {}
            for d, n in v.items():
                counts.setdefault(n, []).append(d)
            print(f"{k}: " + "; ".join(f"{n} windows: {len(ds)} domains ({', '.join(SHORT.get(x, x) for x in ds) if len(ds) <= 12 else 'lattice'})"
                                        for n, ds in sorted(counts.items())))
        else:
            print(f"{k}: {v}")
    print("\n=== data problems ===")
    print("\n".join(PROBLEMS) if PROBLEMS else "none")
    return 0


if __name__ == "__main__":
    sys.exit(main())
