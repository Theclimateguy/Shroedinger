#!/usr/bin/env python3
"""EXPLORATORY (not frozen) figures for docs/EXPLORATION_PHYSICS_2026-09-29.md."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.patches import Rectangle

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import explore_physics_analysis as A  # noqa: E402

XP = A.XP
FIG = XP / "figures"
SEASONS = ("JFM", "JAS")
C_OCEAN, C_LAND, C_MIX = "#1f6fb4", "#c8702a", "#8a8a8a"


def season_mean(fz, tag_fmt, key="P_anch"):
    parts = [A.load_xp(tag_fmt.format(s=s), fz) for s in SEASONS]
    if any(p is None for p in parts):
        return None
    return np.mean([p[key] for p in parts], axis=0)


def draw_map(ax, fz, values, title, cmap="viridis", vmin=None, vmax=None,
             coast=None):
    v = np.asarray(values, float)
    fin = v[np.isfinite(v)]
    vmin = float(np.quantile(fin, 0.02)) if vmin is None else vmin
    vmax = float(np.quantile(fin, 0.98)) if vmax is None else vmax
    cm = plt.get_cmap(cmap)
    for t, val in zip(fz.tiles, v):
        lo0 = ((t["lon0"] + 180.0) % 360.0) - 180.0
        w = t["dlon"]
        spans = ([(lo0, 180.0 - lo0), (-180.0, lo0 + w - 180.0)]
                 if lo0 + w > 180.0 else [(lo0, w)])
        fc = cm((val - vmin) / (vmax - vmin + 1e-12)) if np.isfinite(val) else "0.8"
        for x0, wid in spans:
            ax.add_patch(Rectangle((x0, t["lat0"]), wid, 6.0, facecolor=fc,
                                   edgecolor="none"))
    if coast is not None:
        ax.contour(coast[0], coast[1], coast[2], levels=[0.5], colors="k",
                   linewidths=0.4)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-62, 62)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([-120, -60, 0, 60, 120])
    ax.set_yticks([-60, -30, 0, 30, 60])
    ax.tick_params(labelsize=7)
    sm = plt.cm.ScalarMappable(cmap=cm, norm=plt.Normalize(vmin, vmax))
    plt.colorbar(sm, ax=ax, fraction=0.025, pad=0.01).ax.tick_params(labelsize=7)


def coastline():
    inv = xr.open_dataset(_HERE.parent / "data/b4cov/era5_invariants_global.nc")
    lsm = inv["lsm"].squeeze().values
    glat = inv["latitude"].values
    glon = inv["longitude"].values
    shift = np.argmin(np.abs(glon - 180.0))
    out = (np.sort(((glon + 180.0) % 360.0) - 180.0), glat,
           np.roll(lsm, -shift, axis=1))
    inv.close()
    return out


def fig_balance(fz):
    xs = [A.load_xp(f"era5850_vort_full_{s}", fz) for s in SEASONS]
    R = np.mean([A._rrot(x) for x in xs], axis=0)
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axs[0]
    mix = ~fz.is_ocean & ~fz.is_land
    for m, c, lab in ((fz.is_ocean, C_OCEAN, "океан"), (mix, C_MIX, "смешанные"),
                      (fz.is_land, C_LAND, "суша")):
        ax.scatter(R[m], fz.P[m], s=9, c=c, alpha=0.55, label=lab, linewidths=0)
    bins = np.quantile(R, np.linspace(0, 1, 11))
    for m, c in ((fz.is_ocean, C_OCEAN), (fz.is_land, C_LAND)):
        xm, ym = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            s = m & (R >= lo) & (R <= hi)
            if s.sum() > 3:
                xm.append(np.mean(R[s]))
                ym.append(np.mean(fz.P[s]))
        ax.plot(xm, ym, "-o", c=c, lw=2, ms=4, mec="w")
    ax.set_xlabel("вихревая доля мезомасштаба R_rot (50–400 км)")
    ax.set_ylabel("сопряжённость P")
    ax.set_title("а) P и вихревая доля, ERA5 850 гПа, 902 ячейки", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1]
    f = 2 * 7.292e-5 * np.abs(np.sin(np.deg2rad(fz.lat)))
    ax.scatter(f[fz.is_ocean] * 1e4, fz.P[fz.is_ocean], s=9, c=C_OCEAN, alpha=0.5,
               linewidths=0, label="океан")
    ax.scatter(f[fz.is_land] * 1e4, fz.P[fz.is_land], s=9, c=C_LAND, alpha=0.5,
               linewidths=0, label="суша")
    Pi = season_mean(fz, "era5850_vort_full_iso_{s}")
    if Pi is not None:
        rows = sorted(set(np.round(np.abs(fz.lat), 1)))
        ax.plot([2 * 7.292e-5 * np.sin(np.deg2rad(r)) * 1e4 for r in rows],
                [np.mean(Pi[fz.is_ocean & (np.round(np.abs(fz.lat), 1) == r)])
                 for r in rows], "-s", c="k", ms=4, lw=1.5,
                label="океан, изотропная сетка")
    ax.set_xlabel("параметр Кориолиса |f|, 10⁻⁴ с⁻¹")
    ax.set_ylabel("сопряжённость P")
    ax.set_title("б) P и параметр Кориолиса", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax = axs[2]
    t3 = json.loads((XP / "analysis_t3.json").read_text())
    names = {"cape_mean": "энергия\nнеустойчивости", "eke_syn": "вихревая\nэнергия",
             "land_frac": "доля суши", "abs_lat": "широта"}
    x = np.arange(len(names))
    raw = [t3["mediation"][k]["rho_raw"] for k in names]
    giv = [t3["mediation"][k]["rho_given_R_rot"] for k in names]
    ax.bar(x - 0.18, raw, 0.36, color="#555555", label="исходная связь с P")
    ax.bar(x + 0.18, giv, 0.36, color="#d08a2a", label="после учёта R_rot")
    ax.axhline(0, c="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(list(names.values()), fontsize=8)
    ax.set_ylabel("ранговая корреляция")
    ax.set_title("в) что остаётся от факторов карты", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_balance.png", dpi=170)
    plt.close(fig)


def fig_vertical(fz):
    t4 = json.loads((XP / "analysis_t4.json").read_text())
    levels = [L for L in (925, 850, 700, 600, 500, 250, 50) if str(L) in t4]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axs[0]
    cols = plt.get_cmap("plasma")(np.linspace(0.05, 0.9, 5))
    for i, k in enumerate(["0-12", "12-24", "24-36", "36-48", "48-60"]):
        ax.plot([t4[str(L)]["ocean_by_abs_lat"][k] for L in levels],
                range(len(levels)), "-o", c=cols[i], label=f"|φ| {k}°", ms=4)
    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels([f"{L} гПа" for L in levels])
    ax.set_xlabel("P над океаном")
    ax.set_title("а) широтный ход на разных уровнях (модель IFS-HR)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1]
    ax.plot([t4[str(L)]["rho_vs_850"] for L in levels], range(len(levels)),
            "-o", c="k", label="сходство с картой 850 гПа")
    ax.plot([t4[str(L)]["rho_P_Rrot"] for L in levels], range(len(levels)),
            "-s", c="#d08a2a", label="ρ(P, вихревая доля)")
    ax.plot([t4[str(L)]["zonal_share"] for L in levels], range(len(levels)),
            "-^", c=C_OCEAN, label="зональная доля дисперсии")
    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels([f"{L} гПа" for L in levels])
    ax.set_xlim(-0.2, 1.05)
    ax.axvline(0, c="k", lw=0.5)
    ax.set_title("б) структура карты по уровням", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax = axs[2]
    for b, c in (("tropics(<20)", "#b33"), ("subtropics(20-35)", "#d08a2a"),
                 ("midlat(>=35)", C_OCEAN)):
        xs_, ys_ = [], []
        for i, L in enumerate(levels):
            k = f"{b}|clean"
            if k in t4[str(L)]["contrast"]:
                xs_.append(t4[str(L)]["contrast"][k]["ocean_minus_land"])
                ys_.append(i)
        ax.plot(xs_, ys_, "-o", c=c, label=b, ms=4)
    ax.axvline(0, c="k", lw=0.5)
    ax.set_yticks(range(len(levels)))
    ax.set_yticklabels([f"{L} гПа" for L in levels])
    ax.set_xlabel("P океан − P суша (ячейки без рельефа выше 760 м)")
    ax.set_title("в) контраст суша–океан по уровням", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig2_vertical.png", dpi=170)
    plt.close(fig)


def fig_models(fz):
    t8 = json.loads((XP / "analysis_t8.json").read_text())
    t10 = json.loads((XP / "analysis_t10.json").read_text())
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axs[0]
    for model, c, lab in (("2d", "k", "двумерная турбулентность"),
                          ("sqg", "#d08a2a", "поверхностная квазигеострофическая")):
        rows = sorted([v for v in t8.values() if v["model"] == model],
                      key=lambda v: v["share"])
        if rows:
            ax.errorbar([v["share"] for v in rows], [v["P_anch"] for v in rows],
                        yerr=[2 * v["P_anch_se"] for v in rows], fmt="-o", c=c,
                        label=lab, ms=5, capsize=2)
    ax.axhspan(0.17, 0.41, color=C_OCEAN, alpha=0.10)
    ax.text(0.98, 0.40, "размах карты ERA5", ha="right", va="top", fontsize=8,
            color=C_OCEAN)
    ax.set_xlabel("доля независимых мелких источников в притоке")
    ax.set_ylabel("сопряжённость P (тот же оценщик)")
    ax.set_title("а) каскад против независимых источников", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1]
    a_km = np.array([200, 400, 800, 1600])
    cols = plt.get_cmap("viridis")(np.linspace(0.05, 0.9, 4))
    for i, beta in enumerate((0.0, 0.45, 1.0, 2.0)):
        k = f"beta{beta}_s20.5"
        if k in t10:
            g = [t10[k]["G_by_coarse_km"][str(a)] for a in a_km]
            ax.loglog(a_km, g, "-o", c=cols[i], ms=4,
                      label=f"модулятор k^-{beta:g}: показатель {t10[k]['G_exponent']}")
    ax.loglog(a_km, [0.25, 0.12, 0.04, 0.01], "--s", c="r", ms=5,
              label="ERA5, 12 областей (статья III)")
    ax.set_xlabel("размер грубой полосы, км")
    ax.set_ylabel("G(a)")
    ax.set_title("б) закон «только грубая полоса» без каскада", fontsize=10)
    ax.legend(frameon=False, fontsize=7.5)
    ax = axs[2]
    keys = [k for k in t10 if t10[k]["G_exponent"] is not None]
    ax.bar(np.arange(len(keys)) - 0.18, [t10[k]["R2_coarse_only"] for k in keys],
           0.36, color="#555", label="описание «по грубой полосе»")
    ax.bar(np.arange(len(keys)) + 0.18, [t10[k]["R2_separation"] for k in keys],
           0.36, color="#bbb", label="описание «по расстоянию между полосами»")
    ax.axhline(0.996, c="r", ls="--", lw=1)
    ax.axhline(0.27, c="r", ls=":", lw=1)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([k.replace("beta", "β=").replace("_s2", "\ns²=") for k in keys],
                       fontsize=7.5)
    ax.set_ylabel("R²")
    ax.set_title("в) синтетика и наблюдения (красные линии — ERA5)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_models.png", dpi=170)
    plt.close(fig)


def fig_objects(fz):
    t9 = json.loads((XP / "analysis_t9.json").read_text())
    regs = list(t9["by_region"])
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axs[0]
    x = np.arange(len(regs))
    ax.bar(x - 0.2, [t9["by_region"][r]["precipitation"] for r in regs], 0.4,
           color=C_OCEAN, label="с осадками")
    ax.bar(x + 0.2, [t9["by_region"][r]["moisture_front"] for r in regs], 0.4,
           color="#d08a2a", label="с влажностными фронтами")
    ax.axhline(0, c="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.split('_')[1]}\n{t9['by_region'][r]['P']:.2f}"
                        for r in regs], fontsize=7.5)
    ax.set_ylabel("согласованность размещения (ранг. корр.)")
    ax.set_title("а) с чем совмещена активность вихря (области по убыванию P)",
                 fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1]
    names = {"precipitation": "осадки", "moisture_front": "влажностные\nфронты",
             "vapour_transport": "поток влаги", "divergence_envelope": "огибающая\nдивергенции"}
    x = np.arange(len(names))
    ax.bar(x - 0.2, [t9["tests"][k]["rho_all_tiles"] for k in names], 0.4,
           color="#555", label="по 144 ячейкам")
    ax.bar(x + 0.2, [t9["tests"][k]["within_region_median"] for k in names], 0.4,
           color="#d08a2a", label="внутри областей (медиана)")
    for i, k in enumerate(names):
        ax.text(i + 0.2, t9["tests"][k]["within_region_median"] + 0.02,
                f"{t9['tests'][k]['within_region_n_positive']}/12",
                ha="center", fontsize=8)
    ax.axhline(0, c="k", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(list(names.values()), fontsize=8)
    ax.set_ylabel("связь согласованности с P")
    ax.set_title("б) где активность привязана к фронтам, P выше", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_objects.png", dpi=170)
    plt.close(fig)


def fig_maps(fz):
    coast = coastline()
    xs = [A.load_xp(f"era5850_vort_full_{s}", fz) for s in SEASONS]
    R = np.mean([A._rrot(x) for x in xs], axis=0)
    panels = [(fz.P, "P, ERA5 850 гПа (зафиксированная карта)", "viridis"),
              (R, "вихревая доля мезомасштаба R_rot", "viridis")]
    Pt = season_mean(fz, "era5850_vort_trans_{s}")
    if Pt is not None:
        Pf = np.mean([x["P_anch"] for x in xs], axis=0)
        panels.append((Pt - Pf, "изменение P после удаления стационарной и суточной частей", "RdBu_r"))
    Pd = season_mean(fz, "era5850_div_full_{s}")
    if Pd is not None:
        panels.append((Pd, "P поля дивергенции", "viridis"))
    for L in (500, 250):
        Pm = season_mean(fz, f"model{L}" + "_vort_full_{s}")
        if Pm is not None:
            panels.append((Pm, f"P, модель IFS-HR, {L} гПа", "viridis"))
    n = len(panels)
    fig, axs = plt.subplots((n + 1) // 2, 2, figsize=(15, 3.6 * ((n + 1) // 2)))
    axs = np.atleast_2d(axs)
    for ax, (v, title, cmap) in zip(axs.ravel(), panels):
        if cmap == "RdBu_r":
            lim = float(np.quantile(np.abs(v), 0.98))
            draw_map(ax, fz, v, title, cmap, -lim, lim, coast)
        else:
            draw_map(ax, fz, v, title, cmap, coast=coast)
    for ax in axs.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_maps.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fz = A.Frozen()
    for f in (fig_balance, fig_vertical, fig_models, fig_objects, fig_maps):
        try:
            f(fz)
            print("ok", f.__name__)
        except Exception as e:  # exploratory: report and continue
            print("FAILED", f.__name__, repr(e))


if __name__ == "__main__":
    main()
