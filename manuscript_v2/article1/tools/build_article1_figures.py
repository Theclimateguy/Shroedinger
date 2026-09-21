#!/usr/bin/env python3
"""Пакет иллюстраций статьи 1 по требованию редакции: каждый рисунок >= 300 dpi и
не больше 25 x 17,5 см. Все рисунки альбомные, поэтому ширина фиксируется на 17,5 см
(высота тогда < 17,5 см) -- предел соблюдён при любом прочтении ориентации 25 x 17,5.

Пиксели не пересчитываются: физический размер задаётся метаданными разрешения
(dpi = пиксели / 17,5 см), поэтому качество исходников сохраняется полностью.
Выход: journal_package/figures/Fig1..5.png + .tif (LZW), figures_report.txt и zip.
Запуск из корня репозитория: python manuscript_v2/article1/tools/build_article1_figures.py"""
from __future__ import annotations
import math
import zipfile
from pathlib import Path
from PIL import Image

ART = Path(__file__).resolve().parents[1]
SRC = ART / "figures"
PKG = ART / "journal_package"
OUT = PKG / "figures"
MAX_W_CM, MAX_H_CM, MIN_DPI = 17.5, 17.5, 300
# журнальная нумерация -> исходный файл
FIGS = {1: "fig1_profiles.png", 2: "fig3_geometry.png", 3: "fig4_free_running.png",
        4: "fig2_map.png", 5: "fig5_congo_amazon.png"}

OUT.mkdir(parents=True, exist_ok=True)
lines = ["Рисунок  пиксели      dpi   размер, см     проверка"]
for n, name in FIGS.items():
    im = Image.open(SRC / name)
    if im.mode != "RGB":  # прозрачность -> белый фон
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.convert("RGBA").split()[-1])
        im = bg
    w, h = im.size
    # PNG хранит целое число пикселей на метр -> округляем вверх, чтобы размер не превысил предел
    ppm = math.ceil(max(w / (MAX_W_CM / 100), h / (MAX_H_CM / 100)))
    dpi = ppm * 0.0254
    w_cm, h_cm = w / ppm * 100, h / ppm * 100
    ok = dpi >= MIN_DPI and w_cm <= MAX_W_CM and h_cm <= MAX_H_CM
    im.save(OUT / f"Fig{n}.png", dpi=(dpi, dpi), optimize=True)
    im.save(OUT / f"Fig{n}.tif", dpi=(dpi, dpi), compression="tiff_lzw")
    lines.append(f"Fig{n}     {w}x{h:<6} {dpi:6.1f}  {w_cm:5.2f} x {h_cm:5.2f}   {'OK' if ok else 'НАРУШЕНИЕ'}")
    assert ok, name

report = "\n".join(lines) + "\n"
(PKG / "figures_report.txt").write_text(report, encoding="utf-8")
print(report)

with zipfile.ZipFile(PKG / "article1_figures_300dpi.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in sorted(OUT.glob("Fig*.*")):
        z.write(f, f"figures/{f.name}")
    for extra in ("figures_report.txt", "figure_captions.txt"):
        if (PKG / extra).exists():
            z.write(PKG / extra, extra)
print("zip:", PKG / "article1_figures_300dpi.zip")
