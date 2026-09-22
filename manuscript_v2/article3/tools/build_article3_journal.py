#!/usr/bin/env python3
"""Build the journal-format manuscript of article 3 (Izvestiya RAN, Ser. Geogr.).

Reuses the article-1 tool (manuscript_v2/article1/tools/build_article1_journal.py):
plain-text author-year citations, unnumbered GOST list + Latin References,
typography checks, character count against the 45 000 limit, docx via pandoc,
pdf via latexmk.

Input : manuscript_v2/article3/article3_v2_src.tex   (\\cite{key} placeholders)
        manuscript_v2/references_v2.bib              (shared master bibliography)
Output: manuscript_v2/article3/article3_v2.tex / .docx / .pdf

Usage: python manuscript_v2/article3/tools/build_article3_journal.py [--no-pdf] [--no-docx]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAN = HERE.parent                      # manuscript_v2/article3
ROOT = MAN.parent                      # manuscript_v2
sys.path.insert(0, str(ROOT / "article1" / "tools"))
import build_article1_journal as B  # noqa: E402

B.MAN = MAN
B.SRC = MAN / "article3_v2_src.tex"
B.BIB = ROOT / "references_v2.bib"
B.OUT_TEX = MAN / "article3_v2.tex"
B.OUT_DOCX = MAN / "article3_v2.docx"

B.JOURNAL_ABBR.update({
    "Physical Review Letters": ("Phys. Rev. Lett.",) * 2,
    "Physical Review D": ("Phys. Rev. D",) * 2,
    "The European Physical Journal B": ("Eur. Phys. J. B",) * 2,
    "Physics of Fluids": ("Phys. Fluids",) * 2,
    "Econometrica": ("Econometrica",) * 2,
    "Canadian Journal of Remote Sensing": ("Can. J. Remote Sens.",) * 2,
    "PNAS Nexus": ("PNAS Nexus",) * 2,
})
B.TITLE_OVERRIDE.update({
    "HoskinsValdes1990": "On the existence of storm-tracks",
    "Chang2002": "Storm track dynamics",
    "Roux2000": "A wavelet-based method for multifractal image analysis. III. Applications to high-resolution satellite images of cloud structure",
    "CraigCohen2006": "Fluctuations in an equilibrium convective ensemble. Part I: Theoretical formulation",
})
B.EN_TITLE.update({
    "Isachenko1991": "Landscape Science and Physical-Geographical Regionalisation",
    "Khoroshev2016": "Multiscale Organisation of the Geographical Landscape",
    "Khoroshev2010": "Multiscale organisation of inter-component relations in the landscape",
    "Armand1988": "Self-Organisation and Self-Regulation of Geographical Systems",
    "Kolomyts1999": "Polymorphism of landscape-zonal systems",
    "Baibar2023": "Landscape invariants as order parameters of a dynamic system",
    "Krenke2019": "Spatial organisation of the regional mesoclimate",
})
B.CITY_RU.update({"Princeton": "Princeton"})
B.CITY_EN.update({"Princeton": "Princeton"})
B.PUBL_RU.update({"Товарищество научных изданий КМК": "Т-во науч. изд. КМК",
                  "Princeton University Press": "Princeton Univ. Press",
                  "Высшая школа": "Высшая школа"})
B.PUBL_EN.update({"Товарищество научных изданий КМК": "KMK Scientific Press",
                  "Princeton University Press": "Princeton Univ. Press",
                  "Наука": "Nauka Publ.", "Высшая школа": "Vysshaya Shkola Publ."})

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--no-docx", action="store_true")
    a = ap.parse_args()
    txt = B.build_tex()
    B.char_count(txt)
    if not a.no_pdf:
        B.build_pdf()
    if not a.no_docx:
        B.build_docx()
