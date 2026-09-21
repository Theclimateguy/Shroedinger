#!/usr/bin/env python3
"""Build the journal-format manuscript of article 2 (Izvestiya RAN, Ser. Geogr.).

Reuses the article-1 tool (manuscript_v2/article1/tools/build_article1_journal.py):
plain-text author-year citations, unnumbered GOST list + Latin References,
typography checks (no guillemets, no "yo", decimal points), character count
against the 45 000 limit, docx via pandoc, pdf via latexmk.

Input : manuscript_v2/article2/article2_v4_src.tex   (\\cite{key} placeholders)
        manuscript_v2/references_v2.bib              (shared master bibliography)
Output: manuscript_v2/article2/article2_v4.tex / .docx / .pdf

Usage: python manuscript_v2/article2/tools/build_article2_journal.py [--no-pdf] [--no-docx]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAN = HERE.parent                      # manuscript_v2/article2
ROOT = MAN.parent                      # manuscript_v2
sys.path.insert(0, str(ROOT / "article1" / "tools"))
import build_article1_journal as B  # noqa: E402

B.MAN = MAN
B.SRC = MAN / "article2_v4_src.tex"
B.BIB = ROOT / "references_v2.bib"
B.OUT_TEX = MAN / "article2_v4.tex"
B.OUT_DOCX = MAN / "article2_v4.docx"

B.JOURNAL_ABBR.update({
    "Nature Geoscience": ("Nat. Geosci.",) * 2,
    "Reviews of Geophysics": ("Rev. Geophys.",) * 2,
    "British Journal of Psychology": ("Br. J. Psychol.",) * 2,
    "Bulletin of the American Meteorological Society": ("Bull. Am. Meteorol. Soc.",) * 2,
    "Surveys in Geophysics": ("Surv. Geophys.",) * 2,
})
B.TITLE_OVERRIDE.update({  # journal rule: sentence case, proper nouns kept
    "HoskinsValdes1990": "On the existence of storm-tracks",
    "Chang2002": "Storm track dynamics",
    "Blackmon1976": "A climatological spectral study of the 500 mb geopotential height of the Northern Hemisphere",
    "HoskinsHodges2019": "The annual cycle of Northern Hemisphere storm tracks. Part I: Seasons",
    "Trenberth1997": "The definition of El Niño",
    "Newman2016": "The Pacific Decadal Oscillation, revisited",
    "Henley2015": "A tripole index for the Interdecadal Pacific Oscillation",
    "RiemannCampe2009": "Global climatology of convective available potential energy (CAPE) and convective inhibition (CIN) in ERA-40 reanalysis",
})
B.EN_TITLE.update({
    "Isachenko1991": "Landscape Science and Physical-Geographical Regionalisation",
})
B.PUBL_RU.update({"Высшая школа": "Высшая школа"})
B.PUBL_EN.update({"Высшая школа": "Vysshaya Shkola Publ."})

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
