# Target-journal author rules (papers 1–2)

Journal: «Известия РАН. Серия географическая» (Izvestiya RAN, Seriya Geograficheskaya),
https://izvestia.igras.ru. The manuscripts `manuscript_v2/article1/article1_v7*`
and `manuscript_v2/article2/article2_v4*` follow the author rules dated **17.11.2025**.
The rule files are third-party documents and are **not redistributed** here; local
copies (`*.pdf`, `*.txt` in this folder) are git-ignored.

| Document | URL | SHA-256 of the copy used (retrieved 2026-09-15) |
|---|---|---|
| Правила для авторов (ред. 17.11.2025) | https://izvestia.igras.ru/jour/manager/files/Правиладляавторов17.11.2025.pdf | `da87925e32a2e207f56c515b005413ff60f1aaa3969384b149127743feb0fda9` |
| Правила оформления References | https://izvestia.igras.ru/jour/manager/files/ПравилаоформленияReferences.pdf | `3de864e3b01c0d1dd14d7ce604651a0d8f86a609823acf6d54b4a94972e4cd79` |

Requirements encoded in the build tools
(`manuscript_v2/article1/tools/build_article1_journal.py`,
`manuscript_v2/article2/tools/build_article2_journal.py`):

- volume 20–45 thousand characters with spaces (text + tables); at most 8 figures;
- abstract 200–250 words, no citations or non-standard abbreviations; about 10 keywords;
- sections: problem statement, data and methods, results (italic sub-headings),
  discussion, conclusion, funding;
- in-text citations (Author, year) / (Author et al., year), several sources sorted
  alphabetically and separated by semicolons;
- typography: “ ” quotes, no «ё», decimal point, en-dash minus;
- two unnumbered reference lists: traditional (GOST; Cyrillic first, then Latin) and
  Latin `References` (transliteration table of the journal, translated titles in
  brackets, `(In Russ.)`, abbreviated journal names, DOI as `https://doi.org/...`);
  English titles in sentence case;
- tables (one per page, title above) and figures (one per page, caption below, no
  title on the figure field) after `References`; English block at the end;
- Times New Roman 12 pt, 1.5 spacing (abstract 1.0), margins 2/2/3/1.5 cm;
  figures also as separate files, at least 300 dpi.
