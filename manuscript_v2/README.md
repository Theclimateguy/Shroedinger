# manuscript_v2 — рукописи серии

Три статьи серии разнесены по папкам (с 2026-09-15). Общий мастер-файл
библиографии — `references_v2.bib` (в корне); статьи 1–3 подключают его как
`../references_v2.bib`. Метаданные записи Zenodo — `zenodo_record_metadata.md`.

Прежние редакции статей (v4–v6 статьи 1, `early/`, v1–v3 статьи 2, v1 статьи 3,
пакеты Overleaf) удалены из ветки 30.09.2026; они остаются в истории git и в тегах
до `v6.4`.

**Внимание (30.09.2026).** После выпуска рукописей найден артефакт оценщика: широтный
ход карты сопряженности на ячейках 700 км создается сеткой «широта–долгота» и
суррогатами, а не атмосферой. Статьи 2 и 3 затронуты существенно (зональная часть
карты, факторы, сезонная устойчивость, согласие с моделью); статья 1 — только в
первой паре полос (<50 | 50–100 км). Подробности: `docs/RECONCILIATION.md`
(дополнение от 30.09.2026), `docs/EXPLORATION_PHYSICS_2026-09-29.md`.

## article1/ — статья 1 (ПОДАНА в «Известия РАН. Серия географическая», 2026-08-23)
«Сопряжённость иерархических уровней атмосферной циркуляции как региональный инвариант».
- `article1_v7_src.tex` — исходник с `\cite{}`; `tools/build_article1_journal.py`
  собирает из него журнальный формат: `article1_v7.tex` / `.docx` / `.pdf`
  (author-year в тексте, ГОСТ-список + References; запуск из корня репозитория:
  `python manuscript_v2/article1/tools/build_article1_journal.py`).
- `references_article1.bib` — библиография статьи 1 (подмножество мастера).
- `figures/` — fig1…fig5 (журнальная нумерация) и figP_* (рисунки v4/v5).
- `journal_package/` — пакет подачи (docx статьи, сведения об авторе, письмо, Fig1–5).

## article2/ — статья 2 (в работе; целевой журнал тот же)
«Географическая дифференциация сопряжённости иерархических уровней атмосферной
циркуляции: атрибуция и планетарная карта».
- `article2_v4_src.tex` — ТЕКУЩАЯ версия в формате журнала (2026-09-21: явная цель — карта районирования,
  упрощённый текст, русские надписи на рисунках) (правила 17.11.2025, `docs/journal/`):
  исходник с `\cite{}`; `tools/build_article2_journal.py` (обёртка над инструментом статьи 1) собирает
  `article2_v4.tex` / `.pdf` / `.docx` (author-year, ГОСТ-список + References, проверка типографики,
  счётчик знаков; запуск из корня: `python manuscript_v2/article2/tools/build_article2_journal.py`).
  Весь маршрут статьи 2 — одним скриптом: `python reproducibility/reproduce_article2.py`
  (`--stage download|compute|figures|manuscript|check|all`; описание — `reproducibility/article2_README.md`).
- `references_article2.bib` — подмножество мастера (устарело; v3 берёт мастер); `figures/fig2_*` — 10 рисунков (в v3 используются 6)
  (обновляются этапом `figures` в `reproducibility/reproduce_article2.py`; три новых —
  `clean_experiments/describe_B20_geography.py`).

## article3/ — статья 3 (в работе; целевой журнал тот же)
«Иерархия атмосферной циркуляции: закон наследования между масштабными уровнями
и его значение для районирования» (2026-09-22, переписана вокруг закона структуры
Phases 27–30).
- `article3_v2_src.tex` — исходник с `\cite{}`; `tools/build_article3_journal.py` собирает
  `article3_v2.tex` / `.pdf` / `.docx` (запуск из корня:
  `python manuscript_v2/article3/tools/build_article3_journal.py`).
- `figures/ru_fig1_scheme_draw.png` (схема «на пальцах», черновик для автора),
  `ru_fig2_matrix`, `ru_fig3_decay`, `ru_fig4_tiles`, `ru_fig5_P_vs_h` —
  `clean_experiments/visualize_article3_ru.py`, `visualize_article3_scheme_ru.py`.

Сборка любой статьи: `cd manuscript_v2/articleN && latexmk -xelatex articleN_vK.tex`.
