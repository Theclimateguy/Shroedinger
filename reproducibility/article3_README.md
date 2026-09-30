# Воспроизведение статьи 3 серии

> **Известное ограничение (30.09.2026).** Форма закона воспроизводится любым общим
> модулятором активности (синтетическая проверка: R² 0,99 против 0,2) и не свидетельствует
> о каскаде; амплитуда закона на ячейках 700 км затронута артефактом сетки и суррогатов.
> Маршрут ниже воспроизводит числа статьи в исходном виде; подробности —
> `docs/RECONCILIATION.md`, `docs/EXPLORATION_PHYSICS_2026-09-29.md`.

**«Иерархия атмосферной циркуляции: закон наследования между масштабными
уровнями и его значение для районирования»**
(рукопись в формате журнала: `manuscript_v2/article3/article3_v2_src.tex`).

Единая точка входа — [`reproduce_article3.py`](reproduce_article3.py):

```bash
python reproducibility/reproduce_article3.py                  # всё подряд
python reproducibility/reproduce_article3.py --stage check    # только проверка артефактов
python reproducibility/reproduce_article3.py --stage figures  # рисунки 1–5 (без загрузок)
python reproducibility/reproduce_article3.py --stage compute --only B27
```

| Этап | Что делает |
|---|---|
| `download` | загружает первичные поля (4 загрузчика, возобновляемо) |
| `compute` | 9 шагов расчёта в порядке зависимостей: Phase 27 (единицы, тайлы, единицы ×0.7), Phase 28, Phase 29 (модель, тесты), Phase 30 |
| `figures` | рисунки 1–5 с русскими надписями (300 dpi) в `manuscript_v2/article3/figures/` |
| `manuscript` | собирает рукопись (`article3_v2.tex/.pdf/.docx`): ссылки «автор, год», список по ГОСТ + References, проверка типографики, счётчик знаков |
| `check` | проверяет наличие 6 ожидаемых артефактов и печатает SHA-256 |

Все производные таблицы включены в репозиторий, поэтому `check` и `figures`
работают без загрузки данных. Статья 3 использует как **входы** три
закоммиченных артефакта статей 1–2 (не пересчитываются здесь; маршруты —
[`article1_README.md`](article1_README.md), [`article2_README.md`](article2_README.md)):
`results/experiment_A3_robustness/units/*.json` (сопряжённость $P$ по 12
областям), `results/experiment_B20_armB_global_map/tiles_*.json` (тайловая $P$,
вихревая энергия), `results/experiment_A2_intermittency/tiles/*/*.json` (блок
перемежаемости), а также `results/experiment_B13_free_running/model_*.json`
(контроль C29-1).

## 1. Версия

- Тег: `v6.4` — первый тег с журнальной редакцией статьи 3 (v2), Phases 27–30 и
  единым скриптом воспроизведения; численные результаты прежних фаз идентичны `v6.0`.
- Архив версий: Zenodo, концепт-DOI **10.5281/zenodo.19565769**.

## 2. Первичные источники данных

| Продукт | Переменные | Покрытие | Доступ |
|---|---|---|---|
| ERA5 | u, v на 850 гПа, 6 ч, 0,25° | 12 областей, окна W9–W12 (2023–2024); глобально ЯФМ/ИАС 2023 | CDS, `~/.cdsapirc` |
| ERA5, месячные | ТПО, CAPE, LAI (ковариаты тайлов) | глобально 2021–2024 | CDS (в составе `download_b20_era5_global.py`) |
| ERA5, инварианты | орография, маска суши | глобально | CDS |
| HighResMIP `highresSST-present` (ECMWF-IFS-HR r1i1p1f1) | ua850, va850 | глобально, ЯФМ+ИАС 2014 | ESGF, без учётной записи |

Первичные массивы в репозиторий не включаются; включены точные скрипты их
получения и все производные таблицы.

## 3. Маршрут «результат → данные → загрузчик → расчёт → артефакт»

| Результат статьи | Данные | Загрузчик | Расчёт | Артефакт |
|---|---|---|---|---|
| Связь через ступень 0.35 (12/12 областей, p = 0.0002), профиль по разделению 0.27/0.35/0.25/0.16/0.09; суррогатные уровни 0.16/0.02/<0.01 против 0.64/0.41/0.21/0.10; тайлы 0.36 [0.35; 0.37]; индекс растяжения +0.07 и его карта (ρ = 0.14) | `data/b3`, `data/b20global`, `data/b4cov` | `download_b3_era5_wind.py`, `download_b20_era5_global.py`, `download_b4_covariates.py` | `experiment_B27_mechanism.py` (`units` → `tiles` ×2 сезона → `tests`) | `results/experiment_B27_mechanism/summary.json`, `report.md`; шарды `units/`, `tiles/` |
| Рис. 2 (матрица ковариаций), рис. 3а (G(a) по областям) | шарды B27 | — | `visualize_article3_ru.py` | `manuscript_v2/article3/figures/ru_fig2_matrix.png`, `ru_fig3_decay.png` |
| Закон формы: 99.6 % против 27 %, разность 0.62–0.83 (12/12, p < 10⁻⁴); показатель убывания ≈ −1.5; независимость от размера области (0.93–1.02; уровень ковариаций −20 %) | шарды B27 + единицы ×0.7 | те же | `experiment_B27_mechanism.py --stage units --extent 0.7`, затем `experiment_B28_inheritance_depth.py` | `results/experiment_B27_mechanism/units_x0.7/`, `results/experiment_B28_inheritance_depth/summary.json`, `report.md` |
| Закон на 902 ячейках: отношение разностей 0.052 [0.048; 0.057], 100 % ячеек, ρ с CAPE −0.005, с вихревой энергией 0.02 (рис. 4); модель IFS-HR: 0.62–0.78 (12/12), амплитуда ρ = 0.93 (рис. 3б, табл. 1); прибавка амплитуды над факторами и блоком перемежаемости +0.047 (p = 0.001), обратная +0.014, ρ(G, σ²) = 0.93; надёжность 0.79 против 0.87 | шарды B27; `data/b13hrmip`; тайлы B20B и A2 (статья 2) | `download_b13_highresmip.py` | `experiment_B29_structure_law.py` (`model` → `tests`) | `results/experiment_B29_structure_law/summary.json`, `report.md`, `tile_amplitude.npz`, `model/*.json` |
| Закон на соседних парах после поправки: r = 0.99, отношения 1.02/1.06/1.09; формула (2): ρ(P, h) = 0.63–0.64 (p = 0.001), по областям 0.80; надёжность h/P 0.71/0.72 (рис. 5) | шарды B27, B29 | — | `experiment_B30_unification.py` | `results/experiment_B30_unification/summary.json`, `report.md`, `tile_fields.npz` |
| Рис. 1 (схема), рис. 4, рис. 5 | результаты B29, B30 | — | `visualize_article3_scheme_ru.py`, `visualize_article3_ru.py` | `ru_fig1_scheme_draw.png`, `ru_fig4_tiles.png`, `ru_fig5_P_vs_h.png` |
| Табл. 1 ($G$ ERA5 и модель, $P$ по областям) | результаты B29 (`A_regions_era5`, `region_A_model`), A3 | — | `experiment_B29_structure_law.py` | `results/experiment_B29_structure_law/summary.json` |
| Файлы рукописи (tex, pdf, docx), объём в знаках | `article3_v2_src.tex`, `manuscript_v2/references_v2.bib` | — | `manuscript_v2/article3/tools/build_article3_journal.py` (этап `manuscript`) | `manuscript_v2/article3/article3_v2.*` |

Зафиксированные до вычислений протоколы: `docs/PROTOCOL_PHASE27_MECHANISM.md`,
`PHASE28_INHERITANCE_DEPTH`, `PHASE29_STRUCTURE_LAW` (с отклонением D1 —
исправленное определение контроля C29-1), `PHASE30_UNIFICATION`. Разведочные
расчёты, не входившие в протоколы (обрезка лестницы в Phase 28, форма профилей),
помечены в `report.md` соответствующей фазы и в статью как результаты не вошли.
Обзор литературы и состязательный поиск приоритета — `docs/LIT_REVIEW_ARTICLE3_MECHANISM.md`.

## 4. Окружение

Общее для всех статей серии: [`requirements.txt`](requirements.txt)
(Python ≥ 3.11; расчёты выполнены на 3.13). Внешние программы нужны только
этапу `manuscript`: `pandoc` (docx) и `latexmk` с XeLaTeX (pdf).

Время счёта (12 ядер, 25 ГБ): Phase 27 — единицы ≈ 45 мин, тайлы ≈ 35 мин на
сезон, единицы ×0.7 ≈ 30 мин; Phase 29 модель ≈ 10 мин; тесты фаз 28–30 —
минуты.

## 5. Ожидаемые артефакты (SHA-256)

| Артефакт | SHA-256 |
|---|---|
| `results/experiment_B27_mechanism/summary.json` | `b61b98a4e0858fdb9e9d42df64d9a791b67f9fa7695977fde92ef3e08751c10e` |
| `results/experiment_B28_inheritance_depth/summary.json` | `9660fdba3fd647f2958ceaa52a32526f1845a2847cfdecf09fe52cad346a3f0a` |
| `results/experiment_B29_structure_law/summary.json` | `e071c15a250f64b46d3ede6cfbef1ea0d00955f800a511f5353123eab92bc749` |
| `results/experiment_B30_unification/summary.json` | `5cf9ea9082f1bf8ef3db9dbf238ae1815fb434a04b53e375e446509cd667cd6e` |
| `results/experiment_B29_structure_law/tile_amplitude.npz` | `502177824bd89b4f8ece05a9c41ac81f4c8e4ab4b9ac72cde6b63281e8a53abd` |
| `results/experiment_B30_unification/tile_fields.npz` | `66c20e9f55f6f34aa9133b835438e94138361142f52021f0dead17a9fc2334a8` |

Оговорка о битовой воспроизводимости — как в маршрутах статей 1–2: суррогатные,
перестановочные и бутстрэп-ансамбли используют фиксированные зёрна
(`SEED_SUR = 20260811`, `SEED = 20260921/20260922`); при иных версиях NumPy/SciPy
хэши могут отличаться в последних знаках при неизменных выводах.
