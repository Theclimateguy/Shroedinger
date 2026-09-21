# Воспроизведение статьи 2 серии

**«География сопряженности иерархических уровней атмосферной циркуляции:
планетарная карта, ее факторы и районирование»**
(рукопись в формате журнала: `manuscript_v2/article2/article2_v4_src.tex`).

Единая точка входа — [`reproduce_article2.py`](reproduce_article2.py). Один
скрипт проходит весь путь от первичных данных до файлов рукописи:

```bash
python reproducibility/reproduce_article2.py                  # всё подряд
python reproducibility/reproduce_article2.py --stage check    # только проверка артефактов
python reproducibility/reproduce_article2.py --stage figures  # рисунки и описательный слой (без загрузок)
python reproducibility/reproduce_article2.py --stage compute --only B20
```

| Этап | Что делает |
|---|---|
| `download` | загружает первичные поля (9 загрузчиков, возобновляемо) |
| `compute` | 20 шагов расчёта в порядке зависимостей (см. разд. 3) |
| `figures` | описательный слой (профили, типология и схема районирования, карта необъясненной части, таблица ячеек, справочник названных мест) и рисунки 1, 3, 5 с русскими надписями; обновляет копии в `manuscript_v2/article2/figures/` |
| `manuscript` | собирает рукопись в формате журнала (`article2_v4.tex/.pdf/.docx`): ссылки «автор, год», список по ГОСТ + References, проверка типографики, счетчик знаков; отсутствие `pandoc`/`latexmk` не прерывает маршрут |
| `check` | проверяет наличие 16 ожидаемых артефактов и печатает SHA-256 |

Все производные таблицы включены в репозиторий, поэтому `check` и `figures`
работают без загрузки данных. Для статьи 2 нужны **только** перечисленные
ниже файлы репозитория. Маршрут статьи 1 — [`article1_README.md`](article1_README.md).

## 1. Версия

- Последний выпущенный тег — `v6.2`. Код, добавленный после него
  (Phase 25, описательный слой, рисунки с русскими надписями, сборщик рукописи,
  правила журнала в `docs/journal/`), войдет в следующий тег (`v6.3`,
  **еще не выпущен**); численные результаты прежних фаз не менялись.
- Архив версий: Zenodo, концепт-DOI **10.5281/zenodo.19565769**.

## 2. Первичные источники данных

| Продукт | Переменные | Покрытие | Доступ |
|---|---|---|---|
| ERA5 | u, v на 850 гПа, 6 ч, 0,25° | 12 областей (окна 2021–2024) + глобально ЯФМ/ИАС 2023 | CDS, учётная запись + `~/.cdsapirc` |
| ERA5, суточные 00Z | u, v на 850 гПа | 12 областей, 1979–2024 (длинная запись) | CDS |
| ERA5, инварианты | орография, маска суши, ТПО, LAI (lai_lv+lai_hv) | глобально | CDS |
| CAPE (ERA5, месячные) | CAPE | глобально, 1979–2024 | CDS |
| Климатические индексы | ONI, PDO, AMO, DMI, IPO-TPI | ряды 1979–2024 | NOAA PSL, без учётной записи |
| HighResMIP `highresSST-present` (ECMWF-IFS-HR r1i1p1f1) | ua850, va850 | глобально, ЯФМ+ИАС 2014 | ESGF |
| HighResMIP, переходный расчёт | ua850, va850 | эпохи 1979–1986 и 2007–2014 | ESGF |
| ERA5, месячные гидроклиматические | tp, cp, tcwv; u, v на 850/500 гПа | глобально 0,5°, 2021–2024 | CDS |

Первичные массивы в репозиторий не включаются; включены точные скрипты их
получения и все производные таблицы.

## 3. Маршрут «результат → данные → загрузчик → расчет → артефакт»

Все расчеты — в `clean_experiments/`; многостадийные скрипты возобновляемы,
последовательность стадий закодирована в оркестраторе. Нумерация таблиц и
рисунков — по редакции v4.

| Результат статьи | Данные | Загрузчик | Расчет | Артефакт |
|---|---|---|---|---|
| Рис. 1, 2, 4, 6; табл. 1–3; все числа описания карты (провинции, широтный ход, полушария, суша и океан, сезоны, типология, 12 областей, карта необъясненной части) | ячейки части B + половины AUDIT-2b | — | `describe_B20_geography.py` (описательно, без протокола) | `results/describe_B20_geography/summary.json` (в т.ч. `named_cells` — справочник «название места → ячейка»), `provinces.json` |
| Рис. 1, 3, 5 с русскими надписями (300 dpi, без пересчета) | результаты B20 и B22 | — | `visualize_article2_ru.py` | `manuscript_v2/article2/figures/ru_fig*.png` |
| Факторы карты: R² = 0.536, вклад одной вихревой энергии 0.456, свойства территории 0.482, корреляции рис. 3, наклон спектра 0.570, контроль (индекс листовой поверхности), сезонная устойчивость 0.016/0.048, согласие с частью A 0.840 | ERA5 глобально `data/b20global`; `data/b4cov`; CAPE `data/b21` | `download_b20_era5_global.py`, `download_b4_covariates.py`, `download_b21_cape_indices.py` | `experiment_B20_armB_global_map.py` (`tiles`×2 сезона → `tests`) | `results/experiment_B20_armB_global_map/summary.json` |
| 144 ячейки в 12 областях: R² = 0.404, связи внутри областей (10/12 и 11/12), согласие с областями 0.87, вычитание спектра | ERA5 `data/b3`, `data/b2b`; `data/b4cov` | `download_b3_era5_wind.py`, `download_b2b_era5_wind.py`, `download_b4_covariates.py` | `experiment_B20_p_geography.py` (`tiles` → `tests`) | `results/experiment_B20_p_geography/summary.json` |
| Статистики перемежаемости: R² = 0.761 (83 % воспроизводимой дисперсии) | `data/b20global` | те же | `experiment_A2_intermittency_headtohead.py` | `results/experiment_A2_intermittency/summary.json` |
| Предел воспроизводимости R² = 0.91; прирост +0.207 на независимых половинах | `data/b20global` | те же | `experiment_A2b_splithalf.py` | `results/experiment_A2b_splithalf/summary_halves.json` |
| Необъясненная часть карты воспроизводима (r = 0.64) и не связана со спектром | половины A2b + факторы | — | `experiment_A4_residual_structure.py` | `results/experiment_A4_residual/summary.json` |
| Дополнительные факторы необъясненной части: сдвиг ветра ρ = –0.38, прирост +0.024 при пороге +0.03 | `data/b25hydro` | `download_b25_hydroclim.py` | `experiment_B25_residual_candidates.py` | `results/experiment_B25_residual_candidates/summary.json` |
| Описательные величины текста: разложения дисперсии 51/49 и 51/10/39; «расстояние до гор ничего не добавляет» (ρ = –0.09, прирост –0.001) | таблицы ячеек частей A и B, `data/b4cov` | — | `verify_article2_descriptives.py` (D1–D4) | `results/verify_article2_descriptives/summary.json` |
| Модельный расчет: ρ = 0.869 при пределе 0.897 (рис. 5); многолетний тренд по реанализу | HighResMIP `data/b13hrmip`; `data/b20global`; `data/b17daily`; `data/b21` | `download_b13_highresmip.py` и выше | `experiment_B22_qt_round2.py` | `results/experiment_B22_qt_round2/summary.json` |
| Длинная запись 1979–2024: повышенная межгодовая изменчивость в трех очагах конвекции | суточные `data/b17daily` | `download_b17_era5_daily.py` | `experiment_B17_penv_long_record.py` (`series` → `tests`) | `results/experiment_B17_penv_long_record/summary.json` |
| Связь с Междекадным тихоокеанским колебанием (T = 0.453; 0.602; p = 0.001), исключение Эль-Ниньо | `data/b17daily`; индексы `data/b21` | `download_b21_cape_indices.py` | `experiment_B21_qt_tests.py` (`p5-series` → `p1-series` → `tests`) | `results/experiment_B21_qt_tests/summary.json` |
| Смена фазы 1998–1999 гг.: сопряженность не опережает обычные характеристики; модельная проверка тренда | `data/b17daily`, `data/b21`, `data/b23hrmip` | `download_b23_hrmip_transient.py` | `experiment_B23_drift_stratigraphy.py` (`esyn-series` → `tests-ondisk` → `model-series` → `tests-model`) | `summary_ondisk.json`, `summary_model.json` |
| Файлы рукописи (tex, pdf, docx), объем в знаках | `article2_v4_src.tex`, `manuscript_v2/references_v2.bib` | — | `manuscript_v2/article2/tools/build_article2_journal.py` (этап `manuscript`) | `manuscript_v2/article2/article2_v4.*` |

Зафиксированные до вычислений протоколы: `docs/PROTOCOL_PHASE20_P_GEOGRAPHY.md`,
`PHASE17`, `PHASE21`, `PHASE22`, `PHASE23`, `PHASE25`,
`AUDIT2_INTERMITTENCY_HEADTOHEAD`, `AUDIT2B_SPLITHALF_DECONTAMINATION`,
`AUDIT4_RESIDUAL_STRUCTURE`; сводный журнал — `docs/RECONCILIATION.md`.
Описательный слой (`describe_B20_geography.py`, `verify_article2_descriptives.py`)
протоколом не связан: решающих правил и порогов в нем нет. Правила журнала, по
которым оформлена рукопись, — `docs/journal/`.

## 4. Окружение

Общее для всех статей серии: [`requirements.txt`](requirements.txt)
(Python ≥ 3.11; расчеты выполнены на 3.13; Pillow — для рисунков,
python-docx — для этапа `manuscript`):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r reproducibility/requirements.txt
```

Внешние программы нужны только этапу `manuscript`: `pandoc` (docx) и
`latexmk` с XeLaTeX (pdf). Без них численный маршрут выполняется полностью.

## 5. Ожидаемые артефакты (SHA-256)

| Артефакт | SHA-256 |
|---|---|
| `results/experiment_B20_p_geography/summary.json` | `6140fff97389d7ece6774db3a53d96cedc4b277f2bce768392ab6ecc12a86351` |
| `results/experiment_B20_armB_global_map/summary.json` | `fb5295d7e4d5a4171914366ae9bef671bb2dbf5bd088eefc798c8c517b524550` |
| `results/experiment_A2b_splithalf/summary_halves.json` | `4e5b98247f6d731def635018d558e965369973b3dfa547f846fe4c5de4a35109` |
| `results/experiment_A4_residual/summary.json` | `f93fbe054ff9800952ab189e606c43882ff681f26484465195ccb32cb6f9b9b0` |
| `results/experiment_B21_qt_tests/summary.json` | `43a4c95324bf49e1847c74b80cedb7fcffa50b020cffbcd2a78db80847a03905` |
| `results/experiment_B22_qt_round2/summary.json` | `32b647041e6eb8390b3543f01917252fba5eabeac6f2e5e0c7b242d1acb317e2` |
| `results/experiment_B23_drift_stratigraphy/summary_ondisk.json` | `44070ceb895199dda888fe3ff9736ae4425b981721f9fe2b8c8f3d0c7e21a2f2` |
| `results/experiment_B23_drift_stratigraphy/summary_model.json` | `4b9efabe2d6b8d03187bf384e0a1f65ff08eac66800a82d733e9e5a7ba547d77` |
| `results/experiment_B20_armB_global_map/fig1_global_map.png` | `0e6bbb8f79bc66c5e2a7b0538ba692a5cecd4823e9acbacad1b81cab6e36af1f` |
| `results/experiment_B20_p_geography/fig1_tile_maps.png` | `017ce088393292da83ddac377e054653956cf8c5725085a0fc1f9f7827c984a9` |
| `results/experiment_B25_residual_candidates/summary.json` | `17fdebe9307d9da8f50a33b59e9496a011cb48a6c8e16cc4a97ebca39ccbbaed` |
| `results/verify_article2_descriptives/summary.json` | `f240641830d76fbf2656a4fe392ec293e9aa237a0c8f8ef94106d01329d530cb` |
| `results/experiment_B20_armB_global_map/fig3_two_channel.png` | `4f507a9cee69d5c8ab1f6a0b6b289688bd76157a99efe00e7338349e69b31d03` |
| `results/describe_B20_geography/summary.json` (описательный слой) | `30fdd96403224bf09066651a0113a94f7b00fc5a4e52974595bca12997d9f75b` |
| `results/describe_B20_geography/provinces.json` | `6dd89ac3b963d2e7e25107d686e967d7dde80276158e1c181ec4b31b57c55816` |
| `results/experiment_A2_intermittency/summary.json` | `1a068c068fe1e4d46e046a2071a80fa2928fb6bd2dded5c1e0c2eb8d0e301c9e` |
| `results/experiment_B17_penv_long_record/summary.json` | `adbdee1a2fd13bd5e35afa77c9ad49c3046c2431009e3587945e09e954f0cb69` |

Оговорка о битовой воспроизводимости — как в маршруте статьи 1: суррогатные
и перестановочные ансамбли используют фиксированные зёрна; при иных версиях
NumPy/SciPy возможны расхождения в последних знаках, не меняющие вердиктов.

## 6. Ограничения доступа и объёмы

- CDS (ERA5, CAPE): учётная запись + принятие лицензии; глобальный набор
  `data/b20global` — самый крупный (~десятки ГБ), суточная длинная запись
  `data/b17daily` сопоставима.
- ESGF (HighResMIP): открытые узлы, зеркала CEDA.
- NOAA PSL (индексы): свободно.

## 7. Проверочный маршрут без массовой загрузки

Все производные таблицы включены в репозиторий: статистические выводы
статьи проверяются по ним без загрузки первичных полей
(`--stage check`). Тракт тайловых вычислений проверяется на одном сезоне
части B с прерыванием загрузчика после первых месячных файлов: стадия
`tiles` возобновляема по шардам, а согласие пересчитанных шардов с
сохранёнными `tiles_JFM2023.json` — прямой тест конвейера. Полное
воспроизведение карты требует полного сезона.
