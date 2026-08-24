# Воспроизведение статьи 2 серии

**«Глобальная география сопряжённости иерархических уровней атмосферной
циркуляции: атрибуция и планетарная карта»**
(рукопись `manuscript_v2/article2_v1.tex`).

Маршрут от первичных данных до каждой таблицы и рисунка статьи. Для статьи 2
нужны **только** перечисленные ниже файлы репозитория. Оркестратор:
[`reproduce_article2.py`](reproduce_article2.py)
(этапы `download` → `compute` → `figures` → `check` одним запуском).
Маршрут статьи 1 (валидация самой величины) — [`article1_README.md`](article1_README.md).

## 1. Версия

- Тег: `v6.2` (первый тег с маршрутом статьи 2; численные результаты
  идентичны `v6.0`).
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

Первичные массивы в репозиторий не включаются; включены точные скрипты их
получения и все производные таблицы.

## 3. Маршрут «результат → данные → загрузчик → расчёт → артефакт»

Все расчёты — в `clean_experiments/`; многостадийные скрипты возобновляемы
(шардовые кэши), последовательность стадий закодирована в оркестраторе.

| Результат статьи | Данные | Загрузчик | Расчёт | Артефакт |
|---|---|---|---|---|
| Табл. 1 (часть A: 144 тайла, факторы CAPE/вихревая энергия) и рис. 2–3 | ERA5 `data/b3`, `data/b2b`; ковариаты `data/b4cov` | `download_b3_era5_wind.py`, `download_b2b_era5_wind.py`, `download_b4_covariates.py` | `experiment_B20_p_geography.py` (стадии `tiles` → `tests`) | `results/experiment_B20_p_geography/summary.json` |
| Табл. 2–3 (часть B: глобальная карта, ярусы, k80, LAI-контроль) и рис. 1, 4 | ERA5 глобально `data/b20global`; `data/b4cov`; CAPE `data/b21` | `download_b20_era5_global.py`, `download_b4_covariates.py`, `download_b21_cape_indices.py` | `experiment_B20_armB_global_map.py` (стадии `tiles`×2 сезона → `tests`) | `results/experiment_B20_armB_global_map/summary.json` |
| Потолок надёжности R²=0,914; прирост интермиттентности +0,207 | `data/b20global` + тайлы части B | те же | `experiment_A2b_splithalf.py` | `results/experiment_A2b_splithalf/summary_halves.json` |
| Воспроизводимый неспектральный остаток карты | шарды A2b + ковариаты | — | `experiment_A4_residual_structure.py` | `results/experiment_A4_residual/summary.json` |
| Табл. 4 и рис. 6 (тропический избыток, IPO-TPI, исключение ЭНСО) | суточные `data/b17daily`; CAPE+индексы `data/b21` | `download_b17_era5_daily.py`, `download_b21_cape_indices.py` | `experiment_B21_qt_tests.py` (стадии `p5-series` → `p1-series` → `tests`) | `results/experiment_B21_qt_tests/summary.json` |
| Модельная проверка карты (ρ=0,869/0,897) и тренд ERA5, рис. 5 | HighResMIP `data/b13hrmip`; `data/b20global`; `data/b17daily`; `data/b21` | `download_b13_highresmip.py` и выше | `experiment_B22_qt_round2.py` | `results/experiment_B22_qt_round2/summary.json` |
| Табл. 5 (статичность: переход 1998/99, ранги чувствительности; модельная проверка тренда) | `data/b17daily`, `data/b21`, переходный расчёт `data/b23hrmip` | `download_b23_hrmip_transient.py` и выше | `experiment_B23_drift_stratigraphy.py` (стадии `esyn-series` → `tests-ondisk` → `model-series` → `tests-model`) | `results/experiment_B23_drift_stratigraphy/summary_ondisk.json`, `summary_model.json` |
| Согласованность с региональным уровнем (ρ=0,874) | — | — | входит в `tests` части A | там же |

Рисунки рукописи: `visualize_B20_p_geography.py` и
`visualize_B20_armB_global.py` строят карты частей A и B; этап `figures`
оркестратора обновляет копии в `manuscript_v2/figures/` (включая вырезку
панели IPO из рисунка B21).

Зафиксированные до вычислений протоколы: `docs/PROTOCOL_PHASE20_P_GEOGRAPHY.md`,
`PHASE21`, `PHASE22`, `PHASE23`, `AUDIT2B_SPLITHALF_DECONTAMINATION`,
`AUDIT4_RESIDUAL_STRUCTURE`; сводный журнал — `docs/RECONCILIATION.md`.

## 4. Окружение

Общее для всех статей серии: [`requirements.txt`](requirements.txt)
(Python ≥ 3.11; расчёты выполнены на 3.13; для этапа `figures` дополнительно
Pillow):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r reproducibility/requirements.txt
```

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
