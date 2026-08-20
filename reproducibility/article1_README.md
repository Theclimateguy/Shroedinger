# Воспроизведение статьи 1 серии

**«Сопряжённость иерархических уровней атмосферной циркуляции как региональный
инвариант»** (рукопись `manuscript_v2/article1_v5.tex`).

Этот файл — полный маршрут от первичных данных до каждого числа таблицы 2 и
каждого рисунка статьи. Репозиторий содержит много линий исследовательской
программы; для статьи 1 нужны **только** перечисленные ниже файлы.
Оркестратор: [`reproduce_article1.py`](reproduce_article1.py)
(этапы `download` → `compute` → `check` одним запуском).

## 1. Версия

- Тег: `v6.1` (первый тег, содержащий каталог `reproducibility/`;
  численные результаты идентичны `v6.0`, commit `4e22bd4`).
- Архив версий: Zenodo, концепт-DOI **10.5281/zenodo.19565769**
  (разрешается в последнюю версию).

## 2. Первичные источники данных

| Продукт | Переменные | Уровень | Шаг | Окна | Доступ |
|---|---|---|---|---|---|
| ERA5 (реанализ ECMWF) | u, v | 850 гПа | 6 ч | 12 областей × 8 сезонных окон 2017–2024 гг. + 27 контрольных областей (2024 г.) | Copernicus Climate Data Store, требуется бесплатная учётная запись и файл `~/.cdsapirc` |
| MERRA-2 (NASA GMAO) | U850, V850 | 850 гПа | 6 ч | 12 областей, окна W9–W12 (2023–2024 гг.) | NASA GES DISC (Earthdata), требуется учётная запись и `~/.netrc` |
| HighResMIP `highresSST-present`, модель ECMWF-IFS-HR (r1i1p1f1) | ua850, va850 | 850 гПа | 6 ч | глобальные поля, ЯФМ+ИАС 2014 г. | ESGF (открытый узел, учётная запись обычно не требуется) |
| NOAA GEFS (прикладная проверка «ошибка прогноза») | u, v 850 гПа, ансамбль | 850 гПа | 6 ч | окна 2023–2024 гг. | AWS Open Data, без учётной записи |
| WeatherBench 2 (прикладная проверка «потеря мезомасштаба») | прогнозы GraphCast/Pangu/GenCast/HRES + ERA5-эталон | 850 гПа | — | 2020 г. | Google Cloud, без учётной записи |

Первичные массивы в репозиторий **не включаются** (крупные внешние продукты со
своими лицензиями); включены точные скрипты их получения и все производные
таблицы.

## 3. Маршрут «результат → данные → загрузчик → расчёт → артефакт»

Все пути — от корня репозитория; расчёты — в `clean_experiments/`.

| Результат статьи (строка табл. 2 / рисунок) | Данные | Загрузчик | Расчёт | Артефакт |
|---|---|---|---|---|
| Профили и карта (рис. 1–2); строка 7 (разрешённые ступени) | ERA5: `data/b1`, `data/b2`, `data/b2b`, `data/b3` | `download_b1_era5_wind.py`, `download_b2_era5_wind.py`, `download_b2b_era5_wind.py`, `download_b3_era5_wind.py` | `experiment_B2_scale_irreversibility.py` | `results/experiment_B2_scale_irreversibility/summary.json` |
| Строка 1 (сигнатура на отложенных окнах, Δ=+1,12) и строка 3 (сверх спектра, Δ=+0,34) | ERA5 `data/b2b` | `download_b2b_era5_wind.py` | `experiment_B2b_heldout_invariants.py` | `results/experiment_B2b_heldout_invariants/summary.json` (H1, H2) |
| Строка 2 (сверх полуфрактальных, Δ=+0,65; сверх скаттеринг-статистик, Δ=+0,30) | ERA5 `data/b3` | `download_b3_era5_wind.py` | `experiment_B3_scattering_benchmark.py` | `results/experiment_B3_scattering_benchmark/e1_semifractal.json`, `summary.json` (H3a) |
| Строки 5–6 (контроль равновеликими областями; рис. 3) | ERA5 `data/b3`, `data/b2b` | те же | `experiment_B18_equal_km_regions.py` | `results/experiment_B18_equal_km_regions/summary.json` + `fig2_anchored_profiles.png` |
| Строка 4 (MERRA-2: Δ=+1,15; география ρ=0,93) | MERRA-2 `data/b8merra2` | `download_b8_merra2.py` | `experiment_B8_cross_reanalysis.py` | `results/experiment_B8_cross_reanalysis/summary.json` |
| Строка 8 и рис. 4 (свободный расчёт: ρ=0,71/0,93) | HighResMIP `data/b13hrmip` + ERA5-наборы выше | `download_b13_highresmip.py` | `experiment_B13_free_running.py` | `results/experiment_B13_free_running/summary.json` |
| Мера невосстановимости (форм. 5, ρ=−0,93); проверка содержания индекса A | синтетика + ERA5 выше | — | `experiment_B12_estimator_validity.py` | `results/experiment_B12_estimator_validity/summary.json` |
| Табл. 3: ошибка ансамблевого прогноза | GEFS `data/b9gefs` | `download_b9_gefs.py` | `experiment_B9_predictability.py` | `results/experiment_B9_predictability/summary.json` |
| Табл. 3: мезомасштабная ошибка анализа (27 контрольных областей) | ERA5 `data/b10era5` | `download_b10_era5.py` | `experiment_B10_irrecoverability.py` | `results/experiment_B10_irrecoverability/summary.json` |
| Табл. 3: потеря мезомасштаба в ML-моделях | WeatherBench 2 `data/b11wb2` | `download_b11_wb2.py` | `experiment_B11_learned_refinement.py` | `results/experiment_B11_learned_refinement/summary.json` |

Замороженные протоколы проверок: `docs/PROTOCOL_PHASE2_*.md`,
`PHASE2B`, `PHASE3`, `PHASE8`, `PHASE9`, `PHASE10`, `PHASE11`, `PHASE12`,
`PHASE13`, `PHASE18`; сводный журнал вердиктов и отступлений —
`docs/RECONCILIATION.md`.

## 4. Окружение

Python ≥ 3.11 (расчёты статьи выполнены на 3.13); закреплённые версии —
[`requirements_article1.txt`](requirements_article1.txt):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r reproducibility/requirements_article1.txt
```

## 5. Ожидаемые артефакты (SHA-256)

Производные таблицы включены в репозиторий; контрольные суммы версии v6.0/v6.1:

| Артефакт | SHA-256 |
|---|---|
| `results/experiment_B2b_heldout_invariants/summary.json` | `442b0060ee5ba2ebf2bf2971800e81a573b8ad201defcbfd34c413629645be0a` |
| `results/experiment_B3_scattering_benchmark/summary.json` | `7ef7aa42fdf053a8690c20759c584af41c371e06252a3e70d35e1d3821bea2d0` |
| `results/experiment_B3_scattering_benchmark/e1_semifractal.json` | `522185439f9597fe3280707da23f19d62a8d5767f689a21fbb33bf71ca0c1a60` |
| `results/experiment_B8_cross_reanalysis/summary.json` | `5d573deb165fb0613cd93f967df30f848e757b7178a1b32967122510a6a70a37` |
| `results/experiment_B12_estimator_validity/summary.json` | `4d5effc2adda01e9cd09ff864b36c6a3355fad6a1f8e061c1324fd8498094d65` |
| `results/experiment_B13_free_running/summary.json` | `6d71acf0db84669261e9fc0336689cb39229c67ff1861e79322e15ebf46f299c` |
| `results/experiment_B18_equal_km_regions/summary.json` | `f689e14f8713ac242c9525ef5223e7dc9629078837e111d7a2765038fd36c915` |
| `results/experiment_B9_predictability/summary.json` | `c74270d55879e7966ac55f9f2c48ae0f76d5d816eb5b68965c2d614e0aafe342` |
| `results/experiment_B10_irrecoverability/summary.json` | `cf5b15a4c5c0016e401a05e08c6cab53e3b8de43340d8479e5c73a76bb358caa` |
| `results/experiment_B11_learned_refinement/summary.json` | `5ae51f5634943596359175745ad338f60796ed005b964f9f69c7b432df22013c` |

Оговорка о битовой воспроизводимости: перестановочные и суррогатные ансамбли
используют фиксированные зёрна, поэтому статистики воспроизводятся точно на
той же версии зависимостей; при иных версиях NumPy/SciPy возможны расхождения
в последних знаках, не меняющие ни одного вердикта.

## 6. Ограничения доступа

- ERA5: требуется учётная запись CDS и принятие лицензии продукта
  (файл `~/.cdsapirc`).
- MERRA-2: требуется учётная запись NASA Earthdata (`~/.netrc`).
- HighResMIP: открытые узлы ESGF; при недоступности узла см. зеркала CEDA.
- Полный объём загрузки — сотни ГБ; поэтому предусмотрен проверочный маршрут
  (ниже), не требующий массовой загрузки.

## 7. Проверочный маршрут без многотерабайтной загрузки

Все производные таблицы (`clean_experiments/results/**`) включены в
репозиторий: статистические выводы статьи проверяются по ним без загрузки
первичных полей (см. `--stage check` оркестратора). Для проверки самого
вычислительного тракта достаточно одной области и одного окна:

```bash
# загрузчик возобновляем: его можно прервать, как только появится файл
# data/b3/era5_wind850_R7_CONGO__W9_2023JFM.nc (~2 ГБ)
python clean_experiments/download_b3_era5_wind.py

# расчёт одного регион-окна в отдельный каталог (не трогает сохранённые результаты)
python clean_experiments/experiment_B3_scattering_benchmark.py \
    --only R7_CONGO__W9_2023JFM --out-dir /tmp/article1_smoke
```

Полученный профиль (`P_real` в `/tmp/article1_smoke/R7_CONGO__W9_2023JFM.json`)
сравнивается с сохранённым
`results/experiment_B3_scattering_benchmark/R7_CONGO__W9_2023JFM.json`.
