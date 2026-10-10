Пакет подачи в «Известия РАН. Серия географическая» — статья 1 серии.
Раздел: Природные процессы и динамика геосистем.

Файлы:
  article1_v7.docx            — «файл статьи» (шаг 2 на сайте): УДК, заголовок, автор, аннотация,
                                текст, финансирование, Список литературы, References, таблицы 1–3
                                (каждая с новой страницы), рисунки 1–5 с подписями, английский блок.
  article1_v7.pdf             — тот же материал в PDF (для вычитки и как сводный PDF при подаче).
  figures/Fig1…Fig5 (.png и .tif) — рисунки отдельными файлами по требованию редакции: каждый
                                не менее 300 dpi (320–600) и не более 25 × 17,5 см (ширина 17,5 см,
                                высота 4,7–11,0 см); нумерация как в статье, подписи на русском.
  figures_report.txt          — проверка каждого рисунка: пиксели, dpi, размер в см.
  figure_captions.txt         — подрисуночные подписи.
  article1_figures_300dpi.zip — всё перечисленное одним архивом для отправки в редакцию.
  article1_author_info.docx   — сведения об авторе (заполнить поля [ЗАПОЛНИТЬ]).
  article1_cover_letter.docx  — сопроводительное письмо (проставить дату).

Перед подачей заполнить: сведения об авторе (адрес, телефон, степень), дату письма. Авторский договор — по форме издательства с сайта журнала.

Источник: manuscript_v2/article1/article1_v7_src.tex → article1/tools/build_article1_journal.py → article1_v7.tex/.docx/.pdf.
Рисунки: article1/figures/*.png → article1/tools/build_article1_figures.py → figures/, zip.


=== Доработка по рецензии (v8, октябрь 2026) ===
  ../article1_v8.docx / .pdf         — исправленная рукопись (определение V12, новая табл. 2, 5 рисунков).
  ../article1_v8_supplement.pdf      — дополнительные материалы (рис. П1–П3, полная табл. отвергнутых
                                        интерпретаций, чувствительность к сферическому ротору и маске, компоненты дисперсии).
  figures_v8/Fig1…Fig5 (.png, .tif)  — рисунки v8 отдельными файлами (17.5 см, 400 dpi); отчёт — figures_report_v8.txt;
                                        подписи — figure_captions_v8.txt; архив — article1_v8_figures_300dpi.zip.
  response_to_reviewer_v8.md / .docx — ответ рецензенту по пунктам.
  Перед подачей: в сноске рукописи заменить [COMMIT] и [ZENODO-V6.5] на коммит и версионный DOI релиза v6.5;
  проставить дату в ответе рецензенту.
  Источник: ../article1_v8_src.tex → ARTICLE1_VERSION=v8 python manuscript_v2/article1/tools/build_article1_journal.py;
  рисунки: clean_experiments/visualize_article1_v8_method.py, visualize_article1_v8_figures.py →
  ARTICLE1_VERSION=v8 python manuscript_v2/article1/tools/build_article1_figures.py.
