# Prior art к записке «Формализм gen1 на новом знании» (docs/THEORY_NOTE_BUNDLE_REVISITED_2026-10-16.md, разделы 3, 4, 7)

Дата: 2026-10-16. Только обзор литературы (веб-поиск, чтение доступных текстов); ничего не считалось, код и записка не правились, ничего не коммитилось.

Легенда проверки:
- **[V]** — библиографическая запись сверена по Crossref (DOI, том, страницы, год).
- **[T]** — содержание утверждения подтверждено по тексту статьи (препринт/принятая рукопись), цитируются разделы/уравнения.
- **[A]** — подтверждено только по аннотации/сниппету, текст не читал.
- **[U]** — из памяти, не сверено.

Статусы в таблице: **известно** / **частично** / **не найдено**. «Не найдено» значит «веб-поиском не нашёл», а не «доказано отсутствие».

---

## 0. Главное за минуту

1. Нуль Кларка в бездивергентном 2D — **известен** и сформулирован почти слово в слово: Eyink 2006, JFM 549:191–214 (Sec. 2.1.1, аннотация), продолжение результата Kraichnan 1971. Также известно, что ненулевой поток требует относительного поворота главных осей деформации на разных масштабах (там же, уравнение (2.18) для D_t α^(k)).
2. Формула 2D «Π = −2 λ_τ λ_s cos 2Θ» (ваше −k c σ̄ cos 2Δθ) — **известна**: Liao & Ouellette 2014 (PoF 26:045103), 2015 (PRE 91:063004), Fang & Ouellette 2016 (PRL 117:104501, ур. (2)), Yu–Si–Fang 2025 (ур. (1)), De Serio 2026. Новая только запись с дивергентной добавкой −k δ̄ и язык «матрица плотности».
3. «Матрица плотности» ρ = τ/tr τ — это **нормированный тензор напряжений Рейнольдса** (тензор анизотропии Lumley–Newman 1977, барицентрическая карта Banerjee 2007). Математика стандартная; квантовый язык в гидродинамике я не нашёл.
4. «Кривизна сноса» (4.1)–(4.2): **ближайший источник — Fang & Ouellette 2016 (PRL 117:104501)**: они уже показали (лаб. 2D), что мгновенное выравнивание напряжения и деформации слабое, а лаговое вдоль лагранжевых траекторий сильнее, и объяснили это **дифференциальной адвекцией** напряжения и деформации. Критерий (4.2) по сути есть LKH с лагранжевой производной вдоль траектории более мелкой структуры (см. п. 3.4). Коммутатор [D_t^(ℓ), ∂_ℓ] — однострочная алгебра. Новизна возможна только в операциональной части (число сноса Sw на реанализе, проверка нуля), не в теории.
5. Огрубление Жермано на сфере для атмосферы **уже делают**: Sichani–Storer–Aluie, PRR 8:033239 (2026), ERA5, 3000→100 км; Christensen et al. (EGU26-10479), ECMWF, 200/600 гПа. Это ближайший конкурент по пункту (i) раздела 7 записки. **Полный текст PRR не прочитан** (PDF > 10 МБ для fetch) — читать до любой формулировки новизны.

---

## 1. Таблица: утверждение записки → источник → статус

### 1.1 Разделы 3.1–3.2: Π как свёртка нормированного напряжения

| # | Утверждение записки | Источник(и) | Статус | Комментарий |
|---|---|---|---|---|
| A1 | Фильтрованное уравнение даёт подсеточную силу ∇·τ_ℓ и локальный поток Π_ℓ = −τ_ij ∂_j ū_i | Germano 1992 [V]; Meneveau & Katz 2000 [V]; Eyink 2005 [V]; Eyink & Aluie 2009 [V] | **известно** | Основа всего направления. |
| A2 | τ_ℓ — ковариация, симметрична и ⪰ 0 (реализуемость при положительном ядре) | Vreman, Geurts & Kuerten 1994, JFM 278:351 [V], [A] | **известно** | Условие реализуемости; гауссов фильтр положителен. |
| A3 | ρ = τ/tr τ как «матрица плотности», вектор Блоха (k, c, θ_τ) | Lumley & Newman 1977 [V]; Banerjee et al. 2007 [V]; Pope 2000 [V] (учебник, не открывал) | **частично** | Объект известен как нормированный тензор напряжений Рейнольдса / тензор анизотропии b = τ/tr τ − I/d (+ реализуемость, возврат к изотропии по Rotta). Язык «матрица плотности/Блох» в турбулентности: поиск `density matrix` + SGS stress ничего не вернул. Отличить «переименование» от нового содержания нельзя; новизны здесь нет. |
| A4 | Разложение Π на изотропную часть (давление × дивергенция) и девиаторную часть | Для несжимаемого: Eyink 2006 JFM 549:191, Sec. 2.1.1 [T] (изотропная часть — давление, потока не даёт). Для сжимаемого/дивергентного: Aluie 2011 PRL 106:174502 [V], [U] по содержанию. Атмосфера: аннотация Sichani et al. 2026 — «planetary-scale strain acting on anisotropic effective pressure» [A] | **частично** | Явной записи Π = −kδ̄ − k c σ̄ cos 2Δθ для 2D горизонтального дивергентного ветра на изобарической поверхности не нашёл. Это тривиальная алгебра (разложение τ:S на след и девиатор), а Sichani et al. 2026, судя по аннотации, говорят о близкой вещи (эффективное давление × деформация). Возможное полное совпадение — проверить по тексту. |
| A5 | Выражение 2D-потока через угол: Π = −2 λ_τ λ_s cos 2Θ (у вас −k c σ̄ cos 2Δθ, λ_τ = kc, λ_s = σ̄/2) | Liao & Ouellette 2014 PoF 26:045103 [V]; Liao & Ouellette 2015 PRE 91:063004 [V]; Fang & Ouellette 2016 PRL 117:104501, ур. (2) [V], [T] (даёт ссылки [11, 21, 26] на источники формулы); Yu, Si & Fang 2025 arXiv:2511.05800, ур. (1) [T] (приписывает 2D-форму Liao–Ouellette 2014 и Fang–Ouellette 2016); De Serio 2026 PRFluids 11:034602 [A] | **известно** | «|cos 2Θ| — эффективность переноса»; Θ = 45° — нулевой поток, Θ = 90° — выравнивание со сжимающей осью, поток вниз по масштабу. Совпадает с вашим комментарием про «поворот образующих» и с G4. |
| A6 | Соответствие Λ = Tr(Fρ) ↔ Π как «переименование» | — | — | Внутренняя оценка автора записки. Литература это подтверждает: форма «напряжение : деформация» классична. |

### 1.2 Раздел 3.3: безпамятный нуль и источники ненулевого потока

| # | Утверждение | Источник(и) | Статус | Комментарий |
|---|---|---|---|---|
| B1 | Градиентная/Кларк-модель τ = Cℓ² AAᵀ — ведущий член разложения | Leonard 1974(75) [V]; Clark, Ferziger & Reynolds 1979 [V]; Eyink 2006a JFM 549:159 [V]; Meneveau & Katz 2000 [V] | **известно** | В MSG-разложении Eyink это «strongly UV-local» член, он же «Nonlinear Model» (Eyink 2006b, Sec. 2.1.1 [T]). |
| B2 | Ведущий член даёт Π ≡ 0 в бездивергентном 2D (Cayley–Hamilton: S² = σ²I; член с ω² — давление; член skew-strain ортогонален S) | **Eyink 2006b, JFM 549:191–214, Sec. 2.1.1 и аннотация [V], [T]**; Kraichnan 1971, JFM 47:525 [V] (прямой каскад в 2D не может быть сильно локален); Xiao, Wan, Chen & Eyink 2009, JFM 619:1–44 [V], [A] («the strongly local contribution vanishes identically»); Chen et al. 2006 PRL 96:084502 [V] (содержание не читал, цитируется Eyink и Fang–Ouellette как «физический механизм» обратного каскада) | **известно (точная ссылка: Eyink 2006b)** | Eyink: напряжение от близких масштабов ∝ skew-strain (деформация, повёрнутая на 45°), «no energy flux anywhere in space … to first-order in velocity-gradients». Это ровно ваш вывод «директор на 45° к оси деформации» и (3.2) при δ̄ = 0. Рассуждение с Cayley–Hamilton совпадает с вашим. **Правка сноски в записке**: утверждение «ведущий член даёт ноль» нужно приписывать Eyink 2006 JFM 549:191 (не Chen 2006 PRL / Xiao 2009 как первоисточнику); Xiao 2009 — численное подтверждение. |
| B3 | Ненулевой поток требует отклонения директора от 45°: из «лагранжевой истории» или «структуры мельче грубого градиента» | Eyink 2006b аннотация и Sec. 1–2 [T]: слабая локальность — напряжение (skew-Newtonian) от более мелких масштабов ориентировано под π/4 к деформации **того же** масштаба, а не к крупной деформации; нужен «relative rotation between the principal directions of strain at different scales»; уравнение (2.18) для D_t^(k) α^(k) с advective derivative на масштабе k; члены второго порядка (differential strain-rotation, tensile stress along vorticity contours). Fang & Ouellette 2016 [T]: лаг-выравнивание вдоль траекторий | **известно** (качественно, 2D) | Обе ваши «двери» (история и мелкая структура) — центральные пункты теории Eyink. Требование относительного поворота осей между масштабами = «поворот образующих» gen1 в точной форме. |
| B4 | Для δ̄ ≠ 0 безпамятная часть Π^C = −(Cℓ²/4) δ̄ (3σ̄² + ω̄² + δ̄²) | Eyink 2006b использует несжимаемость (Sec. 2.1.1, App. A [T]). Для 3D сжимаемого: Aluie 2011 [V]. Для 2D горизонтального дивергентного ветра на изобарической поверхности явной формулы не нашёл | **не найдено** (как формула); нуль при δ̄ = 0 — известен | Формулу (3.2) я **не пересчитывал** (задача — без расчётов). Следствие «ненулевая только дивергентная часть» тривиально вытекает из B2 + структуры τ^C. Заметить: на изобарической поверхности атмосферы уравнение для горизонтального ветра не 2D-несжимаемое; перенос определён с поправкой на вертикальный и давление. Не проверено, как Sichani et al. 2026 это обрабатывают. |
| B5 | Поток энстрофии Z_ℓ = −τ^ω·∇ω̄, безпамятный член ∝ ∇ω̄·S̄·∇ω̄ + (δ̄/2)|∇ω̄|²; прямой каскад ⇔ ∇ω̄ вдоль сжимающей оси | Chen, Ecke, Eyink, Wang & Xiao 2003, PRL 91:214501 [V] (препринт: «Athermodynamic alignment in the two-dimensional enstrophy cascade», arXiv nlin/0201043 [A]); Eyink 2001 Nonlinearity 14:787 [U]; Zhou, Fang, Ouellette & Xu 2020, PRFluids 5:054602 [V], [A]; Rivera, Aluie & Ecke 2014, PoF 26:055105 [V] (coarse-graining, мыльная плёнка) | **известно** (2D, лаборатория/DNS); **не найдено** для атмосферы | Формулу Z^C с «+(δ̄/2)|∇ω̄|²» нигде не видел (дивергентная добавка — ваша). Петтерссен/фронтогенез как аналогия — классика метеорологии [U]. |
| B6 | Критерий LKH r = (ω̄ + 2 D_tθ_S)/σ̄, |r| < 1 — выравнивание | Lapeyre, Klein & Hua 1999, PoF 11:3729–3737 [V], [T] (скачан, текст сверен: r = (ω + 2 Dφ/Dt)/σ, φ — угол оси деформации; приняты допущения о медленном изменении r и σ вдоль траектории, (1/σ)Dσ/Dt отброшено); Hua & Klein 1998, Physica D 113:98–110 [V] (точный критерий) | **известно** | Определение в записке совпадает с LKH. |

### 1.3 Раздел 4.2: геометрия выравнивания

| # | Утверждение | Источник(и) | Статус | Комментарий |
|---|---|---|---|---|
| C1 | Статистика выравнивания SGS-напряжения и деформации, 3D | Tao, Katz & Meneveau 2002, JFM 457:35–78 [V], [A] (наиболее растягивающий собственный вектор τ ≈ 32° от наиболее сжимающей оси деформации); Horiuti 2003, JFM 491:65–100 [V], [A] (≈42°); Higgins, Parlange & Meneveau 2003, BLM 109:59–83 [V], [A] (**атмосферный приземный слой**, полевые данные; масштабы порядка метров); Higgins, Meneveau & Parlange 2009, BLM 132:1–9 [V] | **известно** | Атмосфера есть, но это микромасштаб, 3D, приземный слой. |
| C2 | То же в 2D | Chen et al. 2006 [V]; Liao & Ouellette 2014, 2015 [V]; Fang & Ouellette 2016 [V], [T]: средний угол ≈ 45° на всех масштабах («net energy cascade arises from a weak asymmetry between locally strong forward and inverse transfer events»), на масштабах выше инжекции ≈ 37°, «эффективность» ≈ 27 %; De Serio 2026 [V], [A]: вращающийся бассейн 13 м, угол выравнивания + локальное число Россби управляют направлением/устойчивостью каскада | **известно** (лаборатория) | Тезис F4/G4 («обратный перенос ⇒ директор ближе к растягивающей оси») в этих терминах обычное содержание. |
| C3 | То же на синоптических/мезомасштабных 200–1600 км в реанализе/атмосферных моделях | Sichani, Storer & Aluie 2026 PRR 8:033239 [V], [A] (ERA5, «planetary > 3000 км → синоптика 3000–1000 → мезо 1000–100 км», широтные полосы «вверх/вниз по масштабу»; упоминают «strain acting on anisotropic effective pressure»); Christensen, Kouhen, Storer, Aluie & Marshall, EGU26-10479 [A] (годовые операционные анализы ECMWF, 200 и 600 гПа, разложение на дивергентную/вращательную части, два режима спектра); DFD-2023/2024/2025 тезисы Sichani et al. [A]; Kouhen et al., в подготовке/на рецензии в GRL (по сниппету, [A]) | **не найдено** (статистика угла 2Δθ на 200–1600 км по реанализу); **есть близкие работы** | В аннотациях нет упоминания ни угла выравнивания, ни cos 2Δθ, но тексты не читал. Фраза «анизотропное эффективное давление» подозрительно близка к вашей c·cos 2Δθ декомпозиции. **Главный риск**. |

### 1.4 Раздел 4.3: кривизна сноса

| # | Утверждение | Источник(и) | Статус | Комментарий |
|---|---|---|---|---|
| D1 | Мелкая структура несётся крупной скоростью (sweeping), Эйлеровы корреляции ломаются, Лагранжевы — нет | Kraichnan 1964, Phys. Fluids 7:1723 [V]; Tennekes 1975, JFM 67:561–567 [V]; Chen & Kraichnan 1989, Phys. Fluids A 1:2019 [V]; Kraichnan 1977, JFM 83:349 [V] (Eulerian vs Lagrangian renormalization) | **известно** | Общая идея sweeping. Это не про деформационные оси, а про декорреляцию во времени. |
| D2 | Адвективная производная на масштабе k: D_t^(k) = ∂_t + u^(k)·∇ (горизонтальный подъём ∂_t зависит от ℓ) | **Eyink 2006b, ур. (2.18)–(2.20) [T]**: «D_t^(k) = ∂_t + u^(k)·∇ the advective derivative at scale k»; точная динамика угла ориентации деформации (App. A, ур. (A1)–(A4)) | **известно** | Сам объект «снос своей скоростью масштаба» уже в теории Eyink 2006. |
| D3 | Коммутатор [D_t^(ℓ), ∂_ℓ] = −(∂_ℓ ū_ℓ)·∇ и «кривизна/голономия» U(1)-расслоения директоров | — | **не найдено** (как формулировка) | Это тождество: ∂_ℓ(D_t f) = D_t(∂_ℓ f) + (∂_ℓ ū_ℓ)·∇f, откуда (4.1) за одну строку. «Связность по масштабу» в этом смысле в литературе по турбулентности я не нашёл, но и содержания сверх алгебры нет. |
| D4 | Поправка к LKH за дифференциальный снос мелкой структуры: r_{ℓ′ℓ} = (ω̄_ℓ + 2[D_t^(ℓ)θ_S + u_band·∇θ_S])/σ̄_ℓ | LKH 1999 [T]: лагранжева производная D φ/Dt берётся **вдоль траектории частицы реальной жидкости**, то есть вдоль истинной (полной) скорости. Hua & Klein 1998 [V] (точный критерий через гессиан давления). Fang & Ouellette 2016 [T] (эмпирика) | **частично** | Ключевое наблюдение: D_t^(ℓ)θ_S + u_band·∇θ_S = ∂_tθ_S + ū_ℓ′·∇θ_S = лагранжева производная θ_S вдоль траектории, несомой скоростью ℓ′. В LKH с нефильтрованным градиентом эта поправка заложена автоматически (трассер переносится истинной скоростью). Ваша (4.2) — LKH, примененный к деформации, **огрублённой на ℓ**, с трассером/структурой, несомой более мелкой скоростью. Такую запись именно для огрублённых полей и по масштабам в литературе не нашёл, но концептуально это адаптация, не новая теория. |
| D5 | Дифференциальный снос напряжения и деформации разрушает выравнивание и эффективность переноса | **Fang & Ouellette 2016, PRL 117:104501 [V], [T]**: «strain rate is much better aligned with the stress at times in the past, suggesting that the differential advection of the two is responsible for the inefficient spectral transfer»; лаг-выравнивание Θ_sτ(Δt) вдоль лагранжевых траекторий; условная статистика при медленно меняющемся напряжении. Liao & Ouellette 2015 PRE 92:033017 [V], [T] (Okubo–Weiss, лагранжевы корреляции). 3D-аналог: Xu, Pumir & Bodenschatz 2011, Nat. Phys. 7:709 [V] (pirouette effect: вихрь выровнен с деформацией в более ранний момент из-за адвекции); Pumir, Bodenschatz & Xu 2013, PoF 25:035101 [V]; Hamlington, Schumacher & Dahm 2008, PRE 77:026303 и PoF 20:111703 [V] (локальная/нелокальная деформация) | **частично — ближайшая прямая параллель** | Гипотеза G3 («η убывает со Sw») качественно — результат Fang–Ouellette 2016 (лаборатория, 2D, Re ≈ 270). Отличия вашего проекта: (a) явный безразмерный параметр Sw = |u_band·∇θ_S|/σ̄; (b) обработка в терминах огрублённых полос на реанализе; (c) контроль сдвигом и зеркальными суррогатами. Их эксперимент не вводит ни u_band, ни поправку к r, ни коммутатор. |
| D6 | Лагранжевы подсеточные модели (память τ вдоль траектории) | Meneveau, Lund & Cabot 1996, JFM 319:353–385 [V] (усреднение ошибки идентичности Жермано вдоль траектории **разрешённой** скорости с релаксационным временем; описание по памяти, [U]); Bou-Zeid, Meneveau & Parlange 2005, PoF 17:025105 [V] (масштабно-зависимая версия); Chevillard & Meneveau 2006, PRL 97:174501 [V] (замыкание «недавней деформации жидкого элемента»); Li, Chevillard, Eyink & Meneveau 2009, PRE 79:016305 [V] (матричная экспонента для τ) | **частично** (для G6) | Память τ вдоль траектории — известный приём, но в SGS-моделях решает задачу модели (определение коэффициента), а не диагностику выравнивания; поправки на разность скоростей между масштабами не вводят (в просмотренном — не видел). Для G6 эти работы — первые, что стоит прочесть. |
| D7 | Галилеева инвариантность u_band | — | — | Тривиально (разность скоростей). |

### 1.5 Раздел 7 (Q4): огрубление по Жермано на сфере, атмосфера

| # | Утверждение | Источник(и) | Статус |
|---|---|---|---|
| E1 | Огрубление на сфере корректно (свёртки коммутируют с дифоператорами) | Aluie 2019, GEM 10:9 [V] (arXiv:1808.03323 [A]) | **известно** |
| E2 | Океан: карты Π_ℓ | Aluie, Hecht & Vallis 2018, JPO 48:225 [V]; Storer et al. 2022, Nat. Commun. 13:5314 [V]; Storer et al. 2023, Sci. Adv. 9:eadi7420 [V] (глобальный каскад в океане и «атмосферный отпечаток»); Buzzicotti et al. 2023 (JAMES, по тезисам; запись Crossref найдена как препринт ESSOAr [V-partial]) | **известно** |
| E3 | Атмосфера: Π_ℓ по Жермано на реанализе/анализе | **Sichani, Storer & Aluie 2026, PRR 8:033239 [V], [A]**; Christensen et al., EGU26-10479 [A]; тезисы APS DFD 2023 (76-е собрание, «Energy cascades in the troposphere and stratosphere»), 2024, 2025 (K29.1 «Cross-scale kinetic energy pathways in the atmosphere») [A] | **известно** (последние 3 года, группа Aluie) |
| E4 | Атмосфера, 850 гПа, вихрь в полосах 200–1600 км, огибающие | — | **не найдено** |
| E5 | Спектральные потоки (не Жермано) в атмосфере | Lindborg (структурные функции), Nastrom–Gage и др. | вне объёма; не искал подробно |

Дополнительно: тезисы утверждают, что вклад дивергентной составляющей и топографии/конвекции существенен для мезомасштабного потока (Christensen et al. [A]; тезис DFD 2023 [A]). Это пересекается с F5 записки (CAPE понижает P сверх спектра) и с G5 (вакуумная/когерентная части) — проверить по полным текстам.

---

## 2. Проверка точных ссылок, о которых просил запрос

| Ссылка из запроса / записки | Результат проверки |
|---|---|
| Eyink 2006 JFM | Две статьи в одном томе: (a) «Multi-scale gradient expansion of the turbulent stress tensor», JFM **549**, 159–190, DOI 10.1017/S0022112005007895 [V]; (b) «A turbulent constitutive law for the two-dimensional inverse energy cascade», JFM **549**, 191–214, DOI 10.1017/S0022112005007883 [V]. Результат «ведущий член даёт нулевой поток в 2D» — в (b), Sec. 2.1.1 [T]. В (a) изложен общий формализм и 3D-результаты (skew-strain, vortex stretching) [T]. |
| Chen et al. 2006 PRL | Chen, Ecke, Eyink, Rivera, Wan, Xiao, «Physical mechanism of the two-dimensional inverse energy cascade», PRL **96**, 084502 (2006) [V]. Содержание не читал; утверждение «ведущий член даёт 0» там, судя по тому, как его цитируют, не первоисточник (см. B2). |
| Xiao et al. 2009 JFM | Xiao, Wan, Chen, Eyink, «Physical mechanism of the inverse energy cascade of two-dimensional turbulence: a numerical investigation», JFM **619**, 1–44 (2009) [V], абстракт [A]: vortex thinning; сильно локальный вклад исчезает тождественно. |
| Chen et al. 2003 PRL | Chen, Ecke, Eyink, Wang, Xiao, «Physical mechanism of the two-dimensional enstrophy cascade», PRL **91**, 214501 (2003) [V]. В записке — «каскад энстрофии», верно. |
| Tao, Katz, Meneveau 2002 JFM | JFM **457**, 35–78 [V]. Верно. |
| Lapeyre, Klein, Hua 1999 | Phys. Fluids **11**(12), 3729–3737, «Does the tracer gradient vector align with the strain eigenvectors in 2D turbulence?» [V]. |
| «Lapeyre, Hua & Klein 2001 (каскад трассера)» | Исправление названия: Lapeyre, Hua, Klein, «Dynamics of the orientation of active and passive scalars in two-dimensional turbulence», Phys. Fluids **13**, 251–264 (2001) [V]. Это не «каскад трассера», а динамика ориентации градиента (активного и пассивного скаляра). |
| «Aluie & Eyink 2009 (локальность)» | Две части: Eyink & Aluie, Part I «Smooth coarse graining», PoF **21**, 115107; Aluie & Eyink, Part II «Sharp spectral filter», PoF **21**, 115108 (2009) [V]. |
| Leonard 1974 | Adv. Geophys. **18A**, 237–248 [V]; Crossref датирует 1975 (том трудов симпозиума, 1974). |
| Meneveau–Lund–Cabot 1996 | JFM **319**, 353–385 [V]. |
| Storer et al. 2022 | Nat. Commun. **13**, 5314 [V]. |
| Rotta 1951; Petterssen 1936 | **[U]**, не сверял (Rotta, Z. Phys. 129:547; Petterssen, Geofys. Publ. 11(6)). |

---

## 3. Что в записке уже не нужно считать «ново» и что остаётся

**Не новое (известная физика, не заявлять):**
- Нуль Кларка в бездивергентном 2D, 45° к оси деформации, Cayley–Hamilton (Eyink 2006b; Kraichnan 1971).
- Условие ненулевого потока: относительный поворот главных осей деформации между масштабами; weakly local механизм (Eyink 2006b).
- Формула Π = −2 λ_τ λ_s cos 2Θ и «эффективность |cos 2Θ|» (Liao–Ouellette 2014/2015; Fang–Ouellette 2016).
- «Лаг-выравнивание вдоль лагранжевых траекторий и роль дифференциальной адвекции» — на уровне идеи и лабораторной эмпирики (Fang–Ouellette 2016; pirouette effect в 3D).
- Тензор ρ = τ/tr τ и возврат к изотропии (Lumley–Newman 1977, Rotta).
- Критерий r с вращением осей деформации (LKH 1999) и его эквивалент (2.18) у Eyink.
- Карты Π_ℓ по Жермано на сфере (Aluie 2019; океан — 2018–2023; атмосфера — Sichani et al. 2026, Christensen et al.).

**Возможная новизна (после чтения полных текстов Sichani 2026 и Christensen):**
1. **Статистика угла 2Δθ и эффективности c·cos 2Δθ на синоптике/мезомасштабах в реанализе** (200–1600 км, 850 гПа), включая тест «это несёт b3» (G1) и связь со стационарным шаблоном/P. Поиск не нашёл готового. Риск: группа Aluie имеет инструмент, а «анизотропное эффективное давление» в Sichani 2026 — прямо рядом.
2. **Число сноса Sw как диагностика и проверка нуля (лаг 5 суток, зеркальные суррогаты)** на огрублённых полосах. Идея — Fang–Ouellette 2016; операционализация (u_band·∇θ_S, нормировка на σ̄, поправка к r) нова. Теорию сверх LKH+Eyink (2.18) это не даёт.
3. **Явная дивергентная добавка к 2D-формулам** (Π^C ∝ δ̄, Z^C с (δ̄/2)|∇ω̄|²): формул не нашёл; но тривиальны и требуют проверки на корректность 2D-несжимаемости применительно к 850 гПа.
4. **Связь геометрии с P** — вне охвата данного обзора (нет внешней литературы про P; см. audits).

**Где новизны нет:** язык расслоения/голономии/матрицы плотности для уровней 0–1 (подтверждает раздел 8 записки: это переименование); «нуль Кларка» как результат; «alignment determines cascade direction» как тезис.

**Рекомендации по порядку чтения перед любой претензией:**
1. Sichani–Storer–Aluie PRR 8:033239 (полный текст, Methods и раздел про «effective pressure»; нужно открыть через доступ к APS, fetch не потянул размер) — проверить, есть ли там разложение Π = изотропная + девиаторная часть и статистика углов.
2. Christensen et al. (полная версия, если выйдет) и Kouhen et al. (GRL).
3. Fang & Ouellette 2016 (принятая рукопись скачана и прочитана Sec. 1–2; дочитать условную статистику) и Liao–Ouellette 2014/2015.
4. Meneveau–Lund–Cabot 1996 и Chevillard–Meneveau 2006 для G6.

---

## 4. Ограничения обзора

- Часть источников подтверждена только по аннотации/сниппетам ([A]); тексты Tao–Katz–Meneveau 2002, Higgins 2003, Xiao 2009, Chen 2006 PRL, De Serio 2026, Sichani 2026, Christensen EGU26, Hamlington 2008, Pumir 2013 не читал.
- Прочитаны по тексту: Eyink 2006a, 2006b (arXiv-версии: nlin/0512022, nlin/0512023, поиск по ключевым пунктам, не вся статья), LKH 1999 (r), Fang & Ouellette 2016 (принятая рукопись), аннотация и введение Liao–Ouellette 2015 (PRE 92:033017), Yu–Si–Fang 2025 (ур. (1) и атрибуция, через HTML-версию). Liao–Ouellette 2014 и PRE 91:063004 по тексту не читал: ссылки на них как на источник формулы cos 2Θ взяты из списка литературы Fang–Ouellette 2016 ([11, 21, 26] у ур. (2)).
- Утверждения «не найдено» ограничены поиском в сети в рамках этой сессии; рекомендуется Google Scholar/Web of Science по `"stress-strain alignment" atmosphere`, `"coarse-graining" "alignment angle" ERA5`.
- Формулы записки (3.1)–(3.2), (4.1)–(4.2) не пересчитывались. Замечено лишь, что (4.1) — тождество, а (3.2) при δ̄ = 0 согласуется с Eyink 2006b.

---

## 5. BibTeX (все записи сверены по Crossref [V], кроме отмеченных)

```bibtex
@article{Germano1992,
  author  = {Germano, M.},
  title   = {Turbulence: the filtering approach},
  journal = {Journal of Fluid Mechanics},
  volume  = {238},
  pages   = {325--336},
  year    = {1992},
  doi     = {10.1017/S0022112092001733}
}

@article{Eyink2005Locality,
  author  = {Eyink, Gregory L.},
  title   = {Locality of turbulent cascades},
  journal = {Physica D: Nonlinear Phenomena},
  volume  = {207},
  number  = {1--2},
  pages   = {91--116},
  year    = {2005},
  doi     = {10.1016/j.physd.2005.05.018}
}

@article{Eyink2006MSG,
  author  = {Eyink, Gregory L.},
  title   = {Multi-scale gradient expansion of the turbulent stress tensor},
  journal = {Journal of Fluid Mechanics},
  volume  = {549},
  pages   = {159--190},
  year    = {2006},
  doi     = {10.1017/S0022112005007895}
}

@article{Eyink2006Constitutive,
  author  = {Eyink, Gregory L.},
  title   = {A turbulent constitutive law for the two-dimensional inverse energy cascade},
  journal = {Journal of Fluid Mechanics},
  volume  = {549},
  pages   = {191--214},
  year    = {2006},
  doi     = {10.1017/S0022112005007883}
}

@article{EyinkAluie2009I,
  author  = {Eyink, Gregory L. and Aluie, Hussein},
  title   = {Localness of energy cascade in hydrodynamic turbulence. {I}. Smooth coarse graining},
  journal = {Physics of Fluids},
  volume  = {21},
  number  = {11},
  pages   = {115107},
  year    = {2009},
  doi     = {10.1063/1.3266883}
}

@article{AluieEyink2009II,
  author  = {Aluie, Hussein and Eyink, Gregory L.},
  title   = {Localness of energy cascade in hydrodynamic turbulence. {II}. Sharp spectral filter},
  journal = {Physics of Fluids},
  volume  = {21},
  number  = {11},
  pages   = {115108},
  year    = {2009},
  doi     = {10.1063/1.3266948}
}

@article{EyinkSreenivasan2006,
  author  = {Eyink, Gregory L. and Sreenivasan, Katepalli R.},
  title   = {Onsager and the theory of hydrodynamic turbulence},
  journal = {Reviews of Modern Physics},
  volume  = {78},
  number  = {1},
  pages   = {87--135},
  year    = {2006},
  doi     = {10.1103/RevModPhys.78.87}
}

@article{Kraichnan1971,
  author  = {Kraichnan, Robert H.},
  title   = {Inertial-range transfer in two- and three-dimensional turbulence},
  journal = {Journal of Fluid Mechanics},
  volume  = {47},
  number  = {3},
  pages   = {525--535},
  year    = {1971},
  doi     = {10.1017/S0022112071001216}
}

@article{Chen2006PRL,
  author  = {Chen, Shiyi and Ecke, Robert E. and Eyink, Gregory L. and Rivera, Michael and Wan, Minping and Xiao, Zuoli},
  title   = {Physical mechanism of the two-dimensional inverse energy cascade},
  journal = {Physical Review Letters},
  volume  = {96},
  number  = {8},
  pages   = {084502},
  year    = {2006},
  doi     = {10.1103/PhysRevLett.96.084502}
}

@article{Xiao2009JFM,
  author  = {Xiao, Zuoli and Wan, Minping and Chen, Shiyi and Eyink, Gregory L.},
  title   = {Physical mechanism of the inverse energy cascade of two-dimensional turbulence: a numerical investigation},
  journal = {Journal of Fluid Mechanics},
  volume  = {619},
  pages   = {1--44},
  year    = {2009},
  doi     = {10.1017/S0022112008004266}
}

@article{Chen2003PRL,
  author  = {Chen, Shiyi and Ecke, Robert E. and Eyink, Gregory L. and Wang, Xiao and Xiao, Zuoli},
  title   = {Physical mechanism of the two-dimensional enstrophy cascade},
  journal = {Physical Review Letters},
  volume  = {91},
  number  = {21},
  pages   = {214501},
  year    = {2003},
  doi     = {10.1103/PhysRevLett.91.214501}
}

@article{Zhou2020PRFluids,
  author  = {Zhou, Zeyou and Fang, Lei and Ouellette, Nicholas T. and Xu, Haitao},
  title   = {Vorticity gradient stretching in the direct enstrophy transfer process of two-dimensional turbulence},
  journal = {Physical Review Fluids},
  volume  = {5},
  number  = {5},
  pages   = {054602},
  year    = {2020},
  doi     = {10.1103/PhysRevFluids.5.054602}
}

@article{Rivera2014PoF,
  author  = {Rivera, Michael K. and Aluie, Hussein and Ecke, Robert E.},
  title   = {The direct enstrophy cascade of two-dimensional soap film flows},
  journal = {Physics of Fluids},
  volume  = {26},
  number  = {5},
  pages   = {055105},
  year    = {2014},
  doi     = {10.1063/1.4873579}
}

@article{Tao2002JFM,
  author  = {Tao, Bo and Katz, Joseph and Meneveau, Charles},
  title   = {Statistical geometry of subgrid-scale stresses determined from holographic particle image velocimetry measurements},
  journal = {Journal of Fluid Mechanics},
  volume  = {457},
  pages   = {35--78},
  year    = {2002},
  doi     = {10.1017/S0022112001007443}
}

@article{Horiuti2003,
  author  = {Horiuti, Kiyosi},
  title   = {Roles of non-aligned eigenvectors of strain-rate and subgrid-scale stress tensors in turbulence generation},
  journal = {Journal of Fluid Mechanics},
  volume  = {491},
  pages   = {65--100},
  year    = {2003},
  doi     = {10.1017/S0022112003005299}
}

@article{Higgins2003BLM,
  author  = {Higgins, Chad W. and Parlange, Marc B. and Meneveau, Charles},
  title   = {Alignment trends of velocity gradients and subgrid-scale fluxes in the turbulent atmospheric boundary layer},
  journal = {Boundary-Layer Meteorology},
  volume  = {109},
  number  = {1},
  pages   = {59--83},
  year    = {2003},
  doi     = {10.1023/A:1025484500899}
}

@article{Higgins2009BLM,
  author  = {Higgins, Chad W. and Meneveau, Charles and Parlange, Marc B.},
  title   = {Geometric alignments of the subgrid-scale force in the atmospheric boundary layer},
  journal = {Boundary-Layer Meteorology},
  volume  = {132},
  number  = {1},
  pages   = {1--9},
  year    = {2009},
  doi     = {10.1007/s10546-009-9385-3}
}

@article{LiaoOuellette2014,
  author  = {Liao, Yang and Ouellette, Nicholas T.},
  title   = {Geometry of scale-to-scale energy and enstrophy transport in two-dimensional flow},
  journal = {Physics of Fluids},
  volume  = {26},
  number  = {4},
  pages   = {045103},
  year    = {2014},
  doi     = {10.1063/1.4871107}
}

@article{LiaoOuellette2015PRE91,
  author  = {Liao, Yang and Ouellette, Nicholas T.},
  title   = {Long-range ordering of turbulent stresses in two-dimensional flow},
  journal = {Physical Review E},
  volume  = {91},
  number  = {6},
  pages   = {063004},
  year    = {2015},
  doi     = {10.1103/PhysRevE.91.063004}
}

@article{LiaoOuellette2015PRE92,
  author  = {Liao, Yang and Ouellette, Nicholas T.},
  title   = {Correlations between the instantaneous velocity gradient and the evolution of scale-to-scale fluxes in two-dimensional flow},
  journal = {Physical Review E},
  volume  = {92},
  number  = {3},
  pages   = {033017},
  year    = {2015},
  doi     = {10.1103/PhysRevE.92.033017}
}

@article{FangOuellette2016,
  author  = {Fang, Lei and Ouellette, Nicholas T.},
  title   = {Advection and the efficiency of spectral energy transfer in two-dimensional turbulence},
  journal = {Physical Review Letters},
  volume  = {117},
  number  = {10},
  pages   = {104501},
  year    = {2016},
  doi     = {10.1103/PhysRevLett.117.104501}
}

@article{DeSerio2026,
  author  = {De Serio, Francesca},
  title   = {Local organization of the turbulent energy cascade in geophysical flows},
  journal = {Physical Review Fluids},
  volume  = {11},
  number  = {3},
  pages   = {034602},
  year    = {2026},
  doi     = {10.1103/mkwg-hmgn}
}

@misc{YuSiFang2025,
  author        = {Yu, Ziyue and Si, Xinyu and Fang, Lei},
  title         = {Turbulence at low {R}eynolds numbers},
  year          = {2025},
  eprint        = {2511.05800},
  archivePrefix = {arXiv},
  primaryClass  = {physics.flu-dyn}
}

@article{LapeyreKleinHua1999,
  author  = {Lapeyre, Guillaume and Klein, Patrice and Hua, Bach Lien},
  title   = {Does the tracer gradient vector align with the strain eigenvectors in 2{D} turbulence?},
  journal = {Physics of Fluids},
  volume  = {11},
  number  = {12},
  pages   = {3729--3737},
  year    = {1999},
  doi     = {10.1063/1.870234}
}

@article{HuaKlein1998,
  author  = {Hua, Bach Lien and Klein, Patrice},
  title   = {An exact criterion for the stirring properties of nearly two-dimensional turbulence},
  journal = {Physica D: Nonlinear Phenomena},
  volume  = {113},
  number  = {1},
  pages   = {98--110},
  year    = {1998},
  doi     = {10.1016/S0167-2789(97)00143-7}
}

@article{LapeyreHuaKlein2001,
  author  = {Lapeyre, Guillaume and Hua, Bach Lien and Klein, Patrice},
  title   = {Dynamics of the orientation of active and passive scalars in two-dimensional turbulence},
  journal = {Physics of Fluids},
  volume  = {13},
  number  = {1},
  pages   = {251--264},
  year    = {2001},
  doi     = {10.1063/1.1324705}
}

@article{Okubo1970,
  author  = {Okubo, Akira},
  title   = {Horizontal dispersion of floatable particles in the vicinity of velocity singularities such as convergences},
  journal = {Deep Sea Research and Oceanographic Abstracts},
  volume  = {17},
  number  = {3},
  pages   = {445--454},
  year    = {1970},
  doi     = {10.1016/0011-7471(70)90059-8}
}

@article{Weiss1991,
  author  = {Weiss, Jeffrey},
  title   = {The dynamics of enstrophy transfer in two-dimensional hydrodynamics},
  journal = {Physica D: Nonlinear Phenomena},
  volume  = {48},
  number  = {2--3},
  pages   = {273--294},
  year    = {1991},
  doi     = {10.1016/0167-2789(91)90088-Q}
}

@article{Kraichnan1964,
  author  = {Kraichnan, Robert H.},
  title   = {Kolmogorov's hypotheses and {E}ulerian turbulence theory},
  journal = {The Physics of Fluids},
  volume  = {7},
  number  = {11},
  pages   = {1723--1734},
  year    = {1964},
  doi     = {10.1063/1.2746572}
}

@article{Kraichnan1977,
  author  = {Kraichnan, Robert H.},
  title   = {Eulerian and {L}agrangian renormalization in turbulence theory},
  journal = {Journal of Fluid Mechanics},
  volume  = {83},
  number  = {2},
  pages   = {349--374},
  year    = {1977},
  doi     = {10.1017/S0022112077001232}
}

@article{Tennekes1975,
  author  = {Tennekes, H.},
  title   = {Eulerian and {L}agrangian time microscales in isotropic turbulence},
  journal = {Journal of Fluid Mechanics},
  volume  = {67},
  number  = {3},
  pages   = {561--567},
  year    = {1975},
  doi     = {10.1017/S0022112075000468}
}

@article{ChenKraichnan1989,
  author  = {Chen, Shiyi and Kraichnan, Robert H.},
  title   = {Sweeping decorrelation in isotropic turbulence},
  journal = {Physics of Fluids A: Fluid Dynamics},
  volume  = {1},
  number  = {12},
  pages   = {2019--2024},
  year    = {1989},
  doi     = {10.1063/1.857475}
}

@article{MeneveauLundCabot1996,
  author  = {Meneveau, Charles and Lund, Thomas S. and Cabot, William H.},
  title   = {A {L}agrangian dynamic subgrid-scale model of turbulence},
  journal = {Journal of Fluid Mechanics},
  volume  = {319},
  pages   = {353--385},
  year    = {1996},
  doi     = {10.1017/S0022112096007379}
}

@article{BouZeid2005,
  author  = {Bou-Zeid, Elie and Meneveau, Charles and Parlange, Marc},
  title   = {A scale-dependent {L}agrangian dynamic model for large eddy simulation of complex turbulent flows},
  journal = {Physics of Fluids},
  volume  = {17},
  number  = {2},
  pages   = {025105},
  year    = {2005},
  doi     = {10.1063/1.1839152}
}

@article{ChevillardMeneveau2006,
  author  = {Chevillard, Laurent and Meneveau, Charles},
  title   = {Lagrangian dynamics and statistical geometric structure of turbulence},
  journal = {Physical Review Letters},
  volume  = {97},
  number  = {17},
  pages   = {174501},
  year    = {2006},
  doi     = {10.1103/PhysRevLett.97.174501}
}

@article{Li2009PRE,
  author  = {Li, Yi and Chevillard, Laurent and Eyink, Gregory and Meneveau, Charles},
  title   = {Matrix exponential-based closures for the turbulent subgrid-scale stress tensor},
  journal = {Physical Review E},
  volume  = {79},
  number  = {1},
  pages   = {016305},
  year    = {2009},
  doi     = {10.1103/PhysRevE.79.016305}
}

@article{Xu2011Pirouette,
  author  = {Xu, Haitao and Pumir, Alain and Bodenschatz, Eberhard},
  title   = {The pirouette effect in turbulent flows},
  journal = {Nature Physics},
  volume  = {7},
  pages   = {709--712},
  year    = {2011},
  doi     = {10.1038/nphys2010}
}

@article{Pumir2013PoF,
  author  = {Pumir, Alain and Bodenschatz, Eberhard and Xu, Haitao},
  title   = {Tetrahedron deformation and alignment of perceived vorticity and strain in a turbulent flow},
  journal = {Physics of Fluids},
  volume  = {25},
  number  = {3},
  pages   = {035101},
  year    = {2013},
  doi     = {10.1063/1.4795547}
}

@article{Hamlington2008PRE,
  author  = {Hamlington, Peter E. and Schumacher, J{\"o}rg and Dahm, Werner J. A.},
  title   = {Local and nonlocal strain rate fields and vorticity alignment in turbulent flows},
  journal = {Physical Review E},
  volume  = {77},
  number  = {2},
  pages   = {026303},
  year    = {2008},
  doi     = {10.1103/PhysRevE.77.026303}
}

@article{Hamlington2008PoF,
  author  = {Hamlington, Peter E. and Schumacher, J{\"o}rg and Dahm, Werner J. A.},
  title   = {Direct assessment of vorticity alignment with local and nonlocal strain rates in turbulent flows},
  journal = {Physics of Fluids},
  volume  = {20},
  number  = {11},
  pages   = {111703},
  year    = {2008},
  doi     = {10.1063/1.3021055}
}

@incollection{Leonard1975,
  author    = {Leonard, A.},
  title     = {Energy cascade in large-eddy simulations of turbulent fluid flows},
  booktitle = {Turbulent Diffusion in Environmental Pollution, Advances in Geophysics},
  volume    = {18A},
  pages     = {237--248},
  year      = {1975},
  doi       = {10.1016/S0065-2687(08)60464-1},
  note      = {Proceedings of a symposium held 1974, Charlottesville}
}

@article{Clark1979,
  author  = {Clark, R. A. and Ferziger, J. H. and Reynolds, W. C.},
  title   = {Evaluation of subgrid-scale models using an accurately simulated turbulent flow},
  journal = {Journal of Fluid Mechanics},
  volume  = {91},
  number  = {1},
  pages   = {1--16},
  year    = {1979},
  doi     = {10.1017/S002211207900001X}
}

@article{MeneveauKatz2000,
  author  = {Meneveau, Charles and Katz, Joseph},
  title   = {Scale-invariance and turbulence models for large-eddy simulation},
  journal = {Annual Review of Fluid Mechanics},
  volume  = {32},
  pages   = {1--32},
  year    = {2000},
  doi     = {10.1146/annurev.fluid.32.1.1}
}

@article{Piomelli1991,
  author  = {Piomelli, Ugo and Cabot, William H. and Moin, Parviz and Lee, Sangsan},
  title   = {Subgrid-scale backscatter in turbulent and transitional flows},
  journal = {Physics of Fluids A: Fluid Dynamics},
  volume  = {3},
  number  = {7},
  pages   = {1766--1771},
  year    = {1991},
  doi     = {10.1063/1.857956}
}

@article{Vreman1994,
  author  = {Vreman, Bert and Geurts, Bernard and Kuerten, Hans},
  title   = {Realizability conditions for the turbulent stress tensor in large-eddy simulation},
  journal = {Journal of Fluid Mechanics},
  volume  = {278},
  pages   = {351--362},
  year    = {1994},
  doi     = {10.1017/S0022112094003745}
}

@article{LumleyNewman1977,
  author  = {Lumley, J. L. and Newman, G. R.},
  title   = {The return to isotropy of homogeneous turbulence},
  journal = {Journal of Fluid Mechanics},
  volume  = {82},
  number  = {1},
  pages   = {161--178},
  year    = {1977},
  doi     = {10.1017/S0022112077000585}
}

@article{Banerjee2007,
  author  = {Banerjee, S. and Krahl, R. and Durst, F. and Zenger, Ch.},
  title   = {Presentation of anisotropy properties of turbulence, invariants versus eigenvalue approaches},
  journal = {Journal of Turbulence},
  volume  = {8},
  pages   = {N32},
  year    = {2007},
  doi     = {10.1080/14685240701506896}
}

@book{Pope2000,
  author    = {Pope, Stephen B.},
  title     = {Turbulent Flows},
  publisher = {Cambridge University Press},
  year      = {2000},
  doi       = {10.1017/CBO9780511840531}
}

@article{Aluie2011PRL,
  author  = {Aluie, Hussein},
  title   = {Compressible turbulence: the cascade and its locality},
  journal = {Physical Review Letters},
  volume  = {106},
  number  = {17},
  pages   = {174502},
  year    = {2011},
  doi     = {10.1103/PhysRevLett.106.174502}
}

@article{Aluie2019GEM,
  author  = {Aluie, Hussein},
  title   = {Convolutions on the sphere: commutation with differential operators},
  journal = {GEM -- International Journal on Geomathematics},
  volume  = {10},
  pages   = {9},
  year    = {2019},
  doi     = {10.1007/s13137-019-0123-9}
}

@article{AluieHechtVallis2018,
  author  = {Aluie, Hussein and Hecht, Matthew and Vallis, Geoffrey K.},
  title   = {Mapping the energy cascade in the {N}orth {A}tlantic {O}cean: the coarse-graining approach},
  journal = {Journal of Physical Oceanography},
  volume  = {48},
  number  = {2},
  pages   = {225--244},
  year    = {2018},
  doi     = {10.1175/JPO-D-17-0100.1}
}

@article{Storer2022NatCommun,
  author  = {Storer, Benjamin A. and Buzzicotti, Michele and Khatri, Hemant and Griffies, Stephen M. and Aluie, Hussein},
  title   = {Global energy spectrum of the general oceanic circulation},
  journal = {Nature Communications},
  volume  = {13},
  pages   = {5314},
  year    = {2022},
  doi     = {10.1038/s41467-022-33031-3}
}

@article{Storer2023SciAdv,
  author  = {Storer, Benjamin A. and Buzzicotti, Michele and Khatri, Hemant and Griffies, Stephen M. and Aluie, Hussein},
  title   = {Global cascade of kinetic energy in the ocean and the atmospheric imprint},
  journal = {Science Advances},
  volume  = {9},
  number  = {51},
  pages   = {eadi7420},
  year    = {2023},
  doi     = {10.1126/sciadv.adi7420}
}

@article{Sichani2026PRR,
  author  = {Hadi Sichani, Pejman and Storer, Benjamin A. and Aluie, Hussein},
  title   = {Atmospheric transfer of kinetic energy across planetary-scale circulation and weather},
  journal = {Physical Review Research},
  volume  = {8},
  number  = {3},
  pages   = {033239},
  year    = {2026},
  doi     = {10.1103/1jf1-rm75}
}

@misc{Christensen2026EGU,
  author = {Christensen, Hannah and Kouhen, Salah and Storer, Benjamin and Aluie, Hussein and Marshall, David},
  title  = {Local kinetic energy fluxes in the atmospheric mesoscales},
  howpublished = {EGU General Assembly 2026, abstract EGU26-10479},
  year   = {2026},
  note   = {conference abstract; not verified beyond the abstract page}
}
```

Записи [U] (не вносить в .bib без сверки): Rotta 1951 (Z. Phys. 129:547); Petterssen 1936 (Geofys. Publ. 11(6)); Eyink 2001 Nonlinearity 14:787; Buzzicotti et al. 2023 JAMES (в Crossref нашёлся только препринт ESSOAr, DOI 10.22541/essoar.168500394.43917698/v1).
