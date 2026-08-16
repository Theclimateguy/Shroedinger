# Phase 9 -> manuscript: replacement text for Sect. 6.2

Source of every number: `clean_experiments/results/experiment_B9_predictability/`
(`summary.json`, `report.md`, `figB9_A_vs_predictability.png`); protocol
`docs/PROTOCOL_PHASE9_PREDICTABILITY.md`, verdict **NEGATIVE**.

The present Sect. 6.2 states an untested hypothesis. It can now be replaced by
a tested result. The result does not support the hypothesis as it was written,
and the honest version is stronger than the speculative one: it converts a
promise into a measurement, a falsification, and a corrected mechanism.

---

## English draft

### 6.2 The relation to predictability, tested

The literature on forecast error growth assigns cross-scale interaction the
role of a channel through which uncertainty at small scales contaminates the
synoptic level and limits predictability \cite{Lorenz1969,Zhang2007,RotunnoSnyder2008,Judt2020}.
The regional climatology of that interaction constructed here yields a
directional prediction: regimes of high asymmetry should lose predictability
faster. We tested that prediction directly against an operational ensemble
prediction system, under criteria fixed before the forecast data were
retrieved (protocol and deviations in the repository).

Forecasts are taken from the NOAA GEFS v12 ensemble (control plus five
perturbed members, 850~hPa wind on a $0.5^\circ$ grid), initialised at 00~UTC
every fifth day of each seasonal window, at lead times of 12 to 168~h;
17~initialisations per window, the same twelve regions and the same four
windows of 2023--2024 that carry the descriptors, with the eight regions of
2021--2022 held back as an independent out-of-sample test in the opposite
ENSO phase. Forecast fields are decomposed into the same octave levels as the
descriptors and verified in two independent ways: against the forecasting
system's own analysis, and against ERA5. The relative error
$r_b(\tau)=e_b(\tau)/\sqrt{2}\sigma_b$ and the relative ensemble spread
$q_b(\tau)$ are referred to their saturation value, and the growth rates
$\lambda$ and $\mu$ are the slopes of $\operatorname{logit}r$ and
$\operatorname{logit}q$ in lead time, so that the growth rate is separated
from the level of the initial error.

**The prediction is not confirmed.** Across the twelve regions the rank
correlation between $A$ and the error growth rate is $\rho=-0.25$ against the
system's own analysis and $\rho=-0.32$ against ERA5: the sign is opposite to
the one predicted, and neither is significant. The out-of-sample sample of
2021--2022 reproduces the same wrong sign ($\rho=-0.33$, $n=8$). The reason is
visible in the controls. The error growth rate is governed by baroclinicity,
not by the architecture of the hierarchy: it correlates with the absolute
latitude of the domain at $\rho=+0.86$ and with the logarithm of the band
variance at $\rho=+0.81$, whereas $A$ is *anti*-correlated with both
($-0.58$ and $-0.65$). A plain spectral characteristic --- the band variance
--- predicts regional error growth far better than $A$ does ($0.81$ against
$0.25$), so the descriptor fails the placebo control that the earlier phases
of this work passed.

**What the data show instead.** Two quantities do follow $A$. The growth of
the ensemble spread, which requires no verifying analysis at all, correlates
with $A$ at $\rho=+0.60$ ($p=0.019$); the association survives the removal of
absolute latitude, land fraction, band variance, spectral slope and the
initial error ($\rho=+0.62$, $p=0.016$), survives with region fixed effects
across the 48 region-windows ($p=0.046$), and is stable under leaving out any
one region ($\rho$ between $+0.48$ and $+0.85$). The predictability horizon
$T_{50}$ likewise shortens as $A$ grows ($\rho=-0.51$). But on the independent
years of 2021--2022 the spread association weakens to $\rho=+0.26$ ($n=8$,
not significant), and the sign disagreement between the error-based and
spread-based measures is by itself sufficient, under the criteria fixed in
advance, to record the outcome as negative.

**The corrected reading.** In the regions of high asymmetry the forecast error
does not grow faster; it starts higher. The relative error at the 12~h lead
ranges from 0.27 over Europe to 0.61 over Amazonia and correlates with $A$ at
$\rho=+0.48$, and in ten of the 48 region-windows the 200--400~km band is
already saturated at the shortest lead available --- there the forecast
carries no information about the placement of mesoscale variability from the
outset, and a growth rate is not a measurable quantity. The measured growth
rate is correspondingly compressed where the initial error is large
($\rho=-0.76$ between the two), which accounts for the negative sign found
above.

This is not the mechanism proposed in the hypothesis, and it is closer to the
interpretation of Sect.~6.1. A large $A$ does not describe a regime in which
information is lost faster during a forecast; it describes a regime in which
the mesoscale placement is already unrecoverable at the moment of analysis.
The applied consequence stated earlier --- that regions of large $A$ are those
in which statistical refinement from large-scale fields is least warranted ---
is supported by these numbers; the extension of that statement to the rate of
error growth in ensemble forecasting is not.

Two limitations bound this conclusion. With twelve regions the design detects
only strong associations ($|\rho|\geq0.50$ at $\alpha=0.05$, one-sided), and
this was stated before the data were examined; a null here is not evidence of
a small effect but the absence of a large one. And the spread-based
association admits a competing explanation that the present material cannot
exclude: the initial perturbations of an ensemble prediction system may
themselves be under-dispersive in the moist tropics, in which case $\mu$ would
be reporting a property of the forecasting system rather than of the
atmosphere.

---

## Russian draft

### 6.2 Связь с предсказуемостью: проверка

Литература о росте ошибки прогноза отводит межмасштабному взаимодействию роль
канала, по которому неопределённость малых масштабов заражает синоптический
уровень и ограничивает предсказуемость \cite{Lorenz1969,Zhang2007,RotunnoSnyder2008,Judt2020}.
Построенная здесь региональная климатология этого взаимодействия даёт
направленное предсказание: режимы с высокой асимметрией должны терять
предсказуемость быстрее. Это предсказание проверено напрямую по оперативной
ансамблевой системе прогноза, по критериям, зафиксированным до обращения к
прогностическим данным (протокол и все отступления --- в репозитории).

Прогнозы взяты из ансамбля NOAA GEFS v12 (контрольный и пять возмущённых
членов, ветер на 850~гПа, сетка $0{,}5^\circ$), старты в 00~UTC каждые пять
суток сезонного окна, заблаговременности от 12 до 168~ч; 17~стартов на окно,
те же двенадцать регионов и те же четыре окна 2023--2024~гг., на которых
построены дескрипторы, а восемь регионов 2021--2022~гг. оставлены как
независимая выборка в противоположной фазе ЭНЮК. Прогностические поля
разлагаются на те же октавные уровни, что и дескрипторы, и верифицируются
двумя независимыми способами: по собственному анализу прогностической системы
и по ERA5. Относительная ошибка $r_b(\tau)$ и относительный разброс ансамбля
$q_b(\tau)$ отнесены к уровню насыщения, а скорости роста $\lambda$ и $\mu$
определены как наклоны $\operatorname{logit}r$ и $\operatorname{logit}q$ по
заблаговременности --- так скорость роста отделена от уровня начальной ошибки.

**Предсказание не подтвердилось.** По двенадцати регионам ранговая корреляция
$A$ со скоростью роста ошибки составляет $\rho=-0{,}25$ при верификации по
собственному анализу системы и $\rho=-0{,}32$ по ERA5: знак противоположен
предсказанному, значимости нет. Независимая выборка 2021--2022~гг.
воспроизводит тот же неверный знак ($\rho=-0{,}33$, $n=8$). Причина видна из
контролей. Скорость роста ошибки определяется бароклинностью, а не
архитектурой иерархии: она коррелирует с модулем широты области при
$\rho=+0{,}86$ и с логарифмом дисперсии полосы при $\rho=+0{,}81$, тогда как
$A$ связан с обоими *отрицательно* ($-0{,}58$ и $-0{,}65$). Простая
спектральная характеристика --- дисперсия полосы --- предсказывает
региональный рост ошибки существенно лучше, чем $A$ ($0{,}81$ против
$0{,}25$), то есть дескриптор не проходит контроль плацебо, который он
проходил на предыдущих этапах работы.

**Что данные показывают вместо этого.** За $A$ следуют две величины. Рост
разброса ансамбля, для которого верифицирующий анализ вообще не нужен,
коррелирует с $A$ при $\rho=+0{,}60$ ($p=0{,}019$); связь сохраняется после
удаления модуля широты, доли суши, дисперсии полосы, наклона спектра и
начальной ошибки ($\rho=+0{,}62$, $p=0{,}016$), сохраняется при
фиксированных эффектах региона по 48 регион-окнам ($p=0{,}046$) и устойчива к
исключению любого одного региона ($\rho$ от $+0{,}48$ до $+0{,}85$). Горизонт
предсказуемости $T_{50}$ также сокращается с ростом $A$ ($\rho=-0{,}51$). Но
на независимых 2021--2022~гг. связь с разбросом ослабевает до $\rho=+0{,}26$
($n=8$, незначимо), а расхождение знаков между мерами по ошибке и по разбросу
само по себе достаточно, по заранее зафиксированным критериям, чтобы
зарегистрировать результат как отрицательный.

**Исправленное прочтение.** В регионах с высокой асимметрией ошибка прогноза
не растёт быстрее --- она стартует выше. Относительная ошибка на 12~ч
меняется от 0,27 над Европой до 0,61 над Амазонией и коррелирует с $A$ при
$\rho=+0{,}48$, а в десяти из 48 регион-окон полоса 200--400~км насыщена уже
на минимальной доступной заблаговременности: там прогноз с самого начала не
несёт информации о размещении мезомасштабной изменчивости, и скорость роста
не является измеримой величиной. Измеряемая скорость роста соответственно
сжата там, где начальная ошибка велика ($\rho=-0{,}76$ между ними), чем и
объясняется найденный отрицательный знак.

Это не тот механизм, который предполагала гипотеза, и он ближе к трактовке
разд.~6.1. Большое $A$ описывает не режим, в котором информация теряется
быстрее по ходу прогноза, а режим, в котором размещение мезомасштаба уже
невосстановимо в момент анализа. Сформулированное ранее прикладное следствие
--- что регионы с большим $A$ суть те, где статистическое уточнение по
крупномасштабным полям наименее обосновано, --- этими числами
поддерживается; распространение того же утверждения на скорость роста ошибки
в ансамблевом прогнозе --- нет.

Заключение ограничено двумя обстоятельствами. При двенадцати регионах схема
обнаруживает лишь сильные связи ($|\rho|\geq0{,}50$ при $\alpha=0{,}05$,
односторонне), и это было заявлено до обращения к данным: отрицательный
результат здесь означает не малый эффект, а отсутствие большого. Кроме того,
связь с разбросом допускает конкурирующее объяснение, которое настоящий
материал исключить не может: начальные возмущения ансамблевой системы могут
сами по себе быть недостаточно дисперсными во влажных тропиках, и тогда $\mu$
сообщает о свойстве прогностической системы, а не атмосферы.

---

## Knock-on edits required elsewhere

- **Conclusions, point 5**: "a testable hypothesis relating the quantities
  obtained to regional limits of predictability has been formulated" must
  become "...has been formulated and tested: the index does not predict the
  rate of ensemble error growth, and the regional differences in that rate
  are governed by baroclinicity; what does follow the index is the level of
  mesoscale error already present at analysis time."
- **Sect. 6.1**: the downscaling paragraph can now cite a measured quantity
  (the 12 h relative error in the 200-800 km bands, 0.27 to 0.61 across the
  twelve regions) instead of arguing from the descriptor alone.
- **Data section**: add GEFS v12 and the AWS Open Data mirror.
- **Abstract**: if the abstract advertises the predictability hypothesis, it
  should now advertise the test and its negative outcome.
- **Figure**: `figB9_A_vs_predictability.png` (three panels: A against
  lambda, mu, T50) is publication-ready at 170 dpi and can be dropped in as
  the Sect. 6.2 figure.
