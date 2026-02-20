<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# Прекрасно. Теперь подытож все наши эксперименты в этой ветке общения их постановки и результаты (в таблицах) по примеру файлов Numerical_evidence в контексте этого пространства. Назовем Numerical_evidence_3

Собрал “Numerical_evidence_3” в стиле ваших `Numerical evidence` заметок: постановки + критерии + результаты в таблицах, и сохранил как `Numerical_evidence_3.md`.[^1]

## Numerical_evidence_3: постановки

Ниже — краткая карта именно **тех экспериментов, которые мы провели в этой ветке**, с акцентом на G2 и на варианте (iii) “transport + pointer из $F^{\text{phys}}$”.[^1]


| ID | Эксперимент | Постановка (коротко) | Что логируем | Проверка/критерий |
| :-- | :-- | :-- | :-- | :-- |
| NE3‑G2‑1 | Toy‑chain pointer‑from‑curvature | База $(x,k)$ с $L_x=3$, $K=6$, $SU(2)$ линк‑переменные, плакеты $\Rightarrow F^{\text{phys}}_{x\mu}$, pointer из агрегированного $F_{\text{tot}}(k)$, необратимые прыжки $k\to k+1$ + вертикальный transport $V_k$. | $dS^{hor}$ через $S_{hor}(k)=k\Delta S$, $dQ_{in}=-\langle\Delta K_\Omega\rangle$, $\Lambda_w,\Lambda_{unw}$, pointer‑когерентность. | Устойчивый наклон $1/T_{eff}$ и высокий $R^2$ в регрессии $dS^{hor}\sim (1/T)dQ_{in}$ по $\varepsilon$. |
| NE3‑G2‑2 | Final single‑qubit G2(μ) (exp5) | Один кубит $\mathbb C^2$, связь $A(\mu)=\beta I + r(\cos\phi(\mu)\sigma_y+\sin\phi(\mu)\sigma_x)$ с профилем $\omega(\mu)$, слои $k$ по $\mu$, $F^{phys}(k)$ из $(A_k,A_{k+1})$, pointer = eigbasis $F^{phys}(k)$, MCWF прыжки $k\to k+1$ + transport $U_k=e^{-iA_k d\mu}$. | $dS^{hor}$, $dQ_{in}$, $\Lambda_w$, pointer‑когерентность, rate прыжков; скан по профилям $\omega(\mu)$. | Таблица “profile $\to T_{eff},R^2,\langle\Lambda_w\rangle$” как решающая диагностика режима (квазиреверсивный vs неравновесный). |

## Numerical_evidence_3: результаты (таблицы)

### NE3‑G2‑1 (toy‑chain): регрессия по $\varepsilon$

Табличные результаты для фита $dS^{hor}\approx (1/T_{eff})\,dQ_{in}+b$ по амплитудам возмущения $\varepsilon$.


| eps | a_1_over_Teff_from_Shor | r2_Shor | b_Shor | mean_jump | mean_Lambda_w | mean_Coh_ptr_w |
| --: | --: | --: | --: | --: | --: | --: |
| 0.05 | 1.121753 | 0.861393 | 0.000651 | 0.00936 | -0.165427 | 0.883279 |
| 0.15 | 1.146075 | 0.877617 | 0.000456 | 0.00950 | -0.158832 | 1.002393 |
| 0.30 | 1.203128 | 0.861686 | 0.000324 | 0.00910 | -0.137219 | 1.184369 |
| 0.60 | 1.080912 | 0.863213 | 0.000616 | 0.00924 | -0.060245 | 1.401215 |

Глобальный фит (все $\varepsilon$ вместе) для $dS^{hor}$ даёт $1/T_{eff}\approx 1.132789$ и $R^2\approx 0.865313$.


| a_1_over_Teff_global_from_Shor | b_global_Shor | r2_global_Shor | mean_jump | mean_Lambda_w |
| --: | --: | --: | --: | --: |
| 1.132789 | 0.000531 | 0.865313 | 0.0093 | -0.130431 |

### NE3‑G2‑2 (single‑qubit exp5): скан профилей $\omega(\mu)$

Сводка “как в Theory_finale”: одинаковые $r,\beta,\Delta_\mu,K$ и разные профили $\omega(\mu)$.


| profile | Teff | inv_Teff | R2 | mean_Lambda_w | mean_Coh_ptr_w | mean_jump_rate |
| :-- | --: | --: | --: | --: | --: | --: |
| constant | 0.030730 | 32.541586 | 0.995764 | 0.022616 | 0.000442 | 0.003325 |
| gaussian | 0.010164 | 98.391213 | 0.898253 | 0.000994 | 0.000019 | 0.003325 |
| oscillating | 0.009453 | 105.789705 | 0.220476 | 0.022616 | 0.000442 | 0.003325 |

## Артефакты (для статьи)

Черновик текста “Numerical_evidence_3” сохранён в `Numerical_evidence_3.md`.
Данные и логи: `NE3_G2_toychain_timeseries.csv`, `NE3_G2_toychain_fit_by_eps.csv`, `NE3_G2_toychain_fit_global.csv`, а также профили single‑qubit: `NE3_G2_qubit_profiles_summary.csv` и по‑профильные timeseries/fit‑CSV.

Если хочешь, завтра можно сделать ровно один “последний штрих” перед статьёй: добавить в `Numerical_evidence_3.md` короткий абзац-интерпретацию, почему у `oscillating` низкий $R^2$ в этой реализации (то есть явное отделение “квазиреверсивного” и “неравновесного” режимов G2).
<span style="display:none">[^2]</span>

<div align="center">⁂</div>

[^1]: Numerical-evidence-experiments.md

[^2]: Theory_finale.txt

