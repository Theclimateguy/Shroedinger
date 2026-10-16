# Applied landscape for an "inheritance fidelity" forecast-verification diagnostic: competing metrics, users, gaps

Scope reminder: the diagnostic is the correlation between amplitude envelopes of adjacent scale bands (200–400 / 400–800 / 800–1600 km, 850 hPa vorticity), forecast minus reanalysis at matched valid times, one number per system and lead time. The result to place in context: conditional-mean-like forecasts (MSE-trained GraphCast, Pangu, Aurora, and the IFS ensemble mean) over-couple scales by 24–35 % at day 5. Single realisations (HRES, one ENS member, one GenCast member) do not. Matching the power spectrum does not reproduce the effect, and the diagnostic has no reliable geographic map.

Verification tags used below: **[V]** means the bibliographic details were checked against the publisher, arXiv or PMLR page in this session. **[M]** means the details come from memory or secondary citation and were not re-checked in this session.

---

## Q1. Competing and complementary verification metrics: what each detects, and whether it can tell a uniform loss of fine variance from a selective, coarse-locked one

### Takeaway
Every metric in routine use for ML-weather evaluation does one of three things:
- measures scale-by-scale amplitude (spectra, effective resolution, sharpness, SpecDiv);
- measures marginal or pointwise skill and calibration (RMSE/ACC, CRPS, spread-skill, rank histograms, IQD);
- measures neighbourhood or object agreement for thresholded fields, mostly precipitation (FSS, SAL, MODE, pooled CRPS).

None of these is a cross-scale statistic. So none can, by construction, separate "fine variance reduced everywhere" from "fine variance kept only where the coarse envelope is large". Two families could in principle:
- local wavelet spectra (Buschow & Friederichs), whose output contains the needed information but whose published scores average it away;
- generic learned or high-order scores (signature-kernel scores, discriminator "critics", KE-transfer diagnostics). These are either uninterpretable or measure dynamical transfer rather than static organisation.

### Cited Findings

**Power and energy spectra, spectral ratios, effective resolution**
- WeatherBench 2 provides energy spectra as a diagnostic, alongside headline RMSE/ACC/SEEPS (deterministic) and CRPS/spread-skill (probabilistic) [V] — [Rasp et al. 2024, JAMES 16, e2023MS004019](https://doi.org/10.1029/2023ms004019); [metric list in code](https://github.com/google-research/weatherbench2/blob/main/weatherbench2/metrics.py)
- WB2 also ships EnergyScore and RankHistogram metrics, plus derived variables such as vorticity, divergence, eddy kinetic energy and geostrophic/ageostrophic wind. It has no cross-scale or texture metric [V] — [WB2 API docs](https://weatherbench2.readthedocs.io/en/latest/api.html)
- Bonavita (2024) found that the spectra of ML forecasts (Pangu, FourCastNet, GraphCast) differ from ERA5 and IFS. Scales below about 300–400 km are not properly represented, and balance relations are violated [V] — [Bonavita 2024, GRL 51, e2023GL107377](https://doi.org/10.1029/2023gl107377)
  - Bonavita also noted that ML spectra do not match ensemble-mean spectra either: ML forecasts lack the ensemble mean's signature energy drop at synoptic scales at 3–5 days, while having reduced small-scale variability. Spectra therefore cannot place ML forecasts and the ensemble mean in the same class. This is directly relevant, because the inheritance diagnostic does put GraphCast/Pangu/Aurora and the IFS ensemble mean in one class.
- Effective resolution is defined as the smallest scale reproduced with realistic amplitude. Selz et al. argue that the training lead-time interval controls it, and that smooth forecasts score better by ignoring small scales [V, preprint] — [Selz, Bruinsma, Craig, Markou, Turner, Vaughan 2025, ESS Open Archive, doi:10.22541/essoar.174139239.94807670/v1](https://essopenarchive.org/doi/pdf/10.22541/essoar.174139239.94807670/v1?onload=true)
- Rodwell et al. (2025) use spherical-harmonic spectra of spread and error to compare IFS ENS with AIFS-CRPS and AIFS-Diffusion. At n ≥ 70 (≲500 km) the AIFS ensembles become over-dispersive, and ensemble-mean error exceeds its theoretical maximum. The authors call this "curious" small-scale behaviour but do not diagnose it beyond second-order spectra [V] — [Rodwell, Clare, Lock, Lonitz, Chevallier 2025, Met. Apps 32(5), e70071](https://doi.org/10.1002/met.70071)
- ChaosBench introduces "physics-based" SpecDiv (a KL-type divergence of normalised high-wavenumber power spectra) and SpecRes (RMSE of spectra). Both are purely amplitude-spectral [V] — [Nathaniel et al. 2024, arXiv:2402.00712](https://arxiv.org/abs/2402.00712); [ChaosBench metrics page](https://leap-stc.github.io/ChaosBench/metrics.html). The NeurIPS 2024 D&B venue is [M].
- Ebert-Uphoff et al. (2025) review sharpness metrics: gradient/total-variation, spectral and wavelet-TV. They propose "Gaussian Blur Equivalence" and show that gradient- and spectral-based metrics disagree on the same image. All measure blur amplitude, not cross-scale organisation [V] — [Ebert-Uphoff et al. 2025, AIES 4(3), e240083, doi:10.1175/AIES-D-24-0083.1](https://doi.org/10.1175/aies-d-24-0083.1)

**Spectral and dynamical "physics" diagnostics closest in spirit**
- Chen, Oskarsson, Driscoll & Schemm (2026) analyse KE spectra, difference-KE spectra and spectral KE transfer across scales for NeuralGCM-ENS, FourCastNet 3, AIFS-ENS and GenCast against IFS-ENS:
  - AIFS-ENS, GenCast and FourCastNet 3 have realistic KE magnitudes but do not capture the expected upscale transfer;
  - AIFS-ENS and GenCast pile up KE at high wavenumbers;
  - all four under-represent rapid early small-scale spread growth.

  This is the nearest competitor that explicitly addresses scale coupling. It measures energy flux (third-order, dynamical) from winds, not the static amplitude modulation of fine scales by coarse scales [V, preprint] — [arXiv:2609.18489](https://arxiv.org/abs/2609.18489)
- PhysMetrics.Weather (2026) is a framework with conservation, spectral and dynamical metrics for ML weather models. It contains no cross-scale texture/organisation metric, as far as the abstract shows [V, preprint] — [Kasteleyn et al., arXiv:2606.10642](https://arxiv.org/html/2606.10642)
- Selz & Craig (2023): Pangu fails to reproduce rapid small-scale error growth (no butterfly effect). Error growth rate and structure stay synoptic-like regardless of perturbation amplitude [V] — [Selz & Craig 2023, GRL 50, e2023GL105747](https://doi.org/10.1029/2023gl105747)

**Neighbourhood, object and structure scores (designed for precipitation)**
- FSS (Roberts & Lean 2008), SAL (Wernli et al. 2008), MODE (Davis et al. 2006) and the wider intercomparison (Gilleland et al. 2009) are summarised in the wavelet-verification literature. Their purpose is to avoid the double penalty for displaced, thresholded features [V for summary] — [Buschow & Friederichs 2020, ASCMO 6, 13–30](https://ascmo.copernicus.org/articles/6/13/2020/ascmo-6-13-2020.html). Exact citations are [M]:
  - Roberts & Lean 2008, MWR 136, 78–97, doi:10.1175/2007MWR2123.1
  - Wernli et al. 2008, MWR 136, 4470–4487, doi:10.1175/2008MWR2415.1
  - Davis, Brown & Bullock 2006, MWR 134, 1772–1784
  - Gilleland et al. 2009, WAF 24, 1416–1430
- Wavelet-based verification:
  - Buschow et al. (2019) use a redundant discrete wavelet transform. Under local stationarity, its output links to local spatial covariances, giving a local wavelet spectrum at every grid point and scale [V] — [Buschow, Pidstrigach & Friederichs 2019, GMD 12, 3401–3418 (wv_verif)](https://gmd.copernicus.org/articles/12/3401/2019/gmd-12-3401-2019.pdf)
  - The scores either average local spectra in space or compare maps of dominant scale. The SAD extension adds scale, anisotropy and direction [V] — [Buschow & Friederichs 2021, QJRMS, doi:10.1002/qj.3964](https://doi.org/10.1002/qj.3964)
  - Casati et al. (2004) intensity-scale: Met. Apps 11, 141–154. Casati (2010) intensity-scale developments: WAF 25, 113–143 [M]
- GenCast's spatially pooled CRPS (average- and max-pooled over 120–3,828 km regions) is described by its authors as neighbourhood verification adapted to the sphere, used "to evaluate skill at forecasting spatial structure" [V] — [Price et al. 2025, Nature 637, 84–90 (online Dec 2024), doi:10.1038/s41586-024-08252-9](https://www.nature.com/articles/s41586-024-08252-9). Volume/pages are [M]; DOI and content are [V].

**Probabilistic and distributional scores**
- CRPS (fair and almost-fair), spread-skill and rank histograms are standard in WB2 and in AI-ensemble papers. All are marginal, per grid point [V] — [WB2 metrics.py](https://github.com/google-research/weatherbench2/blob/main/weatherbench2/metrics.py); [AIFS-CRPS, Lang et al. 2024, arXiv:2412.15832](https://arxiv.org/abs/2412.15832). Gneiting & Raftery 2007, JASA 102, 359–378 is [M].
- The Integrated Quadratic Distance is the score divergence of the CRPS. It compares marginal CDFs of model and observed climatologies, so it is a distributional, not spatial, tool [V] — [Thorarinsdottir, Gneiting & Gissibl 2013, SIAM/ASA JUQ 1, doi:10.1137/130907550](https://doi.org/10.1137/130907550)

**Generic "realism" scores**
- Ben-Bouallègue (2026, ECMWF) proposes three realism types:
  - functional realism (scoring functions);
  - structural realism (statistical characteristics such as bias, activity, spectra, effective resolution, sharpness);
  - physical realism (falsification tests, conservation, expert inspection).

  He states that conventional scoring rules are insensitive to physical realism and that "hallucinations usually go unnoticed". The paper does not discuss cross-scale coupling [V] — [Ben-Bouallègue 2026, IOP journal (ISSN 2515-7620, i.e. Environmental Research Communications [M]), doi:10.1088/2515-7620/ae9f1f](https://iopscience.iop.org/article/10.1088/2515-7620/ae9f1f); [arXiv:2602.00622](https://arxiv.org/html/2602.00622v2)
- Elberkennou et al. (2026) argue that "every fixed metric has a blind spot", because each fixes a representation in advance. They train a discriminator whose logit serves as a realism score. Realism degrades with lead time, and NWP scores as more realistic than ML [V, preprint] — [arXiv:2609.18381](https://arxiv.org/abs/2609.18381)
- A signature-kernel scoring rule has been proposed as a spatio-temporal diagnostic for probabilistic ML weather forecasts. It targets higher-order path and dependence structure [V title/abstract only, from alphaXiv listing] — [arXiv:2510.19110 (Univ. Warwick)](https://www.alphaxiv.org/abs/2510.19110)

### Inferences
- **Order of the statistic is the crux.** Spectra, spectral ratios, effective resolution, SpecDiv/SpecRes, CRPS, the energy score and IQD are second-order or marginal. A correlation of band envelopes is a fourth-order statistic: the correlation of squared band amplitudes across space. Two fields can share every band variance, and so the spectrum, while differing in where the fine variance sits relative to the coarse envelope. This explains why spectrum matching did not reproduce the effect.
  - It is a theoretical guarantee, not a contingent finding, and should be stated as such.
  - The same blindness applies to Rodwell-style spread/error spectra and to Subich's AMSE criterion.
- **Selective vs uniform loss.** The possible ratings for telling a uniform loss of fine variance from a selective, coarse-locked loss:
  - **No:** spectra and effective resolution; sharpness/GBE; SpecDiv; CRPS; spread-skill; rank histograms; IQD; RMSE/ACC.
  - **Indirectly or weakly:** FSS and SAL-S on thresholded fields, because selective retention concentrates exceedances under coarse maxima and changes object peakedness. Max-pooled CRPS at a given pool size likewise depends on where extremes are placed.
  - **In principle yes, not as published:** local-wavelet-spectrum methods (Buschow et al.). Correlating local RDWT energy at scale j with scale j+1 across grid points is essentially the inheritance diagnostic. The diagnostic should therefore be framed as a cross-scale summary of a known representation, not as a new representation.
  - **In principle yes, uninterpretable:** learned critics and signature-kernel scores.
  - **Related but different:** KE-transfer spectra, which measure flux, not static modulation.
- **Distinct niche.** The one thing the published literature does not do is turn the cross-scale correlation of local amplitudes into a single number per system and lead time, verified against truth, that classifies "conditional-mean-like vs realisation-like". Bonavita's observation that ML spectra and ensemble-mean spectra look different supports the claim that the envelope statistic reveals a shared mechanism (averaging) that spectra hide.

### Gaps
- No paper was found that tests whether FSS, SAL or MODE respond to a controlled "coarse-locked" fine-variance perturbation at fixed spectrum. A synthetic-null experiment would settle this.
- WeatherBench 2's paper discusses blurring and spectra, but I did not open the full WB2 text to quote its caveats.
- The exact ECMWF headline scorecard composition (which parameters and scores) was not retrieved.
- The signature-kernel paper (2510.19110) and the "dynamical fidelity through predictive structure" preprint (arXiv:2609.34627, DTU/Cambridge/Exeter, seen only as an alphaXiv listing) were not read in full.

---

## Q2. Can multivariate and spatial scoring rules detect wrong spatial correlation or texture?

### Takeaway
There is a well-established result that the energy score is weak at detecting misspecified dependence. The variogram score (p = 0.5) is the standard fix. A 2026 study shows the failure mode bites for learned emulators: good marginals, joint spatial structure worse than climatology, missed by the energy score and caught by the variogram score.

These scores target pairwise (second-order-like) dependence. They need an ensemble, and they are not designed to tell whether fine-scale amplitude is modulated by coarse amplitude. The inheritance diagnostic works on single deterministic fields and targets exactly that modulation.

### Cited Findings
- Scheuerer & Hamill (2015) proposed variogram-based proper scores. They argued that the energy score detects incorrectly specified correlations only weakly, and showed the variogram scores are distinctly more discriminative for correlation structure. Recommended settings are p = 0.5 with unit weights [V existence, NOAA repository; details from search summary] — [NOAA IR record](https://repository.library.noaa.gov/view/noaa/22327). MWR 143, 1321–1334, doi:10.1175/MWR-D-14-00269.1 is [M].
- The variogram score is proper but not strictly proper. It is insensitive to simple biases, so it should be paired with a bias-sensitive score [V via search summary of multiple sources] — [Alexander et al. arXiv:2101.12693](https://arxiv.org/pdf/2101.12693); [Ziel & Berk 2019, arXiv:1910.07325](https://arxiv.org/pdf/1910.07325)
- Alexander, Coulon, Han & Meng (2022) ran a simulation study across realistic DGPs with differing dependence structures. The energy score generally has poorer discrimination, and the variogram score with p = 0.5 has the lowest error rate. They cite Pinson & Tastu (2013) for the energy score's lack of sensitivity to correlation errors [V] — [Annals of Operations Research, doi:10.1007/s10479-022-04611-9](https://doi.org/10.1007/s10479-022-04611-9). Pinson & Tastu 2013 (DTU technical report, "Discrimination ability of the energy score") is [M].
- Howard & Barnes (2026) built a diffusion model for subseasonal sea level at 8 tide gauges:
  - marginal skill is positive everywhere, but joint spatial structure is worse than climatological draws;
  - a shuffle-based permutation decomposition shows the failure is missed by the energy score and caught by the variogram score;
  - the failure persists from 0.7 to 170 equivalent years of training data in Lorenz-96 and is reproduced by a linear baseline;
  - dynamical ensembles do not fail, while deterministic emulators do.

  [V, preprint] — [arXiv:2608.08954](https://arxiv.org/abs/2608.08954)
- Allen et al. developed weighted (threshold- and region-weighted) multivariate verification tools for high-impact events, including kernel-score generalisations [V arXiv existence] — [arXiv:2209.04872](https://arxiv.org/pdf/2209.04872). The journal venue is [M].
- Lang, Leutbecher & Hatfield (2026, ECMWF) trained AIFS-CRPS with:
  - the almost-fair CRPS;
  - a fair global energy score;
  - a graph energy score on node neighbourhoods.

  Skill is broadly similar (the graph energy score is somewhat better in the tropics; the global energy score is slightly degraded). On spectra, "any form of explicit scale-awareness improves realism", which they attribute to differing effective weights per scale [V, preprint] — [arXiv:2607.19161](https://arxiv.org/abs/2607.19161)
- FGN is trained only on per-location CRPS (marginals), with learned functional perturbations, yet is reported to yield realistic joint spatial structure [V] — [Alet et al. 2025, arXiv:2506.10772](https://arxiv.org/abs/2506.10772); restated in the [WeatherNext 3 report, arXiv:2609.03582](https://arxiv.org/pdf/2609.03582)

### Inferences
- The multivariate-scoring literature is about whether the ensemble's joint distribution has the right dependence between locations. The inheritance diagnostic asks something adjacent but different: within one field, is the local amplitude at fine scale too predictable from local amplitude at coarse scale? A forecast can have the right pairwise covariance (variogram) at each scale while having too strong amplitude modulation across scales, because covariance is second-order and modulation is fourth-order. The variogram score with p = 0.5 is not strictly second-order, so it may carry some sensitivity, but it has never been shown to target modulation.
- Howard & Barnes (2026) are a useful ally. They show empirically that learned emulators can have "good marginals, bad joint structure" and that the standard multivariate score (energy score) is blind to it. That motivates targeted structure diagnostics, of which inheritance fidelity is one.
- Caution: FGN, GenCast and AIFS-CRPS are claimed to have realistic joint structure according to pooled CRPS and spectra. The finding that one GenCast member does not over-couple is consistent with these claims. The diagnostic adds discriminating power mainly for deterministic and mean-like products, and for sharpened deterministic products, not for well-built generative ensembles.

### Gaps
- I did not find any study that applies the variogram score or energy score to global ML-weather fields specifically to test texture or cross-scale modulation.
- I did not verify whether the Pinson & Tastu (2013) report has a DOI.
- I found no paper that directly compares a fourth-order (amplitude-modulation) statistic with variogram or energy scores.

---

## Q3. ML model development: spectral and CRPS losses, diffusion, sharpening, and how developers check that fine structure is realistic rather than copied

### Takeaway
Developers check fine-scale realism almost exclusively with:
- power spectra and effective resolution;
- spatially pooled CRPS, regional wind power and tropical-cyclone tracks/intensity as joint-structure proxies;
- visual case studies.

Newer work (2026) adds KE-transfer spectra, learned critics and scale-aware training scores. Explicit discussion of "hallucinated" small scales exists (ECMWF realism framework). I found no published test of whether sharpened or generated fine scales are copied or modulated from the large-scale field. That is the most defensible niche for the inheritance diagnostic, especially for AMSE-type sharpened deterministic models whose spectra are, by design, correct.

### Cited Findings
- **AMSE (Subich et al. 2025).** MSE decomposes per total wavenumber into PSD and coherence terms. MSE is minimised when the amplitude ratio sqrt(PSD_x/PSD_y) equals the coherence, which causes the smoothing. AMSE separates "decorrelation" loss from "spectral amplitude" loss and drives PSD_x toward PSD_y regardless of predictability. Fine-tuning GraphCast with AMSE:
  - raises effective resolution from 1,250 km to 160 km;
  - improves lagged-ensemble CRPS, TC intensity and surface wind extremes.

  For long leads the AMSE model has more small-scale energy than the truth, giving a "noise-based" effective resolution. The authors also argue that WB2-style verification "may conflate the effects of more-optimal smoothing with forecast skill" [V] — [Subich, Husain, Separovic & Yang 2025, ICML, PMLR 267, 57191–57211](https://proceedings.mlr.press/v267/subich25a.html); [arXiv:2501.19374](https://arxiv.org/abs/2501.19374)
- **AMSE caveat on modulation.** AMSE constrains only PSD and spectral coherence with truth. It places no constraint on how the restored small-scale amplitude is distributed in space relative to the large scales. This is inferred from the AMSE formula, which uses only PSD_k and Coh_k summed over zonal wavenumbers (same source).
- **AIFS-CRPS.** It is trained with an almost-fair CRPS, with noise injected into the processor. It outperforms IFS ENS for most variables and leads (Feb–Sep 2024 evaluation) [V] — [Lang et al. 2024, arXiv:2412.15832](https://arxiv.org/abs/2412.15832). Its small-scale over-dispersion and drift are shown by power spectra [V] — [Rodwell et al. 2025](https://doi.org/10.1002/met.70071)
- **GenCast.** The authors assert members are "sharp individual weather trajectories with realistic power spectra, rather than ... conditional means". Joint structure is checked with average/max-pooled CRPS (GenCast beats ENS on 98.1 % / 97.6 % of 5,400 pooled targets), regional wind power summed over 5,344 wind farms (about 20 % CRPS gain to day 2), and TC tracks [V] — [Price et al., Nature, doi:10.1038/s41586-024-08252-9](https://www.nature.com/articles/s41586-024-08252-9)
- **NeuralGCM.** It uses "a mix of filtered and spectral loss functions" (per Subich et al.'s characterisation) [V as characterisation] — [Subich et al. 2025](https://arxiv.org/html/2501.19374). Kochkov et al. 2024, Nature 632, 1060–1066 is [M].
- **Mosaic (2026).** It names three spectral-degradation failure modes: spectral damping (statistical), high-frequency aliasing (architectural), and residual high-frequency leakage (parametric). Success is claimed as "near-perfect spectral alignment" of individual members, i.e. still spectral evaluation [V, preprint] — [Zhdanov, Lucic, Welling & van de Meent, arXiv:2604.16429](https://arxiv.org/html/2604.16429v3)
- **ArchesWeatherGen** is a generative/diffusion model built on a deterministic ArchesWeather model (Inria/ECMWF) [V title from alphaXiv listing] — [arXiv:2412.12971](https://www.alphaxiv.org/abs/2412.12971). Details of its structure evaluation were not read.
- **Probabilistic ML ensembles and KE transfer.** Probabilistic MLWP models can be skilful while misrepresenting inter-scale KE transfer. Spatially uncorrelated stochastic perturbations (AIFS-ENS, GenCast) accumulate excess high-wavenumber KE [V, preprint] — [Chen et al. 2026, arXiv:2609.18489](https://arxiv.org/abs/2609.18489)
- **Hallucinations.** Ben-Bouallègue (2026) names generative-model "hallucinations" as features that look statistically plausible but contradict physics. He says they "usually go unnoticed with common verification practices" and that detecting them requires new tools [V] — [arXiv:2602.00622](https://arxiv.org/html/2602.00622v2), citing Rathkopf 2025, arXiv:2504.08526 [V as citation]
- **Data-driven forecasts as conditional means.** Ben-Bouallègue states that MSE-trained data-driven forecasts are "to some extent" equivalent to a conditional mean up to a certain range [V] — same source. Selz et al. note that the expected relation to an ensemble mean is hard to see in practice, citing Bonavita [V] — [Selz et al. 2025](https://essopenarchive.org/doi/pdf/10.22541/essoar.174139239.94807670/v1?onload=true)
- **Learned critic.** A learned discriminator is proposed precisely because fixed metrics miss artefacts such as blurring, periodic irregularities and unphysical spatial structure [V, preprint] — [arXiv:2609.18381](https://arxiv.org/abs/2609.18381)
- **Mainstream ML models.** Lam et al. 2023 (GraphCast) is Science 382, 1416–1421. Bi et al. 2023 (Pangu) is Nature 619, 533–538. Bodnar et al. 2025 (Aurora) is Nature 641. Lang et al. 2024 (AIFS) is arXiv:2406.01465. All [M].

### Inferences
- **Literature support for the finding.** The literature supports the theoretical picture behind the diagnostic. MSE training gives the conditional mean (Subich, Ben-Bouallègue). Bonavita suspected but could not show with spectra that ML forecasts behave like an ensemble mean. The inheritance diagnostic supplies an operational test of "conditional-mean-likeness" that does not rely on the spectrum. That is its cleanest selling point.
- **Highest-value application: sharpening without real sampling.** AMSE fine-tuning, spectral-loss terms, post-hoc spectral nudging and other "sharpening" all restore spectral amplitude. If restored fine variance is placed where the conditional mean already had it (i.e. under large coarse envelopes), spectra will look perfect but inheritance will stay excessive. The prediction is testable on Subich's AMSE-GraphCast. A null result there, with AMSE fixing both spectrum and over-coupling, would also be informative and should be anticipated.
- **Generative ensembles.** For FGN, GenCast, AIFS-CRPS and Mosaic, the developers' own pooled CRPS and spectra already say members are realistic. The project's finding (a GenCast member does not over-couple) agrees. The added value there is modest: it confirms rather than discovers.
- **"Copied" small scales.** No source was found that explicitly discusses generative fine scales being "copied" from coarse conditioning. The concept is implicit in residual-diffusion designs (next question), but untested.

### Gaps
- Whether any developer has computed cross-scale amplitude correlations (e.g. turbulence-style amplitude-modulation coefficients) on ML weather fields was not found. This absence is not proof of non-existence; a targeted search on "amplitude modulation coefficient" plus "weather forecast" is recommended.
- FGN's own structure-evaluation methods (pooled CRPS? spectra?) were not read in full.
- The AIFS documentation's evaluation section was not consulted.
- Brenowitz et al. ("practical probabilistic benchmark for AI weather models", lagged ensembles) is [M], not checked.

---

## Q4. Downscaling and nowcasting: how generative realism is evaluated, and the relevance of RainFARM-type stochastic downscaling

### Takeaway
Generative downscaling papers evaluate realism with:
- radially averaged power spectra, or log-spectral distance on them;
- pixel-wise and pooled CRPS;
- rank histograms and spread-skill;
- FSS;
- PDF tails;
- case studies.

None tests whether generated fine variance is statistically locked to the coarse input. RainFARM is the textbook case of fine variance explicitly tied to the coarse field: it rescales the synthetic fine field box-by-box by the coarse value. It is therefore a natural maximal-inheritance reference. However, the diagnostic's current band range (200–1600 km) is far coarser than downscaling target scales (1–25 km), so transfer is not automatic.

### Cited Findings
- **CorrDiff.** A UNet first predicts the conditional mean, and a diffusion model then generates the residual (25 km to 2 km over Taiwan). Evaluation used:
  - MAE/CRPS;
  - power spectra of KE, T2m and radar reflectivity;
  - log-PDFs, spread-skill and rank histograms;
  - frontal and typhoon case studies.

  The ensembles are under-dispersive, and "Calibration of model uncertainty remains challenging" [V] — [Mardani et al. 2025, Communications Earth & Environment, doi:10.1038/s43247-025-02042-5](https://www.nature.com/articles/s43247-025-02042-5). The volume/article number (6, 124) is [M].
- **GAN and VAE-GAN downscaling (Harris et al. 2022).** IFS precipitation is downscaled to radar resolution (×10). Evaluation used pixel-wise and pooled CRPS, RALSD (radially averaged log-spectral distance), rank histograms, FSS, ROC and heavy-rain cases. The authors found little correlation between Leinonen's full-2D log-spectral distance and good predictions, so they switched to the radially averaged version. RainFARM is a baseline and is "missing a lot of detail" below about 100 km [V] — [Harris, McRae, Chantry, Dueben & Palmer 2022, JAMES 14(10), e2022MS003120](https://doi.org/10.1029/2022ms003120)
- **Leinonen et al.** Stochastic super-resolution GAN for time-evolving fields, IEEE TGRS, doi:10.1109/TGRS.2020.3032790 [V DOI via Harris citation; volume/pages M]
- **Ravuri et al. 2021 (DGMR nowcasting).** Nature 597, 672–677. It used radially averaged PSD, pooled CSI and an expert-forecaster preference study [M]
- **Price & Rasp 2022.** "Increasing the accuracy and resolution of precipitation forecasts using deep generative models", AISTATS [M]
- **RainFARM.** It estimates the coarse field's space-time spectrum and extrapolates it as a power law, generates a Gaussian field, and applies exp(g). It then multiplies the fine field in each coarse box by the ratio P/R̃ (coarse observed over coarse synthetic), so the result "behaves exactly as the original field P" when aggregated. Stochasticity is "manifest only on small scales". The method is stated to be meaningful only at about 1–50 km [V] — [Rebora, Ferraris, von Hardenberg & Provenzale 2006, J. Hydrometeorol. 7(4), doi:10.1175/JHM517.1](https://doi.org/10.1175/jhm517.1)
- Spatially enhanced loss functions (neighbourhood, Fourier or wavelet filters) have been tested for NN training [V] — [Lagerquist & Ebert-Uphoff, "Can we integrate spatial verification methods into neural-network loss functions…", arXiv:2203.11141](https://arxiv.org/pdf/2203.11141v2). Author list and venue are [M].
- Threshold-weighted, spatial verification of extreme precipitation from an AI model vs high-resolution NWP has been done. Its authors note most AIWP benchmarking is point-to-point (except Radford et al. 2025a, object-based) [V, preprint] — [arXiv:2510.25045](https://arxiv.org/html/2510.25045)

### Inferences
- **RainFARM as a positive control.** Because RainFARM multiplies fine-scale fluctuations by the coarse value box-by-box, local fine amplitude is proportional to coarse intensity. That is maximal inheritance by construction.
  - For a downscaling benchmark, RainFARM could serve as the "over-inherited" reference and a pure-noise or phase-randomised field as the "under-inherited" one, with truth (radar or km-scale model) in between.
  - This would give inheritance fidelity a concrete calibration scale in an applied setting.
- **CorrDiff-type residual diffusion is a second target.** The regression mean already carries the coarse-locked part, and the diffusion residual adds stochastic detail. Whether the residual's amplitude is too tightly modulated by the mean (copied texture) is exactly the question the diagnostic asks. Spectra and rank histograms, which CorrDiff reports, cannot answer it.
- **Candid limitation.** The validated configuration is 200–1600 km bands of 850 hPa vorticity against ERA5. Downscaling and nowcasting operate at 1–25 km, on precipitation, reflectivity, winds and temperature. Moving there requires:
  - re-validating band choices;
  - handling intermittent, non-Gaussian fields (precipitation zeros);
  - finding truth with adequate effective resolution.

  Until done, claims of relevance to downscaling are conjecture.

### Gaps
- No downscaling paper was found that evaluates cross-scale amplitude modulation or "conditional texture"; this is the gap the diagnostic could fill.
- Volume, pages and article numbers for CorrDiff, Leinonen 2021 and DGMR were not re-verified.
- I did not find a published RainFARM vs GAN comparison using anything beyond spectra, CRPS and FSS.

---

## Q5. Sector use cases where fine-structure organisation matters, plus ML ensembles, data assimilation and emulators

### Takeaway
The most concrete applied harms of conditional-mean-like structure are documented as:
- intensity low bias (tropical cyclones);
- poor small-scale error growth (ensembles, the butterfly effect);
- wrong background covariances in data assimilation.

Each already has a direct, user-relevant metric (TC intensity error; spread-skill and DKE spectra; DA cycling stability and analysis error). Inheritance fidelity is a diagnostic upstream of these, not a replacement. Its plausible applied value is as a cheap screen that flags "this product is mean-like even though its spectrum looks fine". It may also be a candidate explanatory variable for sector failures, but that link is not yet demonstrated.

### Cited Findings
- **Tropical cyclones (DeMaria et al. 2025).** FourCastNet v1/v2, GraphCast and Pangu have track errors comparable to the best operational models but intensity errors worse than climatology-persistence. The low bias grows fast in the first 24–48 h. An idealised Rankine vortex shows that minimising RMS under track error favours weakening vortices [V] — [DeMaria et al. 2025, AIES, doi:10.1175/AIES-D-24-0085.1](https://doi.org/10.1175/aies-d-24-0085.1); [arXiv:2409.06735](https://arxiv.org/abs/2409.06735)
- **AMSE and TCs.** AMSE fine-tuning "fixes the weak intensity bias in the prediction of tropical cyclones" in GraphCast [V] — [Subich et al. 2025](https://proceedings.mlr.press/v267/subich25a.html)
- **Wind power.** GenCast's regional wind-power CRPS (summed over wind farms in 120–480 km pools) is about 20 % better than ENS to 2 days, and significant to 7 days. GenCast-Perturbed (a perturbed deterministic model) gains much less [V] — [Price et al. 2025](https://www.nature.com/articles/s41586-024-08252-9)
- **Data assimilation with ML backgrounds (Slivinski et al.).** In EnKF cycling of real surface-pressure observations:
  - deterministic ML models (GraphCast, SFNO, Pangu) accumulate small-scale noise and diverge;
  - spectral filtering stabilises them, but errors stay about 3× UFS;
  - they "do not accurately estimate background ensemble covariances", and the vertical correlation in ML models is too strong and extends too high;
  - training for the best RMSE "does not necessarily yield accurate estimation of spatial and inter-variable correlations";
  - NeuralGCM-stochastic correlations did not degrade.

  [V, preprint; full author list M] — [arXiv:2412.18016](https://arxiv.org/pdf/2412.18016)
- **GraphCast TL/AD models.** They show unphysical responses that "could undermine" 4D-Var and lead to inaccurate error covariances [V abstract] — [Tian et al., AMS 2025 abstract](https://ui.adsabs.harvard.edu/abs/2025AMS...10547813T/abstract)
- **Ensemble perturbations in DA.** Conversely, GraphCast can evolve ensemble perturbations stably in month-long JEDI cycling, with realistic spread (NCAR poster) [V, poster] — [Toedtli et al., ECMWF workshop poster](https://events.ecmwf.int/event/488/contributions/5740/attachments/3576/6015/Poster_Toedtli.pdf)
- **Butterfly effect and big ensembles.** Selz & Craig: the butterfly-effect failure means large cheap ML ensembles "only make sense if the error growth properties of the model are realistic" [V] — [Selz & Craig 2023](https://doi.org/10.1029/2023gl105747)
- **Extreme precipitation.** Evaluating AI precipitation extremes with threshold-weighted spatial methods is an active area [V, preprint] — [arXiv:2510.25045](https://arxiv.org/html/2510.25045)
- **Emulator ensembles.** Learned emulators can show good marginals and bad joint structure in subseasonal sea-level forecasting, relevant to coastal flooding [V, preprint] — [Howard & Barnes 2026, arXiv:2608.08954](https://arxiv.org/abs/2608.08954)
- **S2S.** ChaosBench ties high spectral divergence to "blurring" that limits S2S extreme-event usefulness [V] — [Nathaniel et al. 2024](https://arxiv.org/html/2402.00712)

### Inferences
- **DA is the most mechanistically aligned use case.** A background that is too coherent (cross-variable and vertical correlation too strong, per Slivinski) is the same family of defect as a field whose fine scales are too coherent with coarse scales.
  - An ML background with over-inherited fine structure would encode spurious scale-coupled covariances if used in hybrid or ensemble B estimation.
  - This is a hypothesis, not a demonstrated link. It would need an experiment correlating inheritance excess with analysis-increment statistics.
- **TC intensity.** The over-coupling at 200–1600 km is too coarse to resolve TC inner cores (tens of km). Its relevance is to vortex-environment structure, not the intensity metric itself. DeMaria's Rankine-vortex argument already explains the bias via the double penalty. Inheritance fidelity adds little here beyond confirming mean-likeness.
- **Wind power ramps and convective initiation.** These depend on 1–100 km structure and timing. With the current 200–1600 km bands, the diagnostic cannot speak to them directly. Relevance would require the downscaling-scale extension in Q4.
- **Fire weather.** No specific literature was gathered. Fire-weather indices depend mostly on near-surface marginals (T, RH, wind) and their co-occurrence, which marginal and multivariate scores already cover. Expected added value is low.
- **Climate emulators.** One number per system and lead could become one number per emulator and season, a "texture realism" check alongside spectra. This is speculative, and no source was retrieved for climate-emulator structure evaluation (e.g. ACE).

### Gaps
- No sources were collected on wind-ramp forecasting with ML, convective-initiation verification, or fire-weather ML evaluation; these use cases remain unsubstantiated.
- No study links any texture or cross-scale metric to DA performance.
- Climate-emulator evaluation (ACE, NeuralGCM climate runs) was not searched.

---

## Q6. Who would use it, and what gaps in current evaluation practice it could fill (with a candid statement of where it adds little)

### Takeaway
Realistic adopters are evaluation-framework maintainers and model developers, not end users:
- WeatherBench-X/WB2, whose framework is modular and easily extended with a new Metric class;
- ECMWF diagnostics and verification staff, who explicitly call for "structural realism" tools;
- ML labs fine-tuning deterministic models for sharpness (ECCC's AMSE work, NVIDIA CorrDiff, ECMWF AIFS);
- ChaosBench and PhysMetrics-style physics-metric suites.

The gap it fills is narrow but real: a spectrum-independent, single-number test of conditional-mean-likeness that works on deterministic forecasts. It adds little for:
- well-calibrated generative ensembles already validated by pooled CRPS and spectra;
- end-user impact metrics (TC intensity, wind power);
- anything needing a map or km-scale resolution.

### Cited Findings
- **WeatherBench-X** is the modular successor to the WB2 evaluation code. Data loaders, interpolations, metrics and aggregation are interoperable classes, and the WB2 benchmark now runs on it. This is a low-friction home for a new metric [V] — [WeatherBench-X docs](https://weatherbench-x.readthedocs.io/en/latest/); [GitHub README v2025.02.1](https://github.com/google-research/weatherbenchX/blob/v2025.02.1/README.md)
- **ECMWF realism framework.** It explicitly calls for new testing protocols for structural and physical realism, and lists spectra, effective resolution and sharpness as the current structural tools [V] — [Ben-Bouallègue 2026](https://arxiv.org/html/2602.00622v2)
- **ECMWF spectral diagnostics.** ECMWF already runs power-spectral spread/error diagnostics to compare IFS ENS and AIFS ensembles and to evaluate cycle upgrades. A complementary cross-scale statistic would slot into the same diagnostic workflow [V] — [Rodwell et al. 2025](https://doi.org/10.1002/met.70071); [Rodwell workshop slides](https://events.ecmwf.int/event/383/contributions/4647/attachments/2740/4643/Diagnostics-WS_Rodwell.pdf)
- **ECCC.** The AMSE authors (Environment and Climate Change Canada) explicitly want forecasts that "retain realistic variation at finer scales" and note verification may conflate optimal smoothing with skill [V] — [Subich et al. 2025](https://arxiv.org/html/2501.19374)
- **CIRA/NOAA.** Its groups published sharpness metrics for AI imagery and operational TC evaluation, so they are an applied-verification community actively seeking structure metrics [V] — [Ebert-Uphoff et al. 2025](https://doi.org/10.1175/aies-d-24-0083.1); [DeMaria et al. 2025](https://doi.org/10.1175/aies-d-24-0085.1)
- **Physics-metric benchmarks.** ChaosBench (S2S) and PhysMetrics.Weather package "physics-based" metrics, currently spectral, conservation and dynamical. A cross-scale texture metric is absent [V] — [ChaosBench](https://arxiv.org/abs/2402.00712); [PhysMetrics.Weather](https://arxiv.org/html/2606.10642)
- **Learned critics.** Proponents argue fixed metrics have blind spots. This cuts both ways: it validates the search for targeted metrics, but it also offers a more general (if less interpretable) competitor [V] — [arXiv:2609.18381](https://arxiv.org/abs/2609.18381)

### Inferences

**Gaps it fills**
1. A spectrum-independent check of conditional-mean-likeness. This is the main one: it discriminates "sharpened mean" from "realisation" when spectra match (AMSE, spectral-loss and nudged models).
2. A unifying classification. It puts MSE-ML models and the NWP ensemble mean in one class, which spectra fail to do (Bonavita).
3. Applicability to single deterministic fields. No ensemble is needed, unlike the variogram and energy scores.
4. A cheap, scalar, lead-time-resolved number that fits WB-X scorecards.
5. A possible bridge to the "hallucination / copied texture" concern for generative downscaling, pending a km-scale extension.

**Where it adds little (candid)**
1. For GenCast, FGN, AIFS-CRPS and Mosaic, realistic-member claims already rest on pooled CRPS, wind power and spectra. The finding (a GenCast member is fine) confirms rather than discovers.
2. The over-coupling result is theoretically expected for any conditional mean. Reviewers may see it as a demonstration of a known consequence of averaging, not a new defect. The novelty claim must rest on the estimator and the spectrum-independence, not on the existence of the effect.
3. It is not a proper score and not decision-relevant by itself. Sector users will keep using TC intensity error, wind-power CRPS, FSS and SEEPS.
4. It provides one number per system and lead with no reliable map, so it cannot localise problems, unlike local wavelet spectra, SAL/MODE or pooled CRPS maps.
5. It is validated only at 200–1600 km on 850 hPa vorticity against ERA5. ERA5's own effective resolution limits trust in the 200–400 km band, and km-scale uses are untested.
6. Its information is in principle extractable from the local wavelet spectra of Buschow et al. (the same RDWT output). Novelty is in the summary statistic and its application, not the representation.
7. KE-transfer diagnostics (Chen et al. 2026) and learned critics are competing 2026 answers to "fixed spectra are insufficient"; the field is moving fast.

### Gaps

**Unresolved questions**
- No evidence was found of any existing cross-scale amplitude-modulation metric in operational scorecards (ECMWF, WB2/WB-X, NCEP). The ECMWF scorecard composition itself was not retrieved.
- WMO/WGNE intercomparison plans for AI models were not searched.
- No information was found on whether WeatherBench-X maintainers accept external metric contributions, beyond the modular design.

**Bibliographic verification summary**
- **Verified this session (DOI, arXiv or publisher page seen):**
  - Bonavita 2024 GRL; Selz & Craig 2023 GRL; Subich et al. 2025 ICML/PMLR 267
  - Rodwell et al. 2025 Met. Apps; Ebert-Uphoff et al. 2025 AIES; DeMaria et al. 2025 AIES
  - Ben-Bouallègue 2026 (IOP DOI + arXiv); Mardani et al. 2025 CEE (DOI); Harris et al. 2022 JAMES
  - Rebora et al. 2006 JHM; Thorarinsdottir et al. 2013 JUQ; Alexander et al. 2022 Ann. Oper. Res.
  - Rasp et al. 2024 JAMES; Price et al. GenCast (Nature DOI); Buschow et al. 2019 GMD; Buschow & Friederichs 2020 ASCMO; Buschow & Friederichs QJRMS (DOI)
  - arXiv preprints: Alet et al. 2506.10772; Lang et al. 2412.15832; Lang, Leutbecher & Hatfield 2607.19161; Howard & Barnes 2608.08954; Chen et al. 2609.18489; Elberkennou et al. 2609.18381; Zhdanov et al. 2604.16429; Nathaniel et al. 2402.00712; Kasteleyn et al. 2606.10642; Slivinski et al. 2412.18016; Selz et al. 2025 ESSOAr
  - WeatherBench-X docs
- **From memory, not re-checked:**
  - Roberts & Lean 2008; Wernli et al. 2008; Davis et al. 2006; Gilleland et al. 2009
  - Casati et al. 2004/2010; Gneiting & Raftery 2007; Scheuerer & Hamill 2015 (vol/pages); Pinson & Tastu 2013
  - Hamill 2001 (rank histograms, MWR 129, 550–560); Fortin et al. 2014 (spread-skill, J. Hydrometeor. 15, 1708–1713); Skamarock 2004 (effective resolution, MWR 132, 3019–3032)
  - Kochkov et al. 2024 NeuralGCM; Lam et al. 2023; Bi et al. 2023; Bodnar et al. 2025; Lang et al. AIFS 2406.01465
  - Ravuri et al. 2021; Price & Rasp 2022; Leinonen et al. 2021 vol/pages; CorrDiff vol/article number; GenCast vol/pages
  - Allen et al. venue; ChaosBench NeurIPS venue; Brenowitz et al. lagged-ensemble benchmark; Lagerquist & Ebert-Uphoff venue
- **Titles only (alphaXiv listing):** arXiv:2510.19110 (signature kernel scoring rule); arXiv:2609.34627 (dynamical fidelity); arXiv:2412.12971 (ArchesWeatherGen).
