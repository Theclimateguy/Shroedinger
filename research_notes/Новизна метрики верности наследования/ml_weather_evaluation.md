# ML weather and ensemble-mean evaluation: smoothing and realism beyond power spectra, and whether anyone has measured cross-scale coupling

Scope: what ML weather prediction (MLWP) papers and ensemble-verification papers report about blurring, smoothing and realism; which diagnostics they use; and whether any of them measures scale-to-scale amplitude or envelope coupling, intermittency, or multifractal structure in ML forecasts or ensemble means. Literature up to October 2026.

Verification legend. [V-primary]: bibliographic details and content checked in the paper's own text (arXiv full text read through alphaXiv). [V-ref]: bibliographic details checked only through the reference list of another paper that I read; content not read directly. [UNVERIFIED]: noted in the text below. I could not reach the Crossref API from the sandbox, so DOIs come from paper texts, reference lists or publisher landing pages that showed up in search results.

---

## Q1. What do the flagship MLWP and benchmark papers (WeatherBench 2, GraphCast, GenCast, Pangu, NeuralGCM, AIFS/AIFS-CRPS, Stormer, FourCastNet 3, Aurora, FuXi, ArchesWeather) say about blurring, and how do they measure it?

### Takeaway
Every flagship paper treats blurring as a result of training on MSE/MAE, i.e. as an approximation to the conditional mean. They diagnose it almost entirely with zonal or spherical-harmonic power spectra, plus three secondary symptoms: (a) categorical precipitation scores, (b) bias in derived nonlinear quantities such as wind speed, and (c) tropical cyclone and peak-wind intensity. None of them uses higher-order, phase-sensitive or cross-scale statistics. WeatherBench 2 says outright that a matching spectrum is "necessary but not sufficient" for realism.

### Cited Findings
- **WeatherBench 2** (Rasp et al. 2024, JAMES 16, e2023MS004019, doi:10.1029/2023MS004019; arXiv 2308.15560) [V-primary].
  - Its "Energy Spectra" metric is the zonal DFT power spectrum averaged over 30–60° latitude (Sec. 4.5).
  - Spectra show that the IFS ENS mean loses power with lead time, starting at the smallest wavelengths and then reaching longer ones, which is "consistent with the model of upscale error growth" (Lorenz 1969). AI models lose small-scale power from 6 h to 3 d and then hold a roughly constant spectrum. The authors say the models "optimize their blurring" for their 24–72 h training windows (Sec. 5.3).
  - Explicit caveat: a spectrum that matches observations is "a necessary but not sufficient condition" for realism (Sec. 5.3).
  - Sec. 6.3 gives the mechanism: MSE makes models "hedge their bets" by predicting the mean of the forecast distribution, and "the mean of many states is not a realistic state itself". The evidence it cites is spectra, SEEPS precipitation (blurrier models such as the ENS mean score worse), the GraphCast wind-speed bias (models "have difficulties learning correlations between variables"), and case studies of under-represented local intensity. — [WB2 arXiv](https://arxiv.org/abs/2308.15560)
  - WB2's probabilistic metrics (CRPS, spread-skill) are "univariate statistics" that "ignore spatial and temporal correlations" (Sec. 6.3). — [WB2 arXiv](https://arxiv.org/abs/2308.15560)
- **GraphCast** (Lam et al. 2023, Science 382, 1416–1421, doi:10.1126/science.adi2336; arXiv 2212.12794) [V-primary].
  - The paper says longer autoregressive MSE training "encourages it to express its uncertainty by predicting spatially smoothed outputs".
  - The authors fit "optimal blurring" filters (linear, homogeneous, isotropic, weights depending only on total wavenumber) to GraphCast and to HRES. They state that "in optimizing for MSE, we seek to approximate a conditional expectation which averages over predictive uncertainty".
  - They name the ENS ensemble mean as "an alternative way to approximate a conditional expectation" (Suppl. 7.4).
  - Diagnostics used: spectra of predictions and targets, spectral decomposition of MSE, and RMSE as a function of truncation wavenumber (Suppl. 7.5). Spurious peaks near 100 km wavelength come from the grid-to-mesh transforms. — [GraphCast arXiv](https://arxiv.org/abs/2212.12794)
- **GenCast** (Price et al.; arXiv 2312.15796; published in Nature 2025 [V-ref; volume and pages not checked]) [V-primary for the arXiv content].
  - GenCast samples are sharp, with spherical-harmonic spectra that "closely match" ERA5. "As expected, the GenCast ensemble mean is blurry, like GraphCast's states." The Fig. 2 caption describes the GraphCast state as "blurry, similar to GenCast's ensemble mean".
  - It notes that a nonlinear function of a mean is biased (wind speed) and treats this as "a simple test of GenCast's ability to model cross-variable dependence structure" (Suppl. F.9.1).
  - Spatial structure is tested with neighbourhood **average-pooled and max-pooled CRPS** at pool sizes of 120–3828 km (Sec. 8.1, Suppl. E.6). GraphCast-Perturbed "performs especially badly on max-pooling due to blurring".
  - It also evaluates tropical-cyclone tracks (REV) and regional wind power. — [GenCast arXiv](https://arxiv.org/abs/2312.15796)
- **NeuralGCM** (Kochkov et al. 2024, Nature 632, 1060–1066, doi:10.1038/s41586-024-07744-y; arXiv 2311.07222) [V-primary].
  - "All the ML models make significantly blurrier forecasts than source data ERA5." "Blurry forecasts correspond to physically inconsistent atmospheric conditions." Ensemble means "are inherently smooth at long lead times", while individual members "remain sharp".
  - Diagnostics: zonal power spectra of many derived variables (vorticity, divergence, vertical velocity, ageostrophic wind, EKE, IVT), plus geostrophic/ageostrophic vertical structure.
  - Its deterministic loss combines filtered MSE (high wavenumbers progressively removed at longer leads to counter the double penalty) with a spectrum-matching loss. — [NeuralGCM arXiv](https://arxiv.org/abs/2311.07222)
- **AIFS-CRPS** (Lang et al.; arXiv 2412.15832; later in npj Artificial Intelligence 2(1), 2026 [V-ref]) [V-primary for arXiv].
  - MSE training "incentivises the smoothing of forecast fields with lead-time and a reduction of forecast activity" because of the double penalty.
  - AIFS trained with MSE "progressively lose[s] energy at higher wavenumbers with lead time", whereas AIFS-CRPS members keep stable spectra.
  - Diagnostics: Z500/T850 spectra by lead time, CRPS, ensemble-mean RMSE, spread, and MJO. — [AIFS-CRPS arXiv](https://arxiv.org/abs/2412.15832)
- **Stormer** (Nguyen et al.; arXiv 2312.03876) [V-primary].
  - Its inference averages forecasts made with different time-step combinations; the authors say this "achieves a similar effect to ensembling in NWP" and is "a Monte Carlo integration approach" to initial-condition sensitivity.
  - The gain beyond 7 days is "attributed to ... averaging out multiple forecasts". In effect this is deliberate conditional-mean behaviour. No realism diagnostics are reported. — [Stormer arXiv](https://arxiv.org/abs/2312.03876)
- **Pangu-Weather** (Bi et al. 2023, Nature 619, 533–538, doi:10.1038/s41586-023-06185-3) [V-ref]. Its own blurring statements were not read. Bonavita notes that Pangu uses an L1 (MAE) loss, so it estimates a conditional median rather than a mean. — [Bonavita arXiv](https://arxiv.org/abs/2309.08473)
- **FourCastNet 3** (Bonev et al. 2025, arXiv 2507.12144) [V-ref]. It combines spatial CRPS with spectral CRPS over all resolved wavenumbers, "encouraging realistic spatial correlations and power spectra" (as described by Chen et al. 2026). — [Chen et al. 2026](https://arxiv.org/abs/2609.18489)
- **Aurora** (Bodnar et al. 2025, Nature 641, 1180–1187) and **FuXi** (Chen et al. 2023, npj Clim Atmos Sci 6, 190, doi:10.1038/s41612-023-00512-1) [V-ref]. I did not read their own blurring diagnostics (see Gaps).

### Inferences
- In the flagship literature, "blurring" is a spectral-amplitude concept, explained by the conditional mean, the double penalty and spectral bias.
- No flagship paper asks how the structure that survives is arranged across scales, for example whether fine-scale amplitude is still independent of the coarse field. The one non-spectral structural probe is GenCast's pooled CRPS, which scores marginal skill after coarse-graining, not coupling.
- WB2's "necessary but not sufficient" caveat names exactly the gap that a beyond-spectrum metric such as inheritance fidelity tries to fill. It is a good sentence to cite as motivation.

### Gaps
- I did not read the primary texts of Aurora, FuXi, FourCastNet v1/v2/3, ArchesWeather/ArchesWeatherGen or AIFS deterministic (Lang et al. 2024, arXiv 2406.01465) for their own blurring diagnostics. Their content is known only through secondary citations.
- GenCast's Nature volume and pages were not checked (only the arXiv version was read).

---

## Q2. What do critical-evaluation papers show, and which diagnostics beyond spectra do they use?

### Takeaway
The critical literature goes well beyond spectra into the following:
- dynamical balances (geostrophic/ageostrophic ratio, divergence/vorticity ratio, diagnosed vertical velocity, hydrostatic residuals, lapse-rate distributions);
- conservation budgets;
- butterfly-effect and DKE-spectrum error growth;
- spectral kinetic-energy transfer;
- effective resolution and fair rescaling;
- TC and peak-wind intensity;
- multivariate proper scores (variogram);
- learned discriminators;
- process-level predictive structure;
- regeneration of fine scales from truncated initial conditions.

Several papers find that spectra can look realistic while cross-scale energy transfer or joint structure is wrong. None measures envelope or amplitude coupling between scale bands.

### Cited Findings
- **Bonavita 2024**, "On some limitations of current machine learning weather prediction models", GRL 51(12), e2023GL107377, doi:10.1029/2023GL107377 (preprint arXiv 2309.08473) [V-primary for preprint; journal details V-ref in two reference lists].
  - Effective resolution: Pangu's spectra fall below ERA5 from about wavenumber 60–80, so its effective resolution is "closer to 500-700 km".
  - Balances: the ratio of ageostrophic to geostrophic wind is too small and keeps falling with lead time; the ratio of divergence to vorticity is "significantly and progressively reduced"; diagnosed vertical velocities are about 40% weaker than ERA5 hindcasts.
  - Tropical cyclones: Doksuri and Lee become too shallow. — [arXiv](https://arxiv.org/abs/2309.08473)
- **Bonavita 2024, Sec. 3.1 "Is Pangu-Weather emulating the ECMWF ensemble forecast mean?"** [V-primary]. This is the closest direct test of "ML equals ensemble mean". Using spectra only, Bonavita finds that Pangu differs from the ENS mean (EM):
  - At 12–24 h, EM spectra "closely track" ERA5 while Pangu is already heavily damped above wavenumber 60.
  - At 120 h, EM loses energy at synoptic scales, where "ensemble averaging acts to smooth out unpredictable forecast modes" (Leith 1974), but keeps more high-wavenumber energy than Pangu.
  - Conclusion: Pangu has "both too little energy at short forecast ranges and high wave numbers, and too much energy in the medium range at synoptic and sub-synoptic scales" compared with EM. — [arXiv](https://arxiv.org/abs/2309.08473)
- **Ben Bouallègue et al. 2024**, "The rise of data-driven weather forecasting…", BAMS 105, E864–E883, doi:10.1175/BAMS-D-23-0162.1 (arXiv 2307.10128) [V-primary for preprint; the issue number is reported inconsistently as 3 or 6 in different sources].
  - They measure forecast activity (standard deviation of the anomaly). Pangu's activity is similar to IFS because "this metric is dominated by larger scales", even though small scales are visibly smoothed. The EM, by contrast, "becomes smoother as the forecast uncertainty increases", and they state "the ensemble mean cannot be considered as a plausible scenario".
  - Pangu "clearly underestimates" TC intensity, with "too-weak gradients and too-weak maximum wind speed".
  - They recommend "multivariate verification that accounts for the correlation between variables". — [arXiv](https://arxiv.org/abs/2307.10128)
- **Selz & Craig 2023**, "Can artificial intelligence-based weather prediction models simulate the butterfly effect?", GRL 50(20), e2023GL105747 [V-ref], and **Selz & Craig 2026**, JGR: Machine Learning and Computation 3(3), e2025JH001180 [V-ref]. As summarised by Chen et al. (2026) and Bonavita: Pangu (and in 2026 also Aurora) does not reproduce the rapid, amplitude-dependent initial growth of small perturbations, and the DKE spectral peak shifts in ways "inconsistent with the butterfly effect". — [Chen et al. 2026 arXiv](https://arxiv.org/abs/2609.18489); [citedrive record of 2026 paper](https://www.citedrive.com/en/discovery/can-aibased-weather-prediction-models-simulate-the-butterfly-effect-the-role-of-architecture-and-implementation/)
- **Hakim & Masanam 2024**, "Dynamical tests of a deep-learning weather prediction model", Artificial Intelligence for the Earth Systems 3(3), e230090, doi:10.1175/AIES-D-23-0090.1 (arXiv 2309.10867) [V-ref for venue: the AIES citation appears in Ben-Bouallègue 2026's reference list, while one preprint version says it was prepared for AGU Advances].
  - They run Pangu through four idealised experiments: a Matsuno–Gill response to tropical heating, extratropical cyclogenesis from a localized jet perturbation, geostrophic adjustment in about 6 h, and hurricane genesis above a perturbation threshold.
  - Conclusion: Pangu "encodes realistic physics". This is a counterpoint to Bonavita. — [arXiv](https://arxiv.org/abs/2309.10867)
- **Charlton-Perez et al. 2024**, "Do AI models produce better weather forecasts than physics-based models? A quantitative evaluation case study of Storm Ciarán", npj Clim Atmos Sci 7, 93, doi:10.1038/s41612-024-00638-w [V-ref plus a university press release]. Track and rapid deepening were captured, but all four AI models underestimated peak winds. Ben-Bouallègue (2026) summarises it as tracks right (realism type 1), maximum wind too weak (type 2), and strengthening processes sometimes poorly represented (type 3). — [Reading press release](https://www.reading.ac.uk/news/2024/Research-News/AI-weather-forecasts-captured-Ciarans-destructive-path); [Ben-Bouallègue 2026](https://arxiv.org/abs/2602.00622)
- **Subich, Husain, Separovic & Yang 2025**, "Fixing the double penalty in data-driven weather forecasting through a modified spherical harmonic loss function", ICML 2025, PMLR 267 (arXiv 2501.19374) [V-ref via PMLR/arXiv listing; full text not read].
  - The loss separately penalises decorrelation error and spectral-amplitude error.
  - Fine-tuning GraphCast with it reportedly raises effective resolution from about 1,250 km to 160 km and improves ensemble spread, TC strength and surface wind extremes.
  - **Relevance:** the loss is explicitly amplitude- and decorrelation-based per wavenumber. It fixes the spectrum and still says nothing about cross-band envelope coupling. — [PMLR](https://proceedings.mlr.press/v267/subich25a.html); [arXiv](https://arxiv.org/abs/2501.19374)
- **Selz, Bruinsma, Craig, Markou, Turner, Vaughan (and Allen in the 2026 version)**, "On the effective resolution of AI weather prediction models", ESSOAr preprint doi:10.22541/essoar.174139239.94807670/v1 (2025); EMS2025-541 abstract [peer-reviewed venue UNVERIFIED].
  - They argue mathematically that "an ideal, perfectly trained AI model follows the mean of the predictive distribution" over the lead-time window used in the loss, and test this on Aurora, Pangu, GraphCast and GenCast forecasts.
  - Lower effective resolution improves MSE through the double penalty. They propose spectral rescaling for fair comparison and apply it to ECMWF ENS/HRES as well. — [EMS2025 abstract](https://meetingorganizer.copernicus.org/EMS2025/EMS2025-541.html)
- **Rodwell, Clare, Lock, Lonitz & Chevallier 2025**, "Power spectra of physics-based and data-driven ensembles", Meteorological Applications 32(5), e70071, doi:10.1002/met.70071 [V-ref via Ben-Bouallègue 2026; content not read because the Wiley page returned 403]. ECMWF seminar material says variances of physical-model ensembles saturate quickly at small scales, while errors at synoptic and planetary scales reach only about 50% of saturation by day 10. — [ECMWF slides](https://events.ecmwf.int/event/383/contributions/4647/attachments/2740/4643/Diagnostics-WS_Rodwell.pdf)
- **Ebert-Uphoff et al. 2025**, "Measuring sharpness of AI-generated meteorological imagery", AIES 4(3), e240083, doi:10.1175/AIES-D-24-0083.1 [V-ref].
  - The paper surveys sharpness metrics from other fields, introduces "Gaussian blur equivalence", and demonstrates the metrics on GraphCast.
  - The individual metrics could not be retrieved. Per the abstract, it frames sharpness as a single-field property, not cross-scale coupling. — [NSF PAR record](https://par.nsf.gov/biblio/10614981-measuring-sharpness-ai-generated-meteorological-imagery)
- **Li et al. 2025**, "Exploring the differences in atmospheric mesoscale kinetic energy spectra between AI based and physics based models", Sci Rep 15, 15504, doi:10.1038/s41598-025-99815-x [V-ref]. Pangu misses the mesoscale −5/3 slope and is too weak below 1000 km. Spectral **downscale energy flux** is stronger at larger mesoscales but weakest at smaller mesoscales. — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12049474/)
- **Kim et al. 2026**, "A spectral test of the butterfly effect and physical consistency in the diffusion-based GenCast's ensembles", npj Clim Atmos Sci 9 [V-ref; repository abstract only]. GenCast shows weak planetary-scale error growth and a persistent, flattened KE tail at high wavenumbers from the first step. — [GIST repository](https://scholar.gist.ac.kr/handle/local/34241)
- **Chen, Oskarsson, Driscoll & Schemm 2026**, "Butterfly effect and the kinetic energy cascade in probabilistic MLWP models", arXiv 2609.18489 (submitted to GRL) [V-primary].
  - They diagnose ΔKE(n, τ), the change in spectral KE relative to τ = 0, as a signature of cross-scale transfer.
  - FCN3, AIFS-ENS and GenCast "produce realistic KE spectral magnitudes but do not capture the expected upscale transfer of KE". NeuralGCM-ENS reproduces upscale transfer but underestimates mesoscale KE.
  - Key sentence: "MLWP models can misrepresent the known scale transfer of kinetic energy despite producing skilful weather forecasts." — [arXiv](https://arxiv.org/abs/2609.18489)
- **Kasteleyn et al. 2026**, "PhysMetrics.Weather", arXiv 2606.10642 [V-primary].
  - Nine metrics: dry-air, water and energy drift; effective resolution, spectral divergence (W1 between normalised spectra) and spectral residual; geostrophic and hydrostatic excess RMSE; lapse-rate Wasserstein distance. Models evaluated on WB2 2020 data.
  - MLWP hydrostatic RMSE falls *below* the ERA5 reference, which the authors read as "fine-scale smoothing, resulting in less extreme, artificially balanced states".
  - The paper has no intermittency or cross-scale coupling metric, and its own comparison table lists no such category. — [arXiv](https://arxiv.org/abs/2606.10642)
- **Elberkennou et al. 2026**, "Every fixed metric has a blind spot: a learned atmospheric critic for scoring forecast realism", NeurIPS 2026 workshop, arXiv 2609.18381 [V-primary].
  - A SqueezeNet discriminator gives a reverse-KL realism score. Fixed metrics (zonal log-spectrum, mean Wasserstein distance, SCWD, MMD) each miss at least one synthetic corruption.
  - "Spectral metrics do not distinguish discrepancies between individually-realistic constituent fields" (field splice, hemisphere splice).
  - Unrealism grows with lead time for GraphCast, Pangu, FuXi and U-Cast but stays flat for HRES and SWIFT. The critic could in principle pick up cross-scale coupling errors, but it does not identify which statistic differs. — [arXiv](https://arxiv.org/abs/2609.18381)
- **Bohn Lassen et al. 2026**, "Evaluating dynamical fidelity through predictive structure in physical representations", arXiv 2609.34627 [V-primary]. They freeze predictive relationships found in ERA5 (EP flux, wave-1/2 forcing → NAM evolution) and test whether GraphCast, FengWu and Pangu preserve the conditional mean and variance. "Lower state-space forecast error does not imply closer agreement with the conditional dynamics." This is process-level, not scale-band coupling. — [arXiv](https://arxiv.org/abs/2609.34627)
- **Howard & Barnes 2026**, "Do AI forecast ensembles sample the correct conditional distribution?", arXiv 2608.08954 (submitted to GRL) [V-primary].
  - A diffusion model has positive CRPS skill but *negative* variogram-score skill. The failure is "invisible to the energy score".
  - The worst pair is the one with the highest observed correlation (Atlantic City–Battery, r = 0.98), "consistent with the DDPM over-correlating this already highly correlated pair".
  - In Lorenz-96, a deterministic emulator reproduces the joint-structure failure (VS/CRPS ratio 0.42–0.58) and a dynamical ensemble does not (about 0.91–1.02). — [arXiv](https://arxiv.org/abs/2608.08954)
- **Fan et al. 2025/2026**, "Learning more physically realistic dynamics in ML-based weather forecasting with latent-space constraints", arXiv 2510.04006 v2 [V-primary].
  - Section 4.2 "Cross-Scale Dynamics": wavenumbers above 6 are removed from the initial conditions, and the authors track whether the model regenerates fine-scale Z500 power. The MSE model saturates at a fine-scale power fraction of about 0.5×10⁻³ against ERA5's 2.1×10⁻³; the latent-space model approaches ERA5.
  - The authors conclude the difference "primarily stem[s] from the way the learned dynamics represent interactions between large and small scales". Measured with spectra only. — [arXiv](https://arxiv.org/abs/2510.04006)
- **Hassanzadeh, Li, Sun et al. 2026**, "Missing the butterfly and predicting the past", arXiv 2608.25835 [V-primary, targeted sections].
  - They attribute the missing butterfly effect and the high skill to coarse-graining of the training data. AI models "implicitly learn how fast, small scales affect large scales without inheriting their rapid error growth".
  - They argue *against* conditional-mean optimisation or blurring as the primary cause of the missing butterfly effect, because MSE-trained networks reproduce it in Lorenz-96 and diffusion models (GenCast, FourCastNet2) still miss it. — [arXiv](https://arxiv.org/abs/2608.25835)
- **Ben-Bouallègue 2026**, "'What is a realistic forecast?' Assessing data-driven weather forecasts, a journey from verification to falsification", arXiv 2602.00622 [V-primary]. A conceptual taxonomy:
  - functional realism: scoring;
  - structural realism: diagnostics such as bias, activity, spectra, effective resolution and sharpness;
  - physical realism: falsification against knowledge, e.g. Hakim & Masanam, Bonavita.
  
  It mentions no cross-scale or intermittency diagnostic. — [arXiv](https://arxiv.org/abs/2602.00622)

### Inferences
- The field already accepts that matching spectra are not enough. Three lines of work show realistic spectra coexisting with wrong multi-scale structure: Chen et al. 2026 (wrong spectral energy transfer), Howard & Barnes 2026 (wrong joint dependence) and Elberkennou et al. 2026 (spectra blind to splices).
- **Closest conceptual neighbours to inheritance fidelity:**
  - Chen et al. ΔKE(n) and Li et al. spectral fluxes: cross-scale *energy transfer*, which is phase-blind and spectrum-averaged.
  - Fan et al.: whether fine scales are generated *from* large scales, measured by spectra only.
  - Howard & Barnes: over-correlation of already strongly correlated components in data-driven emulators. This is spatial joint dependence, not scale-band envelope dependence, but the direction (excess coupling, reproduced by a deterministic emulator) parallels the user's over-coupling result.
- Bonavita's Sec. 3.1 is the main prior test of whether a deterministic ML model behaves like the ensemble mean. Using spectra, he found they differ: Pangu is damped at high wavenumber early, while the EM is damped at synoptic scales later. That inheritance fidelity puts Pangu (×1.27) and the IFS EM (×1.24) at nearly the same over-coupling may be a useful point of contrast: a metric beyond spectra unifies two systems that a spectral test separated. This is an inference, and it should be checked against lead times and scales comparable to Bonavita's (he used 12, 24 and 120 h).

### Gaps
- Full texts of Rodwell et al. 2025 (Met Apps), Ebert-Uphoff et al. 2025 (AIES), Selz & Craig 2023/2026 and Kim et al. 2026 were not read. They may contain scale-dependent diagnostics that I could not inspect.
- I did not check whether Selz et al.'s "effective resolution" paper is now published in a peer-reviewed journal.
- Charlton-Perez et al. 2024 was not read in full; its findings come from the press release and from Ben-Bouallègue's summary.

---

## Q3. Has anyone applied multifractal, cascade or intermittency analysis, wavelet coherence or bicoherence, or cross-scale amplitude or envelope modulation to ML forecasts or to ensemble means?

### Takeaway
I found **no** published study, up to October 2026, that applies any of the following to ML weather forecasts or to ensemble-mean forecasts:
- intermittency statistics: scale-dependent kurtosis or higher-order structure functions;
- Lovejoy–Schertzer multifractal or cascade analysis;
- wavelet coherence or bicoherence;
- amplitude modulation or envelope correlation between scale bands.

Cross-scale questions in this literature are asked only through spectral energy and energy transfer (ΔKE(n), spectral fluxes, DKE-peak migration), through the regeneration of truncated scales, or implicitly through learned discriminators. On the evidence gathered, the inheritance-fidelity metric (anchored rank correlation of band log-envelopes against a phase-randomised surrogate null) has no close prior art in ML-weather or ensemble verification. Its nearest relatives are in turbulence and nonlinear-dynamics work on observations, not forecasts.

### Cited Findings
- Searches combining "intermittency", "multifractal", "structure function", "kurtosis", "wavelet", "bicoherence", "amplitude modulation" and "cross-scale coupling" with GraphCast, Pangu, AI weather or ensemble mean returned no matching forecast-verification paper. Several search engines explicitly reported no combined match. — [WebSearch: intermittency/kurtosis ML weather](https://arxiv.org/pdf/2603.27288); [WebSearch: amplitude modulation / cross-scale](https://arxiv.org/pdf/1812.03683)
- The Lovejoy framework (scaling, structure functions, intermittency, multifractals) is reviewed in Lovejoy 2023 (NPG 30, 311). No application to AI forecast models turned up. A 2025 Schertzer workshop deck lists neural-network rainfall prediction as an open direction, without GraphCast-type models. — [Lovejoy 2023 NPG](https://npg.copernicus.org/articles/30/311/2023/); [Schertzer 2025 slides](https://perso.ensta-paris.fr/~manzaner/Workshop_USP-IPP_2025/Slides/DSchertzer.pdf)
- StretchCast (arXiv 2603.27288) mentions "intermittency" only qualitatively, as the reason humidity spectra depart more at short wavelengths. It computes no intermittency statistic. — [arXiv](https://arxiv.org/pdf/2603.27288)
- PhysMetrics.Weather's benchmark comparison table has only conservation, spectral and dynamical-balance categories (WB2, WB-X, ChaosBench, ExtremeWB, ClimateLearn). No benchmark covers intermittency or cross-scale coupling. — [arXiv 2606.10642](https://arxiv.org/abs/2606.10642)
- **Nearest MLWP neighbours** (all spectral or energy-based, see Q2):
  - Chen et al. 2026: ΔKE(n), i.e. cross-scale KE transfer. — [arXiv](https://arxiv.org/abs/2609.18489)
  - Li et al. 2025: spectral energy flux in Pangu vs MPAS. — [PMC](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12049474/)
  - Fan et al. 2025/26: fine-scale regeneration from truncated initial conditions, interpreted as "interactions between large and small scales". — [arXiv](https://arxiv.org/abs/2510.04006)
- **Outside forecasting**, amplitude modulation of small scales by large scales is an established turbulence diagnostic (wall-bounded flows: envelope of small-scale fluctuations correlated with the large-scale signal). The 2018 arXiv revisit notes the effect is nonlinear and hard to quantify. Cross-frequency phase–amplitude coupling has also been reported in observed air-temperature records ("Multiscale atmospheric dynamics: cross-frequency phase-amplitude coupling in the air temperature"; bibliographic details beyond the title UNVERIFIED). Neither applies these to forecasts. — [arXiv 1812.03683](https://arxiv.org/pdf/1812.03683); [ResearchGate title](https://www.researchgate.net/publication/260440283_Multiscale_Atmospheric_Dynamics_Cross-Frequency_Phase-Amplitude_Coupling_in_the_Air_Temperature)
- The learned atmospheric critic (Elberkennou et al. 2026) could detect a cross-scale coupling defect without naming it. It was not designed or reported as a coupling measure. — [arXiv](https://arxiv.org/abs/2609.18381)

### Inferences
- Novelty at the level of the *diagnostic*: an anchored, surrogate-nulled envelope coupling between adjacent octave bands, applied to forecast-minus-truth across ML and ensemble-mean systems, appears unprecedented in MLWP and ensemble-verification work. That is a candid result from roughly 25 targeted searches and full-text reads of the main critical papers.
- Novelty at the level of the *phenomenon* needs care.
  - Already known: blurring, conditional-mean behaviour, ensemble-mean smoothing, spectra that hide structural errors, and data-driven emulators over-correlating strongly correlated components (Howard & Barnes).
  - New: that conditional means **over-couple adjacent scales** (fine-band envelope slaved to the coarse field), that single realisations do not, and that uniform spectral equalisation cannot reproduce this (≤10%) while selective envelope-modulated damping can.
  - The "spectral equalisation cannot reproduce it" result directly supports, with a mechanism, WB2's "necessary but not sufficient" caveat and Elberkennou et al.'s claim that spectral metrics are blind.
- A reviewer may raise Hassanzadeh et al. 2026, who argue that blurring and the conditional mean are *not* the main cause of the missing butterfly effect. The user's result concerns structure in the forecast state, not error growth, so the two are compatible. The distinction should still be stated explicitly.
- The user's "effect absent at 24 h" fits WB2's observation that AI blurring is optimised for the 24–72 h training windows. It also fits Selz et al.'s argument that the model follows the predictive mean over the loss window: at 24 h the predictive distribution is still narrow at 200–1600 km scales.

### Gaps
- Search coverage is web/arXiv-based. Paywalled 2025–2026 QJRMS, MWR, Met Apps and AIES papers on scale-dependent verification of AI ensembles (for example the full text of Rodwell et al. 2025) could not be read and might contain wavelet or scale-separated statistics.
- I found no wavelet-based (Casati-type intensity-scale) verification of AI global models. I did not exhaustively check regional nowcasting or precipitation ML papers, where wavelet or FSS methods are common but cross-scale coupling is still not.

---

## Q4. Is it stated explicitly in the literature that deterministic ML forecasts and ensemble means behave alike because both are conditional means, and with which diagnostics?

### Takeaway
Yes. The idea is widespread and explicit:
- WB2, GraphCast, GenCast, NeuralGCM, AIFS-CRPS, Bonavita and Selz et al. all say MSE-trained ML forecasts approximate the conditional mean or expectation, as an ensemble mean does.
- GenCast shows it visually and spectrally: the GraphCast state is "similar to GenCast's ensemble mean".
- Stormer averages forecasts deliberately for ensemble-like gains.

The supporting diagnostics are almost entirely power spectra, plus RMSE behaviour, activity, SEEPS, wind-speed bias and pooled CRPS. The one direct test, Bonavita 2024, found *spectral discrepancies* between Pangu and the ENS mean. No paper compares ML forecasts and ensemble means on a cross-scale or structural metric.

### Cited Findings
- **WB2 Sec. 6.3**: MSE models "hedge their bets by predicting the mean of the potential forecast distribution". Sec. 5.3: the ENS mean "will become increasingly smooth with lead time" and AI models "optimize their blurring" for 24–72 h. The diagnostics are spectra and SEEPS. — [arXiv](https://arxiv.org/abs/2308.15560)
- **GraphCast Suppl. 7.4**: optimal blurring filters approximate "a conditional expectation which averages over predictive uncertainty". The ENS mean is "an alternative way to approximate a conditional expectation". The diagnostics are wavenumber-dependent filter transfer functions and spectra. — [arXiv](https://arxiv.org/abs/2212.12794)
- **GenCast Sec. 3 and Fig. 2**: "MSE-trained models like GraphCast are closer to the ensemble mean." The diagnostics are spherical-harmonic spectra at 1 and 15 days, derived wind-speed bias and max-pooled CRPS. — [arXiv](https://arxiv.org/abs/2312.15796)
- **NeuralGCM**: ensemble means "are closer to ERA5 in an average sense, and thus are inherently smooth at long lead times", while ensemble members "remain sharp". The diagnostics are spectra and case studies. — [arXiv](https://arxiv.org/abs/2311.07222)
- **AIFS-CRPS**: the MSE objective incentivises smoothing and reduced activity, while CRPS-trained members stay sharp. The diagnostics are spectra by lead time. — [arXiv](https://arxiv.org/abs/2412.15832)
- **Selz et al. 2025 (ESSOAr / EMS2025)**: "an ideal, perfectly trained AI model follows the mean of the predictive distribution". They support this with a mathematical argument and spectra/effective-resolution analysis of Aurora, Pangu, GraphCast and GenCast, comparing with ENS and HRES. — [EMS abstract](https://meetingorganizer.copernicus.org/EMS2025/EMS2025-541.html)
- **Bonavita 2024 Sec. 3.1**: ML models trained on L2/L1 losses estimate "the posterior mean/median of the forecast distribution". But a direct spectral comparison with the ECMWF EM shows Pangu "differ[s] from ECMWF EM forecasts". The interpretation "is not completely straightforward", and the reduced activity is lead-time dependent ("significant heteroscedasticity"). — [arXiv](https://arxiv.org/abs/2309.08473)
- **Ben Bouallègue et al. 2024 (BAMS)**: Pangu does *not* become smoother with lead time in activity or spectra (not shown), "as we might expect when training toward RMSE", whereas the EM does. This is evidence that Pangu and the EM differ in how their smoothing evolves. — [arXiv](https://arxiv.org/abs/2307.10128)
- **Stormer**: averaging multiple rollouts "achieves a similar effect to ensembling in NWP". — [arXiv](https://arxiv.org/abs/2312.03876)
- **Howard & Barnes 2026**: a *deterministic emulator* driven by perturbed initial conditions reproduces the joint-dependence failure of generative AI ensembles, and a dynamical ensemble does not. The tool is the variogram score. — [arXiv](https://arxiv.org/abs/2608.08954)
- **Counterpoint, Hassanzadeh et al. 2026**: optimising toward a conditional mean is not the primary cause of the missing butterfly effect. Coarse-graining of training data is. — [arXiv](https://arxiv.org/abs/2608.25835)

### Inferences
- "Deterministic ML ≈ conditional mean ≈ ensemble mean" is textbook in this literature, so the user should cite WB2, GraphCast Suppl. 7.4, GenCast Fig. 2, NeuralGCM, Bonavita Sec. 3.1 and Selz et al. rather than present the equivalence as new.
- What is *not* in the literature:
  1. a quantitative demonstration that the ML and EM likeness holds on a *structural, beyond-spectrum* metric;
  2. the sign and size of the shared defect (over-coupling ×1.24–1.35);
  3. the clean contrast with single realisations (HRES, one ENS member, one GenCast member) at truth;
  4. a control showing that spectrum equalisation does not reproduce it.
  
  Point (1) also partly reconciles Bonavita's spectral finding that Pangu ≠ EM. The conditional-mean signature may show up more cleanly in envelope coupling than in spectral shape, because spectral shape depends on training windows, the L1 vs L2 loss and resolution. This is an inference to test, not established.
- The Phase 33 Arm C result, that the IFS ENS mean over-inherits as well, is consistent with the GenCast and WB2 view that blurring is a property of averaging rather than of neural networks. Phrased that way, it is a quantitative extension of an accepted idea, which is easier to defend with reviewers than a claim of a new phenomenon.

### Gaps
- I did not find any paper that runs a *structural* (non-spectral) comparison between an ML deterministic forecast and an ensemble mean, besides Bonavita's spectra and GenCast's pooled CRPS. A paywalled 2025–26 paper could exist, for example Rodwell et al. 2025 Met Apps, which I could not read.
- The final published version of Selz et al. on effective resolution, which may contain ENS-mean comparisons, was not located.
