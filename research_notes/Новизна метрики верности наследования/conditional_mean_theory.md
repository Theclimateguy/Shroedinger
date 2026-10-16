# What a conditional mean does to multiscale structure: is "over-inheritance" of ensemble means / MSE-trained ML forecasts expected or new?

Finding being judged: at 120 h, conditional-mean-like forecasts (GraphCast, Pangu, Aurora, the IFS 50-member ENS mean) show 24-35 % stronger correlation than ERA5 between the log amplitude envelopes of adjacent scale bands (200-400 / 400-800 / 800-1600 km, 850 hPa vorticity). Single realisations (HRES, one ENS member, one GenCast member) do not. A uniform loss of variance does not reproduce it, while a selective loss does (fine variance removed where the coarse envelope is weak). The effect is absent at 24 h. The proposed mechanism is that averaging removes the fine component that is independent of the coarse field and keeps the part slaved to it.

Verification status: items marked **[verified]** were checked against a publisher, arXiv or repository record in this session. Items marked **[from memory, unverified]** are standard references whose details were not re-checked online here. Confirm them before citing.

---

## Q1. Ensemble-mean smoothing literature: does anyone say ensemble means keep small-scale structure locked to predictable large scales?

### Takeaway
It is textbook that the ensemble mean filters out unpredictable, mostly small-scale features. This explains its RMSE advantage, its low-intensity bias, its loss of variance and why it is "not a physical state". The literature also says, implicitly, that the arithmetic mean keeps features whose position the members agree on and flattens features whose position is uncertain. I found no paper that states the cross-scale corollary: that the surviving fine structure is therefore more strongly tied to the coarse envelope than in reality. I also found none that measures this as a correlation between band envelopes.

### Cited Findings
- Surcel, Zawadzki & Yau studied storm-scale precipitation from the 2008 NOAA HWT SSEF ensemble. They found that "averaging filters out nonpredictable features on which the members disagree", and that these features are small-scale. They identified the affected scale range by comparing the statistics of the ensemble mean with those of individual members. **[verified, abstract]** — [AMS 15th Mesoscale Conf. abstract 2013](https://ams.confex.com/ams/15MESO/webprogram/Paper227817.html); [journal abstract record](https://synapsesocial.com/papers/W2045144288). The journal version is usually cited as Surcel, Zawadzki & Yau (2014), *Mon. Wea. Rev.*, 142, 1093–1105, "On the filtering properties of ensemble averaging for storm-scale precipitation forecasts". The volume and pages were **not confirmed** in this session.
- The same work reports two more results. Averaging causes a low-intensity bias. The probability-matched mean (PMM) has the same intensity PDF as the members, but its "spectral structure and spatial distribution" differ, and its better scores come from lower small-scale variability. **[verified, abstract]** — [synapsesocial record](https://synapsesocial.com/papers/W2045144288). This means that matching the one-point distribution does not restore the spatial and spectral structure, which parallels the finding's "uniform loss does not reproduce" control.
- Surcel et al. (2015) define λ0, the scale at which forecast and observed precipitation become fully decorrelated, as a function of lead time. Cited as *J. Atmos. Sci.*, 72, 216–235. **[partially verified, via lecture slides]** — [Zawadzki slides](https://ifaeci.cima.fcen.uba.ar/bde/pres/Zawadzki.pdf). Surcel et al. (2017, *MWR* 145, 3625–3646) report loss of predictability below about 200 km after 24 h, caused by large-scale IC/BC and microphysics uncertainty. **[verified]** — [CAPS PDF](https://twister.caps.ou.edu/papers/Surcell_etal_MWR2017.pdf).
- Feng, Zhang, Toth, Peña & Ravela (2020) write that the arithmetic mean "flattens features whose presence is still predictable" but whose location is uncertain, which reduces their amplitude and variability. They propose a feature-oriented mean that aligns features across members before averaging. *Weather and Forecasting* 35(3), doi:10.1175/WAF-D-19-0213.1. **[verified]** — [NOAA repository](https://repository.library.noaa.gov/view/noaa/31986). The corollary is that features whose location is fixed by a predictable coarse flow survive averaging. This is the closest published statement to the proposed mechanism, but it is about amplitude retention, not cross-scale coupling.
- A BAMS paper on ensemble use cases describes the conventional view: the mean "smooths out small-scale less predictable details, resulting in low RMSE, but the resulting solution is not a physically consistent meteorological solution". It also says the probability-matched mean keeps more of the distribution. **[verified, text snippet]** — [BAMS-D-24-0183.1 PDF](https://opensky.ucar.edu/system/files/2026-02/bams-bams-d-24-0183.1.pdf).
- WeatherBench 2 reports that the IFS ENS mean loses spectral power first at small wavelengths and then at larger ones as lead time grows, "consistent with Lorenz's 1969 upscale error growth model". The effect is larger for humidity and precipitation. **[verified]** — [Rasp et al., WeatherBench 2, arXiv:2308.15560](https://arxiv.org/pdf/2308.15560).
- ECMWF notes that a smooth, ensemble-mean-like forecast gives lower average error in less predictable situations (the double penalty). It also says its FuXi ML model "shares more similarities with the ENS mean than with any of the other deterministic forecasts". **[verified, via search snippet of ECMWF page]** — [ECMWF: A new ML model in the ECMWF web charts](https://www.ecmwf.int/node/28925).
- Leith (1974) "Theoretical skill of Monte Carlo forecasts", *Mon. Wea. Rev.* 102, 409–418, is the classic source for the ensemble mean beating a single forecast in RMSE because unpredictable components are averaged out. **[from memory, unverified]**

### Inferences
- That ML deterministic forecasts and the ENS mean look alike spectrally, and that the ENS mean filters scale-selectively, is already established (WeatherBench 2, ECMWF FuXi note, Surcel 2014). The paper should cite these and should not claim the ML/ENS-mean equivalence as new at the spectral level.
- The step from "averaging removes features whose position is uncertain" (Feng 2020; Surcel 2014) to "the surviving fine structure is a copy of the coarse structure" is natural, but nobody I found writes it down. The new part is the cross-scale observable and its measurement, not the filtering itself.

### Gaps
- I could not access full texts of Surcel 2014/2015 to check whether they examine where in the coarse field the surviving small scales sit, for example co-location with large-scale forcing.
- I found no study of the ENS mean that uses amplitude-envelope, bispectral or phase-coupling diagnostics across scales.

---

## Q2. Predictability theory by scale: which fine structure is predictable?

### Takeaway
Classical theory (Lorenz 1969 and later) predicts upscale error growth: small scales saturate first and their unforced part becomes unpredictable within hours to about a day. The fine structure still present in a conditional mean at 120 h is therefore what is deterministically forced by the predictable large scales: fronts, jets, orography. This directly supports the "slaved component" interpretation. The 24 h → 120 h onset fits the known timescales of error saturation at about 200–800 km. None of these papers discusses envelope coupling.

### Cited Findings
- WeatherBench 2 finds that the ENS-mean spectral drop-off starts at small wavelengths and moves to larger ones with lead time, as in Lorenz's 1969 upscale error growth. **[verified]** — [arXiv:2308.15560](https://arxiv.org/pdf/2308.15560).
- Durran & Gingrich (2014) "Atmospheric predictability: why butterflies are not of practical importance", *J. Atmos. Sci.* 71(7), 2476–2488. With a k^-5/3 background spectrum, the loss of predictability from small, fixed-amplitude initial errors is nearly independent of their spatial scale. **[verified via AGU abstract; page range 2476–2488 likely but one citation gives 2476–2478]** — [AGU FM14 abstract](https://agu.confex.com/agu/fm14/webprogram/Paper3830.html).
- Sun & Zhang (2016) "Intrinsic versus practical limits of atmospheric predictability and the significance of the butterfly effect", *J. Atmos. Sci.* 73, 1419–1438. In a moist, convectively unstable environment, rapid upscale growth makes forecast error increasingly insensitive to the scale and amplitude of initial perturbations. **[verified, abstract]** — [PSU PDF](https://adapt.psu.edu/ZHANG/papers/SunZhang2016JAS.pdf).
- Surcel et al. (2017): predictability lost below about 200 km within 24 h because of large-scale IC/BC and physics uncertainty. **[verified]** — [CAPS PDF](https://twister.caps.ou.edu/papers/Surcell_etal_MWR2017.pdf).
- The following are standard references that were **[from memory, unverified]** in this session:
  - Lorenz (1969), *Tellus* 21, 289–307: finite predictability of multiscale flows.
  - Tribbia & Baumhefner (2004), *MWR* 132, 703–713: upscale error cascade, with baroclinic scales then dominating.
  - Zhang et al. (2007), *JAS* 64, 3579–3594: multistage error growth from convective to meso- to synoptic scales.
  - Selz (2019), *JAS* 76, 1293–1305: intrinsic limit of about 15 days, estimated with a stochastic convection scheme.
- Subich et al. (2025) note that 2 m temperature hardly smooths even in standard GraphCast, which they attribute to "the strong influence of elevation". This is an example of fine structure tied to a fixed or predictable forcing surviving MSE training. **[verified]** — [arXiv:2501.19374v2](https://arxiv.org/html/2501.19374v2).

### Inferences
- Predictability theory splits fine-scale variance into (i) a forced or balanced part that is a function of the predictable large-scale state (frontogenesis on a predictable trough, jet-entrance ageostrophic circulations, orographic waves) and (ii) a free or intrinsic part that saturates within hours to days. A conditional mean keeps (i) and averages out (ii). The direction and timing of the finding (zero at 24 h, full at 120 h) therefore follow from textbook theory.
- The Subich observation on 2 m temperature is anecdotal support. Forced fine structure is exactly what MSE keeps.

### Gaps
- I found no paper that quantifies what fraction of fine-scale (200–800 km) 850 hPa vorticity variance at day 5 is "forced/slaved" versus "free". That fraction is what would predict the 24–35 % magnitude.

---

## Q3. Statistical / optimal estimation and ML: the conditional mean removes unpredictable variance. Is the coupling increase a known consequence?

### Takeaway
That the L2-optimal point forecast is the conditional mean, and that it damps each scale by its forecast–truth coherence, is textbook and has recently been formalised for weather ML. A simple identity (law of total covariance) also predicts that correlation with anything measurable from the conditioning information is inflated in the conditional mean. That is a qualitative theoretical expectation for the over-inheritance sign. However, no weather or ML paper I found states or measures it as an increase in cross-scale coupling. Note also that the "uniform loss" null is analytically guaranteed to show no effect for a log-envelope correlation, so on its own it is a weak control.

### Cited Findings
- Gneiting (2011) "Making and evaluating point forecasts", *JASA* 106(494), 746–762, doi:10.1198/jasa.2011.r10138. The squared-error-optimal (Bayes) point forecast is the mean of the predictive distribution, and a scoring function should be consistent with the functional being targeted. **[verified]** — [RePEc record](https://ideas.repec.org/a/bes/jnlasa/v106i494y2011p746-762.html); [arXiv:0912.0902](https://alphaxiv.org/abs/0912.0902).
- Subich, Husain, Separovic & Yang (2025) "Fixing the double penalty in data-driven weather forecasting through a modified spherical harmonic loss function", ICML 2025, PMLR 267:57191–57211, arXiv:2501.19374. **[verified]** — [PMLR](https://proceedings.mlr.press/v267/subich25a.html); [arXiv html v2](https://arxiv.org/html/2501.19374v2). Main points:
  - Expected MSE is minimised at σ_X = ρ σ_Y. Per total wavenumber, the optimal amplitude ratio equals the forecast–truth coherence, so fine, low-coherence scales are damped.
  - MSE splits into an amplitude term and a decorrelation term, 2√(PSD_x PSD_y)(1 − Coh_k).
  - Effective resolution of GraphCast at 5 days is about 1,250 km, about 160 km after AMSE fine-tuning.
  - The loss "does not model relations between different total wavenumbers" and relies on approximate independence of scale-separated modes. Cross-scale structure is explicitly outside their framework.
- Bonavita (2023/2024), "On the limitations of data-driven weather forecasting models", arXiv:2309.08473: Pangu-Weather forecasts lack the physical consistency of physics-based models, and their deterministic-score advantage is largely due to these "peculiarities" (smoothing). **[verified as arXiv preprint; journal version not confirmed in this session]** — [arXiv:2309.08473](https://arxiv.org/pdf/2309.08473). The often-cited title "On some limitations of current ML weather prediction models" (GRL 2024) could **not** be confirmed.
- WeatherBench 2: AI models lose small-scale variability between 6 h and about 3 days, then keep a roughly constant spectrum, which the authors link to 24–72 h training-optimisation windows. **[verified, via search summary of the paper]** — [arXiv:2308.15560](https://arxiv.org/pdf/2308.15560). This is relevant to the 24 h vs 120 h contrast: in ML models, blurring is partly set by the training window and is not purely an effect of error growth.
- Lippe et al. (2023) "PDE-Refiner: achieving accurate long rollouts with neural PDE solvers", NeurIPS 2023, arXiv:2308.05732. MSE-trained neural PDE surrogates neglect "non-dominant", mostly high-frequency spatial components, and diffusion-style refinement fixes this. **[verified]** — [arXiv:2308.05732](https://arxiv.org/abs/2308.05732v1).
- Fan et al. (2025/2026) "Learning more physically realistic dynamics in ML-based weather forecasting with latent-space constraints", arXiv:2510.04006. Variable-wise losses ignore error covariance and produce "overly smooth and physically unrealistic long-range forecasts". The paper is about cross-variable rather than cross-scale coupling. **[verified]** — [arXiv:2510.04006](https://arxiv.org/abs/2510.04006).
- Hassanzadeh et al. (2026) "Missing the butterfly and predicting the past: features or bugs of accurate AI weather models?", arXiv:2608.25835. AI models miss the butterfly effect and "implicitly learn how fast, small scales affect large scales", which the authors attribute to coarse-graining of training data. **[verified, abstract]** — [arXiv:2608.25835](https://arxiv.org/abs/2608.25835). This is relevant to the framing that fine scales in AI models act as functions of large scales, but it does not discuss the conditional mean or envelope coupling.
- GenCast (Price et al. 2025, *Nature* 637, 84–90, doi:10.1038/s41586-024-08252-9) is the generative ensemble whose single members serve as "realisations" in the finding. **[verified]** — [RePEc](https://ideas.repec.org/a/nat/nature/v637y2025i8044d10.1038_s41586-024-08252-9.html).

### Inferences (theory, my derivation; no single source found that states it for this observable)
- **Law of total covariance argument.** Let I be the information the forecast conditions on (the initial state) and F the fine-band quantity. For any h that is a function of I: Cov(E[F|I], h) = Cov(F, h), while Var(E[F|I]) ≤ Var(F). So corr(E[F|I], h) = corr(F, h) / η, where η = corr(F, E[F|I]) ≤ 1 (Pearson's correlation ratio). At 120 h the coarse band (800–1600 km) is still largely predictable, so it is approximately a function of I. The conditional mean's fine band therefore correlates more with it than the truth's fine band does, by a factor of about 1/η_fine. η_fine falls with lead time, so the inflation grows from about 0 at 24 h to large at 120 h. This is regression-dilution / attenuation logic run in reverse: truth's cross-scale correlation is "diluted" by the independent fine component that the conditional mean removes.
  - Caveats: the observable is the log *envelope* of a band-passed conditional mean, not the conditional mean of an envelope. The coarse band is itself only partly predictable at 120 h. So the identity is heuristic, not exact. It still gives the sign and the lead-time dependence, and roughly 1/η as an order of magnitude.
  - Implication for novelty: the *sign* of over-inheritance is theoretically expected from elementary probability once one accepts that coarse scales are more predictable than fine ones. This should be stated openly as the expectation, which the measurement then confirms and quantifies.
- **The uniform-loss null is mathematically guaranteed.** Multiplying a band by a constant c adds log c to its log envelope everywhere, and Pearson correlation is invariant to that. "Same spectrum, structure untouched → no change" is therefore an identity, not an empirical finding. The informative control is the *selective* one, plus ideally a spectrum-matched null where the removed fine variance is independent of the coarse envelope. That should give no inflation; inflation should appear only when the removed part is the coarse-independent part.
- ML (GraphCast/Pangu/Aurora) and the ENS mean showing the same signature, while single members, HRES and a GenCast member do not, is what the conditional-mean view predicts. It is consistent with the ECMWF FuXi ≈ ENS-mean observation and with WeatherBench 2. The cross-scale-coupling version of this equivalence is new.

### Gaps
- I found no paper that derives or measures how a conditional mean or MSE training changes inter-scale amplitude-envelope correlation, bispectrum or phase coupling in weather fields.
- I could not verify whether Bonavita's paper appeared in GRL.

---

## Q4. Turbulence / LES / slaving: conditional averages, stochastic estimation, amplitude modulation

### Takeaway
Turbulence theory has the exact formal objects: the ideal LES subgrid model as a conditional average given the resolved field (Langford & Moser 1999), linear stochastic estimation (Adrian), the slaving principle and inertial manifolds, and large-to-small amplitude modulation measured with band envelopes and Hilbert transforms (Mathis et al. 2009). Together they make two points. A conditional average keeps only the part of the small scales that is a function of the large scales. The envelope correlation of band-passed fields is a standard measure of cross-scale coupling. The Langford & Moser result that the conditional average explains only a small part of the subgrid force in isotropic turbulence means the independent part is large. Removing it should therefore strongly change coupling statistics, but nobody seems to have measured that.

### Cited Findings
- Langford & Moser (1999) "Optimal LES formulations for isotropic turbulence", *J. Fluid Mech.* 398, 321–346. **[verified title/venue; volume/pages from memory]** — [Cambridge](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/optimal-les-formulations-for-isotropic-turbulence/70E8F0564EADC6DB0DBA1CF2450A791F). Main points:
  - The ideal subgrid model is a conditional average over real fields whose large scales match the LES field. It reproduces all single-time multi-point statistics and has minimum instantaneous error.
  - It is approximated by stochastic estimation from two-point correlations.
  - In forced isotropic turbulence, the errors between the real and estimated subgrid force were "nearly as large as the subgrid force itself".
- Haselbacher, Moser et al. (2002, CTR summary) continue the optimal-LES line. **[verified]** — [CTR PDF](https://dept.aem.umn.edu/~kmahesh/globalassets/publpdf/conference/haselbacher_ctrsum_02.pdf).
  - Note: the claim that an ideal LES "reproduces all single-time multi-point statistics" refers to the resolved-scale statistics of an LES driven by the conditional-mean *subgrid force*. It is not a claim that a conditional-mean *field* keeps the statistics of the small scales. This distinction matters when citing it.
- Mathis, Hutchins & Marusic (2009) "Large-scale amplitude modulation of the small-scale structures in turbulent boundary layers", *J. Fluid Mech.* 628, 311–337, doi:10.1017/S0022112009006946. **[verified]** — [Univ. Melbourne record](https://findanexpert.unimelb.edu.au/scholarlywork/313259-large-scale%20amplitude%20modulation%20of%20the%20small-scale%20structures%20in%20turbulent%20boundary%20layers).
  - The method is to Hilbert-transform the band-filtered small-scale signal and correlate its envelope with the large-scale signal. This is methodologically the same family as the log-envelope inter-band correlation used in the finding, and should be cited as the methodological ancestor.
  - The modulation coefficient increases with Re. A cross-flow comparison appeared in *Phys. Fluids* (2009). **[verified]** — [Univ. Melbourne record](https://findanexpert.unimelb.edu.au/scholarlywork/315723-comparison-of-large-scale-amplitude-modulation-in-turbulent-boundary-layers--pipes--and-channel-flows).
- The following are **[from memory, unverified]**:
  - Adrian (1977/1979) and Adrian & Moin (1988, *JFM* 190, 531–559) on linear stochastic estimation. The LSE estimate of small scales is a linear function of the condition, i.e. perfectly "slaved".
  - Haken's slaving principle (*Synergetics*, 1977/1983).
  - Inertial-manifold and approximate-inertial-manifold theory (Foias, Temam and coworkers), in which the small scales are approximated as a function Φ of the large scales. This is the deterministic analogue of the conditional mean.

### Inferences
- In this language, a conditional-mean forecast replaces the fine scales by their "approximate inertial manifold / stochastic estimate" given the predictable large scales. By construction, this maximises the share of fine-scale variance that is a function of the coarse scales. Higher envelope coupling is the expected statistical signature of such a parameterisation. The finding can be framed as an empirical measurement of the "slaved fraction" seen through amplitude-modulation statistics.
- The Langford & Moser result that the conditional mean captures a small fraction of subgrid variance, combined with the finding (24–35 % coupling inflation), suggests a quantitative bridge. The inflation size should scale with the unexplained fraction (1/η). That bridge has not been made explicit in the literature.

### Gaps
- I found no turbulence paper that compares amplitude-modulation coefficients between a DNS field and its conditional-mean / LSE / filtered-ensemble-mean reconstruction. A related but unchecked area is wall-model "predictive inner-outer models" (Marusic, Mathis & Hutchins 2010, *Science*), which reconstruct small scales from large scales via a universal signal plus modulation. This is **[from memory, unverified]**.

---

## Q5. Is there a known result that conditional averaging INCREASES cross-scale envelope correlation? Analogues in image super-resolution and statistical downscaling

### Takeaway
No direct statement was found in meteorology, ML weather, image processing or downscaling. The closest analogues are three. (a) Image restoration: the MMSE/posterior-mean estimate averages over plausible high-frequency explanations and leaves the off-manifold, blurry result (Blau & Michaeli 2018). (b) Statistical downscaling: deterministic regression ("perfect prognosis") gives only the large-scale-explained local variance. Variance inflation of that output was criticised because the resulting fine structure is fully determined by the large scale and misrepresents spatial and temporal structure (von Storch 1999; Maraun 2013). (c) Ensemble post-processing: the probability-matched mean restores the one-point PDF but not the spatial structure (Surcel 2014). Downscaling (b) is the closest conceptual precedent, "fine structure = deterministic function of coarse predictors". It has been discussed qualitatively but not quantified as an envelope-coupling inflation.

### Cited Findings
- Blau & Michaeli (2018) "The perception-distortion tradeoff", CVPR 2018, arXiv:1711.06077. **[verified]** — [CVF open access](https://openaccess.thecvf.com/content_cvpr_2018/papers/Blau_The_Perception-Distortion_Tradeoff_CVPR_2018_paper.pdf); [arXiv](https://arxiv.org/pdf/1711.06077).
  - The MMSE estimate is the posterior mean, a likelihood-weighted average of all explanations. It often lies off the natural-image manifold and is blurry.
  - Lower distortion necessarily makes outputs more distinguishable from real images, for any distortion measure.
- Posterior-Mean Rectified Flow (Ohayon et al., ICLR 2025, arXiv:2410.00418) predicts the posterior mean and then transports it to the image distribution. This is an explicit two-stage "conditional mean + restore realism" design. **[verified, abstract]** — [arXiv:2410.00418](https://arxiv.org/pdf/2410.00418).
- Maraun (2013) "Bias correction, quantile mapping, and downscaling: revisiting the inflation issue", *J. Climate* 26(6), 2137–2143, doi:10.1175/JCLI-D-12-00821.1. **[verified]** — [GEOMAR repository](https://oceanrep.geomar.de/id/eprint/21175).
  - Quantile mapping or variance inflation used to downscale gridbox averages to points causes problems like those of inflation in perfect-prognosis downscaling. These include misrepresented spatial and temporal structure and overestimated area-mean extremes.
  - Stochastic, randomised small-scale variability is needed instead.
- von Storch (1999) "On the use of 'inflation' in statistical downscaling", *J. Climate* 12, 3505–3506. Inflation is not meaningful, and randomisation (adding random small-scale variability) should be used instead. **[bibliographic details verified; content via secondary sources only]** — [Hereon publications](https://publications.hereon.de/id/18870/index.php); content as described in [Maraun 2013](https://oceanrep.geomar.de/id/eprint/21175/1/jcli-d-12-00821.1.pdf).
- Surcel et al. (2014): the PMM has member-like intensity PDFs but different spectral and spatial structure. **[verified, abstract]** — [synapsesocial record](https://synapsesocial.com/papers/W2045144288).
- The following are **[from memory, unverified]**:
  - Ledig et al. (2017, CVPR) "Photo-realistic single image super-resolution using a GAN" (SRGAN): MSE-trained SR gives over-smooth textures.
  - Mathieu, Couprie & LeCun (2016, ICLR) "Deep multi-scale video prediction beyond mean square error": an L2 loss gives blurry predictions because it averages over possible futures.

### Inferences
- In the image and downscaling literature the posterior mean is criticised for *missing texture* (variance, perceptual realism). The structural point here goes further. The texture that remains is not a random subsample of the true texture: it is concentrated where the conditioning signal is strong (edges in images, strong coarse envelopes in weather). Restoring variance by uniform inflation, as in downscaling inflation or a constant spectral rescaling, would keep the over-coupling. The "inflation" debate (von Storch 1999; Maraun 2013) is the closest precedent for this logic and supports the finding's design choice of contrasting uniform and selective variance loss.
- Practical corollary, consistent with the literature: spectral fixes such as AMSE fine-tuning (Subich 2025) or variance inflation may restore spectra without restoring the coupling statistic. Only adding fine variance that is *independent* of the coarse field (randomisation, generative sampling as in GenCast) should restore truth-like coupling. This is a testable, not-yet-published prediction: apply the metric to AMSE-fine-tuned GraphCast versus GenCast members.

### Gaps
- I found no paper that reports an *increase* in inter-scale amplitude-envelope correlation (or bicoherence, or amplitude-modulation coefficient) as a consequence of conditional averaging, MSE training or deterministic downscaling. Searches on "envelope correlation", "cross-scale coupling", "smoothing" and "ML weather" returned nothing relevant.
- The search was limited (about 20 queries). A dedicated alphaXiv or Google Scholar sweep on "amplitude modulation" × "ensemble mean", and on "bicoherence" × "neural weather model", is still recommended before claiming priority.

---

### Summary judgment for the report writer (inferences, consolidated)
- **Textbook / known, cite and do not claim:**
  - The conditional mean is the L2-optimal forecast (Gneiting 2011).
  - It damps each scale by its coherence (Subich 2025).
  - The ensemble mean filters unpredictable small scales and is not a physical state (Leith 1974; Surcel 2014; BAMS use-case paper).
  - Arithmetic averaging flattens features with uncertain location (Feng 2020).
  - Deterministic ML forecasts resemble the ENS mean spectrally (WeatherBench 2; ECMWF FuXi note).
  - Upscale error growth makes fine scales unpredictable first (Lorenz 1969 and later).
  - The conditional mean of small scales given large scales is a formal object in optimal LES / stochastic estimation (Langford & Moser 1999; Adrian).
  - Envelope-based amplitude-modulation diagnostics exist (Mathis et al. 2009).
- **Theoretically expected, not empirically new as a sign.** A one-line law-of-total-covariance argument predicts that a conditional mean inflates correlation with any predictable quantity by about 1/η. The sign and lead-time growth of over-inheritance should be presented as the predicted consequence. The uniform-loss null is an identity for log-envelope correlations and should be described as such.
- **Apparently new (no prior statement found):**
  1. Identifying an inter-scale amplitude-envelope coupling statistic as a diagnostic that tells conditional-mean forecasts apart from realisations, beyond spectra.
  2. Its measurement: 24–35 % over-coupling at 120 h, about 0 at 24 h, for 200–1600 km bands of 850 hPa vorticity.
  3. Showing that ML deterministic models and the physical ENS mean are indistinguishable on this *structural* (not merely spectral) axis, while HRES, single ENS members and GenCast members are truth-like.
  4. The selective-vs-uniform variance-loss demonstration as a mechanism test, best strengthened with a spectrum-matched independent-noise null.
  5. The testable prediction that spectral-loss fixes (AMSE) restore spectra but not coupling, whereas generative sampling restores both.
