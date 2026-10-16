# Priority sweep: cross-scale wavelet-modulus / envelope coupling applied to model-produced weather and climate fields

Date of sweep: 2026-10-15. Diagnostic being protected: the "inheritance fidelity" metric. It is the Spearman correlation between log amplitude envelopes of adjacent octave bands (200-400 / 400-800 / 800-1600 km) of 850 hPa relative vorticity. Each value is anchored by subtracting the median over phase-randomised surrogates, and the metric is reported as forecast minus ERA5. The headline finding is that deterministic ML models (GraphCast, Pangu, Aurora) and the IFS ENS mean over-couple adjacent scales by x1.24-1.35 at 120 h, while single realisations do not.

**Bottom line.** One clear precedent was found for using a cross-scale wavelet-modulus statistic as a realism metric for a model that emulates weather fields: **Brochet et al. (2023, AIES)**. They use second-order scattering coefficients (s21, s22) as RMSE metrics comparing GAN-generated fields with AROME-EPS km-scale NWP fields. This weakens any claim that "cross-scale modulus statistics have never been used to evaluate weather-model or emulator output", and Brochet et al. must be cited. The precedent differs from the inheritance metric in every respect that carries the finding:
- target: unconditional generative samples vs the NWP climatology, not a forecast vs analysis at matched valid times;
- domain: km-scale 10 m wind and 2 m temperature over France, not synoptic 850 hPa vorticity;
- statistic: S2/S1 measures how variable the fine-scale modulus is at a coarser scale, not a correlation between adjacent-band envelopes;
- anchoring: no surrogate anchoring;
- systems: no deterministic ML forecasts or ensemble means;
- result: no over-coupling or conditional-mean result.

No paper was found that measures cross-scale amplitude or envelope coupling, via scattering, AM coefficients, envelope correlations or cascade-weight correlations, in **medium-range NWP or ML forecasts, ensemble means, or global ML weather/climate emulators**.

### Closeness ranking (most to least close)

| # | Work | What it does | Why it is not the inheritance result |
|---|---|---|---|
| 1 | Brochet, Raynaud, Thome, Plu & Rambour 2023, AIES 2(4), doi:10.1175/AIES-D-23-0006.1 [V] | Second-order scattering s21 = S2(j1,j2)/S1(j1) ("sparsity", i.e. organisation of the scale-j1 modulus at scale j2) and s22 ("shape"). Used as RMSE metrics of a GAN vs AROME-EPS. Shown to separate fields that a spectrum-matched Gaussian cannot. | Unconditional generator vs the NWP distribution. Km-scale surface fields. No forecast-vs-truth pairing, no conditional mean, no deterministic ML forecasts, no surrogate anchor. The S2/S1 ratio is a sparsity/intermittency index, not an envelope-envelope correlation between adjacent bands. |
| 2 | Gauvrit, Bouin, Delouis & Boulanger: EGU22 (WOAST), EGU23 abstract, Earth Space Sci. 12(2) e2024EA003865 (2025) [V] | WST used to quantify the "modulation of turbulent scales by submeso- to mesoscales" in the marine ABL (EUREC4A). EGU abstracts mention applying it to LES and asking whether "the models show evidence of such scale couplings". | Turbulence scale (m-km). The published paper's abstract covers observations only. No forecast or emulator evaluation. Cross-scale coupling is the explicit object, so it is conceptually close. |
| 3 | Skinner, Lawrence & Callies 2025, arXiv:2505.00819 (submitted to GRL) [V-abstract/HTML] | Scattering transform on SSH from MITgcm (~2 km) and on vorticity from idealised 2D/SQG runs. Separates flows with near-identical spectra. | Characterises regimes inside simulations. Model-vs-observation comparison is only proposed as future work. No forecasts or ML. |
| 4 | Ren, Laima & Li 2019, AIP Advances, doi:10.1063/1.5125566 [V] | Wall-modelled LES of the ABL, checked by the Mathis-type AM coefficient against earlier studies. | Turbulence surface layer. AM coefficient (signed large scale vs small-scale envelope). LES fidelity, not forecast fidelity. |
| 5 | Gallmeier et al. 2023, GMD 16:7143, doi:10.5194/gmd-16-7143-2023 [V] | Evaluates the LLC4320 ocean model against VIIRS SST using the distribution of submesoscale *structure*, not spectra. Cites Cheng & Ménard's scattering guide. | Uses a learned probabilistic autoencoder, not a scattering or envelope statistic. Ocean SST. Usage-level analogue only ("structure beyond spectra" for model evaluation). |
| 6 | Elberkennou et al. 2026, arXiv:2609.18381 (NeurIPS 2026 workshop) [V-abstract/HTML] | Learned critic for realism of GraphCast, Pangu, FuXi, U-Cast, SWIFT vs IFS HRES and ERA5. | Fixed baselines are global-mean WD, spherical convolutional WD, zonal log-spectrum and RBF-MMD. **No scattering, wavelet-modulus or cross-scale statistic.** This shows the current frontier of ML realism evaluation lacks the inheritance-type statistic. |
| 7 | Lovo, Lancelin, Herbert & Bouchet 2025, AIES, doi:10.1175/AIES-D-24-0094.1 [V] | Scattering network (ScatNet) used as an interpretable *predictor* of extreme heatwaves. | A feature extractor, not an evaluation diagnostic. |
| 8 | Lagrange et al., J. Hydrol. 556:972-979 (issue 2018; online 2016), doi:10.1016/j.jhydrol.2016.06.063 [V] | Scattering transform used to classify radar rainfall images. | Observational classification only. |

---

## Q1. Forward citations of Morel et al. 2022 / Cheng et al. 2024 (scattering spectra) and related Allys/Mallat/Cheng work: any application to atmospheric or climate-model fields, ML emulators or generative weather models?

### Takeaway
The scattering-spectra papers (Morel et al.; Cheng et al. PNAS Nexus) have so far been cited almost only in astrophysics, cosmology, finance, turbulence synthesis and methods papers. Their forward citations include no application to NWP/ML forecasts, ensemble means or climate emulators. The only weather-model use of scattering coefficients as an evaluation metric found is Brochet et al. (2023). It comes through the older Cheng & Ménard (2021) "guide" lineage (s21/s22), not through scattering spectra proper.

### Cited Findings
- OpenAlex lists 20 citing works for Cheng, Morel, Allys, Ménard & Mallat (2024) "Scattering spectra models for physics", PNAS Nexus, doi:10.1093/pnasnexus/pgae103. The only atmosphere-related ones are Lovo et al. 2025 (AIES, ScatNet heatwave predictor) and Gauvrit et al. 2025 (ESS, EUREC4A turbulence). A thesis on extreme climate events with rare-event algorithms (2024, doi:10.70675/c778208c...) was not inspected. The rest cover cosmology/astrophysics (weak lensing, CMB, dust, H I), turbulence synthesis, CFD perspective, alloys and gravitational waves — [OpenAlex cites:W4383045342](https://api.openalex.org/works?filter=cites:W4383045342)
- The 13 citers of Morel et al. (2024 ACHA version, doi:10.1016/j.acha.2024.101724) are multifractal fracture surfaces, cosmic-ray diffusion, financial jumps, InSight seismic data, wind-turbine seismic emissions, Hurst estimation and similar. None is atmospheric or climate-model work — [OpenAlex cites:W4404515608](https://api.openalex.org/works?filter=cites:W4404515608)
- The 8 SSRN-version citers include "Predicting Multivariate Air Pollution: A Gaussian-Mixture Nested Factorial VAE" (IEEE GRSL 2024; how it uses scattering was not checked). The rest are finance, images and marine-mammal acoustics. The arXiv:2204.10177 record has 1 citer (the PNAS Nexus paper) — [OpenAlex cites:W4385062734](https://api.openalex.org/works?filter=cites:W4385062734)
- The 21 citers of Cheng & Ménard (2021) "How to quantify fields or textures? A guide to the scattering transform", arXiv:2112.01288, include Brochet et al. 2023 (AIES), Gallmeier et al. 2023 (GMD, LLC4320 vs VIIRS SST), Prochaska et al. 2023 (TGRS, SST patterns via contrastive learning) and Gauvrit et al. 2025. The rest are cosmology — [OpenAlex cites:W3216678128](https://api.openalex.org/works?filter=cites:W3216678128)
- **Brochet et al. 2023** define S2(j1,j2) = <| |x*ψ_{j1}| * ψ_{j2} |>, s21 = S2/S1 ("sparsity": "the organisation of the S1 map is important to account for the image global structure"), and s22 (Cheng & Ménard's "shape" estimator). They state that "a Gaussian noise field with the exact spectrum of AROME-EPS samples exhibits lower s21". RMSE between GAN and AROME-EPS s21/s22 gives "two new metrics per variable", used next to PSD errors. The abstract concludes there is a "need to go beyond spectral analysis in generation quality assessment" — [Brochet et al. 2023 preprint PDF](https://thome.isir.upmc.fr/papers/Brochet-AIES-23.pdf); [DOI record](https://doi.org/10.1175/aies-d-23-0006.1)
- Brochet et al. 2023 report that reducing the number of generated variables gives "drastic improvement" on the scattering metrics: "The GAN is therefore much more able to identify and generate multi-scale organisation" — [Brochet et al. 2023 PDF](https://thome.isir.upmc.fr/papers/Brochet-AIES-23.pdf)
- They also cite earlier meteorological classification with scattering coefficients ("Garcia et al. 2015"; not retrieved, **unverified**) — [Brochet et al. 2023 PDF](https://thome.isir.upmc.fr/papers/Brochet-AIES-23.pdf)
- The follow-up Brochet, Moldovan, Rabault, Regan & Raynaud (2025), AIES 4(1), doi:10.1175/AIES-D-24-0058.1 (StyleGAN-2 enrichment of AROME-EPS), lists its physical-consistency metrics as PSD, bivariate histograms and multiscale sliced Wasserstein distance. Scattering metrics are **not** in the listed evaluation criteria — [AMS accepted PDF via Exa](https://journals.ametsoc.org/view/journals/aies/aop/AIES-D-24-0058.1/AIES-D-24-0058.1.pdf)
- The OpenAlex citers of Brochet 2023 are Rampal et al. 2025 (JAMES cGAN downscaling), a BAMS 2025 ML-emulator perspective, Brochet 2025, a wave-spectrum diffusion paper, a latent-diffusion large-ensemble paper (IEEE TAI 2025), Chatillon 2023 (cloud texture thesis) and Vozár 2026 (representation learning). Abstracts of those checked do not mention scattering — [OpenAlex cites:W4386708769](https://api.openalex.org/works?filter=cites:W4386708769); [Rampal et al. 2025 abstract](https://doi.org/10.1029/2024ms004668)
- Gauvrit et al. (WOAST, EGU22) say the power spectrum "is blind to position consistency between scales". They applied WST to anemometer series and "2D atmospheric simulations (LES)" — [EGU22-9152](https://doi.org/10.5194/egusphere-egu22-9152)
- The EGU23 abstract asks: "Do the models show evidence of such scale couplings?" (applying WST to EUREC4A data and LES) — [EGU23-13685](https://doi.org/10.5194/egusphere-egu23-13685)
- The Gauvrit et al. 2025 ESS abstract reports "strong modulations of the turbulent scales by submeso-to-mesoscales" in convective-cell conditions, from aircraft/turbulence measurements — [ESS 12(2) e2024EA003865](https://doi.org/10.1029/2024ea003865)
- Skinner, Lawrence & Callies (2025) apply the scattering transform to SSH from an MITgcm North Atlantic run (~2 km) and to vorticity in idealised 2D/SQG runs. It distinguishes flows "with nearly identical power spectra". Comparison with models/altimetry is proposed but not done — [arXiv:2505.00819](https://arxiv.org/abs/2505.00819)
- OpenAlex title/abstract search finds 7 works for "scattering transform" + climate and 0 for "scattering transform" + precipitation. The geoscience hits are WOAST, the EUREC4A pair, a 2026 Indian Ocean Dipole ML paper (no abstract), a CMB emulator and a Zenodo thesis — [OpenAlex search](https://api.openalex.org/works?filter=title_and_abstract.search:%22scattering%20transform%22%20climate)
- The 26 "wavelet phase harmonics" hits are all cosmology/astrophysics/methods — [OpenAlex search](https://api.openalex.org/works?filter=title_and_abstract.search:%22wavelet%20phase%20harmonics%22)
- Web searches for scattering + GraphCast/realism, scattering + climate emulator, scattering + diffusion/downscaling found no paper applying scattering statistics to ML weather forecasts or emulators. The closest new realism work, Elberkennou et al. 2026 (arXiv:2609.18381), uses only mean-WD, SCWD, zonal log-spectrum and RBF-MMD baselines — [arXiv:2609.18381](https://arxiv.org/abs/2609.18381)

### Inferences
- **Brochet et al. (2023) is the clearest precedent for *using a wavelet-modulus cross-scale statistic to judge an emulator of NWP output*.** A sentence such as "cross-scale amplitude statistics have not been used to evaluate weather models" would be false and must be avoided. A defensible wording is: "second-order scattering coefficients have been used to assess unconditional GAN emulation of km-scale NWP fields (Brochet et al. 2023). To our knowledge no study has measured adjacent-scale envelope coupling in forecasts verified against analyses, or contrasted conditional-mean forecasts (deterministic ML, ensemble means) with realisations."
- Formally, s21 = S2(j1,j2)/S1(j1) measures the *variability at scale j2 of the fine-scale envelope |x*ψ_{j1}|*. That is close to, but not the same as, correlating the fine envelope with the adjacent coarse-band envelope. A field can have a highly organised fine envelope (high s21) that is spatially uncorrelated with the coarse-band envelope. The inheritance metric is closer to the scattering-spectra cross-scale covariances (envelope-vs-coarse-band and envelope-envelope covariances in Morel et al. / Cheng et al.). Those covariances, however, have not been applied to weather or climate fields in any work found.
- The scattering lineage in meteorology is Paris-centric (Météo-France/CNRM, ENS/Mallat circle, LOCEAN/Ifremer). A forthcoming preprint from that community (e.g. scattering spectra on AROME/diffusion ensembles or on ArchesWeatherGen) is the most plausible near-term scoop risk. None was found up to 2026-10-15.

### Gaps
- The Semantic Scholar API was rate-limited (HTTP 429). Forward citations came from OpenAlex, which under-indexes arXiv-only preprints and recent workshop papers, so a 2026 arXiv preprint citing scattering spectra could be missed.
- The full texts of Lovo et al. 2025, the 2024 extreme-events thesis, the GRSL 2024 air-pollution VAE and Vozár 2026 were not read. Their abstracts show no evaluation use.
- "Garcia et al. 2015" (scattering for meteorological classification, cited by Brochet) was not located. **Unverified.**
- The Brochet et al. 2023 scattering results tables (numbers for s21/s22 RMSE) were not extracted. The volume/issue (2(4), Oct 2023) was verified via Crossref; the article number was not.

## Q2. Forward citations of Portilla & Simoncelli 2000 in geoscience/meteorology (texture statistics used to validate generative models of cloud or precipitation)

### Takeaway
No paper was found that uses Portilla-Simoncelli joint statistics (including adjacent-scale magnitude correlations) to validate weather or climate model output or generative weather models. The geoscience citations of P-S are SAR/remote-sensing texture classification, de-speckling and image processing.

### Cited Findings
- OpenAlex records 1,803 citers of Portilla & Simoncelli (2000), IJCV 40:49-70, doi:10.1023/A:1026553619983 — [OpenAlex W2127006916](https://api.openalex.org/works/W2127006916)
- Filtered by title/abstract terms, the citer counts are: precipitation 1, cloud 6, weather 7, climate 1, atmospheric 2, rainfall 0, radar 14, satellite 8. No hit evaluates NWP, GCM, ML-forecast or generative-weather output. Hits include a SAR wind-direction GLCM method, SAR de-speckling, land-cover texture, atmospheric-turbulence video stabilisation and a solar-flare CNN explanation — [OpenAlex filtered queries](https://api.openalex.org/works?filter=cites:W2127006916,title_and_abstract.search:cloud)
- Chatillon (2023 thesis, "Physically constrained generative networks for cloud and texture synthesis") generates cloud images for optical-sensor simulation. The abstract emphasises control of Fourier spectral slope and histograms, not cross-scale magnitude correlations — [theses record](https://doi.org/10.70675/847251bcz5a74z492fz9ac0z77d1e9962653)
- "Characterization of Uncertainty in Remotely-Sensed Precipitation Estimates" (2014, ResearchGate DOI) cites P-S. Its content was not inspected — [record](https://doi.org/10.13140/2.1.2256.2569)

### Inferences
- P-S adjacent-scale magnitude correlation is the closest *formal* ancestor of the inheritance metric (correlation of wavelet magnitudes at neighbouring scales). In meteorology it appears to have been absorbed, if at all, only indirectly via scattering coefficients (Brochet 2023). It should be cited as the formal source, with the statement that it has been used for texture synthesis, not for forecast verification.

### Gaps
- OpenAlex keyword filtering covers titles/abstracts only. A paper using P-S statistics deep in its methods with no abstract mention would be missed. The 2014 precipitation-uncertainty item and the 2019 French thesis on "fonds complexes" were not inspected.

## Q3. Forward citations of Mathis et al. 2009 and Hutchins & Marusic 2007 in atmospheric boundary-layer or weather-model evaluation

### Takeaway
The amplitude-modulation literature has spread widely into the atmospheric *surface layer*: about 69 citers of Mathis 2009 and about 56 of Hutchins & Marusic 2007 match "atmospheric". These are field campaigns (QLOA, Minqin), wind tunnels and LES/WMLES, at turbulence scales. A few LES studies verify their simulated AM against observations, which is a usage analogue. None applies AM or envelope coupling to NWP, GCM, ML forecasts or ensemble means at synoptic scales.

### Cited Findings
- Mathis, Hutchins & Marusic (2009), JFM 628:311-337, doi:10.1017/S0022112009006946, has 781 citers in OpenAlex. Of these, 69 match "atmospheric", 4 "weather", 2 "mesoscale", 4 "climate" and 3 "forecast" — [OpenAlex cites:W2158296088 + atmospheric](https://api.openalex.org/works?filter=cites:W2158296088,title_and_abstract.search:atmospheric)
- Hutchins & Marusic (2007), "Large-scale influences in near-wall turbulence", Phil. Trans. R. Soc. A 365:647-664 (OpenAlex W2100776582; 683 citers), has 56 citers matching "atmospheric". None matches mesoscale — [OpenAlex cites:W2100776582](https://api.openalex.org/works?filter=cites:W2100776582,title_and_abstract.search:atmospheric)
- Ren, Laima & Li (2019), AIP Advances, doi:10.1063/1.5125566, run WMLES of the ABL at Re ~10^6-10^7 and analyse AM of near-wall small scales by outer large scales. The positions of zero AM are "consistent with those found in previous studies", i.e. LES AM is checked against earlier results — [OpenAlex abstract](https://api.openalex.org/works/doi:10.1063/1.5125566)
- Other model-based AM uses in the ABL: Fang & Porté-Agel 2015 (BLM, LES of VLSMs, doi:10.1007/s10546-015-0006-z; AM content not checked) and Salesky & Anderson 2018 (JFM, buoyancy effects on LSM modulation in convective ABL LES, doi:10.1017/jfm.2018.711; title-level). Observational examples: Liu, Wang & Zheng 2019 (JFM, AM in high-Re ASL, doi:10.1017/jfm.2018.906) and Heisel et al. 2024 (PNAS, mesoscale ABL structure across roughness transition, doi:10.1073/pnas.2320216121) — [OpenAlex list](https://api.openalex.org/works?filter=cites:W2158296088,title_and_abstract.search:atmospheric)
- The only climate-dynamics citer is Nian, Huang & Fu (2022), Clim. Dyn., doi:10.1007/s00382-022-06175-7, "Amplitude modulation of relative humidity by wind in Northeast China". It uses observational time series (title-level; abstract not available) — [OpenAlex](https://api.openalex.org/works/doi:10.1007/s00382-022-06175-7)
- Arneodo, Bacry, Manneville & Muzy (1998), PRL 80:708, "Analysis of random cascades using space-scale correlation functions" (OpenAlex W2050135689; 71 citers). Geophysical citers are observational: rainfall multifractality (WRR 2006, Phys. Lett. A 2005, Adv. Water Resour. 2008), "Causality across rainfall time scales revealed by continuous wavelet transforms" (JGR 2010, doi:10.1029/2009jd013016), surface-layer wind intermittency at mesoscales (PRL 2010, doi:10.1103/physrevlett.105.254501; PRE 2010), a wavelet multifractal analysis of satellite cloud images (2002 book chapter) and "Multiscale interactions between surface shear stress and velocity" (JGR 2003). None evaluates model output — [OpenAlex cites:W2050135689](https://api.openalex.org/works?filter=cites:W2050135689)

### Inferences
- The AM coefficient has a usage tradition of "check that LES reproduces observed AM". That is a fidelity-diagnostic use, but at turbulence scale and for resolved-turbulence models, not for forecasts. It is a legitimate precedent for the *idea* "cross-scale amplitude coupling as a model-fidelity check". It should be cited as such, with the scale, field and system differences made explicit.

### Gaps
- Keyword filtering cannot see AM content inside LES-validation papers whose abstracts do not mention it. Of the ~120 atmospheric citers, only titles were scanned, plus one abstract (Ren et al.).

## Q4. Meteorology papers using "envelope correlation", "amplitude modulation", "cross-scale modulation", "scale interaction index" or cascade-weight correlations to compare forecasts or ensemble means with analyses

### Takeaway
None found. Scale-dependent verification in meteorology decomposes errors or spread by scale band (ECMWF scale-dependent ensemble verification; wavelet spatial verification by Buschow & Friederichs; Casati intensity-scale). Wavelet model-evaluation scores compare per-scale energies. None correlates envelopes or amplitudes *across* bands. The Lovejoy-Schertzer cascade analyses of NWP (already known) remain the closest forecast-oriented cross-scale use.

### Cited Findings
- ECMWF's scale-dependent verification of ensemble forecasts splits fields into planetary, synoptic and sub-synoptic bands. It finds short-range over-dispersion mainly at synoptic scales. This is per-band, not cross-band coupling — [ECMWF memo 10324](https://www.ecmwf.int/sites/default/files/elibrary/2008/10324-scale-dependent-verification-ensemble-forecasts.pdf)
- Wavelet-based verification (Buschow & Friederichs 2020, ASCMO 6:13; Buschow, Pidstrigach & Friederichs 2019, GMD 12:3401; "SAD" EMS 2021) verifies scale, anisotropy and direction of precipitation forecasts from wavelet energy spectra. There is no inter-scale envelope coupling — [ASCMO 2020](https://ascmo.copernicus.org/articles/6/13/2020/ascmo-6-13-2020.html); [GMD 2019](https://doi.org/10.5194/gmd-12-3401-2019)
- Other wavelet model-evaluation scores compare wavelet coefficients/energies scale by scale. Examples are Livina et al. 2008 (QJRMS, wavelet-coefficient score for 2D climatic fields, doi:10.1002/qj.261), Saux Picart et al. 2012 (GMD 5:223, wavelet spatial comparison of geophysical model fields) and a 2022 Climatic Change multiscale CMIP5 evaluation with wavelet descriptive/diagnostic techniques (doi:10.1007/s10584-021-03269-9) — [QJRMS 2008](https://doi.org/10.1002/qj.261); [GMD 2012](https://doi.org/10.5194/gmd-5-223-2012); [Climatic Change](https://link.springer.com/article/10.1007/s10584-021-03269-9)
- Searches combining "amplitude modulation"/"envelope correlation" with ensemble mean/forecast/reanalysis/synoptic returned only predictability and scale-dependent-verification work. No cross-band envelope statistic appeared — [ECMWF reference](https://www.ecmwf.int/bibcite/reference/75101)

### Inferences
- In forecast verification, the gap "per-band diagnostics exist; cross-band envelope coupling of forecast vs analysis does not" holds up after this sweep.

### Gaps
- No full-text search of QJRMS/MWR/WAF was possible. Absence is inferred from abstract-level and web-index coverage.

## Q5. 2025-2026 ML-weather evaluation preprints using higher-order or cross-scale texture statistics (beyond spectra, CRPS, variogram score)

### Takeaway
Recent ML-weather realism evaluation has moved to learned critics/discriminators and distributional distances (Wasserstein, MMD, spectral divergences). Generative-downscaling evaluation still relies on RALSD/PSD, CRPS, FSS and extremes. No 2025-2026 preprint was found that applies scattering, wavelet-modulus or envelope cross-scale statistics to GraphCast/Pangu/Aurora/GenCast-type forecasts or to ensemble means.

### Cited Findings
- Elberkennou, Demler, Meier, Rispoli, Lehmann & Oskarsson (2026), "Every Fixed Metric Has a Blind Spot: A Learned Atmospheric Critic for Scoring Forecast Realism", arXiv:2609.18381 (NeurIPS 2026 EO workshop). It evaluates GraphCast, Pangu-Weather, FuXi, U-Cast and SWIFT vs IFS HRES and ERA5 Forecast. Its fixed baselines are global-mean WD, spherical convolutional WD, zonal log-spectrum discrepancy and subsampled RBF-MMD. It contains no scattering or wavelet metrics — [arXiv HTML](https://arxiv.org/html/2609.18381)
- A generative precipitation-downscaling intercomparison (arXiv:2512.13987) evaluates with PSD/RALSD and tail metrics (e.g., the U-Net underpowers high wavenumbers). A wavelet-diffusion downscaling generalisation study (arXiv:2609.28749) uses Fourier-amplitude MAE and a 1-D Wasserstein distance. Neither uses texture or cross-scale statistics — [arXiv:2512.13987](https://arxiv.org/pdf/2512.13987); [arXiv:2609.28749](https://arxiv.org/pdf/2609.28749)
- A scattering-based validation exists in 2D *hydrodynamic* turbulence synthesis (arXiv:2506.23659, A&A 2025, doi:10.1051/0004-6361/202453264). It treats WST coefficients as texture statistics with L1 sparsity sensitivity. This is not weather-model output — [arXiv:2506.23659](https://arxiv.org/pdf/2506.23659)
- Searches for multifractal/intermittency/cascade evaluation of GraphCast/Pangu returned none — [search context: WeatherBench 2](https://arxiv.org/pdf/2308.15560); [Bonavita 2024 GRL](https://agupubs.onlinelibrary.wiley.com/doi/10.1029/2023GL107377)

### Inferences
- The field's recognised need is "realism beyond spectra" (learned critics, MMD). Brochet 2023 used scattering for km-scale emulation. The inheritance metric therefore lands in an acknowledged gap with one adjacent precedent. Its distinguishing features are:
  - a single interpretable cross-scale number, anchored on surrogates;
  - verification against analyses at matched valid times;
  - contrast between conditional-mean systems and realisations.
- Positioning suggestion: present inheritance fidelity as a *targeted, interpretable fourth-order cross-scale statistic* in the scattering/P-S family. Cite Brochet et al. 2023 as the prior use of scattering coefficients for weather-emulator realism, and Elberkennou et al. 2026 as evidence that current ML realism benchmarks lack cross-scale coupling.

### Gaps
- The arXiv API query for AROME + scattering/StyleGAN returned nothing (possibly throttled). The alphaXiv full-text tools and Google Scholar "cited by" were not used. ICLR/NeurIPS Climate Change AI workshop proceedings were not systematically browsed.
- **Coverage summary.** Forward-citation lists retrieved in OpenAlex:
  - Cheng et al. 2024 (20)
  - Morel et al. ACHA 2024 (13), SSRN 2023 (8) and arXiv 2022 (1)
  - Cheng & Ménard 2021 (21)
  - Brochet et al. 2023 (7)
  - Arneodo et al. 1998 (71, keyword-filtered)
  - Portilla & Simoncelli 2000 (1,803; 8 keyword filters)
  - Mathis et al. 2009 (781; 5 filters)
  - Hutchins & Marusic 2007 (683; 5 filters)

  Keyword sweeps:
  - OpenAlex: "scattering transform" × {climate, precipitation, weather/atmospheric/cloud/ocean/vorticity}; "scattering spectra"; "wavelet phase harmonics"; scattering × emulator/downscaling × climate/weather.
  - Web: scattering + GraphCast realism; scattering spectra + climate emulator; scattering + QG emulator; multifractal + AI weather; generative downscaling + wavelet texture; scattering + diffusion weather (2025-2026); AM/envelope correlation + ensemble mean.
  - Exa: scattering applied to weather/climate model output (surfaced Brochet 2023).

  Bruns et al. 2000 (AEC) forward citations were not checked for meteorology.
