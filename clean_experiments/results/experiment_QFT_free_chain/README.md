# QFT-Free-Chain Artifacts

This folder contains outputs of `clean_experiments/experiment_QFT_free_chain.py`.

Main files:

- `qft_free_chain_all_summaries.csv`: one row per `(N, m, T)` case with headline diagnostics.
- `qft_free_chain_size_checks.csv`: aggregate over the default 3 size checks (`N=32,64,128`).
- `qft_free_chain_all_pair_defects.csv`: intertwinement defects for each layer pair `n -> n+1`.
- `qft_free_chain_all_layers.csv`: fitted effective-generator parameters by layer.

Per-case subfolders (`case_*`) contain:

- `qft_free_chain_summary.csv`
- `qft_free_chain_pair_defects.csv`
- `qft_free_chain_layer_fit.csv`
- `qft_free_chain_lambda_timeseries.csv`

Conventions used:

- `defect_exact_*`: exact pushforward baseline (expected near machine zero).
- `defect_unitary_*`: fitted Markovian quadratic-only generator.
- `defect_dephasing_*`: fitted Markovian quadratic + dephasing generator.
- `A_mu_*`: candidate inter-layer connection mismatch norm from projected effective generators.
- `Lambda_matter_uniform` / `Lambda_matter_modecount`: weighted aggregates over RG layers.
