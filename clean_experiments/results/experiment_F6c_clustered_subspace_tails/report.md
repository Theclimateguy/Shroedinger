# Experiment F6c Report: Clustered Subspace Tail Fits

## Protocol (predeclared anti-overfit)
- Clustering variables are fixed before fitting: fractal surrogate (z), convective activity (z), vertical exchange proxy (z).
- Number of clusters fixed: K=6.
- Multiple testing correction fixed: bonferroni across cluster LLR p-values.
- Target alpha band fixed: [1.50, 2.00].

## Global tail (reference)
- alpha_mle = 3.161476
- xmin = 3.882716e-05
- dynamic_range = 25.231
- LLR(PL-Exp) = 2184.557548
- LLR p = 7.239659e-78

## Strict outcomes
- any_cluster_passes_corrected_strict = False
- PASS_ALL = False

## Files
- experiment_F6c_cluster_table.csv
- experiment_F6c_cluster_sample_sizes.csv
- experiment_F6c_cluster_tail_metrics.csv
- experiment_F6c_cluster_best_fits.csv
- experiment_F6c_verdict.json
- plot_F6c_global_empirical_tail.png
- plot_F6c_cluster_alpha.png
