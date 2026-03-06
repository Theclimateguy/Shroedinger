# F6c Spatial Panel Visualization

Patch-wise tail fits over |Lambda_local(t,y,x)|.
- panel shape: nt=4380, ny=81, nx=161
- patch grid: 12x25, patch_size=15, stride=6
- fit-available patches: 300/300
- strict SOC-candidate patches (dynamic+LLR+alpha in [1.5,2.0]): 0
- alpha min / q10 / median: 2.7546 / 2.9790 / 3.2272

## Main files
- experiment_F6c_spatial_patch_tail_metrics.csv
- experiment_F6c_spatial_summary.csv
- experiment_F6c_spatial_top20_low_alpha.csv
- plot_F6c_spatial_alpha_map.png
- plot_F6c_spatial_ks_map.png
- plot_F6c_spatial_dynamic_range_map.png
- plot_F6c_spatial_log10p_map.png
- plot_F6c_spatial_soc_candidate_mask.png
- plot_F6c_alpha_vs_f5_composite.png
