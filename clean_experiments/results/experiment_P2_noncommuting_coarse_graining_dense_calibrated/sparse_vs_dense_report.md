# P2 Sparse vs Dense Comparison (Calibrated C009)

## Compact table
panel,panel_rows,panel_events,tile_rows,tile_events,mae_gain_all,r2_gain_all,perm_p_all_max_scale,event_positive_frac,min_fold_gain,pass_all,mean_comm_defect_mean_scale,mean_comm_active_minus_calm_scale,mean_rho_purity_scale,mean_abs_lambda_dm_scale
sparse_72x24,72,24,30168,24,7.427353274072466e-06,0.0033916520811646,0.02,0.7361111111111112,-7.053644117257471e-06,True,0.17573935335255356,0.971817768108413,0.9159777290590375,0.027810440486489337
dense_240x16,240,16,100560,16,3.097983009667672e-07,0.0021003574473055,0.8,0.6875,-9.53741720911475e-06,False,0.15978800757434208,0.8942076758797453,0.916007411303644,0.025341626917381635


## Per-scale
scale_l,sparse_mae_gain,sparse_r2_gain,sparse_perm_p_value,sparse_event_positive_frac,sparse_min_fold_gain,sparse_PASS_ALL,sparse_comm_defect_mean,sparse_comm_active_minus_calm,dense_mae_gain,dense_r2_gain,dense_perm_p_value,dense_event_positive_frac,dense_min_fold_gain,dense_PASS_ALL,dense_comm_defect_mean,dense_comm_active_minus_calm,mae_gain_sign_stable,r2_gain_sign_stable
16.0,6.336639168070564e-06,0.0021818375252353,0.02,0.75,-7.053644117257471e-06,True,0.1350408075375731,0.4099966046591966,1.351097270893957e-06,0.0009215190027392,0.02,0.8125,-2.2812327723632612e-06,True,0.1232044623711743,0.4383705197967638,True,True
32.0,7.767883929863219e-06,0.002180939132256,0.02,0.8333333333333334,-2.5285153465703767e-06,True,0.0477235241905648,0.1457687815033302,4.321955702509008e-06,0.0019299716295931,0.02,0.75,-5.997999864812524e-06,True,0.0475712778616958,0.1434865207192629,True,True
8.0,7.65577956277888e-06,0.0035206602274192,0.02,0.625,-6.99511215309078e-06,True,0.3444537283295228,2.359687918162712,-1.3307608806014878e-07,0.0022179769048996,0.8,0.5,-9.53741720911475e-06,False,0.3085882824901562,2.100765987123209,False,True


## Success criteria (requested)
- perm_not_worse: `False`
- event_positive_frac_up: `False`
- mae_gain_not_smaller: `False`
- r2_gain_not_smaller: `False`
- pass_all_preserved: `False`

## Interpretation
- dense ingest does not preserve full sparse performance; scale-specific degradation is present.
- mae_gain sign flip detected at scales: `['8.0']`.
