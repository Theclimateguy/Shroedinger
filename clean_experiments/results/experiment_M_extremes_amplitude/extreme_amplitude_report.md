# Experiment M Extreme-Event Amplitude Analysis

## Thresholds
- vorticity P90 threshold: 1.932790841e-05
- omega*q P90 threshold: 0.0004440173152

## Model gain summary
- full:
  vertical: gain=0.003412, p=0.0062, n=4380
  horizontal: gain=0.003379, p=0.0062, n=4380
  combined: gain=0.002566, p=0.0125, n=4380
- union_p90:
  vertical: gain=-0.004578, p=0.5016, n=804
  horizontal: gain=-0.004892, p=0.5327, n=804
  combined: gain=-0.007362, p=0.7383, n=804
- intersection_p90:
  vertical: gain=-0.027561, p=0.1340, n=72
  horizontal: gain=-0.029830, p=0.1090, n=72
  combined: gain=-0.152906, p=1.0000, n=72
- non_union:
  combined: gain=0.006986, p=0.0031, n=3576
  horizontal: gain=0.006973, p=0.0031, n=3576
  vertical: gain=0.006579, p=0.0031, n=3576

## Amplitude summary
- full: |residual| ratio=1.000, |lambda_h| ratio=1.000, |lambda_v| ratio=1.000
- union_p90: |residual| ratio=1.458, |lambda_h| ratio=0.867, |lambda_v| ratio=0.840
- intersection_p90: |residual| ratio=2.175, |lambda_h| ratio=0.670, |lambda_v| ratio=0.649
- non_union: |residual| ratio=0.897, |lambda_h| ratio=1.030, |lambda_v| ratio=1.036

## Tail response amplitude
- full / lambda_h: |delta|=0.256481, CI95=[-0.388322, -0.122359]
- full / lambda_v: |delta|=0.256802, CI95=[-0.389592, -0.121525]
- union_p90 / lambda_h: |delta|=0.187113, CI95=[-0.451627, 0.075055]
- union_p90 / lambda_v: |delta|=0.174337, CI95=[-0.452932, 0.108141]
- intersection_p90 / lambda_h: |delta|=0.507432, CI95=[-0.212319, 1.149374]
- intersection_p90 / lambda_v: |delta|=0.507432, CI95=[-0.175228, 1.168443]
- non_union / lambda_h: |delta|=0.298357, CI95=[-0.425496, -0.170693]
- non_union / lambda_v: |delta|=0.308859, CI95=[-0.432912, -0.184359]
