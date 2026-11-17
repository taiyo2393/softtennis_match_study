# Validation-only threshold selection

Rule: maximize presence recall subject to presence precision >= 0.90.
Selected threshold: **0.30** (threshold030_median120).

| threshold | presence precision | presence recall | localized recall | mean error (px) | selected |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 0.30 | 94.4% | 70.2% | 49.6% | 20.3 | yes |
| 0.40 | 96.5% | 67.8% | 48.8% | 18.6 |  |
| 0.50 | 97.5% | 64.5% | 46.3% | 18.9 |  |
| 0.60 | 98.3% | 47.1% | 33.1% | 19.9 |  |
| 0.70 | 100.0% | 3.3% | 2.5% | 11.8 |  |

The test split is not read by this script and must only be evaluated after this selection is frozen.
