# Controlled training results

sample SD (ddof=1); unbounded two-sided 95% Student-t CI; paired objective differences. All accuracy values below are percentages.

| Dataset | Labels | Selection | Mask seed | Objective | Seeds | Mean ± sample SD | 95% CI |
|---|---|---|---|---|---|---|---|
| stanfordcars | clip | prototype | None | ce_soft | [42, 43, 44] | 78.4024 ± 0.0400 | [78.3030, 78.5017] |
| stanfordcars | clip | prototype | None | fcl_auc | [42, 43, 44] | 79.1402 ± 0.0613 | [78.9878, 79.2926] |
| stanfordcars | clip | random_global_exact | 42 | fcl_auc | [42, 43, 44] | 81.4078 ± 0.1528 | [81.0282, 81.7874] |
| stanfordcars | qwen | prototype | None | fcl_auc | [42, 43, 44] | 72.2547 ± 0.0249 | [72.1929, 72.3165] |

## Paired objective differences

Difference in percentage points, fcl_auc − ce_soft.

| Dataset | Labels / selection | Mask seed | Mean difference ± sample SD | 95% CI |
|---|---|---|---|---|
| stanfordcars | clip / prototype | None | 0.7379 ± 0.1013 | [0.4863, 0.9895] |
