# Controlled training results

sample SD (ddof=1); unbounded two-sided 95% Student-t CI; paired objective differences. All accuracy values below are percentages.

| Dataset | Labels | Selection | Mask seed | Objective | Seeds | Mean ± sample SD | 95% CI |
|---|---|---|---|---|---|---|---|
| food101 | clip | prototype | None | ce_soft | [42, 43, 44] | 94.5954 ± 0.0251 | [94.5329, 94.6579] |
| food101 | clip | prototype | None | fcl_auc | [42, 43, 44] | 94.7789 ± 0.0082 | [94.7584, 94.7994] |
| food101 | clip | random_global_exact | 42 | fcl_auc | [42, 43, 44] | 94.6719 ± 0.0258 | [94.6079, 94.7360] |
| food101 | qwen | prototype | None | fcl_auc | [42, 43, 44] | 93.6251 ± 0.0359 | [93.5358, 93.7144] |

## Paired objective differences

Difference in percentage points, fcl_auc − ce_soft.

| Dataset | Labels / selection | Mask seed | Mean difference ± sample SD | 95% CI |
|---|---|---|---|---|
| food101 | clip / prototype | None | 0.1835 ± 0.0332 | [0.1010, 0.2660] |
