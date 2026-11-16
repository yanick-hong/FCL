# Controlled training results

sample SD (ddof=1); unbounded two-sided 95% Student-t CI; paired objective differences. All accuracy values below are percentages.

| Dataset | Labels | Selection | Mask seed | Objective | Seeds | Mean ± sample SD | 95% CI |
|---|---|---|---|---|---|---|---|
| eurosat | clip | prototype | None | ce_soft | [42, 43, 44] | 86.9012 ± 0.0566 | [86.7607, 87.0418] |
| eurosat | clip | prototype | None | fcl_auc | [42, 43, 44] | 92.8519 ± 0.0370 | [92.7598, 92.9439] |
| eurosat | clip | random_global_exact | 42 | fcl_auc | [42, 43, 44] | 94.0617 ± 0.0932 | [93.8302, 94.2933] |
| eurosat | qwen | prototype | None | fcl_auc | [42, 43, 44] | 94.4815 ± 0.0000 | [94.4815, 94.4815] |

## Paired objective differences

Difference in percentage points, fcl_auc − ce_soft.

| Dataset | Labels / selection | Mask seed | Mean difference ± sample SD | 95% CI |
|---|---|---|---|---|
| eurosat | clip / prototype | None | 5.9506 ± 0.0566 | [5.8101, 6.0912] |
