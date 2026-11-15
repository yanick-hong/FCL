# Controlled training results

sample SD (ddof=1); unbounded two-sided 95% Student-t CI; paired objective differences. All accuracy values below are percentages.

| Dataset | Labels | Selection | Mask seed | Objective | Seeds | Mean ± sample SD | 95% CI |
|---|---|---|---|---|---|---|---|
| caltech101 | clip | prototype | None | ce_soft | [42, 43, 44] | 96.3224 ± 0.1042 | [96.0636, 96.5812] |
| caltech101 | clip | prototype | None | fcl_auc | [42, 43, 44] | 95.6876 ± 0.0119 | [95.6579, 95.7173] |
| caltech101 | clip | random_global_exact | 42 | fcl_auc | [42, 43, 44] | 92.6102 ± 0.1076 | [92.3430, 92.8774] |
| caltech101 | qwen | prototype | None | fcl_auc | [42, 43, 44] | 85.5861 ± 0.0978 | [85.3431, 85.8291] |

## Paired objective differences

Difference in percentage points, fcl_auc − ce_soft.

| Dataset | Labels / selection | Mask seed | Mean difference ± sample SD | 95% CI |
|---|---|---|---|---|
| caltech101 | clip / prototype | None | -0.6348 ± 0.1140 | [-0.9180, -0.3516] |
