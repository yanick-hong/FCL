# Controlled training results

sample SD (ddof=1); unbounded two-sided 95% Student-t CI; paired objective differences. All accuracy values below are percentages.

| Dataset | Labels | Selection | Mask seed | Objective | Seeds | Mean ± sample SD | 95% CI |
|---|---|---|---|---|---|---|---|
| dtd | clip | prototype | None | ce_soft | [42, 43, 44] | 70.6028 ± 0.0813 | [70.4010, 70.8047] |
| dtd | clip | prototype | None | fcl_auc | [42, 43, 44] | 64.9291 ± 0.1107 | [64.6540, 65.2041] |
| dtd | clip | random_global_exact | 42 | fcl_auc | [42, 43, 44] | 68.3156 ± 0.4028 | [67.3151, 69.3161] |
| dtd | qwen | prototype | None | fcl_auc | [42, 43, 44] | 59.9468 ± 0.1918 | [59.4704, 60.4232] |

## Paired objective differences

Difference in percentage points, fcl_auc − ce_soft.

| Dataset | Labels / selection | Mask seed | Mean difference ± sample SD | 95% CI |
|---|---|---|---|---|
| dtd | clip / prototype | None | -5.6738 ± 0.1339 | [-6.0063, -5.3412] |
