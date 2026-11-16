# Controlled training results

sample SD (ddof=1); unbounded two-sided 95% Student-t CI; paired objective differences. All accuracy values below are percentages.

| Dataset | Labels | Selection | Mask seed | Objective | Seeds | Mean ± sample SD | 95% CI |
|---|---|---|---|---|---|---|---|
| cifar100 | clip | prototype | None | ce_soft | [42, 43, 44] | 82.4633 ± 0.0551 | [82.3265, 82.6002] |
| cifar100 | clip | prototype | None | fcl_auc | [42, 43, 44] | 83.4200 ± 0.0100 | [83.3952, 83.4448] |
| cifar100 | clip | random_global_exact | 42 | fcl_auc | [42, 43, 44] | 83.6333 ± 0.0321 | [83.5535, 83.7132] |
| cifar100 | qwen | prototype | None | fcl_auc | [42, 43, 44] | 81.1167 ± 0.0757 | [80.9286, 81.3048] |

## Paired objective differences

Difference in percentage points, fcl_auc − ce_soft.

| Dataset | Labels / selection | Mask seed | Mean difference ± sample SD | 95% CI |
|---|---|---|---|---|
| cifar100 | clip / prototype | None | 0.9567 ± 0.0513 | [0.8292, 1.0841] |
