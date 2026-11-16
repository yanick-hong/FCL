# 本轮受控实验台账

快照时间（UTC）：2026-11-16T03:26:38.056275+00:00。这是已取回证据的快照，未完成任务保持待完成。

实验源码：`1fec7c895c11a80493f0153e6ec2870ffa322bed`，分支 `reproduction/recover-fcl`。服务器根目录 `/root/autodl-tmp`。配置见 [controlled_campaign.yaml](../../../configs/controlled_campaign.yaml)，协议和目录用途见 [controlled_experiments.md](../../controlled_experiments.md)。

基础准备和回归验证已通过：七数据集 CLIP 缓存及 manifest；Qwen 官方固定 revision 的全部文件校验；72 项测试和 12 项子测试。初次无约束解码失败的资料保存在 `unconstrained_attempt_qwen/` 和 `preparation/worker_status_unconstrained_attempt.json`，不属于正式矩阵。

队列状态：**running**；当前数据集 `tinyimagenet200`，阶段 `qwen`。完整日志、JSONL 标签、checkpoint、配置和 seed metrics 在远程 `outputs/controlled_correction_20261115/`，由 Git 忽略；本目录保存小型审计证据与汇总。

队列按每个数据集依次执行五项要求：完整 Qwen 标注与原型校正、CLIP 随机 30% 校正比较、随机固定 mask 的 AUC-C、同原型 mask 的 soft-CE、四组各训练三个种子并汇总。掩码选择种子与训练种子分开，三次训练始终复用同一 mask。

| 数据集 | train / val / test | 类别 | Qwen 三 split | 四组 × 三种子 |
|---|---|---|---|---|
| dtd | 1880 / 1880 / 1880 | 47 | completed | completed |
| caltech101 | 2945 / 901 / 4831 | 101 | completed | completed |
| stanfordcars | 7356 / 788 / 8041 | 196 | completed | completed |
| eurosat | 21600 / 2700 / 2700 | 10 | completed | completed |
| cifar100 | 45000 / 5000 / 10000 | 100 | completed | completed |
| food101 | 65650 / 10100 / 25250 | 101 | completed | completed |
| tinyimagenet200 | 90000 / 10000 / 10000 | 200 | running | pending |

合计 328,502 张；划分和类别数以实际缓存为准，未按论文数字重分数据。Cars 的 7356 个训练样本与论文 7410 不同，Caltech 缓存排除背景、实际 101 类；差异保留在报告中。

## 已完成的标签准确率

下表只列有完整 split 证据的实测百分比；校正只发生在训练集。

| 数据集 | 来源 | train | val | test | 原型校正后 train | 随机 30% seed42 后 train |
|---|---|---|---|---|---|---|
| dtd | qwen | 60.2128% | 58.9362% | 58.5638% | 79.2021% | 71.7553% |
| dtd | clip | 52.2872% | 50.5319% | 52.8723% | 73.4574% | 66.6489% |
| caltech101 | qwen | 64.8557% | 68.8124% | 72.7386% | 78.6757% | 75.0424% |
| caltech101 | clip | 92.8693% | 93.8957% | 85.0756% | 97.4533% | 95.1783% |
| stanfordcars | qwen | 53.9967% | 52.4112% | 54.1475% | 71.3703% | 68.2028% |
| stanfordcars | clip | 76.5905% | 77.2843% | 76.7069% | 86.3377% | 83.5508% |
| eurosat | qwen | 44.3843% | 44.3704% | 43.7778% | 67.6435% | 61.0000% |
| eurosat | clip | 43.0046% | 43.5185% | 43.7407% | 62.4306% | 60.0741% |
| cifar100 | qwen | 47.1067% | 47.4400% | 46.5600% | 67.0644% | 62.9756% |
| cifar100 | clip | 75.5800% | 76.3200% | 75.5900% | 87.0889% | 82.8511% |
| food101 | qwen | 51.4836% | 51.4554% | 54.3525% | 72.1478% | 66.0960% |
| food101 | clip | 89.0160% | 89.0990% | 92.0594% | 96.0899% | 92.3229% |

原型按预测类别分别向上取整，实际预算通常略高于 30%；随机 30% 按全局最近整数样本数取整。下表另列与原型人数相同的随机对照，避免把额外校正人数与选择策略的效果混为一谈。所有随机对照在此表采用选择种子 42。

| 数据集 / 来源 | 原型校正人数 / 实际比例 | 随机 30% 人数 / 实际比例 | 匹配原型人数的随机校正后 train |
|---|---|---|---|
| dtd / qwen | 583 / 31.0106% | 564 / 30.0000% | 72.1277% |
| dtd / clip | 586 / 31.1702% | 564 / 30.0000% | 67.1809% |
| caltech101 / qwen | 917 / 31.1375% | 884 / 30.0170% | 75.4160% |
| caltech101 / clip | 914 / 31.0357% | 884 / 30.0170% | 95.2122% |
| stanfordcars / qwen | 2286 / 31.0767% | 2207 / 30.0027% | 68.5971% |
| stanfordcars / clip | 2293 / 31.1718% | 2207 / 30.0027% | 83.8363% |
| eurosat / qwen | 6486 / 30.0278% | 6480 / 30.0000% | 61.0231% |
| eurosat / clip | 6485 / 30.0231% | 6480 / 30.0000% | 60.0833% |
| cifar100 / qwen | 13545 / 30.1000% | 13500 / 30.0000% | 63.0200% |
| cifar100 / clip | 13546 / 30.1022% | 13500 / 30.0000% | 82.8778% |
| food101 / qwen | 19741 / 30.0701% | 19695 / 30.0000% | 66.1340% |
| food101 / clip | 19738 / 30.0655% | 19695 / 30.0000% | 92.3305% |

## 已完成的分类器测试结果

均值、样本标准差和 95% Student-t 区间均以三次独立训练为单位；区间不代表重新划分数据或改变 VLM 标签的全部不确定性。

| 数据集 | 来源 / 选择 / 目标 | 测试均值 ± 样本标准差（%） | 95% CI（%） |
|---|---|---|---|
| dtd | clip / prototype / ce_soft | 70.6028 ± 0.0813 | [70.4010, 70.8047] |
| dtd | clip / prototype / fcl_auc | 64.9291 ± 0.1107 | [64.6540, 65.2041] |
| dtd | clip / random_global_exact / fcl_auc | 68.3156 ± 0.4028 | [67.3151, 69.3161] |
| dtd | qwen / prototype / fcl_auc | 59.9468 ± 0.1918 | [59.4704, 60.4232] |
| caltech101 | clip / prototype / ce_soft | 96.3224 ± 0.1042 | [96.0636, 96.5812] |
| caltech101 | clip / prototype / fcl_auc | 95.6876 ± 0.0119 | [95.6579, 95.7173] |
| caltech101 | clip / random_global_exact / fcl_auc | 92.6102 ± 0.1076 | [92.3430, 92.8774] |
| caltech101 | qwen / prototype / fcl_auc | 85.5861 ± 0.0978 | [85.3431, 85.8291] |
| stanfordcars | clip / prototype / ce_soft | 78.4024 ± 0.0400 | [78.3030, 78.5017] |
| stanfordcars | clip / prototype / fcl_auc | 79.1402 ± 0.0613 | [78.9878, 79.2926] |
| stanfordcars | clip / random_global_exact / fcl_auc | 81.4078 ± 0.1528 | [81.0282, 81.7874] |
| stanfordcars | qwen / prototype / fcl_auc | 72.2547 ± 0.0249 | [72.1929, 72.3165] |
| eurosat | clip / prototype / ce_soft | 86.9012 ± 0.0566 | [86.7607, 87.0418] |
| eurosat | clip / prototype / fcl_auc | 92.8519 ± 0.0370 | [92.7598, 92.9439] |
| eurosat | clip / random_global_exact / fcl_auc | 94.0617 ± 0.0932 | [93.8302, 94.2933] |
| eurosat | qwen / prototype / fcl_auc | 94.4815 ± 0.0000 | [94.4815, 94.4815] |
| cifar100 | clip / prototype / ce_soft | 82.4633 ± 0.0551 | [82.3265, 82.6002] |
| cifar100 | clip / prototype / fcl_auc | 83.4200 ± 0.0100 | [83.3952, 83.4448] |
| cifar100 | clip / random_global_exact / fcl_auc | 83.6333 ± 0.0321 | [83.5535, 83.7132] |
| cifar100 | qwen / prototype / fcl_auc | 81.1167 ± 0.0757 | [80.9286, 81.3048] |
| food101 | clip / prototype / ce_soft | 94.5954 ± 0.0251 | [94.5329, 94.6579] |
| food101 | clip / prototype / fcl_auc | 94.7789 ± 0.0082 | [94.7584, 94.7994] |
| food101 | clip / random_global_exact / fcl_auc | 94.6719 ± 0.0258 | [94.6079, 94.7360] |
| food101 | qwen / prototype / fcl_auc | 93.6251 ± 0.0359 | [93.5358, 93.7144] |

DTD 的 12 个最佳 checkpoint 已在 CPU 独立重算，测试正确数精确一致；695 项审计全部通过。范围和证据见 [独立审计](dtd_independent_audit.json)。

Caltech-101 的 12 个最佳 checkpoint 已在 CPU 独立重算，测试正确数精确一致；663 项审计全部通过。范围和证据见 [独立审计](caltech_independent_audit.json)。

已完成整组训练 72/84 次；运行中的组不提前计入完成数。

## 论文基线后续队列

状态：**pending_core**；已完成 0/84 项。详见 [状态证据](baseline_followup/followup_status.json)和 [完整计划](baseline_followup/followup_plan.json)。

该有限队列先等待上述七数据集核心队列全部成功，再执行六个主数据集 × CLIP/Qwen × 七种其余方法；84 项中 12 项为 zero-shot 评估、72 项为 100-epoch seed42 训练。FCL 已在核心队列按来源各三种子执行，不重复计数。输入沿用核心的同一缓存和原型纠错 mask，逐项验证完整输出后才计为完成。

此后续队列只覆盖主表其余方法；论文的超参数、提示词、逐 epoch 测试曲线和特征骨干消融仍须另行完成，不包含在 84 项基线工作中。

模型下载来源、文件 SHA、实际软件版本、Cars 官方标签与压缩包校验见 `preparation/`；每个 manifest 的缓存、类别表、样本顺序、原始索引及官方标注文件指纹见 `manifests/`。完整矩阵只在七个数据集都完成且总汇总通过后标为完成。

正文/附录逐表逐图的范围、历史协议冲突和剩余缺口见 [论文实验覆盖清单](paper_coverage_inventory.md)。其中不可确认的历史模板或指标语义单独列明，不能用当前重建设置冒充已经恢复的历史事实。
