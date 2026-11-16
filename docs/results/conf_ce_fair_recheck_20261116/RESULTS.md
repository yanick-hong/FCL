# Conf+CE 与 FCL+CE 公平复查结果

状态：50/50次正式训练完成；独立审计通过，30个最终best checkpoint均已在CPU重新计算验证集和测试集准确率。

比较范围是两种纠错样本选择策略之后使用同一种混合硬标签CE。两组纠错人数、特征缓存、划分和参数搜索预算相同。以下均为测试准确率百分数，±表示三个训练种子的样本标准差，差值单位为百分点。

| Dataset | Conf+CE mean ± SD | FCL+CE mean ± SD | FCL − Conf (pp) | LR (Conf / FCL) |
|---|---:|---:|---:|---:|
| CIFAR-100 | 82.733 ± 0.015 | 81.810 ± 0.053 | -0.923 | 0.004 / 0.004 |
| Caltech-101 (101 classes) | 96.895 ± 0.218 | 90.927 ± 0.043 | -5.968 | 0.004 / 0.004 |
| EuroSAT | 66.593 ± 0.064 | 60.802 ± 0.113 | -5.790 | 0.004 / 0.004 |
| Food-101 | 95.147 ± 0.028 | 94.486 ± 0.008 | -0.661 | 0.004 / 0.004 |
| Tiny-ImageNet-200 | 83.733 ± 0.023 | 82.220 ± 0.035 | -1.513 | 0.004 / 0.004 |

按数据集三种子均值比较，FCL+CE领先 0/5，Conf+CE领先 5/5。本轮结果不支持“FCL+CE至少在两到三个数据集占优”。全部候选与所有不利结果均保留。

## 每个种子与区间

| Dataset | Method | Seed 42 | Seed 43 | Seed 44 | Mean 95% t CI |
|---|---|---:|---:|---:|---|
| CIFAR-100 | Conf+CE | 82.750 | 82.720 | 82.730 | [82.695, 82.771] |
| CIFAR-100 | FCL+CE | 81.790 | 81.870 | 81.770 | [81.679, 81.941] |
| Caltech-101 (101 classes) | Conf+CE | 96.916 | 97.102 | 96.667 | [96.353, 97.437] |
| Caltech-101 (101 classes) | FCL+CE | 90.892 | 90.913 | 90.975 | [90.820, 91.034] |
| EuroSAT | Conf+CE | 66.667 | 66.556 | 66.556 | [66.433, 66.752] |
| EuroSAT | FCL+CE | 60.926 | 60.704 | 60.778 | [60.521, 61.084] |
| Food-101 | Conf+CE | 95.152 | 95.172 | 95.117 | [95.077, 95.217] |
| Food-101 | FCL+CE | 94.479 | 94.495 | 94.483 | [94.465, 94.506] |
| Tiny-ImageNet-200 | Conf+CE | 83.760 | 83.720 | 83.720 | [83.676, 83.791] |
| Tiny-ImageNet-200 | FCL+CE | 82.260 | 82.200 | 82.200 | [82.134, 82.306] |

## 实际配置与限制

- 两方法同一学习率网格0.001/0.002/0.004；seed42按最高验证准确率选LR，并列选较小LR。所有十个条件均选中0.004，该结论限于预设网格。
- AdamW、weight decay 0.0001、CosineAnnealingLR(T_max=100)、100epochs、patience101、batch1024；相同Xavier线性头与全数据shuffle，梯度裁剪5。
- CPU执行，torch/BLAS线程2、DataLoader workers2，最多三个数据集同时运行。现有GPU队列、源码与配置未改。
- seed42参与LR选择并被纳入42/43/44汇总；95% t区间按自由度2计算，只作给定数据、标签、mask及选中LR下的探索性、条件性描述。
- 恢复的Caltech为101类，与原文102类不同；新缓存不能证明与所有历史稿件缓存相同。新数据属于独立受控复查，未替换论文旧数值，论文实验说明和设置未改。
- 独立pilot仅用于资源/CLI检查，未计入50次正式训练和参数选择。

## 文件位置

- 本地易读汇总：本文件；机器可读汇总：`summary.csv`、`results.json`；全部候选：`all_trials.csv`。
- 每次命令、配置、原始metrics、验证曲线和seal：`trials/<dataset>/<method>/lr-*/seed-*/`。
- 输入与源码：`plan.json`、`preflight.json`、`study_binding.json`、`search_protocol.json`；独立审计：`final_audit.json`。
- 本地完整模型与运行日志：`F:/CodeProjects/FCL2/outputs/conf_ce_fair_recheck_20261116/`；完整备份：同级同名`.zip`。
- AutoDL原件：`/root/autodl-tmp/outputs/conf_ce_fair_recheck_20261116/`；审计原件：`/root/autodl-tmp/outputs/conf_ce_fair_audits/final-audit-20261116.json`。
- 正式实验源码HEAD：`1fec7c895c11a80493f0153e6ec2870ffa322bed`。

完整备份SHA256：`caf163ae6aa4660fbfe41f38b5115dca3e90c1a60e1cd63363b21f69a9a20e67`。
