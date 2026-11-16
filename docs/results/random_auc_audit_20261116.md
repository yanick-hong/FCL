# Random AUC-C versus prototype AUC-C audit

日期：2026-11-16。此审计只读检查受控实验代码和归档结果，没有修改训练源码、配置、论文或远端 HEAD。

## 结论

没有发现随机 AUC-C 使用了更容易的目标、更多纠错样本或错误标签路径。随机结果在部分数据集上较高是真实归档结果，但标签质量和分类器测试精度是不同指标。原型筛选的样本是特征离群点；它们虽然包含更高比例的伪标签错误，却也可能是线性分类器更难学习的样本。随机 mask 还会改变 `y_obs` 的类别组成和 PKBatchSampler 的批次，因此这是选择策略比较，不是完全相同输入的成对优化比较。

| Dataset | Prototype AUC-C | Random global exact AUC-C | Random - prototype (pp) |
|---|---:|---:|---:|
| CIFAR-100 | 83.4200 | 83.6333 | +0.2133 |
| EuroSAT | 92.8519 | 94.0617 | +1.2099 |
| DTD | 64.9291 | 68.3156 | +3.3865 |
| Food-101 | 94.7789 | 94.6719 | -0.1069 |
| Stanford Cars | 79.1402 | 81.4078 | +2.2675 |
| Caltech-101 | 95.6876 | 92.6102 | -3.0773 |

纠错标签质量方向相反。例如 CIFAR-100 原型校正标签准确率为 87.0889%，随机 seed 42 为 82.8511%；EuroSAT 原型为 62.4306%，随机 seed 42 为 60.0741%。原型选中区域的错误精度也更高（CIFAR-100：38.2327% 对 24.2370%；EuroSAT：64.7032% 对 56.8990%）。因此随机 AUC-C 的测试准确率较高不能解释为随机标签校正更准确。

## 代码核查

- `scripts/compare_label_corrections.py` 按 CLIP 伪标签分组，计算归一化原型和 `1-cos(feature, center)`，选择每类 `ceil(rho*n_class)` 个最大距离样本；选择过程不读取真值。
- 随机 mask 是不放回均匀抽样。`random_global_exact` 使用 `floor(rho*N+0.5)`；另有 `random_matched_prototype` 条件匹配原型实际纠错人数。
- 选中位置才替换为真值并设置 `s=0`；未选中位置保留伪标签并设置 `s=1`。`run_controlled_training.py` 会复核缓存、伪标签、mask、样本顺序、真值替换和哈希。
- `batch_objective` 对 `s=0` 使用 CE，对 `s=1` 使用 AUC-C；AUC-C 后验与 CLIP 先验均按实现要求 detach。现有核心测试覆盖 AUC-C 数值/梯度、detach、mask 重放和训练指纹，审计记录为 13 项通过。

## 可比性限制

CIFAR-100 的原型/随机 global exact 纠错数为 13,546/13,500，EuroSAT 为 6,485/6,480。差异很小，但严格预算比较应使用已保存的 `random_matched_prototype` mask。更大的差异来自 mask 组成：PKBatchSampler 按 `y_obs` 分组，不同 mask 会改变类别平衡批次。

建议后续同时报告 matched-budget random AUC-C；若要完全隔离 sampler 影响，再增加固定 index batch stream 的次级复跑。本次没有用这些建议性复跑替换现有结果。
