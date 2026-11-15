# AUC-C 参数敏感性消融

本消融只改变 AUC-C 的一个参数：

- $μ$ sweep: $μ=0,0.25,0.5,0.75,1$，固定 $τ=0.5$；
- $τ$ sweep: $τ=0.25,0.5,1,2$，固定 $μ=0.5$。

所有实验使用100%正确的 corrected subset，因此不混入人工纠正错误率因素。五个
已准备数据集（DTD、Caltech-101、EuroSAT、CIFAR-100、Tiny-ImageNet）各使用
42/43/44三个种子，共135个独立训练。特征、CLIP posterior、原型过滤掩码、
observed labels、数据划分和训练协议保持不变；每个参数值都从独立初始化开始。

训练使用冻结CLIP ViT-L/14特征、线性分类头、P=16、K=16、$λ=1$，最多80轮，
验证准确率10轮不提高时早停。结果包括逐epoch history、参数快照、最佳/末轮权重、
metrics.json和汇总图表。测试集只在恢复最佳验证权重后使用一次。

```bash
python ablation/parameter_sensitivity/run.py
python ablation/parameter_sensitivity/summarize.py --campaign <campaign>
```
