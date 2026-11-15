# 不完美人工纠正：FCL+AUC-C 消融实验

## 问题与控制变量

使用已生成的五个数据集缓存，不重算特征、不改变过滤结果。仅在 corrected subset
中模拟错误；人工正确率为 100%、95%、90%、85%、80%。原始未纠正标签、CLIP 概率、
校正标记 s、验证和测试标签均保持不变。Food-101 和 Stanford Cars 不参与。

设纠正子集大小为 M、训练集大小为 N、人工正确率为 a，则新增错误数
E=round((1-a)M)，新增错误占全训练集比例为 E/N，而不是 1-a。
当 M/N 约为 0.3 时，五档新增错误约占训练集 0%、1.5%、3%、4.5%、6%。
实际 M/N 可能因逐类向上取整略高于0.3，以每组配置及 metrics 中计数为准。
原型及过滤结果在人工纠正前已经固定，本实验不改变初始 VLM 噪声或重算原型，
因此不用于回答“高 VLM 噪声导致原型污染”的另一项研究问题。

每档、每数据集以种子 42/43/44 独立训练，共 75 次。噪声种子为训练种子+1000，
独立于模型随机数流。各档共用同一初始模型种子，但不继承上一档权重。
三次重复同时反映人工错误实例和训练随机性，不将两者方差分离。

## 错误生成

仅模拟器能使用训练真值：检查原 corrected 标签确实正确；在每个真实类别内部
随机排列 corrected 样本，用归一化类内顺序合并为近似类别均衡的全局顺序。
选前 round((1-accuracy)*N_corrected) 个作为错误样本。各档错误集合嵌套，
同一样本跨档使用同一错误类别，错误类别在非真值类别中均匀抽样。
记录实际错误数/正确率，避免将整数取整后的数值误称为精确百分比。

学习器只接收特征、CLIP 概率、s 和模拟人工标签；训练真值不传入损失。
原方法的 P×K 分组按实际 observed labels 进行，故人工错误也可能通过分组影响
批次组成。这是完整训练流程的敏感性，而不是“固定每个批次”的损失隔离实验。

## 训练与评估

冻结 CLIP ViT-L/14，768维特征，线性分类头；P=16,K=16；rho=0.3，mu=0.5，
lambda=1，tau=0.5。AdamW 创建学习率1e-3、weight_decay=1e-4。
OneCycleLR max_lr=5e-3、div_factor=10、final_div_factor=10、pct_start=0.15，
实际起始调度学习率5e-4。最多80 epoch，验证准确率严格提高时保存 best，
10轮无改善早停，首次达到最高值的 epoch 优先。

加载 best.ckpt 后测试一次，不用测试集选超参数/epoch/随机种子。
每轮记录 train_loss/CE/AUC、验证损失/准确率和学习率。
为减少运行成本使用一阶梯度等价的分块 AUC，实现与原损失数值/梯度回归验证；
不更换损失、不用近似采样对替代完整批内成对项。

## 文件与命令

```bash
source /root/autodl-tmp/scripts/activate_autodl.sh
python ablation/human_correction_noise/test_ablation.py
python ablation/human_correction_noise/run.py
# 完成后独立检查输入未变、噪声嵌套、记录/权重以及汇总完整性
python ablation/human_correction_noise/audit.py --campaign ablation/human_correction_noise/results/<时间戳>
```

run.py 默认生成带北京时间时间戳的 results/<campaign>/，使用 --resume PATH
恢复同一批实验；按配置及源文件指纹拒绝不一致的续跑。
每个 dataset/acc_*/seed_*/ 下保存 metrics.json、config.json、run_metadata.json、
history.csv、best.ckpt、last.ckpt。inputs/ 保存独立扰动标签，不覆盖 dataset/processed。

summarize.py 只汇总一个明确 campaign，输出 summary.csv、paired_drops.csv、
comparison.md、accuracy_vs_human_correctness.png/pdf 和 accuracy_drop.png/pdf。
均值±样本标准差基于3次重复；下降值按同种子与100%基线配对计算，单位为百分点。
实验属于模拟噪声压力测试，不代表真实标注员错误分布，也不单独证明 AUC 优于 CE。
