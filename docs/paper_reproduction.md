# 论文复现协议与证据

这里区分三个层次：论文记录的目标、旧代码可恢复的实现、实际执行得到的结果。`configs/paper_targets.yaml` 只是已发表稿件中的参考百分比，运行器不会把它们复制成实验结果。只有真实运行产生的 `metrics.json` 才能作为复现结果。

本轮新增的随机校正、同 mask 损失对照与三种子矩阵采用独立的 [受控实验协议](controlled_experiments.md)，执行证据见 [台账](results/controlled_20261115/README.md)。该入口转写已核对的附录 Prompt C，并明确使用合法类别编号约束解码；下文通用 annotation CLI 的 A/B/C 模板与其分别记录，不能混用结果。

## 覆盖范围

主实验为 CIFAR-100、Caltech-101、Stanford Cars、EuroSAT、Food-101、Tiny-ImageNet 六个数据集，分别使用 CLIP 与 Qwen 标签来源。对应论文表 2/3 的方法如下。

| 方法 | 训练监督/实现 |
| --- | --- |
| Zero-shot | 直接评估选定来源的伪标签，不训练分类器 |
| FL-C | 仅在已经校正的 `s=0` 子集上使用 CE |
| NL-C | 仅在未校正的 `s=1` 子集上使用伪标签 CE |
| FS-C | 全部训练样本真实标签 CE；这是 oracle 上界 |
| L2B | 现有冻结特征的一步元学习标签/样本重加权实现 |
| DLD | 现有冻结特征的方向标签扩散实现 |
| CSGN | 现有冻结特征的潜变量/噪声建模实现 |
| FCL | 校正样本 CE，加未校正样本的后验加权 OVR pairwise AUC |

主入口为 `scripts/run_paper_experiments.py`，配置为 `configs/paper_reproduction.yaml`。FL-C/NL-C/FS-C 的精确 CE 控制实验位于该入口中，避免旧 `train_trusted.py` 重新选样和旧 `train_noise.py` 额外的蒸馏/EMA改变基线定义。L2B/DLD/CSGN 使用现有项目接口，属于论文采用的冻结特征适配版本；不能把这些结果宣称为原论文端到端模型的完整复现。DivideMix、FixMatch 等仅在相关工作中被引用，不属于表 2/3 主实验；可保留项目已有附加实现，但不会混入这里的主表。

## 共同管线

1. 使用冻结的 CLIP ViT-L/14 提取 768 维图像特征，保存训练/验证/测试特征、标签、类别映射与稳定的样本顺序。原始图像与缓存必须在各方法间相同。
2. CLIP 或 Qwen 为同一批训练样本提供候选标签。按预测类别分组，单位化图像特征，计算类均值并再次单位化，按 `1-cos(feature, prototype)` 从大到小选择每类 `ceil(rho*n_class)` 个样本。
3. 复现实验用数据集真值模拟对选中样本的人工校正，只在这些位置写入 `y_obs` 并令 `s=0`；其他位置保留伪标签，`s=1`。这不是实际的人工标注质量评估。
4. FCL 以 `y_obs` 训练可信 CE。未校正样本使用 `q=mu*softmax(logits).detach()+(1-mu)*p_vlm` 作为固定配对权重，使用正样本应高于负样本方向的 logistic ranking loss。`mu` 与总损失权重 `lam` 是两个参数。
5. 按验证准确率选择最佳 checkpoint，最后才报告该 checkpoint 的测试准确率。多次查看/选择测试结果不应作为超参数调优依据。

默认种子 42，AdamW `lr=1e-3, weight_decay=1e-4`，`rho=0.3, lam=1, mu=0.5`。运行器默认完整执行 100 epochs，令各现有 CLI 的 patience 大于 epoch 数，跟论文图 2 的横轴对齐。不同方法仍保留各自已经恢复的优化器调度/元学习细节；这是可审计的恢复协议，不声称论文未提供的细节已经确定。P/K 配置记录了恢复选择，论文正文只说 batch size 随数据集变化。

## 标签来源与文件隔离

CLIP 默认读取 `${DATA_ROOT}/processed/<dataset>_clip_vit-l-14_embeddings.pt`。Qwen 默认读取 `<dataset>_qwen_prompt-C_embeddings.pt`，可用 `--cache-template` 指定已有派生缓存。Qwen 派生缓存保留同一组 CLIP 图像特征，把训练接口兼容字段 `clip_probs_*` / `clip_argmax_*` 明确替换为 Qwen 来源并在 metadata 中记录 Qwen、prompt、原缓存和样本对齐信息。这些字段在 Qwen 实验里表示 Qwen 分布，名称仅为兼容旧接口。若只有硬标签，应记录分布是 one-hot，而非伪装成生成模型的校准后验。

Qwen 恢复脚本使用 `Qwen/Qwen2.5-VL-7B-Instruct`；论文的 Qwen-VL-7B 命名不能证明二者版本等同。生成或导入必须先完成；运行器不会在缺失 Qwen 文件时使用 CLIP 标签。`--suite prompts` 使用 A/B/C 三种独立 Qwen 缓存；当前内置 prompt 不是已核实的论文原文，需通过 `--prompt-file` 指定确切历史模板。预检核验 `annotation_metadata[split]` 的 `source_models` 与 `replaced_clip_fields=true`：训练需要 train Qwen 标注，Qwen 零样本需要 test Qwen 标注，未替换的 CLIP split 不报告为 Qwen 指标。已知 prompt variant 与请求不一致时拒绝；缺少 A/B/C 映射信息时报告未核验。

校正缓存按数据集、来源、prompt、rho 分开放在 `processed/paper_labels/` 下。`--prepare-labels --execute` 只创建不存在的校正缓存，不覆盖既有文件。已有校正缓存须包含 `y_obs, s`；Qwen 校正缓存必须显式记录 `meta.label_source=qwen`。使用旧路径时可通过 `--obs-template` 显式指定，并自行审计顺序与来源。

## 执行示例

默认只打印计划和真实 CLI，不加载模型、不写实验结果：

```bash
python scripts/run_paper_experiments.py --dry-run
python scripts/run_paper_experiments.py --suite main --label-source qwen --dry-run
```

预检加载缓存，检查字段、长度、有效标签、概率归一化、argmax 一致性、来源 metadata 和论文训练样本数量。`--strict-paper-splits` 将数量/类别不一致升级为错误；相同数量仍不足以证明样本索引相同。

```bash
python scripts/run_paper_experiments.py --preflight --strict-paper-splits
python scripts/run_paper_experiments.py --datasets cifar100 --methods zero_shot fl_c nl_c fs_c fcl --prepare-labels --execute
python scripts/run_paper_experiments.py --label-source qwen --prompt C --prepare-labels --execute
python scripts/run_paper_experiments.py --seeds 42 43 44 --prepare-labels --execute
```

多种子训练默认复用相同的冻结特征/数据划分/筛选缓存，只改变训练随机种子。若要研究划分方差，必须另外重建并显式指定不同的缓存。

论文其他实验的入口：

```bash
# lambda: CIFAR100 / Food101 / TinyImageNet, 0,1,2,5,10,20,50,100
python scripts/run_paper_experiments.py --suite lambda --methods fcl --prepare-labels --execute
# rho: CIFAR100 / StanfordCars / TinyImageNet, .1,.2,.3,.4,.5
python scripts/run_paper_experiments.py --suite rho --methods fcl --prepare-labels --execute
# prompt: StanfordCars / Caltech101 / EuroSAT, A/B/C (requires three Qwen caches)
python scripts/run_paper_experiments.py --suite prompts --methods zero_shot fcl --prepare-labels --execute
# epoch curves: CIFAR100 / Caltech101, all compared methods, 100 epochs
python scripts/run_paper_experiments.py --suite epochs --prepare-labels --execute
```

结果写入 `outputs/paper/<suite>/<source>/<dataset>/<method>/<setting>/seed-<seed>/<timestamp>/`，保留命令、配置、日志、校正缓存路径及结果。路径根遵循 `DATA_ROOT/OUTPUT_ROOT`。不同入口的历史记录字段略有差异，但都保留逐 epoch 验证记录。论文图 2 的图注没有明确 accuracy 曲线对应验证集还是测试集，不能直接认定当前验证曲线与其指标语义相同；如另行报告逐 epoch 测试曲线，应另存各 epoch checkpoint 再离线评估，不用测试精度选模型。失败运行不会被标成已复现。

## 已发现的论文/旧代码差异

| 问题 | 可证实的差异与复现处理 |
| --- | --- |
| rho 阈值 | 论文称“rho quantile 后距离超过阈值”，同时又称选择最远 rho 比例。选择最远 30% 应取 70% 分位阈值；恢复代码采用 top-k，避免文字歧义。每类 ceil 会令实际总体校正率略大于 rho。 |
| lambda 方向 | 早期脚本与注释有 `AUC + lam*CE`，最终论文公式 8/11 为 `CE + lam*AUC`。主入口采用最终论文定义；旧结果不可在不转换含义时混合。 |
| AUC 符号 | 论文列出的 `phi(u)=softplus(u)` 配合 `f_positive-f_negative` 会在最小化时反向排序。实现采用 `softplus(-(positive-negative)/tau)`，并对另一方向对应翻转。 |
| 归一化 | 论文写每类/配对数平均，恢复代码以总后验配对权重归一化。两者数值尺度不同，影响 lam 含义；当前入口调用恢复代码并保留这一差异，不宣称理论式数值逐项一致。 |
| tau | 早期 `train_auc_ce_CIFAR100_new.py` CLI 默认 .1，训练函数默认 .5；较新的主方法固定 pairwise tau=.5。CLIP 生成分布的温度/模型 logit_scale 是另一个参数。主入口以当前 .5 为准。 |
| 后验梯度 | early new.py 中 `p_model` 曾没有 detach，而论文提出固定后验再更新分类器。当前 FCL 实现 detach 后验，避免损失通过权重自我调整。 |
| epochs | 备份包含 60 epochs 的修改提示、100 epochs 曲线、200/300 epochs 默认及早停，版本不统一。主入口明确默认 100，允许 `--epochs` 覆盖并保存配置。 |
| 划分 | CIFAR100 45k/5k/10k 可明确恢复；Caltech 有随机70/30、ImageFolder80/10/10、每类30/10/rest多版本；Food101存在 train/val 命名与实际拆分差异。所有方法必须使用同一缓存，不能只按数据集名称合并。 |
| 预处理 | 最早CLIP脚本用 ImageNet normalization、方形Resize并有B/32目录与L/14模型名不符。最终论文称CLIP preprocessing；恢复的历史缓存必须注明实际预处理，不能用模型名推断。 |
| 先验尖锐度 | 早期CIFAR特征脚本对余弦直接softmax，没有CLIP logit_scale；其他脚本使用原生缩放。argmax可能相同，但AUC权重不同，重提缓存需记录该设置。 |
| 真值可见性 | 旧脚本将全训练真值一起传入loader，有些可信CE直接读y_true。当前FCL可信CE使用y_obs；FL/NL薄封装只使用允许的子集标签。真值仅用于模拟选中样本校正、oracle上界及评估。 |
| 指标冲突 | Table1 Cars zero-shot 70.17，Table2 为71.16；xlsx部分Qwen Food/Tiny基线复用了CLIP行而最终论文不同。参考配置以最终论文表为主并记录冲突，不把旧表当作新测量。 |

预检通过只说明输入可以运行。是否接近论文仍取决于恢复到的权重、样本顺序、预处理、prompt 与训练协议，必须比较实际日志及 checkpoint。

逐表、逐图的覆盖范围和剩余实验见 [论文实验覆盖清单](results/controlled_20261115/paper_coverage_inventory.md)。正文两个主表共 78 个单元，通用入口的 96 项还包含 18 个额外 Qwen CE 控制。Figure 3 的 lambda 原图只有 0、1、2、5、10、20，配置中的 50/100 是扩展；附录 EuroSAT 四种骨干的比较尚无正式套件，底层提取器可复用，但另三种骨干还缺独立缓存和 12 次训练。Prompt B 未公开历史全文和语言，现有通用模板不能冒充原实验。当前协议保留正文与附录的划分冲突；代码可执行不表示全部图表已复现。

本轮其余主表方法在 `outputs/paper_baseline_recheck/` 另设有限跟进队列：六主数据集 × 两来源 × 七种非 FCL 方法，共 84 项（12 个 zero-shot、72 次 seed42/100-epoch 训练）。它先等待七数据集核心队列全部成功，再使用核心同一缓存和原型掩码，逐项做 CPU 输入预检、真实执行和产物独立核验。FCL 继续引用核心的三种子结果。计划、当前状态和最终版包装器验证均保存在 [本轮台账](results/controlled_20261115/README.md)；当前为等待依赖的队列，不能记为已经跑完主表。

### Repeated runs and result summaries

For a remote GPU run, `--num-workers 0` keeps cached tensors in the parent process; `--P` and `--K` override the dataset sampler values, and `--pairwise-tau` is passed to FCL's pairwise logistic loss. Every executed job records SHA256 digests for the feature cache, observed-label cache, and tracked source/config files in `paper_run.json`, together with a checkout dirty flag. Re-running with `--resume` skips a completed job only when these fingerprints and the full job settings still match. A run that raises an exception is persisted with `status: failed` and its error message.

To compare measured main-suite test accuracy with the published reference table (including explicit missing rows), run:

```bash
python scripts/summarize_paper_results.py --source clip
python scripts/summarize_paper_results.py --source qwen --json
```
