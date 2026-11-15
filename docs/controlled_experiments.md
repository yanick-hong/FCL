# 标签纠正与损失函数的受控实验协议

本文定义本次新增实验的输入、比较条件、统计方式和文件用途。它说明“如何执行和判断完成”，不记录远程任务已经运行到哪一步；实际进度、失败、缺失数据和实测数值由单独的运行台账记录。配置入口是 `configs/controlled_campaign.yaml`，已有论文恢复协议和已知差异分别见 `docs/paper_reproduction.md`、`docs/project_audit.md`。

## 本次五项要求

| 编号 | 实验与报告内容 | 固定条件 |
| --- | --- | --- |
| 1 | 生成 Qwen 原始标签，报告各 split 的标签准确率；对训练集进行原型筛选后，报告校正后标签准确率、人工预算和检错指标。 | 用户已确认使用 `Qwen/Qwen2.5-VL-7B-Instruct`；采用本次明确保存的附录 Prompt C；沿用已有数据划分。 |
| 2 | 以 CLIP 原始伪标签为起点，均匀随机选择全局约 30% 训练样本校正；使用选择种子 42、43、44 报告标签质量。另附与原型实际人数一致的随机校正，比较同预算下的效果。 | 全局固定人数为 `floor(0.3*N+0.5)`，不按类别分层；三份随机掩码分别保存。 |
| 3 | 在随机 30% 校正标签上训练 AUC-C，比较随机筛选与原型筛选对最终分类性能的影响。 | 核心训练固定采用 **选择种子 42** 的随机掩码，再改变训练种子 42、43、44。 |
| 4 | 在同一份原型校正标签和掩码上，将未校正部分的 AUC-C 替换为 CLIP 软标签交叉熵，形成损失函数对照。 | 两种目标共用样本、校正标签、初始化种子、采样流、优化器、训练轮数和模型选择规则。 |
| 5 | 核心条件各执行至少三次独立训练，报告均值、样本标准差和 95% Student-t 置信区间，并对可核验的配对目标报告差值。 | 三个训练种子为 42、43、44；固定掩码内的训练方差与改变掩码产生的方差分别报告。 |

六个主表数据集是 CIFAR-100、Caltech-101、Stanford Cars、EuroSAT、Food-101、Tiny-ImageNet-200；DTD 单独作为补充数据集。入口支持这七个数据集，但某个数据集尚未具备完整输入时，该行必须保留为缺失或失败，不能用其他数据集、模型或论文数字填充。

每个可运行数据集的核心训练矩阵如下，共 4 组 × 3 个训练种子。随机匹配原型预算的标签结果属于补充对照，不自动算入这四组核心训练。

| 标签来源 | 校正选择 | 目标 | 选择种子 | 训练种子 |
| --- | --- | --- | --- | --- |
| CLIP | 原型最远样本 | `fcl_auc` | 无随机选择种子 | 42、43、44 |
| CLIP | 全局随机精确人数 | `fcl_auc` | **固定 42** | 42、43、44 |
| CLIP | 与第一行完全相同的原型掩码 | `ce_soft` | 无随机选择种子 | 42、43、44 |
| Qwen2.5-VL | 原型最远样本 | `fcl_auc` | 无随机选择种子 | 42、43、44 |

## Qwen 模型、提示词与视觉设置

用户已选择 Qwen2.5-VL-7B-Instruct。本次不把论文中的“Qwen-VL-7B”简称解释成一个已确定的历史权重版本；恢复资料存在模型命名、代码版本和实验版本不一致。备份材料是恢复证据，本次用户确认与保存的可执行配置决定当前实验设置。

| 项目 | 本次显式设置 | 记录方式 |
| --- | --- | --- |
| 模型与后端 | `Qwen/Qwen2.5-VL-7B-Instruct`、`qwen2.5-vl` | 配置、模型环境、实际解析的 revision 或本地权重文件 SHA |
| 提示词 | `run_qwen_campaign.py` 中依据附录 C 转写的 `appendix_prompt()`，逐数据集插入有序类别表 | 每个数据集保存 `prompt.txt` 和 `prompt_sha256` |
| 输出 | 仅一个零起始类别编号；严格解析 | 保存每张图的原始输出，非法输出写错误文件并中止，不猜测或回退到 CLIP |
| 生成 | `do_sample=false`、`max_new_tokens=16`、标注种子 42 | JSONL sidecar 和运行配置 |
| 图像预算 | `min_pixels=max_pixels=200704=448*448` | 处理器参数与模型环境 |
| 推理数值与批量 | `bfloat16`、`sdpa`、`batch_size=8` | 模型环境、配置及依赖版本 |
| 标注范围 | train、val、test | 每个 split 独立 manifest、进度、准确率与 overlay 来源 |

448×448 表示本次选定的**等面积像素预算**；Qwen 处理器仍按自身规则处理宽高比和网格，不代表把所有图片强制拉伸成正方形。该预算、批大小等是本次显式设置，不能声称已经从旧论文中唯一恢复。

本次附录 C 转写模板为：

```text
You are a classifier for the <dataset name>. Given the image, choose exactly ONE class from the list below.
0: <class name>
1: <class name>
...
Reply with ONLY the integer id.
```

该模板与通用 `python -m src.utils.annotation qwen --prompt-variant C` 的内置短模板不同，不能只因为都名为 C 就混用。应比较实际 `prompt.txt`、类别顺序和 SHA。代码会保存模型 revision；配置中请求 `main` 不等于一个永久不变的权重版本，恢复运行必须核对首次加载的实际 revision 或本地文件指纹。

Qwen 当前输出离散类别，导出时对应概率行是 one-hot，并标注 `one_hot_from_label`。它表示硬标签，不能被解释为模型校准后的置信分布。Qwen overlay 保留原 CLIP 概率为 `original_clip_probs_*`，供需要 CLIP 软标签的对照读取。

## 固定划分与样本身份

CLIP 特征、真实标签、类别 ID、原始样本顺序和 train/val/test 划分以同一份已验证缓存为准。更换训练种子时不重新划分数据，不重抽校正掩码，也不重新标注 Qwen。

`prepare_annotation_manifests.py` 从缓存中的稳定 sample ID、路径和原始 split 索引建立 manifest；清单只含 `sample_id/path/split`，不把真实类别传给 VLM。CIFAR-100 图片导出会核对官方索引与缓存标签，其他数据集核对原始路径、类别表和 split 不重叠。需要导出图片时，导出图片保持该身份映射。

已知专用划分包括 CIFAR-100 的 45k/5k/10k、Food-101 的 65650/10100/25250、Tiny-ImageNet 的 90k/10k/10k，以及 DTD 官方 partition=1。Caltech 的每类划分、小类别回退和背景类差异、EuroSAT 的保存索引、Cars 的官方带标签测试输入仍应核对各自缓存；不因名称相同就视为划分相同。论文训练人数与当前缓存不符时必须记录差异。

Qwen 标签按 sample ID 对齐到同一 CLIP 特征缓存，按 split 累积导出 overlay。`annotation_metadata[split]` 必须证明该 split 的 Qwen 来源及 `replaced_clip_fields=true`；未覆盖的 CLIP 预测不报告成 Qwen 准确率。尤其 Qwen zero-shot 测试必须有真实的 Qwen test 标签，不能因 train 已替换就把原 CLIP test 结果改名。

## 校正预算和监督边界

原型筛选先将训练特征逐行 L2 归一化，按**预测类别**分组，计算每组均值并再次 L2 归一化。按 `1-cos(feature, prototype)` 从大到小取每组 `ceil(rho*n_group)` 个样本。距离相同按原缓存行顺序决胜，保证确定性。选择只依赖特征和原伪标签，不读取真实类别。

原型预算为 `sum_c ceil(0.3*n_c)`，通常略大于 30%。全局随机预算则固定为 `floor(0.3*N+0.5)`：均匀、无放回、不按预测类别分层。当 `0.3*N` 不是整数时，数学上无法恰好选择 30%，因此取最近的整数人数，并在报告中同时给出请求比例、实际人数和实际比例。第三种 `random_matched_prototype` 只为匹配原型的实际人数，不能与“随机 30%”混为同一条件。

选中掩码记作 `selected_mask`，观察标签遵循：

```text
selected_mask=True : s=0，y_obs=选中位置的数据集真实标签
selected_mask=False: s=1，y_obs=原始伪标签 y_pseudo
```

这是一项 oracle 人工校正模拟，不表示已经完成真人标注，也不衡量真人误标率。完整真实训练标签可用于输入一致性校验和实验后评分；训练批次仅包含特征、`y_obs`、`s` 和来源概率。未校正位置的真实标签不进入训练损失、原型构造、采样分组或后验更新。

校正报告同时保存源标签准确率、校正后准确率、人工人数和比例、错误检出 precision/recall。precision 是“选中样本中原伪标签错误的比例”，recall 是“全部原伪标签错误中被选中的比例”；不是训练后分类器的 precision/recall。分母为零时写 `null`，不伪造一个零或满分结果。

每份 observed-label 文件保存 `y_obs/s/y_pseudo/selected_mask`、原缓存行号和 sample ID，metadata 保存来源、选择方法、选择种子、缓存 SHA、样本顺序 SHA、mask SHA 和 `y_obs` SHA。训练器拒绝不匹配的缓存、顺序、掩码或标签，不在训练时重新选择样本。

## 训练目标和可比性

所有核心组训练冻结 CLIP ViT-L/14 特征上的线性分类头。固定 100 epochs、`patience=0`，按最高验证准确率保存最佳 checkpoint，完成训练后才加载该 checkpoint 评估测试集。测试准确率不参与超参数或 checkpoint 选择。

共同参数为 AdamW，`lr=1e-3`、OneCycleLR `lr_max=5e-3`、`weight_decay=1e-4`、`lambda=1`、`mu=0.5`、pairwise `tau=0.5`、`num_workers=0`。PK 采样按观察标签 `y_obs` 分组。P/K 是恢复协议中的数据集设置：CIFAR-100/Food-101 为 32/32，Caltech-101/EuroSAT 为 8/8，Stanford Cars/Tiny-ImageNet 为 16/16。DTD 本次显式选择 **P=16、K=16**，由 suite 传入；它是本次补充实验设置，不是已恢复的论文常数。

- `fcl_auc`：已校正部分对 `y_obs` 做 CE，未校正部分做后验加权的 soft pairwise AUC-C。后验为 `mu*softmax(model).detach()+(1-mu)*p_source`，来源按 CLIP/Qwen 条件选择。
- `ce_soft`：已校正部分保持同一 CE；未校正部分改成对**原始 CLIP 概率**的 soft CE。Qwen overlay 中使用 `original_clip_probs_train`，不能把 Qwen one-hot 冒充 CLIP 软概率。

两个子集各自取均值后，按 `CE_corrected + lambda*loss_uncorrected` 合并；不是再按两个子集人数重新加权。对照条件保持该归约方式不变。

同一训练种子的两种目标必须具有相同输入/掩码/观察标签 SHA、初始化 SHA、采样流 SHA 和共同协议 SHA。随机 30% 的核心训练始终使用 `random_global_exact_seed-42.pt`；训练种子 43 不意味着改用 `random_global_exact_seed-43.pt`。选择种子 43、44 用于标签纠正质量的变化分析，不能把三种掩码与三次训练混成同一个固定掩码下的九次独立训练。

## 统计报告

先按 dataset、label_source、selection、objective、selection_seed 分组，在每组内要求训练种子 42、43、44 齐全且不重复，并核对输入、源码和共同协议指纹一致。中途失败、缺少测试指标或只完成一两个种子的组不生成完整三种子结论。

令三次测试准确率为 `a_i`，报告均值 `mean(a)`、样本标准差 `s=sqrt(sum((a_i-mean(a))^2)/(n-1))`（`ddof=1`），以及 `mean(a) ± t(0.975,n-1)*s/sqrt(n)`。三次训练时自由度为 2，t 临界值约 4.303；不能用三次结果直接套用 1.96。汇总输出把准确率转为百分数，标准差和差值的单位为百分点。

95% t 区间是均值的不确定性估计，样本只有三个种子时可能很宽；它不覆盖重新划分数据、重新生成 VLM 标签、改变 prompt 或校正掩码造成的全部不确定性。区间按公式原样保存，不截断到 0–100 后伪装成标准 t 区间。

现有 `summarize_controlled_results.py` 自动汇总固定掩码组，并对经过初始化/采样流等指纹核验的 `fcl_auc - ce_soft` 逐训练种子差值计算配对 t 区间。不同选择方法或不同来源的组可并列比较；当前自动配对输出的范围以脚本实际支持为准，不能把两个组各自的置信区间当成差值区间。标签纠正报告中的随机种子均值/样本标准差只描述掩码选择变化，与训练三种子统计分别列出。

## 命令和产物路径

以下命令在项目根目录执行，本地根目录为 `F:\CodeProjects\FCL2`，AutoDL 根目录为 `/root/autodl-tmp`。数据和输出根目录还可通过 `DATA_ROOT/OUTPUT_ROOT/WEIGHTS_ROOT` 指定。示例目标目录必须是新目录；已有输入或固定掩码不会被覆盖。

先建立七个数据集的有序、无标签 manifest；只有对应缓存和原图完整的数据集才能成功：

```bash
python scripts/prepare_annotation_manifests.py --datasets cifar100 caltech101 stanfordcars eurosat food101 tinyimagenet200 dtd
```

Qwen 标注由统一配置驱动。首次执行不加 `--resume`，后续在同一模型/源码/配置下恢复时加上它。`--limit` 只用于小规模验证，不会把未完成标注导出为完整结果。

```bash
python scripts/run_qwen_campaign.py --config configs/controlled_campaign.yaml
python scripts/run_qwen_campaign.py --config configs/controlled_campaign.yaml --resume
```

Qwen 前置标注完整后，可使用受控套件统一规划和分阶段执行。默认 `dry-run` 仅打印计划；执行时会核验三个 split 的标注、模型/提示词/overlay 来源链，以及固定掩码和训练产物的指纹。已完成工作只有在输入、源码、协议和输出清单均相符时才复用；不完整目录保留并报错，不被覆盖。

```bash
python scripts/run_controlled_suite.py --config configs/controlled_campaign.yaml --dry-run
python scripts/run_controlled_suite.py --config configs/controlled_campaign.yaml --phases corrections --execute
python scripts/run_controlled_suite.py --config configs/controlled_campaign.yaml --phases train_random_auc train_soft_ce train_core --execute
python scripts/run_controlled_suite.py --config configs/controlled_campaign.yaml --phases summary --execute
```

套件默认根目录为 `outputs/controlled_correction_20261115/`，其下 `qwen/` 保存标注，`corrections/` 保存固定校正输入，`runs/` 保存训练，`summaries/` 保存汇总，`orchestration/<timestamp>/` 保存本次计划和阶段状态。下面单项入口使用独立的 `outputs/controlled_manual_example/`，便于检查某个条件；执行完整套件时无需再执行这些命令。手工产物没有 suite receipt，不能直接放入套件目录并要求它自动复用。

以 CIFAR-100 为例，分别产生 CLIP 和完成三 split overlay 后的 Qwen 校正对比；路径中的输出目录必须尚不存在：

```bash
python scripts/compare_label_corrections.py --datasets cifar100 --source clip --cache dataset/processed/cifar100_clip_vit-l-14_embeddings.pt --output-dir outputs/controlled_manual_example/corrections/cifar100/clip

python scripts/compare_label_corrections.py --datasets cifar100 --source qwen --prompt C --cache outputs/controlled_correction_20261115/qwen/cifar100/cifar100_qwen_through-test.pt --output-dir outputs/controlled_manual_example/corrections/cifar100/qwen
```

未显式指定 `--cache` 时，CLIP 默认读取 `dataset/processed/<dataset>_clip_vit-l-14_embeddings.pt`，Qwen 默认模板为 `<dataset>_qwen_prompt-C_embeddings.pt`。Qwen campaign 的累积缓存实际保存在它自己的输出目录，使用时应传入 `--cache` 或 `--cache-template`；不能假设两种默认命名已自动互相复制。

下面展示 CLIP 原型固定掩码上训练种子 42 的两个配对目标，以及随机固定掩码上的 AUC-C；同条件再将 `--seed` 改为 43、44，并使用各自新输出目录。其他超参数沿用上述协议，P/K 按数据集显式调整。

```bash
python scripts/run_controlled_training.py --cache dataset/processed/cifar100_clip_vit-l-14_embeddings.pt --obs-labels outputs/controlled_manual_example/corrections/cifar100/clip/prototype.pt --objective fcl_auc --seed 42 --epochs 100 --P 32 --K 32 --output-dir outputs/controlled_manual_example/runs/cifar100/clip/prototype/fcl_auc/seed-42

python scripts/run_controlled_training.py --cache dataset/processed/cifar100_clip_vit-l-14_embeddings.pt --obs-labels outputs/controlled_manual_example/corrections/cifar100/clip/prototype.pt --objective ce_soft --seed 42 --epochs 100 --P 32 --K 32 --output-dir outputs/controlled_manual_example/runs/cifar100/clip/prototype/ce_soft/seed-42

python scripts/run_controlled_training.py --cache dataset/processed/cifar100_clip_vit-l-14_embeddings.pt --obs-labels outputs/controlled_manual_example/corrections/cifar100/clip/random_global_exact_seed-42.pt --objective fcl_auc --seed 43 --epochs 100 --P 32 --K 32 --output-dir outputs/controlled_manual_example/runs/cifar100/clip/random_global_exact/fcl_auc/seed-43
```

Qwen 原型训练把 `--cache` 换成产生其 observed-label 文件的同一份 Qwen overlay，`--obs-labels` 换成 Qwen 的 `prototype.pt`，目标仍为 `fcl_auc`，训练种子同为 42、43、44。文件 SHA 必须匹配；不能在生成掩码后更换 overlay 文件再沿用旧掩码。

仅在所选训练根目录中全部目标组的三种子结果完整时汇总。输出目录也须为空或尚不存在；若某组缺失，先保留为待完成状态，不删掉缺失组后声称整个矩阵完成。

```bash
python scripts/summarize_controlled_results.py --runs-root outputs/controlled_manual_example/runs --output-dir outputs/controlled_manual_example/summary --required-seeds 42 43 44
```

## 目录职责与本次新增文件

| 现有目录 | 用途 |
| --- | --- |
| `src/fcl/` | FCL 主训练器、线性分类头、PK 采样器与 AUC 损失实现。 |
| `src/contrast/` | 旧基线、冻结特征适配器与公共评估/缓存工具。 |
| `src/utils/extract/`、`src/utils/labels/` | 专用特征提取、划分和观察标签准备。 |
| `src/utils/annotation/` | VLM 标签 schema、进度存储、身份对齐、模型推理和 overlay 导出。 |
| `scripts/`、`configs/` | 可执行批量入口和声明式协议配置。 |
| `ablation/` | 既有人工纠正噪声及参数敏感性实验。 |
| `tests/`、`docs/` | 协议/正确性回归测试，以及恢复证据、目录说明和实验协议。 |
| `reference_backup/` | 原论文、旧代码、附录、表格和演示资料；作为证据保留。 |
| `dataset/raw/`、`dataset/processed/`、`dataset/weights/` | 原始数据、可复用缓存/manifest、模型权重；由 Git 忽略。 |
| `outputs/` | 标注进度、固定掩码、训练日志/checkpoint/metrics 和汇总；由 Git 忽略。 |

本次新增入口及职责：

| 文件 | 用途与主要产物 |
| --- | --- |
| `configs/controlled_campaign.yaml` | 本次七数据集范围、Qwen 显式设置、校正比例、训练种子、共同训练协议和统计规则。 |
| `scripts/prepare_annotation_manifests.py` | 保持原缓存顺序导出 `manifest.jsonl/classes.json/manifest_metadata.json`；CIFAR 同时导出可验证图片。 |
| `src/utils/annotation/qwen_batch.py` | Qwen2.5-VL 批量贪心推理，保存视觉/权重/依赖设置。 |
| `scripts/run_qwen_campaign.py` | 可恢复的三 split 标注；保存 `prompt.txt`、JSONL、sidecar、模型环境、错误信息和累积 Qwen overlay。 |
| `scripts/compare_label_corrections.py` | 生成一份 `prototype.pt`、三份 `random_global_exact_seed-*.pt`、三份 `random_matched_prototype_seed-*.pt`，以及 `sample_order.json/report.json/report.csv`。 |
| `scripts/run_controlled_training.py` | 每次执行一个训练种子和一个目标，严格复用固定 observed-label；保存 `config.json/history.csv/status.json/best.ckpt/last.ckpt/metrics.json` 及指纹。 |
| `scripts/run_controlled_suite.py` | 按纠正、随机 AUC、soft-CE、原型主组、汇总阶段统一规划和执行；保存 orchestration 状态及可复用产物的 receipt。 |
| `scripts/summarize_controlled_results.py` | 校验完整三种子及配对指纹，生成 `summary.json/summary.csv/summary.md`。 |
| `tests/test_label_correction_comparison.py` | 检查预算、oracle 边界、来源 split、稳定顺序与掩码可重放。 |
| `tests/test_qwen_campaign.py` | 检查 campaign 提示词、恢复及标签导出衔接。 |
| `tests/test_controlled_training.py`、`tests/test_controlled_statistics.py` | 检查固定掩码输入、两种目标及三种子统计/配对规则。 |
| `tests/test_controlled_suite.py` | 检查阶段顺序、核心矩阵、DTD 的显式 P/K 以及套件指纹/复用边界。 |
| `docs/controlled_experiments.md` | 本文：本次协议和使用方法；不充当远程完成证明。 |

源码、配置和说明提交到 Git；图片、模型、overlay、掩码、完整运行输出保存在受 Git 忽略的运行时目录。实验完成的证据应能回溯到配置、源码、缓存、标签、掩码和 checkpoint 的指纹，而不仅是一张汇总表。
