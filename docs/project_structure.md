# FCL 项目目录与文件职责

本文档定义当前恢复分支的**规范目录**。项目根目录是唯一代码入口；`dataset/`、`outputs/` 和模型权重只作为本地或 AutoDL 运行时目录，不提交到 Git。

## 1. 总体布局

```text
FCL2/
├─ src/                         # 可复用的算法、数据准备和评估代码
│  ├─ fcl/                      # FCL + AUC-C 主方法
│  ├─ contrast/                 # 对比方法和兼容训练器
│  └─ utils/                    # 标注、特征抽取、数据集标签、评估工具
├─ scripts/                    # 面向用户的批处理/实验入口（调用 src）
├─ configs/                    # 路径、数据集协议、论文实验参数和参考结果
├─ ablation/                   # 独立消融实验（代码和配置受 Git 管理）
├─ tests/                      # 单元、协议和安全回归测试
├─ docs/                       # 复现协议、远程环境、实测结果和本目录说明
├─ reference_backup/           # 遗失项目的原始代码、论文、表格和演示资料
├─ .experiment_archive/        # 本地临时上游副本归档（Git 忽略）
├─ dataset/                    # 运行时数据（Git 忽略，见下文）
│  ├─ raw/                     # 原始图片/压缩包，只读
│  ├─ processed/               # 特征、划分、伪标签、原型和审计 JSON
│  │  └─ annotation_manifests/ # 七数据集恢复并核验的原缓存顺序清单/类别表
│  ├─ archives/                # 下载的压缩包
│  └─ weights/                 # 共享模型权重或本地缓存
├─ outputs/                    # 运行时实验输出（Git 忽略）
│  ├─ _preparation/            # 准备任务状态、命令和日志
│  ├─ paper/                  # 论文套件的历史/新增运行
│  ├─ paper_baseline_recheck/  # 核心完成后复跑主表其余方法；含计划、状态和产物 receipt
│  └─ controlled_correction_20261115/
│     ├─ qwen/                # 三 split 标注、提示词、模型环境、累计 overlay
│     ├─ corrections/         # 固定掩码、观察标签、预算和标签质量报告
│     ├─ runs/                # 四个核心条件 × 三训练种子的运行产物
│     ├─ summaries/           # 完整三种子统计与目标配对差值
│     └─ orchestration/       # 分阶段计划和状态；receipt 随各任务产物保存
├─ README.md                   # 快速开始和常用命令
├─ requirements.txt            # 特征准备和训练的基础依赖
├─ requirements-qwen.txt       # 独立 VLM 环境的固定版本推理依赖
├─ pytest.ini                  # 测试发现和 Python 路径配置
└─ .gitignore                  # 数据、权重、缓存和运行产物的边界
```

根目录原有的 `upstream_extract/` 和 `upstream.zip` 已移至 `.experiment_archive/upstream_reference/`，保留用于核对上游差异。`.pytest_cache/`、各目录下的 `__pycache__/`、`.autodl/`、`.venv/` 等也只属于本地状态。

## 2. 源码职责

### `src/fcl/`

- `auc_loss.py`：稳定的 pairwise logistic/AUC 损失。
- `train_auc_ce.py`：主 FCL 训练器，使用已校正样本的 CE 和未校正样本的后验加权 AUC；包含 checkpoint、验证集选择和指标记录。

### `src/contrast/`

- `train_l2b.py`、`train_dld.py`、`train_csgn.py`：冻结 CLIP 特征上的 L2B、DLD、CSGN 适配实现。
- `train_fullsup.py`、`train_trusted.py`、`train_noise.py`、`train_nlprompt.py`、`train_revision_baselines.py`：全监督、可信样本、噪声标签、NLPrompt 及复核后的基线实现。
- `train_dcd.py`、`train_ido_cache.py`、`train_dld_cache.py`：旧实验兼容入口或缓存适配器。
- `cache_experiment.py`：统一缓存校验、结果目录、评估指标和运行元数据工具。

`src/paths.py` 负责读取 `configs/paths.yaml`，允许 `DATA_ROOT`、`OUTPUT_ROOT`、`WEIGHTS_ROOT` 环境变量覆盖默认目录。

对比方法必须使用与 FCL 相同的特征缓存、样本顺序和划分；不能直接把旧脚本里的随机重划分带入主表。

### `src/utils/`

- `annotation/`：VLM 标签导入、JSONL/TSV/旧 TXT 解析、身份对齐、缓存导出和 metadata 校验；`python -m src.utils.annotation` 是通用命令入口。`qwen_batch.py` 提供本次 Qwen2.5-VL 批量贪心推理，并记录权重、视觉处理和依赖环境。
- `extract/`：CLIP ViT-L/14 特征提取器；每个数据集有专用入口，`extract_CLIP_imagefolder.py` 只用于 ImageFolder 协议。
- `labels/`：各数据集的类别表、真实标签和观察标签/划分生成器。
- `eval/`：Zero-shot、准确率、纠正比例和异常样本统计。

## 3. 配置和实验入口

`configs/paths.yaml` 只描述目录；`configs/datasets.yaml` 描述数据集 ID、标签脚本、特征提取器和划分差异；`configs/paper_reproduction.yaml` 描述可执行论文恢复协议（模型、rho、lambda、P/K、epoch、seed 和方法）；`configs/paper_targets.yaml` 只保存论文参考百分比，绝不能当作实测结果写入 metrics。`configs/controlled_campaign.yaml` 单独声明本次七数据集的 Qwen 推理、校正预算、固定掩码、三训练种子与统计规则。

`requirements.txt` 是基础特征/训练依赖，`requirements-qwen.txt` 是独立 VLM 环境的固定版本推理依赖。两者按用途使用；Qwen 环境复用 AutoDL CUDA PyTorch，不通过安装推理依赖改写基础训练环境。

按职责使用入口：

| 任务 | 入口 |
| --- | --- |
| 下载/解压、特征、标签、原型和筛选准备 | `scripts/prepare_materials.py`、`scripts/prepare_labels.py` |
| 检查准备产物和缓存指纹 | `scripts/check_preparation.py` |
| 从七数据集原缓存顺序恢复并核验无标签 manifest | `scripts/prepare_annotation_manifests.py` |
| 本次 Qwen 三 split 标注及累计 overlay | `scripts/run_qwen_campaign.py` |
| 原型、随机精确 30% 与匹配预算校正标签对比 | `scripts/compare_label_corrections.py` |
| 固定掩码的单个训练目标/种子 | `scripts/run_controlled_training.py` |
| 四个核心条件 × 三训练种子的阶段计划与执行 | `scripts/run_controlled_suite.py` |
| 固定掩码三种子统计与同掩码目标配对差值 | `scripts/summarize_controlled_results.py` |
| 论文主实验及 lambda/rho/prompt/epoch 套件 | `scripts/run_paper_experiments.py` |
| 旧缓存批量实验 | `scripts/run_cache_experiments.py` |
| 旧基线和补充对比 | `scripts/run_legacy_baselines.py`、`scripts/run_new_comparisons.py`、`scripts/run_revision_experiments.py` |
| 断点复评、指标汇总和绘图 | `scripts/evaluate_fcl_checkpoints.py`、`scripts/summarize_latest_metrics.py`、`scripts/summarize_paper_results.py`、`scripts/plot_*.py` |
| 远程环境只读诊断 | `scripts/remote_diagnose.py` |
| AutoDL 环境激活 | `scripts/activate_autodl.sh` |

`ablation/human_correction_noise/` 和 `ablation/parameter_sensitivity/` 是独立实验包：`config.yaml` 固定参数，`run.py` 执行，`summarize.py` 汇总，`audit.py`/`test_ablation.py` 做审计和回归。它们的 `results/`、`smoke/` 和日志被忽略。

## 4. 数据和输出约定

每个数据集的处理链为：

```text
raw/archive → processed/*_manifest + *_classes
            → *_clip_vit-l-14_embeddings.pt
            → *_qwen_prompt-{A|B|C}_embeddings.pt（可选）
            → paper_labels/*_source_rho-{rho}.pt
            → prototypes / filtered labels / preparation.json
            → outputs/paper/<suite>/<source>/<dataset>/<method>/...
```

上图中的 manifest 是身份对齐要求。本次已对主六数据集和 DTD 的 `annotation_manifests` 按原缓存顺序恢复并核验；旧缓存若只保留 split indices 或路径，必须经过身份恢复和原图/索引验证，不能根据新枚举的文件顺序猜测。校验记录见 [本轮 manifest 资料](results/controlled_20261115/manifests/)。

Stanford Cars 的官方带标签测试文件和 CLIP 缓存现已齐全，本次 train/val/test 为 **7356/788/8041**。准备命令、官方测试标签路径、划分和缓存 SHA 见 [Cars 准备证据](results/controlled_20261115/preparation/stanfordcars_preparation.json)；与论文记录人数的差异仍须保留。

缓存必须保留 `sample_id` 或等价 manifest 指纹、类别映射、split、特征来源和预处理信息。Qwen overlay 的来源保存于 `annotation_metadata[split]`，包含 `source_models` 和 `replaced_clip_fields=true`；主训练检查 train，Qwen 零样本检查 test，未覆盖的 CLIP split 不计入 Qwen 指标。旧 TXT 标签导入前必须显式确认原始 manifest 顺序。`outputs/` 中的命令、配置、日志、checkpoint 和 metrics 用于审计，但不进入 Git。

本次受控流程使用 `outputs/controlled_correction_20261115/`：先在 `qwen/` 保存实际标注及来源链，再在 `corrections/` 固定校正输入，在 `runs/` 训练，最后在 `summaries/` 汇总。`orchestration/` 保存计划和阶段状态，各任务产物目录的 `.suite_receipt.json` 保存完成时的输入/输出指纹，供严格复用。四个核心条件是 CLIP 原型 AUC-C、CLIP 随机 30% AUC-C、同一 CLIP 原型掩码 soft-CE、Qwen 原型 AUC-C，各自训练种子 42、43、44；随机核心训练固定选择种子 42，其他选择种子只描述校正质量变化。规则与 CLI 见 [受控实验协议](controlled_experiments.md)。

`outputs/paper_baseline_recheck/` 用于主表其余方法的独立复跑队列。它等待核心队列全部成功后，沿用相同缓存和原型掩码，运行六个主数据集 × 两个标签来源 × 七方法（84 项，其中 12 项 zero-shot 评估、72 项训练）。该目录的计划、状态和 receipt 与核心目录分别保存；FCL 的三种子结果继续以核心目录为准。

Qwen 标注和受控训练目前仍在执行，整套实验尚未全部完成。源码入口或输入齐全不代表结果齐全；最新进度和结果证据以 [本轮运行台账](results/controlled_20261115/README.md) 为准。

## 5. 参考资料归档

`reference_backup/` 只保存恢复证据，不作为现行代码入口：

- `paper/`：论文 PDF、抽取文本及其说明。
- `code_temp/`：旧训练、标签、特征和 shell 脚本。
- `对比方法/`：旧版 L2B/DLD/CSGN 脚本和方法论文。
- `Learning from filter-corrected lables/`：原始演示、表格、附录和重复的旧代码。
- `MANIFEST.csv`、`README.md`：来源清单和归档规则。

现行代码的论文恢复协议见 `docs/paper_reproduction.md`，本次受控协议见 `docs/controlled_experiments.md`。`docs/remote_clip_results_seed42.md` 保留历史实测快照；本轮进度与证据见 [运行台账](results/controlled_20261115/README.md)。

## 6. 版本控制边界和整理规则

应提交：`src/`、`scripts/`、`configs/`、`ablation/` 中的源码/配置、`tests/`、`docs/`、`README.md`、基础与 Qwen 两份依赖清单，以及可追溯的 `reference_backup/` 资料。`docs/results/` 仅保存便于审阅的台账、校验与结果证据；完整模型和运行产物仍放在被忽略的运行时目录。

`.gitattributes` 禁用 `docs/results/` 的自动换行转换，保留取证文件的原始字节；否则 Windows 的 `core.autocrlf` 可能使检出后的脚本与已记录 SHA256 不一致。

不应提交：原始数据、压缩包、特征/标签/原型缓存、模型权重、checkpoint、完整实验输出、日志、Python 缓存、虚拟环境、SSH 凭据和临时上游副本。`.gitignore` 已覆盖这些路径；如果某个文件同时匹配“可提交”和“运行时产物”，以数据不可复现且体积大的运行时文件优先保持忽略，并在 `docs/` 记录获取方式。

提交前建议执行：

```bash
git status --short --ignored
git diff --check
python -m pytest -q
```

发现新的数据集划分、标签来源、预处理或远程环境差异时，应同时更新对应的 `configs/*.yaml` 和 `docs/` 证据文件，避免只改脚本而丢失协议。
