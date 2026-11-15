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
│  ├─ archives/                # 下载的压缩包
│  └─ weights/                 # 共享模型权重或本地缓存
├─ outputs/                    # 运行时实验输出（Git 忽略）
│  └─ _preparation/            # 准备任务状态、命令和日志
├─ README.md                   # 快速开始和常用命令
├─ requirements.txt            # 唯一的项目依赖清单
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

- `annotation/`：VLM 标签导入、JSONL/TSV/旧 TXT 解析、身份对齐、缓存导出和 metadata 校验；`python -m src.utils.annotation` 是统一命令入口。
- `extract/`：CLIP ViT-L/14 特征提取器；每个数据集有专用入口，`extract_CLIP_imagefolder.py` 只用于 ImageFolder 协议。
- `labels/`：各数据集的类别表、真实标签和观察标签/划分生成器。
- `eval/`：Zero-shot、准确率、纠正比例和异常样本统计。

## 3. 配置和实验入口

`configs/paths.yaml` 只描述目录；`configs/datasets.yaml` 描述数据集 ID、标签脚本、特征提取器和划分差异；`configs/paper_reproduction.yaml` 描述可执行协议（模型、rho、lambda、P/K、epoch、seed 和方法）；`configs/paper_targets.yaml` 只保存论文参考百分比，绝不能当作实测结果写入 metrics。

按职责使用入口：

| 任务 | 入口 |
| --- | --- |
| 下载/解压、特征、标签、原型和筛选准备 | `scripts/prepare_materials.py`、`scripts/prepare_labels.py` |
| 检查准备产物和缓存指纹 | `scripts/check_preparation.py` |
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

上图中的 manifest 是身份对齐要求。部分旧 CLIP 缓存只保存 split indices 或路径，尚未统一导出独立 manifest 文件；导入 Qwen 标签时必须先从原提取顺序建立该清单，不能假设每个数据集现成具备 manifest。

缓存必须保留 `sample_id` 或等价 manifest 指纹、类别映射、split、特征来源和预处理信息。Qwen overlay 的来源保存于 `annotation_metadata[split]`，包含 `source_models` 和 `replaced_clip_fields=true`；主训练检查 train，Qwen 零样本检查 test，未覆盖的 CLIP split 不计入 Qwen 指标。旧 TXT 标签导入前必须显式确认原始 manifest 顺序。`outputs/` 中的命令、配置、日志、checkpoint 和 metrics 用于审计，但不进入 Git。

## 5. 参考资料归档

`reference_backup/` 只保存恢复证据，不作为现行代码入口：

- `paper/`：论文 PDF、抽取文本及其说明。
- `code_temp/`：旧训练、标签、特征和 shell 脚本。
- `对比方法/`：旧版 L2B/DLD/CSGN 脚本和方法论文。
- `Learning from filter-corrected lables/`：原始演示、表格、附录和重复的旧代码。
- `MANIFEST.csv`、`README.md`：来源清单和归档规则。

现行代码的协议解释见 `docs/paper_reproduction.md`；远程状态和实测结果见 `docs/remote_setup.md`、`docs/remote_clip_results_seed42.md`。

## 6. 版本控制边界和整理规则

应提交：`src/`、`scripts/`、`configs/`、`ablation/` 中的源码/配置、`tests/`、`docs/`、`README.md`、`requirements.txt` 和可追溯的 `reference_backup/` 资料。

不应提交：原始数据、压缩包、特征/标签/原型缓存、模型权重、checkpoint、完整实验输出、日志、Python 缓存、虚拟环境、SSH 凭据和临时上游副本。`.gitignore` 已覆盖这些路径；如果某个文件同时匹配“可提交”和“运行时产物”，以数据不可复现且体积大的运行时文件优先保持忽略，并在 `docs/` 记录获取方式。

提交前建议执行：

```bash
git status --short --ignored
git diff --check
python -m pytest -q
```

发现新的数据集划分、标签来源、预处理或远程环境差异时，应同时更新对应的 `configs/*.yaml` 和 `docs/` 证据文件，避免只改脚本而丢失协议。
