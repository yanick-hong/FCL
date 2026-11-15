# FCL experiments

实验工程：冻结 CLIP 特征 → 过滤并模拟校正标签 → FCL+AUC-C / 对比方法 → 独立评估 → 汇总。

GitHub 已核对版本：`fb65f126274581af6f931c4de1eca0ff8e364c6e`；保留本地后续修复和未提交修改。
本地详细说明位于 docs/；AutoDL 只同步实验所需代码与本说明，不上传论文材料。

## 当前结构

```text
src/
  fcl/train_auc_ce.py          主方法（原入口保留）
  contrast/                   对比方法、全监督参照、缓存公共工具
  utils/extract/              特征提取（不同划分/骨干版本不能混用）
  utils/labels/               各数据集过滤与标签生成
  utils/eval/                 零样本、标签质量、分类评估
  paths.py                    环境变量优先于 YAML 的路径解析
configs/
  paths.yaml                  数据/输出/权重路径
  datasets.yaml               数据集 ID、入口与划分差异说明
scripts/                      批量运行、评估、绘图、汇总
dataset/                      不提交：所有实验准备输入
  archives/                   上传的 ZIP 压缩包
  raw/                        解压后原始数据，只读
  processed/                  特征、原型、过滤校正标签及审计报告
  weights/                    预训练骨干权重（可链接系统已有缓存）
outputs/                      不提交：<dataset>/<method>/<timestamp>/
  _preparation/               准备任务状态、命令与日志
```

不再并列使用 data/dataset 或 output/outputs；统一 dataset 与 outputs。
src 是算法实现，scripts 是批量入口，configs 是声明配置，三者职责不同。
本地 docs/、tests/ 和 .experiment_archive/ 分别用于说明、回归检查和可恢复归档，
不上传 AutoDL。远程保留 .git 版本信息与 .autodl 平台状态。

## AutoDL

项目根目录是 `/root/autodl-tmp`，无需 `FCL/` 外层目录。

```bash
source /root/autodl-tmp/scripts/activate_autodl.sh
python -m pip check
```

路径默认在项目内，也支持 DATA_ROOT、OUTPUT_ROOT、WEIGHTS_ROOT；数据和输出均被 Git 忽略。
requirements.txt 是唯一依赖清单；OpenAI CLIP 在当前实例已安装。
新环境需从官方项目安装：`python -m pip install git+https://github.com/openai/CLIP.git`。

## 最短实验流程（缓存恢复并校验后）

当前准备范围：DTD、Caltech-101、EuroSAT、CIFAR-100、Tiny-ImageNet。
Food-101 与 Stanford Cars 暂缓；保留入口，避免后续恢复时重新找代码。

```bash
# 自动解压、特征提取、原型导出、过滤校正、字段/标签/划分校验；可复用已完成缓存。
python scripts/prepare_materials.py --run-dir "$OUTPUT_ROOT/_preparation/rebuild_20261115"
```

使用各数据集专用脚本：Caltech-101 30/10/剩余划分，DTD 官方 partition=1，
CIFAR-100 45k/5k/10k，EuroSAT 80/10/10，Tiny-ImageNet 90k/10k/10k。
Caltech 样本不足 41 张的类别按原脚本自适应划分，因此最终数量以审计报告为准。
Tiny-ImageNet 的测试来源是官方有标签 val，不使用无标签 test。
本次重建不能等同于已丢失的旧实验；所有结果需与本次缓存指纹绑定。

查看进度：`python scripts/check_preparation.py`。
每个数据集生成四份材料：`*_clip_vit-l-14_embeddings.pt`、`*_observed_labels.pt`、
`*_prototypes.pt`、`*_preparation.json`，均在 dataset/processed/。

```bash
# 先预览；加 --execute 才生成标签，不覆盖已有标签。
python scripts/prepare_labels.py --datasets cifar100 eurosat tinyimagenet200
python scripts/prepare_labels.py --datasets cifar100 eurosat tinyimagenet200 --execute

# 历史主方法 + NLPrompt + DCD 批量入口，包括已完成结果的跳过逻辑。
python scripts/run_cache_experiments.py --datasets cifar100 eurosat tinyimagenet200

# 主训练脚本的历史测试指标可能对应末轮，统一用 best.ckpt 补评估。
python scripts/evaluate_fcl_checkpoints.py --datasets cifar100 eurosat tinyimagenet200

# 显式指定 test；默认 validation 保留旧脚本兼容性。
python scripts/summarize_latest_metrics.py --metric test --format csv --output "$OUTPUT_ROOT/latest_test_comparison.csv"
```

## 人工纠正噪声消融

独立工程位于 `ablation/human_correction_noise/`，方案见该目录 README.md，参数见 config.yaml。
`run.py` 执行五档人工正确率、三个种子的 FCL+AUC-C；`summarize.py` 汇总指定批次。
结果位于 `ablation/human_correction_noise/results/<北京时间时间戳>/`，不混入普通 outputs。
原始缓存只读；扰动标签位于该批次 inputs/，每次训练保存 history.csv、config.json、
run_metadata.json、metrics.json、best.ckpt 和 last.ckpt。

主方法 corrected CE 已改为读取实际 y_obs，训练结束恢复最佳验证权重。
高 VLM 噪声边界实验尚未实现。
