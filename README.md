# FCL

代码、配置和运行脚本进入 Git；数据集、CLIP 特征缓存、模型权重和实验输出留在项目外部。

## 路径配置

默认值位于 [configs/paths.yaml](configs/paths.yaml)。环境变量优先级更高：

```bash
export DATA_ROOT=/mnt/fcl-data
export OUTPUT_ROOT=/mnt/fcl-experiments
export WEIGHTS_ROOT=/mnt/fcl-weights
```

目录约定：

```text
DATA_ROOT/raw/                  # 原始数据，只读
DATA_ROOT/processed/            # 可复用的 CLIP 特征和划分
OUTPUT_ROOT/cache/              # 跨实验共享的 CLIP 权重缓存
OUTPUT_ROOT/<experiment>/       # checkpoint、config、metrics、logs
```

## 推荐流程

```text
src/utils/extract/extract_CLIP_*.py
    -> DATA_ROOT/processed/*_clip_*_embeddings.pt
src/utils/labels/make_obs_labels_*.py
    -> DATA_ROOT/processed/*_observed_labels.pt
src/fcl/train_auc_ce.py / src/contrast/train_*.py
    -> OUTPUT_ROOT/<experiment>/
src/utils/eval/eval_*.py
```

CLIP 特征提取脚本默认会复用已经存在的 `.pt` 缓存；需要重算时显式传入 `--force_rebuild`。

## 运行

从项目根目录执行：

```bash
python src/utils/extract/extract_CLIP_CIFAR100.py
python src/utils/labels/make_obs_labels_caltech101.py
python src/fcl/train_auc_ce.py
```

旧实验入口位于 [scripts/](scripts/)，可通过 `CACHE`、`OBS`、`CKPT` 等变量临时覆盖路径。
