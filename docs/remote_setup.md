# AutoDL 与 VLM 标签管线

当前受控运行已固定使用用户确认的 `Qwen/Qwen2.5-VL-7B-Instruct`，官方 revision 为 `cc594898137f460bfe9f0759e9844b3ce807cfb5`，独立环境依赖见 `requirements-qwen.txt`。Cars 官方带标签测试输入与 CLIP 缓存现已齐全。下文保留早期准备记录；本轮的实际状态、配置与证据以 [运行台账](results/controlled_20261115/README.md) 和 [受控协议](controlled_experiments.md) 为准。

本轮实验源码固定在 `1fec7c895c11a80493f0153e6ec2870ffa322bed`。活动队列核验 HEAD、源码、模型和配置指纹；运行期间远端保留该提交，不执行 `git pull` 或修改源码。之后提交到 GitHub 的文档/结果快照不会自动切换远端运行版本，实验记录继续明确引用实际源码提交。

2026-11-15 已通过 SSH 只读核查：远端 `/root/autodl-tmp` 是现有 Git 项目，origin 为 `https://github.com/yanick-hong/FCL.git`，当时 HEAD 为 `42072ed`。GPU 为 RTX 4090 24 GB，驱动 550.100；base 环境在 `/root/miniconda3`。数据盘约有 47 GB 空余。`dataset/processed` 已有 CIFAR-100、Tiny-ImageNet、Caltech-101、EuroSAT、DTD 的特征与标签缓存；Food-101、Stanford Cars 仍需准备。这些是现场检查结果，不代表之后环境未改变。

```powershell
ssh -p 32132 root@connect.westb.seetacloud.com
python scripts/remote_diagnose.py --interactive-password --output outputs/remote_diagnostic.json
```

## 历史快照：受控运行之前（2026-11-15）

该次核查时，AutoDL 的 Food-101 分片下载、gzip 校验、解压、CLIP 特征/观察标签准备以及八个主套件任务均已完成；实测结果记录在 `docs/remote_clip_results_seed42.md`。当时 Stanford Cars 缺少官方带标签测试文件，尚未纳入完整表；这项输入缺口现已补齐。当时也没有运行中的 Food 下载或实验进程；本轮队列状态以文首链接为准。`dataset/`、`outputs/` 和远端 `.Trash-0/` 属于运行时状态，不纳入版本控制。

密码仅在 OpenSSH 提示时交互输入，不写入脚本、命令参数或 Git。诊断脚本只读取主机、GPU、依赖版本、Git 和缓存清单，连接失败会重试一次。已有密钥可使用 `--identity-file`。连接中断后先检查 AutoDL 实例是否开机、端口是否改变以及本机网络。

## 统一标签文件

所有模型使用同一份按类别 ID 排序的 `classes.json`（字符串数组），以及能追溯到特征抽取顺序的 manifest。manifest 可用 JSONL、JSON 数组或 CSV；每行包含 `sample_id`、`path`、`split`。路径相对于推理时 `--image-root`，也可使用绝对路径。CIFAR 图片可先导出 PNG，以原官方索引形成稳定 sample ID。禁止根据重排后的文件列表推断旧标签顺序。

```json
{"sample_id":"train/image_00001","path":"images/image_00001.jpg","split":"train"}
```

输出 JSONL 每行保存 `sample_id/path/split/label_id/class_name/probabilities/probability_source/source_model/alignment/raw_output`，类别 ID 一律从 0 开始。`.meta.json` 保存 manifest、类别顺序、模型和设置指纹。恢复运行会验证指纹、跳过已完成样本；末行若因中断而不完整，会先备份到 `.partial-tail.bak` 再恢复。无效、越界、重复 ID 或与概率 argmax 不一致的标签会报错；不会截取输出的第一个字符。

旧纯文本预测标签的每行只有一个编号，必须显式确认 manifest 与旧预测的完整原始顺序。旧 `train_label_tf.txt` 是正确/错误指示，`train_label_t.txt` 是真值；预测文件应是 `train_label_pre.txt`。

```bash
python -m src.utils.annotation import \
  --manifest dataset/processed/train_manifest.jsonl --split train \
  --classes dataset/processed/classes.json --input train_label_pre.txt \
  --source-model Qwen/Qwen2.5-VL-7B-Instruct --ordered-legacy \
  --output dataset/processed/qwen_train.jsonl
```

`--label-base 1` 可导入从 1 开始的旧标签；默认从 0 开始。有 sample ID/path 的 JSONL 或 `identity<TAB>label` 文本会按身份对齐，不依赖输入顺序。JSONL 可提供 `probabilities`；缺失时生成 one-hot，并明确标记 `one_hot_from_label`。one-hot 只代表离散标签，不能解释成经过校准的模型置信度。`--allow-partial` 可导入尚未完整的身份标签，缓存导出仍要求完整。

## 可选 Qwen 推理

Qwen2.5-VL 使用可选依赖 `transformers>=4.49, accelerate, pillow`；安装前先核查当前 CUDA/PyTorch。模型及依赖并非项目烟雾测试的必要条件。备份脚本实际加载 `Qwen/Qwen2.5-VL-7B-Instruct`，论文称 Qwen-VL-7B；这是尚须在结果中区分的版本差异。

```bash
python -m src.utils.annotation qwen \
  --manifest dataset/processed/train_manifest.jsonl --split train \
  --classes dataset/processed/classes.json --image-root dataset/raw/food101 \
  --model Qwen/Qwen2.5-VL-7B-Instruct --backend qwen2.5-vl \
  --output dataset/processed/qwen_train.jsonl --resume
```

第一次运行省略 `--resume`；后续恢复加上。`--limit 10` 可先验证十个样本。默认严格解析仅含类别 ID 的 JSON 或单个完整编号/类别名，遇到无效输出保存错误日志后退出；`--continue-on-error` 会继续处理其他图片，最终仍返回失败状态，未成功样本下次恢复会重试。

如确需历史 `Qwen/Qwen-VL-Chat`，显式设置 `--backend qwen-vl-chat --model Qwen/Qwen-VL-Chat --trust-remote-code`；它与 Qwen2.5-VL 是不同架构，依赖及指标分别记录。默认 A/B/C 是当前实现的中英文提示模板，不宣称是论文原提示词；复现原文时用 `--prompt-file` 指定包含 `{classes}` 占位符的确切模板。

## 接入冻结特征训练

```bash
python -m src.utils.annotation export-cache \
  --manifest dataset/processed/train_manifest.jsonl --split train \
  --classes dataset/processed/classes.json \
  --annotations dataset/processed/qwen_train.jsonl \
  --cache dataset/processed/food101_clip_vit-l-14_embeddings.pt \
  --output dataset/processed/food101_qwen_embeddings.pt --replace-clip-fields
```

缓存优先依据 `train_sample_ids` 或 `train_paths` 对齐。旧缓存缺少身份时，只有已找回确切抽取顺序的原始 manifest 才可显式加 `--confirm-cache-manifest-order`，这一假设将保存在元数据里。生成的新缓存保存 `vlm_probs_train/vlm_argmax_train`。现有训练读取 `clip_*` 字段时，`--replace-clip-fields` 显式写入对应 VLM 标签并将原 CLIP 字段保留为 `original_clip_*`；输入缓存不改写。随后用该缓存重新生成原型和筛选标签，再训练，避免混用原 CLIP 筛选分区与 Qwen 标签。
