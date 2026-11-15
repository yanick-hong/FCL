# 项目审计与已知限制

本文件记录恢复项目中发现的会改变实验结论、降低复现稳定性或扩大输入攻击面的事项。修复已经写入当前工作树；论文目标数字仍只作为参考，不能替代实际运行产生的 `metrics.json`。

## 已修复的问题

| 级别 | 位置 | 问题 | 处理 |
| --- | --- | --- | --- |
| 高 | `src/contrast/train_dcd.py` | 已校正样本的原型、训练目标和类别分配曾直接读取真实 `train_labels`，绕过 `y_obs`。在非完美人工校正或标签扰动实验中会泄漏真实答案。 | 改为只使用观察标签。100% 正确的模拟校正下二者数值相同；不能据此证明旧结果全部失效。 |
| 高 | `src/contrast/train_nlprompt.py` | trusted CE 曾直接使用真实训练标签，忽略观察标签中的人工误标。 | 改为使用 `y_obs` 中的校正结果。 |
| 高 | DCD、DLD-cache、IDO-cache、NLPrompt、revision baselines | 早停后保存了 `best.ckpt`，但测试指标仍用最后一轮模型，导致报告结果与选择规则不一致。 | 测试前重新加载验证集最优 checkpoint。 |
| 高 | L2B、DLD、CSGN 的 PK sampler | L2B/DLD 每个 epoch 重复类别随机序列，CSGN 未绑定实验 seed，DLD 另有写死 `seed=42`。 | 使用 seed+epoch 的独立随机流，并传递用户 seed；DLD/CSGN 类别不足时拒绝。FCL/L2B 保留原有类别不足时有放回采样规则。 |
| 高 | `src/contrast/train_l2b.py` | 缺少 CLIP 概率字段时曾伪造均匀分布，实际改变了标签来源。 | 缺失、形状错误、非概率行或 argmax 不一致时直接拒绝。 |
| 高 | 论文运行器、结果汇总器 | resume 曾信任 completed 标记，外部配置变化未进入指纹，NaN 或无测试指标的 metrics 也可能被误认为完成。 | 要求有效有限测试准确率，并将协议参数纳入任务指纹；汇总器遵循 `OUTPUT_ROOT`。 |
| 中 | `src/contrast/cache_experiment.py`、标注导出 | 普通缓存加载未限制为安全张量格式，且缺少尺寸、有限值、标签范围和 `s∈{0,1}` 校验。 | 优先使用 `weights_only=True`，并在统一缓存入口执行 schema 校验。 |
| 高 | Qwen 标注导出 → 论文运行器 | exporter 的来源字段不被 runner 识别；只替换 train 时，test 仍可能保留 CLIP 预测。 | 按 split 核验 `annotation_metadata`、Qwen 模型来源、`replaced_clip_fields` 和已知 prompt variant；Qwen zero-shot 强制要求 Qwen test，未覆盖的 CLIP split 不入 Qwen 指标。 |
| 中 | Windows 论文运行器 | 子进程默认 GBK、父进程按 UTF-8 解码，转发日志可能抛 `UnicodeEncodeError` 而终止有效训练。 | 父子进程统一 UTF-8 日志；已用八方法 CPU 流程验证。 |
| 中 | `scripts/evaluate_fcl_checkpoints.py` | 用标签最大值推断输出类别，split 缺类时无法加载完整分类头。 | 从 checkpoint 权重形状确定类别数。 |

## 尚存的风险与边界

1. 主入口优先安全加载，但旧 PyTorch 的兼容回退仍使用 pickle。旧独立入口如 `train_trusted.py`、`train_noise.py`、部分 `utils/eval/`、`utils/labels/`、`utils/extract/` 还存在未显式传 `weights_only=True` 的加载；在 PyTorch 2.5 等版本下不保证禁用 pickle。仅应读取可信缓存，完整列表可用 `rg -n 'torch.load' src scripts` 查看。此项尚未全面消除。
2. `train_trusted.py` 中的 `pickle.load` 仅用于官方 CIFAR 原始 pickle 包，不能用于用户提供的任意 pickle 文件。
3. DLD、CSGN、L2B 等当前实现是冻结 CLIP 特征上的 cache-level 适配，论文引用的方法原本可能包含图像增强或端到端训练。因此结果应标注为恢复协议/适配实现，不能宣称逐层复现原论文训练管线。
4. Caltech-101 的多个基线实测值明显低于论文表格；该差异已保留在 `docs/remote_clip_results_seed42.md`，没有用论文数字覆盖实测结果。
5. L2B 的 bootstrap 分支按旧实现使用当前 head 的 `argmax`。曾考虑改为固定 VLM 预测，核对 `reference_backup/code_temp/train_l2b.py` 与 `reference_backup/对比方法/train_l2b_CIFAR100.py` 后撤回；缺少算法证据时不以接近论文数字为由改写方法。

## 数据和远程状态限制

- Food-101 已完成下载、`gzip -t` 校验、解压、特征/标签准备和八个主实验；结果见 `docs/remote_clip_results_seed42.md`。原始压缩包和运行输出仍属于 AutoDL/本地运行时数据，不纳入 Git。
- Stanford Cars 缺少官方带标签测试文件 `cars_test_annos_withlabels.mat`，因此不能把 Cars 缺失行写成“已复现”。
- Qwen 主表、三种 prompt 的真实模型生成，以及全部多种子/消融套件尚未完成实测；具备代码入口和集成测试不等于已经复现全部论文结果。
- AutoDL 的 SSH、磁盘和后台下载状态属于外部运行时条件；本地代码、配置和缓存指纹已经与恢复分支绑定，但远程状态变化后必须重新执行诊断和准备检查。

## 验证结果

当前工作树已通过：

```text
29 passed, 12 subtests passed
python -m compileall -q src scripts ablation tests
8/8 main-suite methods: synthetic CPU smoke, jobs=8 errors=0
```

八方法烟雾测试只验证输入准备、命令调用、训练、checkpoint 和 metrics 的衔接，合成数据的数值不属于论文复现结果；本地日志位于 `outputs/audit_smoke_console.log`。

运行前建议执行 `python scripts/run_paper_experiments.py --preflight --strict-paper-splits`；执行后只以输出目录中的 `metrics.json`、命令、配置、日志和 fingerprint 作为结果证据。
