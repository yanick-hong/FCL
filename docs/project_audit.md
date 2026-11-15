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
6. 直接调用旧论文运行器的 `--resume` 只校验输入/源码指纹、完成标记和有限测试准确率，尚不校验 checkpoint/history 的完整字节。缺少 checkpoint 或被修改的输出不能仅凭其 completed 标记认定有效。本轮核心套件通过产物 receipt 管理复用，后续基线包装器另行核验 100-epoch 历史、最佳 checkpoint 的 CPU 测试结果和输出清单 SHA；旧入口本身的限制仍应保留。

## 数据和远程状态限制

- Food-101 已完成下载、`gzip -t` 校验、解压、特征/标签准备和八个主实验；结果见 `docs/remote_clip_results_seed42.md`。原始压缩包和运行输出仍属于 AutoDL/本地运行时数据，不纳入 Git。
- Stanford Cars 的官方带标签测试文件现已找到并校验，CLIP 缓存已生成，划分为 7356/788/8041；详见 `docs/results/controlled_20261115/preparation/stanfordcars_preparation.json`。输入准备完成与训练实验完成分别记录。
- Qwen 主表、三种 prompt 的真实模型生成，以及全部多种子/消融套件尚未完成实测；具备代码入口和集成测试不等于已经复现全部论文结果。
- 按正文/附录逐表逐图核查的缺口见 [论文实验覆盖清单](results/controlled_20261115/paper_coverage_inventory.md)：Prompt B 历史全文未公开，附录骨干实验尚缺正式编排与缓存，图 2 评估集语义未明确。不得把相关工作中提及的方法、额外控制或配置扩展点混计为已完成论文单元。
- AutoDL 的 SSH、磁盘和后台下载状态属于外部运行时条件；本地代码、配置和缓存指纹已经与恢复分支绑定，但远程状态变化后必须重新执行诊断和准备检查。

## 验证结果

此前恢复审计已通过：

```text
29 passed, 12 subtests passed
python -m compileall -q src scripts ablation tests
8/8 main-suite methods: synthetic CPU smoke, jobs=8 errors=0
```

八方法烟雾测试只验证输入准备、命令调用、训练、checkpoint 和 metrics 的衔接，合成数据的数值不属于论文复现结果；本地日志位于 `outputs/audit_smoke_console.log`。

运行前建议执行 `python scripts/run_paper_experiments.py --preflight --strict-paper-splits`；执行后只以输出目录中的 `metrics.json`、命令、配置、日志和 fingerprint 作为结果证据。

## 本轮受控实验发现与修复

| 问题 | 本轮处理与证据 |
| --- | --- |
| 旧缓存缺少图片身份/路径，或同时存在互相冲突的 flat/nested 别名。 | CIFAR 用官方像素、类别表和保存索引恢复；Caltech/EuroSAT 重放原 torchvision 加载器及保存索引；Tiny/DTD 按原 sample ID 和官方标签/partition 恢复。全部别名必须一致，拒绝重新枚举排序、越界/复数索引、非整数标签或 split 重叠。原缓存保持只读。 |
| Caltech 提取器依赖通常不存在的 `_image_files`，会写出空 `sample_paths`。 | 根据 torchvision 的 `categories/y/index` 构造原文件名，保留原相对索引。 |
| manifest 或类别顺序变化后，推理评分可能与冻结缓存行错位。 | 推理前验证完整 manifest、类别表和缓存 SHA；导出再按真实 ID 对齐。重复类名允许用原 WordNet ID 确定性消歧，类别编号不变。 |
| 断点恢复可能混入不同模型文件、处理器、依赖或代码版本。 | 保存全部本地模型文件、推理环境和标注源码指纹，每个 split 的 sidecar 绑定模型环境；不一致时拒绝恢复。 |
| 真实 Qwen 在 DTD val 生成 `48`，超出合法 0–46。 | 旧尝试停止并单独保存，未截断/映射该输出。正式矩阵统一重新生成，使用多 token 类别 ID trie 约束 greedy 解码；仍保留原始输出并严格复核。此为显式的新解码协议，不与旧结果合并。 |
| 随机 30% 与每类 ceil 的原型预算并不严格相同。 | 分别报告实际人数/比例，额外保存与原型人数完全一致的随机对照；随机核心训练固定同一选择种子 42 的 mask，训练种子 42/43/44 不重抽 mask。 |
| 损失对照或多种子可能混用初始化、采样、监督或选择规则。 | 两种目标共用固定输入、头初始化、PK 流、AdamW/OneCycleLR 和 100 epochs；按最高验证准确率首次出现的 checkpoint 测试。保存并核对所有相关 SHA。 |

本轮回归为 **72 passed, 12 subtests passed**。DTD 和 Caltech-101 均已完整完成四组、每组三种子；两者独立审计分别 **695/695** 和 **663/663** 项通过，共 24 个最佳 checkpoint 的 CPU 测试正确数与记录精确相同，亦核验了验证选点、每 epoch 采样流和跨种子独立性。完整审计及范围见 [DTD 审计](results/controlled_20261115/dtd_independent_audit.json)和 [Caltech-101 审计](results/controlled_20261115/caltech_independent_audit.json)。Caltech 未重复从 seed 构造初始化字节，但已核验同种子两目标的初始化一致、跨种子初始化不同；该边界在报告中单独记录。

基线后续队列另通过 **27 项 CPU 合成产物校验测试、12 项计划/依赖/进程锁检查**和 Caltech 的 CLIP/Qwen 双来源预检。它拒绝未封存的旧结果，逐项核对完整 100-epoch 历史、checkpoint CPU 评估与输出清单 SHA；还验证了重复启动被拒绝，以及恢复过程中再次中断不会丢失后续已完成项的 receipt。详见 [最终包装器验证](results/controlled_20261115/preparation/paper_baseline_receipts_v2/final_validation.json)。这些检查不代表 84 项真实基线工作已经完成。旧方法的 CSV 精度和 L2B 最高验证值相同时覆盖 checkpoint 的规则在每项 receipt 中显式记录，不与核心训练器的首次最高值规则混淆。

DTD 同一原型 mask 下，`fcl_auc - ce_soft` 测试差值为 −5.6738 个百分点，95% 配对 t 区间为 [−6.0063, −5.3412]，保留该实测结果。各数据集的标签准确率按 train/val/test 单独记录；例如 Caltech 的 CLIP train 92.8693% 与 test 85.0756% 差别明显，比较论文前必须核对 split 与指标定义。完整矩阵的未完成项继续保留在 [运行台账](results/controlled_20261115/README.md)。
