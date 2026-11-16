# 独立CE实验核验

`audit_ce_results.py`不导入或修改正在运行的wrapper，不改任何study文件。默认核验完全使用Python标准库；只有显式添加`--checkpoints`才导入torch。

## 正在运行时

```bash
/root/miniconda3/bin/python /path/to/audit_ce_results.py /root/autodl-tmp/outputs/conf_ce_fair_recheck_20261116 --allow-partial --verify-inputs --verify-source --runner /path/to/fair_ce_runner.py
```

`--allow-partial`允许sealed结果先于results.json快照出现；仍会拒绝引用不存在的seal、错误参数、坏指纹、错误LR选择和错误统计。不检查worker必须完成，也不把partial通过当成50次训练完成。并发读取只接触已封存的trial，不把未封存目录当成有效结果。

## 最终交付前

```bash
/root/miniconda3/bin/python /path/to/audit_ce_results.py /root/autodl-tmp/outputs/conf_ce_fair_recheck_20261116 --verify-inputs --verify-source --runner /path/to/fair_ce_runner.py --checkpoints --output /root/autodl-tmp/outputs/conf_ce_fair_audits/final-audit.json
```

将两个`/path/to/`换成实际部署的外部审计脚本和冻结wrapper路径。`--output`必须指向study之外的新文件，拒绝覆盖已有报告。不指定`--output`时只输出stdout。此脚本不修改seal，也不为不完整trial补封存。

默认完整五数据集计划需50个sealed trial、30个final run、10个LR选择和10份三seed汇总；子集pilot按其plan中的数据集数相应缩小。final run固定为每个方法在验证集挑出的LR下42/43/44三次训练。42号种子参与了LR选择，置信区间仅作探索性、条件性描述，不能当成独立于调参的新确认样本。

## 核验内容

- Study binding由plan、preflight输入及runtime重新计算；实际输入/源码核查可按flag开启。
- 每个seal的完整文件清单与SHA重新计算。从metrics重建result，逐字段对照seal.result、launch、run_metadata和results.json引用，避免resume只信seal.result。
- 验证命令、实际参数、同人数纠错预算、100轮完整history、验证集首次最高点、测试样本数量及有限准确率。
- 每组三个LR的选择只按best validation accuracy，并列选择最小LR；对照测试分数不进入选择。
- 验证trial/final无重复、final LR与所选LR一致，逐seed准确率、均值、样本标准差和自由度2的95%t区间。
- `--checkpoints`在CPU直接载入每个final best.ckpt，按旧代码相同的L2特征归一化、4096评估批大小及softmax/argmax重算val/test准确率。核对checkpoint epoch等于history首次最高点，last.ckpt epoch为100。

从本地下载副本核验标准库核心时，可使用`--recorded-study-root /root/autodl-tmp/outputs/conf_ce_fair_recheck_20261116`映射记录中的远端run_dir；实际输入/源码和checkpoint复算应在能访问对应缓存的主机执行。

## 不依赖torch的测试

```powershell
python tmp/conf_ce_review/test_audit_ce_results.py
```

8项测试覆盖完整核验、seal.result被改而files未改、checkpoint字节改变、results坏引用、以test赢家替换validation赢家、总体SD冒充样本SD，以及两种partial快照窗口。测试不会加载torch或真实模型文件。
