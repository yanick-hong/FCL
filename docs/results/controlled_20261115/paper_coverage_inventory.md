# 论文全部实验覆盖清单（只读核查）

区分代码支持、任务计划、实际测量和论文历史协议一致性。此项核查未启动训练，未改源码/配置/执行器/远程 HEAD。完成情况采用已同步的 **2026-11-15T16:57:12.240211+00:00** 证据快照，不能作为当前远程状态。逐任务、指标、来源 SHA 见 [机器清单](paper_coverage_inventory.json)，较新状态见 [运行台账](README.md)。

## 按论文表和图核查

| 原文 | 实际范围 | 实现与计划 | 快照证据/缺口 |
|---|---|---|---|
| 正文Table1，p.4，标注质量 | 六数据集，CLIP正确率、UL-ER、FL-ER、校正后正确率及样本/类别数 | `compare_label_corrections.py` 四项均可得；UL-ER=uncorrected_error_rate，FL-ER=selected_error_precision | 主六数据集仅Caltech有当前正式报告；DTD是额外实验；当前划分差异需保留 |
| 正文Table2，p.6，CLIP | 8方法×6数据集=48单元 | 全部方法都有入口；FCL由核心队列提供，其余由跟进提供 | Caltech FCL已有3种子；Caltech CLIP zero-shot可由校正报告的test指标独立取得 |
| 正文Table3，p.7，Qwen | 5方法×6数据集=30单元；zero-shot/L2B/DLD/CSGN/FCL | 全部有入口；Qwen FL-C/NL-C/FS-C共18个是额外控制，表3未列 | Caltech Qwen FCL已有3种子、zero-shot有独立test报告；正式基线跟进未完成 |
| 正文Figure2，p.7 | CIFAR图例NL-C/L2B/DLD/CSGN/FCL；Caltech图例NL-C/L2B/DLD/FCL/FL-C；共10条，epoch1–100 | epochs suite默认16任务，其中6项为图外扩展；现有训练器记录逐epoch val | caption仅写classification accuracy，**未明确val/test**。数值与CLIP主表吻合，CLIP来源属于推断。不能直接将val曲线称论文测试曲线；如需test曲线，须各epoch保存checkpoint后离线评估且不按test选模型，当前best/last不足以回放 |
| 正文Figure3，p.8，lambda | CIFAR/Food/Tiny×[0,1,2,5,10,20]=18设置/来源/种子 | lambda suite可用；配置另加50/100，共24设置，额外6个必须标为扩展 | 当前未完成完整受控lambda消融；默认lambda1只覆盖一个点 |
| 正文Figure4，p.8，rho | CIFAR/Cars/Tiny×[.1,.2,.3,.4,.5]=15设置/来源/种子 | rho suite可用，需各rho独立校正文件 | 当前未完成完整受控rho消融；默认rho.3不能代表整图 |
| 正文Table4，p.8，prompt | Cars/Caltech/EuroSAT×A/B/C×FCL结果和VLM精度=18指标；9训练+9评估 | 必须显式 `--methods zero_shot fcl` 才有18任务；suite默认只产生9个FCL任务 | 当前仅C进入核心队列；Caltech C有测量，A/B缺失；内置通用A/B不能冒充历史模板 |
| 附录Table1/section3，p.1，骨干 | EuroSAT×ViT-L/14、ViT-B/16、ViT-B/32、ResNet-50×L2B/DLD/CSGN/FCL=16单元 | **没有正式backbone suite**；但EuroSAT提取器已有`--model_name`，线性头支持变维，可复用底层实现 | L/14的4项与主表重叠；缺3套独立特征/先验/校正缓存及12训练；ResNet-50的OpenAI CLIP模型id为`RN50`。附录未说明变骨干时伪标签先验是否固定，需明确实验选择 |
| 正文Figure1c，p.2，流程示意 | 86.95%、75.65%、47.68%、67.63%四个数 | 方法流程可复现，现有报告能独立给实测标注质量 | 未给这些数的汇总数据集/划分；caption只称示例图像来自Caltech，数值不等于Caltech主表。不能编造匹配 |

## 核心矩阵与七基线跟进

核心七数据集含额外DTD，四组×三训练种子=84训练，主六数据集占72。四组为CLIP原型+AUC、CLIP随机固定30%+AUC、CLIP同原型mask+softCE、Qwen原型+AUC。只有主六数据集的两来源原型+AUC对应主表FCL，合计12单元/36训练。

跟进计划为六主数据集×两来源×七非FCL方法=84任务（72训练+12零样本评估），对应66个论文单元及18个额外Qwen控制。因此核心FCL12+跟进论文66=全部**78个主表单元**；通用入口的96任务不能都称论文原表。

快照DTD和Caltech完成24/84核心训练。主表Caltech两个FCL单元有三种子，两个zero-shot单元在校正报告中有独立测量，共4个主表单元有当前测量，但正式跟进job完成仍为0。跟进队列必须等整个核心成功；本清单不认证其他审计任务正在加强的wrapper最终实现。

## 历史协议差异与无法恢复的部分

| 项目 | 论文/附录与当前恢复差异 |
|---|---|
| CIFAR/Tiny | 附录90/10训练划分与恢复45k/5k/10k、90k/10k/10k相符 |
| Caltech | 附录70/30后训练内90/10、102类；恢复2945/901/4831、101类；正文训练量与附录也冲突 |
| Food | 附录官方训练90/10应68175/7575；恢复65650/10100/25250，训练量与正文相符 |
| EuroSAT | 附录应17010/1890/8100；恢复21600/2700/2700，训练量与正文相符 |
| Cars | 正文训练7410；恢复7356/788/8041。附录只给90/10比例，未给取整索引 |
| Qwen版本与图像 | 论文Qwen-VL-7B、224×224；当前按用户指定用固定revision Qwen2.5-VL-7B-Instruct、448×448等效预算与合法id约束贪心解码，必须称明确记录的重建选择 |
| Prompt | A原文可恢复但`[0,{num-calsses}]`存在拼写和上界错误，应0..K−1；B只说跨语言、未给语言/全文，无法精确恢复历史B，重建需声明；C已有核对原文实现 |

L2B/DLD/CSGN有冻结特征适配实现，不能宣称引用论文端到端所有组件已逐项复现。DivideMix、FixMatch、MixMatch、MeanTeacher、Snorkel、ActiveAED仅属相关工作/背景引用，不是主表比较器。不能为追论文数字改写正在运行的冻结缓存或混用协议。

主文 `reference_backup/paper/Learning_from_Filter_Corrected_Labels.pdf` 共10页，已视觉核对p.6–8；附录 `reference_backup/Learning from filter-corrected lables/appendix.pdf` 共2页，已视觉核对p.1。另一个较长文件名PDF实际是2页Reproducibility Checklist。表1和图1同时按PDF文本核查。只读调用`build_jobs()`确认48/48/24/15/9(显式双方法18)/16计数，未调用execute。各表、图中默认点、种子计划互相重叠，不能直接相加为总训练量。
