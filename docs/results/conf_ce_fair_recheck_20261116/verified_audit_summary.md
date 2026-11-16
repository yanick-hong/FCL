# Conf / prototype 纠错选择的独立只读审计

本次仅在CPU读取既有缓存与prototype.pt，未训练、未改论文、未写远端文件。数据记录为`selection_audit.json`，可复现的只读脚本为`selection_audit_remote.py`。

已直接调用远端既有`src/contrast/train_revision_baselines.py`的`get_clip_prior()`，并断言输出与`prob.float().clamp_min(1e-6)/prob.float().sum(1,keepdim=True).clamp_min(1e-6)`逐元素完全相等。置信度选择严格使用既有代码的`torch.topk(prior.max(1).values, verified_count, largest=False).indices`。未使用新的阈值、排序或校准。

所调用源码文件SHA256为`87a65690141801845401bdbe860ab31047de3f4a295447ee68f2ba0b5f592468`。

每集验证了实际缓存文件SHA等于既有纠错报告的cache SHA，prototype mask SHA等于报告记录，选中位置的y_obs等于真值，未选中位置的y_obs等于原CLIP伪标签。JSON同时记录两种mask、prior、原始伪标签、两组y_obs和prototype输入文件的完整SHA。所有评分均在固定选择完成后计算，不参与选择。

| Dataset | 相同纠错人数 | 原型校正标签准确率(%) | Conf校正标签准确率(%) | 原型−Conf(pp) |
|---|---:|---:|---:|---:|
| CIFAR-100 | 13,546 | 87.088889 | 90.380000 | −3.291111 |
| Caltech-101 | 914 | 97.453311 | 99.626486 | −2.173175 |
| EuroSAT | 6,485 | 62.430556 | 64.837963 | −2.407407 |
| Food-101 | 19,738 | 96.089871 | 98.779893 | −2.690023 |
| Tiny-ImageNet | 27,092 | 87.938889 | 91.475556 | −3.536667 |

这些是训练标签准确率，不是分类器测试准确率。Conf在五集的同人数纠错质量均较高，不能预先推断公平重训后FCL+CE必然获胜。Caltech缓存为101类，与正文102类不等同。

资源原始证据在`resource_snapshot.json`和`resource_snapshot_v1.json`：cgroup v1 CPU额度22核，容器内存上限90GiB，快照使用约42.81GiB、剩余约47.19GiB。可见128逻辑核与宿主MemAvailable并非容器真实额度。根代理计划三个CPU数据集进程、每个torch两线程和原有两个DataLoader worker，GPU队列保持原状；应先测时并控制BLAS线程与内存预取。

只读快照时，原core仍在Tiny受控训练，现paper baseline followup为pending_core且没有启动训练；其84项计划不包含Conf+CE和FCL+CE。不应修改该队列已封存的源码或配置。原文章两方法的对应metrics.json未在远端outputs发现，旧表数字不能充当本次复跑输入/输出凭证。
