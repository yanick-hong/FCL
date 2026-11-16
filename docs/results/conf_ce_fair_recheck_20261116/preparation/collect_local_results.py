"""Mirror completed records and generate a report from audited metrics."""
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import zipfile

ROOT = Path(__file__).resolve().parents[2]
NAME = 'conf_ce_fair_recheck_20261116'
SOURCE = ROOT / 'outputs' / NAME
DEST = ROOT / 'docs/results' / NAME
archive = ROOT / 'outputs' / (NAME + '.zip')
with zipfile.ZipFile(archive) as bundle:
    for item in bundle.infolist():
        target = (ROOT / 'outputs' / item.filename).resolve()
        if not target.is_relative_to((ROOT / 'outputs').resolve()):
            raise ValueError('Archive member escapes the intended outputs directory')
    bundle.extractall(ROOT / 'outputs')

audit = json.loads((DEST / 'final_audit.json').read_text(encoding='utf-8'))
assert audit['status'] == 'passed' and audit['sealed_trials'] == 50
assert audit['checkpoint_checks']['count'] == 30
payload = json.loads((SOURCE / 'results.json').read_text(encoding='utf-8'))
status = json.loads((SOURCE / 'status.json').read_text(encoding='utf-8'))
assert status['status'] == 'completed' and status['completed_trials'] == 50
assert len(payload['all_trials']) == 50 and len(payload['final_runs']) == 30
assert audit['binding'] == status['binding']
for name in ['results.json', 'summary.csv', 'all_trials.csv', 'status.json', 'study_binding.json', 'worker_launch.json', 'worker.log']:
    shutil.copy2(SOURCE / name, DEST / name)
for path in (SOURCE / 'trials').rglob('*'):
    if path.is_file() and path.suffix in {'.json', '.yaml', '.csv', '.log'}:
        target = DEST / path.relative_to(SOURCE)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)

names = [('cifar100', 'CIFAR-100'), ('caltech101', 'Caltech-101 (101 classes)'), ('eurosat', 'EuroSAT'), ('food101', 'Food-101'), ('tinyimagenet200', 'Tiny-ImageNet-200')]
groups = {(row['dataset'], row['method']): row for row in payload['summary']}
report = [
    '# Conf+CE 与 FCL+CE 公平复查结果', '',
    '状态：50/50次正式训练完成；独立审计通过，30个最终best checkpoint均已在CPU重新计算验证集和测试集准确率。', '',
    '比较范围是两种纠错样本选择策略之后使用同一种混合硬标签CE。两组纠错人数、特征缓存、划分和参数搜索预算相同。以下均为测试准确率百分数，±表示三个训练种子的样本标准差，差值单位为百分点。', '',
    '| Dataset | Conf+CE mean ± SD | FCL+CE mean ± SD | FCL − Conf (pp) | LR (Conf / FCL) |',
    '|---|---:|---:|---:|---:|'
]
wins = 0
for key, title in names:
    conf, proto = groups[key, 'clip_conf_ce'], groups[key, 'fcl_filter_ce']
    difference = proto['mean_percent'] - conf['mean_percent']
    wins += difference > 0
    report.append(f"| {title} | {conf['mean_percent']:.3f} ± {conf['sample_sd_percent']:.3f} | {proto['mean_percent']:.3f} ± {proto['sample_sd_percent']:.3f} | {difference:+.3f} | {conf['learning_rate']:g} / {proto['learning_rate']:g} |")
report += ['', f'按数据集三种子均值比较，FCL+CE领先 {wins}/5，Conf+CE领先 {5-wins}/5。本轮结果不支持“FCL+CE至少在两到三个数据集占优”。全部候选与所有不利结果均保留。', '',
    '## 每个种子与区间', '',
    '| Dataset | Method | Seed 42 | Seed 43 | Seed 44 | Mean 95% t CI |',
    '|---|---|---:|---:|---:|---|']
for key, title in names:
    for method, short in [('clip_conf_ce', 'Conf+CE'), ('fcl_filter_ce', 'FCL+CE')]:
        row = groups[key, method]
        assert row['seeds'] == [42, 43, 44]
        values = row['test_accuracies_percent']
        assert abs(statistics.mean(values) - row['mean_percent']) < 1e-10
        assert abs(statistics.stdev(values) - row['sample_sd_percent']) < 1e-10
        report.append(f"| {title} | {short} | {values[0]:.3f} | {values[1]:.3f} | {values[2]:.3f} | [{row['ci95_low_percent']:.3f}, {row['ci95_high_percent']:.3f}] |")
report += ['', '## 实际配置与限制', '',
    '- 两方法同一学习率网格0.001/0.002/0.004；seed42按最高验证准确率选LR，并列选较小LR。所有十个条件均选中0.004，该结论限于预设网格。',
    '- AdamW、weight decay 0.0001、CosineAnnealingLR(T_max=100)、100epochs、patience101、batch1024；相同Xavier线性头与全数据shuffle，梯度裁剪5。',
    '- CPU执行，torch/BLAS线程2、DataLoader workers2，最多三个数据集同时运行。现有GPU队列、源码与配置未改。',
    '- seed42参与LR选择并被纳入42/43/44汇总；95% t区间按自由度2计算，只作给定数据、标签、mask及选中LR下的探索性、条件性描述。',
    '- 恢复的Caltech为101类，与原文102类不同；新缓存不能证明与所有历史稿件缓存相同。新数据属于独立受控复查，未替换论文旧数值，论文实验说明和设置未改。',
    '- 独立pilot仅用于资源/CLI检查，未计入50次正式训练和参数选择。', '',
    '## 文件位置', '',
    '- 本地易读汇总：本文件；机器可读汇总：`summary.csv`、`results.json`；全部候选：`all_trials.csv`。',
    '- 每次命令、配置、原始metrics、验证曲线和seal：`trials/<dataset>/<method>/lr-*/seed-*/`。',
    '- 输入与源码：`plan.json`、`preflight.json`、`study_binding.json`、`search_protocol.json`；独立审计：`final_audit.json`。',
    '- 本地完整模型与运行日志：`F:/CodeProjects/FCL2/outputs/conf_ce_fair_recheck_20261116/`；完整备份：同级同名`.zip`。',
    '- AutoDL原件：`/root/autodl-tmp/outputs/conf_ce_fair_recheck_20261116/`；审计原件：`/root/autodl-tmp/outputs/conf_ce_fair_audits/final-audit-20261116.json`。',
    '- 正式实验源码HEAD：`1fec7c895c11a80493f0153e6ec2870ffa322bed`。', '',
    f"完整备份SHA256：`{hashlib.sha256(archive.read_bytes()).hexdigest()}`。", ''
]
(DEST / 'RESULTS.md').write_text('\n'.join(report), encoding='utf-8')
print(json.dumps({'completed': 50, 'independently_recomputed_final_models': 30, 'fcl_ce_dataset_wins': int(wins), 'report': str(DEST / 'RESULTS.md'), 'archive_bytes': archive.stat().st_size, 'archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest()}, indent=2))
