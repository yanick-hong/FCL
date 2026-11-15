"""Summarize exactly one paired ablation campaign and render scientific figures."""
import argparse
import csv
import json
from pathlib import Path
import statistics as stats

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES={'dtd':'DTD','caltech101':'Caltech-101','eurosat':'EuroSAT','cifar100':'CIFAR-100','tinyimagenet200':'Tiny-ImageNet'}


def csv_write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',type=Path,required=True)
    a=p.parse_args();root=a.campaign
    config=json.loads((root/'campaign.json').read_text())['config']
    records={}
    for dataset in config['datasets']:
        for accuracy in config['accuracies']:
            for seed in config['seeds']:
                path=root/dataset/f'acc_{round(accuracy*100):03d}'/f'seed_{seed}'/'metrics.json'
                result=json.loads(path.read_text())
                assert result['status']=='completed' and result['test_checkpoint']=='best.ckpt'
                assert result['accuracy_requested']==accuracy and result['seed']==seed
                records[dataset,accuracy,seed]=result
    summary=[];pairs=[];runs=[]
    for dataset in config['datasets']:
        for accuracy in config['accuracies']:
            group=[records[dataset,accuracy,s] for s in config['seeds']]
            values=[100*r['test_accuracy'] for r in group]
            drops=[100*(records[dataset,1.,s]['test_accuracy']-records[dataset,accuracy,s]['test_accuracy']) for s in config['seeds']]
            summary.append({'dataset':dataset,'human_accuracy_requested_pct':100*accuracy,
                            'human_accuracy_actual_pct':100*stats.mean(r['accuracy_actual'] for r in group),
                            'n':len(group),'test_accuracy_mean_pct':stats.mean(values),
                            'test_accuracy_std_pct':stats.stdev(values) if len(values)>1 else 0.,
                            'paired_drop_mean_pp':stats.mean(drops),
                            'paired_drop_std_pp':stats.stdev(drops) if len(drops)>1 else 0.})
            for r,drop in zip(group,drops):
                pairs.append({'dataset':dataset,'human_accuracy_pct':100*accuracy,'seed':r['seed'],'drop_pp':drop})
                runs.append({'dataset':dataset,'human_accuracy_pct':100*accuracy,'seed':r['seed'],
                             'test_accuracy_pct':100*r['test_accuracy'],'best_val_accuracy_pct':100*r['best_val_accuracy'],
                             'best_epoch':r['best_epoch'],'epochs_completed':r['epochs_completed'],
                             'elapsed_seconds':r['elapsed_seconds'],'actual_human_accuracy_pct':100*r['accuracy_actual']})
    csv_write(root/'summary.csv',summary);csv_write(root/'paired_drops.csv',pairs);csv_write(root/'runs.csv',runs)
    lines=['# 人工纠正噪声消融','', '测试准确率：均值 ± 样本标准差（%，3个配对种子）。下降为相对100%正确率基线的百分点差。','',
           '| 数据集 | 100% | 95% | 90% | 85% | 80% |','|---|---:|---:|---:|---:|---:|']
    for dataset in config['datasets']:
        rows=[r for r in summary if r['dataset']==dataset]
        lines.append('| '+NAMES[dataset]+' | '+' | '.join(f"{r['test_accuracy_mean_pct']:.2f} ± {r['test_accuracy_std_pct']:.2f}" for r in rows)+' |')
    lines+=['','只测试 FCL+AUC-C，不能据此单独推断 AUC 优于 CE；合成均匀错误不等价于真实人工标注研究。']
    (root/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for metric,error,filename,ylabel in [('test_accuracy_mean_pct','test_accuracy_std_pct','accuracy_vs_human_correctness','Test accuracy (%)'),
                                        ('paired_drop_mean_pp','paired_drop_std_pp','accuracy_drop','Paired accuracy drop (pp)')]:
        fig,axes=plt.subplots(1,len(config['datasets']),figsize=(17,3.6),squeeze=False)
        for ax,dataset in zip(axes[0],config['datasets']):
            group=[r for r in summary if r['dataset']==dataset]
            ax.errorbar([r['human_accuracy_requested_pct'] for r in group],[r[metric] for r in group],
                        yerr=[r[error] for r in group],marker='o',capsize=3,linewidth=1.5)
            ax.set_title(NAMES[dataset]);ax.set_xlabel('Human correctness (%)')
            ax.set_xticks([100,95,90,85,80]);ax.set_xlim(101,79);ax.grid(alpha=.25)
            if metric.startswith('paired'):ax.axhline(0,color='gray',linewidth=.8)
        axes[0,0].set_ylabel(ylabel)
        fig.suptitle('FCL+AUC-C | mean ± sample SD, 3 paired seeds',fontsize=12)
        fig.tight_layout()
        for suffix in ('png','pdf'):fig.savefig(root/f'{filename}.{suffix}',dpi=200,bbox_inches='tight')
        plt.close(fig)
    print((root/'comparison.md').read_text(),flush=True)


if __name__=='__main__':main()
