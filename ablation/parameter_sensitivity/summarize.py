"""Aggregate parameter sensitivity results and render compact plots."""
import argparse, csv, json, statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES = {'dtd': 'DTD', 'caltech101': 'Caltech-101', 'eurosat': 'EuroSAT',
         'cifar100': 'CIFAR-100', 'tinyimagenet200': 'Tiny-ImageNet'}


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--campaign', type=Path, required=True)
    root = parser.parse_args().campaign.resolve(); config = json.loads((root / 'campaign.json').read_text())['config']
    rows, long = [], []
    for dataset in config['datasets']:
        for sweep, values in [('mu', config['mu_values']), ('tau', config['tau_values'])]:
            for value in values:
                records = []
                for seed in config['seeds']:
                    name = f'{sweep}_{float(value):.2f}'.replace('.', 'p')
                    records.append(json.loads((root / dataset / name / f'seed_{seed}' / 'metrics.json').read_text()))
                scores = [100 * record['test_accuracy'] for record in records]
                vals = {'dataset': dataset, 'sweep': sweep, 'value': float(value),
                        'n': len(scores), 'test_accuracy_mean_pct': statistics.mean(scores),
                        'test_accuracy_std_pct': statistics.stdev(scores),
                        'best_val_mean_pct': statistics.mean(100 * r['best_val_accuracy'] for r in records)}
                rows.append(vals)
                for record, score in zip(records, scores):
                    long.append({'dataset': dataset, 'sweep': sweep, 'value': float(value),
                                 'seed': record['seed'], 'test_accuracy_pct': score,
                                 'best_val_accuracy_pct': 100 * record['best_val_accuracy'],
                                 'best_epoch': record['best_epoch'], 'epochs_completed': record['epochs_completed']})
    write_csv(root / 'summary.csv', rows); write_csv(root / 'runs.csv', long)
    lines = ['# AUC-C parameter sensitivity', '', 'Test accuracy: mean $\\pm$ sample SD over three seeds.', '']
    for sweep in ('mu', 'tau'):
        lines.append(f'## {sweep}')
        lines.append('| Dataset | ' + ' | '.join(str(v) for v in (config['mu_values'] if sweep == 'mu' else config['tau_values'])) + ' |')
        lines.append('|---|' + '---:|' * len(config['mu_values'] if sweep == 'mu' else config['tau_values']))
        for dataset in config['datasets']:
            group = [r for r in rows if r['dataset'] == dataset and r['sweep'] == sweep]
            lines.append('| ' + NAMES[dataset] + ' | ' + ' | '.join(f"{r['test_accuracy_mean_pct']:.2f} $\\pm$ {r['test_accuracy_std_pct']:.2f}" for r in group) + ' |')
        lines.append('')
    (root / 'comparison.md').write_text('\n'.join(lines), encoding='utf-8')
    for sweep in ('mu', 'tau'):
        fig, axes = plt.subplots(1, len(config['datasets']), figsize=(17, 3.6), squeeze=False)
        values = config['mu_values'] if sweep == 'mu' else config['tau_values']
        for ax, dataset in zip(axes[0], config['datasets']):
            group = [r for r in rows if r['dataset'] == dataset and r['sweep'] == sweep]
            ax.errorbar(values, [r['test_accuracy_mean_pct'] for r in group],
                        yerr=[r['test_accuracy_std_pct'] for r in group], marker='o', capsize=3)
            ax.set_title(NAMES[dataset]); ax.set_xlabel(sweep); ax.grid(alpha=.25)
        axes[0, 0].set_ylabel('Test accuracy (%)'); fig.suptitle(f'FCL+AUC-C sensitivity to {sweep}')
        fig.tight_layout()
        for ext in ('png', 'pdf'): fig.savefig(root / f'{sweep}_sensitivity.{ext}', dpi=200, bbox_inches='tight')
        plt.close(fig)
    print((root / 'comparison.md').read_text(), flush=True)


if __name__ == '__main__': main()
