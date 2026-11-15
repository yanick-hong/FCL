"""Independently audit a completed campaign without reusing test data for selection."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from paths import clip_cache, observed_labels_cache


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def audit(root):
    frozen = read_json(root / 'campaign.json')
    config = frozen['config']
    for name, sha in frozen['sources'].items():
        assert digest(ROOT / name) == sha, f'Source changed: {name}'
    total = 0
    records = {}
    dataset_records = []
    for dataset in config['datasets']:
        cp, op = clip_cache(dataset), observed_labels_cache(dataset)
        cache_hash, labels_hash = digest(cp), digest(op)
        cache = torch.load(cp, map_location='cpu', weights_only=True)
        base = torch.load(op, map_location='cpu', weights_only=True)
        truth = cache['train_labels'].long()
        corrected = base['s'].eq(0)
        assert torch.equal(base['y_obs'][corrected], truth[corrected])
        classes = cache['clip_probs_train'].shape[1]
        for seed in config['seeds']:
            previous = torch.zeros(len(truth), dtype=torch.bool)
            previous_labels = base['y_obs'].clone()
            for accuracy in sorted(config['accuracies'], reverse=True):
                name = f'acc_{round(100 * accuracy):03d}'
                run = root / dataset / name / f'seed_{seed}'
                inputs = root / 'inputs' / dataset / f'seed_{seed}' / (name + '.pt')
                scenario = torch.load(inputs, map_location='cpu', weights_only=True)
                observed = scenario['y_obs']
                changed = observed.ne(base['y_obs'])
                count = round((1 - accuracy) * int(corrected.sum()))
                assert torch.equal(scenario['s'], base['s'])
                assert not bool((changed & ~corrected).any())
                assert int(changed.sum()) == count
                assert not bool((previous & ~changed).any()), 'Errors not nested'
                assert torch.equal(observed[previous], previous_labels[previous])
                assert observed.min() >= 0 and observed.max() < classes
                assert torch.equal(changed.nonzero().flatten().sort().values,
                                   scenario['corrupted_indices'].sort().values)
                previous, previous_labels = changed, observed

                metrics = read_json(run / 'metrics.json')
                records[dataset, accuracy, seed] = metrics
                settings = read_json(run / 'config.json')
                metadata = read_json(run / 'run_metadata.json')
                assert metrics['status'] == metadata['status'] == 'completed'
                assert metrics['test_checkpoint'] == 'best.ckpt'
                assert metrics['cache_sha256'] == cache_hash
                assert metrics['base_labels_sha256'] == labels_hash
                assert metrics['scenario_sha256'] == digest(inputs)
                assert metrics['seed'] == seed and metrics['accuracy_requested'] == accuracy
                assert metrics['errors'] == count
                assert abs(metrics['accuracy_actual'] - (1 - count / int(corrected.sum()))) < 1e-12
                for key, value in config.items():
                    assert settings[key] == value, f'Config mismatch: {run}/{key}'
                with (run / 'history.csv').open(newline='', encoding='utf-8') as handle:
                    history = list(csv.DictReader(handle))
                assert len(history) == metrics['epochs_completed']
                assert [int(row['epoch']) for row in history] == list(range(1, len(history) + 1))
                best = max(history, key=lambda row: float(row['val_accuracy']))
                assert int(best['epoch']) == metrics['best_epoch']
                assert float(best['val_accuracy']) == metrics['best_val_accuracy']
                for filename in ('best.ckpt', 'last.ckpt'):
                    weights = torch.load(run / filename, map_location='cpu', weights_only=True)
                    assert weights['fc.weight'].shape == (classes, cache['train_feats'].shape[1])
                    assert all(bool(torch.isfinite(value).all()) for value in weights.values())
                total += 1
        dataset_records.append({'dataset': dataset, 'cache_sha256': cache_hash,
                                'base_labels_sha256': labels_hash,
                                'corrected_count': int(corrected.sum()),
                                'runs_checked': len(config['accuracies']) * len(config['seeds'])})
    expected = len(config['datasets']) * len(config['accuracies']) * len(config['seeds'])
    assert total == expected
    for filename, rows_expected in [('summary.csv', expected // len(config['seeds'])),
                                    ('runs.csv', expected), ('paired_drops.csv', expected)]:
        with (root / filename).open(newline='', encoding='utf-8') as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == rows_expected
        if filename == 'summary.csv':
            assert all(int(row['n']) == len(config['seeds']) for row in rows)
            seen = set()
            for row in rows:
                dataset = row['dataset']
                accuracy = float(row['human_accuracy_requested_pct']) / 100
                assert (dataset, accuracy) not in seen
                seen.add((dataset, accuracy))
                values = [100 * records[dataset, accuracy, seed]['test_accuracy']
                          for seed in config['seeds']]
                drops = [100 * (records[dataset, 1., seed]['test_accuracy'] -
                                records[dataset, accuracy, seed]['test_accuracy'])
                         for seed in config['seeds']]
                expected_values = {'test_accuracy_mean_pct': statistics.mean(values),
                                   'test_accuracy_std_pct': statistics.stdev(values),
                                   'paired_drop_mean_pp': statistics.mean(drops),
                                   'paired_drop_std_pp': statistics.stdev(drops)}
                for key, value in expected_values.items():
                    assert abs(float(row[key]) - value) < 1e-9
    status = read_json(root / 'status.json')
    assert status['status'] == 'completed' and status['completed'] == total and not status['failed']
    result = {'status': 'passed', 'runs_checked': total, 'datasets': dataset_records,
              'checks': ['frozen source hashes', 'unchanged source caches', 'nested label errors',
                         'uncorrected labels unchanged', 'exact error counts', 'fixed run configs',
                         'history/best-epoch consistency', 'finite checkpoint shapes',
                         'summary completeness and independent aggregation'],
              'limitations': 'Does not independently rerun training or recompute test predictions.'}
    (root / 'audit.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    audit(args.campaign.resolve())
