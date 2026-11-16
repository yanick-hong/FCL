"""Independently audit the frozen Conf/FCL CE study; no study files are changed.

The default audit uses only Python's standard library. --checkpoints additionally
uses torch, loads the final checkpoints on CPU and recomputes validation/test
accuracy. Write reports outside the study with --output, or use stdout only.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path, PurePosixPath
import statistics
import subprocess
import sys


METHODS = ('clip_conf_ce', 'fcl_filter_ce')
DATASETS = ('cifar100', 'caltech101', 'eurosat', 'food101', 'tinyimagenet200')
REQUIRED_FILES = ('metrics.json', 'launch.json', 'console.log', 'best.ckpt',
                  'last.ckpt', 'config.yaml', 'run_metadata.json', 'logs/val_metrics.csv')
T95_DF2 = 4.302652729911275


class AuditError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AuditError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def finite_accuracy(value):
    return type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1


def close(actual, expected, context, tolerance=1e-10):
    require(type(actual) in (int, float) and math.isfinite(actual), context + ': invalid number')
    require(math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance),
            f'{context}: {actual!r} != {expected!r}')


def recorded_join(root, *parts):
    return str(PurePosixPath(root).joinpath(*parts)) if str(root).startswith('/') else str(Path(root).joinpath(*parts))


def identity(row):
    return row['dataset'], row['method'], row['learning_rate'], row['seed']


def file_manifest(run):
    return {str(path.relative_to(run)).replace('\\', '/'): sha(path)
            for path in sorted(run.rglob('*'))
            if path.is_file() and path.name not in ('seal.json', 'seal.json.tmp')}


def expected_command(protocol, dataset, method, lr, seed, run_reference, audit):
    return [protocol['python'], recorded_join(protocol['root'], 'src/contrast/train_revision_baselines.py'),
            '--method', method, '--dataset', dataset, '--cache', audit['cache'],
            '--obs-labels', audit['observed_labels'], '--run-dir', run_reference,
            '--epochs', str(protocol['epochs']), '--patience', str(protocol['patience']),
            '--batch-size', str(protocol['batch_size']), '--lr', str(lr),
            '--weight-decay', str(protocol['weight_decay']), '--seed', str(seed)]


def audit_seal(run, study_root, recorded_root, plan, audits, binding):
    """Rebuild a result from hashed metrics instead of trusting seal.result."""
    relative = run.relative_to(study_root / 'trials')
    require(len(relative.parts) == 4, f'Unexpected trial path: {relative}')
    dataset, method, lr_part, seed_part = relative.parts
    require(lr_part.startswith('lr-') and seed_part.startswith('seed-'), f'Invalid trial path: {relative}')
    lr, seed = float(lr_part[3:]), int(seed_part[5:])
    protocol = plan['protocol']
    require(dataset in protocol['datasets'] and method in METHODS, f'Unplanned condition: {relative}')
    require(lr in protocol['learning_rates'], f'Unplanned learning rate: {relative}')
    require(seed in [protocol['tuning_seed'], *protocol['additional_seeds']], f'Unplanned seed: {relative}')
    audit = audits[dataset]
    expected_binding = {'study_binding': binding, 'dataset': dataset, 'method': method,
                        'learning_rate': lr, 'seed': seed, 'input_hashes': audit['hashes']}
    seal = read(run / 'seal.json')
    require(seal['binding'] == expected_binding, f'Trial binding mismatch: {relative}')
    require(all((run / name).is_file() for name in REQUIRED_FILES), f'Missing trial artifact: {relative}')
    require(seal['files'] == file_manifest(run), f'Changed/missing/extra sealed files: {relative}')
    payload = read(run / 'metrics.json')
    launch, metadata = read(run / 'launch.json'), read(run / 'run_metadata.json')
    run_reference = recorded_join(recorded_root, 'trials', *relative.parts)
    require(launch['binding'] == expected_binding, f'Launch binding mismatch: {relative}')
    require(launch['command'] == expected_command(protocol, dataset, method, lr, seed, run_reference, audit),
            f'Launch command differs from the plan: {relative}')
    require(metadata['command'] == launch['command'][1:], f'Executed command differs from launch: {relative}')
    require(metadata['script'] == 'src/contrast/train_revision_baselines.py', f'Unexpected learner script: {relative}')
    require(metadata['git_commit'] == plan['source']['git_commit'], f'Learner Git revision differs: {relative}')
    expected_args = {'method': method, 'dataset': dataset, 'cache': audit['cache'],
                     'obs_labels': audit['observed_labels'], 'run_dir': run_reference,
                     'epochs': protocol['epochs'], 'patience': protocol['patience'],
                     'batch_size': protocol['batch_size'], 'lr': lr,
                     'weight_decay': protocol['weight_decay'], 'seed': seed}
    args = payload['selection']['args']
    require(all(args.get(key) == value for key, value in expected_args.items()), f'Metrics settings mismatch: {relative}')
    require(metadata['args'] == args, f'Run metadata and metrics args differ: {relative}')
    require(metadata['method'] == method and metadata['dataset'] == dataset, f'Run metadata identity mismatch: {relative}')
    require(payload['method'] == method and payload['dataset'] == dataset, f'Metrics identity mismatch: {relative}')
    require(payload['verified_count'] == audit['verified_count'], f'Correction count mismatch: {relative}')
    require(payload['selection']['budget_verified_count'] == audit['verified_count'], f'Selection budget mismatch: {relative}')
    if method == 'clip_conf_ce':
        require(payload['selection']['selected_count'] == audit['verified_count'], f'Conf count mismatch: {relative}')
    close(payload['rho'], audit['rho'], f'{relative}: correction fraction')
    require(payload['epochs_completed'] == protocol['epochs'], f'Incomplete epoch budget: {relative}')
    require(payload['test']['valid_count'] == audit['split_counts']['test'], f'Test valid count mismatch: {relative}')
    require(payload['test']['total_count'] == audit['split_counts']['test'], f'Test total count mismatch: {relative}')
    with (run / 'logs/val_metrics.csv').open(newline='', encoding='utf-8') as handle:
        history = list(csv.DictReader(handle))
    require([int(row['epoch']) for row in history] == list(range(1, protocol['epochs'] + 1)),
            f'Nonconsecutive or incomplete validation history: {relative}')
    values = [float(row['accuracy']) for row in history]
    require(all(finite_accuracy(value) for value in values), f'Invalid validation history: {relative}')
    first_max = values.index(max(values)) + 1
    require(payload['best_epoch'] == first_max, f'Checkpoint is not the first validation maximum: {relative}')
    close(payload['best_val_accuracy'], max(values), f'{relative}: validation maximum', 1e-12)
    require(finite_accuracy(payload['test']['accuracy']), f'Invalid test accuracy: {relative}')
    rebuilt = {'dataset': dataset, 'method': method, 'learning_rate': lr, 'seed': seed,
               'best_val_accuracy': payload['best_val_accuracy'], 'test_accuracy': payload['test']['accuracy'],
               'best_epoch': payload['best_epoch'], 'run_dir': run_reference,
               'metrics_sha256': sha(run / 'metrics.json')}
    require(seal['result'] == rebuilt, f'Seal result differs from independently rebuilt metrics: {relative}')
    return rebuilt


def expected_summary(rows):
    rows = sorted(rows, key=lambda row: row['seed'])
    values = [row['test_accuracy'] * 100 for row in rows]
    mean, sd = statistics.mean(values), statistics.stdev(values)
    half = T95_DF2 * sd / math.sqrt(3)
    return {'learning_rate': rows[0]['learning_rate'], 'seeds': [row['seed'] for row in rows],
            'test_accuracies_percent': values, 'mean_percent': mean, 'sample_sd_percent': sd,
            'ci95_low_percent': mean - half, 'ci95_high_percent': mean + half, 'n': 3}


def audit_references(results, sealed, protocol, allow_partial):
    require(results['selection_rule'] == protocol['selection'], 'Summary selection rule differs from the plan')
    all_rows, final_rows = results['all_trials'], results['final_runs']
    all_ids, final_ids = [identity(row) for row in all_rows], [identity(row) for row in final_rows]
    require(len(set(all_ids)) == len(all_ids), 'Duplicate all_trials identity')
    require(len(set(final_ids)) == len(final_ids), 'Duplicate final_runs identity')
    for kind, rows in [('all_trials', all_rows), ('final_runs', final_rows)]:
        for row in rows:
            key = identity(row)
            require(key in sealed and row == sealed[key], f'{kind} references an absent or mismatched sealed result: {key}')
    require(set(final_ids).issubset(all_ids), 'A final run is absent from all_trials')
    if not allow_partial:
        require(set(all_ids) == set(sealed), 'Not every sealed result is referenced in all_trials')
    selections = {(row['dataset'], row['method']): row for row in results['selected']}
    require(len(selections) == len(results['selected']), 'Duplicate selected condition')
    expected_pairs = {(dataset, method) for dataset in protocol['datasets'] for method in METHODS}
    require(set(selections).issubset(expected_pairs), 'Unplanned selected condition')
    computed_selections = {}
    for pair in expected_pairs:
        candidates = [sealed.get((*pair, lr, protocol['tuning_seed'])) for lr in protocol['learning_rates']]
        if all(candidates):
            selected = min(candidates, key=lambda row: (-row['best_val_accuracy'], row['learning_rate']))
            computed_selections[pair] = selected
        if pair in selections:
            require(pair in computed_selections, f'LR selected before all candidates were sealed: {pair}')
            actual, expected = selections[pair], computed_selections[pair]
            require(actual['learning_rate'] == expected['learning_rate'], f'LR was not selected by validation/tie-break: {pair}')
            close(actual['best_val_accuracy'], expected['best_val_accuracy'], f'{pair}: selected validation score')
            require(actual['criterion'] == protocol['selection'], f'Unexpected LR selection criterion: {pair}')
    for row in final_rows:
        pair = row['dataset'], row['method']
        require(pair in selections, f'Final run without a selected condition: {pair}')
        require(row['learning_rate'] == computed_selections[pair]['learning_rate'], f'Final run uses an unselected LR: {pair}')
    for row in sealed.values():
        if row['seed'] != protocol['tuning_seed']:
            pair = row['dataset'], row['method']
            require(pair in computed_selections and row['learning_rate'] == computed_selections[pair]['learning_rate'],
                    f'Confirmation run uses an unselected LR: {pair}')
    summaries = {(row['dataset'], row['method']): row for row in results['summary']}
    require(len(summaries) == len(results['summary']), 'Duplicate summary condition')
    expected_summary_pairs = set()
    for pair in expected_pairs:
        rows = [row for row in final_rows if (row['dataset'], row['method']) == pair]
        require(len(rows) <= 3, f'Too many final runs: {pair}')
        if len(rows) != 3:
            continue
        expected_summary_pairs.add(pair)
        require(sorted(row['seed'] for row in rows) == sorted([protocol['tuning_seed'], *protocol['additional_seeds']]),
                f'Incorrect final seeds: {pair}')
        require(pair in summaries, f'Missing complete three-seed summary: {pair}')
        expected, actual = expected_summary(rows), summaries[pair]
        for key in ('learning_rate', 'seeds', 'n'):
            require(actual[key] == expected[key], f'{pair}: incorrect {key}')
        require(len(actual['test_accuracies_percent']) == 3, f'{pair}: incorrect value count')
        for actual_value, expected_value in zip(actual['test_accuracies_percent'], expected['test_accuracies_percent']):
            close(actual_value, expected_value, f'{pair}: individual seed accuracy')
        for key in ('mean_percent', 'sample_sd_percent', 'ci95_low_percent', 'ci95_high_percent'):
            close(actual[key], expected[key], f'{pair}: incorrect {key}')
    require(set(summaries) == expected_summary_pairs, 'Summary exists for an incomplete or unplanned condition')
    if not allow_partial:
        require(set(selections) == expected_pairs, 'Not every dataset/method has an LR selection')
        require(len(final_rows) == len(expected_pairs) * 3, 'Incomplete final-run count')
        require(set(summaries) == expected_pairs, 'Incomplete three-seed summary count')
    return final_rows, {'referenced_trials': len(all_rows), 'final_runs': len(final_rows),
                        'selected_conditions': len(selections), 'summaries': len(summaries),
                        'sealed_not_yet_referenced': len(set(sealed) - set(all_ids))}


def verify_environment(plan, audits, verify_inputs, verify_source, runner):
    checked = {'inputs': False, 'source': False, 'wrapper': False}
    if verify_inputs:
        for dataset, audit in audits.items():
            for path_key, hash_key in [('cache', 'cache_sha256'), ('observed_labels', 'obs_sha256')]:
                require(sha(audit[path_key]) == audit['hashes'][hash_key], f'Actual input SHA changed: {dataset}/{path_key}')
        checked['inputs'] = True
    if verify_source:
        root = Path(plan['protocol']['root'])
        actual_git = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        require(actual_git == plan['source']['git_commit'], 'Actual Git revision changed')
        paths = [path for folder in ('src', 'scripts', 'configs')
                 for path in (root / folder).rglob('*')
                 if path.is_file() and path.suffix in ('.py', '.yaml', '.yml', '.json')]
        current = {str(path.relative_to(root)).replace('\\', '/'): sha(path) for path in sorted(paths)}
        require(current == plan['source']['source_sha256'], 'Actual source file set or SHA changed')
        require(runner is not None, '--verify-source also requires --runner to verify the frozen wrapper')
        require(sha(runner) == plan['source']['wrapper_sha256'], 'Actual wrapper SHA changed')
        checked['source'], checked['wrapper'] = True, True
    return checked


def audit_checkpoints(finals, audits, protocol, study_root, recorded_root):
    """Import torch only on explicit request; reproduce legacy CPU evaluation."""
    import torch
    import torch.nn.functional as functional
    torch.set_num_threads(protocol['threads'])
    checks = []
    for dataset in protocol['datasets']:
        rows = [row for row in finals if row['dataset'] == dataset]
        if not rows:
            continue
        bundle = torch.load(audits[dataset]['cache'], map_location='cpu', weights_only=True)
        features = {split: functional.normalize(bundle[f'{split}_feats'].float(), dim=-1) for split in ('val', 'test')}
        labels = {split: bundle[f'{split}_labels'].long() for split in ('val', 'test')}
        del bundle
        for row in rows:
            relative = PurePosixPath(row['run_dir']).relative_to(PurePosixPath(str(recorded_root)))
            run = study_root.joinpath(*relative.parts)
            checkpoint = torch.load(run / 'best.ckpt', map_location='cpu', weights_only=True)
            last = torch.load(run / 'last.ckpt', map_location='cpu', weights_only=True)
            require(checkpoint['epoch'] == row['best_epoch'], f'Best checkpoint epoch mismatch: {row["run_dir"]}')
            require(last['epoch'] == protocol['epochs'], f'Last checkpoint is incomplete: {row["run_dir"]}')
            state = checkpoint['state_dict']
            require(set(state) == {'fc.weight', 'fc.bias'}, f'Unexpected linear checkpoint parameters: {row["run_dir"]}')
            weight, bias = state['fc.weight'].float(), state['fc.bias'].float()
            require(weight.shape[0] == audits[dataset]['classes'] and bias.shape == (weight.shape[0],), 'Checkpoint class count mismatch')
            require(torch.isfinite(weight).all() and torch.isfinite(bias).all(), 'Nonfinite checkpoint weights')
            evaluated = {}
            for split in ('val', 'test'):
                logits = torch.cat([functional.linear(features[split][start:start + 4096], weight, bias)
                                    for start in range(0, len(labels[split]), 4096)])
                require(torch.isfinite(logits).all(), 'Nonfinite recomputed logits')
                require(((labels[split] >= 0) & (labels[split] < weight.shape[0])).all(), 'Invalid evaluation label')
                predictions = logits.float().softmax(-1).argmax(-1)
                evaluated[split] = float(predictions.eq(labels[split]).float().mean())
            close(evaluated['val'], row['best_val_accuracy'], f'{row["run_dir"]}: recomputed validation accuracy', 1e-8)
            close(evaluated['test'], row['test_accuracy'], f'{row["run_dir"]}: recomputed test accuracy', 1e-8)
            checks.append({'dataset': dataset, 'method': row['method'], 'seed': row['seed'],
                           'best_epoch': row['best_epoch'], 'recomputed_val_accuracy': evaluated['val'],
                           'recomputed_test_accuracy': evaluated['test'], 'run_dir': row['run_dir']})
        del features, labels
    return {'device': 'cpu', 'torch': torch.__version__, 'count': len(checks), 'runs': checks}


def audit_study(study_root, *, allow_partial=False, recorded_root=None,
                verify_inputs=False, verify_source=False, runner=None, checkpoints=False):
    study_root = Path(study_root).resolve()
    recorded_root = str(recorded_root or study_root)
    plan, preflight = read(study_root / 'plan.json'), read(study_root / 'preflight.json')
    protocol, audits = plan['protocol'], preflight['datasets']
    require(protocol['methods'] == list(METHODS), 'Unexpected method list')
    require(protocol['learning_rates'] == [.001, .002, .004], 'Unexpected LR grid')
    require(protocol['tuning_seed'] == 42 and protocol['additional_seeds'] == [43, 44], 'Unexpected tuning/final seeds')
    for key, expected in {'epochs': 100, 'patience': 101, 'batch_size': 1024,
                          'weight_decay': .0001, 'data_loader_workers': 2,
                          'scheduler': 'CosineAnnealingLR(T_max=100)'}.items():
        require(protocol.get(key) == expected, f'Unexpected fixed protocol setting: {key}')
    require(bool(protocol['datasets']) and set(protocol['datasets']).issubset(DATASETS), 'Unplanned dataset list')
    require(len(set(protocol['datasets'])) == len(protocol['datasets']) and set(audits) == set(protocol['datasets']), 'Dataset/preflight mismatch')
    expected_count = len(protocol['datasets']) * len(METHODS) * (len(protocol['learning_rates']) + len(protocol['additional_seeds']))
    require(plan['training_count'] == expected_count, 'Incorrect planned training count')
    binding_file = read(study_root / 'study_binding.json')
    binding = fingerprint({'plan': plan, 'inputs': audits, 'runtime': preflight['runtime']})
    require(binding_file['binding'] == binding and binding_file['runtime'] == preflight['runtime'], 'Study binding/preflight/runtime mismatch')
    environment = verify_environment(plan, audits, verify_inputs, verify_source, runner)
    sealed = {}
    for seal_path in sorted((study_root / 'trials').rglob('seal.json')):
        rebuilt = audit_seal(seal_path.parent, study_root, recorded_root, plan, audits, binding)
        require(identity(rebuilt) not in sealed, 'Duplicate sealed trial identity')
        sealed[identity(rebuilt)] = rebuilt
    require(len(sealed) <= expected_count, 'More sealed trials than planned')
    result_path = study_root / 'results.json'
    if result_path.exists():
        finals, counts = audit_references(read(result_path), sealed, protocol, allow_partial)
    else:
        require(allow_partial, 'Missing results.json')
        finals, counts = [], {'referenced_trials': 0, 'final_runs': 0, 'selected_conditions': 0,
                              'summaries': 0, 'sealed_not_yet_referenced': len(sealed)}
    if not allow_partial:
        require(len(sealed) == expected_count, f'Incomplete sealed count: {len(sealed)}/{expected_count}')
        status = read(study_root / 'status.json')
        require(status['status'] == 'completed' and status['binding'] == binding, 'Worker status is not completed under this binding')
        require(status['completed_trials'] == expected_count, 'Worker completed count is inconsistent')
        if 'training_count' in status:
            require(status['training_count'] == expected_count, 'Worker planned count is inconsistent')
    checkpoint_report = audit_checkpoints(finals, audits, protocol, study_root, recorded_root) if checkpoints else None
    return {'status': 'passed' if not allow_partial else 'partial_passed',
            'allow_partial': allow_partial, 'study_root': str(study_root), 'binding': binding,
            'expected_trials': expected_count, 'sealed_trials': len(sealed), **counts,
            'environment_checks': environment, 'checkpoint_checks': checkpoint_report,
            'statistical_limit': 'Seed 42 also selected the LR; seed 42/43/44 CI is exploratory and conditional on that selection.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study_root', type=Path)
    parser.add_argument('--allow-partial', action='store_true')
    parser.add_argument('--recorded-study-root', help='Original study root when auditing a downloaded snapshot')
    parser.add_argument('--verify-inputs', action='store_true')
    parser.add_argument('--verify-source', action='store_true')
    parser.add_argument('--runner', type=Path)
    parser.add_argument('--checkpoints', action='store_true')
    parser.add_argument('--output', type=Path, help='Optional new report outside the study root')
    args = parser.parse_args(argv)
    if args.output:
        output = args.output.resolve()
        require(not output.is_relative_to(args.study_root.resolve()), 'Audit reports must be outside the study directory')
        require(not output.exists(), 'Refusing to overwrite an existing audit report')
    try:
        report = audit_study(args.study_root, allow_partial=args.allow_partial, recorded_root=args.recorded_study_root,
                             verify_inputs=args.verify_inputs or args.checkpoints,
                             verify_source=args.verify_source, runner=args.runner, checkpoints=args.checkpoints)
        exit_code = 0
    except (AuditError, OSError, ValueError, KeyError, TypeError) as error:
        report, exit_code = {'status': 'failed', 'error': str(error)}, 1
    serialized = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + '\n', encoding='utf-8')
    print(serialized)
    return exit_code


if __name__ == '__main__':
    raise SystemExit(main())
