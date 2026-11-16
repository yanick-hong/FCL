"""Finite, transparent Conf-CE/FCL-CE validation-selected learning-rate study.

Default is plan-only. This wrapper never changes project sources or a manuscript.
The test score is recorded for every trial but is never an LR selection input.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

DATASETS = ('cifar100', 'caltech101', 'eurosat', 'food101', 'tinyimagenet200')
METHODS = ('clip_conf_ce', 'fcl_filter_ce')
DEFAULT = {
    'datasets': list(DATASETS), 'methods': list(METHODS),
    'learning_rates': [.001, .002, .004], 'weight_decay': .0001,
    'epochs': 100, 'patience': 101, 'batch_size': 1024,
    'tuning_seed': 42, 'additional_seeds': [43, 44],
    'selection': 'highest best_val_accuracy; ties choose smallest learning rate',
    'data_loader_workers': 2, 'scheduler': 'CosineAnnealingLR(T_max=100)',
    'notes': 'Separate reruns; do not replace historical manuscript entries or claim matching historical protocols.'
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def source_manifest(root):
    paths = []
    for folder in ('src', 'scripts', 'configs'):
        paths.extend(p for p in (root / folder).rglob('*') if p.is_file() and p.suffix in ('.py', '.yaml', '.yml', '.json'))
    return {
        'git_commit': subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip(),
        'source_sha256': {str(p.relative_to(root)).replace('\\', '/'): sha(p) for p in sorted(paths)},
        'wrapper_sha256': sha(__file__),
    }


def select_trial(trials):
    """Only validation and a preregistered LR tie-break participate in selection."""
    if not trials:
        raise ValueError('Cannot select from an empty trial set')
    for row in trials:
        value = row['best_val_accuracy']
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid validation score')
    return min(trials, key=lambda row: (-row['best_val_accuracy'], row['learning_rate']))


def files_manifest(run):
    return {str(p.relative_to(run)).replace('\\', '/'): sha(p) for p in sorted(run.rglob('*'))
            if p.is_file() and p.name not in ('seal.json', 'seal.json.tmp')}


def verify_seal(run, expected):
    sealed = read(run / 'seal.json')
    if sealed.get('binding') != expected or sealed.get('files') != files_manifest(run):
        raise ValueError(f'Refusing changed or mismatched completed trial: {run}')
    return sealed['result']


def paths_for(args, dataset):
    return (args.root / 'dataset/processed' / f'{dataset}_clip_vit-l-14_embeddings.pt',
            args.core_root / 'corrections' / dataset / 'clip/prototype.pt')


def preflight(args, dataset):
    import torch
    sys.path.insert(0, str(args.root))
    sys.path.insert(0, str(args.root / 'src'))
    from scripts import run_controlled_training as controlled
    from scripts import compare_label_corrections as correction
    from contrast.train_revision_baselines import get_clip_prior
    cache_path, obs_path = paths_for(args, dataset)
    bundle, observed, hashes = controlled.load_fixed_inputs(argparse.Namespace(
        cache=cache_path, obs_labels=obs_path, dataset=dataset))
    if observed['meta']['method'] != 'prototype' or observed['meta']['label_source'] != 'clip':
        raise ValueError('Require the original CLIP prototype correction artifact')
    classes = bundle['clip_probs_train'].shape[1]
    for split in ('train', 'val', 'test'):
        prior = bundle.get(f'clip_probs_{split}')
        labels = bundle[f'{split}_labels']
        if (not isinstance(prior, torch.Tensor) or prior.shape != (len(labels), classes)
                or not torch.isfinite(prior).all() or (prior < 0).any()
                or not torch.allclose(prior.sum(1), torch.ones(len(labels)), atol=1e-4, rtol=1e-4)):
            raise ValueError(f'Require real normalized CLIP probabilities in {split}; fallback prohibited')
        if not correction._has_source(bundle, 'clip', split):
            raise ValueError(f'{split} probabilities are not proven to be CLIP')
    prior = get_clip_prior(bundle, 'train', observed['y_obs'], classes)
    pseudo = prior.argmax(1)
    if not torch.equal(pseudo, bundle['clip_probs_train'].argmax(1)):
        raise ValueError('CLIP normalization unexpectedly changed pseudo-labels')
    selected = observed['s'].eq(0)
    count = int(selected.sum())
    if not 0 < count < len(selected):
        raise ValueError('Correction count must be nonzero and smaller than the dataset')
    confidence_mask = torch.zeros(len(selected), dtype=torch.bool)
    confidence_mask[torch.topk(prior.max(1).values, count, largest=False).indices] = True
    if int(confidence_mask.sum()) != count:
        raise ValueError('Confidence and prototype correction budgets differ')
    truth = bundle['train_labels'].long()
    label_quality = {}
    for name, mask in [('clip_conf_ce', confidence_mask), ('fcl_filter_ce', selected)]:
        targets = pseudo.clone()
        targets[mask] = truth[mask]
        if name == 'fcl_filter_ce' and not torch.equal(targets, observed['y_obs']):
            raise ValueError('FCL-CE labels are not CLIP with only selected corrections')
        label_quality[name] = {
            'mask_sha256': correction.tensor_sha256(mask),
            'targets_sha256': correction.tensor_sha256(targets),
            'corrected_label_accuracy': float(targets.eq(truth).double().mean()),
            'selected_source_error_rate': float(pseudo[mask].ne(truth[mask]).double().mean()),
        }
    return {'dataset': dataset, 'cache': str(cache_path), 'observed_labels': str(obs_path),
            'classes': classes, 'split_counts': {s: len(bundle[f'{s}_labels']) for s in ('train', 'val', 'test')},
            'hashes': hashes, 'verified_count': count, 'rho': count / len(selected),
            'selection_audit': label_quality,
            'historical_protocol_note': 'Caltech recovered cache has 101 classes; historical main text uses 102.' if dataset == 'caltech101' else 'New controlled rerun; historical cache equivalence is not assumed.'}


def child_env(args):
    env = {**os.environ, 'OMP_NUM_THREADS': str(args.threads), 'MKL_NUM_THREADS': str(args.threads),
           'PYTHONUTF8': '1', 'PYTHONIOENCODING': 'utf-8', 'TOKENIZERS_PARALLELISM': 'false'}
    for name in ('OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        env[name] = str(args.threads)
    if args.device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''
    return env


def command(args, dataset, method, lr, seed, run):
    cache, observed = paths_for(args, dataset)
    return [str(args.python), str(args.root / 'src/contrast/train_revision_baselines.py'),
            '--method', method, '--dataset', dataset, '--cache', str(cache), '--obs-labels', str(observed),
            '--run-dir', str(run), '--epochs', str(DEFAULT['epochs']), '--patience', str(DEFAULT['patience']),
            '--batch-size', str(DEFAULT['batch_size']), '--lr', str(lr),
            '--weight-decay', str(DEFAULT['weight_decay']), '--seed', str(seed)]


def run_trial(args, dataset, method, lr, seed, binding, integrity):
    run = args.output_root / 'trials' / dataset / method / f'lr-{lr:g}' / f'seed-{seed}'
    trial_binding = {'study_binding': binding, 'dataset': dataset, 'method': method,
                     'learning_rate': lr, 'seed': seed, 'input_hashes': integrity['hashes']}
    if run.exists() and any(run.iterdir()):
        if not args.resume:
            raise FileExistsError(f'Existing trial; use --resume to verify a completed seal: {run}')
        return verify_seal(run, trial_binding)
    run.mkdir(parents=True, exist_ok=True)
    cmd = command(args, dataset, method, lr, seed, run)
    save(run / 'launch.json', {'command': cmd, 'binding': trial_binding, 'started_at': utc(),
                             'threads': args.threads, 'device_request': args.device,
                             'data_loader_workers': 2, 'source_algorithm_modified': False})
    with (run / 'console.log').open('w', encoding='utf-8') as log:
        completed = subprocess.run(cmd, cwd=args.root, env=child_env(args), stdout=log, stderr=subprocess.STDOUT,
                                   timeout=args.timeout_hours * 3600)
    if completed.returncode:
        raise RuntimeError(f'Training failed ({completed.returncode}); inspect {run / "console.log"}')
    payload = read(run / 'metrics.json')
    required_outputs = ('best.ckpt', 'last.ckpt', 'config.yaml', 'run_metadata.json', 'logs/val_metrics.csv')
    if any(not (run / name).is_file() for name in required_outputs):
        raise ValueError(f'Training did not produce every required artifact: {run}')
    with (run / 'logs/val_metrics.csv').open(newline='', encoding='utf-8') as handle:
        history = list(csv.DictReader(handle))
    if [int(row['epoch']) for row in history] != list(range(1, DEFAULT['epochs'] + 1)):
        raise ValueError(f'History does not contain the complete consecutive epoch budget: {run}')
    history_values = [float(row['accuracy']) for row in history]
    history_best = max(history_values)
    if (not math.isclose(history_best, payload['best_val_accuracy'], abs_tol=1e-12)
            or history_values.index(history_best) + 1 != payload['best_epoch']):
        raise ValueError(f'Checkpoint selection does not match validation history: {run}')
    saved_args = payload['selection']['args']
    expected_args = {'method': method, 'dataset': dataset, 'cache': integrity['cache'],
                     'obs_labels': integrity['observed_labels'], 'run_dir': str(run),
                     'epochs': DEFAULT['epochs'], 'patience': DEFAULT['patience'],
                     'batch_size': DEFAULT['batch_size'], 'lr': lr, 'weight_decay': DEFAULT['weight_decay'], 'seed': seed}
    if any(saved_args.get(k) != v for k, v in expected_args.items()):
        raise ValueError(f'Executed settings differ from the planned trial: {run}')
    if (payload['method'] != method or payload['dataset'] != dataset
            or payload['verified_count'] != integrity['verified_count']
            or payload['epochs_completed'] != DEFAULT['epochs']
            or payload['test']['valid_count'] != integrity['split_counts']['test']):
        raise ValueError(f'Incomplete trial or correction-budget mismatch: {run}')
    val, test = payload['best_val_accuracy'], payload['test']['accuracy']
    if any(not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in (val, test)):
        raise ValueError(f'Invalid result accuracy: {run}')
    result = {'dataset': dataset, 'method': method, 'learning_rate': lr, 'seed': seed,
              'best_val_accuracy': val, 'test_accuracy': test, 'best_epoch': payload['best_epoch'],
              'run_dir': str(run), 'metrics_sha256': sha(run / 'metrics.json')}
    save(run / 'seal.json', {'binding': trial_binding, 'result': result, 'files': files_manifest(run), 'sealed_at': utc()})
    return result


def write_results(args, all_trials, finals, selections):
    summary = []
    for dataset in args.datasets:
        for method in METHODS:
            rows = sorted([r for r in finals if r['dataset'] == dataset and r['method'] == method], key=lambda r: r['seed'])
            if len(rows) != 3:
                continue
            values = [r['test_accuracy'] * 100 for r in rows]
            mean, sd = statistics.mean(values), statistics.stdev(values)
            half = 4.302652729911275 * sd / math.sqrt(3)
            summary.append({'dataset': dataset, 'method': method, 'learning_rate': rows[0]['learning_rate'],
                            'seeds': [r['seed'] for r in rows], 'test_accuracies_percent': values,
                            'mean_percent': mean, 'sample_sd_percent': sd,
                            'ci95_low_percent': mean - half, 'ci95_high_percent': mean + half,
                            'n': 3, 'uncertainty_scope': 'training seeds only; fixed data, labels, correction masks and selected LR'})
    save(args.output_root / 'results.json', {'selection_rule': DEFAULT['selection'], 'selected': selections,
                                            'all_trials': all_trials, 'final_runs': finals, 'summary': summary})
    if all_trials:
        with (args.output_root / 'all_trials.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(all_trials[0]))
            writer.writeheader()
            writer.writerows(all_trials)
    if summary:
        with (args.output_root / 'summary.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/root/autodl-tmp'))
    parser.add_argument('--python', type=Path, default=Path(sys.executable))
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--core-root', type=Path)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--jobs', type=int, default=1, help='Parallel datasets; each dataset remains sequential')
    parser.add_argument('--timeout-hours', type=float, default=12)
    parser.add_argument('--resume', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--execute', action='store_true')
    mode.add_argument('--preflight', action='store_true')
    mode.add_argument('--plan-only', action='store_true')
    args = parser.parse_args(argv)
    args.root, args.python = args.root.resolve(), args.python.resolve()
    args.core_root = (args.core_root or args.root / 'outputs/controlled_correction_20261115').resolve()
    args.output_root = (args.output_root or args.root / 'outputs/conf_ce_fair_recheck_20261116').resolve()
    if len(set(args.datasets)) != len(args.datasets) or args.threads < 1 or args.jobs < 1 or not math.isfinite(args.timeout_hours) or args.timeout_hours <= 0:
        raise ValueError('Invalid datasets/thread/timeout arguments')
    args.output_root.mkdir(parents=True, exist_ok=True)
    fixed_source = source_manifest(args.root)
    protocol = {**DEFAULT, 'datasets': args.datasets, 'device': args.device, 'threads': args.threads, 'parallel_datasets': args.jobs,
                'python': str(args.python), 'root': str(args.root), 'core_root': str(args.core_root)}
    plan = {'protocol': protocol, 'source': fixed_source,
            'training_count': len(args.datasets) * len(METHODS) * (len(DEFAULT['learning_rates']) + len(DEFAULT['additional_seeds'])),
            'inputs': {d: list(map(str, paths_for(args, d))) for d in args.datasets}}
    plan_path = args.output_root / 'plan.json'
    if plan_path.exists() and read(plan_path) != plan:
        raise ValueError('Existing study plan differs; use a distinct output-root rather than overwrite')
    save(plan_path, plan)
    if not args.execute and not args.preflight:
        print(json.dumps({'mode': 'plan-only', 'plan': str(plan_path), 'training_count': plan['training_count']}))
        return
    if args.device == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    os.environ['MKL_NUM_THREADS'] = str(args.threads)
    for name in ('OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
        os.environ[name] = str(args.threads)
    import torch
    torch.set_num_threads(args.threads)
    if args.execute and args.device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA explicitly requested but unavailable; refusing silent CPU fallback')
    runtime = {'python': sys.version, 'platform': platform.platform(), 'torch': torch.__version__,
               'cuda_runtime': torch.version.cuda, 'device': args.device,
               'gpu_name': torch.cuda.get_device_name(0) if args.execute and args.device == 'cuda' else None}
    audits = {d: preflight(args, d) for d in args.datasets}
    save(args.output_root / 'preflight.json', {'datasets': audits, 'runtime': runtime})
    if args.preflight:
        print(json.dumps({'mode': 'preflight', 'status': 'passed', 'datasets': args.datasets}))
        return
    binding = fingerprint({'plan': plan, 'inputs': audits, 'runtime': runtime})
    binding_path = args.output_root / 'study_binding.json'
    if binding_path.exists() and read(binding_path)['binding'] != binding:
        raise ValueError('Execution inputs/runtime differ from the sealed study binding')
    save(binding_path, {'binding': binding, 'runtime': runtime})
    # A POSIX advisory lock is released automatically after crashes, enabling safe resume.
    import fcntl
    with (args.output_root / 'worker.lock').open('a+') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('This study already has an active worker') from exc
        all_trials, finals, selections = [], [], []
        state = {'status': 'running', 'pid': os.getpid(), 'started_at': utc(), 'binding': binding,
                 'training_count': plan['training_count'], 'completed_trials': 0, 'datasets': {}}
        mutex, stop = threading.RLock(), threading.Event()

        def one_trial(dataset, method, lr, seed):
            if stop.is_set():
                raise RuntimeError('Stopped scheduling after another dataset failed')
            with mutex:
                state['datasets'][dataset] = {'status': 'running', 'method': method, 'lr': lr, 'seed': seed}
                state['updated_at'] = utc()
                save(args.output_root / 'status.json', state)
            if source_manifest(args.root) != fixed_source:
                raise ValueError('Source/HEAD/wrapper changed during execution')
            if any(sha(Path(audits[dataset][key])) != audits[dataset]['hashes'][hash_key]
                   for key, hash_key in [('cache', 'cache_sha256'), ('observed_labels', 'obs_sha256')]):
                raise ValueError('Study input changed during execution')
            row = run_trial(args, dataset, method, lr, seed, binding, audits[dataset])
            if source_manifest(args.root) != fixed_source:
                raise ValueError('Source/HEAD/wrapper changed during a trial')
            if any(sha(Path(audits[dataset][key])) != audits[dataset]['hashes'][hash_key]
                   for key, hash_key in [('cache', 'cache_sha256'), ('observed_labels', 'obs_sha256')]):
                raise ValueError('Study input changed during a trial')
            with mutex:
                all_trials.append(row)
                state['completed_trials'] = len(all_trials)
                state['updated_at'] = utc()
                write_results(args, all_trials, finals, selections)
                save(args.output_root / 'status.json', state)
            return row

        def one_dataset(dataset):
            try:
                for method in METHODS:
                    candidates = [one_trial(dataset, method, lr, DEFAULT['tuning_seed']) for lr in DEFAULT['learning_rates']]
                    selected = select_trial(candidates)
                    with mutex:
                        selections.append({'dataset': dataset, 'method': method, 'learning_rate': selected['learning_rate'],
                                           'best_val_accuracy': selected['best_val_accuracy'], 'criterion': DEFAULT['selection']})
                        finals.append(selected)
                    for seed in DEFAULT['additional_seeds']:
                        row = one_trial(dataset, method, selected['learning_rate'], seed)
                        with mutex:
                            finals.append(row)
                            write_results(args, all_trials, finals, selections)
                with mutex:
                    state['datasets'][dataset] = {'status': 'completed'}
                    save(args.output_root / 'status.json', state)
            except BaseException:
                stop.set()
                raise

        try:
            with ThreadPoolExecutor(max_workers=min(args.jobs, len(args.datasets))) as pool:
                futures = [pool.submit(one_dataset, dataset) for dataset in args.datasets]
                for future in as_completed(futures):
                    future.result()
            state.update(status='completed', finished_at=utc())
            save(args.output_root / 'status.json', state)
        except BaseException as exc:
            state.update(status='failed', failed_at=utc(), error=repr(exc))
            save(args.output_root / 'status.json', state)
            raise
    print(json.dumps({'status': 'completed', 'results': str(args.output_root / 'results.json')}))


if __name__ == '__main__':
    main()
