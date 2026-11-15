"""Run one-at-a-time mu/tau sensitivity sweeps for FCL+AUC-C."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, subprocess, sys, time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))
from paths import clip_cache, observed_labels_cache
from fcl.train_auc_ce import LinearHead, PKBatchSampler, set_seed
from fcl.auc_loss import soft_pairwise_auc


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def objective(logits, prior, observed, s, config):
    corrected = s.eq(0)
    ce = F.cross_entropy(logits[corrected], observed[corrected]) if corrected.any() else logits.sum() * 0
    auc = soft_pairwise_auc(logits, prior, s, config['mu'], config['tau'], config['auc_chunk_size'])
    return ce + config['lam'] * auc, ce, auc


@torch.no_grad()
def evaluate(model, features, labels):
    total_loss = torch.zeros((), device=features.device)
    correct = torch.zeros((), device=features.device)
    for start in range(0, len(labels), 4096):
        logits = model(features[start:start + 4096])
        y = labels[start:start + 4096]
        total_loss += F.cross_entropy(logits, y, reduction='sum')
        correct += logits.argmax(-1).eq(y).sum()
    return float(total_loss / len(labels)), float(correct / len(labels))


def train_one(bundle, scenario, config, seed, destination, context):
    assert 'train_labels' not in bundle
    destination.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = bundle['train_x'].device
    observed, s = scenario['y_obs'].to(device), scenario['s'].to(device)
    sampler = PKBatchSampler(scenario['y_obs'], config['P'], config['K'], seed)
    model = LinearHead(bundle['train_x'].shape[1], bundle['prior'].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['lr_max'], total_steps=len(sampler) * config['epochs'],
        pct_start=config['onecycle_pct_start'], anneal_strategy='cos',
        div_factor=config['onecycle_div_factor'], final_div_factor=config['onecycle_final_div_factor'])
    visible = {**config, **context, 'seed': seed, 'effective_initial_lr': optimizer.param_groups[0]['lr']}
    write_json(destination / 'config.json', visible)
    started = time.time()
    write_json(destination / 'run_metadata.json', {
        **context, 'method': 'FCL+AUC-C', 'script': 'ablation/parameter_sensitivity/run.py',
        'command': [sys.executable, *sys.argv], 'seed': seed,
        'started_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(), 'config': visible,
        'status': 'running'})
    best, best_epoch, bad = -1., 0, 0
    fields = ['epoch', 'train_loss', 'train_ce', 'train_auc', 'val_loss', 'val_accuracy', 'lr', 'seconds']
    with (destination / 'history.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for epoch in range(1, config['epochs'] + 1):
            tick = time.time(); model.train(); sums = torch.zeros(3, device=device)
            for indices in sampler:
                idx = torch.tensor(indices, dtype=torch.long, device=device)
                logits = model(bundle['train_x'][idx])
                loss, ce, auc = objective(logits, bundle['prior'][idx], observed[idx], s[idx], config)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss in {destination}')
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip_norm'])
                optimizer.step(); scheduler.step()
                sums += torch.stack((loss.detach(), ce.detach(), auc.detach()))
            model.eval(); val_loss, val_acc = evaluate(model, bundle['val_x'], bundle['val_y'])
            means = (sums / len(sampler)).tolist()
            row = dict(zip(fields, [epoch, *means, val_loss, val_acc,
                                    optimizer.param_groups[0]['lr'], time.time() - tick]))
            writer.writerow(row); handle.flush()
            if val_acc > best:
                best, best_epoch, bad = val_acc, epoch, 0
                torch.save(model.state_dict(), destination / 'best.ckpt')
            else:
                bad += 1
            torch.save(model.state_dict(), destination / 'last.ckpt')
            write_json(destination / 'progress.json', {'epoch': epoch, 'best_epoch': best_epoch,
                       'best_val_accuracy': best, 'last': row})
            if epoch == 1 or epoch % 10 == 0:
                print(f"[EPOCH] {context['dataset']} {context['sweep']}={context['value']} seed={seed} "
                      f"epoch={epoch} val={val_acc:.4f} time={row['seconds']:.2f}s", flush=True)
            if bad >= config['patience']:
                break
    model.load_state_dict(torch.load(destination / 'best.ckpt', map_location=device, weights_only=True))
    test_loss, test_acc = evaluate(model, bundle['test_x'], bundle['test_y'])
    metrics = {'status': 'completed', 'method': 'FCL+AUC-C', **context, 'seed': seed,
               'best_epoch': best_epoch, 'epochs_completed': epoch, 'best_val_accuracy': best,
               'test_accuracy': test_acc, 'test_loss': test_loss,
               'elapsed_seconds': time.time() - started, 'test_checkpoint': 'best.ckpt'}
    write_json(destination / 'metrics.json', metrics)
    write_json(destination / 'run_metadata.json', {
        **context, 'method': 'FCL+AUC-C', 'script': 'ablation/parameter_sensitivity/run.py',
        'command': [sys.executable, *sys.argv], 'seed': seed, 'config': visible,
        'status': 'completed', 'ended_at': datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()})
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=HERE / 'config.yaml')
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    stamp = datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    campaign = (args.resume or HERE / 'results' / stamp).resolve(); campaign.mkdir(parents=True, exist_ok=True)
    sources = {str(path.relative_to(ROOT)): digest(path) for path in
               [Path(__file__), HERE / 'config.yaml', ROOT / 'src/fcl/auc_loss.py', ROOT / 'src/fcl/train_auc_ce.py']}
    frozen = {'config': config, 'sources': sources}
    if (campaign / 'campaign.json').exists() and json.loads((campaign / 'campaign.json').read_text()) != frozen:
        raise ValueError('Resume config/source differs from frozen campaign')
    write_json(campaign / 'campaign.json', frozen)
    write_json(campaign / 'runtime.json', {'python': sys.version, 'torch': torch.__version__,
               'cuda': torch.version.cuda, 'numpy': np.__version__, 'git_head': None,
               'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               'timezone': 'Asia/Shanghai'}) if not (campaign / 'runtime.json').exists() else None
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    jobs = [('mu', float(value), config['tau_default']) for value in config['mu_values']]
    jobs += [('tau', float(value), float(config['mu_default'])) for value in config['tau_values']]
    expected = len(config['datasets']) * len(jobs) * len(config['seeds'])
    status = {'campaign': str(campaign), 'expected': expected, 'completed': 0, 'failed': [], 'status': 'running'}
    write_json(campaign / 'status.json', status)
    print('[CAMPAIGN]', campaign, 'expected', expected, flush=True)
    for dataset in config['datasets']:
        cache_path, labels_path = clip_cache(dataset), observed_labels_cache(dataset)
        cache = torch.load(cache_path, map_location='cpu', weights_only=True)
        base = torch.load(labels_path, map_location='cpu', weights_only=True)
        context_base = {'dataset': dataset, 'cache_sha256': digest(cache_path), 'base_labels_sha256': digest(labels_path)}
        bundle = {'train_x': F.normalize(cache['train_feats'].float(), dim=-1).to(device),
                  'prior': cache['clip_probs_train'].float().to(device),
                  'val_x': F.normalize(cache['val_feats'].float(), dim=-1).to(device),
                  'val_y': cache['val_labels'].long().to(device),
                  'test_x': F.normalize(cache['test_feats'].float(), dim=-1).to(device),
                  'test_y': cache['test_labels'].long().to(device)}
        scenario = {'y_obs': base['y_obs'].long(), 's': base['s'].long()}
        assert torch.equal(scenario['y_obs'][scenario['s'].eq(0)], cache['train_labels'][scenario['s'].eq(0)])
        for sweep, value, other in jobs:
            name = f'{sweep}_{value:.2f}'.replace('.', 'p')
            for seed in config['seeds']:
                destination = campaign / dataset / name / f'seed_{seed}'
                context = {**context_base, 'sweep': sweep, 'value': value,
                           'mu': value if sweep == 'mu' else other,
                           'tau': value if sweep == 'tau' else other}
                status['current'] = {'dataset': dataset, 'sweep': sweep, 'value': value,
                                     'seed': seed, 'path': str(destination)}
                write_json(campaign / 'status.json', status)
                metrics_path = destination / 'metrics.json'
                if metrics_path.exists():
                    previous = json.loads(metrics_path.read_text())
                    if previous.get('status') == 'completed' and all(previous.get(k) == v for k, v in context.items()):
                        status['completed'] += 1; continue
                    raise RuntimeError(f'Existing result metadata mismatch: {metrics_path}')
                run_config = {**config, 'mu': context['mu'], 'tau': context['tau']}
                print('[RUN]', dataset, sweep, value, 'seed', seed, flush=True)
                try:
                    result = train_one(bundle, scenario, run_config, seed, destination, context)
                    status['completed'] += 1
                    print('[DONE]', dataset, sweep, value, seed, result['test_accuracy'], flush=True)
                except Exception as error:
                    write_json(destination / 'failure.json', {'error': repr(error)})
                    status['failed'].append({**status['current'], 'error': repr(error)})
                    write_json(campaign / 'status.json', status); raise
                write_json(campaign / 'status.json', status)
        del bundle, cache, base
        if device.type == 'cuda': torch.cuda.empty_cache()
    status['status'] = 'completed'; write_json(campaign / 'status.json', status)
    subprocess.run([sys.executable, str(HERE / 'summarize.py'), '--campaign', str(campaign)], check=True)
    print('[ALL DONE]', campaign, flush=True)


if __name__ == '__main__':
    main()
