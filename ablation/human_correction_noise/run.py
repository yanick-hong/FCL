"""Paired human-correction-noise ablation; no training ground truth enters learner."""
from __future__ import annotations
import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
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
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def make_scenarios(base_observed, s, truth, classes, accuracies, seed):
    """Oracle simulator only. Returns visible labels, never extra clean targets."""
    base_observed, s, truth = [t.cpu().long() for t in (base_observed, s, truth)]
    corrected = torch.where(s.eq(0))[0]
    if not len(corrected):
        raise ValueError('No corrected samples')
    if not torch.equal(base_observed[corrected], truth[corrected]):
        raise ValueError('Baseline corrected labels are not clean')
    rng = np.random.default_rng(seed)
    priority = np.empty(len(corrected))
    y = truth[corrected].numpy()
    for cls in np.unique(y):
        positions = rng.permutation(np.flatnonzero(y == cls))
        priority[positions] = (np.arange(len(positions)) + 0.5) / len(positions)
    ordering = np.lexsort((rng.random(len(corrected)), priority))
    order = corrected[torch.from_numpy(ordering)]
    # Each changed label differs from the true label, even for binary classes.
    wrong = rng.integers(0, classes-1, size=len(corrected))
    wrong = torch.from_numpy(wrong + (wrong >= y)).long()
    replacement = base_observed.clone()
    replacement[corrected] = wrong
    result = {}
    for accuracy in accuracies:
        count = int(round((1-float(accuracy))*len(corrected)))
        indices = order[:count].clone()
        observed = base_observed.clone()
        observed[indices] = replacement[indices]
        assert torch.equal(observed[s.eq(1)], base_observed[s.eq(1)])
        assert int(observed[corrected].ne(truth[corrected]).sum()) == count
        result[float(accuracy)] = {
            'y_obs': observed, 's': s.clone(), 'corrupted_indices': indices,
            'meta': {'noise_seed': seed, 'corrected_count': len(corrected),
                     'errors': count, 'accuracy_requested': float(accuracy),
                     'accuracy_actual': 1-count/len(corrected),
                     'class_error_counts': torch.bincount(truth[indices], minlength=classes).tolist(),
                     'noise_model': 'uniform_wrong_class_nested_class_balanced'}}
    return result


def objective(logits, prior, observed, s, config):
    corrected = s.eq(0)
    ce = F.cross_entropy(logits[corrected], observed[corrected]) if corrected.any() else logits.sum()*0
    auc = soft_pairwise_auc(logits, prior, s, config['mu'], config['tau'], config['auc_chunk_size'])
    return ce + config['lam']*auc, ce, auc


@torch.no_grad()
def evaluate(model, features, labels):
    total_loss = torch.zeros((), device=features.device)
    correct = torch.zeros((), device=features.device)
    for start in range(0, len(labels), 4096):
        logits = model(features[start:start+4096])
        y = labels[start:start+4096]
        total_loss += F.cross_entropy(logits, y, reduction='sum')
        correct += logits.argmax(-1).eq(y).sum()
    return float(total_loss/len(labels)), float(correct/len(labels))


def train_one(bundle, scenario, config, seed, destination, context):
    """bundle intentionally excludes training true labels."""
    assert 'train_labels' not in bundle
    destination.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    device = bundle['train_x'].device
    observed = scenario['y_obs'].to(device)
    s = scenario['s'].to(device)
    sampler = PKBatchSampler(scenario['y_obs'], config['P'], config['K'], seed)
    model = LinearHead(bundle['train_x'].shape[1], bundle['prior'].shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=config['lr_max'], total_steps=len(sampler)*config['epochs'],
        pct_start=config['onecycle_pct_start'], anneal_strategy='cos',
        div_factor=config['onecycle_div_factor'], final_div_factor=config['onecycle_final_div_factor'])
    visible_config = {**config, **context, 'seed':seed, 'scenario':scenario['meta'],
                      'effective_initial_lr': optimizer.param_groups[0]['lr']}
    write_json(destination/'config.json', visible_config)
    started = time.time()
    metadata = {**context, 'method':'FCL+AUC-C', 'script':'ablation/human_correction_noise/run.py',
                'command':[sys.executable,*sys.argv], 'seed':seed, 'started_at':datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(),
                'config':visible_config, 'status':'running'}
    write_json(destination/'run_metadata.json', metadata)
    best, best_epoch, patience = -1., 0, 0
    fields = ['epoch','train_loss','train_ce','train_auc','val_loss','val_accuracy','lr','seconds']
    with (destination/'history.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields);writer.writeheader()
        for epoch in range(1, config['epochs']+1):
            tick = time.time();model.train()
            sums = torch.zeros(3,device=device)
            for indices in sampler:
                idx = torch.tensor(indices, dtype=torch.long, device=device)
                logits = model(bundle['train_x'][idx])
                loss, ce, auc = objective(logits,bundle['prior'][idx],observed[idx],s[idx],config)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f'Nonfinite loss in {destination}')
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip_norm'])
                optimizer.step();scheduler.step()
                sums += torch.stack((loss.detach(),ce.detach(),auc.detach()))
            model.eval()
            vl,va = evaluate(model,bundle['val_x'],bundle['val_y'])
            means = (sums/len(sampler)).tolist()
            row=dict(zip(fields,[epoch,*means,vl,va,optimizer.param_groups[0]['lr'],time.time()-tick]))
            writer.writerow(row);f.flush()
            if va>best:
                best,best_epoch,patience=va,epoch,0
                torch.save(model.state_dict(),destination/'best.ckpt')
            else:
                patience+=1
            torch.save(model.state_dict(),destination/'last.ckpt')
            write_json(destination/'progress.json', {'epoch':epoch,'best_epoch':best_epoch,'best_val_accuracy':best,'last':row})
            if epoch==1 or epoch%10==0:
                print(f"[EPOCH] {context['dataset']} acc={scenario['meta']['accuracy_requested']} seed={seed} epoch={epoch} val={va:.4f} time={row['seconds']:.2f}s",flush=True)
            if patience>=config['patience']:
                break
    model.load_state_dict(torch.load(destination/'best.ckpt',map_location=device,weights_only=True))
    model.eval()
    test_loss,test_acc=evaluate(model,bundle['test_x'],bundle['test_y'])
    metrics={'status':'completed','method':'FCL+AUC-C',**context,'seed':seed,
             **scenario['meta'],'best_epoch':best_epoch,'epochs_completed':epoch,
             'best_val_accuracy':best,'test_accuracy':test_acc,'test_loss':test_loss,
             'elapsed_seconds':time.time()-started,'test_checkpoint':'best.ckpt'}
    write_json(destination/'metrics.json',metrics)
    metadata.update(status='completed',ended_at=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat())
    write_json(destination/'run_metadata.json',metadata)
    return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=HERE/'config.yaml')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--benchmark',action='store_true',help='One epoch on Tiny-ImageNet; separate smoke output, not included in results.')
    args=parser.parse_args()
    config=yaml.safe_load(args.config.read_text())
    if args.benchmark:
        config.update(datasets=['tinyimagenet200'],accuracies=[1.0],seeds=[42],epochs=1,patience=1)
    stamp=datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y%m%d_%H%M%S')
    campaign=args.resume or HERE/('smoke' if args.benchmark else 'results')/stamp
    campaign=campaign.resolve();campaign.mkdir(parents=True,exist_ok=True)
    sources={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),HERE/'config.yaml',ROOT/'src/fcl/auc_loss.py',ROOT/'src/fcl/train_auc_ce.py']}
    frozen={'config':config,'sources':sources}
    frozen_path=campaign/'campaign.json'
    if frozen_path.exists() and json.loads(frozen_path.read_text())!=frozen:
        raise ValueError('Resume config/source differs from the frozen campaign')
    write_json(frozen_path,frozen)
    if not (campaign/'runtime.json').exists():
        try:
            commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        except (OSError,subprocess.CalledProcessError):
            commit=None
        write_json(campaign/'runtime.json',{'python':sys.version,'torch':torch.__version__,
                   'cuda':torch.version.cuda,'numpy':np.__version__,'git_head':commit,
                   'gpu':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   'timezone':'Asia/Shanghai','cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG']})
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    status={'campaign':str(campaign),'expected':len(config['datasets'])*len(config['accuracies'])*len(config['seeds']),
            'completed':0,'failed':[],'status':'running'}
    print('[CAMPAIGN]',campaign,flush=True)
    for dataset in config['datasets']:
        cp,op=clip_cache(dataset),observed_labels_cache(dataset)
        cache=torch.load(cp,map_location='cpu',weights_only=True)
        base=torch.load(op,map_location='cpu',weights_only=True)
        context={'dataset':dataset,'cache_sha256':digest(cp),'base_labels_sha256':digest(op)}
        bundle={'train_x':F.normalize(cache['train_feats'].float(),dim=-1).to(device),
                'prior':cache['clip_probs_train'].float().to(device)}
        for split in ('val','test'):
            bundle[split+'_x']=F.normalize(cache[split+'_feats'].float(),dim=-1).to(device)
            bundle[split+'_y']=cache[split+'_labels'].long().to(device)
        for seed in config['seeds']:
            scenarios=make_scenarios(base['y_obs'],base['s'],cache['train_labels'],bundle['prior'].shape[1],
                                     config['accuracies'],seed+config['noise_seed_offset'])
            for accuracy in config['accuracies']:
                name=f'acc_{round(100*accuracy):03d}'
                destination=campaign/dataset/name/f'seed_{seed}'
                scenario=scenarios[float(accuracy)]
                input_path=campaign/'inputs'/dataset/f'seed_{seed}'/(name+'.pt')
                input_path.parent.mkdir(parents=True,exist_ok=True)
                if not input_path.exists():torch.save(scenario,input_path)
                else:
                    existing=torch.load(input_path,map_location='cpu',weights_only=True)
                    assert torch.equal(existing['y_obs'],scenario['y_obs']) and torch.equal(existing['s'],scenario['s'])
                scenario_context={**context,'scenario_sha256':digest(input_path),'scenario_path':str(input_path)}
                status['current']={'dataset':dataset,'accuracy':accuracy,'seed':seed,'path':str(destination)}
                write_json(campaign/'status.json',status)
                mp=destination/'metrics.json'
                if mp.exists():
                    previous=json.loads(mp.read_text())
                    if previous.get('status')=='completed' and all(previous.get(k)==v for k,v in scenario_context.items()):
                        status['completed']+=1
                        continue
                    raise RuntimeError(f'Existing result metadata mismatch: {mp}')
                print('[RUN]',dataset,name,'seed',seed,flush=True)
                try:
                    result=train_one(bundle,scenario,config,seed,destination,scenario_context)
                    status['completed']+=1
                    print('[DONE]',dataset,name,seed,'test',result['test_accuracy'],'seconds',result['elapsed_seconds'],flush=True)
                except Exception as error:
                    write_json(destination/'failure.json',{'error':repr(error)})
                    status['failed'].append({**status['current'],'error':repr(error)})
                    write_json(campaign/'status.json',status)
                    raise
                write_json(campaign/'status.json',status)
        del bundle,cache,base,scenarios
        if device.type=='cuda':torch.cuda.empty_cache()
    status['status']='completed'
    write_json(campaign/'status.json',status)
    if not args.benchmark:
        subprocess.run([sys.executable,str(HERE/'summarize.py'),'--campaign',str(campaign)],check=True)
    print('[ALL DONE]',campaign,flush=True)


if __name__=='__main__':main()
