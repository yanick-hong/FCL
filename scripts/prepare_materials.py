"""Safely unpack uploaded archives and rebuild five cache-only experiment inputs."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sys
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from paths import DATA_ROOT, RAW_ROOT, PROCESSED_ROOT, OUTPUT_ROOT, clip_cache, observed_labels_cache

SPECS = {
    'dtd': ('dtd.zip', 'dtd', 'dtd', 47, 'make_obs_labels_dtd.py'),
    'caltech101': ('caltech-101.zip', 'caltech-101', 'caltech101', 101, 'make_obs_labels_caltech101.py'),
    'eurosat': ('EuroSAT.zip', '', 'eurosat', 10, 'make_obs_labels_eurosat.py'),
    'cifar100': ('cifar-100-python-pickle.zip', 'cifar-100-python-pickle', 'CIFAR100', 100, 'make_obs_labels_cifar100.py'),
    'tinyimagenet200': ('tiny-imagenet-200.zip', 'tiny-imagenet-200', 'tiny-imagenet-200', 200, 'make_obs_labels_tinyimagenet.py'),
    'food101': ('food-101.tar.gz', 'food-101', 'food-101', 101, 'make_obs_labels_food101.py'),
    'stanfordcars': ('stanford-cars.zip', '', 'stanford-cars', 196, 'make_obs_labels_stanford_cars.py'),
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def dump(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def safe_target(root, name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or '\\' in name:
        raise ValueError(f'Unsafe archive entry: {name}')
    target = root.joinpath(*relative.parts)
    if not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Archive path escapes destination: {name}')
    return target


def extract_archive(dataset):
    filename, prefix, folder, _, _ = SPECS[dataset]
    archives = DATA_ROOT / 'archives'
    archives.mkdir(parents=True, exist_ok=True)
    source = DATA_ROOT / filename
    archive = archives / filename
    if source.exists():
        if archive.exists():
            raise FileExistsError(f'Both upload and archived ZIP exist: {filename}')
        source.rename(archive)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    digest = sha256(archive)
    target = RAW_ROOT / folder
    marker = target / '.fcl_extraction.json'
    pending = target / '.fcl_extracting.json'
    if marker.exists():
        if json.loads(marker.read_text())['archive_sha256'] != digest:
            raise ValueError(f'Archive differs from already extracted data: {dataset}')
        return digest
    if target.exists() and any(target.iterdir()):
        if not pending.exists() or json.loads(pending.read_text())['archive_sha256'] != digest:
            raise FileExistsError(f'Refusing to overwrite untracked raw data: {target}')
    target.mkdir(parents=True, exist_ok=True)
    dump(pending, {'archive_sha256': digest})
    count = 0
    def unpack_member(name, is_dir, source, symlink=False):
        parts = PurePosixPath(name).parts
        if not parts or '__MACOSX' in parts or parts[-1].startswith('._') or parts[-1] == '.DS_Store':
            return
        if symlink:
            raise ValueError(f'Archive symlink rejected: {name}')
        if prefix:
            if parts[0] != prefix:
                raise ValueError(f'Unexpected archive root: {name}')
            parts = parts[1:]
        if not parts:
            return
        dest = safe_target(target, '/'.join(parts))
        if is_dir:
            dest.mkdir(parents=True, exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            with source as src, dest.open('wb') as out:
                shutil.copyfileobj(src, out)
            nonlocal_count[0] += 1

    nonlocal_count = [0]
    if str(archive).lower().endswith(('.tar.gz', '.tgz', '.tar')):
        with tarfile.open(archive) as tf:
            for member in tf:
                if not (member.isfile() or member.isdir()):
                    unpack_member(member.name, False, None, symlink=True)
                else:
                    source = tf.extractfile(member) if member.isfile() else None
                    unpack_member(member.name, member.isdir(), source)
    else:
        with zipfile.ZipFile(archive) as z:
            needed = sum(i.file_size for i in z.infolist())
            if shutil.disk_usage(target).free < needed + 2 * 1024**3:
                raise OSError('Insufficient free space for extraction plus feature caches')
            for info in z.infolist():
                source = z.open(info) if not info.is_dir() else None
                unpack_member(info.filename, info.is_dir(), source,
                              symlink=stat.S_ISLNK(info.external_attr >> 16))
    count = nonlocal_count[0]
    if dataset == 'caltech101':
        for filename in ('101_ObjectCategories.tar.gz', 'Annotations.tar'):
            with tarfile.open(target / filename) as tf:
                for member in tf:
                    dest = safe_target(target, member.name)
                    if member.isdir():
                        dest.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        with tf.extractfile(member) as src, dest.open('wb') as out:
                            shutil.copyfileobj(src, out)
                    else:
                        raise ValueError(f'Non-regular TAR entry: {member.name}')
    dump(marker, {'archive_sha256': digest, 'zip_files': count, 'dataset': dataset})
    pending.unlink()
    print(f'[EXTRACTED] {dataset}: {count} ZIP files -> {target}', flush=True)
    return digest


def extraction_command(dataset, batch, workers):
    base = [sys.executable]
    cache = str(clip_cache(dataset))
    scripts = 'src/utils/extract/'
    if dataset == 'cifar100':
        return base + [scripts+'extract_CLIP_CIFAR100.py', '--root', str(RAW_ROOT/'CIFAR100'), '--cache', cache, '--batch_size', str(batch), '--num_workers', str(workers), '--seed', '42']
    if dataset == 'eurosat':
        return base + [scripts+'extract_CLIP_eurosat.py', '--root', str(RAW_ROOT), '--cache', cache, '--batch_size', str(batch), '--num_workers', str(workers), '--seed', '42']
    if dataset == 'dtd':
        return base + [scripts+'extract_CLIP_dtd.py', '--data_root', str(RAW_ROOT), '--out', cache, '--batch', str(batch), '--workers', str(workers), '--seed', '42']
    if dataset == 'caltech101':
        return base + [scripts+'extract_CLIP_caltech101.py', '--data_root', str(RAW_ROOT), '--out_path', cache, '--batch_size', str(batch), '--num_workers', str(workers), '--seed', '42']
    if dataset == 'food101':
        return base + [scripts+'extract_CLIP_food101.py', '--data_root', str(RAW_ROOT/'food-101'), '--out', cache, '--batch', str(batch), '--workers', str(workers), '--seed', '42']
    if dataset == 'stanfordcars':
        return base + [scripts+'extract_CLIP_stanford_cars.py', '--data_root', str(RAW_ROOT/'stanford-cars'), '--cache', cache, '--batch_size', str(batch), '--workers', str(workers), '--seed', '42']
    return base + [scripts+'extract_CLIP_tinyimagenet_openai.py', '--data_root', str(RAW_ROOT), '--cache', cache, '--batch_size', str(batch), '--workers', str(workers), '--seed', '42']


def audit_and_export(dataset, archive_hash, rho):
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(4)
    cache_path, obs_path = clip_cache(dataset), observed_labels_cache(dataset)
    c = torch.load(cache_path, map_location='cpu', weights_only=True)
    o = torch.load(obs_path, map_location='cpu', weights_only=True)
    classes = SPECS[dataset][3]
    splits = {}
    for split in ('train', 'val', 'test'):
        x, y, p = c[split+'_feats'], c[split+'_labels'], c['clip_probs_'+split]
        assert x.ndim == 2 and x.shape[1] == 768 and len(x) == len(y) > 0
        assert p.shape == (len(y), classes)
        assert torch.isfinite(x).all() and torch.isfinite(p).all()
        assert ((y >= 0) & (y < classes)).all()
        assert (p >= 0).all() and (p <= 1).all()
        assert torch.allclose(p.sum(1), torch.ones(len(y)), atol=2e-3)
        assert torch.equal(p.argmax(1), c['clip_argmax_'+split])
        splits[split] = {'samples': len(y), 'vlm_accuracy': float(p.argmax(1).eq(y).float().mean())}
    x = F.normalize(c['train_feats'].float(), dim=-1)
    y, pseudo, observed, s = c['train_labels'], c['clip_argmax_train'], o['y_obs'], o['s']
    assert s.shape == observed.shape == y.shape
    assert ((s == 0) | (s == 1)).all()
    assert torch.equal(observed[s == 0], y[s == 0])
    assert torch.equal(observed[s == 1], pseudo[s == 1])
    centers = torch.zeros(classes, x.shape[1])
    counts = torch.bincount(pseudo, minlength=classes)
    distances = torch.zeros(len(y))
    for cls in range(classes):
        idx = pseudo.eq(cls)
        if idx.any():
            centers[cls] = F.normalize(x[idx].mean(0), dim=0)
            distances[idx] = 1 - x[idx] @ centers[cls]
    split_ids = c.get('sample_ids')
    if split_ids:
        sets = [set(split_ids[k]) for k in ('train','val','test')]
        assert all(len(sets[i]) == splits[k]['samples'] for i,k in enumerate(('train','val','test')))
        assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    for keys in [('train_indices','val_indices','test_indices'), ('train_idx_from_train50k','val_idx_from_train50k')]:
        if all(isinstance(c.get(k), torch.Tensor) for k in keys):
            groups = [set(c[k].tolist()) for k in keys]
            assert all(not groups[i] & groups[j] for i in range(len(groups)) for j in range(i))
    h = sha256(cache_path)
    torch.save({'prototypes': centers, 'pseudo_class_counts': counts, 'outlier_scores': distances,
                'corrected_mask': s.eq(0), 'cache_sha256': h, 'rho': rho},
               PROCESSED_ROOT / f'{dataset}_prototypes.pt')
    wrong = pseudo.ne(y)
    metrics = {'dataset': dataset, 'classes': classes, 'feature_dim': 768, 'splits': splits,
               'rho_requested': rho, 'rho_actual': float(s.eq(0).float().mean()),
               'corrected': int(s.eq(0).sum()), 'uncorrected': int(s.eq(1).sum()),
               'observed_accuracy': float(observed.eq(y).float().mean()),
               'selected_error_precision': float(wrong[s==0].float().mean()),
               'selected_error_recall': float((wrong & s.eq(0)).sum() / wrong.sum()) if wrong.any() else None,
               'uncorrected_error_rate': float(wrong[s==1].float().mean()) if s.eq(1).any() else None,
               'archive_sha256': archive_hash, 'cache_sha256': h, 'labels_sha256': sha256(obs_path),
               'protocol': 'rebuild_dedicated_extractors_seed42', 'status': 'ready'}
    dump(PROCESSED_ROOT / f'{dataset}_preparation.json', metrics)
    return metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets', nargs='+', choices=list(SPECS), default=list(SPECS))
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--rho', type=float, default=0.3)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--reuse-raw', action='store_true', help='Use existing raw directories without unpacking archives.')
    p.add_argument('--dry-run', action='store_true', help='Plan commands only; do not unpack or execute extraction.')
    a = p.parse_args()
    if not 0 < a.rho <= 1:
        p.error('rho must be in (0, 1]')
    a.run_dir.mkdir(parents=True, exist_ok=True)
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    report = {'started_at': datetime.now(timezone.utc).isoformat(), 'datasets': {},
              'skipped': [], 'config': {**vars(a), 'run_dir': str(a.run_dir)}}
    commands = []
    failures = 0
    for dataset in a.datasets:
        try:
            report['current_dataset'] = dataset
            report['stage'] = 'extract'
            dump(a.run_dir/'status.json', report)
            raw_folder = RAW_ROOT / SPECS[dataset][2]
            if a.reuse_raw and raw_folder.exists():
                digest = 'reused-existing-raw'
                print(f'[REUSE-RAW] {raw_folder}', flush=True)
            elif a.dry_run:
                digest = 'dry-run'
                print(f'[DRY-RUN] would extract {dataset} into {raw_folder}', flush=True)
            else:
                digest = extract_archive(dataset)
            feature_command = extraction_command(dataset, a.batch_size, a.workers)
            label_command = [sys.executable, 'src/utils/labels/'+SPECS[dataset][4],
                             '--cache_path', str(clip_cache(dataset)), '--save_path', str(observed_labels_cache(dataset)),
                             '--top_outlier_pct', str(a.rho)]
            for stage, command, output in [('features',feature_command,clip_cache(dataset)),
                                            ('labels',label_command,observed_labels_cache(dataset))]:
                report['stage'] = stage
                dump(a.run_dir/'status.json', report)
                commands.append({'dataset':dataset, 'stage':stage, 'argv':command})
                dump(a.run_dir/'commands.json', commands)
                if a.dry_run or output.exists():
                    print(f'[REUSE] {output}', flush=True)
                    continue
                print('[RUN]', ' '.join(command), flush=True)
                with (a.run_dir/f'{dataset}_{stage}.log').open('a') as log:
                    subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            if a.dry_run:
                report['datasets'][dataset] = {'status': 'planned', 'archive_sha256': digest,
                                               'raw_root': str(raw_folder), 'commands': commands[-2:]}
            else:
                report['datasets'][dataset] = audit_and_export(dataset, digest, a.rho)
            print('[READY]',json.dumps(report['datasets'][dataset]),flush=True)
        except Exception as error:
            failures += 1
            report['datasets'][dataset] = {'status':'failed','error':repr(error)}
            print('[FAILED]',dataset,repr(error),flush=True)
        dump(a.run_dir/'status.json',report)
    report.update(stage='finished', finished_at=datetime.now(timezone.utc).isoformat(), failures=failures)
    dump(a.run_dir/'status.json',report)
    raise SystemExit(1 if failures else 0)


if __name__ == '__main__':
    main()
