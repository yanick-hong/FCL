"""Print concise preparation status without dumping large JSON/log files."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from paths import OUTPUT_ROOT, PROCESSED_ROOT, clip_cache, observed_labels_cache


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path)
    p.add_argument('--verify', action='store_true', help='Require completion and verify cached file hashes and prototype alignment.')
    a = p.parse_args()
    if a.run_dir:
        source = a.run_dir / 'status.json'
    else:
        candidates = sorted((OUTPUT_ROOT / '_preparation').glob('*/status.json'), key=lambda x: x.stat().st_mtime)
        if not candidates:
            p.error('No preparation status found')
        source = candidates[-1]
    report = json.loads(source.read_text())
    print('Run:', source.parent)
    print('Stage:', report.get('stage'), '| current dataset:', report.get('current_dataset'))
    for dataset, result in report.get('datasets', {}).items():
        if result['status'] == 'ready':
            sizes = '/'.join(str(result['splits'][s]['samples']) for s in ('train','val','test'))
            print(f"{dataset}: READY train/val/test={sizes}, corrected={result['corrected']}, observed_label_accuracy={result['observed_accuracy']:.4%}")
        else:
            print(dataset+': FAILED '+result.get('error',''))
    print('Skipped:', ', '.join(report.get('skipped',[])))
    if a.verify:
        if report.get('stage') != 'finished' or report.get('failures') != 0:
            p.error('Preparation is not complete; inspect the status and logs.')
        import torch
        from prepare_materials import sha256
        from fcl.train_auc_ce import load_clip_cache_strict, load_obs_labels
        expected = set(report['config']['datasets'])
        assert set(report['datasets']) == expected
        for dataset, result in report['datasets'].items():
            assert result['status'] == 'ready'
            cp, op = clip_cache(dataset), observed_labels_cache(dataset)
            assert sha256(cp) == result['cache_sha256'], f'Cache changed: {dataset}'
            assert sha256(op) == result['labels_sha256'], f'Labels changed: {dataset}'
            # Verify the actual main-method loaders accept the prepared schemas.
            cache, obs = load_clip_cache_strict(str(cp)), load_obs_labels(str(op))
            proto = torch.load(PROCESSED_ROOT/f'{dataset}_prototypes.pt', map_location='cpu', weights_only=True)
            assert proto['cache_sha256'] == result['cache_sha256']
            assert torch.equal(proto['corrected_mask'], obs['s'].eq(0))
            assert proto['prototypes'].shape == (result['classes'], 768)
            assert len(cache['train_feats']) == len(obs['y_obs'])
            print(dataset+': VERIFIED (main loaders, SHA256, prototype/label alignment)')


if __name__ == '__main__':
    main()
