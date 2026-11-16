"""Launch and inspect the finite CPU CE recheck without modifying active code."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/root/autodl-tmp')
STUDY = ROOT / 'outputs/conf_ce_fair_recheck_20261116'
PYTHON = Path('/root/miniconda3/bin/python')
RUNNER = STUDY / 'preparation/fair_ce_runner.py'

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(tmp, path)

def env():
    result = dict(os.environ)
    result.update(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2', PYTHONUTF8='1', PYTHONIOENCODING='utf-8')
    return result

def command():
    return [str(PYTHON), str(RUNNER), '--root', str(ROOT), '--device', 'cpu', '--threads', '2', '--jobs', '3', '--execute', '--resume']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['pilot', 'start', 'status'])
    args = parser.parse_args()
    if args.mode == 'pilot':
        destination = STUDY / 'pilot/caltech101_conf_lr-0.002_seed-42'
        if destination.exists():
            raise RuntimeError('Pilot output already exists; preserve rather than overwrite it')
        destination.mkdir(parents=True)
        cmd = [str(PYTHON), str(ROOT / 'src/contrast/train_revision_baselines.py'), '--method', 'clip_conf_ce', '--dataset', 'caltech101', '--cache', str(ROOT / 'dataset/processed/caltech101_clip_vit-l-14_embeddings.pt'), '--obs-labels', str(ROOT / 'outputs/controlled_correction_20261115/corrections/caltech101/clip/prototype.pt'), '--run-dir', str(destination), '--epochs', '100', '--patience', '101', '--batch-size', '1024', '--lr', '0.002', '--weight-decay', '0.0001', '--seed', '42']
        started = time.time()
        with (destination / 'console.log').open('w') as handle:
            completed = subprocess.run(cmd, cwd=ROOT, env=env(), stdout=handle, stderr=subprocess.STDOUT, timeout=1800, preexec_fn=lambda: os.nice(10))
        result = {'command': cmd, 'exit_code': completed.returncode, 'elapsed_seconds': time.time() - started, 'purpose': 'Independent resource/CLI pilot; excluded from the 50 formal trials and parameter selection', 'output': str(destination)}
        if completed.returncode == 0:
            metrics = json.loads((destination / 'metrics.json').read_text())
            result.update(epochs=metrics['epochs_completed'], test_count=metrics['test']['valid_count'])
        write(STUDY / 'pilot_report.json', result)
        print(json.dumps(result))
        if completed.returncode:
            raise SystemExit(completed.returncode)
    elif args.mode == 'start':
        import fcntl
        with (STUDY / 'launch.lock').open('a+') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            launch = STUDY / 'worker_launch.json'
            if launch.exists():
                previous = json.loads(launch.read_text())
                pid = previous['pid']
                cmdline = Path(f'/proc/{pid}/cmdline')
                if cmdline.exists() and str(RUNNER).encode() in cmdline.read_bytes():
                    raise RuntimeError(f'Worker already active: {pid}')
            with (STUDY / 'worker.log').open('a') as handle:
                process = subprocess.Popen(command(), cwd=ROOT, env=env(), stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True, preexec_fn=lambda: os.nice(10))
            result = {'pid': process.pid, 'command': command(), 'started_at': dt.datetime.now(dt.timezone.utc).isoformat(), 'cpu_niceness': 10}
            write(launch, result)
            print(json.dumps(result))
    else:
        status = STUDY / 'status.json'
        result = json.loads(status.read_text()) if status.exists() else {'status': 'not_started'}
        if (STUDY / 'results.json').exists():
            metrics = json.loads((STUDY / 'results.json').read_text())
            result['completed_summaries'] = metrics['summary']
            result['selected_parameters'] = metrics['selected']
        if result.get('status') == 'failed':
            result['worker_log_tail'] = (STUDY / 'worker.log').read_text(errors='replace')[-5000:]
        print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
