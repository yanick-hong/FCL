"""Single finite experiment queue; persisted on the authorized AutoDL instance."""
from pathlib import Path
import datetime as dt
import json
import os
import subprocess
import sys
import traceback

ROOT = Path('/root/autodl-tmp')
OUT = ROOT / 'outputs/controlled_correction_20261115'
DATASETS = ['dtd', 'caltech101', 'stanfordcars', 'eurosat', 'cifar100', 'food101', 'tinyimagenet200']
QWEN_PYTHON = ROOT / '.autodl/qwen-venv/bin/python'
BASE_PYTHON = Path('/root/miniconda3/bin/python')

def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def save(state):
    temporary = OUT / 'worker_status.json.tmp'
    temporary.write_text(json.dumps(state, indent=2))
    temporary.replace(OUT / 'worker_status.json')

def command(state, arguments, logfile):
    state.update(current_command=list(map(str, arguments)), updated_at=now())
    save(state)
    with (OUT / logfile).open('a') as output:
        subprocess.run(list(map(str, arguments)), cwd=ROOT, env={**os.environ, 'OMP_NUM_THREADS':'4',
            'MKL_NUM_THREADS':'4', 'TOKENIZERS_PARALLELISM':'false'}, stdout=output,
            stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, check=True)

if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    if len(sys.argv) == 1:
        if (OUT / 'worker_pid.json').exists():
            raise RuntimeError('Worker already launched; inspect status and process before restarting')
        with (OUT / 'worker.log').open('a') as output:
            worker = subprocess.Popen([str(BASE_PYTHON), str(Path(__file__).resolve()), '--worker'], cwd=ROOT,
                start_new_session=True, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT)
        (OUT / 'worker_pid.json').write_text(json.dumps({'pid':worker.pid,'started_at':now()}))
        print(json.dumps({'pid':worker.pid,'status_file':str(OUT/'worker_status.json')}))
        sys.exit(0)
    source = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()
    state = {'status':'running','pid':os.getpid(),'started_at':now(),'git_commit':source,
             'dataset_order':DATASETS,'datasets':{name:{'qwen':'pending','controlled_suite':'pending'} for name in DATASETS}}
    save(state)
    try:
        for dataset in DATASETS:
            if subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip() != source:
                raise RuntimeError('Source revision changed during campaign')
            state.update(current_dataset=dataset,current_phase='qwen')
            state['datasets'][dataset]['qwen'] = 'running'
            command(state, [QWEN_PYTHON,ROOT/'scripts/run_qwen_campaign.py','--datasets',dataset,
                    '--model-path',Path('dataset/weights/Qwen2.5-VL-7B-Instruct'),'--resume'],f'qwen_{dataset}.log')
            state['datasets'][dataset]['qwen'] = 'completed'
            state.update(current_phase='controlled_suite')
            state['datasets'][dataset]['controlled_suite'] = 'running'
            command(state, [BASE_PYTHON,ROOT/'scripts/run_controlled_suite.py','--datasets',dataset,
                           '--device','cuda','--execute'],f'suite_{dataset}.log')
            state['datasets'][dataset]['controlled_suite'] = 'completed'
            save(state)
        state.update(current_phase='summary_all')
        command(state,[BASE_PYTHON,ROOT/'scripts/run_controlled_suite.py','--phases','summary',
                       '--device','cuda','--execute'],'suite_summary_all.log')
        state.update(status='completed',completed_at=now())
        save(state)
    except Exception as error:
        state.update(status='failed',error=repr(error),failed_at=now())
        save(state)
        traceback.print_exc()
        raise
