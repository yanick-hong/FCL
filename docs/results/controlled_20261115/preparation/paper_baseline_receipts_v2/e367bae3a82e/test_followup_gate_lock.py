"""CPU-only plan/gate/flock validation; no execute() or learner invocation."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
WRAPPER=ROOT/'.autodl/paper_baseline_followup.py'
spec=importlib.util.spec_from_file_location('followup_gate_under_test',WRAPPER)
wrapper=importlib.util.module_from_spec(spec);spec.loader.exec_module(wrapper)
checks=[]
def check(name,value):
    checks.append({'check':name,'passed':bool(value)})
    if not value:raise AssertionError(name)
args=wrapper.parse_args(['--root',str(ROOT),'--resume','--plan-only'])
runner,metric,campaign,paper,datasets=wrapper.configure(args)
plan=wrapper.make_plan(args,runner,campaign,paper,datasets)
check('exactly_84_unique_jobs',len(plan['jobs'])==84 and len({j['id'] for j in plan['jobs']})==84)
check('fixed_100_epochs_seed42',all('--epochs' in j['command'] and j['command'][j['command'].index('--epochs')+1]=='100'
    and j['command'][j['command'].index('--seeds')+1]=='42' for j in plan['jobs']))
check('weak_paper_resume_never_passed',all('--resume' not in j['command'] for j in plan['jobs']))
check('no_torch_imported_by_plan','torch' not in sys.modules)
with tempfile.TemporaryDirectory(prefix='fcl_followup_gate_lock_') as directory:
    root=Path(directory)
    status=root/'worker_status.json'
    gate_plan={'core_status':str(status),'git_commit':plan['git_commit'],
               'core_required_datasets':campaign['datasets']}
    check('missing_core_waits',wrapper.core_state(gate_plan)[0]=='missing')
    def state(value,**extra):
        payload={'status':value,'git_commit':plan['git_commit'],'datasets':{
            d:{'qwen':'completed','controlled_suite':'completed'} for d in campaign['datasets']},**extra}
        status.write_text(json.dumps(payload));return payload
    state('running');check('running_core_waits',wrapper.core_state(gate_plan)[0]=='pending')
    state('failed');check('failed_core_blocked',wrapper.core_state(gate_plan)[0]=='failed')
    state('completed');check('complete_seven_dataset_core_passes',wrapper.core_state(gate_plan)[0]=='completed')
    payload=state('completed');payload['datasets'][campaign['datasets'][0]]['controlled_suite']='failed'
    status.write_text(json.dumps(payload));check('partial_core_blocked',wrapper.core_state(gate_plan)[0]=='inconsistent')
    state('completed',git_commit='wrong');check('different_head_blocked',wrapper.core_state(gate_plan)[0]=='inconsistent')
    child="""import importlib.util,pathlib,sys
spec=importlib.util.spec_from_file_location('lock_child',sys.argv[1]);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
try:
 with module.queue_lock(pathlib.Path(sys.argv[2])):print('acquired')
except RuntimeError as error:
 if 'already holds' not in str(error):raise
 print('busy');sys.exit(23)
"""
    command=[sys.executable,'-c',child,str(WRAPPER),str(root/'queue')]
    with wrapper.queue_lock(root/'queue'):
        process=subprocess.run(command,text=True,capture_output=True,env=wrapper.environment(cpu=True),timeout=20)
        check('concurrent_queue_lock_rejected',process.returncode==23 and process.stdout.strip()=='busy')
    process=subprocess.run(command,text=True,capture_output=True,env=wrapper.environment(cpu=True),timeout=20)
    check('queue_lock_released_on_exit',process.returncode==0 and process.stdout.strip()=='acquired')
print(json.dumps({'status':'passed','checks':checks,'count':len(checks),'training_performed':False,
                  'wrapper_sha256':wrapper.sha(WRAPPER),'git_commit':plan['git_commit']}))
