"""CPU synthetic artifact checks only: no learner is launched or trained."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2),encoding='utf-8')

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

class SyntheticRun:
    def __init__(self,directory,method='fl_c',source='clip',raw_csgn=False,rounded_csgn_tie=False):
        self.root=Path(directory)
        self.run=self.root/'run'
        (self.run/'logs').mkdir(parents=True)
        self.method,self.source=method,source
        self.cache_path,self.obs_path=self.root/'cache.pt',self.root/'obs.pt'
        self.cache={f'{split}_feats':torch.eye(2) for split in ('train','val','test')}
        self.cache.update({f'{split}_labels':torch.tensor([0,1]) for split in ('train','val','test')})
        self.cache.update({f'clip_probs_{split}':torch.eye(2) for split in ('train','val','test')})
        self.cache.update({f'clip_argmax_{split}':torch.tensor([0,1]) for split in ('train','val','test')})
        self.cache['clip_label_embeds']=torch.eye(2)
        self.cache['class_names']=['first','second']
        self.cache['sample_ids']={split:[f'{split}-0',f'{split}-1'] for split in ('train','val','test')}
        if source=='qwen':
            self.cache['annotation_metadata']={split:{'replaced_clip_fields':True,
                'source_models':['Qwen/Qwen2.5-VL-7B-Instruct'],'prompt_variant':'C'}
                for split in ('train','val','test')}
        self.weight,self.bias=torch.eye(2),torch.zeros(2)
        self.checkpoint_epoch=40 if method=='l2b' else 20
        self.history_best_epoch=20
        if raw_csgn:
            self.cache['val_feats']=torch.tensor([[2.,0.],[1.,0.]])
            self.weight=torch.tensor([[1.,0.],[0.,0.]])
            self.bias=torch.tensor([-1.5,0.])
            self.cache['test_feats']=torch.tensor([[-1.,0.],[0.,1.]])
            self.cache['test_labels']=torch.tensor([1,1])
        if rounded_csgn_tie:
            count=10100
            self.cache['val_feats']=torch.tensor([[1.,0.]]).repeat(count,1)
            self.cache['val_labels']=torch.tensor([0]*51+[1]*(count-51))
            self.weight=torch.tensor([[1.,0.],[0.,0.]])
            self.bias=torch.zeros(2)
            self.checkpoint_epoch=40
            self.cache['clip_probs_val']=F.one_hot(self.cache['val_labels'],num_classes=2).float()
            self.cache['clip_argmax_val']=self.cache['val_labels'].clone()
            self.cache['sample_ids']['val']=[f'val-{index}' for index in range(count)]
        val_x=self.cache['val_feats'].float()
        if method!='csgn':
            val_x=F.normalize(val_x,dim=-1)
        logits=val_x@self.weight.T+self.bias
        self.best_val=(logits.argmax(-1)==self.cache['val_labels']).double().mean().item()
        self.best_nll=F.cross_entropy(logits,self.cache['val_labels']).item()
        logits=F.normalize(self.cache['test_feats'],dim=-1)@self.weight.T+self.bias
        test_y=self.cache['test_labels']
        self.test={'accuracy':(logits.argmax(-1)==test_y).float().mean().item(),
                   'nll':F.cross_entropy(logits,test_y).item(),
                   'valid_count':len(test_y),'total_count':len(test_y)}
        self.obs={'y_obs':torch.tensor([0,1]),'s':torch.tensor([0,1]),
                  'selected_mask':torch.tensor([True,False]),'y_pseudo':torch.tensor([0,1]),
                  'meta':{'label_source':source,'method':'prototype','selection':'prototype',
                          'dataset':'synthetic','top_outlier_pct':.3}}
        self.identity={'dataset':'synthetic','method':method,'source':source,'seed':42,'epochs':100}
        self.expected_fingerprint='a'*64
        self.manifest={'status':'completed','measured':True,**self.identity,
                       'job_key':self.identity,'fingerprint':self.expected_fingerprint}
        self.refresh_inputs()
        if method=='zero_shot':
            self.metrics={'method':method,'source':source,'measured':True,'metrics':{
                split:{'accuracy':float(self.cache[f'clip_argmax_{split}'].eq(self.cache[f'{split}_labels']).float().mean()),
                       'n':len(self.cache[f'{split}_labels'])} for split in ('train','val','test')}}
        else:
            self.history=[]
            precision=4 if method=='csgn' else 6 if method in ('l2b','dld') else None
            best_rounded=round(self.best_val,precision) if precision else self.best_val
            for epoch in range(1,101):
                accuracy=best_rounded if epoch in (20,40) else min(.5,best_rounded/2)
                if method in ('fl_c','nl_c','fs_c'):
                    last_logits=torch.tensor([[1.,0.],[1.,0.]])
                    last_nll=F.cross_entropy(last_logits,self.cache['val_labels']).item()
                    row={'epoch':epoch,'train_loss':.5,'val_acc':accuracy,
                         'val_loss':self.best_nll if epoch in (20,40) else last_nll}
                elif method=='dld':
                    row={'epoch':epoch,'val_ce_head':round(self.best_nll,6),'val_acc_head':accuracy}
                elif method=='l2b':
                    row={'epoch':epoch,'val_loss':round(self.best_nll,6),'val_acc':accuracy,'alpha_mean':.5,'beta_mean':.5}
                else:
                    row={'epoch':epoch,'val_loss':round(self.best_nll,6),'val_acc':accuracy,'train_elbo':1.,'train_sup_ce':1.}
                self.history.append(row)
            self.history_path=self.run/'logs'/('val_metrics_dld_linear.csv' if method=='dld' else 'val_metrics.csv')
            self.write_history()
            if method in ('fl_c','nl_c','fs_c'):
                self.metrics={'method':method,'source':source,'measured':True,'epochs_completed':100,
                    'best_epoch':20,'best_val_accuracy':best_rounded,'test':self.test,
                    'selected_train_samples':2 if method=='fs_c' else 1}
            else:
                self.metrics={'method':method,'dataset':'synthetic','status':'completed',
                    'selection':{'metric':'val_accuracy','best_val_accuracy':best_rounded,'best_epoch':20,
                                 'epochs_completed':100,'history':'logs/val_metrics.csv'},'test':self.test}
            if method=='dld':
                payload={'state_dict':{'diffusion.weight':torch.zeros(2,2),'linear.weight':self.weight,'linear.bias':self.bias},
                         'in_dim':2,'num_classes':2,'hidden':4,'tdim':2,'note':'synthetic'}
            elif method=='csgn':
                state={f'{prefix}.{suffix}':value for prefix in ('linear','fc','classifier')
                       for suffix,value in [('weight',self.weight),('bias',self.bias)]}
                payload={'state_dict':state,'linear.weight':self.weight,'linear.bias':self.bias,
                    'clf_state':{'fc.weight':self.weight,'fc.bias':self.bias},'enc_state':{},'prior_state':{},
                    'dec_x_state':{},'dec_y_state':{},'in_dim':2,'num_classes':2,'epoch':self.checkpoint_epoch,
                    'val_acc':self.best_val,'val_loss':self.best_nll,'note':'synthetic'}
            else:
                payload={'fc.weight':self.weight,'fc.bias':self.bias}
            torch.save(payload,self.run/'best.ckpt')
            if method in ('fl_c','nl_c','fs_c'):
                torch.save({'fc.weight':torch.zeros(2,2),'fc.bias':torch.tensor([1.,0.])},self.run/'last.ckpt')
        write_json(self.run/'metrics.json',self.metrics)

    def refresh_inputs(self):
        torch.save(self.cache,self.cache_path)
        self.obs['meta']['cache_sha256']=digest(self.cache_path)
        torch.save(self.obs,self.obs_path)
        self.manifest.update(cache_sha256=digest(self.cache_path),obs_sha256=digest(self.obs_path),
            fingerprints={'cache_sha256':digest(self.cache_path),'obs_sha256':digest(self.obs_path)})
        write_json(self.run/'paper_run.json',self.manifest)
        self.descriptor={'run_dir':str(self.run),'cache':str(self.cache_path),'source':self.source,
            'method':self.method,'epochs':100,'expected_identity':self.identity,
            'expected_fingerprint':self.expected_fingerprint,'cache_sha256':digest(self.cache_path),
            'obs_sha256':digest(self.obs_path),'obs':str(self.obs_path)}

    def write_history(self):
        with self.history_path.open('w',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=self.history[0])
            writer.writeheader();writer.writerows(self.history)

    def write_metrics(self):
        write_json(self.run/'metrics.json',self.metrics)


class ReceiptValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        spec=importlib.util.spec_from_file_location('paper_followup_under_test',ROOT/'.autodl/paper_baseline_followup.py')
        cls.wrapper=importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.wrapper)
        if not hasattr(cls.wrapper,'validate_output_cpu'):
            raise RuntimeError('Wrapper validate_output_cpu interface is not ready')

    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory(prefix='fcl_receipt_artifacts_')
        self.directory=Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def validate(self,run):
        proof=self.wrapper.validate_output_cpu(run.descriptor)
        self.assertIsInstance(proof,dict)
        return proof

    def sealed(self,run):
        args=SimpleNamespace(output_root=run.root)
        runner=SimpleNamespace(job_identity=lambda job:run.identity)
        plan={'wrapper_sha256':'b'*64,'runner_source_sha256':'c'*64,'git_commit':'d'*40}
        receipt={'schema_version':1,'job_key':run.identity,'fingerprint':run.expected_fingerprint,
                 **plan,'run_dir':str(run.run.resolve()),'files':self.wrapper.output_inventory(run.run),
                 'verification':self.validate(run)}
        receipt_path=run.root/'followup_receipts/receipt.json'
        receipt_path.parent.mkdir(parents=True)
        write_json(receipt_path,receipt)
        record={'receipt_path':str(receipt_path),'receipt_sha256':digest(receipt_path),
                'run_dir':str(run.run.resolve()),'metrics_sha256':digest(run.run/'metrics.json')}
        return args,runner,plan,run.identity,run.expected_fingerprint,record

    def resume(self,arguments):
        return self.wrapper.validate_resume_receipt(*arguments)

    def test_internal_ce_methods_valid(self):
        for method in ('fl_c','nl_c','fs_c'):
            with self.subTest(method=method):
                self.validate(SyntheticRun(self.directory/method,method))

    def test_l2b_last_max_tie_is_valid(self):
        self.validate(SyntheticRun(self.directory,'l2b'))

    def test_dld_actual_alternative_history_path_is_valid(self):
        self.validate(SyntheticRun(self.directory,'dld'))

    def test_csgn_raw_validation_features_are_valid(self):
        self.validate(SyntheticRun(self.directory,'csgn',raw_csgn=True))

    def test_csgn_rounded_first_tie_need_not_equal_payload_epoch(self):
        self.validate(SyntheticRun(self.directory,'csgn',rounded_csgn_tie=True))

    def test_zero_shot_clip_and_qwen_without_training_artifacts(self):
        for source in ('clip','qwen'):
            with self.subTest(source=source):
                self.validate(SyntheticRun(self.directory/source,'zero_shot',source))

    def test_missing_best_checkpoint_rejected(self):
        run=SyntheticRun(self.directory)
        (run.run/'best.ckpt').unlink()
        with self.assertRaises(ValueError): self.validate(run)

    def test_missing_internal_last_checkpoint_rejected(self):
        run=SyntheticRun(self.directory)
        (run.run/'last.ckpt').unlink()
        with self.assertRaises(ValueError): self.validate(run)

    def test_short_history_rejected(self):
        run=SyntheticRun(self.directory)
        run.history.pop();run.write_history()
        with self.assertRaises(ValueError): self.validate(run)

    def test_duplicate_epoch_rejected(self):
        run=SyntheticRun(self.directory)
        run.history[-1]['epoch']=99;run.write_history()
        with self.assertRaises(ValueError): self.validate(run)

    def test_finite_but_wrong_test_accuracy_rejected(self):
        run=SyntheticRun(self.directory)
        run.metrics['test']['accuracy']=.25;run.write_metrics()
        with self.assertRaises(ValueError): self.validate(run)

    def test_checkpoint_not_maximum_validation_rejected(self):
        run=SyntheticRun(self.directory)
        for row in run.history: row['val_acc']=.5
        run.write_history()
        run.metrics.update(best_epoch=1,best_val_accuracy=.5);run.write_metrics()
        with self.assertRaises(ValueError): self.validate(run)

    def test_nonfinite_history_rejected(self):
        run=SyntheticRun(self.directory)
        run.history[10]['val_loss']=float('nan');run.write_history()
        with self.assertRaises(ValueError): self.validate(run)

    def test_invalid_classifier_shape_rejected(self):
        run=SyntheticRun(self.directory)
        torch.save({'fc.weight':torch.eye(3),'fc.bias':torch.zeros(3)},run.run/'best.ckpt')
        with self.assertRaises(ValueError): self.validate(run)

    def test_zero_shot_qwen_missing_test_source_metadata_rejected(self):
        run=SyntheticRun(self.directory,'zero_shot','qwen')
        del run.cache['annotation_metadata']['test']
        run.refresh_inputs()
        with self.assertRaises(ValueError): self.validate(run)

    def test_incomplete_manifest_rejected(self):
        run=SyntheticRun(self.directory)
        run.manifest['status']='running';write_json(run.run/'paper_run.json',run.manifest)
        with self.assertRaises(ValueError): self.validate(run)

    def test_output_inventory_detects_finite_metric_edit(self):
        run=SyntheticRun(self.directory)
        before=self.wrapper.output_inventory(run.run)
        run.metrics['test']['accuracy']=.25;run.write_metrics()
        after=self.wrapper.output_inventory(run.run)
        self.assertNotEqual(before,after)

    def test_sealed_receipt_valid_resume(self):
        run=SyntheticRun(self.directory)
        self.assertEqual(self.resume(self.sealed(run)),run.run.resolve())

    def test_resume_missing_receipt_rejected(self):
        arguments=self.sealed(SyntheticRun(self.directory))
        del arguments[-1]['receipt_path']
        with self.assertRaises(ValueError): self.resume(arguments)

    def test_resume_wrong_fingerprint_even_with_updated_receipt_hash_rejected(self):
        arguments=self.sealed(SyntheticRun(self.directory))
        record=arguments[-1]
        receipt=read_json(record['receipt_path'])
        receipt['fingerprint']='e'*64
        write_json(record['receipt_path'],receipt)
        record['receipt_sha256']=digest(record['receipt_path'])
        with self.assertRaises(ValueError): self.resume(arguments)

    def test_resume_modified_receipt_without_updated_hash_rejected(self):
        arguments=self.sealed(SyntheticRun(self.directory))
        record=arguments[-1]
        receipt=read_json(record['receipt_path'])
        receipt['fingerprint']='e'*64
        write_json(record['receipt_path'],receipt)
        with self.assertRaises(ValueError): self.resume(arguments)

    def test_resume_finite_metric_tampering_rejected(self):
        run=SyntheticRun(self.directory)
        arguments=self.sealed(run)
        run.metrics['test']['accuracy']=.25;run.write_metrics()
        with self.assertRaises(ValueError): self.resume(arguments)

    def test_resume_missing_checkpoint_rejected(self):
        run=SyntheticRun(self.directory)
        arguments=self.sealed(run)
        (run.run/'best.ckpt').unlink()
        with self.assertRaises(ValueError): self.resume(arguments)

    def test_resume_added_unsealed_file_rejected(self):
        run=SyntheticRun(self.directory)
        arguments=self.sealed(run)
        (run.run/'extra.txt').write_text('not in the sealed result',encoding='utf-8')
        with self.assertRaises(ValueError): self.resume(arguments)

    def test_resume_incomplete_epoch_proof_rejected(self):
        arguments=self.sealed(SyntheticRun(self.directory))
        record=arguments[-1]
        receipt=read_json(record['receipt_path'])
        receipt['verification']['completed_epochs']=99
        write_json(record['receipt_path'],receipt)
        record['receipt_sha256']=digest(record['receipt_path'])
        with self.assertRaises(ValueError): self.resume(arguments)

    def test_two_interruptions_preserve_later_completed_receipts(self):
        planned=[{'id':f'job-{index}','status':'pending','command':['reviewed.py',str(index)],
                  'preflight_command':['reviewed.py','--preflight',str(index)]} for index in range(3)]
        prior={f'job-{index}':{'id':f'job-{index}','status':'completed',
                   'run_dir':f'/runs/job-{index}','receipt_path':f'/receipts/job-{index}.json',
                   'receipt_sha256':str(index)*64,'metrics_sha256':str(index+1)*64,
                   'fingerprint':str(index+2)*64,'fingerprints':{'cache_sha256':'f'*64},
                   'completed_at':f'completed-{index}'} for index in (0,2)}
        # An interrupted, unfinished middle job must be rebuilt from the plan.
        prior['job-1']={'id':'job-1','status':'running','receipt_path':'not-completed.json'}
        expected={key:{field:value[field] for field in ('run_dir','receipt_path','receipt_sha256',
                      'metrics_sha256','fingerprint','fingerprints','completed_at')}
                  for key,value in prior.items() if value['status']=='completed'}
        recovered=self.wrapper.restore_completed_records(planned,prior)
        self.assertEqual(recovered[1]['status'],'pending')
        self.assertNotIn('receipt_path',recovered[1])
        status_file=self.directory/'interrupted_followup_status.json'
        for interruption in range(2):
            # The traversal has only touched its first completed job when it stops.
            recovered[0]['resumed_from_receipt']=True
            self.wrapper.save(status_file,{'status':'running','jobs':recovered,
                                          'current_job':'job-0','interruption':interruption})
            persisted=self.wrapper.read_json(status_file)
            recovered=self.wrapper.restore_completed_records(planned,
                {record['id']:record for record in persisted['jobs']})
            indexed={record['id']:record for record in recovered}
            for key,fields in expected.items():
                self.assertEqual(indexed[key]['status'],'completed')
                self.assertEqual({field:indexed[key][field] for field in fields},fields)
            self.assertEqual(indexed['job-2']['receipt_path'],'/receipts/job-2.json')

    def test_two_restores_keep_reviewed_commands_instead_of_prior_commands(self):
        planned=[{'id':f'job-{index}','status':'pending','dataset':'reviewed-dataset','source':'clip',
                  'method':'fl_c','cache':'reviewed-cache.pt','observed_labels':'reviewed-prototype.pt',
                  'command':['reviewed_runner.py','--epochs','100','--seed','42'],
                  'preflight_command':['reviewed_runner.py','--preflight']} for index in range(2)]
        original=json.loads(json.dumps(planned))
        prior={record['id']:{'id':record['id'],'status':'completed','dataset':'stale-dataset',
                    'source':'qwen','method':'fcl','cache':'stale-cache.pt',
                    'command':['stale.py','--epochs','1'],'preflight_command':['stale.py','--execute'],
                    'receipt_path':f"/receipts/{record['id']}.json",'receipt_sha256':'a'*64,
                    'metrics_sha256':'b'*64,'run_dir':f"/runs/{record['id']}"}
               for record in planned}
        for _ in range(2):
            recovered=self.wrapper.restore_completed_records(planned,prior)
            persisted=json.loads(json.dumps({'jobs':recovered}))
            prior={record['id']:record for record in persisted['jobs']}
            for index,record in enumerate(recovered):
                for field in ('dataset','source','method','cache','observed_labels','command','preflight_command'):
                    self.assertEqual(record[field],planned[index][field])
                self.assertEqual(record['status'],'completed')
                self.assertEqual(record['receipt_path'],f"/receipts/job-{index}.json")
        self.assertEqual(planned,original)

if __name__=='__main__':
    unittest.main(verbosity=2)
