"""Meaningful pure-Python checks; fake training has no tensor/model dependency."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('fair_ce_runner', Path(__file__).with_name('fair_ce_runner.py'))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class FairStudyTests(unittest.TestCase):
    def test_validation_selection_and_deterministic_tie_break_ignore_test(self):
        trials = [
            {'learning_rate': .004, 'best_val_accuracy': .92, 'test_accuracy': .99},
            {'learning_rate': .002, 'best_val_accuracy': .94, 'test_accuracy': .2},
            {'learning_rate': .001, 'best_val_accuracy': .94, 'test_accuracy': .1},
        ]
        self.assertEqual(runner.select_trial(trials)['learning_rate'], .001)

    def test_seal_detects_artifact_mutation_and_wrong_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            (run / 'metrics.json').write_text('{"score":1}')
            binding = {'source': 'original'}
            runner.save(run / 'seal.json', {'binding': binding, 'files': runner.files_manifest(run), 'result': {'ok': True}})
            self.assertEqual(runner.verify_seal(run, binding), {'ok': True})
            with self.assertRaises(ValueError):
                runner.verify_seal(run, {'source': 'changed'})
            (run / 'metrics.json').write_text('{"score":2}')
            with self.assertRaises(ValueError):
                runner.verify_seal(run, binding)

    def test_fake_cli_jobs_two_keeps_all_trials_and_complete_final_seeds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            output = root / 'study'
            datasets = ['caltech101', 'eurosat']
            inputs = {}
            for name in datasets:
                cache = root / f'{name}.cache'
                obs = root / f'{name}.obs'
                cache.write_bytes(b'cache')
                obs.write_bytes(b'observed')
                inputs[name] = (cache, obs)

            def fake_preflight(args, dataset):
                cache, obs = inputs[dataset]
                return {'dataset': dataset, 'cache': str(cache), 'observed_labels': str(obs),
                        'hashes': {'cache_sha256': runner.sha(cache), 'obs_sha256': runner.sha(obs)},
                        'verified_count': 3, 'split_counts': {'test': 4}}

            def fake_train(cmd, **kwargs):
                values = dict(zip(cmd[2::2], cmd[3::2]))
                run = Path(values['--run-dir'])
                args = {k[2:].replace('-', '_'): v for k, v in values.items()}
                for name in ['epochs', 'patience', 'batch_size', 'seed']:
                    args[name] = int(args[name])
                for name in ['lr', 'weight_decay']:
                    args[name] = float(args[name])
                # Lowest LR has best validation but deliberately worst test.
                val = {.001: .9, .002: .8, .004: .7}[args['lr']]
                test = {.001: .3, .002: .7, .004: .99}[args['lr']] + (args['seed'] - 42) / 1000
                runner.save(run / 'metrics.json', {'method': args['method'], 'dataset': args['dataset'],
                    'selection': {'args': args}, 'verified_count': 3, 'epochs_completed': 100,
                    'best_val_accuracy': val, 'best_epoch': 1, 'test': {'accuracy': test, 'valid_count': 4}})
                for name in ('best.ckpt', 'last.ckpt', 'config.yaml', 'run_metadata.json'):
                    (run / name).write_text('fake artifact')
                (run / 'logs').mkdir()
                with (run / 'logs/val_metrics.csv').open('w', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=['epoch', 'accuracy'])
                    writer.writeheader()
                    writer.writerows({'epoch': e, 'accuracy': val} for e in range(1, 101))
                return types.SimpleNamespace(returncode=0)

            fake_torch = types.SimpleNamespace(set_num_threads=lambda n: None, __version__='fake', version=types.SimpleNamespace(cuda=None))
            fake_fcntl = types.SimpleNamespace(flock=lambda *args: None, LOCK_EX=1, LOCK_NB=2)
            argv = ['--root', str(root), '--output-root', str(output), '--datasets', *datasets,
                    '--device', 'cpu', '--threads', '2', '--jobs', '2', '--execute']
            with patch.dict(sys.modules, {'torch': fake_torch, 'fcntl': fake_fcntl}), \
                 patch.object(runner, 'source_manifest', return_value={'source': 'frozen'}), \
                 patch.object(runner, 'preflight', side_effect=fake_preflight), \
                 patch.object(runner, 'paths_for', side_effect=lambda args, dataset: inputs[dataset]), \
                 patch.object(runner.subprocess, 'run', side_effect=fake_train):
                runner.main(argv)
                result = runner.read(output / 'results.json')
                self.assertEqual(len(result['all_trials']), 20)
                self.assertEqual(len(result['summary']), 4)
                self.assertEqual(len(result['final_runs']), 12)
                self.assertTrue(all(row['learning_rate'] == .001 for row in result['selected']))
                self.assertTrue(all(row['seeds'] == [42, 43, 44] for row in result['summary']))
                self.assertEqual(runner.read(output / 'status.json')['status'], 'completed')
                # Full resume must reuse every sealed output, without invoking fake training.
                with patch.object(runner.subprocess, 'run', side_effect=AssertionError('Training must not repeat')):
                    runner.main(argv + ['--resume'])
                result_after = runner.read(output / 'results.json')
                self.assertEqual(len(result_after['all_trials']), 20)
                self.assertEqual(len(result_after['summary']), 4)
                self.assertTrue(all(row['learning_rate'] == .001 for row in result_after['selected']))


if __name__ == '__main__':
    unittest.main()
