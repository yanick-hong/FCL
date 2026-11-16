"""Standard-library tests for independent result auditing; torch is not loaded."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import statistics
import tempfile
import unittest

MODULE_PATH = Path(__file__).with_name('audit_ce_results.py')
SPEC = importlib.util.spec_from_file_location('ce_audit', MODULE_PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.base = Path(self.temp.name)
        self.study = self.base / 'study'
        protocol = {
            'datasets': ['cifar100'], 'methods': list(AUDIT.METHODS),
            'learning_rates': [.001, .002, .004], 'weight_decay': .0001,
            'epochs': 100, 'patience': 101, 'batch_size': 1024,
            'data_loader_workers': 2, 'scheduler': 'CosineAnnealingLR(T_max=100)',
            'tuning_seed': 42, 'additional_seeds': [43, 44],
            'selection': 'highest best_val_accuracy; ties choose smallest learning rate',
            'threads': 2, 'python': '/python', 'root': str(self.base / 'project'),
        }
        self.plan = {'protocol': protocol, 'training_count': 10,
                     'source': {'git_commit': 'a' * 40, 'source_sha256': {}, 'wrapper_sha256': 'b' * 64}}
        dataset_audit = {
            'cache': str(self.base / 'cache.pt'), 'observed_labels': str(self.base / 'obs.pt'),
            'hashes': {'cache_sha256': 'c' * 64, 'obs_sha256': 'd' * 64},
            'verified_count': 4, 'rho': .4, 'split_counts': {'train': 10, 'val': 6, 'test': 12}, 'classes': 3,
        }
        self.audits = {'cifar100': dataset_audit}
        runtime = {'python': 'fixture', 'torch': 'not imported'}
        self.binding = AUDIT.fingerprint({'plan': self.plan, 'inputs': self.audits, 'runtime': runtime})
        write(self.study / 'plan.json', self.plan)
        write(self.study / 'preflight.json', {'datasets': self.audits, 'runtime': runtime})
        write(self.study / 'study_binding.json', {'binding': self.binding, 'runtime': runtime})
        all_rows, finals, selections, summaries = [], [], [], []
        for method in AUDIT.METHODS:
            for lr, val, test in [(.001, .7, .65), (.002, .8, .66), (.004, .75, .9)]:
                row = self.make_trial(method, lr, 42, val, test)
                all_rows.append(row)
                if lr == .002:
                    finals.append(row)
            selections.append({'dataset': 'cifar100', 'method': method, 'learning_rate': .002,
                               'best_val_accuracy': .8, 'criterion': protocol['selection']})
            for seed, val, test in [(43, .81, .67), (44, .79, .65)]:
                row = self.make_trial(method, .002, seed, val, test)
                all_rows.append(row)
                finals.append(row)
            summaries.append({'dataset': 'cifar100', 'method': method,
                              **AUDIT.expected_summary([row for row in finals if row['method'] == method])})
        self.results = {'selection_rule': protocol['selection'], 'selected': selections,
                        'all_trials': all_rows, 'final_runs': finals, 'summary': summaries}
        write(self.study / 'results.json', self.results)
        write(self.study / 'status.json', {'status': 'completed', 'binding': self.binding, 'completed_trials': 10})

    def tearDown(self):
        self.temp.cleanup()

    def make_trial(self, method, lr, seed, validation, test):
        protocol = self.plan['protocol']
        run = self.study / 'trials/cifar100' / method / f'lr-{lr:g}' / f'seed-{seed}'
        run.mkdir(parents=True)
        audit = self.audits['cifar100']
        binding = {'study_binding': self.binding, 'dataset': 'cifar100', 'method': method,
                   'learning_rate': lr, 'seed': seed, 'input_hashes': audit['hashes']}
        command = AUDIT.expected_command(protocol, 'cifar100', method, lr, seed, str(run), audit)
        args = {'method': method, 'dataset': 'cifar100', 'cache': audit['cache'],
                'obs_labels': audit['observed_labels'], 'run_dir': str(run), 'epochs': 100,
                'patience': 101, 'batch_size': 1024, 'lr': lr, 'weight_decay': .0001, 'seed': seed}
        selection = {'args': args, 'budget_verified_count': 4, 'selected_count': 4}
        metrics = {'method': method, 'dataset': 'cifar100', 'selection': selection,
                   'verified_count': 4, 'rho': .4, 'epochs_completed': 100,
                   'best_val_accuracy': validation, 'best_epoch': 50,
                   'test': {'accuracy': test, 'valid_count': 12, 'total_count': 12}}
        write(run / 'metrics.json', metrics)
        write(run / 'launch.json', {'binding': binding, 'command': command})
        write(run / 'run_metadata.json', {'args': args, 'command': command[1:],
                                         'script': 'src/contrast/train_revision_baselines.py',
                                         'git_commit': 'a' * 40, 'method': method, 'dataset': 'cifar100'})
        for name in ['console.log', 'best.ckpt', 'last.ckpt', 'config.yaml']:
            (run / name).write_text('fixture artifact', encoding='utf-8')
        (run / 'logs').mkdir()
        with (run / 'logs/val_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=['epoch', 'accuracy'])
            writer.writeheader()
            writer.writerows({'epoch': epoch, 'accuracy': validation if epoch in (50, 51) else validation - .01}
                            for epoch in range(1, 101))
        result = {'dataset': 'cifar100', 'method': method, 'learning_rate': lr, 'seed': seed,
                  'best_val_accuracy': validation, 'test_accuracy': test, 'best_epoch': 50,
                  'run_dir': str(run), 'metrics_sha256': AUDIT.sha(run / 'metrics.json')}
        write(run / 'seal.json', {'binding': binding, 'result': result, 'files': AUDIT.file_manifest(run)})
        return result

    def test_complete_standard_library_audit(self):
        report = AUDIT.audit_study(self.study)
        self.assertEqual((report['sealed_trials'], report['final_runs'], report['summaries']), (10, 6, 2))

    def test_seal_result_change_is_detected_even_when_files_are_unchanged(self):
        path = next((self.study / 'trials').rglob('seal.json'))
        seal = AUDIT.read(path)
        seal['result']['test_accuracy'] = .999
        write(path, seal)
        with self.assertRaisesRegex(AUDIT.AuditError, 'independently rebuilt metrics'):
            AUDIT.audit_study(self.study)

    def test_unsealed_artifact_change_is_detected(self):
        path = next((self.study / 'trials').rglob('best.ckpt'))
        path.write_text('changed', encoding='utf-8')
        with self.assertRaisesRegex(AUDIT.AuditError, 'sealed files'):
            AUDIT.audit_study(self.study)

    def test_results_reference_must_match_the_seal(self):
        self.results['all_trials'][0]['best_epoch'] = 51
        write(self.study / 'results.json', self.results)
        with self.assertRaisesRegex(AUDIT.AuditError, 'mismatched sealed result'):
            AUDIT.audit_study(self.study)

    def test_test_winner_cannot_replace_validation_winner(self):
        self.results['selected'][0]['learning_rate'] = .004
        self.results['selected'][0]['best_val_accuracy'] = .75
        write(self.study / 'results.json', self.results)
        with self.assertRaisesRegex(AUDIT.AuditError, 'not selected by validation'):
            AUDIT.audit_study(self.study)

    def test_summary_sd_must_be_sample_sd(self):
        self.results['summary'][0]['sample_sd_percent'] = statistics.pstdev([66, 67, 65])
        write(self.study / 'results.json', self.results)
        with self.assertRaisesRegex(AUDIT.AuditError, 'sample_sd_percent'):
            AUDIT.audit_study(self.study)

    def test_partial_allows_sealed_results_ahead_of_json_snapshot(self):
        self.results['all_trials'] = []
        self.results['final_runs'] = []
        self.results['selected'] = []
        self.results['summary'] = []
        write(self.study / 'results.json', self.results)
        report = AUDIT.audit_study(self.study, allow_partial=True)
        self.assertEqual(report['sealed_not_yet_referenced'], 10)
        with self.assertRaises(AUDIT.AuditError):
            AUDIT.audit_study(self.study)

    def test_partial_allows_first_seal_before_initial_results_write(self):
        (self.study / 'results.json').unlink()
        report = AUDIT.audit_study(self.study, allow_partial=True)
        self.assertEqual(report['referenced_trials'], 0)


if __name__ == '__main__':
    unittest.main()
