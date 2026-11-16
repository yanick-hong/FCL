# Fair CE runner handoff

Files are local tmp only. No source files, configurations, manuscript files, or remote files were changed by this agent.

## Commands

Upload `fair_ce_runner.py` outside active source directories, e.g. `/root/autodl-tmp/outputs/conf_ce_fair_recheck_20261116/_preparation/fair_ce_runner.py`.

```bash
/root/miniconda3/bin/python /root/autodl-tmp/outputs/conf_ce_fair_recheck_20261116/_preparation/fair_ce_runner.py --root /root/autodl-tmp --device cpu --threads 2 --jobs 3 --plan-only
/root/miniconda3/bin/python /root/autodl-tmp/outputs/conf_ce_fair_recheck_20261116/_preparation/fair_ce_runner.py --root /root/autodl-tmp --device cpu --threads 2 --jobs 3 --preflight
/root/miniconda3/bin/python /root/autodl-tmp/outputs/conf_ce_fair_recheck_20261116/_preparation/fair_ce_runner.py --root /root/autodl-tmp --device cpu --threads 2 --jobs 3 --execute --resume
```

The same command options must be used for plan, preflight, and execution. `--datasets caltech101` requires a different output root because it defines a different finite plan. Default is plan-only. Training always calls the unchanged original CLI with two DataLoader workers. CPU mode hides CUDA explicitly. OMP/MKL/OpenBLAS/NumExpr/vecLib thread budgets are bounded equally.

## Verified behavior

- Same three LR values for both methods on all five datasets; seed42 validation selects LR independently for each dataset/method, with lowest-LR tie-break. No test metric participates in selection.
- Selected LR is trained with seeds43/44. Seed42 selected run is included in the final three-seed summary. All 50 trial outputs and test metrics remain stored, including unsuccessful LR candidates.
- Cross-dataset parallelism only; each dataset runs its trials and selection sequentially. Global status/results writes use a mutex, run paths are distinct, and a POSIX process lock prevents duplicate study workers.
- Preflight reuses controlled input validation, checks original CLIP probabilities in train/val/test without fallback, and proves the observed label artifact belongs to the exact cache and sample order. It checks selected labels equal selected truth and unselected labels remain CLIP.
- CPU top-k confidence uses the exact same prior normalization and query count as the underlying CE CLI. Mask/label SHA, query count, classes, split counts, and corrected-label quality are recorded.
- Source/HEAD/wrapper and input SHA are checked before and after each trial. The plan, inputs, runtime, and wrapper bind every completed run. A seal covers every produced file; resume rejects changed files and rejects a different binding. Partial unsealed directories deliberately fail closed instead of overwriting evidence.
- Completion checks require 100 consecutive validation epochs, required output files, first-max validation checkpoint selection, expected arguments, equal query budgets, and complete labeled test count.
- `results.json` contains all trials, final runs, selections, and percentage summaries. `all_trials.csv` and `summary.csv` support later inspection. Summary gives sample SD and df=2 95% Student-t interval; uncertainty is conditional on fixed data, labels, mask, and chosen LR.

## Checks and limits

Three meaningful pure-Python tests pass in `test_fair_ce_runner.py`: validation-over-test selection and LR tie-break; changed SHA/wrong-binding resume rejection; two-dataset jobs2 fake CLI execution preserving all20 trials and12 final runs, plus sealed resume without retraining.

No real training or remote preflight was performed by this agent. Parent should run strict preflight on AutoDL before execution and may benchmark an isolated original-CLI Caltech run in a distinct pilot output directory. Current source tree remains guarded by other queues.

Caltech recovered cache has101 classes while historical main text uses102; all outputs must be described as separate controlled reruns. CPU results cannot be silently substituted for historical GPU numbers. The wrapper logs every trial's test result because the original CLI evaluates it after checkpoint selection; only validation fields are used for choosing LR. No target win count is enforced. No runtime benchmark claim is appropriate during concurrent workloads.

The existing CE CLI reports an unused `confidence_threshold` default in saved args, but that flag does not alter `clip_conf_ce`; the mask is lowest-top1-confidence top-k at the FCL prototype query count.
