# Legacy FCL-CVPR reference backup

This directory is a read-only working copy of legacy material from `F:\Documents\备份\FCL-CVPR`.

- `code_temp/`: legacy feature extraction, prototype/filter label generation, FCL/AUC training and baselines.
- `对比方法/`: legacy CSGN, DLD and L2B comparison scripts and papers.
- `Learning from filter-corrected lables/`: archive contents, when available (the original `.rar` is not modified).
- `MANIFEST.csv`: SHA-256 and byte-size inventory for provenance.

The runnable implementation should live under the version-controlled repository `src/`, `configs/`, and `scripts/`; this folder is a reference source only. Legacy scripts contain hard-coded paths and some missing `utils.*` dependencies, so they should be ported behind the current configuration rather than executed in place.
