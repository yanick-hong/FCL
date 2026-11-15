# AutoDL CLIP main-suite results (seed 42)

This snapshot records measured test accuracy from `/root/autodl-tmp/outputs/paper` on 2026-11-15. Values are percentages; the reference column comes from `configs/paper_targets.yaml` and is not used as a measured result. The protocol uses CLIP ViT-L/14 frozen features, `rho=0.3`, 100 epochs, and the dataset-specific `P/K` values.

| Dataset | Method | Measured | Reference | Status |
|---|---:|---:|---:|---|
| CIFAR-100 | Zero-shot | 75.59 | 75.63 | complete |
| CIFAR-100 | FL-C | 80.91 | 76.70 | complete |
| CIFAR-100 | NL-C | 76.46 | 76.37 | complete |
| CIFAR-100 | FS-C | 84.40 | 85.20 | complete |
| CIFAR-100 | L2B | 81.35 | 82.51 | complete |
| CIFAR-100 | DLD | 82.53 | 82.47 | complete |
| CIFAR-100 | CSGN | 76.65 | 82.74 | complete |
| CIFAR-100 | FCL | 83.41 | 84.26 | complete |
| Caltech-101 | Zero-shot | 85.08 | 92.39 | complete |
| Caltech-101 | FL-C | 90.29 | 91.18 | complete |
| Caltech-101 | NL-C | 86.30 | 87.15 | complete |
| Caltech-101 | FS-C | 96.27 | 97.29 | complete |
| Caltech-101 | L2B | 47.26 | 74.21 | complete |
| Caltech-101 | DLD | 68.00 | 94.93 | complete |
| Caltech-101 | CSGN | 24.38 | 84.87 | complete |
| Caltech-101 | FCL | 95.67 | 95.28 | complete |
| EuroSAT | Zero-shot | 43.74 | 43.00 | complete |
| EuroSAT | FL-C | 88.67 | 91.11 | complete |
| EuroSAT | NL-C | 49.74 | 55.04 | complete |
| EuroSAT | FS-C | 93.74 | 96.44 | complete |
| EuroSAT | L2B | 71.63 | 74.15 | complete |
| EuroSAT | DLD | 73.41 | 75.59 | complete |
| EuroSAT | CSGN | 79.96 | 79.56 | complete |
| EuroSAT | FCL | 92.81 | 92.19 | complete |
| Tiny-ImageNet-200 | Zero-shot | 74.96 | 70.17 | complete |
| Tiny-ImageNet-200 | FL-C | 82.23 | 80.11 | complete |
| Tiny-ImageNet-200 | NL-C | 76.44 | 76.37 | complete |
| Tiny-ImageNet-200 | FS-C | 85.49 | 83.49 | complete |
| Tiny-ImageNet-200 | L2B | 82.75 | 79.22 | complete |
| Tiny-ImageNet-200 | DLD | 83.43 | 78.76 | complete |
| Tiny-ImageNet-200 | CSGN | 83.30 | 80.42 | complete |
| Tiny-ImageNet-200 | FCL | 84.07 | 81.30 | complete |

Stanford Cars has no local cache because the official test labels file was unavailable on AutoDL. Food-101 is being downloaded from the official archive with resumable 8-way Range requests; its main-suite controls and learners will be added after extraction and cache validation.

The Caltech-101 comparison baselines are substantially below the published values, which is retained as an explicit protocol/result discrepancy rather than silently replacing the measured values.
