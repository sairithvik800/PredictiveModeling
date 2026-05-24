# Computational Overhead - Paper Table

| Component | Per-image / per-call | Notes |
|---|---|---|
| SynthSeg subprocess (extraction) | 220.24s (IQR 219.47-221.04) | robust=True, fast=False, CPU; per-image wall-clock |
| Other 16 features (extraction) | 0.35s (IQR 0.34-0.36) | single MRI load shared across features |
| Total extraction per image | 220.6s | SynthSeg median + other-16 median |
| Training - Ridge (Polynomial) (nested CV, n=88) | 0.4s | total wall-clock incl. grid search |
| Training - Bayesian Ridge (Polynomial) (nested CV, n=88) | 0.3s | total wall-clock incl. grid search |
| Training - Huber Regression (nested CV, n=88) | 3.5s | total wall-clock incl. grid search |
| Training - SVR (RBF) (nested CV, n=88) | 0.6s | total wall-clock incl. grid search |
| Training - Random Forest (nested CV, n=88) | 240.3s | total wall-clock incl. grid search |
| Training - XGBoost (nested CV, n=88) | 26.8s | total wall-clock incl. grid search |
| Inference - Ridge (Polynomial) (1x18 vector) | 148.8 us | median over 1000 calls |
| Inference - Bayesian Ridge (Polynomial) (1x18 vector) | 148.7 us | median over 1000 calls |
| Inference - Huber Regression (1x18 vector) | 85.2 us | median over 1000 calls |
| Inference - SVR (RBF) (1x18 vector) | 105.9 us | median over 1000 calls |
| Inference - Random Forest (1x18 vector) | 17640.8 us | median over 1000 calls |
| Inference - XGBoost (1x18 vector) | 649.0 us | median over 1000 calls |

## Hardware

- CPU: 13th Gen Intel(R) Core(TM) i7-13700HX
- RAM: 31.75 GB
- OS: Windows-10-10.0.26200-SP0
- Python: 3.11.14

### Library versions

- SimpleITK: 2.5.5
- nibabel: 5.3.3
- nilearn: 0.13.0
- numpy: 2.3.5
- pandas: 2.3.3
- scipy: 1.16.3
- skimage: 0.26.0
- sklearn: 1.7.1
- tensorflow: not installed

## Notes

- SynthSeg published benchmark: ~15s/image on GPU vs ~60s on CPU (Billot et al., 2023). Our CPU measurements are consistent.
- Deployment to a GPU-equipped host would reduce SynthSeg cost ~4x per the published benchmark.