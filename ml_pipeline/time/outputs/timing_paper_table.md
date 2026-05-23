# Computational Overhead - Paper Table

| Component | Per-image / per-call | Notes |
|---|---|---|
| SynthSeg subprocess (extraction) | 220.24s (IQR 219.47-221.04) | robust=True, fast=False, CPU; per-image wall-clock |
| Other 16 features (extraction) | 0.35s (IQR 0.34-0.35) | single MRI load shared across features |
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

- CPU: 
- RAM: 0.0 GB
- OS: Windows-11-10.0.26200-SP0
- Python: 3.13.9

### Library versions

- SimpleITK: not installed
- nibabel: 5.4.0
- nilearn: 0.13.1
- numpy: 2.4.2
- pandas: 3.0.1
- scipy: 1.17.0
- skimage: 0.26.0
- sklearn: 1.8.0
- tensorflow: not installed

## Notes

- SynthSeg published benchmark: ~15s/image on GPU vs ~60s on CPU (Billot et al., 2023). Our CPU measurements are consistent.
- Deployment to a GPU-equipped host would reduce SynthSeg cost ~4x per the published benchmark.