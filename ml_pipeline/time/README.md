# Computational Overhead Timing Benchmark

CPU-based timing benchmark for the SLANT makespan-prediction pipeline's deployment cost. Produces a paper-ready table covering feature extraction, model training, and inference for the 18-feature deployment subset.

## Environment

Two conda environments are involved due to dependency conflicts in the project:

- `slant` — feature-extraction deps (nibabel, nilearn, skimage, SimpleITK, pywt, antropy). Use for `--phase synthseg`, `--phase others`, and hardware capture.
- `.venv` — xgboost (required by `ml_pipeline/models.py`). Use for `--phase train`.

`--build-table` is pure file I/O and works from either env, but prefer running it from `slant` so the hardware JSON it (re-)captures has the full CPU/RAM info.

## Quickstart

```powershell
# Phase 1: SynthSeg (one-time setup, then ~5.4 hr overnight on CPU)
# Prereqs: install SynthSeg per its README, then set SYNTHSEG_PREDICT_SCRIPT
# and SYNTHSEG_OUTPUT_DIR in ml_pipeline/constants.py to real paths.
# Env gotcha: if .venv is already active, `conda activate slant` does NOT remove
# it from PATH. Either `deactivate` first, or use `conda run -n slant ...` below.
conda activate slant
python ml_pipeline/time/timing_benchmark.py --phase synthseg   # ~5.4 hr (measured: 220s/image x 88 on i7-13700HX, robust+cpu+crop192)

# Phase 2: other 16 features (a few minutes)
conda activate slant
python ml_pipeline/time/timing_benchmark.py --phase others     # ~5 min

# Phase 3: training + inference (a few minutes)
conda deactivate
.venv\Scripts\Activate.ps1
python ml_pipeline/time/timing_benchmark.py --phase train      # ~5 min

# Build the paper table from whatever CSVs exist
conda deactivate
conda activate slant
python ml_pipeline/time/timing_benchmark.py --build-table      # seconds
```

## What gets measured

| Phase | What it times | Output |
|---|---|---|
| Phase 1 (`synthseg`) | SynthSeg CLI subprocess wall-clock per image (88 images) | `outputs/timing_synthseg.csv` |
| Phase 2 (`others`) | 16 non-SynthSeg features per image (88 images) via thin wrappers over `ml_pipeline/features.py` | `outputs/timing_other_features.csv` |
| Phase 3 (`train`) | Nested-CV training wall-clock + per-call inference microseconds for each model in `DEFAULT_MODELS` | `outputs/timing_train_predict.csv` |
| Build table | Aggregates the above + hardware metadata into a paper-ready table | `outputs/timing_paper_table.{csv,md}` |

## Settings

- SynthSeg flags: `robust=True, fast=False, use_cpu=True, crop_size=192` - matches the production extraction that generated the dataset. Faster modes (`--fast`) deferred to future work; see spec for rationale.
- Nested CV: 5-outer x 3-inner with `random_state=42` (hardcoded in `model_eval_pipeline.nested_cv_evaluation`).
- Inference: median of 1000 `model.predict(X_18)` calls per model.

## Hardware

`outputs/timing_hardware.json` captures CPU/RAM/OS/Python/library versions whenever a phase is run. For the richest capture (with `cpu_brand` and `ram_total_gb`), run a phase from the `slant` env (which has `psutil` and `py-cpuinfo` installed).

## Setup prereqs for Phase 1 (SynthSeg)

This is the only phase that requires extra installation:

1. Clone SynthSeg from its repository (BBillot/SynthSeg or freesurfer/SynthSeg)
2. Install in a Python 3.8 conda env named `synthseg_38` per SynthSeg's own README
3. Update two constants in `ml_pipeline/constants.py`:
   - `SYNTHSEG_PREDICT_SCRIPT` = full path to `SynthSeg/scripts/commands/SynthSeg_predict.py`
   - `SYNTHSEG_OUTPUT_DIR` = directory where outputs should land (or use an existing seg-outputs directory if one already exists from prior runs)

## Notes for the paper

- SynthSeg published benchmark: ~15s/image on GPU vs ~60s on CPU (Billot et al., 2023). Our CPU measurements should land near 60s; deployment to a GPU host would reduce SynthSeg cost ~4x.
- "Other 16 features" reflects deployment-realistic shared computation (brain mask and MRI load are computed once per image, shared across features).
- Inference time excludes feature extraction by design (per spec); add the extraction rows if reporting end-to-end.
- The MRIs at `ml_pipeline/csv_files/inputs/raiders_data_3d/` are 4D BOLD volumes; the SynthSeg wrapper handles whatever volume extraction is required. The reported "SynthSeg wall-clock" therefore includes that internal step.

## Re-running just the table

If only `--build-table` is needed (e.g. after manually editing a CSV), it's pure file I/O:

```powershell
conda activate slant
python ml_pipeline/time/timing_benchmark.py --build-table
```

## Files in this directory

| File | Purpose |
|---|---|
| `timing_benchmark.py` | CLI orchestrator (`--phase`, `--build-table`) |
| `hardware_info.py` | Capture CPU/RAM/OS/lib versions to JSON |
| `build_table.py` | Aggregate 3 timing CSVs + hardware.json into paper_table.{csv,md} |
| `phase_synthseg.py` | Phase 1: SynthSeg subprocess timing (deferred; run separately) |
| `phase_others.py` | Phase 2: 16 non-SynthSeg features timing |
| `phase_train.py` | Phase 3: nested-CV training + inference timing |
| `extract_18_features.py` | Thin wrapper invoking `ml_pipeline/features.py` helpers for the 16 non-SynthSeg features |
| `test_build_table.py` | pytest tests for the table builder |
| `outputs/` | Generated timing CSVs, JSON, and paper table |
| `fixtures/` | Sample CSVs used by tests |
| `synthseg_output_timing/` | Fresh SynthSeg outputs from Phase 1 (gitignored) |

## Spec

`ml_pipeline/docs/superpowers/specs/2026-05-16-computational-overhead-design.md`
