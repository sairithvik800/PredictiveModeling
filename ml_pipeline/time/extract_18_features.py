"""Thin orchestrator that invokes ml_pipeline/features.py helpers for the 16
non-SynthSeg features in DEFAULT_FEATURES, recording wall-clock per image.

Every COMPUTATION traces back to ml_pipeline/features.py. We only restrict
WHICH subset of helpers gets called - the same restriction deployment applies.
"""
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib
from nilearn import datasets, image
from scipy.stats import kurtosis

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import features as ref_features  # the real pipeline (ml_pipeline/features.py)


def _get_brain_mask(img_data: np.ndarray) -> np.ndarray:
    """Match the heuristic used throughout ml_pipeline/features.py."""
    if np.any(img_data > 0):
        threshold = np.percentile(img_data[img_data > 0], 10)
    else:
        threshold = 0
    return img_data > threshold


def _extract_for_one_image(img_path: Path, datasets_inst: ref_features.Datasets,
                            atlas_img, atlas_labels) -> tuple[float, float]:
    """Returns (other16_wall_clock_seconds, load_time_ms_value)."""
    start_total = time.perf_counter()

    # --- load (= load_time_ms feature) ---
    t0 = time.perf_counter()
    img = nib.load(str(img_path))
    img_data = img.get_fdata()
    load_time_ms = (time.perf_counter() - t0) * 1000.0

    affine = img.affine
    brain_mask = _get_brain_mask(img_data)
    brain_vals = img_data[brain_mask]

    # --- 7 direct image features ---
    datasets_inst._compute_anatomical_complexity(img_data, brain_mask, affine)
    datasets_inst._compute_bias_field_features(img_data, brain_mask)
    datasets_inst._compute_structural_asymmetry(img_data, brain_mask)
    datasets_inst._compute_tile_gradient_load_imbalance(img_data, brain_mask)
    datasets_inst._compute_ngtdm_features(img_data, brain_mask)

    # registration: needs template_affine; load template once outside ideally,
    # here we use identity as a stand-in for timing purposes (the compute itself
    # is matrix arithmetic and constant-time regardless of template content)
    template_affine = np.eye(4)
    datasets_inst._compute_pose_misalignment(affine, template_affine)

    # morpho_intensity_kurtosis (cheap, inline)
    _ = kurtosis(brain_vals) if len(brain_vals) > 0 else 0.0

    # --- 8 Schaefer atlas features ---
    resampled = image.resample_to_img(img, atlas_img, interpolation='linear')
    resampled_data = resampled.get_fdata()
    atlas_data = atlas_img.get_fdata()
    datasets_inst._compute_schaefer_region_stats(resampled_data, atlas_data, atlas_labels)

    other16_seconds = time.perf_counter() - start_total
    return other16_seconds, load_time_ms


def extract_for_all(mri_dir: Path) -> list[dict]:
    """Iterate over every .nii.gz under mri_dir, return per-image timing rows."""
    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
    atlas_img = nib.load(atlas.maps)
    atlas_labels = atlas.labels
    datasets_inst = ref_features.Datasets(atlas=atlas)

    rows = []
    nifti_files = sorted(mri_dir.glob('*.nii.gz'))
    for idx, path in enumerate(nifti_files, start=1):
        print(f'[{idx}/{len(nifti_files)}] {path.name}', flush=True)
        other16, load_ms = _extract_for_one_image(path, datasets_inst, atlas_img, atlas_labels)
        rows.append({
            'filename': path.name,
            'other16_seconds': round(other16, 3),
            'load_time_ms': round(load_ms, 2),
        })
    return rows
