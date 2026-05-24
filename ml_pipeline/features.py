"""Deployment feature extractor for the SLANT makespan-prediction paper.

Computes the 18 features used by the deployed model (DEFAULT_FEATURES in
ml_pipeline/main.py). The original 1,508-feature exploratory extractor that
produced csv_files/inputs/features_data.csv is archived in the dev repo
<DEV_REPO_URL>; see paper Section 1 for details. The feature selection
pipeline that narrowed 1,508 -> 18 lives in
ml_pipeline/feature_selection_pipeline.py and is fully reproducible against
features_data.csv (paper Section 2).

Run standalone:
    python ml_pipeline/features.py [<mri_dir>]

TODO(heagen): replace <DEV_REPO_URL> with the actual archive repo URL.
"""
import argparse
import os
import gc
import time
import subprocess
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from scipy.ndimage import center_of_mass, binary_erosion, label, sobel, gaussian_filter
from scipy.fft import fftn, fftshift
from scipy.signal import find_peaks
from scipy.stats import kurtosis, median_abs_deviation, moment
from skimage.morphology import ball
from skimage.filters.rank import entropy as skimage_entropy
from nilearn import datasets, image
from constants import SYNTHSEG_CONDA_ENV, SYNTHSEG_PREDICT_SCRIPT, MRI_3D_IMAGE_DATA, FEATURES_CSV_OUTPUT, SYNTHSEG_OUTPUT_DIR

class Datasets:
    def __init__(self, atlas=None):
        """Atlas param retained for backward compat with consumers (e.g.,
        ml_pipeline/time/extract_18_features.py which passes a fetched
        Schaefer atlas). The kept compute methods do not read self attrs.
        """
        self._atlas = atlas

    def _check_synthseg_environment(self):
        """Check if SynthSeg conda environment exists."""
        result = subprocess.run(['conda', 'env', 'list'],
                                capture_output=True, text=True, shell=True)
        return SYNTHSEG_CONDA_ENV in result.stdout

    def _run_synthseg_cli(self, input_path, seg_path, vol_path, qc_path,
                          robust=True, fast=False, use_cpu=False, crop_size=False,
                          timeout=600, max_retries=3):
        """Run SynthSeg via CLI in Python 3.8 conda environment.

        Args:
            timeout: Maximum seconds to wait for SynthSeg (default 600 = 10 min)
            max_retries: Number of retry attempts if TensorFlow initialization fails
        """
        cli_script = SYNTHSEG_PREDICT_SCRIPT

        # Build flags
        flags = []
        if robust:
            flags.append('--robust')
        if fast:
            flags.append('--fast')
        if use_cpu:
            flags.append('--cpu')
        if crop_size:
            flags.extend(['--crop', str(crop_size)])
        flags_str = ' '.join(flags)

        # Build command with CUDA explicitly disabled (-1) for CPU mode
        if use_cpu:
            cmd = f'set CUDA_VISIBLE_DEVICES=-1 && conda run -n {SYNTHSEG_CONDA_ENV} --no-capture-output python "{cli_script}" --i "{input_path}" --o "{seg_path}" --vol "{vol_path}" --qc "{qc_path}" {flags_str}'
        else:
            cmd = f'conda run -n {SYNTHSEG_CONDA_ENV} --no-capture-output python "{cli_script}" --i "{input_path}" --o "{seg_path}" --vol "{vol_path}" --qc "{qc_path}" {flags_str}'

        env = os.environ.copy()

        # Retry logic for intermittent TensorFlow initialization failures
        last_error = None
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    wait_time = 5 * (attempt + 1)  # 10s, 15s for retries
                    print(f"    Retry {attempt + 1}/{max_retries} after {wait_time}s wait...")
                    time.sleep(wait_time)
                    gc.collect()

                result = subprocess.run(cmd, capture_output=True, text=True, shell=True,
                                        env=env, timeout=timeout)

                if result.returncode == 0:
                    # Verify segmentation file was actually created
                    if os.path.exists(seg_path):
                        return True
                    else:
                        last_error = f"Segmentation file not found: {seg_path}"
                        continue

                # Check if it's a TF initialization failure (empty stdout = crashed during import)
                if not result.stdout.strip() and 'tensorflow' in result.stderr.lower():
                    last_error = "TensorFlow initialization failed (will retry)"
                    continue

                # Other error - don't retry
                raise RuntimeError(f"SynthSeg failed: {result.stderr}\nstdout: {result.stdout}")

            except subprocess.TimeoutExpired:
                last_error = f"Timed out after {timeout} seconds"
                continue

        # All retries exhausted
        raise RuntimeError(f"SynthSeg failed after {max_retries} attempts. Last error: {last_error}")

    def _compute_bias_field_features(self, img_data, brain_mask):
        features = {}
        scales = [5, 10, 20, 40]
        scale_variances = []
        for sigma in scales:
            smoothed = gaussian_filter(img_data, sigma=sigma)
            masked_smoothed = smoothed[brain_mask]
            scale_variances.append(np.var(masked_smoothed))
        
        features['bias_multiscale_var_mean'] = np.mean(scale_variances)
        features['bias_multiscale_var_std'] = np.std(scale_variances)
        features['bias_multiscale_var_range'] = np.max(scale_variances) - np.min(scale_variances)
        
        fft_img = fftn(img_data)
        fft_shifted = fftshift(fft_img)
        power_spectrum = np.abs(fft_shifted) ** 2
        
        center = np.array(power_spectrum.shape) // 2
        low_freq_radius = min(power_spectrum.shape) // 8
        
        y, x, z = np.ogrid[:power_spectrum.shape[0], :power_spectrum.shape[1], :power_spectrum.shape[2]]
        distance_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2 + (z - center[2])**2)
        low_freq_mask = distance_from_center <= low_freq_radius
        
        low_freq_energy = np.sum(power_spectrum[low_freq_mask])
        total_energy = np.sum(power_spectrum)
        features['bias_low_freq_energy_ratio'] = low_freq_energy / (total_energy + 1e-8)
        
        grad_x = np.gradient(img_data, axis=0)
        grad_y = np.gradient(img_data, axis=1)
        grad_z = np.gradient(img_data, axis=2)
        
        features['bias_gradient_x_mean'] = np.abs(grad_x[brain_mask]).mean()
        features['bias_gradient_y_mean'] = np.abs(grad_y[brain_mask]).mean()
        features['bias_gradient_z_mean'] = np.abs(grad_z[brain_mask]).mean()
        
        features['bias_monotonicity_score'] = self._compute_monotonicity(img_data, brain_mask)
        
        entropy_map = self._compute_local_entropy_map(img_data, brain_mask, window_size=5)
        features['bias_entropy_map_variance'] = np.var(entropy_map[brain_mask])
        return features

    def _compute_anatomical_complexity(self, img_data, brain_mask, affine):
        features = {}
        
        brain_center = np.array(center_of_mass(brain_mask))
        y, x, z = np.ogrid[:img_data.shape[0], :img_data.shape[1], :img_data.shape[2]]
        distance_from_center = np.sqrt((x - brain_center[1])**2 + (y - brain_center[0])**2 + (z - brain_center[2])**2)
        
        central_radius = min(img_data.shape) * 0.15
        edges = self._compute_edge_magnitude(img_data)
        edge_variance_per_slice = []
        for slice_idx in range(img_data.shape[2]):
            slice_edges = edges[:, :, slice_idx][brain_mask[:, :, slice_idx]]
            if len(slice_edges) > 0:
                edge_variance_per_slice.append(np.var(slice_edges))
        
        if len(edge_variance_per_slice) > 0:
            features['sulcal_depth_proxy_variance'] = np.var(edge_variance_per_slice)
            features['sulcal_depth_proxy_mean'] = np.mean(edge_variance_per_slice)
            features['sulcal_depth_proxy_moment5'] = moment(edge_variance_per_slice, moment=5)
            features['sulcal_depth_proxy_moment6'] = moment(edge_variance_per_slice, moment=6)
        else:
            features['sulcal_depth_proxy_variance'] = 0
            features['sulcal_depth_proxy_mean'] = 0
            features['sulcal_depth_proxy_moment5'] = 0
            features['sulcal_depth_proxy_moment6'] = 0

        brain_intensities = img_data[brain_mask]
        hist, bins = np.histogram(brain_intensities, bins=50)
        peaks, _ = find_peaks(gaussian_filter(hist.astype(float), sigma=1), distance=10)
        if len(peaks) >= 2:
            sorted_peaks = sorted(peaks, key=lambda p: bins[p])
            gm_intensity = bins[sorted_peaks[0]]
            wm_intensity = bins[sorted_peaks[-1]]
            features['gm_wm_contrast'] = np.abs(wm_intensity - gm_intensity) / (gm_intensity + 1e-8)
        else:
            features['gm_wm_contrast'] = 0
        
        z_coords = np.where(brain_mask)[2]
        inferior_threshold = np.percentile(z_coords, 25)
        inferior_mask = brain_mask & (z >= inferior_threshold)
        if inferior_mask.sum() > 0:
            inferior_com = center_of_mass(inferior_mask.astype(float))
            full_com = center_of_mass(brain_mask.astype(float))
            features['inferior_brain_offset'] = np.linalg.norm(np.array(inferior_com) - np.array(full_com))
        else:
            features['inferior_brain_offset'] = 0
        
        return features

    def _compute_structural_asymmetry(self, img_data, brain_mask):
        features = {}

        com = center_of_mass(brain_mask)
        midline_x = int(com[0])
        
        left_hem = brain_mask.copy()
        left_hem[midline_x:, :, :] = 0
        right_hem = brain_mask.copy()
        right_hem[:midline_x, :, :] = 0
        
        left_volume = left_hem.sum()
        right_volume = right_hem.sum()
        features['asymmetry_hemispheric_volume_ratio'] = np.abs(left_volume - right_volume) / (left_volume + right_volume + 1e-8)
        
        if left_volume > 0 and right_volume > 0:
            left_mean = img_data[left_hem].mean()
            right_mean = img_data[right_hem].mean()
            features['asymmetry_hemispheric_intensity_diff'] = np.abs(left_mean - right_mean)
        else:
            features['asymmetry_hemispheric_intensity_diff'] = 0
        
        geometric_center_x = img_data.shape[0] / 2
        features['asymmetry_midline_deviation'] = np.abs(midline_x - geometric_center_x) / img_data.shape[0]

        com_per_slice = []
        for z in range(brain_mask.shape[2]):
            slice_mask = brain_mask[:, :, z]
            if slice_mask.sum() > 10:
                slice_com = center_of_mass(slice_mask)
                com_per_slice.append(slice_com)
        
        if len(com_per_slice) > 1:
            com_array = np.array(com_per_slice)
            features['asymmetry_landmark_variance_x'] = np.var(com_array[:, 0])
            features['asymmetry_landmark_variance_y'] = np.var(com_array[:, 1])
        else:
            features['asymmetry_landmark_variance_x'] = 0
            features['asymmetry_landmark_variance_y'] = 0
        
        return features

    def _compute_pose_misalignment(self, img_affine, template_affine):
        features = {}
        
        img_translation = img_affine[:3, 3]
        template_translation = template_affine[:3, 3]
        translation_diff = np.linalg.norm(img_translation - template_translation)
        features['registration_translation_magnitude'] = translation_diff
        
        img_rotation = img_affine[:3, :3]
        template_rotation = template_affine[:3, :3]
        
        rotation_diff = np.linalg.norm(img_rotation - template_rotation, 'fro')
        features['registration_rotation_difference'] = rotation_diff
        
        img_det = np.linalg.det(img_rotation)
        template_det = np.linalg.det(template_rotation)
        features['registration_scale_difference'] = np.abs(img_det - template_det)
        
        img_scales = np.sqrt(np.sum(img_rotation ** 2, axis=0))
        scale_anisotropy = np.std(img_scales) / (np.mean(img_scales) + 1e-8)
        features['registration_scale_anisotropy'] = scale_anisotropy
        
        return features

    def _compute_edge_magnitude(self, img_data):
        edges_x = sobel(img_data, axis=0)
        edges_y = sobel(img_data, axis=1)
        edges_z = sobel(img_data, axis=2)
        return np.sqrt(edges_x**2 + edges_y**2 + edges_z**2)
    
    def _compute_monotonicity(self, img_data, brain_mask):
        coords = np.where(brain_mask)
        intensities = img_data[coords]
        
        correlations = []
        for axis in range(3):
            positions = coords[axis]
            if len(np.unique(positions)) > 1:
                corr = np.abs(np.corrcoef(positions, intensities)[0, 1])
                correlations.append(corr)
        
        if len(correlations) > 0:
            return np.max(correlations)
        else:
            return 0
    
    def _compute_local_entropy_map(self, img_data, brain_mask, window_size=5):
        img_normalized = ((img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8) * 255).astype(np.uint8)
        selem = ball(window_size)
        
        entropy_map = np.zeros_like(img_data)
        for z in range(img_data.shape[2]):
            if brain_mask[:, :, z].sum() > 0:
                entropy_map[:, :, z] = skimage_entropy(img_normalized[:, :, z], selem[:, :, window_size])
        return entropy_map

    def _compute_ngtdm_features(self, img_data, brain_mask, n_levels=64):
        try:
            from scipy import ndimage

            brain_vals = img_data[brain_mask]
            if len(brain_vals) < 100:
                return {'ngtdm_complexity': 0, 'ngtdm_busyness': 0}

            min_val, max_val = brain_vals.min(), brain_vals.max()
            if max_val - min_val < 1e-8:
                return {'ngtdm_complexity': 0, 'ngtdm_busyness': 0}

            quantized = np.zeros_like(img_data, dtype=np.int32)
            quantized[brain_mask] = np.clip(
                ((img_data[brain_mask] - min_val) / (max_val - min_val) * (n_levels - 1)).astype(np.int32),
                0, n_levels - 1
            )

            kernel = np.array([[[0,0,0],[0,1,0],[0,0,0]],
                            [[0,1,0],[1,0,1],[0,1,0]],
                            [[0,0,0],[0,1,0],[0,0,0]]]) / 6.0

            neighbor_avg = ndimage.convolve(quantized.astype(float), kernel, mode='constant')

            s = np.zeros(n_levels)
            N = np.zeros(n_levels)

            for i in range(n_levels):
                level_mask = (quantized == i) & brain_mask
                if level_mask.sum() > 0:
                    N[i] = level_mask.sum()
                    s[i] = np.abs(i - neighbor_avg[level_mask]).mean()

            Nvp = N.sum()
            p = N / (Nvp + 1e-10)

            complexity = 0.0
            valid_levels = np.where(p > 0)[0]
            for i in valid_levels:
                for j in valid_levels:
                    denom = p[i] + p[j]
                    if denom > 1e-10:
                        complexity += np.abs(i - j) * (p[i] * s[i] + p[j] * s[j]) / denom
            complexity /= (Nvp + 1e-10)

            numerator = np.sum(p * s)
            denominator = 0.0
            for i in valid_levels:
                for j in valid_levels:
                    denominator += np.abs(i * p[i] - j * p[j])
            busyness = numerator / (denominator + 1e-10)

            return {'ngtdm_complexity': complexity, 'ngtdm_busyness': busyness}
        except:
            return {'ngtdm_complexity': 0, 'ngtdm_busyness': 0}

    def _compute_segmentation_complexity(self, seg_path):
        """Compute complexity metrics from a SynthSeg segmentation."""
        seg_img = nib.load(seg_path)
        seg_data = seg_img.get_fdata().astype(int)

        features = {}

        unique_labels = np.unique(seg_data)
        features['n_structures'] = len(unique_labels) - 1
        features['n_segmented_voxels'] = np.sum(seg_data > 0)

        brain_mask = seg_data > 0
        eroded = binary_erosion(brain_mask)
        boundary_voxels = np.sum(brain_mask & ~eroded)
        features['boundary_voxels'] = boundary_voxels
        features['boundary_ratio'] = boundary_voxels / (features['n_segmented_voxels'] + 1)

        total_components = 0
        for lbl in unique_labels[unique_labels > 0]:
            labeled_array, num_features = label(seg_data == lbl)
            total_components += num_features
        features['total_components'] = total_components
        features['fragmentation_index'] = total_components / max(len(unique_labels) - 1, 1)

        return features

    def _compute_schaefer_region_stats(self, resampled_data, atlas_data, atlas_labels):
        """Compute per-ROI stats for the 7 Schaefer ROIs used by the 18-feature
        deployment set.

        For each matched ROI, computes range, kurtosis, median absolute
        deviation, and 5th central moment. Returns a flat dict keyed by
        '<roi_label>_<stat_name>' (e.g. '7Networks_LH_Default_PFC_1_range').
        Substring match handles label variants (the atlas labels typically
        carry an integer suffix after the canonical name).
        """
        needed_substrings = (
            '7Networks_LH_Default_PFC_1',
            '7Networks_RH_Default_PFCdPFCm_2',
            '7Networks_LH_SalVentAttn_PFCl_1',
            '7Networks_LH_Limbic_TempPole_1',
            '7Networks_RH_SomMot_2',
            '7Networks_RH_Cont_PFCl_3',
            '7Networks_RH_DorsAttn_Post_3',
        )
        stats = {}
        for i, label_val in enumerate(atlas_labels[1:], start=1):
            label_str = label_val.decode() if isinstance(label_val, bytes) else str(label_val)
            matched = next((s for s in needed_substrings if s in label_str), None)
            if matched is None:
                continue
            region_values = resampled_data[atlas_data == i]
            region_values = region_values[region_values > 0]
            if len(region_values) == 0:
                stats[f'{matched}_range'] = 0.0
                stats[f'{matched}_kurtosis'] = 0.0
                stats[f'{matched}_med_abs_dev'] = 0.0
                stats[f'{matched}_moment_5'] = 0.0
                continue
            stats[f'{matched}_range'] = float(np.ptp(region_values))
            stats[f'{matched}_kurtosis'] = float(kurtosis(region_values))
            stats[f'{matched}_med_abs_dev'] = float(median_abs_deviation(region_values))
            stats[f'{matched}_moment_5'] = float(moment(region_values, moment=5))
        return stats

    def _compute_tile_gradient_load_imbalance(self, img_data, brain_mask):
        """Compute tile_gradient_load_imbalance feature.

        Divides the image into 16x16x8 tiles; for each tile with sufficient
        brain content, computes the average 3D gradient magnitude under the
        mask. The feature is the max-over-mean ratio across tiles (a
        load-imbalance proxy).

        Returns:
            dict with key 'tile_gradient_load_imbalance' (float).
        """
        tile_features = {}
        if brain_mask.sum() < 100:
            tile_features['tile_gradient_load_imbalance'] = 0.0
            return tile_features

        tile_shape = (16, 16, 8)
        gradient_magnitudes = []

        for i in range(0, img_data.shape[0] - tile_shape[0], tile_shape[0]):
            for j in range(0, img_data.shape[1] - tile_shape[1], tile_shape[1]):
                for k in range(0, img_data.shape[2] - tile_shape[2], tile_shape[2]):
                    tile = img_data[i:i + tile_shape[0],
                                    j:j + tile_shape[1],
                                    k:k + tile_shape[2]]
                    tile_mask = brain_mask[i:i + tile_shape[0],
                                           j:j + tile_shape[1],
                                           k:k + tile_shape[2]]
                    if tile_mask.sum() <= (tile_shape[0] * tile_shape[1] * tile_shape[2] * 0.1):
                        continue
                    try:
                        tile_gradients = np.gradient(tile)
                        grad_magnitude = np.sqrt(sum(g**2 for g in tile_gradients))
                        gradient_magnitudes.append(float(grad_magnitude[tile_mask].mean()))
                    except Exception:
                        continue

        if len(gradient_magnitudes) > 1:
            load_imbalance = float(np.max(gradient_magnitudes) / (np.mean(gradient_magnitudes) + 1e-8))
        else:
            load_imbalance = 0.0
        tile_features['tile_gradient_load_imbalance'] = load_imbalance
        return tile_features


def extract_features_to_csv(mri_dir=None, output_path=None):
    """Extract the 18 deployed features for every .nii.gz in mri_dir and write to CSV.

    Args:
        mri_dir: Directory containing 3D MRI .nii.gz files. Defaults to
            MRI_3D_IMAGE_DATA from constants.py.
        output_path: CSV output path. Defaults to
            <FEATURES_CSV_OUTPUT.parent>/features_data_18.csv so it does NOT
            overwrite the historical 1,506-col features_data.csv.

    Returns:
        DataFrame with one row per MRI, columns: filename + the 18 features
        in DEFAULT_FEATURES order.
    """
    if mri_dir is None:
        mri_dir = Path(MRI_3D_IMAGE_DATA)
    if output_path is None:
        output_path = Path(FEATURES_CSV_OUTPUT).parent / 'features_data_18.csv'

    atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)
    atlas_img = nib.load(atlas.maps)
    atlas_data = atlas_img.get_fdata()
    atlas_labels = atlas.labels

    ds = Datasets(atlas=atlas)

    rows = []
    nifti_files = sorted(Path(mri_dir).glob('*.nii.gz'))
    for idx, img_path in enumerate(nifti_files, start=1):
        print(f'[{idx}/{len(nifti_files)}] {img_path.name}', flush=True)

        t0 = time.perf_counter()
        img = nib.load(str(img_path))
        img_data = img.get_fdata()
        load_time_ms = (time.perf_counter() - t0) * 1000.0
        affine = img.affine

        if np.any(img_data > 0):
            threshold = np.percentile(img_data[img_data > 0], 10)
        else:
            threshold = 0
        brain_mask = img_data > threshold
        brain_vals = img_data[brain_mask]

        anat = ds._compute_anatomical_complexity(img_data, brain_mask, affine)
        bias = ds._compute_bias_field_features(img_data, brain_mask)
        sym = ds._compute_structural_asymmetry(img_data, brain_mask)
        tile = ds._compute_tile_gradient_load_imbalance(img_data, brain_mask)
        ngtdm = ds._compute_ngtdm_features(img_data, brain_mask)
        pose = ds._compute_pose_misalignment(affine, np.eye(4))

        morpho_intensity_kurtosis = float(kurtosis(brain_vals)) if len(brain_vals) > 0 else 0.0

        seg_path = Path(SYNTHSEG_OUTPUT_DIR) / f'{img_path.stem}_seg.nii.gz'
        vol_path = Path(SYNTHSEG_OUTPUT_DIR) / f'{img_path.stem}_vol.csv'
        qc_path = Path(SYNTHSEG_OUTPUT_DIR) / f'{img_path.stem}_qc.csv'
        ds._run_synthseg_cli(
            input_path=str(img_path),
            seg_path=str(seg_path),
            vol_path=str(vol_path),
            qc_path=str(qc_path),
        )
        seg_features = ds._compute_segmentation_complexity(str(seg_path))

        resampled = image.resample_to_img(img, atlas_img, interpolation='linear')
        atlas_stats = ds._compute_schaefer_region_stats(
            resampled.get_fdata(), atlas_data, atlas_labels
        )

        row = {
            'filename': img_path.name,
            'load_time_ms': load_time_ms,
            'synthseg_total_components': seg_features.get('total_components', 0),
            'synthseg_fragmentation_index': seg_features.get('fragmentation_index', 0.0),
            'inferior_brain_offset': anat.get('inferior_brain_offset', 0.0),
            'ngtdm_complexity': ngtdm.get('ngtdm_complexity', 0.0),
            'bias_field_proxy': bias.get('bias_field_proxy', 0.0),
            'morpho_intensity_kurtosis': morpho_intensity_kurtosis,
            'asymmetry_hemispheric_volume_ratio': sym.get('asymmetry_hemispheric_volume_ratio', 0.0),
            'tile_gradient_load_imbalance': tile.get('tile_gradient_load_imbalance', 0.0),
            'registration_scale_difference': pose.get('registration_scale_difference', 0.0),
        }
        row.update({k: atlas_stats.get(k, 0.0) for k in (
            '7Networks_LH_Default_PFC_1_range',
            '7Networks_RH_Default_PFCdPFCm_2_med_abs_dev',
            '7Networks_LH_SalVentAttn_PFCl_1_kurtosis',
            '7Networks_LH_Default_PFC_1_kurtosis',
            '7Networks_LH_Limbic_TempPole_1_moment_5',
            '7Networks_RH_SomMot_2_kurtosis',
            '7Networks_RH_Cont_PFCl_3_kurtosis',
            '7Networks_RH_DorsAttn_Post_3_kurtosis',
        )})
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f'Saved 18-feature CSV to {output_path}')
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract the 18 deployed features from MRIs in a directory.'
    )
    parser.add_argument(
        'mri_dir', nargs='?', default=None,
        help='Directory of 3D .nii.gz files. Defaults to MRI_3D_IMAGE_DATA from constants.py.',
    )
    parser.add_argument(
        '--output', default=None,
        help='Output CSV path. Defaults to features_data_18.csv next to the historical CSV.',
    )
    args = parser.parse_args()
    extract_features_to_csv(
        mri_dir=Path(args.mri_dir) if args.mri_dir else None,
        output_path=Path(args.output) if args.output else None,
    )
