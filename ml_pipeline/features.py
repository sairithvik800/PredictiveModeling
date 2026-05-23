import os
import gc
from pyexpat import features
import numpy as np
import pandas as pd
import nibabel as nib
import time
from scipy.ndimage import center_of_mass, binary_erosion, zoom, sobel, laplace, binary_dilation, distance_transform_edt, label, gaussian_filter
from scipy.stats import skew, kurtosis, moment, kstest, median_abs_deviation, variation
from scipy.stats import entropy as scipy_entropy
from scipy.fft import fftn, fftshift
from scipy.signal import find_peaks
from skimage.morphology import ball
from skimage.filters.rank import entropy as skimage_entropy
from skimage.measure import moments_central
from skimage.feature import graycomatrix, graycoprops, peak_local_max, local_binary_pattern
from skimage.segmentation import watershed, find_boundaries
from skimage.measure import euler_number, marching_cubes
from nilearn import datasets, image
from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix
from nilearn.glm.second_level import SecondLevelModel, make_second_level_design_matrix
from constants import MRI_3D_IMAGE_DATA, MRI_4D_IMAGE_DATA, FEATURES_CSV_OUTPUT, SLANT_RUNTIME_DATA, TARGET_IMAGE, EC2_RUNTIME_DATA
from constants import SYNTHSEG_OUTPUT_DIR, SYNTHSEG_VOLUMES_CSV, SYNTHSEG_QC_CSV, ICC_FEATURES_CSV, SYNTHSEG_CONDA_ENV, SYNTHSEG_PREDICT_SCRIPT
import subprocess
import gzip
import zlib
import pywt
import antropy
import re
import glob as glob_module
import SimpleITK as sitk
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def _register_single_volume_worker(args):
    """Worker function for parallel volume registration.

    Args:
        args: Tuple of (vol_data, ref_data, voxel_spacing, vol_idx)

    Returns:
        Tuple of (vol_idx, translations, rotations) or (vol_idx, None, None) on failure
    """
    vol_data, ref_data, voxel_spacing, vol_idx = args

    import SimpleITK as sitk  # Import in worker to ensure availability

    ref_sitk = sitk.GetImageFromArray(ref_data.T)
    ref_sitk.SetSpacing([float(s) for s in voxel_spacing])

    mov_sitk = sitk.GetImageFromArray(vol_data.T)
    mov_sitk.SetSpacing([float(s) for s in voxel_spacing])

    registration = sitk.ImageRegistrationMethod()
    registration.SetMetricAsMeanSquares()
    registration.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0, minStep=0.001, numberOfIterations=200
    )
    registration.SetInterpolator(sitk.sitkLinear)

    initial_transform = sitk.CenteredTransformInitializer(
        ref_sitk, mov_sitk, sitk.Euler3DTransform(),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration.SetInitialTransform(initial_transform, inPlace=False)

    try:
        final_transform = registration.Execute(ref_sitk, mov_sitk)
        params = final_transform.GetParameters()
        return (vol_idx, params[3:6], params[0:3])  # translations, rotations
    except Exception:
        return (vol_idx, None, None)

class Datasets:
    def __init__(self, atlas=None):
        """Init dataset inheritiance class for nilearn atlases

        Args:
            atlas (nilearn.datasets.Atlas, optional): Atlas object from nilearn. Defaults to None
        """
        self._atlas = atlas
        if atlas:
            if isinstance(self._atlas.maps, str):
                self._atlas_img = nib.load(self._atlas.maps)
            else:
                self._atlas_img = self._atlas.maps
            self._atlas_labels = self._atlas.labels
        self._mri_3d_image_data_dir = MRI_3D_IMAGE_DATA
        self._mri_4d_image_data_dir = MRI_4D_IMAGE_DATA
        self._features_csv_dir = FEATURES_CSV_OUTPUT
        self._target_image_dir = TARGET_IMAGE
        self._synthseg_output_dir = SYNTHSEG_OUTPUT_DIR
        self._synthseg_volumes_csv = SYNTHSEG_VOLUMES_CSV
        self._synthseg_qc_csv = SYNTHSEG_QC_CSV
        self._icc_features_csv = ICC_FEATURES_CSV

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

    def extract_features_to_csv(self):
        """Extract features from MRI images then save to CSV file
        """
        records = self._extract_region_features()
        columns = self._setup_columns()
        df = pd.DataFrame(records, columns=columns)
        df = self._extract_size_features(df)
        df = self._extract_computational_features(df)
        df = self._extract_watershed_features(df)
        df = self._extract_preprocessing_features(df)
        df = self._extract_postprocessing_features(df)
        df = self._extract_registration_features(df)
        df = self._extract_artifact_features(df)
        df = self._extract_image_features(df)
        df = self._extract_network_derived_features(df)
        df = self._extract_information_theoretic_features(df)
        df = self._extract_frequency_texture_features(df)
        df = self._extract_preprocessing_proxy_features(df)
        df = self._extract_topological_complexity_features(df)
        df = self._extract_deep_learning_predictors(df)
        df = self._extract_metadata_features(df)
        df = self._extract_hypothesis_features(df)
        #df = self._extract_motion_features(df)
        df = self._extract_temporal_quality_features(df)
        df = self._extract_synthseg_features(df)
        df = self._extract_structural_quality_features(df)
        df = self._extract_morphometric_features(df)
        df.to_csv(self._features_csv_dir, index=False)
        print(f'Saved region-wise features to {self._features_csv_dir}')
    
    def _extract_watershed_features(self, df):
        for index, fname in enumerate(df['filename'], start=1):
            print(f'Image {index}: Processing 3D Watershed features for {fname}')
            img_path = os.path.join(self._mri_3d_image_data_dir, fname)
            img = nib.load(img_path)
            img_data = img.get_fdata()
            if np.any(img_data > 0):
                threshold = np.percentile(img_data[img_data > 0], 10)
            else:
                threshold = 0
            brain_mask = img_data > threshold

            distance = distance_transform_edt(brain_mask)
            local_maxima = peak_local_max(-distance, min_distance=5, labels=brain_mask)
            markers = np.zeros_like(brain_mask, dtype=int)
            markers[tuple(local_maxima.T)] = np.arange(len(local_maxima)) + 1

            watershed_labels = watershed(-distance, markers=markers, mask=brain_mask)
            df.loc[df['filename'] == fname, 'watershed_regions'] = len(np.unique(watershed_labels) - 1) 
            
            bounadries = find_boundaries(watershed_labels, mode='inner')
            boundary_density = bounadries.sum() / brain_mask.sum()
            df.loc[df['filename'] == fname, 'watershed_boundary_density'] = boundary_density
            
        for index, fname in enumerate(df['filename'], start=1):
            print(f'Image {index}: Processing 4D Watershed features for {fname}')
            fpath = os.path.join(self._mri_4d_image_data_dir, fname)
            img = nib.load(fpath)
            img_data = img.get_fdata()
            temporal_mean = img_data.mean(axis=3)
            temporal_std = img_data.std(axis=3)
            if np.any(temporal_mean > 0):
                threshold = np.percentile(temporal_mean[temporal_mean > 0], 10)
            else:
                threshold = 0
            brain_mask = temporal_mean > threshold

            distance = distance_transform_edt(brain_mask)
            local_maxima = peak_local_max(-distance, min_distance=5, labels=brain_mask)
            markers = np.zeros_like(brain_mask, dtype=int)
            markers[tuple(local_maxima.T)] = np.arange(len(local_maxima)) + 1

            watershed_labels = watershed(-distance, markers=markers, mask=brain_mask)
            n_segments = len(np.unique(watershed_labels)) - 1
            df.loc[df['filename'] == fname, 'watershed_regions'] = n_segments

            boundaries = find_boundaries(watershed_labels, mode='outer')
            boundary_density = boundaries.sum() / brain_mask.sum()
            df.loc[df['filename'] == fname, 'watershed_boundary_density'] = boundary_density
            
            region_variances = []
            for region_id in range(1, n_segments + 1):
                region_mask = watershed_labels == region_id
                if region_mask.sum() > 0:
                    region_variances.append(temporal_std[region_mask].mean()) 

            df.loc[df['filename'] == fname, 'watershed_region_variances_mean'] = np.mean(region_variances)
            df.loc[df['filename'] == fname, 'watershed_region_variances_std'] = np.std(region_variances)
            df.loc[df['filename'] == fname, 'watershed_region_variances_max'] = np.max(region_variances)
            df.loc[df['filename'] == fname, 'watershed_region_variances_min'] = np.min(region_variances)
        return df
    
    def _extract_preprocessing_features(self, df):
        template_img = nib.load(self._target_image_dir)
        template_data = template_img.get_fdata()
        template_mask = template_data > (template_data.max() * 0.1)
        for index, fname in enumerate(df['filename'], start=1):
            print(f'Image {index}: Processing preprocessing features for {fname}...')
            fpath = os.path.join(self._mri_3d_image_data_dir, fname)
            img = nib.load(fpath)
            img_data = img.get_fdata()
            img_affine = img.affine
            
            if np.any(img_data > 0):
                threshold = np.percentile(img_data[img_data > 0], 10)
            else:
                threshold = 0
            brain_mask = img_data > threshold
            
            bias_features = self._compute_bias_field_features(img_data, brain_mask)
            for key, value in bias_features.items():
                df.loc[df['filename'] == fname, key] = value
            
            hist_features = self._compute_histogram_complexity(img_data, brain_mask)
            for key, value in hist_features.items():
                df.loc[df['filename'] == fname, key] = value
            
            anat_features = self._compute_anatomical_complexity(img_data, brain_mask, img_affine)
            for key, value in anat_features.items():
                df.loc[df['filename'] == fname, key] = value
            
            norm_features = self._compute_normalization_difficulty(img_data, brain_mask)
            for key, value in norm_features.items():
                df.loc[df['filename'] == fname, key] = value
        
        return df

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

    def _compute_histogram_complexity(self, img_data, brain_mask):
        features = {}
        brain_intensities = img_data[brain_mask]

        hist, bin_edges = np.histogram(brain_intensities, bins=100)
        hist_normalized = hist / hist.sum()

        smoothed_hist = gaussian_filter(hist_normalized.astype(float), sigma=2)
        features['histogram_kurtosis'] = kurtosis(brain_intensities)
        features['histogram_moment3'] = moment(brain_intensities, moment=3)
        features['histogram_moment4'] = moment(brain_intensities, moment=4)
        features['histogram_moment5'] = moment(brain_intensities, moment=5)
        features['histogram_moment6'] = moment(brain_intensities, moment=6)
        features['histogram_variation_coefficient'] = variation(brain_intensities)

        normalized_intensities = (brain_intensities - brain_intensities.mean()) / (brain_intensities.std() + 1e-8)
        ks_stat, ks_pvalue = kstest(normalized_intensities, 'norm')
        features['histogram_ks_statistic'] = ks_stat

        percentile_01 = np.percentile(brain_intensities, 1)
        percentile_99 = np.percentile(brain_intensities, 99)
        features['histogram_dynamic_range'] = percentile_99 - percentile_01
        features['histogram_robust_range_ratio'] = (percentile_99 - percentile_01) / (brain_intensities.max() - brain_intensities.min() + 1e-8)

        unique_count = len(np.unique(brain_intensities))
        features['histogram_unique_count'] = unique_count
        features['effective_bits_per_voxel'] = np.log2(unique_count + 1e-8)

        features['sparsity_ratio'] = np.sum(hist == 0) / len(hist)
        nonzero = brain_intensities[brain_intensities > 0]
        if len(nonzero) > 0:
            max_val, min_val = nonzero.max(), nonzero.min()
            features['log_dynamic_range'] = np.log10(max_val / (min_val + 1e-10))
        else:
            features['log_dynamic_range'] = 0

        # Peak detection for multimodality
        smoothed_for_peaks = gaussian_filter(hist_normalized.astype(float), sigma=3)
        try:
            peaks, properties = find_peaks(smoothed_for_peaks, prominence=0.01)
            features['histogram_peak_count'] = len(peaks)

            if len(peaks) >= 2:
                peak_positions = bin_edges[peaks]
                features['histogram_peak_separation'] = (peak_positions[-1] - peak_positions[0]) / (bin_edges[-1] - bin_edges[0])

                # Valley depth between two highest peaks
                sorted_peaks = sorted(peaks, key=lambda p: smoothed_for_peaks[p], reverse=True)[:2]
                sorted_peaks = sorted(sorted_peaks)  # Sort by position
                valley_idx = (sorted_peaks[0] + sorted_peaks[1]) // 2
                if valley_idx < len(smoothed_for_peaks):
                    valley_depth = smoothed_for_peaks[valley_idx]
                    peak_heights = [smoothed_for_peaks[p] for p in sorted_peaks]
                    features['histogram_bimodal_depth'] = valley_depth / max(peak_heights)
                else:
                    features['histogram_bimodal_depth'] = 1
            else:
                features['histogram_peak_separation'] = 0
                features['histogram_bimodal_depth'] = 1
        except:
            features['histogram_peak_count'] = 0
            features['histogram_peak_separation'] = 0
            features['histogram_bimodal_depth'] = 1

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

    def _compute_normalization_difficulty(self, img_data, brain_mask):
        features = {}
        brain_intensities = img_data[brain_mask]
        
        mean_intensity = brain_intensities.mean()
        std_intensity = brain_intensities.std()
        outlier_mask = np.abs(brain_intensities - mean_intensity) > 3 * std_intensity
        features['normalization_outlier_percentage'] = outlier_mask.sum() / len(brain_intensities)
        
        mad = median_abs_deviation(brain_intensities)
        features['normalization_mad_std_ratio'] = mad / (std_intensity + 1e-8)
        
        grad_magnitude = self._compute_edge_magnitude(img_data)
        grad_brain = grad_magnitude[brain_mask]
        features['normalization_gradient_mean'] = grad_brain.mean()
        features['normalization_gradient_std'] = grad_brain.std()
        features['normalization_gradient_skewness'] = skew(grad_brain)
        features['normalization_gradient_kurtosis'] = kurtosis(grad_brain)
        return features

    def _extract_postprocessing_features(self, df):        
        for index, fname in enumerate(df['filename'], start=1):
            print(f'Image {index}: Processing postprocessing features for {fname}')
            fpath = os.path.join(self._mri_3d_image_data_dir, fname)
            img = nib.load(fpath)
            img_data = img.get_fdata()
            
            if np.any(img_data > 0):
                threshold = np.percentile(img_data[img_data > 0], 10)
            else:
                threshold = 0
            brain_mask = img_data > threshold
            
            tile_features = self._compute_tile_overlap_complexity(img_data, brain_mask)
            for key, value in tile_features.items():
                df.loc[df['filename'] == fname, key] = value
            
            seg_features = self._compute_segmentation_load(img_data, brain_mask)
            for key, value in seg_features.items():
                df.loc[df['filename'] == fname, key] = value
            
            asym_features = self._compute_structural_asymmetry(img_data, brain_mask)
            for key, value in asym_features.items():
                df.loc[df['filename'] == fname, key] = value
        return df

    def _compute_tile_overlap_complexity(self, img_data, brain_mask):
        features = {}
        
        shape = np.array(img_data.shape)
        tile_size = shape // 3
        overlap_variances = []
        boundary_consistencies = []
        tile_correlations = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    start = tile_size * np.array([i, j, k])
                    end = start + tile_size
                    end = np.minimum(end, shape)
                    
                    tile = img_data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
                    tile_mask = brain_mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
                    
                    if tile_mask.sum() == 0:
                        continue

                    if i < 2:
                        next_start = tile_size * np.array([i+1, j, k])
                        next_end = np.minimum(next_start + tile_size, shape)
                        
                        # Overlap region
                        overlap_start = np.maximum(start, next_start)
                        overlap_end = np.minimum(end, next_end)
                        
                        if np.all(overlap_end > overlap_start):
                            overlap_region = img_data[overlap_start[0]:overlap_end[0], 
                                                    overlap_start[1]:overlap_end[1], 
                                                    overlap_start[2]:overlap_end[2]]
                            overlap_mask = brain_mask[overlap_start[0]:overlap_end[0], 
                                                    overlap_start[1]:overlap_end[1], 
                                                    overlap_start[2]:overlap_end[2]]
                            
                            if overlap_mask.sum() > 10:
                                overlap_variances.append(np.var(overlap_region[overlap_mask]))
                    
                    edges = self._compute_edge_magnitude(tile)
                    if tile_mask.sum() > 0:
                        if i > 0:
                            boundary_edges = edges[0, :, :][tile_mask[0, :, :]]
                            if len(boundary_edges) > 0:
                                boundary_consistencies.append(np.mean(boundary_edges))
        if len(boundary_consistencies) > 0:
            features['tile_boundary_consistency_mean'] = np.mean(boundary_consistencies)
            features['tile_boundary_consistency_std'] = np.std(boundary_consistencies)
        else:
            features['tile_boundary_consistency_mean'] = 0
            features['tile_boundary_consistency_std'] = 0
        return features

    def _compute_segmentation_load(self, img_data, brain_mask):
        features = {}

        thresholds = [20, 40, 60, 80]
        fragmentation_scores = []
        for thresh_percentile in thresholds:
            thresh_val = np.percentile(img_data[brain_mask], thresh_percentile)
            thresh_mask = (img_data > thresh_val) & brain_mask
            
            labeled_array, num_features = label(thresh_mask)
            fragmentation_scores.append(num_features)
        
        features['segmentation_fragmentation_mean'] = np.mean(fragmentation_scores)
        features['segmentation_fragmentation_max'] = np.max(fragmentation_scores)
        
        edges = self._compute_edge_magnitude(img_data)
        brain_edges = edges[brain_mask]
        
        features['segmentation_boundary_irregularity'] = np.var(brain_edges)
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

    def _extract_registration_features(self, df):
        template_img = nib.load(self._target_image_dir)
        template_data = template_img.get_fdata()
        template_affine = template_img.affine
        
        for index, fname in enumerate(df['filename'], start=1):
            print(f'Image {index}: Processing registration features for {fname}...')
            fpath = os.path.join(self._mri_3d_image_data_dir, fname)
            img = nib.load(fpath)
            img_data = img.get_fdata()
            img_affine = img.affine
            
            pose_features = self._compute_pose_misalignment(img_affine, template_affine)
            for key, value in pose_features.items():
                df.loc[df['filename'] == fname, key] = value
            
            img_resampled = image.resample_to_img(img, template_img, interpolation='linear')
            img_data_resampled = img_resampled.get_fdata()
            
            template_mask = template_data > (template_data.max() * 0.1)
            img_mask = img_data_resampled > (img_data_resampled.max() * 0.1)
            common_mask = template_mask & img_mask
            
            if common_mask.sum() > 100:
                template_values = template_data[common_mask]
                img_values = img_data_resampled[common_mask]
                
                correlation = np.corrcoef(template_values, img_values)[0, 1]
                df.loc[df['filename'] == fname, 'registration_initial_correlation'] = correlation
                
                hist_t, bins = np.histogram(template_values, bins=50, density=True)
                hist_i, _ = np.histogram(img_values, bins=bins, density=True)
                hist_intersection = np.minimum(hist_t, hist_i).sum()
                df.loc[df['filename'] == fname, 'registration_histogram_intersection'] = hist_intersection
            else:
                df.loc[df['filename'] == fname, 'registration_initial_correlation'] = 0
                df.loc[df['filename'] == fname, 'registration_histogram_intersection'] = 0
        
        return df

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

    def _extract_artifact_features(self, df):
        for index, fname in enumerate(df['filename'], start=1):
            print(f'Image {index}: Processing artifact features for {fname}...')
            fpath = os.path.join(self._mri_3d_image_data_dir, fname)
            img = nib.load(fpath)
            img_data = img.get_fdata()
            
            if np.any(img_data > 0):
                threshold = np.percentile(img_data[img_data > 0], 10)
            else:
                threshold = 0
            brain_mask = img_data > threshold
            
            fft_img = fftn(img_data)
            power_spectrum = np.abs(fft_img) ** 2
            
            center = np.array(power_spectrum.shape) // 2
            y, x, z = np.ogrid[:power_spectrum.shape[0], :power_spectrum.shape[1], :power_spectrum.shape[2]]
            distance_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2 + (z - center[2])**2)
            
            high_freq_threshold = min(power_spectrum.shape) // 4
            high_freq_mask = distance_from_center > high_freq_threshold
            
            high_freq_energy = np.sum(power_spectrum[high_freq_mask])
            total_energy = np.sum(power_spectrum)
            df.loc[df['filename'] == fname, 'artifact_high_freq_ratio'] = high_freq_energy / (total_energy + 1e-8)
            
            edges = self._compute_edge_magnitude(img_data)
            brain_edges = edges[brain_mask]
            
            edge_intensity_ratio = np.percentile(brain_edges, 95) / (np.mean(img_data[brain_mask]) + 1e-8)
            df.loc[df['filename'] == fname, 'artifact_gibbs_ringing_proxy'] = edge_intensity_ratio
            
            local_snrs = []
            window_size = 20 
            for i in range(0, img_data.shape[0] - window_size, window_size):
                for j in range(0, img_data.shape[1] - window_size, window_size):
                    for k in range(0, img_data.shape[2] - window_size, window_size):
                        window = img_data[i:i+window_size, j:j+window_size, k:k+window_size]
                        window_mask = brain_mask[i:i+window_size, j:j+window_size, k:k+window_size]
                        
                        if window_mask.sum() > 10:
                            signal = window[window_mask].mean()
                            noise = window[window_mask].std()
                            if noise > 0:
                                local_snrs.append(signal / noise)
            
            if len(local_snrs) > 0:
                df.loc[df['filename'] == fname, 'artifact_snr_heterogeneity'] = np.std(local_snrs)
                df.loc[df['filename'] == fname, 'artifact_snr_mean'] = np.mean(local_snrs)
            else:
                df.loc[df['filename'] == fname, 'artifact_snr_heterogeneity'] = 0
                df.loc[df['filename'] == fname, 'artifact_snr_mean'] = 0
        
        return df
    
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

    def _extract_image_features(self, df):
        """Extract features related to image structure

        Args:
            df (pandas.DataFrame): DataFrame containing image data

        Returns:
            pandas.DataFrame: DataFrame with extracted computational features
        """
        # MNI305 template features
        template_img = nib.load(self._target_image_dir)
        template_data = template_img.get_fdata()
        template_mask = template_data > (template_data.max() * 0.1)
        
        template_center_of_mass = center_of_mass(template_mask)
        template_center_of_mass_normalized = np.array(template_center_of_mass) / np.array(template_data.shape)
        template_volume = template_mask.sum()
        template_voxel_dimensions = np.sqrt((template_img.affine[:3, :3] ** 2).sum(axis=0))

        print('Computing image features...')
        for index, fname in enumerate(df['filename'], start=1):
            print(f'Image {index}: Processing {fname}...')
            fpath = os.path.join(self._mri_3d_image_data_dir, fname)
            img = nib.load(fpath)
            img_data = img.get_fdata()
            img_affine = img.affine
            voxel_dimensions = np.sqrt((img_affine[:3, :3] ** 2).sum(axis=0))
            
            img_resampled = image.resample_to_img(img, self._atlas_img, interpolation='linear', copy_header=True, force_resample=True)
            img_data_resampled = img_resampled.get_fdata()
            
            if np.any(img_data > 0):
                threshold = np.percentile(img_data[img_data > 0], 10)
            else:
                threshold = 0
            brain_mask = img_data > threshold

            # Brain Mask Properties
            df.loc[df['filename'] == fname, 'brain_mask'] = brain_mask.sum()
            df.loc[df['filename'] == fname, 'brain_tissue_ratio'] = brain_mask.sum() / img_data.size
            df.loc[df['filename'] == fname, 'brain_volume_mm3'] = brain_mask.sum() * np.prod(voxel_dimensions) # voxel volume
            
            image_center_of_mass = center_of_mass(brain_mask)
            image_center_of_mass_normalized = np.array(image_center_of_mass) / np.array(img_data.shape)

            # MNI305 Template Comparisons
            df.loc[df['filename'] == fname, 'template_center_of_mass_distance'] = np.linalg.norm(template_center_of_mass_normalized - image_center_of_mass_normalized) # np.linalg.norm is Euclidean distance
            if template_volume > 0:
                df.loc[df['filename'] == fname, 'template_volume_ratio'] = brain_mask.sum() / template_volume
            else:
                df.loc[df['filename'] == fname, 'template_volume_ratio'] = np.nan
            
            zoom_factors = voxel_dimensions / template_voxel_dimensions
            resampled_brain = zoom(brain_mask.astype(float), zoom_factors, order=1) > 0.5
            
            min_shape = np.minimum(resampled_brain.shape, template_mask.shape)
            resampled_brain_crop = resampled_brain[:min_shape[0], :min_shape[1], :min_shape[2]]
            template_mask_crop = template_mask[:min_shape[0], :min_shape[1], :min_shape[2]]
            
            intersection = np.logical_and(resampled_brain_crop, template_mask_crop).sum()
            union = resampled_brain_crop.sum() + template_mask_crop.sum()
            if union > 0:
                dice_coefficient = (2.0 * intersection) / union
                df.loc[df['filename'] == fname, 'template_dice_overlap'] = dice_coefficient
            else:
                df.loc[df['filename'] == fname, 'template_dice_overlap'] = np.nan
            
            # Image Quality
            signal = img_data[img_data > 0].mean()
            noise = img_data[img_data > 0].std()
            if signal > 0:
                cov = noise / signal
            else:
                cov = 0
            df.loc[df['filename'] == fname, 'coefficient_of_variation'] = cov

            # Spatial Complexity Metrics
            edges_x = sobel(img_data, axis=0)
            edges_y = sobel(img_data, axis=1)
            edges_z = sobel(img_data, axis=2)
            edge_magnitude = np.sqrt(edges_x**2 + edges_y**2 + edges_z**2)
            edge_density = (edge_magnitude > edge_magnitude.mean()).sum() / img_data.size
            df.loc[df['filename'] == fname, 'edge_density'] = edge_density

            # Edge entropy (gradient-based complexity)
            edge_brain = edge_magnitude[brain_mask]
            if len(edge_brain) > 0:
                edge_hist, _ = np.histogram(edge_brain, bins=100)
                edge_entropy = scipy_entropy(edge_hist + 1e-8)
            else:
                edge_entropy = 0
            df.loc[df['filename'] == fname, 'edge_entropy'] = edge_entropy

            laplacian_var = np.var(laplace(img_data))
            df.loc[df['filename'] == fname, 'laplacian_variance'] = laplacian_var
            
            img_normalized = ((img_data - img_data.min()) / (img_data.max() - img_data.min()) * 255).astype(np.uint8)

            # Multi-slice aggregated GLCM features
            slice_indices = [int(img_data.shape[0] * p) for p in [0.2, 0.35, 0.5, 0.65, 0.8]]
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]

            contrasts, homogeneities, correlations, entropies = [], [], [], []
            for idx in slice_indices:
                if idx < img_normalized.shape[0]:
                    slice_2d = img_normalized[idx, :, :]
                    try:
                        glcm = graycomatrix(slice_2d, [1], angles, 256, symmetric=True, normed=True)
                        contrasts.append(graycoprops(glcm, 'contrast').mean())
                        homogeneities.append(graycoprops(glcm, 'homogeneity').mean())
                        correlations.append(graycoprops(glcm, 'correlation').mean())
                        # Joint entropy
                        glcm_norm = glcm / (glcm.sum() + 1e-10)
                        entropies.append(-np.sum(glcm_norm * np.log2(glcm_norm + 1e-10)))
                    except:
                        pass

            if len(contrasts) > 0:
                df.loc[df['filename'] == fname, 'glcm_contrast_aggregated'] = np.mean(contrasts)
                df.loc[df['filename'] == fname, 'glcm_homogeneity_aggregated'] = np.mean(homogeneities)
                df.loc[df['filename'] == fname, 'glcm_correlation_aggregated'] = np.mean(correlations)
                df.loc[df['filename'] == fname, 'glcm_joint_entropy_aggregated'] = np.mean(entropies)
            else:
                df.loc[df['filename'] == fname, 'glcm_contrast_aggregated'] = 0
                df.loc[df['filename'] == fname, 'glcm_homogeneity_aggregated'] = 0
                df.loc[df['filename'] == fname, 'glcm_correlation_aggregated'] = 0
                df.loc[df['filename'] == fname, 'glcm_joint_entropy_aggregated'] = 0

            # Keep original texture_contrast for backward compatibility
            mid_slice = img_normalized[img_normalized.shape[0]//2, :, :]
            try:
                glcm = graycomatrix(mid_slice, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
                contrast = graycoprops(glcm, 'contrast')[0, 0]
            except:
                contrast = 0
            df.loc[df['filename'] == fname, 'texture_contrast'] = contrast
            
            # Fractal Dimension 
            df.loc[df['filename'] == fname, 'fractal_dimension'] = self._estimate_fractal_dimension(brain_mask)

            # Intensity Distribution
            brain_intensities = img_data[brain_mask]
            m3 = moment(brain_intensities, moment=3)
            m4 = moment(brain_intensities, moment=4)
            n = len(brain_intensities)
            bimodality = (m3**2 + 1) / (m4 + 3*(n-1)**2/((n-2)*(n-3)))
            df.loc[df['filename'] == fname, 'bimodality_coefficient'] = bimodality

            # Network Tiles
            features = self._extract_tile_complexity_features(img_data, brain_mask)
            for key, value in features.items():
                df.loc[df['filename'] == fname, key] = value
                
            #High Correlation Network Variations - Schaefer Atlas Only
            features = self._extract_network_interactions(img_data_resampled, self._atlas_data)
            for key, value in features.items():
                df.loc[df['filename'] == fname, key] = value
            
            # Preprocessing Difficulty Proxies
            preprocessing_features = self._extract_preprocessing_difficulty(img_data, brain_mask)
            for key, value in preprocessing_features.items():
                df.loc[df['filename'] == fname, key] = value
            
        return df
    
    def _estimate_fractal_dimension(self, mask):
        max_box_size = min(mask.shape) // 4
        scales = np.logspace(0, np.log2(max_box_size), num=10, base=2, dtype=int)
        scales = np.unique(scales)
        counts = []
        
        for scale in scales:
            count = 0
            for i in range(0, mask.shape[0], scale):
                for j in range(0, mask.shape[1], scale):
                    for k in range(0, mask.shape[2], scale):
                        box = mask[i:i+scale, j:j+scale, k:k+scale]
                        if box.any():
                            count += 1
            counts.append(count)
        
        coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
        return -coeffs[0]
    
    def _extract_tile_complexity_features(self, img_data, brain_mask):
        shape = np.array(img_data.shape)
        tile_size = shape // 3

        tile_edge_densities = []
        tile_intensity_ranges = []
        tile_nonzero_fractions = []
        tile_entropies = []

        for i in range(3):
            for j in range(3):
                for k in range(3):
                    start = tile_size * np.array([i, j, k])
                    end = start + tile_size

                    tile = img_data[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
                    tile_mask = brain_mask[start[0]:end[0], start[1]:end[1], start[2]:end[2]]

                    if tile_mask.sum() > 0:
                        tile_data = tile[tile_mask]

                        # Edge detection per tile
                        edges = np.sqrt(
                            sobel(tile, axis=0)**2 +
                            sobel(tile, axis=1)**2 +
                            sobel(tile, axis=2)**2
                        )
                        tile_edge_densities.append((edges > edges.mean()).sum() / tile.size)

                        # Intensity range
                        tile_intensity_ranges.append(
                            np.percentile(tile_data, 95) - np.percentile(tile_data, 5)
                        )

                        # Non-zero fraction (sparsity)
                        tile_nonzero_fractions.append(tile_mask.sum() / tile_mask.size)

                        # Tile entropy
                        if len(tile_data) > 10:
                            hist, _ = np.histogram(tile_data, bins=50)
                            tile_entropy = scipy_entropy(hist + 1e-8)
                            tile_entropies.append(tile_entropy)

        result = {
            'tile_edge_density_min': np.min(tile_edge_densities),
            'tile_edge_density_max': np.max(tile_edge_densities),
            'tile_edge_density_mean': np.mean(tile_edge_densities),
            'tile_edge_density_std': np.std(tile_edge_densities),
            'tile_intensity_range_min': np.min(tile_intensity_ranges),
            'tile_intensity_range_max': np.max(tile_intensity_ranges),
            'tile_intensity_range_mean': np.mean(tile_intensity_ranges),
            'tile_intensity_range_std': np.std(tile_intensity_ranges),
            'tile_nonzero_min': np.min(tile_nonzero_fractions),
            'tile_nonzero_max': np.max(tile_nonzero_fractions),
            'tile_nonzero_mean': np.mean(tile_nonzero_fractions),
            'tile_nonzero_std': np.std(tile_nonzero_fractions),
        }

        # Add tile entropy features if computed
        if len(tile_entropies) > 0:
            result['tile_entropy_mean'] = np.mean(tile_entropies)
            result['tile_entropy_std'] = np.std(tile_entropies)
            result['tile_entropy_max'] = np.max(tile_entropies)
            result['tile_entropy_range'] = np.ptp(tile_entropies)
        else:
            result['tile_entropy_mean'] = 0
            result['tile_entropy_std'] = 0
            result['tile_entropy_max'] = 0
            result['tile_entropy_range'] = 0

        return result
    
    def _extract_network_interactions(self, img_data, atlas_data):        
        high_correlation_pairs = [
            ('7Networks_RH_Cont_PFCl_2', '7Networks_LH_Default_pCunPCC_1'),
            ('7Networks_RH_Default_PFCv_1', '7Networks_LH_SomMot_3'),
        ]
        
        features = {}
        
        for region1_name, region2_name in high_correlation_pairs:
            idx1 = self._atlas_labels.index(region1_name)
            idx2 = self._atlas_labels.index(region2_name)
            
            mask1 = (atlas_data == idx1)
            mask2 = (atlas_data == idx2)
            
            if mask1.any() and mask2.any():
                data1 = img_data[mask1]
                data2 = img_data[mask2]
                
                if data2.var() > 0:
                    features[f'{region1_name}_vs_{region2_name}_var_ratio'] = \
                        data1.var() / data2.var()
        
        return features
    
    def _extract_preprocessing_difficulty(self, img_data, brain_mask):
        features = {}

        # Original bias field proxy
        grid_samples = []
        for x in np.linspace(0, img_data.shape[0]-1, 5, dtype=int):
            for y in np.linspace(0, img_data.shape[1]-1, 5, dtype=int):
                for z in np.linspace(0, img_data.shape[2]-1, 5, dtype=int):
                    if brain_mask[x, y, z]:
                        grid_samples.append(img_data[x, y, z])

        if len(grid_samples) > 0:
            features['bias_field_proxy'] = np.std(grid_samples) / (np.mean(grid_samples) + 1e-8)
        else:
            features['bias_field_proxy'] = 0

        # NEW: Estimate WM/GM from histogram peaks for CJV and tissue-specific CV
        brain_intensities = img_data[brain_mask]
        if len(brain_intensities) < 100:
            # Insufficient data
            features['cjv_coefficient'] = 0
            features['cv_white_matter'] = 0
            features['cv_gray_matter'] = 0
            features['tissue_contrast_ratio'] = 0
            return features

        hist, bins = np.histogram(brain_intensities, bins=100)
        smoothed = gaussian_filter(hist.astype(float), sigma=2)
        try:
            peaks, _ = find_peaks(smoothed, distance=10)

            if len(peaks) >= 2:
                # Get two highest peaks
                sorted_peaks = sorted(peaks, key=lambda p: smoothed[p], reverse=True)[:2]
                # Sort by position (lower peak = GM, higher = WM typically)
                sorted_peaks = sorted(sorted_peaks)
                gm_peak, wm_peak = sorted_peaks[0], sorted_peaks[1]

                # Define intensity ranges around peaks
                gm_range = (bins[max(0, gm_peak-5)], bins[min(len(bins)-1, gm_peak+5)])
                wm_range = (bins[max(0, wm_peak-5)], bins[min(len(bins)-1, wm_peak+5)])

                gm_mask = (img_data >= gm_range[0]) & (img_data <= gm_range[1]) & brain_mask
                wm_mask = (img_data >= wm_range[0]) & (img_data <= wm_range[1]) & brain_mask

                if gm_mask.sum() > 10 and wm_mask.sum() > 10:
                    gm_values = img_data[gm_mask]
                    wm_values = img_data[wm_mask]

                    # CJV: Coefficient of Joint Variation
                    features['cjv_coefficient'] = (np.std(wm_values) + np.std(gm_values)) / (np.abs(np.mean(wm_values) - np.mean(gm_values)) + 1e-8)

                    # Tissue-specific CV
                    features['cv_white_matter'] = np.std(wm_values) / (np.mean(wm_values) + 1e-8)
                    features['cv_gray_matter'] = np.std(gm_values) / (np.mean(gm_values) + 1e-8)

                    # Tissue contrast
                    features['tissue_contrast_ratio'] = np.abs(np.mean(wm_values) - np.mean(gm_values)) / (np.mean(wm_values) + np.mean(gm_values) + 1e-8)
                else:
                    features['cjv_coefficient'] = 0
                    features['cv_white_matter'] = 0
                    features['cv_gray_matter'] = 0
                    features['tissue_contrast_ratio'] = 0
            else:
                # Less than 2 peaks
                features['cjv_coefficient'] = 0
                features['cv_white_matter'] = 0
                features['cv_gray_matter'] = 0
                features['tissue_contrast_ratio'] = 0
        except:
            features['cjv_coefficient'] = 0
            features['cv_white_matter'] = 0
            features['cv_gray_matter'] = 0
            features['tissue_contrast_ratio'] = 0

        return features
    
    def _extract_size_features(self, df):
        size_features = {}
        for index, fname in enumerate(df['filename'], start=1):
            print(f'Image {index}: Processing size features for {fname}')
            fpath = os.path.join(self._mri_3d_image_data_dir, fname)
            img = nib.load(fpath)
            img_data = img.get_fdata()
            size_features['img_data_size'] = img_data.size
            
            img_resampled = image.resample_to_img(img, self._atlas_img, interpolation='linear', copy_header=True, force_resample=True)
            img_resampled_data = img_resampled.get_fdata()
            size_features['img_resampled_data_size'] = img_resampled_data.size
            
            for i in range(1, len(self._atlas_labels)):
                region_mask = (self._atlas_data == i)
                size_features[f'atlas_region_{i}_mask_size'] = region_mask.size
                region_values = img_resampled_data[region_mask]
                size_features[f'atlas_region_{i}_size'] = region_values.size
                size_features[f'atlas_region_{i}_nonzero'] = (region_values > 0).sum()
                            
            total_nonzero = sum(size_features[f'atlas_region_{i}_nonzero'] for i in range(1, len(self._atlas_labels)))
            size_features['total_region_nonzero'] = total_nonzero
            
            for key, value in size_features.items():
                df.loc[df['filename'] == fname, key] = value
                
        return df
            
    def _extract_computational_features(self, df):
        for index, fname in enumerate(df['filename'], start=1):
            print(f'Image {index}: Processing computational features for {fname}')
            fpath = os.path.join(self._mri_3d_image_data_dir, fname)
            
            file_size = os.path.getsize(fpath)
            df.loc[df['filename'] == fname, 'file_size_bytes'] = file_size
            
            uncompressed_size = 80 * 80 * 41 * 4
            df.loc[df['filename'] == fname, 'compression_ratio'] = file_size / uncompressed_size

            start = time.perf_counter()
            img = nib.load(fpath)
            img_data = img.get_fdata()
            load_time_ms = (time.perf_counter() - start) * 1000
            df.loc[df['filename'] == fname, 'load_time_ms'] = load_time_ms

            raw_bytes = img_data.astype(np.float32).tobytes()
            compressed_bytes = gzip.compress(raw_bytes, compresslevel=6)
            df.loc[df['filename'] == fname, 'true_compression_ratio'] = len(raw_bytes) / len(compressed_bytes)
            
            hist, bins = np.histogram(img_data[img_data > 0], bins=256)
            shannon_entropy = scipy_entropy(hist + 1e-8)
            df.loc[df['filename'] == fname, 'shannon_entropy'] = shannon_entropy
            
            gradient_x = np.gradient(img_data, axis=0)
            gradient_y = np.gradient(img_data, axis=1)
            gradient_z = np.gradient(img_data, axis=2)
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2 + gradient_z**2)
            grad_hist, _ = np.histogram(gradient_magnitude[gradient_magnitude > 0], bins=256)
            grad_entropy = scipy_entropy(grad_hist + 1e-8)
            df.loc[df['filename'] == fname, 'gradient_entropy'] = grad_entropy
            
            normalized_gradient_mean = gradient_magnitude[img_data > 0].mean() / (img_data[img_data > 0].mean() + 1e-8)
            df.loc[df['filename'] == fname, 'normalized_gradient_mean'] = normalized_gradient_mean
            
            fft_data = fftn(img_data)
            fft_magnitude = np.abs(fftshift(fft_data))
            power_spectrum = fft_magnitude.flatten()
            power_spectrum = power_spectrum / np.sum(power_spectrum)
            spectral_entropy = scipy_entropy(power_spectrum[power_spectrum > 0] + 1e-8)
            df.loc[df['filename'] == fname, 'spectral_entropy'] = spectral_entropy

            # Spectral flatness (Wiener entropy)
            psd = fft_magnitude.flatten()
            psd = psd[psd > 0]  # Filter zeros first
            if len(psd) > 0:
                geometric_mean = np.exp(np.mean(np.log(psd)))
                arithmetic_mean = np.mean(psd)
                spectral_flatness = geometric_mean / arithmetic_mean
            else:
                spectral_flatness = 0
            df.loc[df['filename'] == fname, 'spectral_flatness'] = spectral_flatness

            unique_values = len(np.unique(img_data[img_data > 0]))
            if unique_values > 0:
                effective_bits = np.log2(unique_values)
            else:
                effective_bits = 0
            df.loc[df['filename'] == fname, 'effective_bits'] = effective_bits
            
        return df
    
    def _extract_network_derived_features(self, df):
        print('Extracting network-derived features...')
        default_range_cols = [col for col in df.columns if 'Default' in col and '_range' in col]
        if default_range_cols:
            df['default_network_mean_range'] = df[default_range_cols].mean(axis=1)

        default_iqr_cols = [col for col in df.columns if 'Default' in col and '_iqr' in col]
        if default_iqr_cols:
            df['default_network_max_iqr'] = df[default_iqr_cols].max(axis=1)

        salvent_mean_cols = [col for col in df.columns if 'SalVentAttn' in col and '_mean' in col]
        if salvent_mean_cols:
            df['salventattr_network_heterogeneity'] = df[salvent_mean_cols].std(axis=1)

        sommot_range_cols = [col for col in df.columns if 'SomMot' in col and '_range' in col]
        if default_range_cols and sommot_range_cols:
            default_range_mean = df[default_range_cols].mean(axis=1)
            sommot_range_mean = df[sommot_range_cols].mean(axis=1)
            df['default_somatomotor_variability_ratio'] = default_range_mean / (sommot_range_mean + 1e-10)

        frontal_networks = ['Default', 'SalVentAttn', 'Cont']
        posterior_networks = ['Vis', 'DorsAttn']
        frontal_cols = [col for col in df.columns if any(net in col for net in frontal_networks) and '_range' in col]
        posterior_cols = [col for col in df.columns if any(net in col for net in posterior_networks) and '_range' in col]
        if frontal_cols and posterior_cols:
            df['frontal_posterior_variability_gradient'] = df[frontal_cols].mean(axis=1) - df[posterior_cols].mean(axis=1)

        if 'registration_rotation_difference' in df.columns and 'default_network_mean_range' in df.columns:
            df['variability_weighted_registration_error'] = df['registration_rotation_difference'] * df['default_network_mean_range']

        range_cols = [col for col in df.columns if '_range' in col and 'network' not in col]
        if range_cols:
            threshold = df[range_cols].quantile(0.75, axis=1)
            high_var_count = (df[range_cols].T > threshold.values).sum()
            df['high_variability_region_count'] = high_var_count.values

        default_mean_cols = [col for col in df.columns if 'Default' in col and '_mean' in col]
        if default_iqr_cols and default_mean_cols:
            default_iqr = df[default_iqr_cols].mean(axis=1)
            default_mean = df[default_mean_cols].mean(axis=1)
            df['iqr_to_mean_ratio_default'] = default_iqr / (default_mean + 1e-10)

        default_kurtosis_cols = [col for col in df.columns if 'Default' in col and '_kurtosis' in col]
        if default_kurtosis_cols and 'default_network_mean_range' in df.columns:
            default_kurtosis = df[default_kurtosis_cols].mean(axis=1)
            df['kurtosis_variability_interaction'] = default_kurtosis * df['default_network_mean_range']

        if 'watershed_boundary_density' in df.columns and 'default_network_mean_range' in df.columns:
            df['tissue_boundary_variability_index'] = df['watershed_boundary_density'] * df['default_network_mean_range']

        return df

    def _extract_information_theoretic_features(self, df):
        """Extract information-theoretic complexity features

        Computes entropy variants, compression-based complexity, and
        information measures that correlate with computational cost.
        """
        # Load MNI305 template once for NCD calculations
        template_img = nib.load(self._target_image_dir)
        template_data = template_img.get_fdata()
        template_bytes = template_data.astype(np.float32).tobytes()

        for index, fname in enumerate(df['filename'], start=1):
            print(f'Image {index}: Processing information-theoretic features for {fname}')
            fpath = os.path.join(self._mri_3d_image_data_dir, fname)
            img = nib.load(fpath)
            img_data = img.get_fdata()

            # Create brain mask
            if np.any(img_data > 0):
                threshold = np.percentile(img_data[img_data > 0], 10)
            else:
                threshold = 0
            brain_mask = img_data > threshold
            brain_intensities = img_data[brain_mask]

            # Skip if insufficient data
            if len(brain_intensities) < 100:
                for feature_name in ['renyi_entropy_alpha2', 'zlib_compression_ratio',
                                    'ncd_vs_template', 'lempel_ziv_complexity',
                                    'sample_entropy', 'approximate_entropy',
                                    'permutation_entropy', 'wavelet_entropy_db4',
                                    'wavelet_detail_energy_ratio', 'svd_entropy']:
                    df.loc[df['filename'] == fname, feature_name] = 0
                continue

            # 1. Renyi Entropy (alpha=2) - Collision entropy
            hist, _ = np.histogram(brain_intensities, bins=256)
            p = hist / (hist.sum() + 1e-10)
            p = p[p > 0]
            renyi_2 = -np.log2(np.sum(p**2)) if len(p) > 0 else 0
            df.loc[df['filename'] == fname, 'renyi_entropy_alpha2'] = renyi_2

            # 2. Zlib Compression Ratio
            raw_bytes = img_data.astype(np.float32).tobytes()
            compressed_zlib = zlib.compress(raw_bytes, level=6)
            zlib_ratio = len(raw_bytes) / len(compressed_zlib)
            df.loc[df['filename'] == fname, 'zlib_compression_ratio'] = zlib_ratio

            # 3. Normalized Compression Distance (NCD) vs MNI305 template
            img_bytes = img_data.astype(np.float32).tobytes()
            c_img = len(gzip.compress(img_bytes, compresslevel=6))
            c_template = len(gzip.compress(template_bytes, compresslevel=6))
            c_combined = len(gzip.compress(img_bytes + template_bytes, compresslevel=6))
            ncd = (c_combined - min(c_img, c_template)) / max(c_img, c_template)
            df.loc[df['filename'] == fname, 'ncd_vs_template'] = ncd

            # 4. Lempel-Ziv Complexity (subsampled for efficiency)
            subsample_size = min(5000, len(brain_intensities))
            subsampled = np.random.choice(brain_intensities, size=subsample_size, replace=False)

            # Discretize to 16 bins
            n_bins = 16
            quantiles = np.percentile(subsampled, np.linspace(0, 100, n_bins + 1))
            symbols = np.digitize(subsampled, quantiles[:-1]) - 1
            symbols = np.clip(symbols, 0, n_bins - 1)
            symbol_string = ''.join(map(str, symbols))

            # LZ76 algorithm
            n = len(symbol_string)
            c = 1
            i = 0
            while i < n:
                j = i + 1
                while j <= n and symbol_string[i:j] in symbol_string[:i]:
                    j += 1
                c += 1
                i = j

            # Corrected normalization
            lz_complexity = c / (n / np.log2(n)) if n > 1 else 0
            df.loc[df['filename'] == fname, 'lempel_ziv_complexity'] = lz_complexity

            # 5-7. Antropy-based entropy measures
            subsample_antropy = np.random.choice(brain_intensities,
                                                    size=min(5000, len(brain_intensities)),
                                                    replace=False)

            try:
                samp_en = antropy.sample_entropy(subsample_antropy, order=2)
                df.loc[df['filename'] == fname, 'sample_entropy'] = samp_en if not np.isnan(samp_en) else 0
            except:
                df.loc[df['filename'] == fname, 'sample_entropy'] = 0

            try:
                app_en = antropy.app_entropy(subsample_antropy, order=2)
                df.loc[df['filename'] == fname, 'approximate_entropy'] = app_en if not np.isnan(app_en) else 0
            except:
                df.loc[df['filename'] == fname, 'approximate_entropy'] = 0

            try:
                perm_en = antropy.perm_entropy(subsample_antropy, normalize=True)
                df.loc[df['filename'] == fname, 'permutation_entropy'] = perm_en if not np.isnan(perm_en) else 0
            except:
                df.loc[df['filename'] == fname, 'permutation_entropy'] = 0

            try:
                svd_en = antropy.svd_entropy(subsample_antropy, normalize=True)
                df.loc[df['filename'] == fname, 'svd_entropy'] = svd_en if not np.isnan(svd_en) else 0
            except:
                df.loc[df['filename'] == fname, 'svd_entropy'] = 0
        else:
            df.loc[df['filename'] == fname, 'sample_entropy'] = 0
            df.loc[df['filename'] == fname, 'approximate_entropy'] = 0
            df.loc[df['filename'] == fname, 'permutation_entropy'] = 0
            df.loc[df['filename'] == fname, 'svd_entropy'] = 0

            # 8-9. Wavelet Entropy (2D slices approach for 3D efficiency)
            try:
                wavelet = 'db4'
                levels = 4
                slice_indices = [int(img_data.shape[0] * p) for p in [0.3, 0.5, 0.7]]

                all_energies = []
                detail_energy = 0
                total_energy = 0

                for idx in slice_indices:
                    if idx < img_data.shape[0]:
                        slice_2d = img_data[idx, :, :]
                        coeffs = pywt.wavedec2(slice_2d, wavelet, level=levels)

                        # Approximation coefficients energy
                        approx_energy = np.sum(coeffs[0]**2)
                        all_energies.append(approx_energy)
                        total_energy += approx_energy

                        # Detail coefficients energy
                        for level_coeffs in coeffs[1:]:
                            for detail in level_coeffs:  # (cH, cV, cD)
                                detail_e = np.sum(detail**2)
                                all_energies.append(detail_e)
                                detail_energy += detail_e
                                total_energy += detail_e

                # Wavelet entropy
                if total_energy > 0:
                    p_wavelet = np.array(all_energies) / total_energy
                    p_wavelet = p_wavelet[p_wavelet > 0]
                    wavelet_entropy = -np.sum(p_wavelet * np.log2(p_wavelet))
                    detail_ratio = detail_energy / total_energy
                else:
                    wavelet_entropy = 0
                    detail_ratio = 0

                df.loc[df['filename'] == fname, 'wavelet_entropy_db4'] = wavelet_entropy
                df.loc[df['filename'] == fname, 'wavelet_detail_energy_ratio'] = detail_ratio
            except:
                df.loc[df['filename'] == fname, 'wavelet_entropy_db4'] = 0
                df.loc[df['filename'] == fname, 'wavelet_detail_energy_ratio'] = 0

        return df

    def _extract_frequency_texture_features(self, df):
        for index, fname in enumerate(df['filename'], start=1):
            print(f'Image {index}: Processing frequency/texture features for {fname}')
            fpath = os.path.join(self._mri_3d_image_data_dir, fname)
            img = nib.load(fpath)
            img_data = img.get_fdata()

            if np.any(img_data > 0):
                threshold = np.percentile(img_data[img_data > 0], 10)
            else:
                threshold = 0
            brain_mask = img_data > threshold

            if brain_mask.sum() < 100:
                for feature_name in ['glcm_imc2', 'lbp_entropy', 'lbp_uniformity',
                                    'high_freq_energy_ratio', 'spectral_centroid']:
                    df.loc[df['filename'] == fname, feature_name] = 0
                continue

            brain_data = img_data[brain_mask]
            img_normalized = np.zeros_like(img_data)
            if brain_data.std() > 0:
                img_normalized[brain_mask] = ((brain_data - brain_data.min()) /
                                             (brain_data.max() - brain_data.min()) * 255).astype(np.uint8)

            mid_idx = img_normalized.shape[0] // 2
            mid_slice = img_normalized[mid_idx, :, :]

            try:
                angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
                glcm = graycomatrix(mid_slice, [1], angles, 256, symmetric=True, normed=True)
                
                glcm_norm = glcm / (glcm.sum() + 1e-10)
                hxy = -np.sum(glcm_norm * np.log2(glcm_norm + 1e-10))

                px = glcm_norm.sum(axis=1)
                py = glcm_norm.sum(axis=0)
                hx = -np.sum(px * np.log2(px + 1e-10))
                hy = -np.sum(py * np.log2(py + 1e-10))

                imc2 = np.sqrt(max(0, 1 - np.exp(-2 * (hxy - min(hx, hy)))))
                df.loc[df['filename'] == fname, 'glcm_imc2'] = imc2
            except:
                df.loc[df['filename'] == fname, 'glcm_imc2'] = 0

            try:
                radius = 1
                n_points = 8 * radius
                lbp = local_binary_pattern(mid_slice, n_points, radius, method='uniform')

                n_bins = int(lbp.max() + 1)
                lbp_hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins))
                lbp_hist = lbp_hist.astype(float)

                p_lbp = lbp_hist / (lbp_hist.sum() + 1e-10)
                p_lbp = p_lbp[p_lbp > 0]
                lbp_ent = -np.sum(p_lbp * np.log2(p_lbp))
                df.loc[df['filename'] == fname, 'lbp_entropy'] = lbp_ent

                lbp_uniformity = np.sum(p_lbp**2)
                df.loc[df['filename'] == fname, 'lbp_uniformity'] = lbp_uniformity
            except:
                df.loc[df['filename'] == fname, 'lbp_entropy'] = 0
                df.loc[df['filename'] == fname, 'lbp_uniformity'] = 0

            try:
                fft_3d = np.fft.fftn(img_data)
                fft_shifted = np.fft.fftshift(fft_3d)
                magnitude = np.abs(fft_shifted)

                center = np.array(magnitude.shape) // 2
                cutoff = 0.4
                y, x, z = np.ogrid[:magnitude.shape[0], :magnitude.shape[1], :magnitude.shape[2]]
                dist_from_center = np.sqrt(((y - center[0]) / center[0])**2 +
                                        ((x - center[1]) / center[1])**2 +
                                        ((z - center[2]) / center[2])**2)
                high_freq_mask = dist_from_center > cutoff

                total_energy = np.sum(magnitude**2)
                high_freq_energy = np.sum(magnitude[high_freq_mask]**2)
                high_freq_ratio = high_freq_energy / (total_energy + 1e-10)
                df.loc[df['filename'] == fname, 'high_freq_energy_ratio'] = high_freq_ratio

                magnitude_flat = magnitude.flatten()
                freqs = np.arange(len(magnitude_flat))
                spectral_centroid = np.sum(freqs * magnitude_flat) / (np.sum(magnitude_flat) + 1e-10)
                spectral_centroid_norm = spectral_centroid / len(magnitude_flat)
                df.loc[df['filename'] == fname, 'spectral_centroid'] = spectral_centroid_norm
            except:
                df.loc[df['filename'] == fname, 'high_freq_energy_ratio'] = 0
                df.loc[df['filename'] == fname, 'spectral_centroid'] = 0

        return df

    def _extract_preprocessing_proxy_features(self, df):
        for index, fname in enumerate(df['filename'], start=1):
            print(f'Image {index}: Processing preprocessing proxy features for {fname}')
            fpath = os.path.join(self._mri_3d_image_data_dir, fname)
            img = nib.load(fpath)
            img_data = img.get_fdata()

            if np.any(img_data > 0):
                threshold = np.percentile(img_data[img_data > 0], 10)
            else:
                threshold = 0
            brain_mask = img_data > threshold

            if brain_mask.sum() < 100:
                df.loc[df['filename'] == fname, 'intensity_uniformity'] = 0
                continue

            try:
                patch_size = 8
                local_means = []

                for i in range(0, img_data.shape[0] - patch_size, patch_size):
                    for j in range(0, img_data.shape[1] - patch_size, patch_size):
                        for k in range(0, img_data.shape[2] - patch_size, patch_size):
                            patch = img_data[i:i+patch_size, j:j+patch_size, k:k+patch_size]
                            patch_mask = brain_mask[i:i+patch_size, j:j+patch_size, k:k+patch_size]

                            if patch_mask.sum() > (patch_size**3 * 0.5):
                                local_means.append(patch[patch_mask].mean())

                if len(local_means) > 1:
                    global_mean = img_data[brain_mask].mean()
                    intensity_uniformity = 1 - (np.std(local_means) / (global_mean + 1e-8))
                    intensity_uniformity = max(0, min(1, intensity_uniformity))
                else:
                    intensity_uniformity = 0

                df.loc[df['filename'] == fname, 'intensity_uniformity'] = intensity_uniformity
            except:
                df.loc[df['filename'] == fname, 'intensity_uniformity'] = 0

        return df

    def _extract_topological_complexity_features(self, df):
        for index, fname in enumerate(df['filename'], start=1):
            print(f'Image {index}: Processing topological features for {fname}')
            fpath = os.path.join(self._mri_3d_image_data_dir, fname)
            img = nib.load(fpath)
            img_data = img.get_fdata()
            voxel_spacing = img.header.get_zooms()[:3]

            if np.any(img_data > 0):
                threshold = np.percentile(img_data[img_data > 0], 10)
            else:
                threshold = 0
            brain_mask = img_data > threshold

            if brain_mask.sum() < 100:
                for feature_name in ['euler_characteristic', 'euler_density',
                                    'betti_0_estimate', 'betti_1_lower_bound',
                                    'perimeter_complexity', 'compactness_3d',
                                    'surface_area_volume_ratio', 'boundary_fractal_proxy']:
                    df.loc[df['filename'] == fname, feature_name] = 0
                continue

            try:
                euler_char = euler_number(brain_mask, connectivity=1)
                volume = brain_mask.sum()
                euler_density = euler_char / volume if volume > 0 else 0

                df.loc[df['filename'] == fname, 'euler_characteristic'] = euler_char
                df.loc[df['filename'] == fname, 'euler_density'] = euler_density

                from scipy.ndimage import label
                labeled_mask, betti_0 = label(brain_mask)
                df.loc[df['filename'] == fname, 'betti_0_estimate'] = betti_0

                betti_1_lb = max(0, betti_0 - euler_char)
                df.loc[df['filename'] == fname, 'betti_1_lower_bound'] = betti_1_lb

                axial_projection = brain_mask.sum(axis=0) > 0
                from skimage.measure import perimeter
                perim = perimeter(axial_projection)
                area = axial_projection.sum()
                perimeter_complex = perim / (np.sqrt(area) + 1e-8) if area > 0 else 0
                df.loc[df['filename'] == fname, 'perimeter_complexity'] = perimeter_complex

                try:
                    verts, faces, _, _ = marching_cubes(brain_mask.astype(float),
                                                    level=0.5,
                                                    spacing=voxel_spacing)

                    v0 = verts[faces[:, 0]]
                    v1 = verts[faces[:, 1]]
                    v2 = verts[faces[:, 2]]
                    cross_product = np.cross(v1 - v0, v2 - v0)
                    triangle_areas = np.linalg.norm(cross_product, axis=1) / 2
                    surface_area = triangle_areas.sum()

                    voxel_volume = np.prod(voxel_spacing)
                    total_volume = volume * voxel_volume

                    if surface_area > 0:
                        compactness = (36 * np.pi * total_volume**2) / (surface_area**3)
                        sa_vol_ratio = surface_area / total_volume
                    else:
                        compactness = 0
                        sa_vol_ratio = 0

                    df.loc[df['filename'] == fname, 'compactness_3d'] = compactness
                    df.loc[df['filename'] == fname, 'surface_area_volume_ratio'] = sa_vol_ratio
                except:
                    df.loc[df['filename'] == fname, 'compactness_3d'] = 0
                    df.loc[df['filename'] == fname, 'surface_area_volume_ratio'] = 0

                from scipy.ndimage import binary_erosion
                perimeters_at_scales = []
                current_mask = brain_mask.copy()

                for scale in range(1, 4):
                    if current_mask.sum() > 100:
                        projection = current_mask.sum(axis=0) > 0
                        if projection.sum() > 0:
                            p = perimeter(projection)
                            perimeters_at_scales.append(p)
                        current_mask = binary_erosion(current_mask)

                if len(perimeters_at_scales) > 1:
                    fractal_proxy = np.var(perimeters_at_scales) / (np.mean(perimeters_at_scales) + 1e-8)
                else:
                    fractal_proxy = 0

                df.loc[df['filename'] == fname, 'boundary_fractal_proxy'] = fractal_proxy

            except Exception as e:
                for feature_name in ['euler_characteristic', 'euler_density',
                                    'betti_0_estimate', 'betti_1_lower_bound',
                                    'perimeter_complexity', 'compactness_3d',
                                    'surface_area_volume_ratio', 'boundary_fractal_proxy']:
                    df.loc[df['filename'] == fname, feature_name] = 0

        return df

    def _extract_deep_learning_predictors(self, df):
        for index, fname in enumerate(df['filename'], start=1):
            print(f'Image {index}: Processing deep learning predictors for {fname}')
            fpath = os.path.join(self._mri_3d_image_data_dir, fname)
            img = nib.load(fpath)
            img_data = img.get_fdata()

            if np.any(img_data > 0):
                threshold = np.percentile(img_data[img_data > 0], 10)
            else:
                threshold = 0
            brain_mask = img_data > threshold

            if brain_mask.sum() < 100:
                for feature_name in ['tile_activation_density_mean',
                                    'tile_gradient_load_imbalance',
                                    'tile_boundary_complexity']:
                    df.loc[df['filename'] == fname, feature_name] = 0
                continue

            try:
                tile_shape = (16, 16, 8)  # Smaller for computational efficiency
                activation_densities = []
                gradient_magnitudes = []
                boundary_densities = []

                for i in range(0, img_data.shape[0] - tile_shape[0], tile_shape[0]):
                    for j in range(0, img_data.shape[1] - tile_shape[1], tile_shape[1]):
                        for k in range(0, img_data.shape[2] - tile_shape[2], tile_shape[2]):
                            tile = img_data[i:i+tile_shape[0],
                                        j:j+tile_shape[1],
                                        k:k+tile_shape[2]]
                            tile_mask = brain_mask[i:i+tile_shape[0],
                                                j:j+tile_shape[1],
                                                k:k+tile_shape[2]]

                            if tile_mask.sum() > (tile_shape[0] * tile_shape[1] * tile_shape[2] * 0.1):
                                tile_data = tile[tile_mask]

                                high_intensity_threshold = np.percentile(tile_data, 75)
                                activation_density = (tile_data > high_intensity_threshold).sum() / len(tile_data)
                                activation_densities.append(activation_density)

                                try:
                                    tile_gradients = np.gradient(tile)
                                    grad_magnitude = np.sqrt(sum(g**2 for g in tile_gradients))
                                    avg_grad = grad_magnitude[tile_mask].mean()
                                    gradient_magnitudes.append(avg_grad)
                                except:
                                    pass

                                try:
                                    from skimage.filters import sobel
                                    edges = sobel(tile)
                                    edge_density = edges[tile_mask].mean()
                                    boundary_densities.append(edge_density)
                                except:
                                    pass

                if len(activation_densities) > 0:
                    tile_activation_mean = np.mean(activation_densities)
                else:
                    tile_activation_mean = 0
                df.loc[df['filename'] == fname, 'tile_activation_density_mean'] = tile_activation_mean

                if len(gradient_magnitudes) > 1:
                    load_imbalance = np.max(gradient_magnitudes) / (np.mean(gradient_magnitudes) + 1e-8)
                else:
                    load_imbalance = 0
                df.loc[df['filename'] == fname, 'tile_gradient_load_imbalance'] = load_imbalance

                if len(boundary_densities) > 0:
                    boundary_complex = np.mean(boundary_densities)
                else:
                    boundary_complex = 0
                df.loc[df['filename'] == fname, 'tile_boundary_complexity'] = boundary_complex

            except Exception as e:
                df.loc[df['filename'] == fname, 'tile_activation_density_mean'] = 0
                df.loc[df['filename'] == fname, 'tile_gradient_load_imbalance'] = 0
                df.loc[df['filename'] == fname, 'tile_boundary_complexity'] = 0

        return df

    def _extract_metadata_features(self, df):
        for index, fname in enumerate(df['filename'], start=1):
            print(f'Image {index}: Processing metadata features for {fname}')
            fpath = os.path.join(self._mri_3d_image_data_dir, fname)
            img = nib.load(fpath)
            img_data = img.get_fdata()
            header = img.header
            affine = img.affine

            try:
                voxel_count = np.prod(img_data.shape[:3])
                df.loc[df['filename'] == fname, 'voxel_count'] = voxel_count

                voxel_spacing = header.get_zooms()[:3]
                voxel_spacing_product = np.prod(voxel_spacing)
                df.loc[df['filename'] == fname, 'voxel_spacing_product'] = voxel_spacing_product

                data_dtype = header.get_data_dtype()
                data_type_bytes = data_dtype.itemsize
                df.loc[df['filename'] == fname, 'data_type_bytes'] = data_type_bytes

                # Indicates scaling/rotation in transformation
                affine_det = np.linalg.det(affine[:3, :3])
                df.loc[df['filename'] == fname, 'affine_determinant'] = affine_det

                try:
                    header_size = header['sizeof_hdr']
                    df.loc[df['filename'] == fname, 'header_data_size'] = header_size
                except:
                    # Some NIfTI versions may not have this field
                    df.loc[df['filename'] == fname, 'header_data_size'] = 0

            except Exception as e:
                df.loc[df['filename'] == fname, 'voxel_count'] = 0
                df.loc[df['filename'] == fname, 'voxel_spacing_product'] = 0
                df.loc[df['filename'] == fname, 'data_type_bytes'] = 0
                df.loc[df['filename'] == fname, 'affine_determinant'] = 0
                df.loc[df['filename'] == fname, 'header_data_size'] = 0

        return df

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

    def _compute_glcm_entropy_dmn(self, img_data_resampled, atlas_data, atlas_labels):
        try:
            dmn_indices = [i for i, label in enumerate(atlas_labels)
                        if isinstance(label, str) and 'Default' in label]

            if len(dmn_indices) == 0:
                return 0.0

            dmn_mask = np.isin(atlas_data, dmn_indices)
            if dmn_mask.sum() < 100:
                return 0.0

            dmn_values = img_data_resampled[dmn_mask]
            min_val, max_val = dmn_values.min(), dmn_values.max()
            if max_val - min_val < 1e-8:
                return 0.0

            img_normalized = np.clip(
                ((img_data_resampled - min_val) / (max_val - min_val) * 255),
                0, 255
            ).astype(np.uint8)

            entropies = []
            shape = img_data_resampled.shape

            for axis in range(3):
                for slice_pct in [0.3, 0.5, 0.7]:
                    idx = int(shape[axis] * slice_pct)

                    if axis == 0:
                        slice_2d = img_normalized[idx, :, :]
                        mask_2d = dmn_mask[idx, :, :]
                    elif axis == 1:
                        slice_2d = img_normalized[:, idx, :]
                        mask_2d = dmn_mask[:, idx, :]
                    else:
                        slice_2d = img_normalized[:, :, idx]
                        mask_2d = dmn_mask[:, :, idx]

                    if mask_2d.sum() > 50:
                        masked_slice = slice_2d * mask_2d.astype(np.uint8)
                        try:
                            glcm = graycomatrix(masked_slice, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4],
                                                256, symmetric=True, normed=True)
                            
                            glcm_sum = glcm.sum(axis=(2, 3))
                            glcm_norm = glcm_sum / (glcm_sum.sum() + 1e-10)
                            glcm_flat = glcm_norm.flatten()
                            glcm_flat = glcm_flat[glcm_flat > 0]
                            entropy = -np.sum(glcm_flat * np.log2(glcm_flat + 1e-10))
                            entropies.append(entropy)
                        except:
                            pass

            return np.mean(entropies) if entropies else 0.0
        except:
            return 0.0

    def _compute_local_entropy_variance(self, img_data, brain_mask, patch_size=8):
        try:
            entropies = []
            shape = img_data.shape

            for i in range(0, shape[0], patch_size):
                for j in range(0, shape[1], patch_size):
                    for k in range(0, shape[2], patch_size):
                        patch = img_data[i:min(i+patch_size, shape[0]),
                                        j:min(j+patch_size, shape[1]),
                                        k:min(k+patch_size, shape[2])]
                        patch_mask = brain_mask[i:min(i+patch_size, shape[0]),
                                            j:min(j+patch_size, shape[1]),
                                            k:min(k+patch_size, shape[2])]

                        if patch_mask.sum() < 20:
                            continue

                        patch_vals = patch[patch_mask]
                        hist, _ = np.histogram(patch_vals, bins=32, density=True)
                        hist = hist[hist > 0]
                        if len(hist) > 1:
                            entropy = -np.sum(hist * np.log2(hist + 1e-10))
                            entropies.append(entropy)

            if len(entropies) < 10:
                return 0.0

            return np.var(entropies)
        except:
            return 0.0

    def _compute_multiscale_entropy_index(self, img_data, brain_mask, max_scale=5):
        try:
            brain_vals = img_data[brain_mask]
            if len(brain_vals) < 500:
                return 0.0

            n_samples = min(5000, len(brain_vals))
            rng = np.random.RandomState(42)  # Reproducibility
            signal = rng.choice(brain_vals, size=n_samples, replace=False)

            signal = (signal - signal.mean()) / (signal.std() + 1e-10)

            mse_sum = 0.0
            for scale in range(1, max_scale + 1):
                n_coarse = len(signal) // scale
                if n_coarse < 100:  # Need sufficient length for SampEn
                    break

                coarse_signal = signal[:n_coarse * scale].reshape(-1, scale).mean(axis=1)

                try:
                    samp_en = antropy.sample_entropy(coarse_signal, order=2)
                    if not np.isnan(samp_en) and not np.isinf(samp_en):
                        mse_sum += samp_en
                except:
                    pass

            return mse_sum
        except:
            return 0.0

    def _compute_tissue_boundary_count(self, img_data, brain_mask):
        try:
            brain_vals = img_data[brain_mask]
            if len(brain_vals) < 500:
                return {'tissue_boundary_count': 0, 'tissue_boundary_density': 0}

            hist, bins = np.histogram(brain_vals, bins=100)
            hist_smooth = gaussian_filter(hist.astype(float), sigma=2)

            peaks, _ = find_peaks(hist_smooth, prominence=hist_smooth.max() * 0.05, distance=10)

            if len(peaks) < 2:
                return {'tissue_boundary_count': 0, 'tissue_boundary_density': 0}

            bin_centers = (bins[:-1] + bins[1:]) / 2
            peak_intensities = bin_centers[peaks]
            thresholds = (peak_intensities[:-1] + peak_intensities[1:]) / 2

            tissue_labels = np.zeros_like(img_data, dtype=int)
            tissue_labels[brain_mask] = 1
            for i, thresh in enumerate(thresholds):
                tissue_labels[img_data > thresh] = i + 2

            boundary_count = 0
            struct = ball(1)

            for label_val in range(1, len(peaks) + 1):
                label_mask = (tissue_labels == label_val)
                dilated = binary_dilation(label_mask, structure=struct)
                boundary = dilated & (tissue_labels > 0) & (~label_mask)
                boundary_count += boundary.sum()

            boundary_count = boundary_count // 2
            boundary_density = boundary_count / (brain_mask.sum() + 1e-10)

            return {'tissue_boundary_count': boundary_count,
                    'tissue_boundary_density': boundary_density}
        except:
            return {'tissue_boundary_count': 0, 'tissue_boundary_density': 0}

    def _compute_label_entropy(self, img_data_resampled, atlas_data, atlas_labels):
        try:
            label_counts = []
            for i in range(1, len(atlas_labels)):
                region_mask = (atlas_data == i) & (img_data_resampled > 0)
                label_counts.append(region_mask.sum())

            label_counts = np.array(label_counts)
            total = label_counts.sum()

            if total == 0:
                return 0.0

            p = label_counts / total
            p = p[p > 0]

            return -np.sum(p * np.log2(p + 1e-10))
        except:
            return 0.0

    def _compute_efc(self, img_data):
        try:
            flat = img_data.flatten()
            flat = flat[flat > 0]

            if len(flat) < 100:
                return 0.0

            x_max = flat.max()
            if x_max < 1e-10:
                return 0.0

            p = flat / x_max
            p = p[p > 0]

            efc = -np.sum(p * np.log(p + 1e-10))

            n = len(p)
            efc_max = n * np.log(n) if n > 1 else 1.0

            return efc / efc_max
        except:
            return 0.0

    def _compute_cnr(self, img_data, brain_mask):
        try:
            brain_vals = img_data[brain_mask]
            if len(brain_vals) < 500:
                return 0.0

            hist, bins = np.histogram(brain_vals, bins=100)
            hist_smooth = gaussian_filter(hist.astype(float), sigma=2)
            peaks, _ = find_peaks(hist_smooth, prominence=hist_smooth.max() * 0.05, distance=10)

            if len(peaks) < 2:
                return 0.0

            bin_centers = (bins[:-1] + bins[1:]) / 2
            peak_values = bin_centers[peaks]
            gm_peak = peak_values[0]
            wm_peak = peak_values[-1]

            gm_vals = brain_vals[(brain_vals > gm_peak * 0.8) & (brain_vals < gm_peak * 1.2)]
            wm_vals = brain_vals[(brain_vals > wm_peak * 0.8) & (brain_vals < wm_peak * 1.2)]

            if len(gm_vals) < 10 or len(wm_vals) < 10:
                return 0.0

            mu_gm, sigma_gm = gm_vals.mean(), gm_vals.std()
            mu_wm, sigma_wm = wm_vals.mean(), wm_vals.std()

            cnr = np.abs(mu_wm - mu_gm) / np.sqrt((sigma_wm**2 + sigma_gm**2) / 2 + 1e-10)

            return cnr
        except:
            return 0.0

    def _compute_fwhm(self, img_data, brain_mask, voxel_spacing):

        try:
            grad_x = np.gradient(img_data, axis=0)
            grad_y = np.gradient(img_data, axis=1)
            grad_z = np.gradient(img_data, axis=2)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)

            brain_signal = img_data[brain_mask]
            brain_grad = grad_mag[brain_mask]

            if len(brain_signal) < 100:
                return 0.0
            
            signal_range = np.percentile(brain_signal, 95) - np.percentile(brain_signal, 5)
            grad_peak = np.percentile(brain_grad, 95)

            if grad_peak < 1e-10:
                return 0.0

            sigma_voxels = signal_range / (grad_peak * np.sqrt(2 * np.pi) + 1e-10)
            sigma_mm = sigma_voxels * np.mean(voxel_spacing)

            fwhm_mm = 2.355 * sigma_mm

            return fwhm_mm
        except:
            return 0.0

    def _compute_snr_dietrich(self, img_data, brain_mask):
        try:
            signal = img_data[brain_mask]
            if len(signal) < 100:
                return 0.0

            mu_signal = signal.mean()

            corner_size = 5
            shape = img_data.shape
            corners = []
            for x in [slice(0, corner_size), slice(-corner_size, None)]:
                for y in [slice(0, corner_size), slice(-corner_size, None)]:
                    for z in [slice(0, corner_size), slice(-corner_size, None)]:
                        corners.append(img_data[x, y, z].flatten())

            background = np.concatenate(corners)
            sigma_background = background.std()

            if sigma_background < 1e-10:
                return 0.0

            rayleigh_factor = np.sqrt(2 / (4 - np.pi))

            snr = mu_signal / (sigma_background * rayleigh_factor)

            return snr
        except:
            return 0.0

    def _extract_hypothesis_features(self, df):
        for index, fname in enumerate(df['filename'], start=1):
            print(f'Image {index}: Processing hypothesis features for {fname}')

            fpath = os.path.join(self._mri_3d_image_data_dir, fname)
            img = nib.load(fpath)
            img_data = img.get_fdata()
            voxel_spacing = img.header.get_zooms()[:3]

            if np.any(img_data > 0):
                threshold = np.percentile(img_data[img_data > 0], 10)
            else:
                threshold = 0
            brain_mask = img_data > threshold

            if brain_mask.sum() < 100:
                continue

            df.loc[df['filename'] == fname, 'efc'] = self._compute_efc(img_data)
            df.loc[df['filename'] == fname, 'snr_dietrich'] = self._compute_snr_dietrich(img_data, brain_mask)
            df.loc[df['filename'] == fname, 'cnr'] = self._compute_cnr(img_data, brain_mask)
            df.loc[df['filename'] == fname, 'fwhm'] = self._compute_fwhm(img_data, brain_mask, voxel_spacing)

            ngtdm = self._compute_ngtdm_features(img_data, brain_mask)
            for key, value in ngtdm.items():
                df.loc[df['filename'] == fname, key] = value

            df.loc[df['filename'] == fname, 'local_entropy_variance'] = \
                self._compute_local_entropy_variance(img_data, brain_mask)
            df.loc[df['filename'] == fname, 'mse_complexity_index'] = \
                self._compute_multiscale_entropy_index(img_data, brain_mask)

            tissue = self._compute_tissue_boundary_count(img_data, brain_mask)
            for key, value in tissue.items():
                df.loc[df['filename'] == fname, key] = value

            img_resampled = image.resample_to_img(img, self._atlas_img, interpolation='linear')
            img_data_resampled = img_resampled.get_fdata()

            df.loc[df['filename'] == fname, 'label_entropy'] = \
                self._compute_label_entropy(img_data_resampled, self._atlas_data, self._atlas_labels)
            df.loc[df['filename'] == fname, 'glcm_entropy_dmn'] = \
                self._compute_glcm_entropy_dmn(img_data_resampled, self._atlas_data, self._atlas_labels)

        return df

    def _extract_region_features(self):
        """Extract features using nilearn atlas

        Returns:
            pandas.DataFrame: DataFrame with extracted region features
        """
        all_features = {} 
        valid_regions = set()
        for index, fname in enumerate(os.listdir(self._mri_3d_image_data_dir), start=1):
            print(f'Image {index}: Processing Atlas feature extraction for {fname}')
            if fname.endswith('.nii') or fname.endswith('.nii.gz'):
                try:
                    fpath = os.path.join(self._mri_3d_image_data_dir, fname)
                    img = nib.load(fpath)
                    img_resampled = image.resample_to_img(img, self._atlas_img, interpolation='linear', copy_header=True, force_resample=True)
                    img_data = img_resampled.get_fdata()
                    self._atlas_data = self._atlas_img.get_fdata()

                    file_features = {}
                    for i in range(1, len(self._atlas_labels)):
                        region_mask = (self._atlas_data == i)
                        region_values = img_data[region_mask]
                        stats = self._extract_region_stats(region_values)
                        file_features[i] = stats
                        
                        if stats is not None:
                            valid_regions.add(i)

                    all_features[fname] = file_features
                    
                except Exception as e:
                    print(f'Error processing {fname}: {e}')
        
        self._valid_regions = sorted(valid_regions)
        
        records = []
        for fname, file_features in all_features.items():
            feature_row = [fname]
            for i in self._valid_regions:
                stats = file_features.get(i)
                if stats is not None:
                    feature_row.extend(stats)
            records.append(feature_row)
        
        return records

    def _extract_region_stats(self, region_data):
        """Extract statistics from a specific brain region

        Args:
            region_data (np.ndarray): region values extracted

        Returns:
            list: List of extracted statistics
        """
        region_data = region_data[region_data > 0]
        if len(region_data) == 0:
            return None        
        stats = [
            np.mean(region_data),
            np.std(region_data),
            median_abs_deviation(region_data),
            kurtosis(region_data),
            np.percentile(region_data, 95),
            np.percentile(region_data, 75),
            np.ptp(region_data),
            moment(region_data, moment=5),
            moment(region_data, moment=6),
            (np.percentile(region_data, 75) - np.percentile(region_data, 25))
        ]
        return stats
        
    def _setup_columns(self):
        """Setup the columns for the feature DataFrame

        Returns:
            list: List of column names for the DataFrame
        """
        columns = ['filename']
        for region_idx in getattr(self, '_valid_regions', range(1, len(self._atlas_labels))):
            label = self._atlas_labels[region_idx]
            for stat in ['mean', 'std', 'med_abs_dev', 'kurtosis',
                        'pct_95', 'pct_75', 'range', 'moment_5', 'moment_6', 'iqr']:
                columns.append(f'{label}_{stat}')
        return columns

    # =====================================================================
    # SynthSeg Batch Runner and Feature Extraction
    # =====================================================================

    def run_synthseg_batch(self, skip_existing=True, use_robust=True,
                        use_fast=False, use_cpu=False, crop_size=192):
        """Run SynthSeg on all 3D images in the dataset.

        Uses subprocess to call SynthSeg CLI in Python 3.8 environment.

        Args:
            skip_existing: Skip images that already have SynthSeg outputs
            use_robust: Use robust mode for better handling of low-quality images
            use_fast: Use fast mode (slightly less accurate but faster)
            use_cpu: Force CPU mode (use if no GPU available)
            crop_size: Size of 3D patches to analyze (default 192). Lower values use less memory.

        Returns:
            dict: Dictionary with paths to combined outputs
        """
        if not self._check_synthseg_environment():
            print(f"Error: Conda environment '{SYNTHSEG_CONDA_ENV}' not found.")
            print("Create it with: conda create -n synthseg_38 python=3.8 -y")
            print("Then install: pip install -r SynthSeg/requirements_python3.8.txt")
            return None

        os.makedirs(self._synthseg_output_dir, exist_ok=True)

        nifti_files = []
        for ext in ['*.nii', '*.nii.gz']:
            nifti_files.extend(glob_module.glob(
                os.path.join(self._mri_3d_image_data_dir, ext)))

        print(f"Found {len(nifti_files)} images to process with SynthSeg")
        all_volumes = []
        all_qc = []

        for idx, input_path in enumerate(nifti_files, start=1):
            base_name = os.path.basename(input_path).replace('.nii.gz', '').replace('.nii', '')
            seg_path = os.path.join(self._synthseg_output_dir, f'{base_name}_synthseg.nii.gz')
            vol_path = os.path.join(self._synthseg_output_dir, f'{base_name}_volumes.csv')
            qc_path = os.path.join(self._synthseg_output_dir, f'{base_name}_qc.csv')

            if skip_existing and os.path.exists(seg_path) and os.path.exists(vol_path):
                print(f"[{idx}/{len(nifti_files)}] Skipping {base_name} (already processed)")
                if os.path.exists(vol_path):
                    vol_df = pd.read_csv(vol_path)
                    vol_df['filename'] = os.path.basename(input_path)
                    all_volumes.append(vol_df)
                if os.path.exists(qc_path):
                    qc_df = pd.read_csv(qc_path)
                    qc_df['filename'] = os.path.basename(input_path)
                    all_qc.append(qc_df)
                continue

            print(f"[{idx}/{len(nifti_files)}] Processing {base_name}...")

            try:
                self._run_synthseg_cli(
                    input_path=input_path,
                    seg_path=seg_path,
                    vol_path=vol_path,
                    qc_path=qc_path,
                    robust=use_robust,
                    fast=use_fast,
                    use_cpu=use_cpu,
                    crop_size=crop_size
                )

                vol_df = pd.read_csv(vol_path)
                vol_df['filename'] = os.path.basename(input_path)
                all_volumes.append(vol_df)

                if os.path.exists(qc_path):
                    qc_df = pd.read_csv(qc_path)
                    qc_df['filename'] = os.path.basename(input_path)
                    all_qc.append(qc_df)

                print(f"  Successfully processed {base_name}")

            except Exception as e:
                print(f"  Error processing {base_name}: {e}")
                continue

            finally:
                # Clean up memory and give TensorFlow time to fully release resources
                gc.collect()
                time.sleep(5)  # Longer pause to allow TF cleanup between subprocess calls

        if all_volumes:
            combined_volumes = pd.concat(all_volumes, ignore_index=True)
            combined_volumes.to_csv(self._synthseg_volumes_csv, index=False)
            print(f"Saved combined volumes to {self._synthseg_volumes_csv}")

        if all_qc:
            combined_qc = pd.concat(all_qc, ignore_index=True)
            combined_qc.to_csv(self._synthseg_qc_csv, index=False)
            print(f"Saved combined QC scores to {self._synthseg_qc_csv}")

        return {
            'volumes_csv': self._synthseg_volumes_csv,
            'qc_csv': self._synthseg_qc_csv,
            'output_dir': self._synthseg_output_dir
        }

    def _extract_synthseg_features(self, df):
        """Extract volumetric and QC features from SynthSeg outputs.

        Requires: run_synthseg_batch() to have been executed first.
        """
        if not os.path.exists(self._synthseg_volumes_csv):
            print("Warning: SynthSeg volumes CSV not found. Run run_synthseg_batch() first.")
            print("Skipping SynthSeg features.")
            return df

        print("Extracting SynthSeg features...")

        volumes_df = pd.read_csv(self._synthseg_volumes_csv)

        qc_df = None
        if os.path.exists(self._synthseg_qc_csv):
            qc_df = pd.read_csv(self._synthseg_qc_csv)

        for index, fname in enumerate(df['filename'], start=1):
            print(f'Image {index}: Processing SynthSeg features for {fname}')

            vol_row = volumes_df[volumes_df['filename'] == fname]
            if vol_row.empty:
                print(f"  Warning: No SynthSeg volumes found for {fname}")
                continue

            vol_row = vol_row.iloc[0]
            vol_cols = [c for c in volumes_df.columns if c not in ['filename', 'subject', 'Unnamed: 0']]
            volumes = {col: vol_row[col] for col in vol_cols if col in vol_row.index}

            left_wm = volumes.get('left cerebral white matter', 0)
            right_wm = volumes.get('right cerebral white matter', 0)
            left_cortex = volumes.get('left cerebral cortex', 0)
            right_cortex = volumes.get('right cerebral cortex', 0)

            vent_keys = [k for k in volumes.keys() if 'ventricle' in k.lower()]
            total_ventricular = sum(volumes.get(k, 0) for k in vent_keys)

            csf_keys = [k for k in volumes.keys() if 'csf' in k.lower() or 'ventricle' in k.lower()]
            non_csf_keys = [k for k in volumes.keys() if k not in csf_keys]
            total_brain = sum(volumes.get(k, 0) for k in non_csf_keys)

            total_icv = sum(volumes.values())

            df.loc[df['filename'] == fname, 'synthseg_total_brain_volume'] = total_brain
            df.loc[df['filename'] == fname, 'synthseg_intracranial_volume'] = total_icv
            df.loc[df['filename'] == fname, 'synthseg_total_ventricular_volume'] = total_ventricular
            df.loc[df['filename'] == fname, 'synthseg_ventricular_ratio'] = total_ventricular / (total_brain + 1e-8)

            df.loc[df['filename'] == fname, 'synthseg_wm_asymmetry'] = abs(left_wm - right_wm) / (left_wm + right_wm + 1e-8)
            df.loc[df['filename'] == fname, 'synthseg_cortex_asymmetry'] = abs(left_cortex - right_cortex) / (left_cortex + right_cortex + 1e-8)

            subcortical_keys = ['thalamus', 'caudate', 'putamen', 'pallidum', 'hippocampus', 'amygdala', 'accumbens']
            total_subcortical = 0
            for struct in subcortical_keys:
                for side in ['left', 'right']:
                    key = f'{side} {struct}'
                    total_subcortical += volumes.get(key, 0)
            df.loc[df['filename'] == fname, 'synthseg_total_subcortical_volume'] = total_subcortical

            hippocampal = volumes.get('left hippocampus', 0) + volumes.get('right hippocampus', 0)
            df.loc[df['filename'] == fname, 'synthseg_hippocampal_volume'] = hippocampal

            cerebellar = (volumes.get('left cerebellum cortex', 0) + volumes.get('right cerebellum cortex', 0) +
                         volumes.get('left cerebellum white matter', 0) + volumes.get('right cerebellum white matter', 0))
            df.loc[df['filename'] == fname, 'synthseg_cerebellar_volume'] = cerebellar

            df.loc[df['filename'] == fname, 'synthseg_brainstem_volume'] = volumes.get('brain-stem', 0)
            df.loc[df['filename'] == fname, 'synthseg_csf_volume'] = volumes.get('CSF', 0)

            total_wm = left_wm + right_wm
            total_gm = left_cortex + right_cortex
            df.loc[df['filename'] == fname, 'synthseg_wm_gm_ratio'] = total_wm / (total_gm + 1e-8)

            if qc_df is not None:
                qc_row = qc_df[qc_df['filename'] == fname]
                if not qc_row.empty:
                    qc_cols = [c for c in qc_df.columns if c not in ['filename', 'subject']]
                    qc_score = qc_row[qc_cols[-1]].values[0] if qc_cols else 0
                    df.loc[df['filename'] == fname, 'synthseg_qc_score'] = qc_score
                    df.loc[df['filename'] == fname, 'synthseg_qc_low_flag'] = 1 if qc_score < 0.6 else 0

            seg_path = os.path.join(self._synthseg_output_dir,
                                    fname.replace('.nii.gz', '').replace('.nii', '') + '_synthseg.nii.gz')
            if os.path.exists(seg_path):
                complexity = self._compute_segmentation_complexity(seg_path)
                for key, value in complexity.items():
                    df.loc[df['filename'] == fname, f'synthseg_{key}'] = value

        return df

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

    def _get_tissue_masks_from_synthseg(self, seg_path):
        """Extract GM, WM, and CSF masks from SynthSeg segmentation.

        SynthSeg labels (FreeSurfer convention):
        - GM: 3, 42 (cerebral cortex), 8, 47 (cerebellar cortex)
        - WM: 2, 41 (cerebral WM), 7, 46 (cerebellar WM)
        - CSF: 4, 5, 43, 44 (ventricles), 14, 15, 24 (3rd, 4th ventricle, CSF)
        """
        seg_img = nib.load(seg_path)
        seg_data = seg_img.get_fdata().astype(int)

        gm_labels = [3, 42, 8, 47]
        wm_labels = [2, 41, 7, 46]
        csf_labels = [4, 5, 43, 44, 14, 15, 24]

        gm_mask = np.isin(seg_data, gm_labels)
        wm_mask = np.isin(seg_data, wm_labels)
        csf_mask = np.isin(seg_data, csf_labels)

        return gm_mask, wm_mask, csf_mask

    # =====================================================================
    # Motion Parameter Estimation (SimpleITK)
    # =====================================================================

    def _extract_motion_features(self, df):
        """Extract motion features from 4D images using SimpleITK registration."""
        print("Extracting motion features")
        for index, fname in enumerate(df['filename'], start=1):
            print(f'Image {index}: Processing motion features for {fname}')
            fpath = os.path.join(self._mri_4d_image_data_dir, fname)

            if not os.path.exists(fpath):
                print(f"  Warning: 4D file not found: {fpath}")
                continue

            try:
                motion_params = self._estimate_motion_parameters_parallel(fpath)
                fd = self._compute_framewise_displacement(motion_params)

                df.loc[df['filename'] == fname, 'motion_mean_fd'] = np.mean(fd[1:])
                df.loc[df['filename'] == fname, 'motion_max_fd'] = np.max(fd)
                df.loc[df['filename'] == fname, 'motion_std_fd'] = np.std(fd[1:])

                translations = np.abs(motion_params[:, :3])
                rotations = np.abs(motion_params[:, 3:])

                df.loc[df['filename'] == fname, 'motion_mean_translation'] = np.mean(translations)
                df.loc[df['filename'] == fname, 'motion_max_translation'] = np.max(translations)
                df.loc[df['filename'] == fname, 'motion_mean_rotation'] = np.mean(rotations)
                df.loc[df['filename'] == fname, 'motion_max_rotation'] = np.max(rotations)

                n_outliers = np.sum(fd > 0.5)
                df.loc[df['filename'] == fname, 'motion_n_outliers'] = n_outliers
                df.loc[df['filename'] == fname, 'motion_pct_outliers'] = 100 * n_outliers / len(fd)

            except Exception as e:
                print(f"  Error extracting motion features for {fname}: {e}")
                continue

        return df

    def _estimate_motion_parameters_sitk(self, nifti_4d_path, ref_volume_idx=0):
        """Estimate 6 rigid-body motion parameters using SimpleITK registration.

        Returns:
            motion_params: (n_volumes, 6) array [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z]
        """
        img = nib.load(nifti_4d_path)
        data = img.get_fdata()
        n_volumes = data.shape[3]
        voxel_spacing = img.header.get_zooms()[:3]

        ref_sitk = sitk.GetImageFromArray(data[:, :, :, ref_volume_idx].T)
        ref_sitk.SetSpacing([float(s) for s in voxel_spacing])

        motion_params = np.zeros((n_volumes, 6))

        registration = sitk.ImageRegistrationMethod()
        registration.SetMetricAsMeanSquares()
        registration.SetOptimizerAsRegularStepGradientDescent(
            learningRate=1.0, minStep=0.001, numberOfIterations=200
        )
        registration.SetInterpolator(sitk.sitkLinear)

        for vol_idx in range(n_volumes):
            if vol_idx == ref_volume_idx:
                continue

            mov_sitk = sitk.GetImageFromArray(data[:, :, :, vol_idx].T)
            mov_sitk.SetSpacing([float(s) for s in voxel_spacing])

            initial_transform = sitk.CenteredTransformInitializer(
                ref_sitk, mov_sitk, sitk.Euler3DTransform(),
                sitk.CenteredTransformInitializerFilter.GEOMETRY
            )
            registration.SetInitialTransform(initial_transform, inPlace=False)

            try:
                final_transform = registration.Execute(ref_sitk, mov_sitk)
                params = final_transform.GetParameters()
                motion_params[vol_idx, :3] = params[3:6]
                motion_params[vol_idx, 3:] = params[0:3]
            except Exception as e:
                print(f"    Volume {vol_idx} registration failed: {e}")
                continue

        return motion_params

    def _compute_framewise_displacement(self, motion_params, radius=50):
        """Compute framewise displacement from motion parameters (Power et al., 2012).

        FD = |Δtrans_x| + |Δtrans_y| + |Δtrans_z| + r*|Δrot_x| + r*|Δrot_y| + r*|Δrot_z|

        Args:
            motion_params: (n_volumes, 6) array [trans_x/y/z, rot_x/y/z]
            radius: Brain radius in mm for converting rotations to arc length (default 50mm)

        Returns:
            fd: (n_volumes,) array of framewise displacement values
        """
        motion_mm = motion_params.copy()
        motion_mm[:, 3:] *= radius

        diff = np.diff(motion_mm, axis=0)
        fd = np.abs(diff).sum(axis=1)

        return np.insert(fd, 0, 0)

    def _estimate_motion_parameters_parallel(self, nifti_4d_path, ref_volume_idx=0, n_workers=None):
        """Estimate motion parameters using parallel volume registration.

        Args:
            nifti_4d_path: Path to 4D NIfTI file
            ref_volume_idx: Index of reference volume (default: 0)
            n_workers: Number of parallel workers (default: cpu_count - 1, max 8)

        Returns:
            motion_params: (n_volumes, 6) array [trans_x, trans_y, trans_z, rot_x, rot_y, rot_z]
        """
        if n_workers is None:
            n_workers = min(mp.cpu_count() - 1, 8)

        img = nib.load(nifti_4d_path)
        data = img.get_fdata()
        n_volumes = data.shape[3]
        voxel_spacing = img.header.get_zooms()[:3]

        ref_data = data[:, :, :, ref_volume_idx]
        motion_params = np.zeros((n_volumes, 6))

        # Prepare work items for parallel processing
        work_items = [
            (data[:, :, :, vol_idx], ref_data, voxel_spacing, vol_idx)
            for vol_idx in range(n_volumes) if vol_idx != ref_volume_idx
        ]

        print(f"    Registering {len(work_items)} volumes using {n_workers} workers...")

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            results = list(executor.map(_register_single_volume_worker, work_items))

        # Collect results
        for vol_idx, trans, rot in results:
            if trans is not None:
                motion_params[vol_idx, :3] = trans
                motion_params[vol_idx, 3:] = rot

        return motion_params

    # =====================================================================
    # Temporal Quality Metrics (DVARS, tSNR, AR1)
    # =====================================================================

    def _extract_temporal_quality_features(self, df):
        """Extract temporal quality metrics from 4D images."""
        print("Extracting temporal quality features...")

        for index, fname in enumerate(df['filename'], start=1):
            print(f'Image {index}: Processing temporal quality features for {fname}')
            fpath = os.path.join(self._mri_4d_image_data_dir, fname)

            if not os.path.exists(fpath):
                print(f"  Warning: 4D file not found: {fpath}")
                continue

            try:
                img = nib.load(fpath)
                data = img.get_fdata()

                mean_img = np.mean(data, axis=3)
                brain_mask = mean_img > np.percentile(mean_img[mean_img > 0], 25)

                dvars_raw, dvars_std = self._compute_dvars(data, brain_mask)
                df.loc[df['filename'] == fname, 'temporal_dvars_mean'] = np.mean(dvars_raw[1:])
                df.loc[df['filename'] == fname, 'temporal_dvars_std'] = np.std(dvars_raw[1:])
                df.loc[df['filename'] == fname, 'temporal_dvars_max'] = np.max(dvars_raw[1:])
                df.loc[df['filename'] == fname, 'temporal_pct_high_dvars'] = 100 * np.sum(dvars_std[1:] > 1.5) / (len(dvars_std) - 1)

                tsnr_map, tsnr_median, tsnr_mean, tsnr_std = self._compute_tsnr(data, brain_mask)
                df.loc[df['filename'] == fname, 'temporal_tsnr_median'] = tsnr_median
                df.loc[df['filename'] == fname, 'temporal_tsnr_mean'] = tsnr_mean
                df.loc[df['filename'] == fname, 'temporal_tsnr_std'] = tsnr_std

                ar1_median, ar1_mean = self._compute_ar1(data, brain_mask)
                df.loc[df['filename'] == fname, 'temporal_ar1_median'] = ar1_median
                df.loc[df['filename'] == fname, 'temporal_ar1_mean'] = ar1_mean

            except Exception as e:
                print(f"  Error extracting temporal quality features for {fname}: {e}")
                continue

        return df

    def _compute_dvars(self, data_4d, brain_mask):
        """Compute DVARS (D temporal variance of timecourses).

        DVARS_t = sqrt( (1/N) * Σ(x_{i,t} - x_{i,t-1})² )
        """
        brain_data = data_4d[brain_mask]

        global_mean = np.mean(brain_data)
        if global_mean > 0:
            brain_scaled = (brain_data / global_mean) * 1000
        else:
            brain_scaled = brain_data

        diff = np.diff(brain_scaled, axis=1)
        dvars_raw = np.sqrt(np.mean(diff**2, axis=0))
        dvars_raw = np.insert(dvars_raw, 0, 0)

        diff_std = np.std(diff)
        if diff_std > 0:
            dvars_std = dvars_raw / diff_std
        else:
            dvars_std = dvars_raw

        return dvars_raw, dvars_std

    def _compute_tsnr(self, data_4d, brain_mask):
        """Compute temporal signal-to-noise ratio.

        tSNR = mean(timeseries) / std(timeseries)
        """
        mean_signal = np.mean(data_4d, axis=3)
        std_signal = np.std(data_4d, axis=3)

        std_signal[std_signal == 0] = np.inf

        tsnr_map = mean_signal / std_signal
        tsnr_map[~brain_mask] = 0

        tsnr_brain = tsnr_map[brain_mask]
        tsnr_median = np.median(tsnr_brain)
        tsnr_mean = np.mean(tsnr_brain)
        tsnr_std = np.std(tsnr_brain)

        return tsnr_map, tsnr_median, tsnr_mean, tsnr_std

    def _compute_ar1(self, data_4d, brain_mask):
        """Compute lag-1 autocorrelation (vectorized).

        AR(1) = corr(x_t, x_{t-1})
        """
        brain_data = data_4d[brain_mask]

        brain_demean = brain_data - np.mean(brain_data, axis=1, keepdims=True)

        numerator = np.sum(brain_demean[:, 1:] * brain_demean[:, :-1], axis=1)
        denominator = np.sum(brain_demean**2, axis=1)
        denominator[denominator == 0] = 1

        ar1_values = numerator / denominator

        return np.median(ar1_values), np.mean(ar1_values)

    # =====================================================================
    # Structural Quality Metrics (CJV, CNR, SNR)
    # =====================================================================

    def _extract_structural_quality_features(self, df):
        """Extract structural quality metrics from 3D images.

        Uses SynthSeg segmentations for tissue masks if available,
        otherwise falls back to intensity-based thresholding.
        """
        print("Extracting structural quality features...")

        for index, fname in enumerate(df['filename'], start=1):
            print(f'Image {index}: Processing structural quality features for {fname}')
            fpath = os.path.join(self._mri_3d_image_data_dir, fname)

            if not os.path.exists(fpath):
                print(f"  Warning: 3D file not found: {fpath}")
                continue

            try:
                img = nib.load(fpath)
                img_data = img.get_fdata()

                seg_path = os.path.join(
                    self._synthseg_output_dir,
                    fname.replace('.nii.gz', '').replace('.nii', '') + '_synthseg.nii.gz'
                )

                if os.path.exists(seg_path):
                    gm_mask, wm_mask, csf_mask = self._get_tissue_masks_from_synthseg(seg_path)
                else:
                    if np.any(img_data > 0):
                        threshold = np.percentile(img_data[img_data > 0], 10)
                    else:
                        threshold = 0
                    brain_mask = img_data > threshold

                    p33 = np.percentile(img_data[brain_mask], 33)
                    p66 = np.percentile(img_data[brain_mask], 66)

                    gm_mask = brain_mask & (img_data >= p33) & (img_data < p66)
                    wm_mask = brain_mask & (img_data >= p66)
                    csf_mask = brain_mask & (img_data < p33)

                gm_values = img_data[gm_mask]
                wm_values = img_data[wm_mask]

                if len(gm_values) > 0 and len(wm_values) > 0:
                    mu_gm, mu_wm = np.mean(gm_values), np.mean(wm_values)
                    sigma_gm, sigma_wm = np.std(gm_values), np.std(wm_values)

                    df.loc[df['filename'] == fname, 'structural_gm_mean'] = mu_gm
                    df.loc[df['filename'] == fname, 'structural_wm_mean'] = mu_wm
                    df.loc[df['filename'] == fname, 'structural_gm_std'] = sigma_gm
                    df.loc[df['filename'] == fname, 'structural_wm_std'] = sigma_wm

                    cjv = (sigma_wm + sigma_gm) / (abs(mu_wm - mu_gm) + 1e-8)
                    df.loc[df['filename'] == fname, 'structural_cjv'] = cjv

                    df.loc[df['filename'] == fname, 'structural_gm_wm_contrast'] = abs(mu_wm - mu_gm) / (mu_wm + mu_gm + 1e-8)

                    air_mask = img_data < np.percentile(img_data[img_data > 0], 5)
                    sigma_air = np.std(img_data[air_mask]) if np.any(air_mask) else 1e-8

                    cnr = abs(mu_gm - mu_wm) / (np.sqrt(sigma_air**2 + sigma_wm**2 + sigma_gm**2) + 1e-8)
                    df.loc[df['filename'] == fname, 'structural_cnr'] = cnr

                    brain_mask = gm_mask | wm_mask | csf_mask
                    snr = np.mean(img_data[brain_mask]) / (sigma_air * np.sqrt(2 / (4 - np.pi)) + 1e-8)
                    df.loc[df['filename'] == fname, 'structural_snr'] = snr

                brain_mask = gm_mask | wm_mask
                masked_image = img_data * brain_mask
                variance_laplacian = np.var(laplace(masked_image))
                df.loc[df['filename'] == fname, 'structural_variance_laplacian'] = variance_laplacian

                grad_x = np.gradient(img_data, axis=0)
                grad_y = np.gradient(img_data, axis=1)
                grad_z = np.gradient(img_data, axis=2)
                grad_mag = np.sqrt(grad_x**2 + grad_y**2 + grad_z**2)
                df.loc[df['filename'] == fname, 'structural_gradient_magnitude_mean'] = np.mean(grad_mag[brain_mask])

            except Exception as e:
                print(f"  Error extracting structural quality features for {fname}: {e}")
                continue

        return df

    # =====================================================================
    # Quick Morphometrics
    # =====================================================================

    def _extract_morphometric_features(self, df):
        """Extract quick morphometric features from 3D images (nibabel only)."""
        print("Extracting morphometric features...")

        for index, fname in enumerate(df['filename'], start=1):
            print(f'Image {index}: Processing morphometric features for {fname}')
            fpath = os.path.join(self._mri_3d_image_data_dir, fname)

            if not os.path.exists(fpath):
                print(f"  Warning: 3D file not found: {fpath}")
                continue

            try:
                img = nib.load(fpath)
                data = img.get_fdata()
                voxel_vol = np.prod(img.header.get_zooms()[:3])

                if np.any(data > 0):
                    threshold = np.percentile(data[data > 0], 25)
                else:
                    threshold = 0
                mask = data > threshold
                brain_vals = data[mask]

                total_voxels = np.sum(mask)
                brain_volume_ml = total_voxels * voxel_vol / 1000
                df.loc[df['filename'] == fname, 'morpho_brain_volume_ml'] = brain_volume_ml

                eroded = binary_erosion(mask)
                boundary_voxels = np.sum(mask & ~eroded)
                surface_area = boundary_voxels * (voxel_vol ** (2/3))
                df.loc[df['filename'] == fname, 'morpho_surface_area_mm2'] = surface_area

                if surface_area > 0:
                    sphericity = (np.pi ** (1/3) * (6 * total_voxels * voxel_vol) ** (2/3)) / surface_area
                else:
                    sphericity = 0
                df.loc[df['filename'] == fname, 'morpho_sphericity'] = sphericity

                hist, bins = np.histogram(brain_vals, bins=256)
                hist_norm = hist / hist.sum()
                hist_norm = hist_norm[hist_norm > 0]
                intensity_entropy = -np.sum(hist_norm * np.log2(hist_norm))
                df.loc[df['filename'] == fname, 'morpho_intensity_entropy'] = intensity_entropy

                intensity_cv = np.std(brain_vals) / (np.mean(brain_vals) + 1e-8)
                df.loc[df['filename'] == fname, 'morpho_intensity_cv'] = intensity_cv

                df.loc[df['filename'] == fname, 'morpho_intensity_kurtosis'] = kurtosis(brain_vals)
                df.loc[df['filename'] == fname, 'morpho_intensity_skewness'] = skew(brain_vals)

                n = len(brain_vals)
                if n > 2:
                    m3 = skew(brain_vals)
                    m4 = kurtosis(brain_vals) + 3
                    bimodality = (m3**2 + 1) / m4
                else:
                    bimodality = 0
                df.loc[df['filename'] == fname, 'morpho_intensity_bimodality'] = bimodality

                coords = np.where(mask)
                bbox_volume = ((coords[0].max() - coords[0].min() + 1) *
                              (coords[1].max() - coords[1].min() + 1) *
                                (coords[2].max() - coords[2].min() + 1))
                df.loc[df['filename'] == fname, 'morpho_bbox_volume_ratio'] = total_voxels / (bbox_volume + 1e-8)

            except Exception as e:
                print(f"  Error extracting morphometric features for {fname}: {e}")
                continue

        return df

class HarvardOxfordAtlas(Datasets):
    def __init__(self):
        atlas = datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
        super().__init__(atlas=atlas)
class SchaeferAtlas(Datasets):
    def __init__(self, n_rois=100, networks=7, resolution_mm=2):
        atlas = datasets.fetch_atlas_schaefer_2018(n_rois=n_rois, yeo_networks=networks, resolution_mm=resolution_mm)
        super().__init__(atlas=atlas)
class AALAtlas(Datasets):   
    def __init__(self, version='SPM12'):
        atlas = datasets.fetch_atlas_aal(version=version)
        super().__init__(atlas=atlas)
class extractSLANTData:
    def __init__(self):
        self._features_csv_dir = FEATURES_CSV_OUTPUT
        self._slant_runtime_data_dir = SLANT_RUNTIME_DATA
        self._ec2_runtime_data_dir = EC2_RUNTIME_DATA
        self.use_paper_dataset = True
    
    def load_data_to_csv(self):
        existing_df = pd.read_csv(self._features_csv_dir)

        # Drop any existing makespan columns to avoid duplicates
        makespan_cols = [col for col in existing_df.columns if col.startswith('makespan') or col.startswith('preproc_time_sec') or col.startswith('postproc_time_sec')]
        if makespan_cols:
            print(f'Dropping existing makespan columns: {makespan_cols}')
            existing_df = existing_df.drop(columns=makespan_cols)

        makespan_df = self._extract_makespan_data()

        if self.use_paper_dataset:
            merged_df = existing_df.merge(makespan_df[['filename', 'makespan (sec)']], on='filename', how='left')
            merged_df.rename(columns={'makespan (sec)': 'makespan'}, inplace=True)
        else:
            makespan_df['makespan'] = makespan_df['preproc_time_sec'] + makespan_df['postproc_time_sec']
            merged_df = existing_df.merge(makespan_df[['filename', 'makespan']], on='filename', how='left')

        merged_df.to_csv(self._features_csv_dir, index=False)
        print(f'Added makespan data to {self._features_csv_dir}')

    def _extract_makespan_data(self):
        if self.use_paper_dataset:
            df = pd.read_excel(self._slant_runtime_data_dir)
        else:
            df = pd.read_csv(self._ec2_runtime_data_dir)
        return df