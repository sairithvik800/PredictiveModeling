"""
File for context on how/what each feature that was selected by the @feature_selection_pipeline.py pipeline is calculated

Features:
FEATURES_50 = [
    # >=50% selection frequency
    '7Networks_LH_Default_PFC_1_range',                   # 100.0%
    'synthseg_total_components',                           #  99.8%
    'morpho_intensity_kurtosis',                           #  99.0%
    'bias_field_proxy',                                    #  63.5%
    '7Networks_RH_SalVentAttn_TempOccPar_2_med_abs_dev',  #  53.0%
    'tile_gradient_load_imbalance',                        #  51.0%
]

FEATURES_30 = FEATURES_50 + [
    # >=30% selection frequency
    'synthseg_fragmentation_index',                        #  45.5%
    'ngtdm_complexity',                                    #  45.5%
    '7Networks_RH_Default_PFCdPFCm_2_med_abs_dev',        #  38.8%
    '7Networks_LH_SalVentAttn_PFCl_1_kurtosis',           #  35.5%
    'load_time_ms',                                        #  31.8%
    'registration_scale_difference',                       #  31.2%
    '7Networks_LH_Vis_5_med_abs_dev',                      #  30.5%
]

FEATURES_15 = FEATURES_30 + [
    # >=15% selection frequency (natural cliff at 12% terminates the set)
    'inferior_brain_offset',                               #  28.2%
    '7Networks_LH_Default_PFC_1_kurtosis',                 #  27.8%
    '7Networks_LH_Vis_2_kurtosis',                         #  27.8%
    'morpho_bbox_volume_ratio',                            #  27.5%
    'registration_rotation_difference',                    #  24.2%
    '7Networks_LH_SalVentAttn_ParOper_1_kurtosis',         #  17.0%
    '7Networks_LH_Limbic_TempPole_1_moment_5',             #  15.5%
    '7Networks_RH_DorsAttn_Post_1_iqr',                    #  15.2%
]
"""

# 1)

# Neighborhood Gray-Tone Difference Matrix (NGTDM) complexity - (ngtdm_complexity)
# 

from scipy import ndimage

def ngtdm_complexity(mri_image_path):
    n_levels = 64
    
    mri_image = nib.load(mri_image_path)
    img_data = mri_image.get_fdata()
    brain_mask = get_brain_mask(img_data)
    
    brain_vals = img_data[brain_mask]
    min_val, max_val = brain_vals.min(), brain_vals.max()
    
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
    
    return complexity

#2)

# 7 Networks features based on Schaefer Atlas 2018 - (7Networks_RH_Default_PFCdPFCm_2_med_abs_dev, 7Networks_LH_Default_PFC_1_range, 7Networks_LH_SalVentAttn_PFCl_1_kurtosis, 7Networks_LH_Default_PFC_1_kurtosis)
# cortical voxels in the MNI152 space assigned to 100 regions/ROIs across 7 networks 
# https://nilearn.github.io/dev/modules/generated/nilearn.datasets.fetch_atlas_schaefer_2018.html

from nilearn import datasets, image
import nibabel as nib
import numpy as np
from scipy.stats import median_abs_deviation, kurtosis, moment
import pandas as pd

atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2)

def region_based_features(atlas):
    atlas_img = nib.load(atlas.maps)
    atlas_labels = atlas.labels
    
    all_features = {}
    valid_regions = set()
    for mri_image_path in "path_to_mri_images":
        mri_image = nib.load(mri_image_path)
        resampled_img = image.resample_to_img(mri_image, atlas_img, interpolation='linear')
        img_data = resampled_img.get_fdata()
        atlas_data = atlas_img.get_fdata()
        
        file_features = {}
        for i in range(1, len(atlas_labels)):
            region_mask = (atlas_data == i)
            region_values = img_data[region_mask]
            stats = region_stats(region_values)
            file_features[i] = stats
            valid_regions.add(i)
            
        all_features[mri_image_path] = file_features
        
    # dataframe construction
    valid_regions = sorted(valid_regions)
    records = region_list(all_features, valid_regions)
    columns = region_columns(atlas_labels, valid_regions)
    df = pd.DataFrame(records, columns=columns)
    return df

def region_stats(values):
    values = values[values > 0]
    if len(values) == 0:
        return None        
    stats = [
        median_abs_deviation(values),
        kurtosis(values),
        np.ptp(values), # range
    ]
    return stats

def region_list(all_features, valid_regions):
    records = []
    for fname, file_features in all_features.items():
        feature_row = [fname]
        for i in valid_regions:
            stats = file_features.get(i)
            if stats is not None:
                feature_row.extend(stats)
        records.append(feature_row)
    return records

def region_columns(atlas_labels, valid_regions):
    columns = ['filename']
    for i in valid_regions:
        label = atlas_labels[i]
        for stat in ['med_abs_dev', 'kurtosis', 'range']:
            columns.append(f'{label}_{stat}')
    return columns

# MRI Signal Intensity - (bias_field_proxy, morpho_intensity_kurtosis)
# Bias field proxy: standard deviation of sampled intensities / mean of sampled intensities
# Morpho intensity kurtosis: kurtosis of brain voxel intensities

def signal_intensity(mri_image_path):
    mri_image = nib.load(mri_image_path)
    img_data = mri_image.get_fdata()
    brain_mask = get_brain_mask(img_data)
    
    grid_samples = []
    for x in np.linspace(0, img_data.shape[0]-1, 5, dtype=int):
        for y in np.linspace(0, img_data.shape[1]-1, 5, dtype=int):
            for z in np.linspace(0, img_data.shape[2]-1, 5, dtype=int):
                if brain_mask[x, y, z]:
                    grid_samples.append(img_data[x, y, z]) 
    
    if len(grid_samples) > 0:
        bias_field_proxy = np.std(grid_samples) / (np.mean(grid_samples) + 1e-10)
    else:
        bias_field_proxy = 0
    
    brain_vals = img_data[brain_mask]
    if len(brain_vals) > 0:
        morpho_intensity_kurtosis = kurtosis(brain_vals)
    else:
        morpho_intensity_kurtosis = 0
    
    return bias_field_proxy, morpho_intensity_kurtosis

# 3)

# Synthseg features - (synthseg_total_components, synthseg_fragmentation_index)
# 
# https://github.com/BBillot/SynthSeg

# First run SynthSeg on the MRI images to get segmentation maps, then calculate:

def synthseg_features():
    seg_path = "" # Directory where SynthSeg segmentation maps are stored
    seg_data = nib.load(seg_path).get_fdata().astype(int)
    labels = np.unique(seg_data)
    n_structures = len(labels) - 1
    
    features = {}
    total_components = 0
    for label in labels[labels > 0]:
        labeled_array, num_features = label(seg_data == label)
        total_components += num_features
        
    fragmentation_index = total_components / max(n_structures, 1)
    features = {
        'synthseg_total_components': total_components,
        'synthseg_fragmentation_index': fragmentation_index
    }
    
    return features

# 4)

# Computational performance features - (load_time_ms)
# Load time: time taken to load the MRI image in milliseconds

import time

def computational_performance(mri_image_path):
    start = time.perf_counter()
    mri_image = nib.load(mri_image_path)
    img_data = mri_image.get_fdata()
    load_time_ms = (time.perf_counter() - start) * 1000  
    return load_time_ms

# 5)
# Spatial & geometric features - (tile_gradient_load_imbalance, inferior_brain_offset)

from scipy.ndimage import center_of_mass

def spatial_geometric_features(mri_image_path):
    mri_image = nib.load(mri_image_path)
    img_data = mri_image.get_fdata()
    
    brain_mask = get_brain_mask(img_data)
    if brain_mask.sum() < 100:
        tile_gradient_load_imbalance = 0
    else:
        tile_shape = (16, 16, 8)
        gradient_magnitudes = []

        for i in range(0, img_data.shape[0] - tile_shape[0], tile_shape[0]):
            for j in range(0, img_data.shape[1] - tile_shape[1], tile_shape[1]):
                for k in range(0, img_data.shape[2] - tile_shape[2], tile_shape[2]):
                    tile = img_data[i:i+tile_shape[0], j:j+tile_shape[1], k:k+tile_shape[2]]
                    tile_mask = brain_mask[i:i+tile_shape[0], j:j+tile_shape[1], k:k+tile_shape[2]]

                    if tile_mask.sum() > (tile_shape[0] * tile_shape[1] * tile_shape[2] * 0.1):
                        tile_gradients = np.gradient(tile)
                        grad_magnitude = np.sqrt(sum(g**2 for g in tile_gradients))
                        avg_grad = grad_magnitude[tile_mask].mean()
                        gradient_magnitudes.append(avg_grad)

        if len(gradient_magnitudes) > 1:
            tile_gradient_load_imbalance = np.max(gradient_magnitudes) / (np.mean(gradient_magnitudes) + 1e-8)
        else:
            tile_gradient_load_imbalance = 0
            
    y, x, z = np.ogrid[:img_data.shape[0], :img_data.shape[1], :img_data.shape[2]]
    z_coords = np.where(brain_mask)[2]
    inferior_threshold = np.percentile(z_coords, 25)
    inferior_mask = brain_mask & (z >= inferior_threshold)
    if inferior_mask.sum() > 0:
        inferior_com = center_of_mass(inferior_mask.astype(float))
        full_com = center_of_mass(brain_mask.astype(float))
        inferior_brain_offset = np.linalg.norm(np.array(inferior_com) - np.array(full_com))
    else:
        inferior_brain_offset = 0
    
    return tile_gradient_load_imbalance, inferior_brain_offset

# Brain mask helper function
def get_brain_mask(image_data):
    if np.any(image_data > 0):
        threshold = np.percentile(image_data[image_data > 0], 10)
    else:
        threshold = 0
    return image_data > threshold

# X)

# Registration features - (registration_scale_difference, registration_rotation_difference)
# 

def registration_features(mri_image_path, template_image_path):
    template_img = nib.load(template_image_path)
    template_data = template_img.get_fdata()
    template_affine = template_img.affine
    
    img = nib.load(mri_image_path)
    img_data = img.get_fdata()
    img_affine = img.affine
    
    template_rotation = template_affine[:3, :3]
    img_rotation = img_affine[:3, :3]
    
    registration_rotation_difference = np.linalg.norm(img_rotation - template_rotation, ord='fro')
    
    img_det = np.linalg.det(img_rotation)
    template_det = np.linalg.det(template_rotation)
    registration_scale_difference = np.abs(img_det - template_det)
    
    return registration_scale_difference, registration_rotation_difference

# X) 

# Morphological features - (morpho_bbox_volume_ratio)
#

def morphological_features(mri_image_path):
    mri_image = nib.load(mri_image_path)
    img_data = mri_image.get_fdata()
    brain_mask = get_brain_mask(img_data)
    
    total_voxels = brain_mask.sum()
    coords = np.where(brain_mask)
    bbox_volume = ((coords[0].max() - coords[0].min() + 1) *
                   (coords[1].max() - coords[1].min() + 1) *
                    (coords[2].max() - coords[2].min() + 1))
    morpho_bbox_volume_ratio = total_voxels / (bbox_volume + 1e-10)
    return morpho_bbox_volume_ratio