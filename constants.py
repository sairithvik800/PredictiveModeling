# Features CSV from feature extraction
FEATURES_CSV_OUTPUT = r'C:\Users\heage\OneDrive\Research\SLANT\PredictiveModeling\csv_files\inputs\features_data.csv'

# Experiment logging for feature selection
EXPERIMENT_LOG_CSV = r'C:\Users\heage\OneDrive\Research\SLANT\PredictiveModeling\csv_files\outputs\experiment_log.csv'

# Directory to save CV predictions
CSV_OUTPUT_DIR = r'C:\Users\heage\OneDrive\Research\SLANT\PredictiveModeling\csv_files'

# ---- Paths required by reference/features.py ----

# Directories of MRI input volumes
MRI_3D_IMAGE_DATA = r'C:\Users\heage\OneDrive\Research\SLANT\PredictiveModeling\csv_files\inputs\raiders_data_3d'
MRI_4D_IMAGE_DATA = r'C:\Users\heage\OneDrive\Research\SLANT\PredictiveModeling\csv_files\inputs\raiders_data_3d'  # same dir; files are 4D BOLD

# Registration template (placeholder; only required if registration_features is run)
TARGET_IMAGE = r''  # set to MNI152 template path if registration features are exercised

# SLANT/EC2 runtime data (placeholders; only required if those extractors are run)
SLANT_RUNTIME_DATA = r''
EC2_RUNTIME_DATA = r''

# SynthSeg outputs and conda env
SYNTHSEG_OUTPUT_DIR = r'C:\Users\heage\OneDrive\Research\SLANT\PredictiveModeling\time\synthseg_output_timing'
SYNTHSEG_VOLUMES_CSV = SYNTHSEG_OUTPUT_DIR + r'\all_volumes.csv'
SYNTHSEG_QC_CSV = SYNTHSEG_OUTPUT_DIR + r'\all_qc.csv'
SYNTHSEG_CONDA_ENV = 'synthseg_38'  # confirm via: conda env list

# Inter-rater reliability features CSV (placeholder; not needed by this benchmark)
ICC_FEATURES_CSV = r''

# Path to the SynthSeg prediction script (used by reference/features._run_synthseg_cli)
SYNTHSEG_PREDICT_SCRIPT = r'C:\Users\heage\OneDrive\Research\SLANT\PredictiveModeling\reference\SynthSeg\scripts\commands\SynthSeg_predict.py'