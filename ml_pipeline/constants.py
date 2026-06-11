# Features CSV from feature extraction
FEATURES_CSV_OUTPUT = r'C:\Users\heage\OneDrive\Research\SLANT\PredictiveModeling\ml_pipeline\csv_files\inputs\features_data.csv'
FEATURES_18_CSV_OUTPUT = r'C:\Users\heage\OneDrive\Research\SLANT\PredictiveModeling\ml_pipeline\csv_files\inputs\features_data_18.csv'

# Experiment logging for feature selection
EXPERIMENT_LOG_CSV = r'C:\Users\heage\OneDrive\Research\SLANT\PredictiveModeling\ml_pipeline\csv_files\outputs\experiment_log.csv'

# Directory to save CV predictions
CSV_OUTPUT_DIR = r'C:\Users\heage\OneDrive\Research\SLANT\PredictiveModeling\ml_pipeline\csv_files'

# Directories of MRI input volumes
MRI_3D_IMAGE_DATA = r'C:\Users\heage\OneDrive\Research\SLANT\PredictiveModeling\ml_pipeline\csv_files\inputs\raiders_data_3d'

# Registration template
TARGET_IMAGE = r''

# SLANT/EC2 runtime data (placeholders; only required if those extractors are run)
SLANT_RUNTIME_DATA = r''
EC2_RUNTIME_DATA = r''

# SynthSeg
SYNTHSEG_OUTPUT_DIR = r'C:\Users\heage\OneDrive\Research\SLANT\PredictiveModeling\ml_pipeline\time\synthseg_output_timing'
SYNTHSEG_VOLUMES_CSV = SYNTHSEG_OUTPUT_DIR + r'\all_volumes.csv'
SYNTHSEG_QC_CSV = SYNTHSEG_OUTPUT_DIR + r'\all_qc.csv'
SYNTHSEG_CONDA_ENV = 'synthseg_38'
SYNTHSEG_PREDICT_SCRIPT = r'C:\Users\heage\OneDrive\Research\SLANT\PredictiveModeling\reference\SynthSeg\scripts\commands\SynthSeg_predict.py'