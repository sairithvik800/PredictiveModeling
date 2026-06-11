"""Filesystem paths for the ML pipeline, resolved relative to this package.

Paths are derived from this file's location so the repo runs unchanged on any
machine after the environment is created. Values are kept as ``str`` for
compatibility with the string-concatenation and ``pd.read_csv`` call sites in
main.py and features.py.
"""

from pathlib import Path

_PKG_DIR = Path(__file__).resolve().parent   # .../ml_pipeline
_REPO_DIR = _PKG_DIR.parent                   # repository root
_CSV = _PKG_DIR / "csv_files"

# Feature CSVs produced by feature extraction.
FEATURES_CSV_OUTPUT = str(_CSV / "inputs" / "features_data.csv")        # full 1506-col exploratory input (read by main.py)
FEATURES_18_CSV_OUTPUT = str(_CSV / "inputs" / "features_data_18.csv")  # 18-feature deploy extractor output (written by features.py)

# Experiment logging for the stochastic feature-selection mode.
EXPERIMENT_LOG_CSV = str(_CSV / "outputs" / "experiment_log.csv")

# Base directory under which CV predictions are written (main.py appends '/outputs/...').
CSV_OUTPUT_DIR = str(_CSV)

# --- Paths to data NOT shipped with this repo; only needed to re-run feature extraction. ---

# Directory of MRI input volumes.
MRI_3D_IMAGE_DATA = str(_CSV / "inputs" / "raiders_data_3d")

# SynthSeg segmentation step of feature extraction.
SYNTHSEG_OUTPUT_DIR = str(_PKG_DIR / "time" / "synthseg_output_timing")
SYNTHSEG_CONDA_ENV = "synthseg_38"
SYNTHSEG_PREDICT_SCRIPT = str(
    _REPO_DIR / "reference" / "SynthSeg" / "scripts" / "commands" / "SynthSeg_predict.py"
)
