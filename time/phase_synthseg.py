"""Phase 1: time SynthSeg subprocess wall-clock per image across all 88 images.

Uses reference/features.py's _run_synthseg_cli with production flags:
robust=True, fast=False, use_cpu=True, crop_size=192. Writes fresh outputs to
time/synthseg_output_timing/ so the production seg files are untouched.
"""
import csv
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Order matters: reference/ must be ahead of repo root so `import features`
# resolves to reference/features.py (which defines Datasets), not the root
# features.py (which is reference-only documentation code).
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'reference'))

import constants
import features as ref_features

ROBUST = True
FAST = False
USE_CPU = True
CROP_SIZE = 192


def run() -> Path:
    timing_dir = Path(__file__).parent / 'synthseg_output_timing'
    timing_dir.mkdir(parents=True, exist_ok=True)

    datasets_inst = ref_features.Datasets()
    if not datasets_inst._check_synthseg_environment():
        raise RuntimeError(
            f'SynthSeg conda env {constants.SYNTHSEG_CONDA_ENV!r} not found. '
            'Create with: conda create -n synthseg_38 python=3.8 -y'
        )

    mri_dir = Path(constants.MRI_3D_IMAGE_DATA)
    nifti_files = sorted(mri_dir.glob('*.nii.gz'))
    rows = []

    for idx, input_path in enumerate(nifti_files, start=1):
        base_name = input_path.name.replace('.nii.gz', '').replace('.nii', '')
        seg_path = timing_dir / f'{base_name}_synthseg.nii.gz'
        vol_path = timing_dir / f'{base_name}_volumes.csv'
        qc_path = timing_dir / f'{base_name}_qc.csv'

        print(f'[{idx}/{len(nifti_files)}] {input_path.name}', flush=True)

        start = time.perf_counter()
        try:
            datasets_inst._run_synthseg_cli(
                input_path=str(input_path),
                seg_path=str(seg_path),
                vol_path=str(vol_path),
                qc_path=str(qc_path),
                robust=ROBUST,
                fast=FAST,
                use_cpu=USE_CPU,
                crop_size=CROP_SIZE,
            )
            elapsed = time.perf_counter() - start
            print(f'  {elapsed:.1f}s', flush=True)
        except Exception as e:
            elapsed = -1.0
            print(f'  ERROR: {e}', flush=True)

        rows.append({
            'filename': input_path.name,
            'synthseg_seconds': round(elapsed, 2),
        })

    out_path = Path(__file__).parent / 'outputs' / 'timing_synthseg.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'synthseg_seconds'])
        writer.writeheader()
        writer.writerows(rows)
    return out_path


if __name__ == '__main__':
    out = run()
    print(f'Wrote {out}')
