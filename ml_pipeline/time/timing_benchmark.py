"""CLI orchestrator for the timing benchmark.

Usage:
    python time/timing_benchmark.py --phase synthseg
    python time/timing_benchmark.py --phase others
    python time/timing_benchmark.py --phase train
    python time/timing_benchmark.py --phase all
    python time/timing_benchmark.py --build-table
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(HERE))  # sibling imports without the `time.` prefix (avoids stdlib collision)


def _run_synthseg() -> None:
    from phase_synthseg import run
    print(f'Phase 1 wrote: {run()}')


def _run_others() -> None:
    from phase_others import run
    print(f'Phase 2 wrote: {run()}')


def _run_train() -> None:
    from phase_train import time_training_per_model, write_train_predict_csv
    from main import DEFAULT_FEATURES
    rows = time_training_per_model(DEFAULT_FEATURES)
    out = HERE / 'outputs' / 'timing_train_predict.csv'
    write_train_predict_csv(rows, out)
    print(f'Phase 3 wrote: {out}')


def _capture_hardware() -> None:
    from hardware_info import write_hardware_json
    out = HERE / 'outputs' / 'timing_hardware.json'
    write_hardware_json(out)
    print(f'Hardware wrote: {out}')


def _build_table() -> None:
    from build_table import build_paper_table
    out_dir = HERE / 'outputs'
    build_paper_table(
        synthseg_csv=out_dir / 'timing_synthseg.csv',
        others_csv=out_dir / 'timing_other_features.csv',
        train_csv=out_dir / 'timing_train_predict.csv',
        hardware_json=out_dir / 'timing_hardware.json',
        out_csv=out_dir / 'timing_paper_table.csv',
        out_md=out_dir / 'timing_paper_table.md',
    )
    print(f'Table wrote: {out_dir / "timing_paper_table.csv"} and .md')


def main() -> None:
    parser = argparse.ArgumentParser(description='Run the SLANT computational overhead benchmark.')
    parser.add_argument('--phase', choices=['synthseg', 'others', 'train', 'all'],
                        help='Which measurement phase to run.')
    parser.add_argument('--build-table', action='store_true',
                        help='Aggregate existing CSVs into the paper table.')
    args = parser.parse_args()

    if not args.phase and not args.build_table:
        parser.error('Pass either --phase {synthseg,others,train,all} or --build-table.')

    if args.phase == 'synthseg':
        _capture_hardware()
        _run_synthseg()
    elif args.phase == 'others':
        _capture_hardware()
        _run_others()
    elif args.phase == 'train':
        _capture_hardware()
        _run_train()
    elif args.phase == 'all':
        _capture_hardware()
        _run_synthseg()
        _run_others()
        _run_train()
        _build_table()

    if args.build_table:
        _capture_hardware()
        _build_table()


if __name__ == '__main__':
    main()
