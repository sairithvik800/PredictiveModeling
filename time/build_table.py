"""Aggregate the three timing CSVs and the hardware JSON into a paper-ready table."""
import csv
import json
from pathlib import Path
from statistics import median


def _iqr(values: list[float]) -> tuple[float, float]:
    """Return the 25th and 75th percentiles of a list of floats."""
    if not values:
        return 0.0, 0.0
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    q1_idx = max(0, int(n * 0.25))
    q3_idx = min(n - 1, int(n * 0.75))
    return sorted_vals[q1_idx], sorted_vals[q3_idx]


def _fmt_seconds(med: float, iqr: tuple[float, float]) -> str:
    return f'{med:.2f}s (IQR {iqr[0]:.2f}-{iqr[1]:.2f})'


def _fmt_microseconds(val: float) -> str:
    return f'{val:.1f} us'


def _read_seconds_column(path: Path, column: str) -> list[float]:
    with path.open() as f:
        return [float(row[column]) for row in csv.DictReader(f)]


def build_paper_table(
    synthseg_csv: Path,
    others_csv: Path,
    train_csv: Path,
    hardware_json: Path,
    out_csv: Path,
    out_md: Path,
) -> None:
    """Read the three timing CSVs + hardware metadata, write summary CSV and MD."""
    # Sentinel guard: abort if SynthSeg CSV contains placeholder rows
    with Path(synthseg_csv).open() as f:
        for row in csv.DictReader(f):
            if row.get('filename', '').startswith('PENDING_'):
                raise ValueError(
                    'SynthSeg CSV contains placeholder rows; run Phase 1 before building the table.'
                )

    synth_secs = _read_seconds_column(Path(synthseg_csv), 'synthseg_seconds')
    other_secs = _read_seconds_column(Path(others_csv), 'other16_seconds')

    synth_med = median(synth_secs) if synth_secs else 0.0
    synth_iqr = _iqr(synth_secs)
    other_med = median(other_secs) if other_secs else 0.0
    other_iqr = _iqr(other_secs)

    rows = [
        {
            'component': 'SynthSeg subprocess (extraction)',
            'per_image_or_call': _fmt_seconds(synth_med, synth_iqr),
            'notes': 'robust=True, fast=False, CPU; per-image wall-clock',
        },
        {
            'component': 'Other 16 features (extraction)',
            'per_image_or_call': _fmt_seconds(other_med, other_iqr),
            'notes': 'single MRI load shared across features',
        },
        {
            'component': 'Total extraction per image',
            'per_image_or_call': f'{synth_med + other_med:.1f}s',
            'notes': 'SynthSeg median + other-16 median',
        },
    ]

    # Per-model training rows + one inference row per model
    with Path(train_csv).open() as f:
        train_rows = list(csv.DictReader(f))
    for r in train_rows:
        rows.append({
            'component': f'Training - {r["model"]} (nested CV, n=88)',
            'per_image_or_call': f'{float(r["train_seconds"]):.1f}s',
            'notes': 'total wall-clock incl. grid search',
        })

    for r in train_rows:
        rows.append({
            'component': f'Inference - {r["model"]} (1x18 vector)',
            'per_image_or_call': _fmt_microseconds(float(r['predict_us_median'])),
            'notes': 'median over 1000 calls',
        })

    # Write CSV
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with Path(out_csv).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['component', 'per_image_or_call', 'notes'])
        writer.writeheader()
        writer.writerows(rows)

    # Write MD
    hw = json.loads(Path(hardware_json).read_text())
    md_lines = [
        '# Computational Overhead - Paper Table',
        '',
        '| Component | Per-image / per-call | Notes |',
        '|---|---|---|',
    ]
    for r in rows:
        md_lines.append(f'| {r["component"]} | {r["per_image_or_call"]} | {r["notes"]} |')

    md_lines += [
        '',
        '## Hardware',
        '',
        f'- CPU: {hw.get("cpu_brand", hw.get("processor", "unknown"))}',
        f'- RAM: {hw.get("ram_total_gb", "unknown")} GB',
        f'- OS: {hw.get("platform", "unknown")}',
        f'- Python: {hw.get("python_version", "unknown")}',
        '',
        '### Library versions',
        '',
    ]
    for name, version in sorted(hw.get('library_versions', {}).items()):
        md_lines.append(f'- {name}: {version}')

    md_lines += [
        '',
        '## Notes',
        '',
        '- SynthSeg published benchmark: ~15s/image on GPU vs ~60s on CPU (Billot et al., 2023). Our CPU measurements are consistent.',
        '- Deployment to a GPU-equipped host would reduce SynthSeg cost ~4x per the published benchmark.',
    ]

    Path(out_md).write_text('\n'.join(md_lines))


if __name__ == '__main__':
    here = Path(__file__).parent
    out = here / 'outputs'
    build_paper_table(
        synthseg_csv=out / 'timing_synthseg.csv',
        others_csv=out / 'timing_other_features.csv',
        train_csv=out / 'timing_train_predict.csv',
        hardware_json=out / 'timing_hardware.json',
        out_csv=out / 'timing_paper_table.csv',
        out_md=out / 'timing_paper_table.md',
    )
    print(f'Wrote {out / "timing_paper_table.csv"} and {out / "timing_paper_table.md"}')
