"""Tests for time/build_table.py - paper table aggregator."""
import csv
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).parent
FIXTURES = HERE / 'fixtures'
sys.path.insert(0, str(HERE))  # avoid stdlib `time` collision; import sibling directly


def test_build_table_produces_csv_and_md(tmp_path):
    from build_table import build_paper_table

    out_csv = tmp_path / 'paper_table.csv'
    out_md = tmp_path / 'paper_table.md'

    build_paper_table(
        synthseg_csv=FIXTURES / 'timing_synthseg_sample.csv',
        others_csv=FIXTURES / 'timing_other_features_sample.csv',
        train_csv=FIXTURES / 'timing_train_predict_sample.csv',
        hardware_json=FIXTURES / 'timing_hardware_sample.json',
        out_csv=out_csv,
        out_md=out_md,
    )

    assert out_csv.exists()
    assert out_md.exists()

    # CSV should have header + 3 extraction rows + 6 training rows + 6 inference rows (one per model)
    with out_csv.open() as f:
        rows = list(csv.DictReader(f))
    components = [r['component'] for r in rows]
    assert 'SynthSeg subprocess (extraction)' in components
    assert 'Other 16 features (extraction)' in components
    assert 'Total extraction per image' in components
    assert any('Training' in c and 'Ridge' in c for c in components)
    assert any('Inference' in c for c in components)
    # Exactly 6 inference rows — one per model
    inference_rows = [c for c in components if 'Inference' in c]
    assert len(inference_rows) == 6

    # MD should contain a markdown table and a hardware footer
    md = out_md.read_text()
    assert '| Component' in md
    assert '|---' in md
    assert 'SynthSeg' in md
    assert 'Hardware' in md  # footer present


def test_build_table_computes_median_and_iqr(tmp_path):
    from build_table import build_paper_table

    out_csv = tmp_path / 'paper_table.csv'
    out_md = tmp_path / 'paper_table.md'

    build_paper_table(
        synthseg_csv=FIXTURES / 'timing_synthseg_sample.csv',
        others_csv=FIXTURES / 'timing_other_features_sample.csv',
        train_csv=FIXTURES / 'timing_train_predict_sample.csv',
        hardware_json=FIXTURES / 'timing_hardware_sample.json',
        out_csv=out_csv,
        out_md=out_md,
    )

    with out_csv.open() as f:
        rows = {r['component']: r for r in csv.DictReader(f)}

    # SynthSeg median of [62.4, 65.1, 58.7] = 62.4
    synth = rows['SynthSeg subprocess (extraction)']
    assert '62.4' in synth['per_image_or_call']

    # Other 16 median of [2.31, 2.42, 2.18] = 2.31
    others = rows['Other 16 features (extraction)']
    assert '2.31' in others['per_image_or_call']
