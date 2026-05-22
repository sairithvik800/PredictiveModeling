"""Phase 2: time the 16 non-SynthSeg features across all 88 images."""
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))  # for sibling import without `time.` prefix

import constants
from extract_18_features import extract_for_all


def run() -> Path:
    rows = extract_for_all(Path(constants.MRI_3D_IMAGE_DATA))
    out_path = Path(__file__).parent / 'outputs' / 'timing_other_features.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['filename', 'other16_seconds', 'load_time_ms'])
        writer.writeheader()
        writer.writerows(rows)
    return out_path


if __name__ == '__main__':
    out = run()
    print(f'Wrote {out}')
