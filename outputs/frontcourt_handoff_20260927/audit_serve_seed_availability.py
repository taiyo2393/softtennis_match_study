"""Measure four-player seed availability without silently changing seed rules."""
import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(r'C:\Users\ta239\Research\softtennis_match_study\study')
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('selector', ROOT / 'scripts/96_select_four_players_pilot.py')
selector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selector)
results = []
for suffix in ('044', '056'):
    path = OUT / f'serve_phase_{suffix}/bytetrack.csv'
    with path.open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    by_frame = defaultdict(list)
    for row in rows:
        by_frame[int(row['source_frame'])].append(row)
    first = 114710 if suffix == '044' else 154051
    valid = []
    failures = []
    for frame in range(first, first + 180):
        try:
            seeds = selector.seed_by_court_geometry(by_frame[frame], frame, 3840, 2160)
            valid.append((frame, seeds))
        except RuntimeError as exc:
            failures.append((frame, str(exc)))
    result = dict(candidate_id=f'match1_rally_window_{suffix}',
                  first_frame=first, last_frame=first+179,
                  first_valid_seed_frame=valid[0][0] if valid else None,
                  frames_with_four_seed_candidates=len(valid),
                  frames_without_four_seed_candidates=len(failures),
                  first_valid_seed_ids=valid[0][1] if valid else None,
                  first_failure=failures[0] if failures else None,
                  warning='Seed availability is not identity accuracy')
    results.append(result)
(OUT / 'serve_seed_availability.json').write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
for result in results:
    print(json.dumps(result, ensure_ascii=False))
