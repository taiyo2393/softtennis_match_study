"""Create an isolated 056 pilot starting when four candidates first satisfy geometry."""
import csv
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
source = OUT / 'serve_phase_056/bytetrack.csv'
destination = OUT / 'serve_phase_056_deferred'
destination.mkdir(exist_ok=False)
with source.open(encoding='utf-8-sig', newline='') as file:
    reader = csv.DictReader(file)
    fields = reader.fieldnames
    rows = [r for r in reader if int(r['source_frame']) >= 154086]
with (destination / 'bytetrack.csv').open('w', encoding='utf-8-sig', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
(destination / 'origin.json').write_text(json.dumps({
    'source_tracker_file': str(source),
    'tracker_original_first_frame': 154051,
    'deferred_seed_first_frame': 154086,
    'dropped_initial_frames': 35,
    'warning': 'Candidate selection only; no independent identity or position ground truth',
}, ensure_ascii=False, indent=2), encoding='utf-8')
print(len(rows), 'detections copied')
