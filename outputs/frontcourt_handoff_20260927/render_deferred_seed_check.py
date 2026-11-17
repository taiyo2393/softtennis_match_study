"""Render candidate IDs at the first viable deferred seed and ten frames later."""
import csv
from pathlib import Path

import cv2

ROOT = Path(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study')
OUT = Path(__file__).resolve().parent
cap = cv2.VideoCapture(str(ROOT / 'data/raw/match1.mp4'))
for pilot, frame_id, chosen_ids in (
    ('044', 114710, (1, 2, 3, 5)),
    ('044', 114790, (1, 2, 3, 5)),
    ('056', 154086, (1, 2, 4, 42)),
    ('056', 154096, (1, 2, 4, 42)),
    ('056', 154116, (1, 2, 4, 42)),
):
    with (OUT / f'serve_phase_{pilot}/bytetrack.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    assert ok
    for row in rows:
        if int(row['source_frame']) != frame_id or int(row['track_id']) not in chosen_ids:
            continue
        x1, y1, x2, y2 = (round(float(row[k])) for k in ('x1', 'y1', 'x2', 'y2'))
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 6)
        cv2.putText(frame, f"ID {row['track_id']}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
    frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    assert cv2.imwrite(str(OUT / f'deferred_seed_F{frame_id}.jpg'), frame)
cap.release()
