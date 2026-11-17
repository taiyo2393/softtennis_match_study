"""Render continuous point context for boundary and side-mapping review."""
import csv
from pathlib import Path

import cv2

ROOT = Path(r'C:\Users\ta239\Research\softtennis_match_study\study')
OUT = Path(__file__).resolve().parent
FPS = 60000 / 1001
WINDOWS = [
    ('044', 'match1_rally_window_044', 114710, 115820),
    ('056', 'match1_rally_window_056', 154050, 154940),
]
cap = cv2.VideoCapture(str(ROOT / 'data/raw/match1.mp4'))
assert cap.isOpened()
rows = []
for name, candidate, first, last in WINDOWS:
    destination = OUT / f'{name}_full_point_context.mp4'
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*'mp4v'),
                             FPS, (1920, 1080))
    assert writer.isOpened(), destination
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    for source_frame in range(first, last + 1):
        ok, image = cap.read()
        assert ok, source_frame
        assert image.shape[:2] == (2160, 3840)
        image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_AREA)
        cv2.rectangle(image, (0, 1030), (420, 1080), (0, 0, 0), -1)
        cv2.putText(image, f'{name}  source F{source_frame}', (12, 1065),
                    cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
        writer.write(image)
    writer.release()
    rows.append(dict(candidate_id=candidate, clip_file=destination.name,
                     first_source_frame=first, last_source_frame=last,
                     start_confirmed='', point_end_confirmed='',
                     winner_team_confirmed='', winner_camera_side_confirmed='',
                     review_status='requires_independent_human_review'))
cap.release()
with (OUT / 'full_point_context_queue.csv').open('w', encoding='utf-8-sig', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print('Wrote', len(rows), 'continuous clips')
