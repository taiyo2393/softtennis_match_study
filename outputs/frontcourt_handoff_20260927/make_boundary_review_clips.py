"""Render short, source-frame-numbered review clips from the original match."""
import csv
from pathlib import Path

import cv2

ROOT = Path(r'C:\Users\ta239\Research\softtennis_match_study\study')
OUT = Path(__file__).resolve().parent
VIDEO = ROOT / 'data/raw/match1.mp4'
FPS = 60000 / 1001
CLIPS = [
    ('044_serve_review', 'match1_rally_window_044', 'serve_contact', 114720, 114780,
     114740, 114760, 'Confirm valid serve and racket contact; do not use candidate-window start'),
    ('044_point_end_review', 'match1_rally_window_044', 'point_end', 115750, 115810,
     115776, 115790, 'Locate decisive rule event separately from broadcast score change'),
    ('056_serve_review', 'match1_rally_window_056', 'serve_contact', 154060, 154110,
     154081, 154086, 'Confirm valid serve and racket contact through camera transition'),
    ('056_point_end_review', 'match1_rally_window_056', 'point_end', 154815, 154935,
     154840, 154880, 'Locate decisive rule event; score changes after candidate-window end'),
]
cap = cv2.VideoCapture(str(VIDEO))
assert cap.isOpened(), VIDEO
rows = []
for name, candidate, event, first, last, proposed_first, proposed_last, question in CLIPS:
    dest = OUT / f'{name}.mp4'
    writer = cv2.VideoWriter(str(dest), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (1920, 1080))
    assert writer.isOpened(), dest
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    for frame_index in range(first, last + 1):
        ok, image = cap.read()
        assert ok, frame_index
        image = cv2.resize(image, (1920, 1080), interpolation=cv2.INTER_AREA)
        cv2.rectangle(image, (0, 0), (530, 60), (0, 0, 0), -1)
        cv2.putText(image, f'{candidate[-3:]} {event} F{frame_index}',
                    (14, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        writer.write(image)
    writer.release()
    rows.append(dict(candidate_id=candidate, event=event, clip_file=dest.name,
                     clip_first_source_frame=first, clip_last_source_frame=last,
                     provisional_first_source_frame=proposed_first,
                     provisional_last_source_frame=proposed_last,
                     reviewer_question=question, confirmed_source_frame='',
                     event_status='requires_independent_human_review'))
cap.release()
with (OUT / 'boundary_review_queue.csv').open('w', encoding='utf-8-sig', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f'Wrote {len(rows)} review clips and queue')
