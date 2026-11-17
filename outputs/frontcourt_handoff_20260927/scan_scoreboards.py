"""Create an unbiased visual queue for score changes in all candidate windows."""
import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study')
OUT = Path(__file__).resolve().parent
FPS = 60000 / 1001
with (OUT / 'rally_label_review.csv').open(encoding='utf-8-sig', newline='') as stream:
    candidates = list(csv.DictReader(stream))
assert len(candidates) == 63

cap = cv2.VideoCapture(str(ROOT / 'data/raw/match1.mp4'))
assert cap.isOpened()
video_last_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
queue = []
panels = []
for index, candidate in enumerate(candidates):
    start = float(candidate['candidate_start_sec'])
    end = float(candidate['candidate_end_sec'])
    # Fixed sampling schedule for every candidate; later frames can include
    # replays or another point, so they are evidence to review, not labels.
    samples = [('start', start), ('end_minus_1s', end - 1),
               ('end_plus_1s', end + 1), ('end_plus_3s', end + 3)]
    tiles = []
    sample_frames = {}
    for kind, timestamp in samples:
        frame_index = min(video_last_frame, max(0, round(timestamp * FPS)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        assert ok, (candidate['candidate_id'], kind, frame_index)
        crop = frame[:350, :1250].copy()
        assert crop.shape == (350, 1250, 3)
        # Scale the complete scoreboard uniformly, keeping both point digits.
        crop = cv2.resize(crop, (900, 252), interpolation=cv2.INTER_AREA)
        cv2.rectangle(crop, (650, 0), (900, 28), (0, 0, 0), -1)
        cv2.putText(crop, f'{kind} F{frame_index}', (652, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
        tiles.append(crop)
        sample_frames[kind] = frame_index
    title = np.zeros((36, 1800, 3), dtype=np.uint8)
    label = f"{candidate['candidate_id']}  {candidate['legacy_window_status']}"
    cv2.putText(title, label, (8, 25), cv2.FONT_HERSHEY_SIMPLEX,
                .63, (255, 255, 255), 1)
    panel = np.vstack([title, np.hstack(tiles[:2]), np.hstack(tiles[2:])])
    panels.append(panel)
    page_name = f'scoreboard_scan_{index // 3:02d}.jpg'
    queue.append(dict(candidate_id=candidate['candidate_id'],
                      legacy_window_status=candidate['legacy_window_status'],
                      page_image=page_name, **{f'{k}_source_frame': v for k, v in sample_frames.items()},
                      pre_score='', post_score='', provisional_winner_team='',
                      score_visibility='', broadcast_transition_or_replay='',
                      reviewer='', review_status='visual_review_pending'))

cap.release()
for old_page in OUT.glob('scoreboard_scan_[0-9][0-9].jpg'):
    old_page.unlink()
for page in range((len(panels) + 2) // 3):
    group = panels[page * 3:(page + 1) * 3]
    sheet = np.vstack(group)
    assert cv2.imwrite(str(OUT / f'scoreboard_scan_{page:02d}.jpg'), sheet,
                       [cv2.IMWRITE_JPEG_QUALITY, 92])
with (OUT / 'scoreboard_scan_queue.csv').open('w', encoding='utf-8-sig', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(queue[0]))
    writer.writeheader()
    writer.writerows(queue)
print(f'Candidates: {len(queue)}; pages: {(len(panels) + 2) // 3}; score labels remain pending')
