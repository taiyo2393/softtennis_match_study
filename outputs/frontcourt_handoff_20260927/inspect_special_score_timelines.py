"""One-second broadcast-score timelines for three non-simple candidates."""
import csv
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r'C:\Users\ta239\Research\softtennis_match_study\study')
OUT = Path(__file__).resolve().parent
FPS = 60000 / 1001
with (OUT / 'rally_label_review.csv').open(encoding='utf-8-sig', newline='') as file:
    by_id = {r['candidate_id']: r for r in csv.DictReader(file)}
cap = cv2.VideoCapture(str(ROOT / 'data/raw/match1.mp4'))
for suffix in ('025', '048', '050'):
    row = by_id[f'match1_rally_window_{suffix}']
    end = float(row['candidate_end_sec'])
    tiles = []
    for offset in range(-3, 9):
        frame_id = round((end + offset) * FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        assert ok, frame_id
        crop = cv2.resize(frame[:350, :1250], (900, 252),
                          interpolation=cv2.INTER_AREA)
        cv2.rectangle(crop, (650, 0), (900, 28), (0, 0, 0), -1)
        cv2.putText(crop, f'{offset:+d}s F{frame_id}', (655, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, .56, (255, 255, 255), 1)
        tiles.append(crop)
    sheet = np.vstack([np.hstack(tiles[i:i+2]) for i in range(0, 12, 2)])
    assert cv2.imwrite(str(OUT / f'uncertain_{suffix}_score_timeline.jpg'), sheet)
cap.release()
