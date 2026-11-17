"""Overlay the F492 fixed-camera calibration on later point snapshots."""
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study')
OUT = Path(__file__).resolve().parent
cal = json.loads((ROOT / 'config/match1_court_plane_calibration.json').read_text(encoding='utf-8'))
corners = np.rint(np.asarray(cal['image_points_px'][:4], dtype=float) / 2).astype(np.int32)
net = np.rint(np.asarray(cal['image_points_px'][4:], dtype=float) / 2).astype(np.int32)
capture = cv2.VideoCapture(str(ROOT / 'data/raw/match1.mp4'))
assert capture.isOpened()
for frame_index in (492, 114790, 154116):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    assert ok
    frame = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(OUT / f'calibration_raw_F{frame_index}.jpg'), frame)
    cv2.polylines(frame, [corners], True, (255, 0, 255), 3)
    for index, point in enumerate(corners, start=1):
        x, y = map(int, point)
        cv2.circle(frame, (x, y), 12, (0, 255, 255), 3)
        cv2.putText(frame, f'C{index}', (x+14, y-12), cv2.FONT_HERSHEY_SIMPLEX,
                    .75, (0, 255, 255), 2)
    for index, point in enumerate(net, start=1):
        x, y = map(int, point)
        cv2.circle(frame, (x, y), 11, (0, 255, 0), 3)
        cv2.putText(frame, f'N{index}', (x+14, y-12), cv2.FONT_HERSHEY_SIMPLEX,
                    .75, (0, 255, 0), 2)
    cv2.rectangle(frame, (0, 1030), (700, 1080), (0, 0, 0), -1)
    cv2.putText(frame, f'F{frame_index}  F492 calibration overlay', (10, 1065),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
    assert cv2.imwrite(str(OUT / f'calibration_transfer_F{frame_index}.jpg'), frame)
    tiles = []
    points = [(f'C{i}', p) for i, p in enumerate(corners, 1)]
    points += [(f'N{i}', p) for i, p in enumerate(net, 1)]
    for label, point in points:
        x, y = map(int, point)
        tile = frame[max(0,y-110):min(1080,y+110), max(0,x-150):min(1920,x+150)]
        tile = cv2.copyMakeBorder(tile, max(0,220-tile.shape[0]), 0, 0,
                                  max(0,300-tile.shape[1]), cv2.BORDER_CONSTANT)
        cv2.putText(tile, label, (10,30), cv2.FONT_HERSHEY_SIMPLEX,
                    .9, (255,255,255), 2)
        tiles.append(tile)
    sheet = np.concatenate([np.concatenate(tiles[:3], axis=1),
                            np.concatenate(tiles[3:], axis=1)], axis=0)
    assert cv2.imwrite(str(OUT / f'calibration_transfer_zoom_F{frame_index}.jpg'), sheet)
capture.release()


