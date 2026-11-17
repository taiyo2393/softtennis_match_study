"""Exploratory pixel-offset check for the existing F492 doubles-court outline.

This checks line appearance, not court-plane accuracy or player foot points.
"""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study')
OUT = Path(__file__).resolve().parent
cal = json.loads((ROOT / 'config/match1_court_plane_calibration.json').read_text(encoding='utf-8'))
corners = np.asarray(cal['image_points_px'][:4], np.float32) / 2
edges = (('right_sideline',1,2),
         ('near_baseline',2,3), ('left_sideline',3,0))
cap = cv2.VideoCapture(str(ROOT / 'data/raw/match1.mp4'))
rows = []
for frame_id in (492, 114790, 154116):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    assert ok
    frame = cv2.resize(frame, (1920,1080), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation = hsv[:,:,1].astype(np.float32)
    value = hsv[:,:,2].astype(np.float32)
    # White tape on dark blue court: simultaneous bright and low-saturation.
    whiteness = value - .7*saturation
    for name, a, b in edges:
        p, q = corners[a], corners[b]
        direction = q-p
        normal = np.array([-direction[1], direction[0]], np.float32)
        normal /= np.linalg.norm(normal)
        samples = p + np.linspace(.12,.88,150)[:,None]*direction
        scores = []
        for offset in range(-16,17):
            coords = samples + normal*offset
            x = np.rint(coords[:,0]).astype(int)
            y = np.rint(coords[:,1]).astype(int)
            valid = (x>=0)&(x<1920)&(y>=0)&(y<1080)
            v = whiteness[y[valid],x[valid]]
            scores.append(float(np.quantile(v,.6)))
        best = int(np.argmax(scores))-16
        rows.append(dict(source_frame=frame_id, line=name,
                         best_normal_offset_px=best,
                         score_at_calibration=round(scores[16],2),
                         score_at_best=round(max(scores),2),
                         interpretation='exploratory_line_appearance_not_independent_landmark_error'))
cap.release()
with (OUT/'court_line_alignment_probe.csv').open('w',encoding='utf-8-sig',newline='') as file:
    writer=csv.DictWriter(file,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
for row in rows:
    print(row)
