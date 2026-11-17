"""Propose a snapshot after the valid second serve; IDs require human review."""
import csv
import json
from pathlib import Path
import cv2
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=Path(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study')
FRAME=115403
FPS=60000/1001
SLOTS={1:'near_rear',2:'near_front',422:'far_rear',20:'far_front'}
selection_confirmed=False
if (OUT/'snapshot_selection_reviews.csv').exists():
    with (OUT/'snapshot_selection_reviews.csv').open(encoding='utf-8-sig',newline='') as f:
        selection_confirmed=any(
            r['candidate_id']=='match1_rally_window_044'
            and r['source_frame']==str(FRAME)
            and {int(t) for t in r['selected_track_ids'].split(';')}==set(SLOTS)
            and r['review_status']=='human_confirmed' for r in csv.DictReader(f))
with (OUT/'full_rally_044/bytetrack.csv').open(encoding='utf-8-sig',newline='') as f:
    candidates=[r for r in csv.DictReader(f)
                if int(r['source_frame'])==FRAME and int(r['track_id']) in SLOTS]
assert len(candidates)==4
cap=cv2.VideoCapture(str(ROOT/'data/raw/match1.mp4'))
cap.set(cv2.CAP_PROP_POS_FRAMES,FRAME)
ok,frame=cap.read()
assert ok
cap.release()
preview=cv2.resize(frame,(1920,1080),interpolation=cv2.INTER_AREA)
cal=json.loads((ROOT/'config/match1_court_plane_calibration.json').read_text(encoding='utf-8'))
H=np.asarray(cal['image_to_court_homography'],np.float32)
rows=[]
for r in candidates:
    tid=int(r['track_id'])
    x1,y1,x2,y2=[round(float(r[k])/2) for k in ('x1','y1','x2','y2')]
    cv2.rectangle(preview,(x1,y1),(x2,y2),(0,255,255),2)
    label={422:(600,520),2:(825,525),20:(1040,530),1:(1150,570)}[tid]
    cv2.putText(preview,f'ID{tid}',label,cv2.FONT_HERSHEY_SIMPLEX,.7,(0,255,255),2)
    cv2.line(preview,(label[0]+30,label[1]+5),(x1,y1),(0,255,255),1)
    point=np.float32([[[float(r['bottom_center_x']),float(r['bottom_center_y'])]]])
    x,y=cv2.perspectiveTransform(point,H)[0,0]
    rows.append(dict(candidate_id='match1_rally_window_044',sample_source_frame=FRAME,
                     serve_attempt_number=2,serve_validity='human_confirmed_valid',
                     sampling_rule='30_frames_after_last_possible_effective_serve_contact',
                     time_after_contact_min_sec=f'{(FRAME-115373)/FPS:.3f}',
                     time_after_contact_max_sec=f'{(FRAME-115365)/FPS:.3f}',
                     player_slot=SLOTS[tid],tracker_id=tid,
                     bottom_center_x_px=r['bottom_center_x'],bottom_center_y_px=r['bottom_center_y'],
                     court_x_normalized=f'{x:.5f}',court_y_normalized=f'{y:.5f}',
                     player_selection_review_status=('human_confirmed_at_snapshot' if selection_confirmed else 'human_pending'),
                     identity_review_status='ai_proposed_human_pending',
                     position_review_status='box_bottom_center_not_ground_truth',
                     calibration_status='visual_alignment_only_transfer_error_unmeasured'))
cv2.putText(preview,f'044 F{FRAME}: second serve +0.501 to +0.634 sec (candidate)',
            (20,1040),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
cv2.imwrite(str(OUT/'044_second_snapshot_F115403.jpg'),preview)
with (OUT/'044_second_snapshot_candidates.csv').open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print('4 candidate positions, exact contact and identity review pending')
