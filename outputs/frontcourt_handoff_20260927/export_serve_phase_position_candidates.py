"""Export two provisional four-box snapshots on a shared post-serve sampling rule."""
import csv
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(r'C:\Users\ta239\Research\softtennis_match_study\study')
OUT = Path(__file__).resolve().parent
FPS = 60000 / 1001
calibration = json.loads((ROOT / 'config/match1_court_plane_calibration.json').read_text(encoding='utf-8'))
assert calibration['image_size_px'] == {'width': 3840, 'height': 2160}
H = np.asarray(calibration['image_to_court_homography'], dtype=np.float32)
samples = (
    ('044', 114790, 114740, 114760, OUT / 'serve_phase_044/bytetrack_selected_four.csv'),
    ('056', 154116, 154081, 154086,
     OUT / 'serve_phase_056_deferred/bytetrack_four_player_reconnection.csv'),
)
result = []
for suffix, frame, contact_first, contact_last, path in samples:
    with path.open(encoding='utf-8-sig', newline='') as file:
        candidates = [r for r in csv.DictReader(file) if int(r['source_frame']) == frame]
    assert len(candidates) == 4
    assert len({r['player_slot'] for r in candidates}) == 4
    points = np.float32([[[float(r['bottom_center_x']), float(r['bottom_center_y'])]
                         for r in candidates]])
    transformed = cv2.perspectiveTransform(points, H)[0]
    for candidate, (court_x, court_y) in zip(candidates, transformed):
        result.append(dict(candidate_id=f'match1_rally_window_{suffix}',
                           sample_source_frame=frame,
                           sampling_rule='30_frames_after_last_possible_serve_contact',
                           sampling_validity_status=('invalid_effective_serve_anchor_first_fault'
                                                     if suffix=='044' else
                                                     'effective_serve_anchor_review_pending'),
                           time_after_contact_min_sec=f'{(frame-contact_last)/FPS:.3f}',
                           time_after_contact_max_sec=f'{(frame-contact_first)/FPS:.3f}',
                           player_slot=candidate['player_slot'],
                           tracker_id=candidate['track_id'],
                           bottom_center_x_px=candidate['bottom_center_x'],
                           bottom_center_y_px=candidate['bottom_center_y'],
                           court_x_normalized=f'{court_x:.5f}',
                           court_y_normalized=f'{court_y:.5f}',
                           detection_confidence=candidate['confidence'],
                           position_source=candidate.get('position_source') or 'observed_original',
                           identity_review_status='not_independently_confirmed',
                           position_review_status='box_bottom_center_not_ankle_ground_truth',
                           calibration_status='visual_doubles_line_alignment_only_transfer_error_unmeasured'))
with (OUT / 'serve_phase_position_snapshot_candidates.csv').open('w', encoding='utf-8-sig', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=list(result[0]))
    writer.writeheader()
    writer.writerows(result)
for row in result:
    print(row['candidate_id'][-3:], row['player_slot'], row['tracker_id'],
          row['court_x_normalized'], row['court_y_normalized'],
          row['time_after_contact_min_sec'], row['time_after_contact_max_sec'])
