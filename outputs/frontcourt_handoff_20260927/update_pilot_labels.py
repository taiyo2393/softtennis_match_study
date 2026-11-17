"""Add evidence-backed, provisional labels for the two reviewed pilot windows."""
import csv
from pathlib import Path

path=Path(__file__).resolve().parent/'rally_label_review.csv'
with path.open(encoding='utf-8-sig',newline='') as f:
    reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
data={
 'match1_rally_window_044':dict(
    winner_team_id='上岡俊介・丸山海斗',winner_camera_side='far',
    winner_evidence_source_frame='115787',winner_evidence_type='broadcast_score_change_bottom_0_to_1',
    review_note='pilot_011; scoreboard bottom team 0 to 1 between F115783 and F115787; far side orange jersey shows 丸山; exact serve and point-end frames pending',
    boundary_review_status='end_interval_known_start_pending',
    winner_review_status='broadcast_score_provisional_human_pending'),
 'match1_rally_window_056':dict(
    winner_team_id='橋場柊一郎・菊山太陽',winner_camera_side='',
    winner_evidence_source_frame='154925',winner_evidence_type='broadcast_score_change_top_2_to_3',
    review_note='pilot_013; scoreboard top team 2 to 3 between F154920 and F154925; a far-side player raises arms near F154890; team-to-camera-side mapping and decisive point-ending event require motion review',
    boundary_review_status='point_end_interval_review_pending',
    winner_review_status='broadcast_score_provisional_human_pending')}
for r in rows:
    if r['candidate_id'] in data:
        if r.get('winner_camera_side_review_status')=='human_confirmed':
            raise RuntimeError('Refusing to overwrite human-reviewed side label: '+r['candidate_id'])
        r.update(data[r['candidate_id']]);r['team_side_mapping_status']=('pending_independent_video_review' if r['candidate_id']=='match1_rally_window_056' else 'video_appearance_provisional_human_pending')
        r['reviewer']='Codex video-frame review 2026-09-27'
with path.open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)
