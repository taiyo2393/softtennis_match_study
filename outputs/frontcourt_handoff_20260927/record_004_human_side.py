import csv
from pathlib import Path
p=Path(__file__).resolve().parent
for name in ['rally_label_review.csv','human_review_log.csv']:
 with (p/name).open(encoding='utf-8-sig',newline='') as f:
  rd=csv.DictReader(f);fields=rd.fieldnames;rows=list(rd)
 if name=='rally_label_review.csv':
  r=next(r for r in rows if r['candidate_id']=='match1_rally_window_004')
  r.update(winner_camera_side='near',winner_camera_side_review_status='human_confirmed',winner_team_id='上岡俊介・丸山海斗',team_side_mapping_status='initial_anchor_same_first_game',winner_evidence_source_frame='6953',winner_evidence_type='human_video_side_review_and_scoreboard_bottom_3_to_4',winner_review_status='human_side_plus_score_team_derived',reviewer='human_user side; Codex scoreboard linkage 2026-11-02')
  note='User answered near after 004_score_point_context_h264.mp4. Bottom score 3 to 4 in first game agrees with initial team-side anchor. Exact boundaries and positions unconfirmed.'
  if note not in r['review_note']:r['review_note']=(r['review_note']+' | '+note).strip(' |')
 elif not any(r['review_id']=='20261102_004_winner_side' for r in rows):rows.append(dict(review_id='20261102_004_winner_side',date='2026-11-02',candidate_id='match1_rally_window_004',field='winner_camera_side',value='near',reviewer='human_user',evidence='004_score_point_context_h264.mp4; user response 手前',prior_value='unreviewed',scope='winner camera side only; team derived separately; exact boundaries and positions not confirmed'))
 with (p/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print('004 near recorded; team derived; exact boundaries preserved')
