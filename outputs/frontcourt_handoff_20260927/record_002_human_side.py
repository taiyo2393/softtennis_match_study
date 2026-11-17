import csv
from pathlib import Path
p=Path(__file__).resolve().parent
for name in ['rally_label_review.csv','human_review_log.csv','score_point_context_review_queue.csv']:
 with (p/name).open(encoding='utf-8-sig',newline='') as f:
  rd=csv.DictReader(f);fields=rd.fieldnames;rows=list(rd)
 if name=='rally_label_review.csv':
  r=next(r for r in rows if r['candidate_id']=='match1_rally_window_002')
  r.update(winner_camera_side='near',winner_camera_side_review_status='human_confirmed',winner_team_id='上岡俊介・丸山海斗',team_side_mapping_status='initial_anchor_same_first_game',winner_evidence_source_frame='2697',winner_evidence_type='human_video_side_review_and_scoreboard_bottom_1_to_2',winner_review_status='human_side_plus_score_team_derived',reviewer='human_user side; Codex scoreboard linkage 2026-11-02')
  note='User explicitly answered near after 002_score_point_context_h264.mp4. Bottom points 1 to 2 in first game agree with initial near-team anchor. Team label derived; exact service/end frames and position accuracy not confirmed.'
  if note not in r['review_note']:r['review_note']=(r['review_note']+' | '+note).strip(' |')
 elif name=='human_review_log.csv':
  if not any(r['review_id']=='20261102_002_winner_side' for r in rows):rows.append(dict(review_id='20261102_002_winner_side',date='2026-11-02',candidate_id='match1_rally_window_002',field='winner_camera_side',value='near',reviewer='human_user',evidence='002_score_point_context_h264.mp4; user response 手前',prior_value='unreviewed',scope='winner camera side only; team derived separately; exact boundaries and positions not confirmed'))
 else:
  r=next(r for r in rows if r['candidate_id']=='match1_rally_window_002');r['winner_camera_side']='near';r['review_status']='human_winner_side_confirmed_team_derived_boundaries_pending'
 with (p/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print('002 near recorded; team derived; boundary fields preserved')
