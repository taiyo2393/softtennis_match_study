import csv
from pathlib import Path
p=Path(__file__).resolve().parent
for name in ['rally_label_review.csv','human_review_log.csv']:
 with (p/name).open(encoding='utf-8-sig',newline='') as f:
  rd=csv.DictReader(f);fields=rd.fieldnames;rows=list(rd)
 if name=='rally_label_review.csv':
  r=next(r for r in rows if r['candidate_id']=='match1_rally_window_006')
  r.update(winner_camera_side='far',winner_camera_side_review_status='human_confirmed',winner_team_id='上岡俊介・丸山海斗',team_side_mapping_status='initial_anchor_first_game_change_and_ai_checkpoint_consistent',winner_evidence_source_frame='12557',winner_evidence_type='human_video_side_review_and_scoreboard_bottom_1_to_2',winner_review_status='human_side_plus_score_team_derived',reviewer='human_user side; Codex scoreboard linkage 2026-11-02')
  note='User answered far after 006_score_point_context_h264.mp4. Bottom score 1 to 2 in second game agrees with initial team-side anchor after the first-game court change. Exact boundaries and positions unconfirmed.'
  if note not in r['review_note']:r['review_note']=(r['review_note']+' | '+note).strip(' |')
 elif not any(r['review_id']=='20261102_006_winner_side' for r in rows):rows.append(dict(review_id='20261102_006_winner_side',date='2026-11-02',candidate_id='match1_rally_window_006',field='winner_camera_side',value='far',reviewer='human_user',evidence='006_score_point_context_h264.mp4; user response 奥',prior_value='unreviewed',scope='winner camera side only; team derived separately; exact boundaries and positions not confirmed'))
 with (p/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print('006 far recorded; team derived; exact boundaries preserved')
