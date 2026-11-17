import csv
from pathlib import Path
p=Path(__file__).resolve().parent
for name in ['rally_label_review.csv','full_point_context_queue.csv','human_review_log.csv']:
 with (p/name).open(encoding='utf-8-sig',newline='') as f:
  reader=csv.DictReader(f); fields=reader.fieldnames;rows=list(reader)
 if name=='rally_label_review.csv':
  r=next(x for x in rows if x['candidate_id']=='match1_rally_window_056')
  r.update(winner_camera_side='near',winner_camera_side_review_status='human_confirmed',winner_team_id='橋場柊一郎・菊山太陽',team_side_mapping_status='initial_human_anchor_and_ai_game_checkpoints_consistent',winner_review_status='human_side_plus_score_team_and_checkpoint_mapping_derived',reviewer='human_user side; Codex scoreboard and checkpoint mapping 2026-10-03')
  note='2026-10-03 user explicitly labeled 056 winner near after 056_full_point_context_h264.mp4. Score top 2 to 3 and final-game side checkpoint after first two points yield Hashiba/Kikuyama. Human confirmed camera side only; boundaries, serve validity, positions and continuous IDs are separate.'
  if note not in r['review_note']:r['review_note']+=' | '+note
 elif name=='full_point_context_queue.csv':
  r=next(x for x in rows if x['candidate_id']=='match1_rally_window_056')
  r.update(clip_file='056_full_point_context_h264.mp4',winner_camera_side_confirmed='near',review_status='winner_side_human_confirmed_team_derived_other_fields_pending')
 elif not any(x['review_id']=='20261003_056_winner_side' for x in rows):
  rows.append(dict(review_id='20261003_056_winner_side',date='2026-10-03',candidate_id='match1_rally_window_056',field='winner_camera_side',value='near',reviewer='human_user',evidence='056_full_point_context_h264.mp4; user response 手前',prior_value='unknown',scope='camera side only; team derived from score and checkpoint mapping; no serve/boundary/position/ID confirmation'))
 with (p/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print('056 human winner side recorded; team derived separately')
