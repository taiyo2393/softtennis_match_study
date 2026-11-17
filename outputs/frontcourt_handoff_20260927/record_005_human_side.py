import csv
from pathlib import Path
p=Path(__file__).resolve().parent
for name in ['rally_label_review.csv','human_review_log.csv','score_suggestion_blind_review.csv']:
 with (p/name).open(encoding='utf-8-sig',newline='') as f:
  rd=csv.DictReader(f);fields=rd.fieldnames;rows=list(rd)
 if name=='rally_label_review.csv':
  r=next(r for r in rows if r['candidate_id']=='match1_rally_window_005')
  r.update(winner_camera_side='far',winner_camera_side_review_status='human_confirmed',winner_team_id='上岡俊介・丸山海斗',team_side_mapping_status='initial_anchor_first_game_change_and_ai_checkpoint_consistent',winner_evidence_source_frame='9171',winner_evidence_type='human_video_side_review_and_scoreboard_bottom_0_to_1',winner_review_status='human_side_plus_score_team_derived',reviewer='human_user side; Codex scoreboard linkage 2026-11-02')
  note='User explicitly answered far for 005_score_point_context_h264.mp4. Initial bottom team changed to far after game 1; bottom points 0 to 1 agree. Exact boundaries and position accuracy unconfirmed.'
  if note not in r['review_note']:r['review_note']=(r['review_note']+' | '+note).strip(' |')
 elif name=='human_review_log.csv':
  if not any(r['review_id']=='20261102_005_winner_side' for r in rows):rows.append(dict(review_id='20261102_005_winner_side',date='2026-11-02',candidate_id='match1_rally_window_005',field='winner_camera_side',value='far',reviewer='human_user',evidence='005_score_point_context_h264.mp4; user response 奥',prior_value='unreviewed',scope='winner camera side only; team derived separately; no independent score-row or boundary judgment'))
 else:
  r=next(r for r in rows if r['candidate_id']=='match1_rally_window_005');r['winner_camera_side']='far';r['reviewer']='human_user_2026-11-02';r['review_status']='partial_camera_side_only_score_row_linkage_unreviewed'
 with (p/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print('005 far recorded; blind score-row judgment remains blank')
