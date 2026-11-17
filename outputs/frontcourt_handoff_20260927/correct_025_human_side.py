import csv
from pathlib import Path
p=Path(__file__).resolve().parent
for name in ['rally_label_review.csv','human_review_log.csv']:
 with (p/name).open(encoding='utf-8-sig',newline='') as f:
  rd=csv.DictReader(f);fields=rd.fieldnames;rows=list(rd)
 if name=='rally_label_review.csv':
  r=next(r for r in rows if r['candidate_id']=='match1_rally_window_025')
  r.update(winner_camera_side='near',winner_camera_side_review_status='human_confirmed',winner_team_id='上岡俊介・丸山海斗',team_side_mapping_status='initial_anchor_game4_completion_and_checkpoint_consistent',winner_review_status='human_side_plus_score_team_derived',winner_evidence_type='human_corrected_video_side_and_post_reset_bottom_0_to_1',reviewer='human_user corrected side; Codex scoreboard linkage 2026-11-02')
  note='User corrected previous far answer to near (ごめん，手前). This supersedes prior side answer and resolves score/mapping conflict. Bottom0-to1 after game2-2 reset agrees with near initial-team mapping. Team derived; exact boundaries remain pending.'
  if note not in r['review_note']:r['review_note']+=' | '+note
 else:
  if not any(r['review_id']=='20261102_025_winner_side_correction' for r in rows):rows.append(dict(review_id='20261102_025_winner_side_correction',date='2026-11-02',candidate_id='match1_rally_window_025',field='winner_camera_side',value='near',reviewer='human_user',evidence='User explicitly corrected prior answer: ごめん，手前; 025_full_score_context_h264.mp4',prior_value='far_human_answer_superseded',scope='winner camera side correction only; team derived separately; exact boundaries and positions unconfirmed'))
 with (p/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
q=p/'025_SCORE_SIDE_CONFLICT.md'
with q.open('a',encoding='utf-8') as f:f.write('\n## 解消\n\n2026-11-02、ユーザーが「ごめん，手前」と訂正。旧far回答は履歴として残し、現在値nearへ更新。下段増点と初期側対応に整合し、得点チームは上岡・丸山と導出。以前の矛盾記述は訂正前の履歴であり、現在の未解決事項ではない。厳密な境界時刻は引き続き未確定。\n')
print('025 corrected to near; history preserved; team derived; prior conflict resolved')
