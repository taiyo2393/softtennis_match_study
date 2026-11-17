import csv
from pathlib import Path
p=Path(__file__).resolve().parent
for name in ['rally_label_review.csv','human_review_log.csv']:
 with (p/name).open(encoding='utf-8-sig',newline='') as f:
  rd=csv.DictReader(f);fields=rd.fieldnames;rows=list(rd)
 if name=='rally_label_review.csv':
  r=next(r for r in rows if r['candidate_id']=='match1_rally_window_025')
  r.update(winner_camera_side='far',winner_camera_side_review_status='human_confirmed',winner_team_id='',team_side_mapping_status='conflict_human_far_vs_score_bottom_and_initial_mapping_pending',winner_review_status='human_camera_side_confirmed_team_unresolved',winner_evidence_type='human_video_side_review',reviewer='human_user side 2026-11-02')
  note='User answered far for serve-started point in 025_full_score_context_h264.mp4. AI scoreboard reading bottom0-to1 after game reset conflicts with initial-anchor rule mapping bottom near. Preserve human side, leave team unknown; review score linkage/camera mapping; no source boundaries changed.'
  if note not in r['review_note']:r['review_note']=(r['review_note']+' | '+note).strip(' |')
 else:
  if not any(r['review_id']=='20261102_025_winner_side' for r in rows):rows.append(dict(review_id='20261102_025_winner_side',date='2026-11-02',candidate_id='match1_rally_window_025',field='winner_camera_side',value='far',reviewer='human_user',evidence='025_full_score_context_h264.mp4; user response 奥',prior_value='unreviewed',scope='camera side only; conflicts with inferred scoreboard/team-side mapping; winner team unresolved'))
 with (p/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
(p/'025_SCORE_SIDE_CONFLICT.md').write_text('''# 025 得点側と表示の矛盾

2026-11-02、ユーザーは025_full_score_context_h264.mp4のサーブから始まるポイントの勝者を「奥」と回答。台帳に人手の側ラベルfarを記録。得点チームは空欄を維持。

既存の画像読取では開始後のゲーム表示2-2、ポイント0-0から下段0-1へ進行。通常ゲーム4ゲーム終了後は奇数ゲーム後のチェンジ2回で、初期下段手前と同じ対応のはず。026のチェックポイントも黒・橙が手前。一方、人手の回答は奥。人手回答をAIに合わせて反転せず、チーム名を機械的に確定しない。

再確認対象：同じポイントへの表示更新の対応、放送映像の編集・リプレイ・時差、試合先頭のアンカーと後続の対応、質問したラリーの取り違え。スコア表示だけで解決したことにしない。
''',encoding='utf-8')
print('025 human far preserved; winner team blank; explicit mapping conflict recorded')
