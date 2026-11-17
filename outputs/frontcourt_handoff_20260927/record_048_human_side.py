import csv
from pathlib import Path
p=Path(__file__).resolve().parent
for name in ['rally_label_review.csv','human_review_log.csv','score_suggestion_blind_review.csv']:
 with (p/name).open(encoding='utf-8-sig',newline='') as f:
  rd=csv.DictReader(f);fields=rd.fieldnames;rows=list(rd)
 if name=='rally_label_review.csv':
  r=next(r for r in rows if r['candidate_id']=='match1_rally_window_048')
  r.update(winner_camera_side='far',winner_camera_side_review_status='human_confirmed',winner_team_id='橋場柊一郎・菊山太陽',team_side_mapping_status='initial_anchor_and_game8_checkpoint_consistent',winner_review_status='human_side_plus_score_team_derived',winner_evidence_type='human_video_side_and_top_2_to_3_before_camera_exit',reviewer='human_user side; Codex scoreboard linkage 2026-11-02')
  note='User answered far after 048_finish_score_context_h264.mp4. Top2-to3 with game3-4 and game8 side mapping yields Hashiba/Kikuyama. Score update before camera exit is not point-ending frame; exact boundaries unconfirmed.'
  if note not in r['review_note']:r['review_note']=(r['review_note']+' | '+note).strip(' |')
 elif name=='human_review_log.csv':
  if not any(r['review_id']=='20261102_048_winner_side' for r in rows):rows.append(dict(review_id='20261102_048_winner_side',date='2026-11-02',candidate_id='match1_rally_window_048',field='winner_camera_side',value='far',reviewer='human_user',evidence='048_finish_score_context_h264.mp4; user response 奥',prior_value='unreviewed',scope='winner side only; team derived from score and mapping; exact end frame and tracking unconfirmed'))
 else:
  r=next(r for r in rows if r['candidate_id']=='match1_rally_window_048');r['winner_camera_side']='far';r['reviewer']='human_user_2026-11-02';r['review_status']='partial_camera_side_only_score_row_linkage_unreviewed'
 with (p/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
(p/'FOCUSED_BOUNDARY_REVIEW_STATUS.md').write_text('''# 重点3箇所の確認結果（2026-11-02）

025：ユーザーは初回答の奥を手前へ訂正。下段増点・ゲーム2-2での側対応と整合し、上岡・丸山の得点と導出。開始時に残った前ゲーム表示を基準に単純差分を取らない。

048：ユーザーは奥と回答。上段2→3と第8ゲームの側対応と整合し、橋場・菊山の得点と導出。得点表示がカメラ退出より先に変わること自体は異常でなく、カメラ退出を得点成立時刻としてはいけない。

050/051：ユーザーが同じラリーと確認。元の63候補を残したまま同じポイント群へ対応づけ、得点を二重計上しない。得点側は人手未確認。上段3→4は機械の候補として1件だけ保持。

現在は候補63行・暫定ポイント群62。人手得点側確認11件。人手による接続同一性確認2件。これらは本人同定精度、位置精度、独立な勝者チーム判定件数を意味しない。人手の側回答に、AIによる放送スコア読取と側対応を組み合わせてチームを導出している。厳密境界・位置誤差評価は別項目。

次は確認済み11ラリーの勝者ラベルと概算位置取得可能性を対応づけ、同じ基準で取得可能／欠測を報告する。通常候補の逐次質問は再開せず、既存の固定検証標本に基づく必要な確認をまとめる。未確認の機械候補を確定勝者として統計へ使わない。
''',encoding='utf-8')
print('048 far recorded; top winner team derived; exact end not confirmed')
