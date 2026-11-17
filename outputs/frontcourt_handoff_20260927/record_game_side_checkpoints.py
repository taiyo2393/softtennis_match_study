import csv
from pathlib import Path
p=Path(__file__).resolve().parent
checkpoints=[(1,'0-0','near',0),(5,'0-1','far',1),(15,'1-1','far',1),(21,'2-1','near',2),(26,'2-2','near',2),(33,'2-3','far',3),(39,'3-3','far',3),(44,'3-4','near',4),(53,'4-4','near',4),(55,'4-4','far',5),(56,'4-4','far',5)]
with (p/'team_side_game_checkpoints.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=['candidate_id','game_score_top_bottom','bottom_team_camera_side','rule_change_count','evidence_image','review_status','scope']);w.writeheader()
 for n,score,side,count in checkpoints:w.writerow(dict(candidate_id=f'match1_rally_window_{n:03}',game_score_top_bottom=score,bottom_team_camera_side=side,rule_change_count=count,evidence_image=f'game_side_probe_{n:03}.jpg',review_status='ai_visual_checkpoint_consistent_with_rule',scope='uniform-pair and court orientation at checkpoint; not continuous person ID or all intervening points'))
with (p/'rally_label_review.csv').open(encoding='utf-8-sig',newline='') as f:
 r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)
r=next(x for x in rows if x['candidate_id']=='match1_rally_window_044')
r['team_side_mapping_status']='initial_human_anchor_and_ai_game_checkpoints_consistent'
r['winner_review_status']='human_side_plus_score_team_and_checkpoint_mapping_derived'
note='2026-10-03: game-start checkpoints 001,005,015,021,026,033,039,044 visually preserve court orientation and alternate black/orange vs white pairs as expected; 044 bottom team near agrees with human winner-side label. Prior AI far-side conflict resolved; continuous identity and boundaries not certified.'
if note not in r['review_note']:r['review_note']+=' | '+note
with (p/'rally_label_review.csv').open('w',encoding='utf-8-sig',newline='') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
(p/'GAME_SIDE_CHECKPOINT_FINDINGS.md').write_text('''# ゲーム進行とコート側の照合

63候補の開始時ゲーム表示を既存画像から目視した。最初に新ゲーム表示が出る候補は005（0-1）、015（1-1）、021（2-1）、026（2-2）、033（2-3）、039（3-3）、044（3-4）、053（4-4）。候補窓が全ポイントを網羅するとは限らず、この番号をポイント番号と同一視しない。

各候補開始2秒後の元画像を同じ16:9縮尺で抽出。初期の手前ペアは黒・橙、相手は白系。各通常ゲームの手前は順に黒・橙、白系、白系、黒・橙、黒・橙、白系、白系、黒・橙で、奇数ゲーム後のチェンジと一致。審判台・背景を含むカメラ方向も各確認点で同じ。

044では初期の手前＝下段上岡・丸山の対応に戻る。人手確認済みの044勝者「手前」と下段ポイント0→1は整合。旧AIの奥ラベルによる対応矛盾を解消した。得点チームは人手の側ラベルと放送表示・AIのチェックポイント照合から導出した値であり、チーム名への独立した人手回答ではない。

ファイナル開始053は黒・橙が手前。055（ポイント1-1）、056（2-1）は白系が手前で、最初の2ポイント後のチェンジと一致。056の上段2→3という表示変化からは手前の勝利が示唆されるが、人手の得点側ラベルは未入力のまま保持した。

これはゲームごとの静止画チェックポイント照合であり、全カット・リプレイ・全ポイントを連続監査したものではない。全63候補へ確定ラベルを一括伝播していない。4人の連続ID精度、接地点誤差、再連結正解は別の検証事項。
''',encoding='utf-8')
print('11 checkpoints saved; 044 mapping conflict resolved; 056 outcome remains human pending')
