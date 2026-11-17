import csv,json,cv2,numpy as np
from pathlib import Path
p=Path(__file__).resolve().parent;root=Path(r'C:\Users\ta239\Research\softtennis_match_study\study');H=np.array(json.loads((root/'config/match1_court_plane_calibration.json').read_text(encoding='utf-8'))['image_to_court_homography'],np.float32)
with (p/'additional_snapshot_person_detections.csv').open(encoding='utf-8-sig') as f:ds=list(csv.DictReader(f))
selected={'001':{0:'near_right',1:'near_left',4:'far_left',5:'far_right'},'025':{0:'near_right',1:'near_left',3:'far_left',5:'far_right'}};out=[]
for r in ds:
 n=r['candidate_id'];i=int(r['detection_index'])
 if i not in selected[n]:continue
 x=(float(r['x1'])+float(r['x2']))/2;y=float(r['y2']);q=cv2.perspectiveTransform(np.float32([[[x,y]]]),H)[0,0]
 out.append(dict(candidate_id=n,sample_frame=r['sample_frame'],image_slot=selected[n][i],detection_index=i,bottom_x_px=x,bottom_y_px=y,court_x_candidate=float(q[0]),court_y_candidate=float(q[1]),selection_status='ai_visual_four_player_proposal_human_pending',anchor_status='legacy_contact_effective_service_unverified',position_status='bbox_bottom_approximate_not_ground_truth',identity_scope='single_image_no_temporal_ID'))
assert len(out)==8
with (p/'additional_four_player_snapshot_proposals.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
with (p/'rally_label_review.csv').open(encoding='utf-8-sig') as f:labels=list(csv.DictReader(f))
rows=[]
for r in labels:
 if r['winner_camera_side_review_status']!='human_confirmed':continue
 n=r['candidate_id'][-3:];new=n in selected;old=n in ['044','056']
 rows.append(dict(candidate_id=r['candidate_id'],winner_side=r['winner_camera_side'],winner_team_derived=r['winner_team_id'],snapshot_status='human_four_player_selection_confirmed' if old else 'ai_four_player_selection_proposed' if new else 'not_processed_anchor_pending',sample_frame={'001':117,'025':60006,'044':115403,'056':154116}.get(n,''),service_anchor_status='effective_service_valid_contact_interval_provisional' if old else 'legacy_contact_effective_service_pending' if new else 'service_contact_unestablished',approximate_position_candidates='available' if old or new else 'not_extracted',position_accuracy_status='not_independently_evaluated',known_ground_plane_limitation='one_player_airborne_human_confirmed' if n=='056' else 'not_fully_assessed',scope='one_frame_feasibility_not_continuous_tracking_accuracy'))
assert len(rows)==11
with (p/'labeled_rally_position_inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(p/'LABELED_POSITION_FEASIBILITY.md').write_text('''# 得点側確認11候補の配置取得状況

044・056：4人選択は人手確認済み。概算座標候補あり。正確な接触時刻と独立位置誤差は未評価。056では1人が空中にいることを確認済み。概算の足元代表点は残すが、直接観測した地面接地点とは呼ばない。

001・025：既存サービス時刻から30フレーム後（F117・F60006）を新規処理。YOLO11m、imgsz1280、person/conf0.25で検出。画像目視で4選手候補を選び、概算座標8件を別表へ保存。検出番号Dは画像内の一時番号で、追跡IDではない。審判等を含む全人物検出と4選手候補を分離。有効サービスかどうかと接触時刻は改めて確認が必要で、主要分析値にはまだ採用しない。

002・003・004・005・006・007・048：サーブ接触基準が未整備のため未抽出。画像から取得できない欠測、検出失敗、位置取得率0として扱わない。

したがって11件中4件に位置候補を用意したが、「4/11の位置取得成功率」と報告しない。時刻整備待ちと測定上の欠測を別記する。現時点の4件は得点チームが上岡・丸山3件、橋場・菊山1件で偏っており、2ラリー以上へ増えただけで配置と得点の関連を推定しない。

位置定義はまず箱下端中央に基づく概算値と明示する。精密接地の全件質問は行わず、代表場面の独立注釈で誤差を評価し、研究したい配置差を区別できるか判断する。浮遊・遮蔽の状態はフラグに残す。人手正解をAI候補で埋めない。
''',encoding='utf-8')
print('11 rows inventoried; 2 existing human-reviewed snapshots + 2 new AI-selected snapshots; 7 anchor-pending. Eight new candidate positions, no ground truth promotion.')
