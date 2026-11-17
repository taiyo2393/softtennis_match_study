import csv,json,cv2,numpy as np
from pathlib import Path
p=Path(__file__).resolve().parent;root=Path(r'C:\Users\ta239\Research\softtennis_match_study\study');H=np.array(json.loads((root/'config/match1_court_plane_calibration.json').read_text(encoding='utf-8'))['image_to_court_homography'],np.float32)
with (p/'048_snapshot_person_candidates.csv').open(encoding='utf-8-sig') as f:raw=list(csv.DictReader(f))
slots={0:'near_right',2:'near_left',4:'far_left',1:'far_right'};out=[]
for i,slot in slots.items():
 r=next(r for r in raw if int(r['detection_index'])==i);x=(float(r['x1'])+float(r['x2']))/2;y=float(r['y2']);q=cv2.perspectiveTransform(np.float32([[[x,y]]]),H)[0,0]
 out.append(dict(candidate_id='048',source_frame=125911,image_slot=slot,detection_index=i,bottom_x_px=x,bottom_y_px=y,court_x_candidate=float(q[0]),court_y_candidate=float(q[1]),selection_status='ai_visual_selected_human_pending',position_status='bbox_bottom_approximate_not_ground_truth',contact_first_possible_frame=125859,contact_last_possible_frame=125881,contact_status='ai_visual_bracket_human_pending',effective_serve_status='sequence_supported_human_pending',evidence='048_revised_contact_detail.jpg;048_refined_anchor_person_candidates.jpg'))
with (p/'048_selected_position_candidates.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
with (p/'labeled_rally_position_inventory.csv').open(encoding='utf-8-sig') as f:rs=list(csv.DictReader(f))
r=next(r for r in rs if r['candidate_id'].endswith('_048'));r.update(snapshot_status='ai_four_player_selection_proposed',sample_frame=125911,service_anchor_status='ai_contact_bracket_effective_service_pending',approximate_position_candidates='four_candidates',known_ground_plane_limitation='not_fully_assessed')
with (p/'labeled_rally_position_inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=rs[0]);w.writeheader();w.writerows(rs)
(p/'POSITION_PILOT_CURRENT_STATUS.md').write_text('''# 位置・得点パイロットの現在地（2026-11-02）

得点側の人手確認済み11候補をすべて、サービス後の位置候補を採る段階まで処理した。9候補に4人分の概算座標、005・006に3人分と遮蔽による未解決1人を保存。ただし、サービスが有効であることの人手確認と接触時刻の根拠は候補ごとに異なるため、統計投入可能な11ラリーができたという意味ではない。

048：前の粗い探索範囲より後の2099.2〜2100.2秒を再点検。F125859〜125881をAIの接触候補範囲として保持し、末尾+30のF125911で4選手を選択。初期の切り出しはサーバーを十分に含んでいなかったため破棄せず履歴として残し、今回の048_revised_contact_detail.jpgを判断根拠にする。サーバーの画面側は奥左で、旧粗探索表の奥右という領域推定も修正。箱下端・校正転用の独立誤差は未評価。

005・006：採取時点の前後24ソースフレーム（約0.40秒）を6フレーム刻みで確認。隠れた選手は前後で見えるが、中心時点の足元を直接読める保証は得られない。追跡・補間の可能性はあるが独立正解ではないため、中心時点を別の画像に差し替えず欠測を維持。今回補間座標は生成していない。

次の研究上の判断材料は、概算位置の誤差が大まかな配置差を識別するのに十分かどうか。代表場面を独立注釈し、画像上の誤差・コート変換誤差・選手選択の誤りを分離して測る。空中状態を全件質問する方式には戻らない。サービス時刻の幅による変動も併記する。人手注釈のないAI候補同士を比較して位置精度と呼ばない。

得点チームの偏りは上岡・丸山9件、橋場・菊山2件で、便宜的かつ単一試合の標本。9対2の勝者分布で配置の優劣や予測モデル性能を論じない。次の拡張は別の得点側も含む、勝敗・追跡可否で恣意的に選ばない標本を使う。候補050/051の同一点統合も維持する。
''',encoding='utf-8')
assert len(out)==4 and len(rs)==11
print('048 four coordinates saved; inventory complete:9 full approximate snapshots,2 partial. Occlusion positions remain missing; no inferred coordinates added.')
