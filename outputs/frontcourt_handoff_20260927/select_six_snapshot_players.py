import csv,json,cv2,numpy as np
from pathlib import Path
p=Path(__file__).resolve().parent;root=Path(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study');H=np.array(json.loads((root/'config/match1_court_plane_calibration.json').read_text(encoding='utf-8'))['image_to_court_homography'],np.float32)
def read(n):
 with (p/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
raw=read('six_service_snapshot_person_candidates.csv');anchors={r['candidate_id'][-3:]:r for r in read('six_service_contact_brackets.csv')}
sel={'002':{'near_left':0,'near_right':1,'far_left':4,'far_right':2},'003':{'near_left':0,'near_right':1,'far_left':4,'far_right':2},'004':{'near_left':1,'near_right':0,'far_left':4,'far_right':2},'005':{'near_left':0,'near_right':1,'far_left':None,'far_right':2},'006':{'near_left':0,'near_right':1,'far_left':3,'far_right':None},'007':{'near_left':0,'near_right':1,'far_left':4,'far_right':2}}
rows=[];cap=cv2.VideoCapture(str(root/'data/raw/match1.mp4'))
for n,slots in sel.items():
 f=int(anchors[n]['sample_candidate_frame']);cap.set(1,f);ok,im=cap.read();assert ok;im=cv2.resize(im,(1920,1080))
 for slot,i in slots.items():
  r=next((r for r in raw if r['candidate_id']==n and int(r['detection_index'])==i),None) if i is not None else None
  row=dict(candidate_id=n,source_frame=f,image_slot=slot,detection_index=i if i is not None else '',bottom_x_px='',bottom_y_px='',court_x_candidate='',court_y_candidate='',selection_status='ai_visual_selected_human_pending' if r else 'no_usable_full_player_box',position_status='bbox_bottom_approximate_not_ground_truth' if r else 'candidate_position_missing_occlusion',missing_reason='' if r else 'far_black_player_occluded_no_distinct_full_box' if n=='005' else 'far_orange_player_mostly_occluded_head_racket_box_D12_not_ground_position',anchor_status='ai_contact_bracket_effective_service_pending',identity_scope='single_image_only')
  if r:
   x=(float(r['x1'])+float(r['x2']))/2;y=float(r['y2']);q=cv2.perspectiveTransform(np.float32([[[x,y]]]),H)[0,0];row.update(bottom_x_px=x,bottom_y_px=y,court_x_candidate=float(q[0]),court_y_candidate=float(q[1]))
   a,b,c,d=[round(float(r[k])/2) for k in ['x1','y1','x2','y2']];cv2.rectangle(im,(a,b),(c,d),(0,255,255),2);cv2.putText(im,f'{slot} D{i}',(a,max(20,b-10)),0,.55,(0,255,255),2)
  rows.append(row)
 missing=sum(i is None for i in slots.values());cv2.putText(im,f'{n} F{f}: {4-missing} approximate position candidates, {missing} unresolved',(20,1030),0,.7,(255,255,255),2);cv2.imwrite(str(p/f'{n}_selected_match_player_candidates.jpg'),im)
cap.release();assert len(rows)==24 and sum(bool(r['bottom_x_px']) for r in rows)==22
with (p/'six_service_selected_position_candidates.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
inventory=read('labeled_rally_position_inventory.csv')
for r in inventory:
 n=r['candidate_id'][-3:]
 if n in sel:
  complete=all(i is not None for i in sel[n].values());r['snapshot_status']='ai_four_player_selection_proposed' if complete else 'ai_three_positions_one_occlusion_unresolved';r['sample_frame']=anchors[n]['sample_candidate_frame'];r['service_anchor_status']='ai_contact_bracket_effective_service_pending';r['approximate_position_candidates']='four_candidates' if complete else 'three_candidates_one_missing';r['known_ground_plane_limitation']='not_fully_assessed' if complete else 'one_far_player_occluded_no_usable_footpoint_candidate'
with (p/'labeled_rally_position_inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=inventory[0]);w.writeheader();w.writerows(inventory)
(p/'SIX_SNAPSHOT_SELECTION_FINDINGS.md').write_text('''# 追加6候補の4選手選択と概算位置

2026-11-02、各候補の画像をAIが目視して試合選手と審判等を分離。D番号は当該1画像の検出番号で、連続IDではない。サーブ接触範囲・有効性・座標誤差の人手正解は未確定。

002/003/004/007：それぞれ4選手の枠をAIが選択し、箱下端中央と既存ホモグラフィによる概算座標を保存。005：奥左の黒い選手は手前の白い選手の背後に重なり、独立した全身枠を選べない。006：奥右の橙の選手が手前白い選手に遮蔽され、D12は頭・ラケット付近の小枠。これを選手全身として採ると足元が大幅に誤るため、地上位置候補へ変換しない。両候補とも他の3人だけ座標候補を保存し、残る1人は空欄。

24枠のうち22個の概算候補、2個の未解決位置。これは人手正解に対する22/24の精度でも、ラリー全体の可用率でもない。遮蔽時に既存検出から足元候補を選べないというAI所見であり、手動での復元が不可能との断定ではない。4人が揃う近傍画像へ採取時点をずらしていない。

得点側確認済み11候補の現状：8件に4人分の概算候補（うち044/056の4人選択のみ人手確認）、2件に3人分と未解決1人、048はサーブ接触未整備。位置候補がある全件を主要分析適格に昇格させていない。次の優先事項は独立位置誤差の小規模評価と、遮蔽2件の人手修正で補えるかの検討。勝敗別の比較はまだ行わない。
''',encoding='utf-8')
print('24 slots:22 approximate coordinates,2 unresolved occlusions. Four complete snapshots,two partial. Updated 11-row inventory.')
