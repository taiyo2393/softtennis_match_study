import csv
from pathlib import Path
p=Path(__file__).resolve().parent
specs=[('002',30,32,'near_left','possible_serve_retry','Service motion followed by reset-like positions33-35s; inspect later service before sampling'),('003',69,71,'near_right','service_followed_by_play','Service motion69-70s followed by play71-74s; validity not human confirmed'),('004',105.5,109.5,'far_right','attempt_sequence_unresolved','Far-side action while near players reposition; service contact not located'),('005',132,134,'near_right','possible_serve_retry','Motion132.5s followed by reset-like positions134.5-138.5s; do not assume effective first attempt'),('006',174,176,'near_left','service_followed_by_play','Toss174-175s, follow-through176s, play177-179s'),('007',247,250,'near_right','possible_serve_retry','Motion247-249s followed by reset-like positions250-253s; inspect later service'),('048',2097,2100,'far_right','service_followed_by_play','Service starts in distant view; play2100.5-2103.5s')]
rows=[]
for n,a,b,server,cat,note in specs:rows.append(dict(candidate_id='match1_rally_window_'+n,first_action_search_start_sec=a,first_action_search_end_sec=b,server_camera_region_candidate=server,sequence_category=cat,contact_status='coarse_search_interval_not_contact_label',effective_serve_status='unreviewed',evidence=f'{n}_service_context_grid.jpg',reviewer='Codex sparse-frame review 2026-11-02',notes=note))
with (p/'remaining_service_search_candidates.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(p/'REMAINING_SERVICE_SEARCH_FINDINGS.md').write_text('''# 得点側確認済み7件のサーブ探索

1秒間隔の9画像×7候補を原動画から抽出し、AIが動作とその後の文脈を目視した。時刻は接触ラベルではなく次の詳細探索範囲。

003・006・048ではサーブ動作からプレーへ移る系列が見える。003は69〜71秒、006は174〜176秒、048は2097〜2100秒を詳細探索する。これらを有効サービス確定としない。

002・005・007では最初のサービス動作後に構え直しらしい姿勢が見える。フォルト、レット、トスやり直し、その他を静止画列だけでは区別できず、試行結果はunknown。候補開始から最初のスイングだけを基準に位置を採取しない。004も開始付近の動作とその後の構えを含め試行系列が未解決。

次は候補窓の後半まで見て有効サービス候補を探し、その候補範囲だけ細かく確認する。人手確認済みの勝者ラベルは変更せず、粗い探索範囲の中央を接触時刻として埋めない。今回、7件の確定接触ラベルや位置座標を新たに作ったわけではない。
''',encoding='utf-8')
print('7 service searches classified: 3 service-to-play sequences; 3 possible retries; 1 unresolved sequence. No contact labels promoted.')
