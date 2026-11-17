import csv
from pathlib import Path
p=Path(__file__).resolve().parent
spec=[('002',37,38,'near_left','later_action_followed_by_play','Two action sequences; earlier30-32s followed by reset; later37-38s followed by sustained play'),('003',69,71,'near_right','first_observed_action_followed_by_play','No retry established in inspected window'),('004',111.5,113,'near_left','later_action_followed_by_play','Earlier far-side motion did not establish server identity; later near orange player serves111.5-112.5s'),('005',143.5,144.5,'near_right','later_action_followed_by_play','Earlier132.5s action followed by reset; later143.5s motion then144.5s play'),('006',175,176,'near_left','first_observed_action_followed_by_play','Service motion then play in coarse sequence'),('007',259,260,'near_right','later_action_followed_by_play','Earlier247-249s action followed by reset; later259s toss and260s follow-through'),('048',2098.5,2099.5,'far_right','first_observed_action_followed_by_play','Distant service then sustained play; precise contact unresolved')]
rows=[]
for n,a,b,region,sequence,note in spec:rows.append(dict(candidate_id='match1_rally_window_'+n,service_leading_to_play_search_start_sec=a,service_leading_to_play_search_end_sec=b,server_camera_region_candidate=region,sequence_type=sequence,attempt_number='',previous_attempt_outcome='unknown',service_validity='ai_sequence_supported_human_pending',contact_status='coarse_search_range_not_confirmed_contact',evidence=f'{n}_later_service_context_grid.jpg' if n in ['002','004','005','007'] else f'{n}_service_context_grid.jpg',notes=note))
with (p/'effective_service_search_intervals.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
(p/'LATER_SERVICE_SEQUENCE_FINDINGS.md').write_text('''# 後続サーブ動作の探索結果

002・004・005・007について候補窓の後半まで1秒間隔で原画像を抽出し、プレーへ続くサーブ動作を探索した。粗い画像列によるAI判断で、人手の有効サービス判定・正確な接触フレームではない。

002は37〜38秒、004は111.5〜113秒、005は143.5〜144.5秒、007は259〜260秒に、その後プレーへ続く動作候補を確認。004は以前の遠方選手の動きからサーバーを推測した所見を修正し、後半の手前左の橙色選手のサーブを候補とする。以前の探索表は履歴で、採取基準は今回のeffective_service_search_intervals.csvを参照する。

003・006・048も合わせ7件の探索範囲を作成した。attempt_numberと前試行結果は空欄／unknown。構え直しだけでファーストフォルトやセカンドサービスと断定しない。タイムスタンプ中央を機械的に接触時刻へ変換しない。

次はこれらの狭い範囲を細かく抽出して接触の前後を囲む。画像更新が約30fps相当の局所区間を含むため、ソース59.94fpsの隣接1枚へ無理に断定せず、接触範囲とその幅を残す。得点表示や人物が見やすい時点を後から選んで位置採取しない。
''',encoding='utf-8')
print('7 provisional service-to-play search ranges saved; attempt outcomes and contact frames not confirmed')
