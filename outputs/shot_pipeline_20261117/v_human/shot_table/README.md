# 本人ラベルによる打球表

正解は `../../labels_human/human_labels_20261118.csv` の condition=without_candidates の32件だけ。labels/ は読まない。
元の `../../build_shot_table.py` の LABEL_COLUMNS と ADDED_COLUMNS を同じ順序・定義で保持し、末尾に rally_no と前後打球間隔を追加。
point/frame/side/is_serve/outcome を point/contact_frame/hitter_side/is_serve/shot_outcome に対応させた。
rally は本人の rally_no の文字列表現、shot_index は同じ点・rally_no 内の時刻順連番。hitter_slot・confidence・tracking_issue・note は本人が付けていないため空欄。
第7点の2つの流れを分離。第9点の同一rally_no内の2サーブ、第11点の最初の非サーブ印を変更しない。

座標は元スクリプトの position_index / nearest_position / pair_metrics を import して使用。
正規化コート座標 x×10.97、y×23.77 m、ネットy=11.885 m。接触フレーム±3F内の最近観測、同距離は早い方。補間・外挿・範囲内へのクリップなし。
全4人のraw座標は保持。第8点手前は線審追跡のため near_positions_trusted=False とし、そのペアの横幅・前後差を空欄にする。
trusted=True はこの既知の除外に該当しない意味で、追跡精度の保証ではない。game/pair/point_winner等の文脈は元と同じ。

間隔は同じpoint・rally_no内だけ。前後がない場合は空欄。秒換算=F×1001/60000。
間隔26件、中央値84F（1.401秒）、最短28F（0.467秒）、最長512F。
第9点の2本のサーブ間も本人の同じrally_noに従い含む。

| point | shots | four_players |
| --- | --- | --- |
| 7 | 8 | 7 |
| 8 | 8 | 8 |
| 9 | 8 | 7 |
| 10 | 4 | 4 |
| 11 | 4 | 4 |

実行: `study\.venv\Scripts\python.exe -B -X utf8 outputs/shot_pipeline_20261117/v_human/shot_table/build_shot_table.py`
