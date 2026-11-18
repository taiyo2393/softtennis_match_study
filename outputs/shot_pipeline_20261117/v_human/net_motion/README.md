# 本人ラベルによる前衛モーション再集計

本人の without_candidates 32打球、5点・6つのrally_no区間を対象とする。各打球に対する相手側の前衛を追うため、対象イベントは32件（約30件）。
末尾の打球も+Wまで解析する。labels/、旧shot_table、Claude目視参照は使用しない。
打球時刻・side・is_serve・outcome・rally_noは本人の値。打者slot・モーション有無・方向は本人の正解にはない。
したがって以下は自動分類の**構成比**で、モーション認識の精度評価ではない。

## v2から引き継ぐ規則

`../../net_motion/v2/classify_net_motion_v2.py` をimportし、load_coordinates / event_tracks / classify_event / smooth_positions / paper_tables とv1.summarizeを使用。
分類時の終点は min(同じpoint・rally_no内の次の打球, c+round(W×fps))。主条件W=0.4秒、D=0.3m、感度条件W=0.3/0.4/0.5秒、D=0.2/0.3/0.5m。
fps=60000/1001、主条件24F≈0.4004秒。次打球が前衛の打球でも例外なし。区間始点は打球前6Fの生座標中央値（4観測以上必要）。
中心中央値5F→中心平均3Fで平滑化、角度75°以上・各分割4F以上・各変位>=Dで単/複を分類。
欠測20%超、連続欠測2F超、端点欠測、tracking segment境界等は判定不能。定数・方向定義はv2のまま。
3方向と8方向は区切り角度が違うため、8方向の前3区分合計は3方向の前と一致しない場合がある。

## 前衛の選択と仮定

非サーブ: 打球時に相手側2人のうちネットから近い方（箱位置）。2人が揃わない場合は役割不明。
ネット距離差0.5m未満はrole_ambiguousとして記録し、v2同様、自動的には除外しない。
サーブ: 次の打球がレシーブ側の場合、そのフレームでレシーブ側2人のうち**ネットから遠い方をレシーバーと仮定**し、もう1人を前衛に選ぶ。
打者slotの本人正解はないためこの仮定を検証していない。同距離はslot名順。足首条件の役割選択も箱位置で行う。
次が同じ側・次打球なし・次打球時の2人位置不足の場合は判定不能。第9点F27464は次が奥側のサーブF27976のため、このレシーバー仮定を適用できない。
第11点F32755には本人がサーブ印を付けていないため非サーブとして処理。is_serveの補完、ラリー分割の変更をしない。

## 箱と足首

箱下端は既存正規化座標×(10.97,23.77)m。load_coordinatesで箱からの射影再現を全17022行でassert。
最大正規化誤差=3.11e-15。射影式の一致であり実位置の精度検証ではない。
足首はpose/pose_cache直下の既存CSVだけを使用。旧A/Bと本人rally_noの名前は対応させず、各キャッシュの実フレーム範囲で選ぶ。
左右conf>=0.3は中点、片方だけならその足首、両方欠測は欠測。ファイルなし・範囲外だけ箱fallback。
キャッシュ範囲内の低信頼やslot欠測は箱で埋めない。前6Fから終点までで足首と箱が混在する場合はv2同様判定不能。
各イベントにfoot_source_counts_json / bbox_fallback_frames / ankle_valid_frames / uses_bbox_fallback / mixed_ankle_bboxを保存。
第8点手前は線審追跡の問題があるため、主に解釈するのは **exclude_p8_near**。allも監査用に併記。

| coordinate_method | events | bbox_fallback_events | mixed_proxy_events | bbox_fallback_frames | ankle_valid_frames |
| --- | --- | --- | --- | --- | --- |
| ankle | 32 | 4 | 0 | 124 | 806 |
| bbox | 32 | 0 | 0 | 0 | 0 |

## 主条件の分類件数（第8点手前を含む全件）

| coordinate_method | classification | n |
| --- | --- | --- |
| ankle | compound | 7 |
| ankle | none | 5 |
| ankle | single | 16 |
| ankle | unclassifiable | 4 |
| bbox | compound | 3 |
| bbox | none | 7 |
| bbox | single | 18 |
| bbox | unclassifiable | 4 |

## 主条件の割合（サーブ含む）

分母: presenceは判定可能、complexityとfinal方向はモーションあり、single方向は単。判定不能は「なし」にしない。
summary_main.csvにはサーブ除外条件も保存。summary.csvは全18条件（足元2×W3×D3）と全スコープを含む。

| coordinate_method | scope | metric | category_ja | n | denominator | percent | total_shots | classifiable | unclassifiable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ankle | all | availability | 判定不能 | 4 | 32 | 12.5 | 32 | 28 | 4 |
| ankle | all | presence | あり | 23 | 28 | 82.143 | 32 | 28 | 4 |
| ankle | all | presence | なし | 5 | 28 | 17.857 | 32 | 28 | 4 |
| ankle | all | complexity | 単 | 16 | 23 | 69.565 | 32 | 28 | 4 |
| ankle | all | complexity | 複 | 7 | 23 | 30.435 | 32 | 28 | 4 |
| ankle | all | single_3dir | 前 | 7 | 16 | 43.75 | 32 | 28 | 4 |
| ankle | all | single_3dir | 横 | 2 | 16 | 12.5 | 32 | 28 | 4 |
| ankle | all | single_3dir | 後ろ | 7 | 16 | 43.75 | 32 | 28 | 4 |
| ankle | all | single_8dir | 左斜め前 | 1 | 16 | 6.25 | 32 | 28 | 4 |
| ankle | all | single_8dir | まっすぐ前 | 3 | 16 | 18.75 | 32 | 28 | 4 |
| ankle | all | single_8dir | 右斜め前 | 3 | 16 | 18.75 | 32 | 28 | 4 |
| ankle | all | single_8dir | 左横 | 0 | 16 | 0.0 | 32 | 28 | 4 |
| ankle | all | single_8dir | 右横 | 0 | 16 | 0.0 | 32 | 28 | 4 |
| ankle | all | single_8dir | 左斜め後ろ | 2 | 16 | 12.5 | 32 | 28 | 4 |
| ankle | all | single_8dir | まっすぐ後ろ | 3 | 16 | 18.75 | 32 | 28 | 4 |
| ankle | all | single_8dir | 右斜め後ろ | 4 | 16 | 25.0 | 32 | 28 | 4 |
| ankle | all | final_3dir | 前 | 10 | 23 | 43.478 | 32 | 28 | 4 |
| ankle | all | final_3dir | 横 | 5 | 23 | 21.739 | 32 | 28 | 4 |
| ankle | all | final_3dir | 後ろ | 8 | 23 | 34.783 | 32 | 28 | 4 |
| ankle | all | final_8dir | 左斜め前 | 2 | 23 | 8.696 | 32 | 28 | 4 |
| ankle | all | final_8dir | まっすぐ前 | 5 | 23 | 21.739 | 32 | 28 | 4 |
| ankle | all | final_8dir | 右斜め前 | 5 | 23 | 21.739 | 32 | 28 | 4 |
| ankle | all | final_8dir | 左横 | 0 | 23 | 0.0 | 32 | 28 | 4 |
| ankle | all | final_8dir | 右横 | 0 | 23 | 0.0 | 32 | 28 | 4 |
| ankle | all | final_8dir | 左斜め後ろ | 3 | 23 | 13.043 | 32 | 28 | 4 |
| ankle | all | final_8dir | まっすぐ後ろ | 4 | 23 | 17.391 | 32 | 28 | 4 |
| ankle | all | final_8dir | 右斜め後ろ | 4 | 23 | 17.391 | 32 | 28 | 4 |
| ankle | exclude_p8_near | availability | 判定不能 | 3 | 28 | 10.714 | 28 | 25 | 3 |
| ankle | exclude_p8_near | presence | あり | 22 | 25 | 88.0 | 28 | 25 | 3 |
| ankle | exclude_p8_near | presence | なし | 3 | 25 | 12.0 | 28 | 25 | 3 |
| ankle | exclude_p8_near | complexity | 単 | 15 | 22 | 68.182 | 28 | 25 | 3 |
| ankle | exclude_p8_near | complexity | 複 | 7 | 22 | 31.818 | 28 | 25 | 3 |
| ankle | exclude_p8_near | single_3dir | 前 | 6 | 15 | 40.0 | 28 | 25 | 3 |
| ankle | exclude_p8_near | single_3dir | 横 | 2 | 15 | 13.333 | 28 | 25 | 3 |
| ankle | exclude_p8_near | single_3dir | 後ろ | 7 | 15 | 46.667 | 28 | 25 | 3 |
| ankle | exclude_p8_near | single_8dir | 左斜め前 | 1 | 15 | 6.667 | 28 | 25 | 3 |
| ankle | exclude_p8_near | single_8dir | まっすぐ前 | 2 | 15 | 13.333 | 28 | 25 | 3 |
| ankle | exclude_p8_near | single_8dir | 右斜め前 | 3 | 15 | 20.0 | 28 | 25 | 3 |
| ankle | exclude_p8_near | single_8dir | 左横 | 0 | 15 | 0.0 | 28 | 25 | 3 |
| ankle | exclude_p8_near | single_8dir | 右横 | 0 | 15 | 0.0 | 28 | 25 | 3 |
| ankle | exclude_p8_near | single_8dir | 左斜め後ろ | 2 | 15 | 13.333 | 28 | 25 | 3 |
| ankle | exclude_p8_near | single_8dir | まっすぐ後ろ | 3 | 15 | 20.0 | 28 | 25 | 3 |
| ankle | exclude_p8_near | single_8dir | 右斜め後ろ | 4 | 15 | 26.667 | 28 | 25 | 3 |
| ankle | exclude_p8_near | final_3dir | 前 | 9 | 22 | 40.909 | 28 | 25 | 3 |
| ankle | exclude_p8_near | final_3dir | 横 | 5 | 22 | 22.727 | 28 | 25 | 3 |
| ankle | exclude_p8_near | final_3dir | 後ろ | 8 | 22 | 36.364 | 28 | 25 | 3 |
| ankle | exclude_p8_near | final_8dir | 左斜め前 | 2 | 22 | 9.091 | 28 | 25 | 3 |
| ankle | exclude_p8_near | final_8dir | まっすぐ前 | 4 | 22 | 18.182 | 28 | 25 | 3 |
| ankle | exclude_p8_near | final_8dir | 右斜め前 | 5 | 22 | 22.727 | 28 | 25 | 3 |
| ankle | exclude_p8_near | final_8dir | 左横 | 0 | 22 | 0.0 | 28 | 25 | 3 |
| ankle | exclude_p8_near | final_8dir | 右横 | 0 | 22 | 0.0 | 28 | 25 | 3 |
| ankle | exclude_p8_near | final_8dir | 左斜め後ろ | 3 | 22 | 13.636 | 28 | 25 | 3 |
| ankle | exclude_p8_near | final_8dir | まっすぐ後ろ | 4 | 22 | 18.182 | 28 | 25 | 3 |
| ankle | exclude_p8_near | final_8dir | 右斜め後ろ | 4 | 22 | 18.182 | 28 | 25 | 3 |
| bbox | all | availability | 判定不能 | 4 | 32 | 12.5 | 32 | 28 | 4 |
| bbox | all | presence | あり | 21 | 28 | 75.0 | 32 | 28 | 4 |
| bbox | all | presence | なし | 7 | 28 | 25.0 | 32 | 28 | 4 |
| bbox | all | complexity | 単 | 18 | 21 | 85.714 | 32 | 28 | 4 |
| bbox | all | complexity | 複 | 3 | 21 | 14.286 | 32 | 28 | 4 |
| bbox | all | single_3dir | 前 | 6 | 18 | 33.333 | 32 | 28 | 4 |
| bbox | all | single_3dir | 横 | 4 | 18 | 22.222 | 32 | 28 | 4 |
| bbox | all | single_3dir | 後ろ | 8 | 18 | 44.444 | 32 | 28 | 4 |
| bbox | all | single_8dir | 左斜め前 | 0 | 18 | 0.0 | 32 | 28 | 4 |
| bbox | all | single_8dir | まっすぐ前 | 2 | 18 | 11.111 | 32 | 28 | 4 |
| bbox | all | single_8dir | 右斜め前 | 5 | 18 | 27.778 | 32 | 28 | 4 |
| bbox | all | single_8dir | 左横 | 0 | 18 | 0.0 | 32 | 28 | 4 |
| bbox | all | single_8dir | 右横 | 0 | 18 | 0.0 | 32 | 28 | 4 |
| bbox | all | single_8dir | 左斜め後ろ | 3 | 18 | 16.667 | 32 | 28 | 4 |
| bbox | all | single_8dir | まっすぐ後ろ | 2 | 18 | 11.111 | 32 | 28 | 4 |
| bbox | all | single_8dir | 右斜め後ろ | 6 | 18 | 33.333 | 32 | 28 | 4 |
| bbox | all | final_3dir | 前 | 6 | 21 | 28.571 | 32 | 28 | 4 |
| bbox | all | final_3dir | 横 | 7 | 21 | 33.333 | 32 | 28 | 4 |
| bbox | all | final_3dir | 後ろ | 8 | 21 | 38.095 | 32 | 28 | 4 |
| bbox | all | final_8dir | 左斜め前 | 1 | 21 | 4.762 | 32 | 28 | 4 |
| bbox | all | final_8dir | まっすぐ前 | 2 | 21 | 9.524 | 32 | 28 | 4 |
| bbox | all | final_8dir | 右斜め前 | 5 | 21 | 23.81 | 32 | 28 | 4 |
| bbox | all | final_8dir | 左横 | 0 | 21 | 0.0 | 32 | 28 | 4 |
| bbox | all | final_8dir | 右横 | 0 | 21 | 0.0 | 32 | 28 | 4 |
| bbox | all | final_8dir | 左斜め後ろ | 5 | 21 | 23.81 | 32 | 28 | 4 |
| bbox | all | final_8dir | まっすぐ後ろ | 2 | 21 | 9.524 | 32 | 28 | 4 |
| bbox | all | final_8dir | 右斜め後ろ | 6 | 21 | 28.571 | 32 | 28 | 4 |
| bbox | exclude_p8_near | availability | 判定不能 | 3 | 28 | 10.714 | 28 | 25 | 3 |
| bbox | exclude_p8_near | presence | あり | 20 | 25 | 80.0 | 28 | 25 | 3 |
| bbox | exclude_p8_near | presence | なし | 5 | 25 | 20.0 | 28 | 25 | 3 |
| bbox | exclude_p8_near | complexity | 単 | 17 | 20 | 85.0 | 28 | 25 | 3 |
| bbox | exclude_p8_near | complexity | 複 | 3 | 20 | 15.0 | 28 | 25 | 3 |
| bbox | exclude_p8_near | single_3dir | 前 | 5 | 17 | 29.412 | 28 | 25 | 3 |
| bbox | exclude_p8_near | single_3dir | 横 | 4 | 17 | 23.529 | 28 | 25 | 3 |
| bbox | exclude_p8_near | single_3dir | 後ろ | 8 | 17 | 47.059 | 28 | 25 | 3 |
| bbox | exclude_p8_near | single_8dir | 左斜め前 | 0 | 17 | 0.0 | 28 | 25 | 3 |
| bbox | exclude_p8_near | single_8dir | まっすぐ前 | 2 | 17 | 11.765 | 28 | 25 | 3 |
| bbox | exclude_p8_near | single_8dir | 右斜め前 | 4 | 17 | 23.529 | 28 | 25 | 3 |
| bbox | exclude_p8_near | single_8dir | 左横 | 0 | 17 | 0.0 | 28 | 25 | 3 |
| bbox | exclude_p8_near | single_8dir | 右横 | 0 | 17 | 0.0 | 28 | 25 | 3 |
| bbox | exclude_p8_near | single_8dir | 左斜め後ろ | 3 | 17 | 17.647 | 28 | 25 | 3 |
| bbox | exclude_p8_near | single_8dir | まっすぐ後ろ | 2 | 17 | 11.765 | 28 | 25 | 3 |
| bbox | exclude_p8_near | single_8dir | 右斜め後ろ | 6 | 17 | 35.294 | 28 | 25 | 3 |
| bbox | exclude_p8_near | final_3dir | 前 | 5 | 20 | 25.0 | 28 | 25 | 3 |
| bbox | exclude_p8_near | final_3dir | 横 | 7 | 20 | 35.0 | 28 | 25 | 3 |
| bbox | exclude_p8_near | final_3dir | 後ろ | 8 | 20 | 40.0 | 28 | 25 | 3 |
| bbox | exclude_p8_near | final_8dir | 左斜め前 | 1 | 20 | 5.0 | 28 | 25 | 3 |
| bbox | exclude_p8_near | final_8dir | まっすぐ前 | 2 | 20 | 10.0 | 28 | 25 | 3 |
| bbox | exclude_p8_near | final_8dir | 右斜め前 | 4 | 20 | 20.0 | 28 | 25 | 3 |
| bbox | exclude_p8_near | final_8dir | 左横 | 0 | 20 | 0.0 | 28 | 25 | 3 |
| bbox | exclude_p8_near | final_8dir | 右横 | 0 | 20 | 0.0 | 28 | 25 | 3 |
| bbox | exclude_p8_near | final_8dir | 左斜め後ろ | 5 | 20 | 25.0 | 28 | 25 | 3 |
| bbox | exclude_p8_near | final_8dir | まっすぐ後ろ | 2 | 20 | 10.0 | 28 | 25 | 3 |
| bbox | exclude_p8_near | final_8dir | 右斜め後ろ | 6 | 20 | 30.0 | 28 | 25 | 3 |

## 井箟ほかとの比較

| coordinate_method | metric | category | category_ja | paper_percent | all_n | all_denominator | all_percent | exclude_p8_near_n | exclude_p8_near_denominator | exclude_p8_near_percent |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bbox | presence | motion | あり | 93.1 | 21 | 28 | 75.0 | 20 | 25 | 80.0 |
| bbox | presence | none | なし | 6.9 | 7 | 28 | 25.0 | 5 | 25 | 20.0 |
| bbox | complexity | single | 単 | 72.8 | 18 | 21 | 85.714 | 17 | 20 | 85.0 |
| bbox | complexity | compound | 複 | 27.2 | 3 | 21 | 14.286 | 3 | 20 | 15.0 |
| bbox | single_3dir | front | 前 | 60.3 | 6 | 18 | 33.333 | 5 | 17 | 29.412 |
| bbox | single_3dir | lateral | 横 | 21.1 | 4 | 18 | 22.222 | 4 | 17 | 23.529 |
| bbox | single_3dir | back | 後ろ | 18.6 | 8 | 18 | 44.444 | 8 | 17 | 47.059 |
| bbox | single_8dir | left_front | 左斜め前 | 31.6 | 0 | 18 | 0.0 | 0 | 17 | 0.0 |
| bbox | single_8dir | straight_front | まっすぐ前 | 2.7 | 2 | 18 | 11.111 | 2 | 17 | 11.765 |
| bbox | single_8dir | right_front | 右斜め前 | 27.1 | 5 | 18 | 27.778 | 4 | 17 | 23.529 |
| bbox | single_8dir | left_lateral | 左横 | 9.4 | 0 | 18 | 0.0 | 0 | 17 | 0.0 |
| bbox | single_8dir | right_lateral | 右横 | 12.1 | 0 | 18 | 0.0 | 0 | 17 | 0.0 |
| bbox | single_8dir | left_back | 左斜め後ろ | 6.1 | 3 | 18 | 16.667 | 3 | 17 | 17.647 |
| bbox | single_8dir | straight_back | まっすぐ後ろ | 1.8 | 2 | 18 | 11.111 | 2 | 17 | 11.765 |
| bbox | single_8dir | right_back | 右斜め後ろ | 11.1 | 6 | 18 | 33.333 | 6 | 17 | 35.294 |
| bbox | final_3dir | front | 前 | 58.9 | 6 | 21 | 28.571 | 5 | 20 | 25.0 |
| bbox | final_3dir | lateral | 横 | 21.3 | 7 | 21 | 33.333 | 7 | 20 | 35.0 |
| bbox | final_3dir | back | 後ろ | 19.8 | 8 | 21 | 38.095 | 8 | 20 | 40.0 |
| ankle | presence | motion | あり | 93.1 | 23 | 28 | 82.143 | 22 | 25 | 88.0 |
| ankle | presence | none | なし | 6.9 | 5 | 28 | 17.857 | 3 | 25 | 12.0 |
| ankle | complexity | single | 単 | 72.8 | 16 | 23 | 69.565 | 15 | 22 | 68.182 |
| ankle | complexity | compound | 複 | 27.2 | 7 | 23 | 30.435 | 7 | 22 | 31.818 |
| ankle | single_3dir | front | 前 | 60.3 | 7 | 16 | 43.75 | 6 | 15 | 40.0 |
| ankle | single_3dir | lateral | 横 | 21.1 | 2 | 16 | 12.5 | 2 | 15 | 13.333 |
| ankle | single_3dir | back | 後ろ | 18.6 | 7 | 16 | 43.75 | 7 | 15 | 46.667 |
| ankle | single_8dir | left_front | 左斜め前 | 31.6 | 1 | 16 | 6.25 | 1 | 15 | 6.667 |
| ankle | single_8dir | straight_front | まっすぐ前 | 2.7 | 3 | 16 | 18.75 | 2 | 15 | 13.333 |
| ankle | single_8dir | right_front | 右斜め前 | 27.1 | 3 | 16 | 18.75 | 3 | 15 | 20.0 |
| ankle | single_8dir | left_lateral | 左横 | 9.4 | 0 | 16 | 0.0 | 0 | 15 | 0.0 |
| ankle | single_8dir | right_lateral | 右横 | 12.1 | 0 | 16 | 0.0 | 0 | 15 | 0.0 |
| ankle | single_8dir | left_back | 左斜め後ろ | 6.1 | 2 | 16 | 12.5 | 2 | 15 | 13.333 |
| ankle | single_8dir | straight_back | まっすぐ後ろ | 1.8 | 3 | 16 | 18.75 | 3 | 15 | 20.0 |
| ankle | single_8dir | right_back | 右斜め後ろ | 11.1 | 4 | 16 | 25.0 | 4 | 15 | 26.667 |
| ankle | final_3dir | front | 前 | 58.9 | 10 | 23 | 43.478 | 9 | 22 | 40.909 |
| ankle | final_3dir | lateral | 横 | 21.3 | 5 | 23 | 21.739 | 5 | 22 | 22.727 |
| ankle | final_3dir | back | 後ろ | 19.8 | 8 | 23 | 34.783 | 8 | 22 | 36.364 |

比較値は既存v2.PAPERから継承。論文原文は今回再照合していない。旧READMEは値の出典を依頼文中の要約としており、特に8方向の論文値は合計101.9%で整合しない。
原表の数値・分母の照合前に論文値を確定した引用として使わない。値を勝手に正規化・修正せず、全条件比較CSVにもそのまま保存。
打球後の短時間区間、代理座標、標本、方向の角度定義は論文の観察手続と同一ではない。意図的移動・フェイントをこの軌跡だけから確認したとはいえない。
対象は同一試合5点の32イベントで、除外・判定不能後の分母はさらに小さい。構成比の差から戦術や一般化性能を主張しない。

出力: motions.csv（全576行）、motions_main.csv、summary.csv、summary_main.csv、paper_comparison.csv（主条件）、
paper_comparison_all_conditions.csv、role_selection.csv、pose_cache_ranges.csv、projection_validation.csv、trajectories_main.csv、input_manifest.json。

実行: `study\.venv\Scripts\python.exe -B -X utf8 outputs/shot_pipeline_20261117/v_human/net_motion/classify_net_motion_on_human.py`
