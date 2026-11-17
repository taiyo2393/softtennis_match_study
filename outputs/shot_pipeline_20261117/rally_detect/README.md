# match2 第7〜11点：音と場面によるラリー区間推定

**結論：今回の単純な音・場面規則では、サーブと最終打球の区間を十分正確に自動決定できなかった。** Trueのみ・Aを含む条件では、固定テストのラリー重なりは 3/3 だが開始MAE 282.7F、終了MAE 234.7F。固定テストではIoU≥0.5も両境界±10Fも 0 / 3。1点抜き全5点では候補259件（51.8件/点）を確認し、既存ラベルの不足14打球を追加する見積もり。全True場面のv1 331件からは21.8%減るが、ラベル前後だけで抽出していた前回190件より確認数が増える。区間自動化の成功や人の時間短縮は主張できない。

本実験は、既存の暫定打球ラベルに対し、候補を確認する件数と取りこぼしを測る予備実験。人の確認時間・修正時間は実測していない。区間の終わりは最後の打球であり、ボールが落ちて得点が確定する時刻ではない。

## 再実行

リポジトリルートのPowerShellで実行：

```powershell
& study\.venv\Scripts\python.exe -B -X utf8 -u outputs\shot_pipeline_20261117\rally_detect\detect_rallies.py
```

今回の実行時間は 11.8 秒（図の生成・評価を含む）。追加パッケージなし。WAVはリポジトリ外の一時ディレクトリで生成・削除し、残さない。書込先はこのフォルダだけ。

## 入力と推定の分離

点区間は point_inventory.csv の [display_from_frame, next_display_from_frame)。場面は scenes/scenes.csv を直接読み、effective_scenes.csv・poseは使用しない。音声は各点全体に前後1.5秒を付けて48 kHz monoで抽出する。v1の固定設定（4–12 kHz、2 msエネルギー/1 ms刻み、0.5秒局所中央値との差、しきい値1、最小間隔0.15秒）で全区間からピークを出す。ラベルから抽出範囲を決めない。フィルタ・ピーク探索は点全体で行い、整数候補Fの場面所属でTrue/uncertainを分ける。場面の境界を越えて塊を結合しない。Falseの音も図の背景用に計測するが、ラリー推定には渡さない。

特徴は既存v2の add_features をそのまま呼ぶ（半値幅、低/高域の上昇、両帯域上昇、低域ピーク遅延、50 ms後の残存/低下、20/100 ms集中度、スペクトル平坦度）。新しい全点候補を candidate_features.csv に保存。候補時刻は変更しない。

図の音の値は1F内のv1スコアの最大値（鋭いピークを保持）。推定自体は元の1 ms刻みで行い、図用の集約値を使わない。

ラリー推定関数の入力は候補と規則のみ。ラベルは規則の訓練選択と評価だけに渡す。ラベル内の各rallyの最初/最後のcontact_frameを正解区間とし、両端を含む。第7点Aを除外する条件ではAを検出対象外とし、そこだけに出た検出は余分と数える。Aの位置で推定を削る等の正解利用はしない。Aを含む/除く条件は別々に訓練選択する。

## 事前に固定した探索と選択基準

G=0.8,1.0,1.2,1.4,1.6,1.8,2.0秒、K=3,4,5,6。G秒以下で隣接候補を連結し、K候補以上を採る。形状規則は none / sharp / v2 の3種類。sharpは半値幅≤15 msかつ50 ms後の対数エネルギー低下≥1、v2はさらに両帯域上昇≥0.5。形状除去は連結の前に行い、拍手等でつながる塊を切る機会を与える。

サーブ前除去は none / rhythmic_jump1 / rhythmic_jump2。先頭の3〜8候補が0.4〜1.1秒の間隔、間隔の変動係数≤0.3、先頭から新開始候補まで≤4秒を満たし、新開始候補の高域上昇が先頭部分の中央値より1または2以上、全先頭候補より大きく、両帯域上昇も先頭中央値より0.5以上大きい最初の候補までを除去する。除去後にもK件を要求する。これは『弱い等間隔の先頭音』という仮説であり、ボールつきの識別正解があるわけではない。塊の最後の採用候補を終了とする。

cluster_v1（G,Kのみ、28通り）、shape_only（84通り）、shape_and_serve（252通り）をそれぞれ訓練だけで選択し、比較する。主結果はshape_and_serve（不要な規則なしも選択可）。選択指標は候補F1とラリーIoUを用いる次式を等重みで最大化：

`0.5 × 候補F1 + 0.5 × 2Σ(対応ラリーのIoU)/(正解ラリー数+検出ラリー数)`

同点はclear一致数が多い、開始/終了MAE和が小さい、候補が少ない、規則が単純、ID順。この基準は本スクリプトに固定し、テストの出力を見て修正しない。高再現率の保証や時間最小化の基準ではない。

今回、探索した先頭除去規則は全条件で除去を発動せず、主規則はすべてserve_trim=noneが選ばれた。shape_onlyとshape_and_serveの結果は同じ。したがって候補削減は音形状規則等の効果で、サーブ前ボールつきの除去ができた結果ではない。この保守的な等間隔規則が機能しなかったことを示し、強さの跳びだけを使う等の別規則まで否定する結果ではない。テスト観察後の規則追加・再選択は行っていない。

固定分割は第7・8点で選択し、第9〜11点を評価。1点抜きは4点で選択して残り1点を評価し、5回のテスト予測を合算する。率は点平均ではなく件数合算。比較用に1点抜きの第9〜11点だけも併記。Trueのみ/True+uncertainの各条件で同じ探索を独立実施する。

## 評価定義

ラリーは同一点内で閉区間が1F以上重なれば対応候補。最大対応数、次にIoU和最大の1対1割当。単なる重なり再現率も別列で保存し、分裂・結合・IoU≥0.5・両境界±10F以内も記録する。開始/終了のずれは検出−正解（正=遅れ）。MAEは対応したラリーのみで、見逃しは分母に入らない。未対応検出は、正解と重なる断片(extra_fragment)と、正解ラリーとの重なりなし(extra_no_labeled_rally)に分け、場所と場面IDを保存する。後者がリプレイ・拍手かは自動断定しない。

候補は検出区間を形成した、形状/先頭規則を通過した候補。打球は同一点・±10F以内で最大対応数、次に絶対ずれ和最小の1対1割当。clear再現率も全ラベル共通割当に基づく。適合率目安=一致数/候補数、F1=2一致数/(候補数+全ラベル数)。候補/秒の分母は検出ラリーの閉区間長の和、候補/打球は全対象ラベル数。raw比較とoracle比較ではそれぞれ場面全長/正解ラリー長が秒の分母になるため、率の直接比較に注意。

作業量は1候補=1確認と仮定。追加打球数=全ラベル−1対1一致数。確認+追加は件数の和で、取りこぼしを探すための全動画確認時間や区間修正は含まない。したがって候補件数の削減率を人の時間削減率と呼ばない。

## 主結果：固定分割と1点抜き

| 条件 | 評価範囲 | 対応ラリー | 正解ラリー | 余分 | 開始MAE(F) | 終了MAE(F) | 確認候補 | clear再現率 | 全再現率 | 適合率目安 | F1 | 候補/秒 | 候補/打球 | 候補/点 | 追加打球 | 全場面候補から削減率 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| true_only, A=除外 | fixed_test_09_11 | 3 | 3 | 0 | 282.667 | 234.667 | 118 | 0.944 | 0.906 | 0.246 | 0.387 | 3.322 | 3.688 | 39.333 | 3 | 0.203 |
| true_only, A=除外 | loo_all_07_11 | 5 | 5 | 3 | 234.000 | 163.200 | 259 | 0.907 | 0.849 | 0.282 | 0.423 | 3.359 | 3.012 | 51.800 | 13 | 0.218 |
| true_only, A=除外 | loo_common_09_11 | 3 | 3 | 0 | 282.667 | 234.667 | 118 | 0.944 | 0.906 | 0.246 | 0.387 | 3.322 | 3.688 | 39.333 | 3 | 0.203 |
| true_only, A=含む | fixed_test_09_11 | 3 | 3 | 0 | 282.667 | 234.667 | 118 | 0.944 | 0.906 | 0.246 | 0.387 | 3.322 | 3.688 | 39.333 | 3 | 0.203 |
| true_only, A=含む | loo_all_07_11 | 6 | 6 | 2 | 215.667 | 138.333 | 259 | 0.914 | 0.846 | 0.297 | 0.440 | 3.359 | 2.846 | 51.800 | 14 | 0.218 |
| true_only, A=含む | loo_common_09_11 | 3 | 3 | 0 | 282.667 | 234.667 | 118 | 0.944 | 0.906 | 0.246 | 0.387 | 3.322 | 3.688 | 39.333 | 3 | 0.203 |
| true_plus_uncertain, A=除外 | fixed_test_09_11 | 3 | 3 | 8 | 282.667 | 234.667 | 203 | 0.944 | 0.906 | 0.143 | 0.247 | 3.362 | 6.344 | 67.667 | 3 | 0.372 |
| true_plus_uncertain, A=除外 | loo_all_07_11 | 5 | 5 | 13 | 344.200 | 163.200 | 389 | 0.907 | 0.849 | 0.188 | 0.307 | 3.347 | 4.523 | 77.800 | 13 | 0.358 |
| true_plus_uncertain, A=除外 | loo_common_09_11 | 3 | 3 | 8 | 282.667 | 234.667 | 204 | 0.944 | 0.906 | 0.142 | 0.246 | 3.328 | 6.375 | 68.000 | 3 | 0.368 |
| true_plus_uncertain, A=含む | fixed_test_09_11 | 3 | 3 | 8 | 282.667 | 234.667 | 203 | 0.944 | 0.906 | 0.143 | 0.247 | 3.362 | 6.344 | 67.667 | 3 | 0.372 |
| true_plus_uncertain, A=含む | loo_all_07_11 | 5 | 6 | 13 | 344.200 | 163.200 | 388 | 0.914 | 0.846 | 0.198 | 0.322 | 3.365 | 4.264 | 77.600 | 14 | 0.360 |
| true_plus_uncertain, A=含む | loo_common_09_11 | 3 | 3 | 8 | 282.667 | 234.667 | 203 | 0.944 | 0.906 | 0.143 | 0.247 | 3.362 | 6.344 | 67.667 | 3 | 0.372 |

## 点別の確認数と追加数（主規則）

| 場面 | Aを含む | 分割 | train/test | 点 | 規則 | 確認候補 | 追加打球 | 追加clear | F1 | 余分ラリー |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| true_only | True | fixed | train | 7 | r006 | 100 | 5 | 2 | 0.385 | 2 |
| true_only | True | fixed | train | 8 | r006 | 41 | 6 | 2 | 0.657 | 0 |
| true_only | True | fixed | test | 9 | r006 | 70 | 2 | 1 | 0.222 | 0 |
| true_only | True | fixed | test | 10 | r006 | 18 | 1 | 0 | 0.480 | 0 |
| true_only | True | fixed | test | 11 | r006 | 30 | 0 | 0 | 0.636 | 0 |
| true_only | True | loo_07 | test | 7 | r006 | 100 | 5 | 2 | 0.385 | 2 |
| true_only | True | loo_08 | test | 8 | r006 | 41 | 6 | 2 | 0.657 | 0 |
| true_only | True | loo_09 | test | 9 | r006 | 70 | 2 | 1 | 0.222 | 0 |
| true_only | True | loo_10 | test | 10 | r006 | 18 | 1 | 0 | 0.480 | 0 |
| true_only | True | loo_11 | test | 11 | r006 | 30 | 0 | 0 | 0.636 | 0 |
| true_only | False | fixed | train | 7 | r042 | 100 | 4 | 2 | 0.336 | 1 |
| true_only | False | fixed | train | 8 | r042 | 41 | 6 | 2 | 0.657 | 0 |
| true_only | False | fixed | test | 9 | r042 | 70 | 2 | 1 | 0.222 | 0 |
| true_only | False | fixed | test | 10 | r042 | 18 | 1 | 0 | 0.480 | 0 |
| true_only | False | fixed | test | 11 | r042 | 30 | 0 | 0 | 0.636 | 0 |
| true_only | False | loo_07 | test | 7 | r006 | 100 | 4 | 2 | 0.336 | 3 |
| true_only | False | loo_08 | test | 8 | r042 | 41 | 6 | 2 | 0.657 | 0 |
| true_only | False | loo_09 | test | 9 | r042 | 70 | 2 | 1 | 0.222 | 0 |
| true_only | False | loo_10 | test | 10 | r042 | 18 | 1 | 0 | 0.480 | 0 |
| true_only | False | loo_11 | test | 11 | r042 | 30 | 0 | 0 | 0.636 | 0 |
| true_plus_uncertain | True | fixed | train | 7 | r033 | 136 | 5 | 2 | 0.301 | 5 |
| true_plus_uncertain | True | fixed | train | 8 | r033 | 41 | 6 | 2 | 0.657 | 0 |
| true_plus_uncertain | True | fixed | test | 9 | r033 | 96 | 2 | 1 | 0.168 | 2 |
| true_plus_uncertain | True | fixed | test | 10 | r033 | 77 | 1 | 0 | 0.143 | 6 |
| true_plus_uncertain | True | fixed | test | 11 | r033 | 30 | 0 | 0 | 0.636 | 0 |
| true_plus_uncertain | True | loo_07 | test | 7 | r069 | 144 | 5 | 2 | 0.287 | 5 |
| true_plus_uncertain | True | loo_08 | test | 8 | r033 | 41 | 6 | 2 | 0.657 | 0 |
| true_plus_uncertain | True | loo_09 | test | 9 | r033 | 96 | 2 | 1 | 0.168 | 2 |
| true_plus_uncertain | True | loo_10 | test | 10 | r033 | 77 | 1 | 0 | 0.143 | 6 |
| true_plus_uncertain | True | loo_11 | test | 11 | r033 | 30 | 0 | 0 | 0.636 | 0 |
| true_plus_uncertain | False | fixed | train | 7 | r033 | 136 | 4 | 2 | 0.261 | 6 |
| true_plus_uncertain | False | fixed | train | 8 | r033 | 41 | 6 | 2 | 0.657 | 0 |
| true_plus_uncertain | False | fixed | test | 9 | r033 | 96 | 2 | 1 | 0.168 | 2 |
| true_plus_uncertain | False | fixed | test | 10 | r033 | 77 | 1 | 0 | 0.143 | 6 |
| true_plus_uncertain | False | fixed | test | 11 | r033 | 30 | 0 | 0 | 0.636 | 0 |
| true_plus_uncertain | False | loo_07 | test | 7 | r069 | 144 | 4 | 2 | 0.249 | 5 |
| true_plus_uncertain | False | loo_08 | test | 8 | r033 | 41 | 6 | 2 | 0.657 | 0 |
| true_plus_uncertain | False | loo_09 | test | 9 | r033 | 96 | 2 | 1 | 0.168 | 2 |
| true_plus_uncertain | False | loo_10 | test | 10 | r105 | 78 | 1 | 0 | 0.141 | 6 |
| true_plus_uncertain | False | loo_11 | test | 11 | r033 | 30 | 0 | 0 | 0.636 | 0 |

## 規則追加の比較（Trueのみ・Aを含む）

| 規則群 | 範囲 | 候補 | 追加 | clear再現 | F1 | 余分ラリー | 開始MAE | 終了MAE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cluster_v1 | fixed_test_09_11 | 148 | 0 | 1.000 | 0.356 | 0 | 320.667 | 234.667 |
| cluster_v1 | loo_all_07_11 | 331 | 7 | 0.966 | 0.398 | 1 | 367.000 | 166.600 |
| shape_and_serve | fixed_test_09_11 | 118 | 3 | 0.944 | 0.387 | 0 | 282.667 | 234.667 |
| shape_and_serve | loo_all_07_11 | 259 | 14 | 0.914 | 0.440 | 2 | 215.667 | 138.333 |
| shape_only | fixed_test_09_11 | 118 | 3 | 0.944 | 0.387 | 0 | 282.667 | 234.667 |
| shape_only | loo_all_07_11 | 259 | 14 | 0.914 | 0.440 | 2 | 215.667 | 138.333 |

## 前回値とラリー正解を使う上限参照

依頼時の参考値は旧ラベル前後±90F内のv1=190件、F1約0.59〜0.62、旧候補を『最初の打球−15F〜最後の打球+15F』に限ると136件・F1約0.73。この±15Fは audio_learned/README.md の前回追記に記載されていた。今回は正解区間を最初〜最後の打球そのものと指定されているので、oracle_gt_rallyは追加幅0F、oracle_gt_rally_plusminus15が前回を再現する条件。追加幅0Fだと旧候補は120件となり、136件と食い違う原因は区間定義である。前回の104件はラベル±10F内の候補の多対1集計であり、1対1一致数ではない。以下は既存190件と新規の全場面候補を、今回と同じ±10F・最大数1対1で再評価したもの。旧方式の最近傍greedy割当と一致数が異なる可能性がある。実際に136件の条件は旧greedyで83一致・F1=0.731、今回の最大対応数割当で84一致・F1=0.740となる。oracleはいずれも正解区間を利用する参考値で、自動ラリー検出の成績ではない。A除外oracleはAの候補も正解を使って除くが、自動方式のA除外はそこを負例として数える。

| 入力 | Aを含む | 区間 | 候補 | clear再現 | 全再現 | 適合率目安 | F1 | 旧greedy F1 | 候補/打球 | 候補/点 | 追加 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| historical | True | all_input_candidates | 190 | 0.966 | 0.923 | 0.442 | 0.598 | 0.591 | 2.088 | 38.000 | 7 |
| historical | True | oracle_gt_rally | 120 | 0.931 | 0.868 | 0.658 | 0.749 | 0.739 | 1.319 | 24.000 | 12 |
| historical | True | oracle_gt_rally_plusminus15 | 136 | 0.966 | 0.923 | 0.618 | 0.740 | 0.731 | 1.495 | 27.200 | 7 |
| true_only | True | all_input_candidates | 331 | 0.966 | 0.923 | 0.254 | 0.398 | — | 3.637 | 66.200 | 7 |
| true_only | True | oracle_gt_rally | 119 | 0.931 | 0.868 | 0.664 | 0.752 | — | 1.308 | 23.800 | 12 |
| true_only | True | oracle_gt_rally_plusminus15 | 134 | 0.966 | 0.923 | 0.627 | 0.747 | — | 1.473 | 26.800 | 7 |
| true_plus_uncertain | True | all_input_candidates | 606 | 0.966 | 0.923 | 0.139 | 0.241 | — | 6.659 | 121.200 | 7 |
| true_plus_uncertain | True | oracle_gt_rally | 119 | 0.931 | 0.868 | 0.664 | 0.752 | — | 1.308 | 23.800 | 12 |
| true_plus_uncertain | True | oracle_gt_rally_plusminus15 | 134 | 0.966 | 0.923 | 0.627 | 0.747 | — | 1.473 | 26.800 | 7 |
| historical | False | all_input_candidates | 190 | 0.963 | 0.919 | 0.416 | 0.572 | 0.565 | 2.209 | 38.000 | 7 |
| historical | False | oracle_gt_rally | 115 | 0.944 | 0.884 | 0.661 | 0.756 | 0.746 | 1.337 | 23.000 | 10 |
| historical | False | oracle_gt_rally_plusminus15 | 127 | 0.963 | 0.919 | 0.622 | 0.742 | 0.732 | 1.477 | 25.400 | 7 |
| true_only | False | all_input_candidates | 331 | 0.963 | 0.919 | 0.239 | 0.379 | — | 3.849 | 66.200 | 7 |
| true_only | False | oracle_gt_rally | 114 | 0.944 | 0.884 | 0.667 | 0.760 | — | 1.326 | 22.800 | 10 |
| true_only | False | oracle_gt_rally_plusminus15 | 125 | 0.963 | 0.919 | 0.632 | 0.749 | — | 1.453 | 25.000 | 7 |
| true_plus_uncertain | False | all_input_candidates | 606 | 0.963 | 0.919 | 0.130 | 0.228 | — | 7.047 | 121.200 | 7 |
| true_plus_uncertain | False | oracle_gt_rally | 114 | 0.944 | 0.884 | 0.667 | 0.760 | — | 1.326 | 22.800 | 10 |
| true_plus_uncertain | False | oracle_gt_rally_plusminus15 | 125 | 0.963 | 0.919 | 0.632 | 0.749 | — | 1.453 | 25.000 | 7 |

## 選択された主規則

| 場面 | Aを含む | 分割 | G秒 | K | 形状 | 先頭除去 |
| --- | --- | --- | --- | --- | --- | --- |
| true_only | True | fixed | 0.800 | 3 | v2 | none |
| true_only | True | loo_07 | 0.800 | 3 | v2 | none |
| true_only | True | loo_08 | 0.800 | 3 | v2 | none |
| true_only | True | loo_09 | 0.800 | 3 | v2 | none |
| true_only | True | loo_10 | 0.800 | 3 | v2 | none |
| true_only | True | loo_11 | 0.800 | 3 | v2 | none |
| true_only | False | fixed | 1.000 | 3 | v2 | none |
| true_only | False | loo_07 | 0.800 | 3 | v2 | none |
| true_only | False | loo_08 | 1.000 | 3 | v2 | none |
| true_only | False | loo_09 | 1.000 | 3 | v2 | none |
| true_only | False | loo_10 | 1.000 | 3 | v2 | none |
| true_only | False | loo_11 | 1.000 | 3 | v2 | none |
| true_plus_uncertain | True | fixed | 0.800 | 6 | v2 | none |
| true_plus_uncertain | True | loo_07 | 1.000 | 6 | v2 | none |
| true_plus_uncertain | True | loo_08 | 0.800 | 6 | v2 | none |
| true_plus_uncertain | True | loo_09 | 0.800 | 6 | v2 | none |
| true_plus_uncertain | True | loo_10 | 0.800 | 6 | v2 | none |
| true_plus_uncertain | True | loo_11 | 0.800 | 6 | v2 | none |
| true_plus_uncertain | False | fixed | 0.800 | 6 | v2 | none |
| true_plus_uncertain | False | loo_07 | 1.000 | 6 | v2 | none |
| true_plus_uncertain | False | loo_08 | 0.800 | 6 | v2 | none |
| true_plus_uncertain | False | loo_09 | 0.800 | 6 | v2 | none |
| true_plus_uncertain | False | loo_10 | 1.200 | 6 | v2 | none |
| true_plus_uncertain | False | loo_11 | 0.800 | 6 | v2 | none |

## 主規則のラリー対応と余分検出の場所

| 場面 | Aを含む | 分割 | train/test | 点 | 対応 | 正解 | 検出場面 | 検出開始 | 検出終了 | 開始ずれF | 終了ずれF | IoU |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| true_only | True | fixed | train | 7 | matched | A | p07_s012 | 22742.000 | 22928.000 | -124.000 | -14.000 | 0.313 |
| true_only | True | fixed | train | 7 | matched | B | p07_s012 | 23293.000 | 23958.000 | -221.000 | -26.000 | 0.643 |
| true_only | True | fixed | train | 7 | extra_no_labeled_rally |  | p07_s001 | 21631.000 | 22242.000 |  |  |  |
| true_only | True | fixed | train | 7 | extra_no_labeled_rally |  | p07_s012 | 22977.000 | 23244.000 |  |  |  |
| true_only | True | fixed | train | 8 | matched | B | p08_s001 | 25151.000 | 25910.000 | -101.000 | 86.000 | 0.754 |
| true_only | True | fixed | test | 9 | matched | B | p09_s003 | 27407.000 | 28683.000 | -573.000 | 521.000 | 0.143 |
| true_only | True | fixed | test | 10 | matched | B | p10_s009 | 30457.000 | 30767.000 | -30.000 | 142.000 | 0.447 |
| true_only | True | fixed | test | 11 | matched | B | p11_s001 | 32563.000 | 33103.000 | -245.000 | 41.000 | 0.471 |
| true_only | True | loo_07 | test | 7 | matched | A | p07_s012 | 22742.000 | 22928.000 | -124.000 | -14.000 | 0.313 |
| true_only | True | loo_07 | test | 7 | matched | B | p07_s012 | 23293.000 | 23958.000 | -221.000 | -26.000 | 0.643 |
| true_only | True | loo_07 | test | 7 | extra_no_labeled_rally |  | p07_s001 | 21631.000 | 22242.000 |  |  |  |
| true_only | True | loo_07 | test | 7 | extra_no_labeled_rally |  | p07_s012 | 22977.000 | 23244.000 |  |  |  |
| true_only | True | loo_08 | test | 8 | matched | B | p08_s001 | 25151.000 | 25910.000 | -101.000 | 86.000 | 0.754 |
| true_only | True | loo_09 | test | 9 | matched | B | p09_s003 | 27407.000 | 28683.000 | -573.000 | 521.000 | 0.143 |
| true_only | True | loo_10 | test | 10 | matched | B | p10_s009 | 30457.000 | 30767.000 | -30.000 | 142.000 | 0.447 |
| true_only | True | loo_11 | test | 11 | matched | B | p11_s001 | 32563.000 | 33103.000 | -245.000 | 41.000 | 0.471 |
| true_only | False | fixed | train | 7 | matched | B | p07_s012 | 22742.000 | 23958.000 | -772.000 | -26.000 | 0.358 |
| true_only | False | fixed | train | 7 | extra_no_labeled_rally |  | p07_s001 | 21631.000 | 22242.000 |  |  |  |
| true_only | False | fixed | train | 8 | matched | B | p08_s001 | 25151.000 | 25910.000 | -101.000 | 86.000 | 0.754 |
| true_only | False | fixed | test | 9 | matched | B | p09_s003 | 27407.000 | 28683.000 | -573.000 | 521.000 | 0.143 |
| true_only | False | fixed | test | 10 | matched | B | p10_s009 | 30457.000 | 30767.000 | -30.000 | 142.000 | 0.447 |
| true_only | False | fixed | test | 11 | matched | B | p11_s001 | 32563.000 | 33103.000 | -245.000 | 41.000 | 0.471 |
| true_only | False | loo_07 | test | 7 | matched | B | p07_s012 | 23293.000 | 23958.000 | -221.000 | -26.000 | 0.643 |
| true_only | False | loo_07 | test | 7 | extra_no_labeled_rally |  | p07_s001 | 21631.000 | 22242.000 |  |  |  |
| true_only | False | loo_07 | test | 7 | extra_no_labeled_rally |  | p07_s012 | 22742.000 | 22928.000 |  |  |  |
| true_only | False | loo_07 | test | 7 | extra_no_labeled_rally |  | p07_s012 | 22977.000 | 23244.000 |  |  |  |
| true_only | False | loo_08 | test | 8 | matched | B | p08_s001 | 25151.000 | 25910.000 | -101.000 | 86.000 | 0.754 |
| true_only | False | loo_09 | test | 9 | matched | B | p09_s003 | 27407.000 | 28683.000 | -573.000 | 521.000 | 0.143 |
| true_only | False | loo_10 | test | 10 | matched | B | p10_s009 | 30457.000 | 30767.000 | -30.000 | 142.000 | 0.447 |
| true_only | False | loo_11 | test | 11 | matched | B | p11_s001 | 32563.000 | 33103.000 | -245.000 | 41.000 | 0.471 |
| true_plus_uncertain | True | fixed | train | 7 | matched | A | p07_s012 | 22742.000 | 22928.000 | -124.000 | -14.000 | 0.313 |
| true_plus_uncertain | True | fixed | train | 7 | matched | B | p07_s012 | 23293.000 | 23958.000 | -221.000 | -26.000 | 0.643 |
| true_plus_uncertain | True | fixed | train | 7 | extra_no_labeled_rally |  | p07_s001 | 21631.000 | 22242.000 |  |  |  |
| true_plus_uncertain | True | fixed | train | 7 | extra_no_labeled_rally |  | p07_s012 | 22977.000 | 23244.000 |  |  |  |
| true_plus_uncertain | True | fixed | train | 7 | extra_no_labeled_rally |  | p07_s020 | 24429.000 | 24584.000 |  |  |  |
| true_plus_uncertain | True | fixed | train | 7 | extra_no_labeled_rally |  | p07_s022 | 24649.000 | 24841.000 |  |  |  |
| true_plus_uncertain | True | fixed | train | 7 | extra_no_labeled_rally |  | p07_s022 | 24908.000 | 25077.000 |  |  |  |
| true_plus_uncertain | True | fixed | train | 8 | matched | B | p08_s001 | 25151.000 | 25910.000 | -101.000 | 86.000 | 0.754 |
| true_plus_uncertain | True | fixed | test | 9 | matched | B | p09_s003 | 27407.000 | 28683.000 | -573.000 | 521.000 | 0.143 |
| true_plus_uncertain | True | fixed | test | 9 | extra_no_labeled_rally |  | p09_s001 | 26687.000 | 26962.000 |  |  |  |
| true_plus_uncertain | True | fixed | test | 9 | extra_no_labeled_rally |  | p09_s001 | 27093.000 | 27245.000 |  |  |  |
| true_plus_uncertain | True | fixed | test | 10 | matched | B | p10_s009 | 30457.000 | 30767.000 | -30.000 | 142.000 | 0.447 |
| true_plus_uncertain | True | fixed | test | 10 | extra_no_labeled_rally |  | p10_s001 | 29068.000 | 29220.000 |  |  |  |
| true_plus_uncertain | True | fixed | test | 10 | extra_no_labeled_rally |  | p10_s003 | 29276.000 | 29534.000 |  |  |  |
| true_plus_uncertain | True | fixed | test | 10 | extra_no_labeled_rally |  | p10_s004 | 29545.000 | 29625.000 |  |  |  |
| true_plus_uncertain | True | fixed | test | 10 | extra_no_labeled_rally |  | p10_s005 | 29870.000 | 30029.000 |  |  |  |
| true_plus_uncertain | True | fixed | test | 10 | extra_no_labeled_rally |  | p10_s007 | 30114.000 | 30344.000 |  |  |  |
| true_plus_uncertain | True | fixed | test | 10 | extra_no_labeled_rally |  | p10_s014 | 31137.000 | 31313.000 |  |  |  |
| true_plus_uncertain | True | fixed | test | 11 | matched | B | p11_s001 | 32563.000 | 33103.000 | -245.000 | 41.000 | 0.471 |
| true_plus_uncertain | True | loo_07 | test | 7 | missed_gt | A |  |  |  |  |  |  |
| true_plus_uncertain | True | loo_07 | test | 7 | matched | B | p07_s012 | 22742.000 | 23958.000 | -772.000 | -26.000 | 0.358 |
| true_plus_uncertain | True | loo_07 | test | 7 | extra_no_labeled_rally |  | p07_s001 | 21631.000 | 22242.000 |  |  |  |
| true_plus_uncertain | True | loo_07 | test | 7 | extra_no_labeled_rally |  | p07_s003 | 22297.000 | 22481.000 |  |  |  |
| true_plus_uncertain | True | loo_07 | test | 7 | extra_no_labeled_rally |  | p07_s020 | 24429.000 | 24584.000 |  |  |  |
| true_plus_uncertain | True | loo_07 | test | 7 | extra_no_labeled_rally |  | p07_s022 | 24649.000 | 24841.000 |  |  |  |
| true_plus_uncertain | True | loo_07 | test | 7 | extra_no_labeled_rally |  | p07_s022 | 24908.000 | 25077.000 |  |  |  |
| true_plus_uncertain | True | loo_08 | test | 8 | matched | B | p08_s001 | 25151.000 | 25910.000 | -101.000 | 86.000 | 0.754 |
| true_plus_uncertain | True | loo_09 | test | 9 | matched | B | p09_s003 | 27407.000 | 28683.000 | -573.000 | 521.000 | 0.143 |
| true_plus_uncertain | True | loo_09 | test | 9 | extra_no_labeled_rally |  | p09_s001 | 26687.000 | 26962.000 |  |  |  |
| true_plus_uncertain | True | loo_09 | test | 9 | extra_no_labeled_rally |  | p09_s001 | 27093.000 | 27245.000 |  |  |  |
| true_plus_uncertain | True | loo_10 | test | 10 | matched | B | p10_s009 | 30457.000 | 30767.000 | -30.000 | 142.000 | 0.447 |
| true_plus_uncertain | True | loo_10 | test | 10 | extra_no_labeled_rally |  | p10_s001 | 29068.000 | 29220.000 |  |  |  |
| true_plus_uncertain | True | loo_10 | test | 10 | extra_no_labeled_rally |  | p10_s003 | 29276.000 | 29534.000 |  |  |  |
| true_plus_uncertain | True | loo_10 | test | 10 | extra_no_labeled_rally |  | p10_s004 | 29545.000 | 29625.000 |  |  |  |
| true_plus_uncertain | True | loo_10 | test | 10 | extra_no_labeled_rally |  | p10_s005 | 29870.000 | 30029.000 |  |  |  |
| true_plus_uncertain | True | loo_10 | test | 10 | extra_no_labeled_rally |  | p10_s007 | 30114.000 | 30344.000 |  |  |  |
| true_plus_uncertain | True | loo_10 | test | 10 | extra_no_labeled_rally |  | p10_s014 | 31137.000 | 31313.000 |  |  |  |
| true_plus_uncertain | True | loo_11 | test | 11 | matched | B | p11_s001 | 32563.000 | 33103.000 | -245.000 | 41.000 | 0.471 |
| true_plus_uncertain | False | fixed | train | 7 | matched | B | p07_s012 | 23293.000 | 23958.000 | -221.000 | -26.000 | 0.643 |
| true_plus_uncertain | False | fixed | train | 7 | extra_no_labeled_rally |  | p07_s001 | 21631.000 | 22242.000 |  |  |  |
| true_plus_uncertain | False | fixed | train | 7 | extra_no_labeled_rally |  | p07_s012 | 22742.000 | 22928.000 |  |  |  |
| true_plus_uncertain | False | fixed | train | 7 | extra_no_labeled_rally |  | p07_s012 | 22977.000 | 23244.000 |  |  |  |
| true_plus_uncertain | False | fixed | train | 7 | extra_no_labeled_rally |  | p07_s020 | 24429.000 | 24584.000 |  |  |  |
| true_plus_uncertain | False | fixed | train | 7 | extra_no_labeled_rally |  | p07_s022 | 24649.000 | 24841.000 |  |  |  |
| true_plus_uncertain | False | fixed | train | 7 | extra_no_labeled_rally |  | p07_s022 | 24908.000 | 25077.000 |  |  |  |
| true_plus_uncertain | False | fixed | train | 8 | matched | B | p08_s001 | 25151.000 | 25910.000 | -101.000 | 86.000 | 0.754 |
| true_plus_uncertain | False | fixed | test | 9 | matched | B | p09_s003 | 27407.000 | 28683.000 | -573.000 | 521.000 | 0.143 |
| true_plus_uncertain | False | fixed | test | 9 | extra_no_labeled_rally |  | p09_s001 | 26687.000 | 26962.000 |  |  |  |
| true_plus_uncertain | False | fixed | test | 9 | extra_no_labeled_rally |  | p09_s001 | 27093.000 | 27245.000 |  |  |  |
| true_plus_uncertain | False | fixed | test | 10 | matched | B | p10_s009 | 30457.000 | 30767.000 | -30.000 | 142.000 | 0.447 |
| true_plus_uncertain | False | fixed | test | 10 | extra_no_labeled_rally |  | p10_s001 | 29068.000 | 29220.000 |  |  |  |
| true_plus_uncertain | False | fixed | test | 10 | extra_no_labeled_rally |  | p10_s003 | 29276.000 | 29534.000 |  |  |  |
| true_plus_uncertain | False | fixed | test | 10 | extra_no_labeled_rally |  | p10_s004 | 29545.000 | 29625.000 |  |  |  |
| true_plus_uncertain | False | fixed | test | 10 | extra_no_labeled_rally |  | p10_s005 | 29870.000 | 30029.000 |  |  |  |
| true_plus_uncertain | False | fixed | test | 10 | extra_no_labeled_rally |  | p10_s007 | 30114.000 | 30344.000 |  |  |  |
| true_plus_uncertain | False | fixed | test | 10 | extra_no_labeled_rally |  | p10_s014 | 31137.000 | 31313.000 |  |  |  |
| true_plus_uncertain | False | fixed | test | 11 | matched | B | p11_s001 | 32563.000 | 33103.000 | -245.000 | 41.000 | 0.471 |
| true_plus_uncertain | False | loo_07 | test | 7 | matched | B | p07_s012 | 22742.000 | 23958.000 | -772.000 | -26.000 | 0.358 |
| true_plus_uncertain | False | loo_07 | test | 7 | extra_no_labeled_rally |  | p07_s001 | 21631.000 | 22242.000 |  |  |  |
| true_plus_uncertain | False | loo_07 | test | 7 | extra_no_labeled_rally |  | p07_s003 | 22297.000 | 22481.000 |  |  |  |
| true_plus_uncertain | False | loo_07 | test | 7 | extra_no_labeled_rally |  | p07_s020 | 24429.000 | 24584.000 |  |  |  |
| true_plus_uncertain | False | loo_07 | test | 7 | extra_no_labeled_rally |  | p07_s022 | 24649.000 | 24841.000 |  |  |  |
| true_plus_uncertain | False | loo_07 | test | 7 | extra_no_labeled_rally |  | p07_s022 | 24908.000 | 25077.000 |  |  |  |
| true_plus_uncertain | False | loo_08 | test | 8 | matched | B | p08_s001 | 25151.000 | 25910.000 | -101.000 | 86.000 | 0.754 |
| true_plus_uncertain | False | loo_09 | test | 9 | matched | B | p09_s003 | 27407.000 | 28683.000 | -573.000 | 521.000 | 0.143 |
| true_plus_uncertain | False | loo_09 | test | 9 | extra_no_labeled_rally |  | p09_s001 | 26687.000 | 26962.000 |  |  |  |
| true_plus_uncertain | False | loo_09 | test | 9 | extra_no_labeled_rally |  | p09_s001 | 27093.000 | 27245.000 |  |  |  |
| true_plus_uncertain | False | loo_10 | test | 10 | matched | B | p10_s009 | 30457.000 | 30767.000 | -30.000 | 142.000 | 0.447 |
| true_plus_uncertain | False | loo_10 | test | 10 | extra_no_labeled_rally |  | p10_s001 | 29013.000 | 29220.000 |  |  |  |
| true_plus_uncertain | False | loo_10 | test | 10 | extra_no_labeled_rally |  | p10_s003 | 29276.000 | 29534.000 |  |  |  |
| true_plus_uncertain | False | loo_10 | test | 10 | extra_no_labeled_rally |  | p10_s004 | 29545.000 | 29625.000 |  |  |  |
| true_plus_uncertain | False | loo_10 | test | 10 | extra_no_labeled_rally |  | p10_s005 | 29870.000 | 30029.000 |  |  |  |
| true_plus_uncertain | False | loo_10 | test | 10 | extra_no_labeled_rally |  | p10_s007 | 30114.000 | 30344.000 |  |  |  |
| true_plus_uncertain | False | loo_10 | test | 10 | extra_no_labeled_rally |  | p10_s014 | 31137.000 | 31313.000 |  |  |  |
| true_plus_uncertain | False | loo_11 | test | 11 | matched | B | p11_s001 | 32563.000 | 33103.000 | -245.000 | 41.000 | 0.471 |

## 研究上の結論と限界

- ラベルはClaudeによる暫定ラベルで、人が独立検証した完全な正解ではない。clearも独立正解ではない。未ラベルの実打球が候補にあると適合率目安を過小評価し、未ラベル打球の見逃しは測れない。
- v1の固定設定は第7・8点で選択済み。したがって1点抜きはラリー規則のみの条件付き交差検証で、第7・8点をテストにした回にも上流の設定選択が残る。全パイプラインの独立汎化評価ではない。
- 場面分類の参照に第9・10点自身の打球フレームが使われている（scenes/README.md）。固定分割も場面分類を含めた完全独立テストではない。既存の第9点の音の観察に基づくv2仮説もあり、再利用された探索的評価である。第7点p07_s012と第11点p11_s001以外は俯瞰判定の目視確認が限定される。
- Trueは俯瞰角度の支持であり、ライブである保証はない。uncertainの包含で候補/余分ラリーが増えても、それだけで拍手やリプレイを識別できたとはいえない。ラベル外ラリーの実在も未確認。
- 単なるラリー重なりは開始/終了を正しく推定できたことを意味しない。IoU・境界ずれ・見逃しを併せて判断する。音だけではボールつき・バウンド・ラケット接触を安定識別できない可能性があり、規則追加の効果は同じテスト点のcluster_v1/shape_only比較で判断する。
- 5点・同一試合の小標本であり、新規性や実用的な時間短縮はこの結果だけでは示せない。次は独立に確認した別の連続区間で、候補確認・不足探索・区間修正を含む人の所要時間を映像のみと比較する。

## 出力

- `summary.csv`：固定訓練/固定テスト/固定全点参考/1点抜き全点/共通第9〜11点、全条件・3規則群。
- `per_point_metrics.csv`：点別の確認候補、追加打球、clear/all再現率、F1、境界/分裂/結合。
- `rally_matches.csv`：正解対応、開始/終了の符号付きずれ、見逃し、余分検出の場面とフレーム。
- `rally_overlap_edges.csv`：全重なりの辺。1対1対応から漏れた分裂・結合も監査可能。
- `label_matches.csv` / `selected_candidates.csv`：打球ごとの不足と、実際に確認対象となる候補。
- `candidate_features.csv`：全点を抽出したv1候補と既存v2と同じ特徴。Falseも監査用に保存。
- `baseline_comparison.csv`：旧v1・新全場面v1・正解区間内oracleの統一指標。
- `selected_configs.json` / `param_search.csv`：各foldで訓練だけを使った選択と全探索。
- `timeline_{true_only|true_plus_uncertain}_point_XX.png`：場面、音の値、候補、GT、固定/1点抜きの区間。A含む/除外の予測は別段。青=clear、橙=inferred、GT Aは斜線。
- `protocol.json` / `run_metadata.json`：探索定義・入力ハッシュ・入力不変確認・実行環境。
- `verification.json`：保存CSVの件数・一意性・境界と、別実装DPによる全120点/モデル行の照合検証。
- 256 KiB超の生成物はこのフォルダの `.gitignore` で除外。再実行で再生成する。
