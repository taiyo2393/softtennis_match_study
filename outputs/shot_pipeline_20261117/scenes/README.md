# match2 第7〜11点：場面分割と俯瞰カメラの仮判定

固定校正を使えるカメラ画角かを再点検するための準備データ。自動判定は正解ラベルではない。**俯瞰画角の一致と、ライブ／リプレイの区別は別問題**で、同じ俯瞰画角のリプレイは True になり得る。True も校正精度・選手ID・足元位置・打球の実在を保証しない。

## 再実行

リポジトリルートの PowerShell：

```powershell
& study\.venv\Scripts\python.exe -B -X utf8 -u outputs\shot_pipeline_20261117\scenes\detect_scenes.py
& study\.venv\Scripts\python.exe -B -X utf8 -u outputs\shot_pipeline_20261117\scenes\recount.py
```

前者は特徴計測・分割・判定・シート・再集計まで実行する。後者は画像を読まず、現在の場面判定から再集計する。しきい値だけ変更して既存のフレーム特徴を再利用する場合は前者に `--reuse-features` を付ける（入力動画・特徴定義を変えた場合には使用不可）。生成物のみ本フォルダに上書きし、既存の tracking / audio / labels / shot_table / 元動画は変更しない。`scene_review.csv` は上書きしない。git commit は行わない。

## 入力・フレーム番号

- 動画パスと FPS は `../tracking/point_07/protocol.json` の `source_video` / `source_fps`（60000/1001）。
- 対象は `outputs/match2_point_inventory_20261117/point_inventory.csv` の第7〜11点、`display_from_frame` から `next_display_from_frame - 1`。全て0始まり、開始・終了を含む。
- 第7点 F21600–25139、第8点 F25140–26684、第9点 F26685–29009、第10点 F29010–32549、第11点 F32550–33584。合計11,985フレーム。
- 各読み出し区間の先頭で1回だけ seek し、後は全て `grab` で進め、必要な画像を `retrieve`。各フレームで `CAP_PROP_POS_FRAMES` を assert する。`make_review_sheets.py` と同じ0始まりの規約。参照画像の収集は最小〜最大参照フレームを1回の連続走査、特徴計測とシート作成は点ごとに各1回の連続走査。画像寸法1920×1080とFPSも assert。
- 参照は `outputs/shot_table_20261117/shot_table.csv` の第4〜6点の全 `contact_frame_ref` と、`../labels/point_09.csv` / `point_10.csv` の rally B / confidence=clear の `contact_frame`。第7・8・11点のラベルは参照に使わない。使用Fは `reference_frames.csv` に保存。
- 再集計する座標は **`../shot_table.csv`（第7〜11点）**。第4〜6点の参照表とは別ファイル。音候補は `../audio/candidates.csv`（v1全候補、190件）、ラベルは `../labels/point_07.csv`〜`point_11.csv`（91件）。他の版・採択済み候補に置き換えない。

## 分割規則（detect_scenes.py の定数）

全フレームを160×90に縮小。HSVのH/Sヒストグラム（24×16ビン、L1正規化）と3×3 Gaussianぼかし後のグレースケールを使う。histは Bhattacharyya 距離（0〜1）、grayは画素絶対差の平均 / 255。

| 種類 | 候補条件 |
|---|---|
| 急な切替 | 隣接hist ≥ 0.28、または隣接gray ≥ 0.16、または（hist ≥ 0.12 かつ gray ≥ 0.085） |
| 緩やかな切替 | 12F前との（hist ≥ 0.18 かつ gray ≥ 0.045）、またはgray ≥ 0.095 |

候補の間隔が6F以下なら同一群。急な切替を含む群は急変候補の局所群ごとに `hist_adj + gray_adj` 最大のフレームを切替先頭とする。緩変候補を含む群は、急変の有無によらず「最初の緩変候補−12+1」と「最後の緩変候補+1」も境界にし、混合映像を独立した場面として残す（両方を含む群は `abrupt_and_gradual`）。12Fは約0.20秒。ワイプ／ディゾルブの正確な開始・終了の保証はなく、疑わしい前後の映像を確認するための包絡区間である。選手の大きな動きやパンも拾うため、同一ショットを細かく分割する場合がある。

10F未満の場面は原則として前の場面へ吸収、点の先頭なら次へ吸収。`scene_cuts.csv` はしきい値を超えた**全フレーム**の差分・急変/緩変フラグ・群ID・提案境界・採用境界・吸収理由を保存する。境界自体のレコードも含み、`group_rule=point_start/point_end` は区間端、`absorbed_short_*` は短区間の吸収。候補フレーム数と採用カット数は一致しない。点境界は強制し、点をまたいで場面を結合しない。全フレームを重複・欠落なく場面へ割り当てる。

## カメラ判定規則

コート線の有無だけでは通過させず、**参照画像との画面構造の相関と青コートの占有率のAND条件**を使う。Hough線分は採用しない。

1. グレースケール画像の上1/6（得点表示など）を除き、80×38へ縮小。平均を引いてL2正規化し、参照集合との最大内積を求める。これは Pearson 相関に相当する（`ref_corr`）。
2. 下2/3で OpenCV HSV の H=90〜130、S≥60、V≥40 の画素割合を求める（`blue_ratio`）。Hは0〜179。
3. フレームごとに main支持 = corr≥0.92 **かつ** blue≥0.45、other支持 = corr<0.65 **または** blue<0.10。
4. 場面内のmain支持率が98%以上なら **True**、そうでなくother支持率が70%以上なら **False**、残りは **uncertain**。uncertainをFalseに算入しない。

`scenes.csv` に相関の最小・10%点・中央値・最大、青割合の10%点・中央値、main/other支持率を記録する。ズームやパン、ワイプの一部を含むと uncertain になり得る。しきい値は説明可能な暫定定数であり、独立した正解による感度・特異度の検証は未実施。初回シート確認で寄り映像にも相関0.8台が出たため、Trueの条件を相関0.92・青割合0.45・支持率98%へ厳しくした。第9・10点は自身の参照を含むため、その点の一致率を独立評価と解釈できない。異常座標との対応が良くなるようにしきい値を最適化していない。

## 目視確認と修正

`scene_sheets/point_XX_sheet_YY.jpg` は1枚最大10場面。各行は場面ID・両端を含む範囲・自動判定・代表特徴量と、先頭／中央／末尾の元動画の全画面（各320px幅）。一覧は `scene_sheet_index.csv`。

`scene_review_template.csv` を本フォルダ内で `scene_review.csv` にコピーし、`claude_label` に **True / False / uncertain** を記入する。空欄はautoのまま。`review_note` に根拠やリプレイの疑いを残せる。`scene_id` とフレーム範囲は変更しない。`recount.py` は記入済み判定を優先し、不正ラベル・未知/重複ID・古い境界を拒否する。集計時の実効判定は `effective_scenes.csv` の `effective_label` / `label_source` で追跡できる。

先頭・中央・末尾だけでは短い挿入を見落とし得る。特に第7点 F23994、第8点 F25962付近のディゾルブ、uncertain場面、異常座標がTrueに残った場面を動画で確認する。境界自体が誤っている場合はラベル修正だけで混在を解決できないため、分割条件を直して検出を再実行し、新しいテンプレートで再レビューする。

## 再集計の定義・出力

`recount.recount()` を関数として呼び出せる。接触フレーム（音は整数の `candidate_frame`）を含む場面に割り当て、True / False / uncertain / outside（対象外）を分離する。全入力が対象場面に一意に入ることもassertする。

- 異常のスクリーニング条件は F1/F2 の **y>26.0 m または y<11.885 m**（等号を含まない、欠損は除外）。この固定範囲からの逸脱自体を物理的不可能や別カメラの証明とはしない。`abnormal_shot_rows.csv` はどちらかが逸脱した打球行、`abnormal_positions.csv` は選手別の逸脱座標。`recount_summary.csv` は両方の数え方と上限/下限別件数を残す。
- `audio_scene_membership.csv` はv1全候補。`label_scene_membership.csv` は全ラベル。`labels_other_camera.csv` はFalseのラベルを point / rally / shot_index / contact_frame とともに列挙。uncertainは `labels_uncertain_camera.csv` に分離。
- `point07_rally_A_scenes.csv` は指定の閉区間 **F22776〜23032（257F）** と各場面の交差。元の音声READMEは半開区間 [22776,23032)（256F）だが、今回の問い合わせは末尾を含めた。差は末尾1Fのみ。
- `frame_features.csv` は全フレームの根拠値、`detection_protocol.json` は入力・定数を記録する。

## 結果（再集計時の判定に依存）

<!-- RECOUNT_RESULTS_START -->
実効判定はレビュー適用 0 場面。未修正部分は自動仮判定。

| 点 | 場面数 | True F / 全F | True割合 | False F | uncertain F | シート数 |
|---|---:|---:|---:|---:|---:|---:|
| 7 | 23 | 1915/3540 | 54.10% | 520 | 1105 | 3 |
| 8 | 16 | 817/1545 | 52.88% | 298 | 430 | 2 |
| 9 | 7 | 1378/2325 | 59.27% | 262 | 685 | 1 |
| 10 | 29 | 381/3540 | 10.76% | 828 | 2331 | 3 |
| 11 | 7 | 558/1035 | 53.91% | 447 | 30 | 1 |
| all | 82 | 5049/11985 | 42.13% | 2355 | 4581 | 10 |

| 実効カメラ判定 | 異常打球行（上限超/下限未満） | 異常選手座標 | 音v1候補 | ラベル（clear） |
|---|---:|---:|---:|---:|
| True | 27 (27/0) | 27 | 180 | 91 (58) |
| False | 0 (0/0) | 0 | 7 | 0 (0) |
| uncertain | 0 (0/0) | 0 | 3 | 0 (0) |
| outside | 0 (0/0) | 0 | 0 | 0 (0) |
| all | 27 (27/0) | 27 | 190 | 91 (58) |

第7点 rally A（閉区間F22776–23032）：
- p07_s012 F22721–23986、判定 True、交差 F22776–23032（257F）。

ディゾルブ指定フレームの所属（境界の正確さを保証するものではない）：
- 第7点 F23994：p07_s013 F23987–24181、False。
- 第8点 F25962：p08_s002 F25957–25987、uncertain。

26m超は全27打球行。このうち第7点rally Aが4行、rally Bのみで合計23行。元の『23件』と全ラベルの集計母数はこの4行分異なる。

別カメラFalseのラベル一覧は [labels_other_camera.csv](labels_other_camera.csv)、uncertain一覧は [labels_uncertain_camera.csv](labels_uncertain_camera.csv)。Trueに残る異常も [abnormal_shot_rows.csv](abnormal_shot_rows.csv) で確認する。
<!-- RECOUNT_RESULTS_END -->

これは再解析の準備であり、既存の打球表・音声評価・ラベルを修正した結果ではない。別カメラ混入を除去した後も、Trueに残る異常座標には固定校正のずれ、足元でないbbox下端、追跡IDの誤り等があり得る。確認後に独立した足元参照・校正チェックと、ライブ／リプレイの重複確認を行ってから研究結果を確定する。

## 実行・確認記録

指定の `study/.venv/Scripts/python.exe`（OpenCV 5.0.0）で動画からの特徴抽出を実行後、シートで混合映像の境界を点検して分割・判定定数を修正し、同じ全フレーム特徴から `--reuse-features` で最終出力を生成した。`recount.py` 単独の再実行も終了コード0。

- 全11,985Fの連続性、一意な場面被覆、全場面10F以上をassert。参照31F。
- 差分しきい値を超える2,102Fが候補CSVに全て存在することを別途照合。短区間の吸収境界16件は境界レコードに記録。最終82場面（点内境界77箇所）。
- JPEG10枚の読込・寸法（幅1220、高さ210×収録場面数）、インデックスと全82IDの対応、テンプレートの空欄を検証。第7〜11点のシートから画像・文字の配置と代表的な画角／切替を目視点検した。全フレームの手動正解付けは行っていない。
- 一時CSVでレビューの優先適用、未記入場面のauto維持、不正ラベル・古い境界の拒否を確認（一時CSVは削除、本番 `scene_review.csv` は未作成）。
- Git差分は本 `scenes/` の新規ファイルのみ。既存の追跡・音・ラベル等に変更なし。JPEGは既存のgitignore対象。
