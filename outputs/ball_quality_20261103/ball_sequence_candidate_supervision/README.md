# 確認済み候補を時系列入力へ接続したデータ

36候補（実球21・誤検出15）を、対応するmedian背景とBGR t−3..tの4画像へ接続した。
全画面24候補、分割入力12候補、計34フレーム/ビュー。画像ハッシュ・整数source frame・座標変換を保存した。
`sequence_cases.json`と`candidate_index.csv`は候補の正誤を学習するための索引で、球中心や全画面の非可視状態の教師には変換していない。

`legacy_label_audit.json`は追加質問前の既存300標本の点検記録。
300件中249件にvisibleと中心座標、51件にnot_visibleがあり、reviewer_idは空欄だった。
ただし旧レビュー画面の保存処理はreviewer_idを書かないため、空欄を「人手未確認」の根拠にしてはならない。

**その後、ユーザーが300件は自分で球をクリックしたと明示した。**
現在の出典確認は`../balltrack_manual_finetune_pilot/manual_annotation_provenance.json`と`../ball_annotation_provenance.json`を参照。
既存300件は手動中心注釈として追加学習に使用可能。36候補の円による正誤確認とは注釈の種類が異なる。
元CSVの作成者欄・座標・可視性は上書きせず保持した。

候補特徴の試作結果は`../ball_tracknet_candidate_feature_comparison/`と`../ball_tracknet_common_view_comparison/`。
通常の球検出器の追加学習は、候補円を中心GTへ昇格させず、実際にクリックされた300件を使う別実験へ進めた。
