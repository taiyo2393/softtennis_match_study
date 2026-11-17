# 009のAI選手指定によるN2修正

親フォルダの自動出力がN2に非競技者を追跡したため、AIが画像F21855で実際のサーバーraw148を指定した。F21848〜22147に存在する300個の元検出枠だけをN2へ結び付けた修正版。

指定窓F21915〜21975は変更せず、N1・F1・F2の725観測をそのまま保持した。4人分の観測があるのは16/31要求時点で、残る15時点は奥F1が欠測。N2は正しいサーバーを指して見えるが、人手の独立本人正解はなく、完全自動の追跡成功には数えない。

- `seed_design.json`: AI指定時点・raw ID・適用範囲・限界。
- `protocol.json`: 元の自動出力への参照と修正版の位置づけ。
- `four_player_path.csv`: N2だけを置き換えた元観測の結合経路。
- `prefix_frame_features.csv` / `snapshot_player_features.csv` / `prefix_player_summary_features.csv`: 同じ指定窓の配置候補。
- `AI_seeded_review_h264.mp4`: 対象範囲の修正動画。
- `candidate_sequence_grid.jpg`: AIが点検した修正画像列。
- `verification.json`: 元枠1025件・他3人725件の保持・379値・300動画フレームの検査。

得点側は未確認。モデル学習、独立精度評価、座標の補間はしていない。詳細は親フォルダのREADME。
