# 得点表示候補のブラインド映像確認（16件）

4層（境界complete/未解決 × 単純な1点増加提案あり/なし）から固定乱数20260927で各4件を抽出し、別の固定乱数20260928で確認順をシャッフルした。下の一覧には提案した増点行や層を表示しない。

各動画は候補開始1秒前から候補終了3秒後までの連続映像。H.264映像・AAC音声、1920×1080、縦横比16:9。候補窓に複数ポイントやリプレイが混ざる場合は `linked_to_this_rally=unknown` とする。

確認順：①候補窓が1ポイントに対応するか、②得点成立事象と表示更新がそのポイントに対応するか、③増点行とチーム、④チームの手前/奥。接触・終了のフレームは別途連続映像で標定する。

|順番|候補|候補窓の動画内時刻|連続動画|
|---:|---|---|---|
|1|022|1.000–17.000秒|[H.264動画](/C:/Users/ta239/AndroidStudioProjects/tennis_ai_app2/outputs/frontcourt_handoff_20260927/022_score_validation_h264.mp4)|
|2|061|1.000–13.500秒|[H.264動画](/C:/Users/ta239/AndroidStudioProjects/tennis_ai_app2/outputs/frontcourt_handoff_20260927/061_score_validation_h264.mp4)|
|3|057|1.000–29.500秒|[H.264動画](/C:/Users/ta239/AndroidStudioProjects/tennis_ai_app2/outputs/frontcourt_handoff_20260927/057_score_validation_h264.mp4)|
|4|043|1.000–17.500秒|[H.264動画](/C:/Users/ta239/AndroidStudioProjects/tennis_ai_app2/outputs/frontcourt_handoff_20260927/043_score_validation_h264.mp4)|
|5|013|1.000–8.000秒|[H.264動画](/C:/Users/ta239/AndroidStudioProjects/tennis_ai_app2/outputs/frontcourt_handoff_20260927/013_score_validation_h264.mp4)|
|6|032|1.000–19.500秒|[H.264動画](/C:/Users/ta239/AndroidStudioProjects/tennis_ai_app2/outputs/frontcourt_handoff_20260927/032_score_validation_h264.mp4)|
|7|005|1.000–20.500秒|[H.264動画](/C:/Users/ta239/AndroidStudioProjects/tennis_ai_app2/outputs/frontcourt_handoff_20260927/005_score_validation_h264.mp4)|
|8|021|1.000–8.500秒|[H.264動画](/C:/Users/ta239/AndroidStudioProjects/tennis_ai_app2/outputs/frontcourt_handoff_20260927/021_score_validation_h264.mp4)|
|9|029|1.000–24.000秒|[H.264動画](/C:/Users/ta239/AndroidStudioProjects/tennis_ai_app2/outputs/frontcourt_handoff_20260927/029_score_validation_h264.mp4)|
|10|012|1.000–6.500秒|[H.264動画](/C:/Users/ta239/AndroidStudioProjects/tennis_ai_app2/outputs/frontcourt_handoff_20260927/012_score_validation_h264.mp4)|
|11|048|1.000–29.000秒|[H.264動画](/C:/Users/ta239/AndroidStudioProjects/tennis_ai_app2/outputs/frontcourt_handoff_20260927/048_score_validation_h264.mp4)|
|12|011|1.000–49.500秒|[H.264動画](/C:/Users/ta239/AndroidStudioProjects/tennis_ai_app2/outputs/frontcourt_handoff_20260927/011_score_validation_h264.mp4)|
|13|037|1.000–14.000秒|[H.264動画](/C:/Users/ta239/AndroidStudioProjects/tennis_ai_app2/outputs/frontcourt_handoff_20260927/037_score_validation_h264.mp4)|
|14|023|1.000–24.500秒|[H.264動画](/C:/Users/ta239/AndroidStudioProjects/tennis_ai_app2/outputs/frontcourt_handoff_20260927/023_score_validation_h264.mp4)|
|15|063|1.000–18.500秒|[H.264動画](/C:/Users/ta239/AndroidStudioProjects/tennis_ai_app2/outputs/frontcourt_handoff_20260927/063_score_validation_h264.mp4)|
|16|014|1.000–25.000秒|[H.264動画](/C:/Users/ta239/AndroidStudioProjects/tennis_ai_app2/outputs/frontcourt_handoff_20260927/014_score_validation_h264.mp4)|

判定欄は [score_suggestion_blind_review.csv](/C:/Users/ta239/AndroidStudioProjects/tennis_ai_app2/outputs/frontcourt_handoff_20260927/score_suggestion_blind_review.csv)。各動画で①増点行（top/bottom/none/unclear）、②表示更新が候補のポイントに対応するか（yes/no/unknown）、③必要ならチームと画面側を記入する。提案値はこの票には含めない。現時点では全16件とも未確認。
