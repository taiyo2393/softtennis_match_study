# match2 第7〜11点：人の確認を反映したコート図（review2）

5本の point_07〜11.mp4 は打者に human_confirmed_hitters_p07_11.csv の confirmed_hitter を使用し「確認済」と表示する。自動判定と異なる打球には自動名も表示。第7点6は回答の名前を追跡情報から N2 と解釈しており、要再確認。時刻・側・ラリーは既存の本人ラベル。位置・人物自動対応・球候補の精度は未検証。

正解表32行との一致：変更前 25/32（未決定7）、変更後 32/32（未決定0）。同じ例で規則を調整した照合であり、独立評価ではない。

## A〜E の規則

- A：25打球は旧自動結果を本人了承、6打球は明示回答、1打球は回答を再解釈。元の回答と解釈を別列に保存。番号は点内の行順。
- B：球曖昧は棄権し手首が決まれば採用。両距離が各枠幅25%未満なら球曖昧。単独可視枠内の球、片方欠測かつ相方の近くに球なし、の一般規則を追加。hitter_auto.csvは自動結果のまま保持。rule_version=review2_20261119。
- C：枠欠測中はラリー内の最終観測位置を保持（offscreen_hold）。右図の破線白抜き円＋画面外・推定。模式線にも使用。次の観測で解除。
- D：同側の人物名が欠測中に非審判の新trackが現れたら、既知の相方との重複を禁じて空き名へ対応。時間・距離ゲートを外し、競合候補は前位置への距離最小。初接続はreentry_vacant_slot。
- E：奥サーブ時24.07mをアンカーとする。開始〜サーブの下隠れ・持ち上がり・25.27m超も保持。枠下端が安定値より高さ6%上がり42F以内に戻る区間は前後y2で時間線形補間。横位置は従来観測のまま。奥側のみ。射影は変更しない。

既存の下隠れ判定（同Fの手前枠との横重なりが奥枠幅10%以上、手前y1<奥y2）、前後アンカー補間、従来速度・高さフィルタは継続。サーブ保持と持ち上がりは優先し、補正値を中央値で崩さない。横には追加平滑化をしない。打球者間の破線は説明用で、球軌道・バウンド・3D位置ではない。

review2b：奥側だけ、同一選手・ラリー内の高さ適格枠（下隠れ含む）の±15元F・90パーセンタイルを足元y2の包絡とする。3観測未満は変更せず、元y2より小さくしない。速度判定・下隠れアンカー・持ち上がり判定へ適用し、サーブアンカーを優先する。0.5m以上の直接補正観測はobserved_x_depth_envelope、補正量はenvelope_shift_m。手前側と既存横位置は保持する。短い本当の後退を抑える可能性があり、長い1m以上補正区間の拡大画像と統計をreview2_report.mdに記録する。

## 再実行

```powershell
& study/.venv/Scripts/python.exe -B -X utf8 outputs/shot_pipeline_20261117/court_view_p07_11/build_court_view.py --review2
& study/.venv/Scripts/python.exe -B -X utf8 outputs/shot_pipeline_20261117/court_view_p07_11/verify_court_view.py
```

球候補・姿勢は保存済み出力を使用。元動画は59.94fps、ffmpegは C:/ffmpeg/bin/ffmpeg.exe。表示動画は1920×1080、15fps、無音、原則2元Fごと。正確な元F対応は video_source_frame_map_p*.csv。

詳細な精度、再認識原因と割り当て区間、奥行き件数、白線との差、持ち上がり補正統計、限界は [review2_report.md](review2_report.md)。検証結果は review2_verification.json。

入力tracking_v2/・labels_human/・pose/・元動画は読み取りのみ。変更前の派生表はreview2_before/、打者はhitter_auto_before_review2.csv。大きなCSVと画像・動画はgit除外。commit/pushはしない。


## review2c（2026-11-19）

現行の5動画と配置CSVはreview2c。far包絡75%、far/near速度棄却30元F超の再初期化、再接続の空き名判定は最終観測から1秒超。詳細・残存誤差・比較表はreview2_report.mdのH節。過去節は当時の結果として保存。
