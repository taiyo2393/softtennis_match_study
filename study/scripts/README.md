# 実装予定スクリプト

| 順序 | ファイル | 役割 |
| --- | --- | --- |
| 00 | `00_probe_video.py` | fps、解像度、総フレーム数、ハッシュを記録 |
| 01 | `01_detect_cuts.py` | カット候補を作成（主カメラ判定は次段階） |
| 02 | `02_build_cut_review.py` | 候補の前後を並べたレビュー画像とHTMLを作成 |
| 03 | `03_review_cuts_server.py` | ブラウザでカット前後の区間を分類し、判定CSVを保存 |
| 04 | `04_build_main_segments.py` | 判定済みカットから解析可能な主カメラ連続区間を生成 |
| 05 | `05_detect_fixed_main_camera.py` | 固定主カメラの画角との一致度で解析区間を抽出 |
| 06 | `06_build_fixed_view_audit.py` | 主・非主カメラを各10件ずつ抽出して精度監査 |
| 07 | `07_review_fixed_view_audit_server.py` | 20画像を盲検で確認し、監査結果を保存 |
| 08 | `08_score_court_geometry.py` | ネット・コート線の固定配置が見える度合いを算出 |
| 09 | `09_build_analysis_ready_segments.py` | コート幾何と残像なしを両方満たす区間だけを確定 |
| 10 | `10_track_players_pilot.py` | 解析可能な定点区間で人物候補を追跡し、ID監査用CSV・動画を出力 |
| 11 | `11_review_tracking_server.py` | 対象前衛の追跡IDとID切替・検出漏れを2秒区間で監査 |
| 12 | `12_build_tracking_review_clips.py` | 監査用の各2秒区間を、必ず終了する短い動画として生成 |
| 13 | `13_build_clean_annotation_proxy.py` | 基準条件用の追跡枠なし1080p60映像を生成 |
| 14 | `14_trial_annotation_server.py` | manual / assisted を分離して試技ラベルを保存する入力画面 |
| 15 | `15_propose_rally_boundaries.py` | 定点主カメラへの出入りと視覚カットからラリー境界候補を作成（確定はしない） |
| 16 | `16_build_boundary_review_clips.py` | ラリー終了候補の前後3秒・15fps動画を作成 |
| 17 | `17_review_rally_boundaries_server.py` | ラリー終了候補を人が確定・除外するローカル画面 |
| 18 | `18_build_rally_window_candidates.py` | 確認済み終了候補と直前の定点画角再入から完全ラリー候補窓を作成 |
| 19 | `19_build_rally_start_review_clips.py` | 完全ラリー候補窓の開始候補を前後3秒動画にする |
| 20 | `20_review_rally_starts_server.py` | ラリー開始候補を人が確認するローカル画面 |
| 21 | `21_detect_audio_onset_candidates.py` | 放送音声の急峻な音響イベント候補を抽出し、境界候補の前後音響を要約 |
| 32 | `32_prepare_motion_prediction_pilot.py` | 既確認の主カメラ区間から再現可能な20窓を抽出し、動作予測用レビュー動画を生成 |
| 33 | `33_review_motion_prediction_events_server.py` | 相手後衛グラウンドストロークを1窓1件で確認し、予測イベントCSVへ保存 |
| 34 | `34_run_racketvision_balltrack_pilot.py` | 隔離したRacketVision BallTrackで、1本の固定主カメラ候補動画から球座標候補CSVを生成 |
| 35 | `35_build_balltrack_overlay.py` | BallTrack候補点を元動画へ重ね、採否を視覚監査する動画を生成 |
| 53 | `53_extract_target_front_tracks.py` | 4Kの対象前衛だけを高解像度で追跡し、根拠動画を作成 |
| 54 | `54_review_highres_front_tracks_server.py` | 高解像度・対象前衛追跡を人が監査する画面 |
| 55 | `55_propose_shot_direction_candidates.py` | インパクト後だけのBallTrack候補列から打球方向候補と根拠動画を作成 |
| 56 | `56_review_shot_direction_server.py` | AI候補を人が確認・修正し、対象前衛視点の実打球方向を保存する画面 |
| 57 | `57_build_front_motion_timepoints.py` | 方向ラベルを参照せず、打球前の対象前衛の横移動代理指標を時点別に作成 |
| 58 | `58_summarize_front_motion_timepoints.py` | 時点別の方向動作率・条件付き一致率を探索的に要約 |
| 59 | `59_extract_target_front_pose.py` | 監査済み対象前衛枠の中からYOLO Poseで左右足首候補を抽出 |
| 60 | `60_build_target_front_pose_previews.py` | 足首点を重ねたPose監査動画を作成 |
| 61 | `61_review_target_front_pose_server.py` | 足首点が対象前衛の足元を追えているかを人が監査する画面 |
| 62 | `62_build_front_motion_pose_timepoints.py` | 監査済み足首中点から、打球前の前衛横移動を時点別に作成 |
| 63 | `63_summarize_semiautomated_pilot.py` | AI候補率・人手確認後の一致・追跡監査を予備研究として要約 |
| 03 | `03_extract_pose.py` | 対象前衛の4K cropからPoseを取得 |
| 04 | `04_build_review_queue.py` | 要確認フレームを一覧化 |
| 05 | `05_make_features.py` | 骨盤・足の変位と速度を作成 |
| 06 | `06_run_analysis.py` | 時点別統計と分類評価 |

追跡IDはカットをまたいで接続しない。すべての出力には `video_id` と原動画時刻を含める。

## 動作予測パイロット

`32` は既に確認済みの定点主カメラ窓を、固定シードで20件だけ選ぶ。これはイベントの自動検出器ではなく、定義・欠測・作業時間を確認する予備標本である。各窓の中から相手後衛グラウンドストロークを1件だけ人手で確定する。見つからなければ除外するため、20窓が20イベントになる前提ではない。

```powershell
& .\.venv\Scripts\python.exe .\scripts\32_prepare_motion_prediction_pilot.py `
  --input .\data\raw\match1.mp4 --video-id match1

& .\.venv\Scripts\python.exe .\scripts\33_review_motion_prediction_events_server.py `
  --video-id match1
```

ブラウザで `http://127.0.0.1:8775/` を開く。インパクトは動画内時間ではなく、保存時に原動画時刻・原動画フレームへ変換される。音声候補の精度が低かったため、音だけで採用してはならない。

## RacketVision BallTrack の技術検証

`34` は、RacketVisionの学習済みMS-TrackNetV3を**球候補生成**にだけ使う。未来フレームを使う背景中央値・時系列処理はイベント候補作成では許容するが、前衛横移動を予測するモデルの入力には使わない。最初は1本だけ実行し、CSVの可視率と重ね描画を監査してから対象を増やす。

```powershell
& .\third_party\racketvision-env\Scripts\python.exe .\scripts\34_run_racketvision_balltrack_pilot.py `
  --video-id match1 --candidate-id match1_motion_pilot_001
```

出力は `data/balltrack/racketvision/<video_id>/<candidate_id>/balltrack_predictions.csv`。これは正解ラベルではなく、各フレームの `visible` と `confidence` を持つAI候補である。

```powershell
& .\third_party\racketvision-env\Scripts\python.exe .\scripts\35_build_balltrack_overlay.py `
  --video-id match1 --candidate-id match1_motion_pilot_001
```

`33_review_motion_prediction_events_server.py` を再起動すると、候補動画カードから重ね描画を開ける。点が球へ追従しているかは、この動画で監査する。候補数や可視率だけで精度を主張してはならない。

## 実打球方向のAI補助確認

インパクトフレームは人手確定済みであることを前提に、`55` は打球後だけのBallTrack候補列から画面上の左右方向を提案する。これは連続的なボール座標を人手で入力する処理ではない。AI候補の根拠動画を見て、対象前衛から見た実打球方向を確認・修正する。

```powershell
& .\.venv\Scripts\python.exe .\scripts\55_propose_shot_direction_candidates.py --video-id match1
& .\.venv\Scripts\python.exe .\scripts\56_review_shot_direction_server.py --video-id match1
```

ブラウザで `http://127.0.0.1:8782/` を開く。保存先は `annotations/shot_direction_review_match1.csv`。ここで得た方向は、打球前の前衛動作を作る入力ではなく、前衛動作との対応を判定するための事後的な正解基準である。

## 実行例

```powershell
& 'C:/Users/ta239/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' .\scripts\01_detect_cuts.py `
  --input .\data\raw\match1.mp4 --video-id match1 `
  --output .\data\segments\match1_cut_candidates.csv
```

`threshold=0.12` は候補抽出用の初期値であり、正解ラベルではない。見逃しを避けるため、候補を目視で確認してから主カメラ区間を確定する。

## 人物追跡の予備検証

`10_track_players_pilot.py` は、最長の解析可能定点区間を既定で選び、人物検出と追跡の候補を出力する。観客・審判を除くため、**足元が画角内のコート周縁に入る人物だけ**を残す。この周縁多角形は現時点では `match1` 固有であり、他大会映像に流用してはならない。`15 fps` は追跡の可否を安価に確認するためだけの設定であり、インパクト前時点の研究解析には使わない。対象前衛の同定、IDの入れ替わり、検出漏れをレビューしてから、4K・59.94 fps のPose抽出へ進む。

```powershell
& .\.venv\Scripts\python.exe .\scripts\10_track_players_pilot.py `
  --input .\data\raw\match1.mp4 --video-id match1
```

```powershell
& .\.venv\Scripts\python.exe .\scripts\11_review_tracking_server.py `
  --video-id match1 --segment-id match1_ready_012
```

監査サーバーを起動する前に、区間動画を作成する。

```powershell
& .\.venv\Scripts\python.exe .\scripts\12_build_tracking_review_clips.py `
  --video-id match1 --segment-id match1_ready_012
```

ブラウザで `http://127.0.0.1:8768/` を開く。各2秒区間について、対象前衛を安定して追えていれば `usable` とそのIDを保存する。IDの入替・対象の欠落・対象前衛ではない区間だけを明示し、全ボックスを人手で修正しない。

```powershell
& 'C:/Users/ta239/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' .\scripts\02_build_cut_review.py `
  --input .\data\raw\match1.mp4 `
  --candidates .\data\segments\match1_cut_candidates.csv `
  --output-dir .\data\segments\review\match1
```

```powershell
& 'C:/Users/ta239/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' .\scripts\03_review_cuts_server.py --video-id match1
```

```powershell
& 'C:/Users/ta239/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' .\scripts\04_build_main_segments.py `
  --video-id match1 --duration-sec 2929.728
```

```powershell
& 'C:/Users/ta239/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' .\scripts\05_detect_fixed_main_camera.py `
  --input .\data\raw\match1.mp4 --video-id match1 --duration-sec 2929.728 --reference-sec 172.8
```

```powershell
& 'C:/Users/ta239/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' .\scripts\06_build_fixed_view_audit.py `
  --input .\data\raw\match1.mp4 --video-id match1
```
