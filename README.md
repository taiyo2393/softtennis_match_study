# ソフトテニス試合動画研究

固定カメラのダブルス試合映像を使った、4人の選手追跡・ボール追跡・ラリー分析の研究。

## フォルダ構成

| フォルダ | 内容 |
| --- | --- |
| `study/` | 元プロジェクト（旧 `Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study`）。元動画・前処理スクリプト・アノテーション。詳細は `study/README.md` |
| `outputs/` | 4人追跡・ボール品質・ラリー分析の実験（旧 `AndroidStudioProjects\tennis_ai_app2\outputs` から移動、2026-11-17） |
| `Ultralytics/` | Ultralytics の設定（git管理外） |

旧 `softtennis_frontcourt_study` の場所には `study/` へのジャンクションを残してある。過去の記録（json・csv）に書かれた絶対パスはそのまま読める。

## git管理の対象

コード・メモ・小さなデータ（256KB以下のcsv/json）だけを管理する。動画・画像・モデル重み・`.venv`・`third_party/`・`node_modules/` は管理しない（`.gitignore` 参照）。
