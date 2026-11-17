"""Preserve human side evidence separately from scoreboard-derived team labels."""
import csv
from pathlib import Path

OUT = Path(__file__).resolve().parent

def read(name):
    with (OUT/name).open(encoding='utf-8-sig', newline='') as f:
        r = csv.DictReader(f)
        return list(r.fieldnames), list(r)

def write(name, fields, rows):
    with (OUT/name).open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

fields, rows = read('rally_label_review.csv')
r = next(r for r in rows if r['candidate_id']=='match1_rally_window_001')
r['winner_camera_side'] = 'near'
r['winner_camera_side_review_status'] = 'human_confirmed'
r['winner_team_id'] = '上岡俊介・丸山海斗'
r['winner_review_status'] = 'human_side_plus_visually_read_score_team_derived'
r['winner_evidence_type'] = 'human_video_side_review_and_scoreboard_bottom_0_to_1'
r['winner_evidence_source_frame'] = '659'
r['team_side_mapping_status'] = 'initial_anchor_human_side_plus_score_reading'
r['reviewer'] = 'human_user side; Codex scoreboard reading 2026-10-03'
note = 'User explicitly answered near for 001_first_point_side_anchor_h264.mp4. Score at F539 is 0-0; F659/F779 bottom point is 1. Team label derived from scoreboard; boundaries and tracking remain unconfirmed.'
if note not in r['review_note']:
    r['review_note'] = (r['review_note']+' | '+note).strip(' |')
write('rally_label_review.csv', fields, rows)

fields, rows = read('human_review_log.csv')
if not any(r['review_id']=='20261003_001_winner_side' for r in rows):
    rows.append(dict(review_id='20261003_001_winner_side', date='2026-10-03', candidate_id='match1_rally_window_001', field='winner_camera_side', value='near', reviewer='human_user', evidence='001_first_point_side_anchor_h264.mp4; user response 手前', prior_value='unreviewed', scope='camera side only; team derived separately from scoreboard; no boundary or tracking confirmation'))
write('human_review_log.csv', fields, rows)

(OUT/'INITIAL_TEAM_SIDE_ANCHOR.md').write_text('''# 試合開始時のチームと側の対応

2026-10-03、ユーザーが最初のポイントの勝者を「手前」と回答。
映像：001_first_point_side_anchor_h264.mp4（元動画0〜14秒、H.264+AAC）。
放送表示はF539でポイント0-0、F659・F779で下段のみ0→1。
下段チーム名は上岡俊介・丸山海斗、上段は橋場柊一郎・菊山太陽。

したがって初期対応は手前＝上岡・丸山、奥＝橋場・菊山。
これは人手による得点側とAIによる表示目視を組み合わせた対応であり、チーム名への独立した人手回答ではない。

044開始時はゲーム3-4、ポイント0-0（F114635）。通常ゲームの第8ゲームで、規則上のチェンジは第1・3・5・7ゲーム後の4回。カメラ側が保たれれば初期対応に戻る。044の人手回答「手前」と下段0→1は整合する。旧AIの奥ラベルはこの証拠に反する。
ただし全ゲーム間の映像切替・カメラ方向を連続監査したわけではないため、規則だけを全ラリーへの確定ラベル伝播に使わない。044の既存台帳のチーム側対応は、その監査完了まで保留する。
''', encoding='utf-8')
print('001 side recorded; initial team-side anchor derived; 044 rule consistency documented without global propagation')
