"""Create a compact clickable index for the frozen score-validation clips."""
import csv
import random
from pathlib import Path

OUT=Path(__file__).resolve().parent
with (OUT/'score_suggestion_validation_sample.csv').open(encoding='utf-8-sig',newline='') as file:
    sample=list(csv.DictReader(file))
blind=list(sample)
random.Random(20260928).shuffle(blind)
blind_rows=[]
for order,row in enumerate(blind,1):
    blind_rows.append(dict(review_order=order,candidate_id=row['candidate_id'],
                           clip_file=row['clip_file'],
                           candidate_start_video_sec='1.000',
                           candidate_end_video_sec=f"{1+float(row['candidate_end_sec'])-float(row['candidate_start_sec']):.3f}",
                           actual_score_change_row='',
                           score_change_linked_to_candidate_point='',
                           winner_team='',winner_camera_side='',
                           start_confirmed='',point_end_confirmed='',
                           reviewer='',review_status='unreviewed'))
with (OUT/'score_suggestion_blind_review.csv').open('w',encoding='utf-8-sig',newline='') as file:
    writer=csv.DictWriter(file,fieldnames=list(blind_rows[0]))
    writer.writeheader()
    writer.writerows(blind_rows)
lines=[
    '# 得点表示候補のブラインド映像確認（16件）',
    '',
    '4層（境界complete/未解決 × 単純な1点増加提案あり/なし）から固定乱数20260927で各4件を抽出し、別の固定乱数20260928で確認順をシャッフルした。下の一覧には提案した増点行や層を表示しない。',
    '',
    '各動画は候補開始1秒前から候補終了3秒後までの連続映像。H.264映像・AAC音声、1920×1080、縦横比16:9。候補窓に複数ポイントやリプレイが混ざる場合は `linked_to_this_rally=unknown` とする。',
    '',
    '確認順：①候補窓が1ポイントに対応するか、②得点成立事象と表示更新がそのポイントに対応するか、③増点行とチーム、④チームの手前/奥。接触・終了のフレームは別途連続映像で標定する。',
    '',
    '|順番|候補|候補窓の動画内時刻|連続動画|',
    '|---:|---|---|---|',
]
for row in blind_rows:
    suffix=row['candidate_id'][-3:]
    path='/' + (OUT/row['clip_file']).as_posix()
    lines.append(f"|{row['review_order']}|{suffix}|{row['candidate_start_video_sec']}–{row['candidate_end_video_sec']}秒|[H.264動画]({path})|")
lines += ['',f"判定欄は [score_suggestion_blind_review.csv](/{(OUT/'score_suggestion_blind_review.csv').as_posix()})。各動画で①増点行（top/bottom/none/unclear）、②表示更新が候補のポイントに対応するか（yes/no/unknown）、③必要ならチームと画面側を記入する。提案値はこの票には含めない。現時点では全16件とも未確認。",'']
(OUT/'SCORE_REVIEW_INDEX.md').write_text('\n'.join(lines),encoding='utf-8')
print('wrote',len(sample),'links')
