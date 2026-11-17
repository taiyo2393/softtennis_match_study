"""Create H.264 motion review clips for non-simple broadcast-score changes."""
import csv
import subprocess
from pathlib import Path

ROOT = Path(r'C:\Users\ta239\Research\softtennis_match_study\study')
OUT = Path(__file__).resolve().parent
FFMPEG = Path(r'C:\ffmpeg\bin\ffmpeg.exe')
with (OUT / 'score_change_suggestions.csv').open(encoding='utf-8-sig', newline='') as stream:
    suggestions = list(csv.DictReader(stream))
with (OUT / 'rally_label_review.csv').open(encoding='utf-8-sig', newline='') as stream:
    candidates = {r['candidate_id']: r for r in csv.DictReader(stream)}
review = []
for suggestion in suggestions:
    if suggestion['suggestion_reason'] == 'one_point_increment_consistent':
        continue
    candidate_id = suggestion['candidate_id']
    end_second = float(candidates[candidate_id]['candidate_end_sec'])
    clip_start = max(0, end_second - 3)
    clip_end = end_second + 5
    suffix = candidate_id[-3:]
    destination = OUT / f'uncertain_{suffix}_h264.mp4'
    command = [str(FFMPEG), '-hide_banner', '-loglevel', 'error', '-y',
               '-ss', f'{clip_start:.3f}', '-i', str(ROOT / 'data/raw/match1.mp4'),
               '-t', f'{clip_end-clip_start:.3f}',
               '-map', '0:v:0', '-map', '0:a:0?',
               '-vf', 'scale=1920:1080:flags=fast_bilinear',
               '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '25',
               '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '96k',
               '-movflags', '+faststart', str(destination)]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL)
    review.append(dict(candidate_id=candidate_id, clip_file=destination.name,
                       clip_start_sec=f'{clip_start:.3f}', clip_end_sec=f'{clip_end:.3f}',
                       score_suggestion_reason=suggestion['suggestion_reason'],
                       score_start=suggestion['score_start'],
                       score_before_end=suggestion['score_before_end'],
                       score_after_1s=suggestion['score_after_1s'],
                       score_after_3s=suggestion['score_after_3s'],
                       observed_broadcast_transition='',
                       corresponding_rally_id='', confirmed_winner_team='',
                       reviewer='', review_status='motion_review_pending'))

assert len(review) == 12
with (OUT / 'uncertain_score_motion_queue.csv').open('w', encoding='utf-8-sig', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(review[0]))
    writer.writeheader()
    writer.writerows(review)
print('H.264 review clips:', len(review))
