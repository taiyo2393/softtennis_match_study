"""Render continuous 16:9 H264 contexts for three scoreboard suggestions."""
import csv
import subprocess
from pathlib import Path

OUT=Path(__file__).resolve().parent
SOURCE=Path(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study\data\raw\match1.mp4')
FFMPEG=Path(r'C:\ffmpeg\bin\ffmpeg.exe')
FPS=60000/1001
with (OUT/'rally_label_review.csv').open(encoding='utf-8-sig',newline='') as file:
    candidates={row['candidate_id'][-3:]:row for row in csv.DictReader(file)}
with (OUT/'score_change_suggestions.csv').open(encoding='utf-8-sig',newline='') as file:
    suggestions={row['candidate_id'][-3:]:row for row in csv.DictReader(file)}
rows=[]
for name in ('002','010','027'):
    candidate=candidates[name]
    suggestion=suggestions[name]
    first=max(0,float(candidate['candidate_start_sec'])-1)
    last=float(candidate['candidate_end_sec'])+3
    destination=OUT/f'{name}_score_point_context_h264.mp4'
    command=[str(FFMPEG),'-hide_banner','-loglevel','error','-y',
             '-ss',f'{first:.6f}','-i',str(SOURCE),'-t',f'{last-first:.6f}',
             '-vf','scale=1920:1080','-c:v','libx264','-preset','veryfast',
             '-crf','27','-pix_fmt','yuv420p','-c:a','aac','-b:a','128k',
             '-movflags','+faststart',str(destination)]
    subprocess.run(command,check=True)
    rows.append(dict(candidate_id=candidate['candidate_id'],
                     clip_file=destination.name,
                     approximate_first_source_frame=round(first*FPS),
                     approximate_last_source_frame=round(last*FPS),
                     suggested_score_row=suggestion['suggested_scoreboard_row'],
                     score_before_end=suggestion['score_before_end'],
                     score_after_1s=suggestion['score_after_1s'],
                     linked_to_this_rally='',winner_camera_side='',
                     start_confirmed='',point_end_confirmed='',
                     review_status='requires_video_review_no_outcome_label'))
with (OUT/'score_point_context_review_queue.csv').open('w',encoding='utf-8-sig',newline='') as file:
    writer=csv.DictWriter(file,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print('wrote',len(rows),'clips')
