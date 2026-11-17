"""Make browser-friendly tracking previews around observed ID losses."""
import csv
import subprocess
from pathlib import Path

OUT=Path(__file__).resolve().parent
FFMPEG=Path(r'C:\ffmpeg\bin\ffmpeg.exe')
FPS=60000/1001
clips=(
    ('044',114835,114880,'far_front_initial_id_loss'),
    ('044',115200,115235,'far_rear_long_id_loss'),
    ('056',154120,154150,'far_rear_relink'),
    ('056',154355,154395,'far_front_long_id_loss'),
    ('056',154625,154665,'near_rear_frame_exit'),
)
rows=[]
for rally,first,last,reason in clips:
    folder=OUT/f'full_rally_{rally}'
    source_start=114710 if rally=='044' else 154086
    destination=OUT/f'full_rally_{rally}_{reason}_h264.mp4'
    command=[str(FFMPEG),'-hide_banner','-loglevel','error','-y',
             '-ss',f'{(first-source_start)/FPS:.6f}','-i',str(folder/'bytetrack.mp4'),
             '-frames:v',str(last-first+1),'-c:v','libx264','-preset','veryfast',
             '-crf','25','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)]
    subprocess.run(command,check=True)
    rows.append(dict(candidate_id=f'match1_rally_window_{rally}',
                     first_source_frame=first,last_source_frame=last,
                     clip_file=destination.name,review_reason=reason,
                     identity_review_status='not_independently_confirmed'))
with (OUT/'full_rally_clip_review_queue.csv').open('w',encoding='utf-8-sig',newline='') as file:
    writer=csv.DictWriter(file,fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(len(rows),'H264 clips')
