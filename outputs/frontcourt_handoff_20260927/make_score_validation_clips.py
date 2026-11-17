"""Render the frozen 16-candidate score-validation sample as H264 clips."""
import csv
import subprocess
from pathlib import Path

OUT=Path(__file__).resolve().parent
SOURCE=Path(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study\data\raw\match1.mp4')
FFMPEG=Path(r'C:\ffmpeg\bin\ffmpeg.exe')
with (OUT/'score_suggestion_validation_sample.csv').open(encoding='utf-8-sig',newline='') as file:
    sample=list(csv.DictReader(file))
for index,row in enumerate(sample,1):
    first=max(0,float(row['candidate_start_sec'])-1)
    last=float(row['candidate_end_sec'])+3
    destination=OUT/row['clip_file']
    command=[str(FFMPEG),'-hide_banner','-loglevel','error','-y',
             '-ss',f'{first:.6f}','-i',str(SOURCE),'-t',f'{last-first:.6f}',
             '-vf','scale=1920:1080','-c:v','libx264','-preset','veryfast',
             '-crf','27','-pix_fmt','yuv420p','-c:a','aac','-b:a','128k',
             '-movflags','+faststart',str(destination)]
    subprocess.run(command,check=True)
    print(index,len(sample),destination.name,destination.stat().st_size,flush=True)
