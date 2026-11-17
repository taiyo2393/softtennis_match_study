"""Mux original match audio into existing frame-labelled H264 point clips."""
import json
import os
import subprocess
from pathlib import Path

OUT=Path(__file__).resolve().parent
SOURCE=Path(r'C:\Users\ta239\Research\softtennis_match_study\study\data\raw\match1.mp4')
FFMPEG=Path(r'C:\ffmpeg\bin\ffmpeg.exe')
FFPROBE=Path(r'C:\ffmpeg\bin\ffprobe.exe')
FPS=60000/1001

def probe(path):
    return json.loads(subprocess.check_output([
        str(FFPROBE),'-v','error','-show_entries',
        'stream=codec_type,codec_name,nb_frames','-of','json',str(path)],text=True))['streams']

for name,first,frames in (('044',114710,1111),('056',154050,891)):
    path=OUT/f'{name}_full_point_context_h264.mp4'
    temp=OUT/f'{name}_full_point_context_audio_tmp.mp4'
    original=probe(path)
    if any(stream['codec_type']=='audio' for stream in original):
        print(name,'already has audio')
        continue
    assert len(original)==1 and original[0]['codec_name']=='h264'
    assert int(original[0]['nb_frames'])==frames
    command=[str(FFMPEG),'-hide_banner','-loglevel','error','-y',
             '-i',str(path),'-ss',f'{first/FPS:.9f}',
             '-t',f'{frames/FPS:.9f}','-i',str(SOURCE),
             '-map','0:v:0','-map','1:a:0','-c:v','copy',
             '-c:a','aac','-b:a','128k','-movflags','+faststart',str(temp)]
    subprocess.run(command,check=True)
    streams=probe(temp)
    assert len(streams)==2 and streams[0]['codec_name']=='h264'
    assert int(streams[0]['nb_frames'])==frames
    assert streams[1]['codec_name']=='aac'
    os.replace(temp,path)
    print(name,frames,'frames + AAC')
