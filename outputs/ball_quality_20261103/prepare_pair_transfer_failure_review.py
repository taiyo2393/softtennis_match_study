"""Render known failure neighborhoods at preserved aspect and explicit slow speed."""
import json,subprocess
from pathlib import Path
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,sha,dump

def main():
    tag='019';c=BASE/f'cohort_tracking/case_{tag}';out=BASE/f'pair_initialization_transfer_{tag}'
    p=json.loads((c/'protocol.json').read_text(encoding='utf-8'));source=c/f'tracking_quality_20261102_stage7/players_{tag}_global/global_review_h264.mp4'
    plan=dict(source_video=str(source),source_video_sha256=sha(source),segments=[dict(first=46480,last=46540,label='A: inspect F1 near background official'),dict(first=46620,last=46700,label='B: inspect F2 at near-player crossing')],crop_on_1920x1080_video_xyxy=[600,480,1400,950],source_fps=60000/1001,repeated_output_frames_per_source=3,display_speed_fraction=1/3,aspect_preserved=True,selection='AI-diagnosed failure neighborhoods, not representative tracking-accuracy sampling',script_sha256=sha(Path(__file__)))
    assert not (out/'failure_review_manifest.json').exists();dump(out/'failure_review_selection.json',plan)
    cap=cv2.VideoCapture(str(source));temp=out/'failure_review_silent_mp4v.mp4';writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),plan['source_fps'],(1920,1080));assert writer.isOpened();cells=[];count=0
    for segment in plan['segments']:
        cap.set(1,segment['first']-p['context_first_source_frame'])
        for f in range(segment['first'],segment['last']+1):
            ok,im=cap.read();assert ok and im.shape[:2]==(1080,1920)
            x1,y1,x2,y2=plan['crop_on_1920x1080_video_xyxy'];crop=im[y1:y2,x1:x2];scale=min(1920/crop.shape[1],1020/crop.shape[0]);small=cv2.resize(crop,(round(crop.shape[1]*scale),round(crop.shape[0]*scale)))
            canvas=np.zeros((1080,1920,3),np.uint8);dx=(1920-small.shape[1])//2;dy=60+(1020-small.shape[0])//2;canvas[dy:dy+small.shape[0],dx:dx+small.shape[1]]=small
            cv2.putText(canvas,f'019 source F{f} | {segment["label"]} | 1/3 speed',(16,39),0,.84,(255,255,255),2,cv2.LINE_AA)
            if f in [46503,46511,46665,46685]:cells.append(cv2.resize(canvas,(960,540)))
            for repeat in range(3):writer.write(canvas);count+=1
    cap.release();writer.release();assert count==426 and len(cells)==4
    dest=out/'failure_review_h264.mp4';subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-i',str(temp),'-an','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(dest)],check=True)
    image=out/'failure_review_QA.jpg';assert cv2.imwrite(str(image),np.vstack([np.hstack(cells[:2]),np.hstack(cells[2:])]))
    cap=cv2.VideoCapture(str(dest));decoded=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(1080,1920);decoded+=1
    cap.release();assert decoded==count
    dump(out/'failure_review_manifest.json',dict(plan,output_frames=count,decoded_frames=decoded,dimensions=[1920,1080],video_sha256=sha(dest),QA_image_sha256=sha(image),QA_source_frames=[46503,46511,46665,46685],is_corrected_tracking_demonstration=False));print('426 failure-review frames decoded, aspect preserved, explicit 1/3 speed.')

if __name__=='__main__':main()
