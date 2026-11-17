"""Show both benefit and counterexample with identical source frames and aspect."""
import subprocess
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,sha,dump
from diagnose_far_identity_failures import load
OUT=BASE/"far_spatial_guard_development"

def main():
    stage=OUT;p=load(stage/'protocol.json');dest=OUT/'comparison_review';dest.mkdir();plan=dict(segments=[dict(tag='006',first=12410,last=12432,repeat=3,label='Official+actual player mixed detection removed'),dict(tag='019',first=46480,last=46540,repeat=3,label='F1 official / mixed box'),dict(tag='019',first=46620,last=46700,repeat=3,label='F2 crossing'),dict(tag='016',first=40625,last=40625,repeat=120,label='Previous false rejection avoided; still image')],crop_native_xyxy=[1200,970,2800,1900],source_fps=60000/1001,output_dimensions=[1920,720],dynamic_speed_fraction=1/3,control_still_duration_seconds=120/(60000/1001),aspect_preserved=True,selection='All changed-row neighborhoods006/019, remaining019 failures, and the previous016 false rejection. Development selection, not accuracy sampling.',script_sha256=sha(Path(__file__)));dump(dest/'selection.json',plan)
    source=load(BASE/'cohort_tracking/case_019/protocol.json')['source_video'];cap=cv2.VideoCapture(source);tables={};slots=['near_front','near_rear','far_rear','far_front'];colors=[(80,255,80),(255,190,40),(255,80,240),(0,230,255)];names=['N1','N2','F1','F2']
    for tag in ['006','016','019']:
        e=next(e for e in p['cases'] if e['tag']==tag);near=[r for r in read(Path(e['temporal_folder'])/'selected_four.csv') if r['player_slot'].startswith('near_')];tables[tag]=[]
        for rs in [near+read(BASE/f'far_identity_guard_development/matched_trial/case_{tag}/baseline/far_pair_path.csv'),near+read(stage/f'case_{tag}/spatial_background_and_near/far_pair_path.csv')]:
            groups=defaultdict(list)
            for row in rs:groups[int(row['source_frame'])].append(row)
            tables[tag].append(groups)
    temp=dest/'comparison_mp4v.mp4';writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),plan['source_fps'],(1920,720));assert writer.isOpened();count=0;QA=[]
    for segment in plan['segments']:
        tag=segment['tag'];cap.set(1,segment['first'])
        for f in range(segment['first'],segment['last']+1):
            ok,raw=cap.read();assert ok and round(cap.get(1))==f+1;canvas=np.zeros((720,1920,3),np.uint8)
            cv2.putText(canvas,f'{tag} source F{f}: {segment["label"]}',(15,26),0,.66,(255,255,255),1);cv2.putText(canvas,'BASELINE',(15,61),0,.75,(255,255,255),2);cv2.putText(canvas,'SPATIAL TRIAL: known failures remain',(975,61),0,.70,(255,255,255),2)
            for side,groups in enumerate(tables[tag]):
                im=raw.copy()
                for r in groups[f]:
                    i=slots.index(r['player_slot']);bb=[round(float(r[k])) for k in ['x1','y1','x2','y2']];cv2.rectangle(im,tuple(bb[:2]),tuple(bb[2:]),colors[i],5);cv2.putText(im,f'{names[i]} raw{r["track_id"]}',(bb[0],max(40,bb[1]-12)),0,1.,colors[i],2)
                x1,y1,x2,y2=plan['crop_native_xyxy'];crop=cv2.resize(im[y1:y2,x1:x2],(960,558),interpolation=cv2.INTER_AREA);canvas[100:658,side*960:(side+1)*960]=crop
            cv2.putText(canvas,'1/3 speed; no coordinate interpolation' if tag!='016' else '016 retained actual player: still image, 2 seconds',(15,699),0,.68,(255,255,255),1)
            if (tag=='019' and f in [46503,46511,46665]) or (tag=='006' and f==12421) or tag=='016':QA.append(cv2.resize(canvas,(960,360)))
            for repeat in range(segment['repeat']):writer.write(canvas);count+=1
    cap.release();writer.release();assert count==615 and len(QA)==5
    out=dest/'before_after_spatial_guard_h264.mp4';subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-i',str(temp),'-an','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],check=True)
    image=dest/'video_QA.jpg';assert cv2.imwrite(str(image),np.vstack(QA));cap=cv2.VideoCapture(str(out));n=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(720,1920);n+=1
    cap.release();assert n==count;dump(dest/'verification.json',dict(status='passed',video_decoded_frames=n,dimensions=[1920,720],aspect_preserved=True,video_sha256=sha(out),QA_image_sha256=sha(image),selection_manifest_sha256=sha(dest/'selection.json'),corrected_complete_tracking=False));print(n,'comparison frames decoded; unresolved failures included.')

if __name__=='__main__':main()
