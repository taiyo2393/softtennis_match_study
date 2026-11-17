"""Create diagnostic media. No source annotations or model outputs are changed."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parent
COMPARE=ROOT/'totnet_comparison'
OUT=COMPARE/'joint_failure_review'
SOURCE=Path(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study')
FFMPEG=Path(r'C:\ffmpeg\bin\ffmpeg.exe')


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def getcases():
    old=read(COMPARE/'test_frame_comparison.csv')
    bykey={(r['candidate_id'],r['local_frame']):r for r in old}
    cases=[]
    for i,r in enumerate(read(COMPARE/'joint_failure_review_candidates.csv'),1):
        cid=r['candidate_id'];t=int(r['local_frame'])
        base=SOURCE/'data/balltrack/racketvision/match1'/cid
        metadata=json.loads((base/'runs/threshold030_median120/run_metadata.json').read_text())
        case=dict(r,case_id=f'{i:02d}_{cid[-3:]}_f{t}',existing_notes=bykey[cid,r['local_frame']]['notes'],
                  fps=metadata['fps'],frame_count=metadata['frame_count'],
                  start_local_frame=max(0,t-30),end_local_frame=min(metadata['frame_count']-1,t+30))
        for name in ('baseline','totnet'):
            for key in ('x_px','y_px','visible'):case[f'{name}_{key}']=bykey[cid,r['local_frame']][f'{name}_{key}']
        cases.append(case)
    return cases


def frame(case,t):
    p=SOURCE/'data/balltrack/racketvision/match1'/case['candidate_id']/'frames'/f'{t:06d}.jpg'
    im=cv2.imread(str(p));assert im is not None,str(p)
    assert im.shape[:2]==(1080,1920),im.shape
    return im


def crop(case,im,w=320,h=180):
    x=max(0,min(im.shape[1]-w,round(float(case['existing_truth_x_px']))-w//2))
    y=max(0,min(im.shape[0]-h,round(float(case['existing_truth_y_px']))-h//2))
    return im[y:y+h,x:x+w].copy(),(x,y)


def sheets(cases):
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',18)
    for page in range(9):
        canvas=Image.new('RGB',(1600,775),'#171b24');d=ImageDraw.Draw(canvas)
        d.text((10,5),'Joint failures: raw native-scale crops; columns t-12, t-6, t, t+6, t+12 (59.94 fps)',font=font,fill='white')
        d.text((10,30),'Crop centered on existing label. Target column has border; no model overlays. AI diagnostic, not independent ground truth.',font=font,fill='white')
        for j,case in enumerate(cases[page*3:page*3+3]):
            t=int(case['local_frame']);top=70+j*235
            d.text((5,top),f"{case['case_id']} | baseline={case['baseline_status']} | TOTNet={case['totnet_status']}",font=font,fill='white')
            for col,delta in enumerate((-12,-6,0,6,12)):
                f=max(0,min(case['frame_count']-1,t+delta));im=frame(case,f);patch,_=crop(case,im)
                x=col*320;y=top+28
                canvas.paste(Image.fromarray(cv2.cvtColor(patch,cv2.COLOR_BGR2RGB)),(x,y))
                d.text((x+5,y+2),str(f),font=font,fill='#00ff88',stroke_width=1,stroke_fill='black')
                if delta==0:d.rectangle((x,y,x+319,y+179),outline='#ffaa00',width=3)
        canvas.save(OUT/f'temporal_page_{page+1:02d}.jpg',quality=94)


def video(case):
    target=int(case['local_frame']);fps=case['fps']
    destination=OUT/'videos'/f"{case['case_id']}.mp4";destination.parent.mkdir(exist_ok=True)
    cmd=[str(FFMPEG),'-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1440x640',
         '-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)]
    log=destination.with_suffix('.log')
    with log.open('wb') as err:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=err)
        count=0
        for slow in (False,True):
            for t in range(case['start_local_frame'],case['end_local_frame']+1):
                im=frame(case,t);patch,(cx,cy)=crop(case,im,480,270)
                canvas=np.full((640,1440,3),22,np.uint8)
                canvas[74:614,:960]=cv2.resize(im,(960,540),interpolation=cv2.INTER_AREA)
                canvas[74:344,960:1440]=patch
                text=lambda s,y:cv2.putText(canvas,s,(12,y),cv2.FONT_HERSHEY_SIMPLEX,.65,(235,235,235),1,cv2.LINE_AA)
                text(f"{case['case_id']}  | raw frames {'0.25x' if slow else '1x'}  | local frame {t} (target {target})",27)
                text('Target only: green=existing label, cyan=baseline, pink=TOTNet. Right crop is native scale.',55)
                for k,s in enumerate(('Markers on target frame only.','Green: existing annotation.',f"Baseline: {case['baseline_status']}",f"TOTNet: {case['totnet_status']}",'No coordinate relabeling.')):
                    cv2.putText(canvas,s,(973,380+34*k),cv2.FONT_HERSHEY_SIMPLEX,.52,(225,225,225),1,cv2.LINE_AA)
                if t==target:
                    x=round(float(case['existing_truth_x_px']));y=round(float(case['existing_truth_y_px']))
                    cv2.drawMarker(canvas,(x//2,74+y//2),(80,255,100),cv2.MARKER_CROSS,14,1)
                    cv2.drawMarker(canvas,(960+x-cx,74+y-cy),(80,255,100),cv2.MARKER_CROSS,18,1)
                    for name,color,radius in (('baseline',(255,220,0),7),('totnet',(203,102,255),5)):
                        if int(case[f'{name}_visible']):
                            px=round(float(case[f'{name}_x_px']));py=round(float(case[f'{name}_y_px']))
                            cv2.circle(canvas,(px//2,74+py//2),radius,color,1)
                            if 0<=px-cx<480 and 0<=py-cy<270:cv2.circle(canvas,(960+px-cx,74+py-cy),radius+2,color,1)
                    cv2.rectangle(canvas,(0,72),(1439,616),(0,180,255),2)
                repeats=4 if slow else 1
                if slow and t==target:repeats+=round(fps*.5)
                for _ in range(repeats):proc.stdin.write(canvas.tobytes());count+=1
        proc.stdin.close();code=proc.wait();assert code==0,log.read_text()
    # Full decode and frame count verification; rendering is not annotation accuracy.
    cap=cv2.VideoCapture(str(destination));decoded=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(640,1440);decoded+=1
    cap.release();assert decoded==count,(decoded,count)
    return dict(case_id=case['case_id'],video=str(destination.resolve()),fps=fps,decoded_frames=decoded,
                expected_frames=count,full_decode_verified=True,sha256=hashlib.sha256(destination.read_bytes()).hexdigest())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['sheets','videos']);parser.add_argument('--cases',nargs='*')
    args=parser.parse_args();OUT.mkdir(exist_ok=True)
    cases=getcases();(OUT/'cases.json').write_text(json.dumps(cases,indent=2,ensure_ascii=False),encoding='utf-8')
    if args.mode=='sheets':sheets(cases)
    else:
        chosen=[c for c in cases if not args.cases or c['case_id'] in args.cases]
        reports=[]
        for c in chosen:
            reports.append(video(c));print('rendered and decoded',c['case_id'],flush=True)
        (OUT/'video_render_report.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')


if __name__=='__main__':main()
