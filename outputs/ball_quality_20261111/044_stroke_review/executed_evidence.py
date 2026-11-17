"""Raw event neighborhoods and AI-provisional stroke records, separate from GT."""
import sys, math, shutil, subprocess
from collections import defaultdict,Counter
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
import build_044_spatiotemporal_pilot as base

OUT=base.HERE/'044_stroke_review'
# Chosen after whole-interval uniform chronological inspection; not benchmark samples.
WINDOWS=[
 dict(id='B01',kind='bounce_candidate',first=115407,last=115423,roi=[940,560,1370,850]),
 dict(id='H02',kind='near_return_candidate',first=115433,last=115451,roi=[1050,480,1420,800]),
 dict(id='C02',kind='unclassified_turn_before_far_return',first=115501,last=115519,roi=[640,490,900,680]),
 dict(id='H03',kind='far_return_candidate',first=115515,last=115531,roi=[640,490,900,680]),
 dict(id='B02',kind='bounce_candidate',first=115563,last=115581,roi=[1130,570,1510,850]),
 dict(id='H04',kind='near_return_candidate',first=115591,last=115609,roi=[1460,520,1900,900]),
 dict(id='H05',kind='far_return_candidate',first=115721,last=115739,roi=[640,480,910,680])]

def evidence():
    assert not (OUT/'fine_review_plan.json').exists()
    base.dump(OUT/'fine_review_plan.json',dict(source='AI whole-interval chronological image inspection, after previous direction-change proposals',
        not_independent_evaluation=True,windows=WINDOWS,script_sha256=base.sha(Path(__file__)),
        policy='Source JPEGs only; raw crop preserves aspect; no ball marker or model boxes on fine reference images. No user truth created.'))
    manifest=[]
    for w in WINDOWS:
        left,top,right,bottom=w['roi'];ww,hh=right-left,bottom-top
        # Full-resolution crop, enlarged uniformly2x. Video retains every source frame.
        width=ww*2;height=hh*2+40
        proc=subprocess.Popen(['C:/ffmpeg/bin/ffmpeg.exe','-y','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s',f'{width}x{height}','-r',str(base.FPS/4),'-i','pipe:0','-an','-c:v','libx264','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/(w['id']+'_raw_quarter_speed.mp4'))],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
        stills=[]
        for f in range(w['first'],w['last']+1):
            path=base.FRAMES/f'{f-114635:06d}.jpg';im=cv2.imread(str(path));assert im is not None
            crop=im[top:bottom,left:right];large=cv2.resize(crop,(ww*2,hh*2),interpolation=cv2.INTER_LINEAR)
            header=np.zeros((40,width,3),np.uint8);cv2.putText(header,f'{w["id"]} raw F{f} / quarter speed',(8,28),0,.65,(255,255,255),1,cv2.LINE_AA)
            proc.stdin.write(np.vstack([header,large]).tobytes())
            manifest.append(dict(event_id=w['id'],source_frame=f,display_frame=f-w['first'],source_image=str(path),sha256=base.sha(path)))
            if (f-w['first'])%2==0:
                sh=np.zeros((30,ww,3),np.uint8);cv2.putText(sh,f'F{f}',(6,21),0,.5,(255,255,255),1,cv2.LINE_AA);stills.append(np.vstack([sh,crop]))
        proc.stdin.close();err=proc.stderr.read();assert proc.wait()==0,err
        while len(stills)%3:stills.append(np.zeros_like(stills[0]))
        cv2.imwrite(str(OUT/(w['id']+'_raw_fine_grid.jpg')),np.vstack([np.hstack(stills[i:i+3]) for i in range(0,len(stills),3)]))
    base.write(OUT/'fine_source_image_manifest.csv',manifest)
    print('Saved7raw event videos and full native-frame provenance; no new labels.',flush=True)

if __name__=='__main__':{'evidence':evidence}[sys.argv[1]]()
