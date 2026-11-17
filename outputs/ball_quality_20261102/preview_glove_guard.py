"""Apply frozen guard on two full local clips; retain original observed coordinates."""
import csv
import argparse
import json
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from evaluate_glove_guard import ROOT,OUT,SOURCE,BASE,read,features,sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--rerender',action='store_true');args=parser.parse_args()
    from ultralytics import YOLO
    choice=json.loads((OUT/'guard_selection.json').read_text());threshold=choice['selected_threshold']
    assert threshold==.12
    net=YOLO(str(SOURCE/'models/yolo11m-pose.pt')) if not args.rerender else None
    results=[]
    for cid in ('match1_motion_pilot_013','match1_motion_pilot_018'):
        dest=OUT/'continuous_predictions'/cid;dest.mkdir(parents=True,exist_ok=True)
        assert args.rerender or not (dest/'balltrack_predictions.csv').exists()
        cached_flags={}
        if args.rerender:
            previous=json.loads((dest/'report.json').read_text())
            (dest/'report_before_render_fix.json').write_text(json.dumps(previous,indent=2),encoding='utf-8')
            cached_flags={r['local_frame']:r['evidence'] for r in previous['flags']}
        rows=read(BASE/cid/'balltrack_predictions.csv');output=[];flags=[]
        base=SOURCE/'data/balltrack/racketvision/match1'/cid
        meta=json.loads((base/'runs/threshold030_median120/run_metadata.json').read_text())
        video=dest/'guard_review_h264.mp4';started=time.perf_counter()
        log=dest/'ffmpeg.log'
        command=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1440x640',
                 '-r',str(meta['fps']),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
        with log.open('wb') as err:
            proc=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=err)
            for t,r in enumerate(rows):
                assert int(r['local_frame'])==t
                im=cv2.imread(str(base/'frames'/f'{t:06d}.jpg'));assert im is not None
                candidate=None;flag=False
                if int(r['visible']):
                    candidate=cached_flags.get(t) if args.rerender else features(net,im,float(r['x_px']),float(r['y_px']))['nearest_eligible_wrist']
                    flag=candidate is not None and candidate['distance_by_person_height']<=threshold
                p=dict(r,raw_x_px=r['x_px'],raw_y_px=r['y_px'],raw_visible=r['visible'],raw_confidence=r['confidence'],
                       glove_guard_flag=int(flag),guard_threshold=threshold,
                       guard_state='withheld_uncertain' if flag else 'observed' if int(r['visible']) else 'missing')
                if flag:
                    p.update(x_px=0,y_px=0,visible=0,confidence=0,position_source='withheld_blue_raised_wrist_candidate')
                    flags.append(dict(local_frame=t,raw_x_px=r['x_px'],raw_y_px=r['y_px'],evidence=candidate))
                else:
                    assert all(p[k]==r[k] for k in ('x_px','y_px','visible','confidence'))
                output.append(p)
                canvas=np.full((640,1440,3),22,np.uint8)
                canvas[74:614,:960]=cv2.resize(im,(960,540),interpolation=cv2.INTER_AREA)
                # Fixed ROI shows the official across this whole clip, independent of labels.
                left,top=480,420
                canvas[74:344,960:1440]=im[top:top+270,left:left+480]
                cv2.putText(canvas,f'{cid[-3:]}  frame {t} | frozen experimental glove guard', (12,27),cv2.FONT_HERSHEY_SIMPLEX,.7,(240,240,240),1,cv2.LINE_AA)
                cv2.putText(canvas,'Green=retained detection | orange X=withheld candidate | right=native-scale fixed crop', (12,55),cv2.FONT_HERSHEY_SIMPLEX,.65,(240,240,240),1,cv2.LINE_AA)
                status='WITHHELD: blue raised wrist' if flag else 'DETECTION RETAINED' if int(p['visible']) else 'NO BASELINE DETECTION'
                for j,s in enumerate((status,'No recovered ball position.','Raw coordinates preserved.','Experimental; not adopted.','No truth markers in video.')):
                    cv2.putText(canvas,s,(972,380+j*34),cv2.FONT_HERSHEY_SIMPLEX,.53,(220,220,220),1,cv2.LINE_AA)
                if int(r['visible']):
                    x=round(float(r['x_px']));y=round(float(r['y_px']));color=(0,175,255) if flag else (80,255,100)
                    points=[((x//2,74+y//2),14)]
                    if left<=x<left+480 and top<=y<top+270:points.append(((960+x-left,74+y-top),18))
                    for point,size in points:
                        if flag:cv2.drawMarker(canvas,point,color,cv2.MARKER_TILTED_CROSS,size,2)
                        else:cv2.circle(canvas,point,7,color,1)
                proc.stdin.write(canvas.tobytes())
                if t%120==0:print(cid,t,'/',len(rows),flush=True)
            proc.stdin.close();assert proc.wait()==0,log.read_text()
        with (dest/'balltrack_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
        cap=cv2.VideoCapture(str(video));decoded=0
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape[:2]==(640,1440);decoded+=1
        cap.release();assert decoded==len(rows)
        report=dict(candidate_id=cid,frames=len(rows),suppressed_frames=len(flags),flags=flags,seconds=time.perf_counter()-started,
                    original_csv_sha256=sha(BASE/cid/'balltrack_predictions.csv'),selection_sha256=sha(OUT/'guard_selection.json'),
                    full_video_decoded_frames=decoded,unchanged_rows_exactly_preserved=True,interpolation=False,
                    rerender_from_cached_flags=args.rerender,
                    warning='Continuous raw output generated but full-interval identity/localization accuracy not established; no active baseline overwrite.')
        (dest/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        results.append({k:v for k,v in report.items() if k!='flags'})
    (OUT/'continuous_preview_report.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
