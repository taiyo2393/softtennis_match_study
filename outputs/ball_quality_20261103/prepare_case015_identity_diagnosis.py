"""Freeze native-box evidence for the known referee initialization failure."""
import csv,hashlib,json
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'ball_quality_20261110';C=BASE/'cohort_tracking/case_015';O=C/'identity_diagnosis';O.mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=json.loads((C/'protocol.json').read_text(encoding='utf-8'));rawpath=C/'crop_runs/augmented_detections_015.csv';raw=list(csv.DictReader(rawpath.open(encoding='utf-8-sig')));byframe=defaultdict(dict)
for r in raw:
    if r['track_id'] in ['1','2','3'] and r['candidate_origin']=='original_full_frame':byframe[int(r['source_frame'])][r['track_id']]=r
frames=[38548,38786,38816,38846,38876,38906,39074,39100,39200];cap=cv2.VideoCapture(p['source_video']);cells=[]
for f in frames:
    cap.set(1,f);ok,im=cap.read();assert ok and round(cap.get(1))==f+1
    for tid,r in byframe[f].items():
        color={'1':(0,0,255),'2':(255,190,40),'3':(80,255,80)}[tid];b=[round(float(r[k])) for k in ['x1','y1','x2','y2']]
        cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),color,5);cv2.putText(im,f'native raw{tid}',(b[0],max(35,b[1]-10)),0,1.2,color,3)
    preview=cv2.resize(im,(1920,1080),interpolation=cv2.INTER_AREA);assert cv2.imwrite(str(O/f'F{f}.jpg'),preview)
    h=np.zeros((32,960,3),np.uint8);cv2.putText(h,f'015 F{f}: raw1 / raw2 / raw3',(10,24),0,.7,(255,255,255),1);cells.append(np.vstack([h,cv2.resize(preview,(960,540))]))
cap.release();assert cv2.imwrite(str(O/'native_correspondence_grid.jpg'),np.vstack([np.hstack(cells[i:i+3]) for i in range(0,len(cells),3)]))
(O/'manifest.json').write_text(json.dumps(dict(source_video_sha256=p['source_video_sha256'],native_detections_sha256=sha(rawpath),source_frames=frames,images={x.name:sha(x) for x in O.glob('*.jpg')},script_sha256=sha(Path(__file__)),scope='Native candidates for AI diagnosis, not a person identity truth dataset.'),indent=2),encoding='utf-8')
print('015 native correspondence evidence saved; no paths modified.')
