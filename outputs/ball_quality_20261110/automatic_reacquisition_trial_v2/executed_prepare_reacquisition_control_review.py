"""Draw frozen control transitions before interpreting guard availability changes."""
import csv,json,hashlib
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent;OUT=ROOT.parent/'ball_quality_20261110/automatic_reacquisition_trial_v2'
SLOTS=['near_front','near_rear','far_rear','far_front'];COLORS=[(80,255,80),(255,190,40),(255,80,240),(0,230,255)]
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf-8'));out=OUT/'control_image_review';out.mkdir();manifest=[]
    cap=cv2.VideoCapture(p['source_video'])
    for ev in p['control_switches']:
        tag=ev['tag'];slot=ev['slot'];frame=ev['frame'];trace=next(r for r in read(OUT/f'case_{tag}/baseline/near_transition_trace.csv') if int(r['source_frame'])==frame and r['player_slot']==slot)
        plan=next(c for c in p['cases'] if c['tag']==tag);before=int(trace['old_observed_frame']);frames=[max(plan['first'],before-12),before,frame,frame+7,frame+13,min(plan['last'],frame+30)]
        for mode in ['baseline','confirm_new_id_opponent_negatives']:
            people=defaultdict(dict)
            for r in read(OUT/f'case_{tag}/{mode}/selected_four.csv'):people[int(r['source_frame'])][r['player_slot']]=r
            cells=[]
            for f in frames:
                cap.set(1,f);ok,im=cap.read();assert ok and round(cap.get(1))==f+1
                for s,r in people[f].items():
                    if not s.startswith('near_'):continue
                    i=SLOTS.index(s);b=[round(float(r[k])) for k in ['x1','y1','x2','y2']];cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),COLORS[i],6);cv2.putText(im,['N1','N2','F1','F2'][i]+' raw'+r['track_id'],(b[0],max(35,b[1]-12)),0,1.2,COLORS[i],3)
                thumb=cv2.resize(im,(960,540),interpolation=cv2.INTER_AREA);head=np.zeros((34,960,3),np.uint8);cv2.putText(head,f'{tag} {mode[:13]} F{f} target present:{slot in people[f]}',(10,25),0,.72,(255,255,255),1);cells.append(np.vstack([head,thumb]))
            file=out/f'{tag}_{slot}_{frame}_{mode}.jpg';cv2.imwrite(str(file),np.vstack([np.hstack(cells[:3]),np.hstack(cells[3:])]))
            manifest.append(dict(control=ev,mode=mode,frames=frames,image_file=file.name,sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    cap.release();(out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8');print('Saved all four frozen control transition sheets, baseline and guardC.')
if __name__=='__main__':main()
