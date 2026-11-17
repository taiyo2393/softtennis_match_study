"""Inventory final-path gaps; render deterministic longest-gap diagnostics."""
import argparse,csv,hashlib,json
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'ball_quality_20261110/cohort_tracking'
SLOTS=['near_front','near_rear','far_rear','far_front'];NAMES=['N1','N2','F1','F2'];COLORS=[(80,255,80),(255,190,40),(255,80,240),(0,230,255)]
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,rows):
    if not rows:return
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);tag=ap.parse_args().tag
    case=BASE/f'case_{tag}';out=case/'gap_diagnostics';out.mkdir();p=json.loads((case/'protocol.json').read_text(encoding='utf-8'))
    result=json.loads((case/'case_result.json').read_text(encoding='utf-8'));rows=read(case/'four_player_path.csv')
    byslot=defaultdict(list);byframe=defaultdict(dict)
    for r in rows:byslot[r['player_slot']].append(r);byframe[int(r['source_frame'])][r['player_slot']]=r
    gaps=[];counts=[];chosen=[]
    for slot in SLOTS:
        rs=sorted(byslot[slot],key=lambda r:int(r['source_frame']));gs=[];changes=0
        for a,b in zip(rs,rs[1:]):
            before,after=int(a['source_frame']),int(b['source_frame']);changes+=a['track_id']!=b['track_id']
            if after-before>1:gs.append(dict(player_slot=slot,before_source_frame=before,after_source_frame=after,
                missing_frames=after-before-1,before_raw_id=a['track_id'],after_raw_id=b['track_id']))
        gaps+=gs;chosen+=sorted(gs,key=lambda r:(-r['missing_frames'],r['before_source_frame']))[:2]
        counts.append(dict(player_slot=slot,observations=len(rs),raw_id_count=len({r['track_id'] for r in rs}),raw_id_changes=changes,gap_runs=len(gs)))
    write(out/'all_final_path_gap_runs.csv',gaps);write(out/'slot_diagnostic_counts.csv',counts)
    dump(out/'protocol.json',dict(selection='Largest two post-initialization observation gaps per slot, ties earliest first. Diagnostic selection only; not representative identity-accuracy sampling.',
        source_path=str(case/'four_player_path.csv'),source_sha256=sha(case/'four_player_path.csv'),
        metrics='Detector raw-ID changes are not person-ID switches. Gap runs do not distinguish physical occlusion, detection loss, or assignment veto.',
        coordinate_policy='No path/coordinates changed; no interpolation or identity GT generated.',script_sha256=sha(Path(__file__))))
    cap=cv2.VideoCapture(p['source_video']);manifest=[]
    for index,g in enumerate(chosen,1):
        slot=g['player_slot'];before=g['before_source_frame'];after=g['after_source_frame']
        anchors=[byframe[before][slot],byframe[after][slot]]
        x1=max(0,int(min(float(r['x1']) for r in anchors))-180);y1=max(0,int(min(float(r['y1']) for r in anchors))-140)
        x2=min(3840,int(max(float(r['x2']) for r in anchors))+180);y2=min(2160,int(max(float(r['y2']) for r in anchors))+140)
        frames=[max(p['context_first_source_frame'],before-12),before,(before+after)//2,after,min(p['context_last_source_frame'],after+6),min(p['context_last_source_frame'],after+12)]
        cells=[]
        for f in frames:
            cap.set(1,f);ok,im=cap.read();assert ok and int(round(cap.get(1)))==f+1
            for s,r in byframe[f].items():
                i=SLOTS.index(s);box=[round(float(r[k])) for k in ['x1','y1','x2','y2']]
                cv2.rectangle(im,tuple(box[:2]),tuple(box[2:]),COLORS[i],6 if s==slot else 3)
                cv2.putText(im,f'{NAMES[i]} raw{r["track_id"]}',(box[0],max(35,box[1]-12)),0,1.1,COLORS[i],2,cv2.LINE_AA)
            crop=im[y1:y2,x1:x2];scale=min(800/crop.shape[1],450/crop.shape[0]);small=cv2.resize(crop,(round(crop.shape[1]*scale),round(crop.shape[0]*scale)),interpolation=cv2.INTER_AREA)
            cell=np.zeros((450,800,3),np.uint8);dy=(450-small.shape[0])//2;dx=(800-small.shape[1])//2;cell[dy:dy+small.shape[0],dx:dx+small.shape[1]]=small
            head=np.zeros((38,800,3),np.uint8);cv2.putText(head,f'{tag} {NAMES[SLOTS.index(slot)]} F{f} | {g["missing_frames"]} missing frames',(12,27),0,.73,(255,255,255),1,cv2.LINE_AA)
            cells.append(np.vstack([head,cell]))
        path=out/f'gap_{index:02d}_{slot}.jpg';assert cv2.imwrite(str(path),np.vstack([np.hstack(cells[:3]),np.hstack(cells[3:])]),[cv2.IMWRITE_JPEG_QUALITY,95])
        manifest.append(dict(g,image_file=path.name,source_frames=frames,constant_crop_xyxy=[x1,y1,x2,y2],aspect_preserved=True,sha256=sha(path)))
    cap.release();dump(out/'image_manifest.json',manifest)
    times=p['requested_input_frames'];init=result['initialized_frame']
    missing=[f for f in times if f<init]
    assert all(not byframe[f] for f in missing)
    dump(out/'initialization_diagnosis.json',dict(initialization_source_frame=init,initialization_confirmation_window=[init-7,init],
        requested_frames_before_initialization=missing,requested_times_before_initialization=len(missing),
        coordinate_rows_at_these_requested_times=0,
        explanation='Four-person joint initialization with eight consecutive source frames delays all four output slots, including visibly present near players. These are unassigned times, not physical player absence.',
        input_times_not_shifted=True,retroactive_fill_performed=False))
    print(json.dumps(dict(counts=counts,diagnostic_sheets=len(manifest),requested_times_before_initialization=len(missing)),ensure_ascii=False))

if __name__=='__main__':main()
