"""Render every requested prefix time with observed IDs; never fill missing boxes."""
import argparse,csv,hashlib,json,re
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'ball_quality_20261110/cohort_tracking'
SLOTS=['near_front','near_rear','far_rear','far_front'];NAMES=['N1','N2','F1','F2'];COLORS=[(80,255,80),(255,190,40),(255,80,240),(0,230,255)]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--variant');ap.add_argument('--case-dir');args=ap.parse_args();tag=args.tag;assert re.fullmatch(r'\d{3}',tag);case=BASE/f'case_{tag}'
    if args.case_dir:
        assert not args.variant;case=Path(args.case_dir).resolve();assert case.is_relative_to(BASE.parent.resolve())
    if args.variant:
        assert re.fullmatch(r'[A-Za-z0-9_]+',args.variant);case=case/args.variant
    out=case/'fixed_input_identity';out.mkdir();p=json.loads((case/'protocol.json').read_text(encoding='utf-8'));assert p['candidate_id'].endswith('_'+tag);byframe=defaultdict(dict)
    with (case/'four_player_path.csv').open(encoding='utf-8-sig',newline='') as f:
        for r in csv.DictReader(f):byframe[int(r['source_frame'])][r['player_slot']]=r
    times=p['requested_input_frames'];cap=cv2.VideoCapture(p['source_video']);pages={'far':[],'near':[]};boxes={'far':[1250,920,2650,1480],'near':[0,1050,3840,2160]}
    for f in times:
        cap.set(1,f);ok,im=cap.read();assert ok and round(cap.get(1))==f+1
        for i,s in enumerate(SLOTS):
            r=byframe[f].get(s)
            if not r:continue
            coords=[round(float(r[k])) for k in ['x1','y1','x2','y2']]
            cv2.rectangle(im,tuple(coords[:2]),tuple(coords[2:]),COLORS[i],5)
            cv2.putText(im,f'{NAMES[i]} raw{r["track_id"]}',(coords[0],max(35,coords[1]-10)),0,1.,COLORS[i],2,cv2.LINE_AA)
        for name,(x1,y1,x2,y2) in boxes.items():
            width=700;height=round(width*(y2-y1)/(x2-x1));thumb=cv2.resize(im[y1:y2,x1:x2],(width,height),interpolation=cv2.INTER_AREA)
            head=np.zeros((34,width,3),np.uint8);missing=','.join(NAMES[i] for i,s in enumerate(SLOTS) if s not in byframe[f]) or 'none'
            cv2.putText(head,f'{tag} F{f}; missing:{missing}',(10,25),0,.65,(255,255,255),1,cv2.LINE_AA);pages[name].append(np.vstack([head,thumb]))
    cap.release()
    for name,cells in pages.items():
        while len(cells)%4:cells.append(np.zeros_like(cells[0]))
        assert cv2.imwrite(str(out/f'{name}_all31.jpg'),np.vstack([np.hstack(cells[i:i+4]) for i in range(0,len(cells),4)]),[cv2.IMWRITE_JPEG_QUALITY,95])
    (out/'manifest.json').write_text(json.dumps(dict(requested_source_frames=times,native_crop_xyxy=boxes,aspect_preserved=True,source_path_sha256=sha(case/'four_player_path.csv'),images={p.name:sha(p) for p in out.glob('*.jpg')},script_sha256=sha(Path(__file__)),scope='All requested times, AI image evidence only; not independent identity GT.'),indent=2),encoding='utf-8')
    print(tag,'rendered all31 requested times, with observed IDs and missing slots.')
if __name__=='__main__':main()
