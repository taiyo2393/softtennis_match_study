"""Inspect every final-path gap after the inherited contact; no point-end GT assumed."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import cv2,numpy as np
from prepare_player_gap_diagnostics import ROOT,BASE,SLOTS,NAMES,COLORS,read,dump,sha

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);tag=ap.parse_args().tag
    assert len(tag)==3 and tag.isdigit()
    case=BASE/f'case_{tag}';out=case/'post_contact_gap_evidence';out.mkdir()
    p=json.loads((case/'protocol.json').read_text(encoding='utf-8'));rows=read(case/'four_player_path.csv');people=defaultdict(dict)
    for r in rows:people[int(r['source_frame'])][r['player_slot']]=r
    gaps=[r for r in read(case/'gap_diagnostics/all_final_path_gap_runs.csv') if int(r['after_source_frame'])>p['contact_candidate_last_source_frame']]
    cap=cv2.VideoCapture(p['source_video']);manifest=[]
    for index,g in enumerate(gaps,1):
        before,after=int(g['before_source_frame']),int(g['after_source_frame']);slot=g['player_slot'];anchors=[people[before][slot],people[after][slot]]
        x1=max(0,int(min(float(r['x1']) for r in anchors))-220);y1=max(0,int(min(float(r['y1']) for r in anchors))-160)
        x2=min(3840,int(max(float(r['x2']) for r in anchors))+220);y2=min(2160,int(max(float(r['y2']) for r in anchors))+160)
        frames=[max(p['context_first_source_frame'],before-12),before,(before+after)//2,after,min(p['context_last_source_frame'],after+6),min(p['context_last_source_frame'],after+12)];cells=[]
        for f in frames:
            cap.set(1,f);ok,im=cap.read();assert ok and round(cap.get(1))==f+1
            for s,r in people[f].items():
                i=SLOTS.index(s);coords=[round(float(r[k])) for k in ['x1','y1','x2','y2']]
                cv2.rectangle(im,tuple(coords[:2]),tuple(coords[2:]),COLORS[i],5)
                cv2.putText(im,f'{NAMES[i]} raw{r["track_id"]}',(coords[0],max(35,coords[1]-12)),0,1.1,COLORS[i],2,cv2.LINE_AA)
            crop=im[y1:y2,x1:x2];scale=min(800/crop.shape[1],450/crop.shape[0]);small=cv2.resize(crop,(round(crop.shape[1]*scale),round(crop.shape[0]*scale)),interpolation=cv2.INTER_AREA)
            cell=np.zeros((488,800,3),np.uint8);dy=38+(450-small.shape[0])//2;dx=(800-small.shape[1])//2;cell[dy:dy+small.shape[0],dx:dx+small.shape[1]]=small
            cv2.putText(cell,f'{tag} {NAMES[SLOTS.index(slot)]} F{f}; gap={g["missing_frames"]}',(12,27),0,.73,(255,255,255),1,cv2.LINE_AA);cells.append(cell)
        path=out/f'gap_{index:02d}_{slot}.jpg';assert cv2.imwrite(str(path),np.vstack([np.hstack(cells[:3]),np.hstack(cells[3:])]),[cv2.IMWRITE_JPEG_QUALITY,95])
        manifest.append(dict(g,image_file=path.name,source_frames=frames,constant_crop_xyxy=[x1,y1,x2,y2],aspect_preserved=True,sha256=sha(path)))
    cap.release();dump(out/'manifest.json',dict(selection='All final-path gaps whose next observation follows inherited contact; no formal point-end certification or accuracy sampling.',source_path_sha256=sha(case/'four_player_path.csv'),script_sha256=sha(Path(__file__)),gaps=manifest,interpolation=False))
    print(tag,'saved',len(manifest),'post-contact gap sheets')
if __name__=='__main__':main()
