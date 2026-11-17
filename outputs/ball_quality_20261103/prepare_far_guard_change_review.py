"""Deterministic changed-observation evidence; no identity GT is generated."""
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,sha,dump
from diagnose_far_identity_failures import OUT,load

def main():
    trial=OUT/'matched_trial';p=load(trial/'protocol.json');assert load(trial/'result.json')['baseline_exact_replays']==10;out=trial/'changed_observation_review';out.mkdir();requests=[];bytag={};fields=['source_frame','player_slot','track_id','x1','y1','x2','y2'];key=lambda r:tuple(r[k] for k in fields)
    for e in p['cases']:
        tag=e['tag'];b=read(trial/f'case_{tag}/baseline/far_pair_path.csv');new=read(trial/f'case_{tag}/slot_near_upper_guard/far_pair_path.csv');bm=defaultdict(dict);nm=defaultdict(dict);nk={key(r) for r in new};bk={key(r) for r in b}
        for r in b:bm[int(r['source_frame'])][r['player_slot']]=r
        for r in new:nm[int(r['source_frame'])][r['player_slot']]=r
        bytag[tag]=(bm,nm)
        for slot in ['far_rear','far_front']:
            lost=sorted(int(r['source_frame']) for r in b if r['player_slot']==slot and key(r) not in nk);runs=[]
            for f in lost:
                if runs and f==runs[-1][-1]+1:runs[-1].append(f)
                else:runs.append([f])
            if runs:
                run=sorted(runs,key=lambda r:(-len(r),r[0]))[0];requests.append(dict(tag=tag,source_frame=run[len(run)//2],slot=slot,reason=f'longest removed run {run[0]}..{run[-1]} ({len(run)})'))
        for r in new:
            if key(r) not in bk:requests.append(dict(tag=tag,source_frame=int(r['source_frame']),slot=r['player_slot'],reason='all newly selected observations'))
    for r in read(BASE/'pair_initialization_transfer_019/AI_failure_probes.csv'):requests.append(dict(tag='019',source_frame=int(r['source_frame']),slot=r['player_slot'],reason='previously AI-diagnosed failure/ambiguity probe'))
    source=load(BASE/'cohort_tracking/case_019/protocol.json')['source_video'];cap=cv2.VideoCapture(source);cells=[];manifest=[]
    for i,r in enumerate(requests,1):
        f=r['source_frame'];bm,nm=bytag[r['tag']];old=bm[f].get(r['slot']);new=nm[f].get(r['slot']);anchors=[row for row in [old,new] if row];assert anchors
        x1=max(0,round(min(float(a['x1']) for a in anchors))-200);x2=min(3840,round(max(float(a['x2']) for a in anchors))+200);y1=max(0,round(min(float(a['y1']) for a in anchors))-150);y2=min(2160,round(max(float(a['y2']) for a in anchors))+150)
        cap.set(1,f);ok,im=cap.read();assert ok and round(cap.get(1))==f+1
        for row,color,label in [(old,(70,70,255),'OLD'),(new,(80,255,80),'NEW')]:
            if row:
                bb=[round(float(row[k])) for k in ['x1','y1','x2','y2']];cv2.rectangle(im,tuple(bb[:2]),tuple(bb[2:]),color,5);cv2.putText(im,f'{label} {row["player_slot"]} raw{row["track_id"]}',(bb[0],max(40,bb[1]-12)),0,.95,color,2)
        crop=im[y1:y2,x1:x2];scale=min(900/crop.shape[1],500/crop.shape[0]);small=cv2.resize(crop,(round(crop.shape[1]*scale),round(crop.shape[0]*scale)));canvas=np.zeros((570,900,3),np.uint8);dx=(900-small.shape[1])//2;dy=70+(500-small.shape[0])//2;canvas[dy:dy+small.shape[0],dx:dx+small.shape[1]]=small
        cv2.putText(canvas,f'{r["tag"]} F{f} {r["slot"]}: old red / new green',(10,27),0,.65,(255,255,255),1);cv2.putText(canvas,r['reason'],(10,55),0,.58,(255,255,255),1)
        file=out/f'{i:02d}_{r["tag"]}_F{f}.jpg';assert cv2.imwrite(str(file),canvas);manifest.append(dict(r,image=file.name,sha256=sha(file),crop_xyxy=[x1,y1,x2,y2],aspect_preserved=True,new_observation_present=new is not None));cells.append(canvas)
    cap.release()
    for page,start in enumerate(range(0,len(cells),6),1):
        group=cells[start:start+6]
        while len(group)%2:group.append(np.zeros_like(cells[0]))
        assert cv2.imwrite(str(out/f'page_{page:02d}.jpg'),np.vstack([np.hstack(group[i:i+2]) for i in range(0,len(group),2)]))
    dump(out/'manifest.json',dict(selection='Longest removed run midpoint per slot/case plus all additions for upper guard, and all seven earlier019 probes. Diagnostic selection, not representative accuracy sample.',items=manifest,script_sha256=sha(Path(__file__)),independent_GT=False));print(len(manifest),'changed-observation evidence cells rendered.')

if __name__=='__main__':main()
