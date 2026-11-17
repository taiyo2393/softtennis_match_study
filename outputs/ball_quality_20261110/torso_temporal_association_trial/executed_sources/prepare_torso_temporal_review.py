"""Inspect every changed row in the torso temporal trial, plus all previous failure probes."""
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,sha,dump
from diagnose_far_identity_failures import load
OUT=BASE/"torso_temporal_association_trial"

def main():
    stage=OUT;p=load(stage/'protocol.json');assert len(load(stage/'result.json')['cases'])==10;old=BASE/'far_identity_guard_development/matched_trial';out=stage/'all_change_review';out.mkdir();requests={};tables={};fields=['source_frame','player_slot','track_id','x1','y1','x2','y2'];key=lambda row:tuple(row[k] for k in fields)
    for e in p['cases']:
        tag=e['tag'];before=read(old/f'case_{tag}/baseline/far_pair_path.csv');after=read(stage/f'case_{tag}/torso_team_soft_cost/far_pair_path.csv');bm={(int(r['source_frame']),r['player_slot']):r for r in before};nm={(int(r['source_frame']),r['player_slot']):r for r in after};tables[tag]=(bm,nm);bk={key(r) for r in before};nk={key(r) for r in after};assert not nk-bk,'Review additions explicitly if future rule creates them.'
        for r in before:
            if key(r) not in nk:requests[tag,int(r['source_frame']),r['player_slot']]='all removed observations; no replacement'
    for r in read(BASE/'pair_initialization_transfer_019/AI_failure_probes.csv'):requests.setdefault(('019',int(r['source_frame']),r['player_slot']),'prior failure/ambiguous probe retained')
    for tag,f,slot in [('016',40625,'far_rear'),('016',40626,'far_rear'),('019',46352,'far_rear'),('002',2319,'far_front'),('009',21945,'far_front'),('009',21975,'far_front'),('006',12421,'far_rear'),('006',12422,'far_rear')]:requests.setdefault((tag,f,slot),'prior true-player/color-counterexample; check retained')
    source=load(BASE/'cohort_tracking/case_019/protocol.json')['source_video'];cap=cv2.VideoCapture(source);items=[];cells=[]
    for i,((tag,f,slot),reason) in enumerate(sorted(requests.items()),1):
        bm,nm=tables[tag];row=bm[f,slot];new=nm.get((f,slot));x1=max(0,round(float(row['x1']))-210);x2=min(3840,round(float(row['x2']))+210);y1=max(0,round(float(row['y1']))-160);y2=min(2160,round(float(row['y2']))+160)
        cap.set(1,f);ok,im=cap.read();assert ok and round(cap.get(1))==f+1
        bb=[round(float(row[k])) for k in ['x1','y1','x2','y2']];color=(70,70,255) if new is None else (80,255,80);cv2.rectangle(im,tuple(bb[:2]),tuple(bb[2:]),color,5);cv2.putText(im,f'{slot} raw{row["track_id"]}',(bb[0],max(40,bb[1]-12)),0,1.,color,2)
        crop=im[y1:y2,x1:x2];scale=min(900/crop.shape[1],500/crop.shape[0]);small=cv2.resize(crop,(round(crop.shape[1]*scale),round(crop.shape[0]*scale)));canvas=np.zeros((570,900,3),np.uint8);dx=(900-small.shape[1])//2;dy=70+(500-small.shape[0])//2;canvas[dy:dy+small.shape[0],dx:dx+small.shape[1]]=small
        cv2.putText(canvas,f'{tag} F{f} {slot}: '+('REMOVED red' if new is None else 'RETAINED green'),(10,27),0,.65,(255,255,255),1);cv2.putText(canvas,reason,(10,55),0,.59,(255,255,255),1)
        file=out/f'{i:02d}_{tag}_F{f}.jpg';assert cv2.imwrite(str(file),canvas);items.append(dict(tag=tag,source_frame=f,slot=slot,reason=reason,removed=new is None,image=file.name,sha256=sha(file),crop_xyxy=[x1,y1,x2,y2],aspect_preserved=True));cells.append(canvas)
    cap.release()
    for page,start in enumerate(range(0,len(cells),6),1):
        group=cells[start:start+6]
        while len(group)%2:group.append(np.zeros_like(cells[0]))
        assert cv2.imwrite(str(out/f'page_{page:02d}.jpg'),np.vstack([np.hstack(group[i:i+2]) for i in range(0,len(group),2)]))
    dump(out/'manifest.json',dict(selection='Every changed row in all10 cases, plus retained earlier AI failure/ambiguous probes. No success-only display.',items=items,script_sha256=sha(Path(__file__)),independent_GT=False));print([(r['tag'],r['source_frame'],r['slot'],r['removed']) for r in items])

if __name__=='__main__':main()
