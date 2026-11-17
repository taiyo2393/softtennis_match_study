"""Display both fixed pose scales, including wrong native assignments."""
from pathlib import Path
import cv2,numpy as np
from run_body_part_localization_probe import OUT,SLOTS,load,box
from run_fixed_cohort_player_case_v3 import sha,dump

EDGES=[(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]
COLORS=[(80,255,80),(255,190,40),(255,80,240),(0,230,255)]
NAMES=['N1','N2','F1','F2']

def selected(item):
    return 'retained_known_development_example' in item['reasons'] or any(r.startswith('maximum_native_far_pair_overlap') for r in item['reasons'])

def main():
    p=load(OUT/'protocol.json');pred=load(OUT/'pose_predictions.json');lookup={(r['tag'],r['source_frame']):r for r in pred['frames']};dest=OUT/'AI_review';dest.mkdir(exist_ok=True);requests=[r for r in p['selected_frames'] if selected(r)];cells=[];manifest=[];cap=cv2.VideoCapture(p['source_video'])
    for i,item in enumerate(requests,1):
        tag,f=item['tag'],item['source_frame'];record=lookup[tag,f];rows=item['native_observations'];far=[r for r in rows if r['player_slot'].startswith('far_')]
        if 'retained_known_development_example' in item['reasons']:
            slot='far_front' if tag=='019' and f in [46665,46666] else 'far_rear';focus=[r for r in far if r['player_slot']==slot] or far
        else:focus=far
        if not focus:focus=[r for r in rows if r['player_slot']=='near_rear'] or rows
        assert focus
        bb=np.asarray([box(r) for r in focus]);l=max(0,int(bb[:,0].min())-180);t=max(0,int(bb[:,1].min())-120);r=min(3840,int(bb[:,2].max())+180);b=min(2160,int(bb[:,3].max())+200)
        cap.set(1,f);ok,raw=cap.read();assert ok and round(cap.get(1))==f+1;canvas=np.zeros((560,1600,3),np.uint8)
        for side,(mode,people) in enumerate(record['modes'].items()):
            im=raw.copy()
            for player in rows:
                idx=SLOTS.index(player['player_slot']);xy=[round(x) for x in box(player)];cv2.rectangle(im,tuple(xy[:2]),tuple(xy[2:]),COLORS[idx],3);cv2.putText(im,NAMES[idx]+f' raw{player["track_id"]}',(xy[0],max(30,xy[1]-9)),0,.65,COLORS[idx],2)
            for person in people:
                if not person['eligible_not_crop_truncated']:continue
                xy=np.asarray(person['box']);kp=np.asarray(person['keypoints'])
                if xy[2]<l or xy[0]>r or xy[3]<t or xy[1]>b:continue
                xy=np.rint(xy).astype(int);cv2.rectangle(im,tuple(xy[:2]),tuple(xy[2:]),(230,230,230),1);cv2.putText(im,f'p{person["pose_index"]}',(xy[0],xy[1]+18),0,.65,(255,255,255),2)
                for a,z in EDGES:
                    if min(kp[a,2],kp[z,2])>=.5:cv2.line(im,tuple(np.rint(kp[a,:2]).astype(int)),tuple(np.rint(kp[z,:2]).astype(int)),(255,255,0),2)
                for j,k in enumerate(kp):
                    if k[2]>=.5:cv2.circle(im,tuple(np.rint(k[:2]).astype(int)),4,(0,0,255) if j in [15,16] else (255,255,0),-1)
            crop=im[t:b,l:r];scale=min(800/(r-l),470/(b-t));small=cv2.resize(crop,(round((r-l)*scale),round((b-t)*scale)),interpolation=cv2.INTER_AREA);dx=side*800+(800-small.shape[1])//2;dy=80+(470-small.shape[0])//2;canvas[dy:dy+small.shape[0],dx:dx+small.shape[1]]=small
            cv2.putText(canvas,f'{tag} F{f}  {mode}',(side*800+10,28),0,.7,(255,255,255),1);cv2.putText(canvas,'colored=native ID; cyan=pose; red=ankle estimates',(side*800+10,55),0,.49,(255,255,255),1)
        file=dest/f'{i:02d}_{tag}_F{f}.jpg';assert cv2.imwrite(str(file),canvas);cells.append(canvas);manifest.append(dict(tag=tag,source_frame=f,reasons=item['reasons'],focus_slots=[r['player_slot'] for r in focus],crop_native_xyxy=[l,t,r,b],aspect_preserved=True,image=file.name,image_sha256=sha(file)))
    cap.release()
    for page,start in enumerate(range(0,len(cells),2),1):
        group=cells[start:start+2];assert cv2.imwrite(str(dest/f'page_{page:02d}.jpg'),np.vstack(group))
    dump(dest/'manifest.json',dict(selection='All retained known development-example frames plus maximum-native-far-pair-overlap frame per case. Not every pose output; no independent accuracy denominator.',items=manifest,pages=(len(cells)+1)//2,script_sha256=sha(Path(__file__)),independent_GT=False));print(len(cells),'comparison frames;', (len(cells)+1)//2,'pages')

if __name__=='__main__':main()
