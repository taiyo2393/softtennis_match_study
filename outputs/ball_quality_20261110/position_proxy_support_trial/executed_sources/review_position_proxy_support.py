"""Inspect every changed early body-mask proxy plus geometric counterexamples."""
from pathlib import Path
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import read,sha,dump
import run_position_proxy_support_head_independent as a

OUT=a.OUT
COLORS=[(80,255,80),(255,190,40),(255,80,240),(0,230,255)]
NAMES=['N1','N2','F1','F2']

def paint_body(im,person,color=(255,255,0)):
    if not a.body_supported(person):return
    kp=np.asarray(person['keypoints']);height=person['box'][3]-person['box'][1];quad=kp[[5,6,12,11],:2];center=quad.mean(axis=0);quad=np.rint(center+1.12*(quad-center)).astype(np.int32);cv2.fillConvexPoly(im,cv2.convexHull(quad),color)
    if kp[0,2]>=.5:cv2.circle(im,tuple(np.rint(kp[0,:2]).astype(int)),max(1,round(.09*height)),color,-1)
    for l,r,fraction in [(5,7,.045),(7,9,.045),(6,8,.045),(8,10,.045),(11,13,.065),(13,15,.065),(12,14,.065),(14,16,.065)]:
        if min(kp[l,2],kp[r,2])<.5:continue
        p1=tuple(np.rint(kp[l,:2]).astype(int));p2=tuple(np.rint(kp[r,:2]).astype(int));rad=max(1,round(fraction*height));cv2.line(im,p1,p2,color,2*rad);cv2.circle(im,p1,rad,color,-1);cv2.circle(im,p2,rad,color,-1)

def matches_for(f,group,poses):
    matched={}
    for mode in a.CROPS:
        for slot,value in a.match_people(group,poses.get((f,mode),[])).items():
            if mode==('far_context' if slot.startswith('far_') else 'court'):matched[slot]=value
    return matched

def panel(raw,group,matches,focus_slot,diagnostic,crop):
    l,t,r,b=crop;canvas=np.zeros((560,1600,3),np.uint8)
    for side in range(2):
        im=raw.copy()
        if side:
            overlay=im.copy()
            for slot,(person,_) in matches.items():
                if slot!=focus_slot:paint_body(overlay,person)
            im=cv2.addWeighted(im,.75,overlay,.25,0)
        for slot,row in group.items():
            i=a.SLOTS.index(slot);bb=[round(float(row[k])) for k in ['x1','y1','x2','y2']];cv2.rectangle(im,tuple(bb[:2]),tuple(bb[2:]),COLORS[i],3);cv2.putText(im,NAMES[i],(bb[0],max(30,bb[1]-10)),0,.8,COLORS[i],2)
        row=group[focus_slot];point=(round(float(row['bottom_center_x'])),round(float(row['bottom_center_y'])));cv2.circle(im,point,9,(0,128,255),-1);cv2.circle(im,point,12,(0,0,0),2)
        if side and focus_slot in matches:
            person,_=matches[focus_slot];kp=np.asarray(person['keypoints'])
            for j in [15,16]:
                if kp[j,2]>=.5:cv2.drawMarker(im,tuple(np.rint(kp[j,:2]).astype(int)),(0,0,255),cv2.MARKER_TILTED_CROSS,16,3)
        view=im[t:b,l:r];scale=min(800/(r-l),470/(b-t));small=cv2.resize(view,(round((r-l)*scale),round((b-t)*scale)));dx=side*800+(800-small.shape[1])//2;dy=80+(470-small.shape[0])//2;canvas[dy:dy+small.shape[0],dx:dx+small.shape[1]]=small
        cv2.putText(canvas,'Native proxy (orange dot)' if side==0 else 'Other-body approximation + model ankles (red)',(side*800+8,55),0,.47,(255,255,255),1)
    cv2.putText(canvas,f'{diagnostic["tag"]} F{diagnostic["source_frame"]} {focus_slot} | boxRisk={diagnostic["geometry_box_overlap_risk"]} bodyRisk={diagnostic["geometry_body_overlap_risk"]}',(10,27),0,.7,(255,255,255),1)
    return canvas

def main():
    p=a.load(OUT/'protocol.json');dest=OUT/'AI_review';dest.mkdir();requests={};tables={};diags={};poses={};all_prefix=[]
    for e in p['cases']:
        tag=e['tag'];tables[tag]=a.native_table(e);poses[tag]=a.all_pose_geometry(tag);rows=read(OUT/f'case_{tag}/position_proxy_support.csv');diags[tag]={(int(r['source_frame']),r['player_slot']):r for r in rows};prefix=read(OUT/f'case_{tag}/fixed_prefix_position_support.csv');all_prefix.extend(prefix)
        for r in prefix:
            if r['geometry_body_overlap_risk']=='1':requests[tag,int(r['source_frame']),r['player_slot']]='Every position-only body-mask change in the fixed input windows'
        box_only=next((r for r in prefix if r['geometry_box_overlap_risk']=='1' and r['geometry_body_overlap_risk']=='0'),None)
        if box_only:requests.setdefault((tag,int(box_only['source_frame']),box_only['player_slot']),'First fixed-window rectangle-only overlap; inspect possible over-masking')
    for tag,f,slot in [('019',46511,'far_rear'),('019',46634,'far_rear'),('019',46406,'far_rear'),('019',46436,'far_rear'),('019',46352,'far_rear'),('016',40625,'far_rear')]:requests.setdefault((tag,f,slot),'Retained mixed/fragment/clear-person diagnostic, not independent GT')
    cap=cv2.VideoCapture(p['source_video']);cells=[];manifest=[]
    for i,((tag,f,slot),reason) in enumerate(sorted(requests.items()),1):
        group=tables[tag][f];row=group[slot];diagnostic=diags[tag][f,slot];focus=[row]+[group[s] for s in diagnostic['native_proxy_in_other_boxes'].split('|') if s in group];bb=np.asarray([a.rect(r) for r in focus]);l=max(0,int(bb[:,0].min())-150);t=max(0,int(bb[:,1].min())-120);r=min(3840,int(bb[:,2].max())+150);b=min(2160,int(bb[:,3].max())+160);crop=[l,t,r,b];cap.set(1,f);ok,raw=cap.read();assert ok and round(cap.get(1))==f+1;im=panel(raw,group,matches_for(f,group,poses[tag]),slot,diagnostic,crop);file=dest/f'{i:02d}_{tag}_F{f}.jpg';assert cv2.imwrite(str(file),im);cells.append(im);manifest.append(dict(tag=tag,source_frame=f,slot=slot,reason=reason,image=file.name,image_sha256=sha(file),crop_native_xyxy=crop,aspect_preserved=True,box_risk=int(diagnostic['geometry_box_overlap_risk']),body_risk=int(diagnostic['geometry_body_overlap_risk'])))
    cap.release()
    for page,start in enumerate(range(0,len(cells),2),1):assert cv2.imwrite(str(dest/f'page_{page:02d}.jpg'),np.vstack(cells[start:start+2]))
    dump(dest/'manifest.json',dict(selection='Every four body-mask change in fixed prefixes, first rectangle-only risk per affected case, retained019 mixed/fragment frames and one clear control. Post-output diagnostic selection; not independent accuracy.',items=manifest,script_sha256=sha(Path(__file__)),independent_GT=False));print(len(cells),'review frames',[(r['tag'],r['source_frame'],r['slot']) for r in manifest])

if __name__=='__main__':main()
