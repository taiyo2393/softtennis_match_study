"""Explain position warnings while preserving every retained native identity row."""
import subprocess
from pathlib import Path
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import read,sha,dump
from review_position_proxy_support import a,OUT,paint_body,matches_for,COLORS,NAMES

def frame_panel(raw,group,matches,slot,diag,crop,label):
    canvas=np.zeros((560,1600,3),np.uint8);l,t,r,b=crop
    for side in range(2):
        im=raw.copy()
        if side:
            overlay=im.copy()
            for other,(person,_) in matches.items():
                if other!=slot:paint_body(overlay,person)
            im=cv2.addWeighted(im,.75,overlay,.25,0)
        for s,row in group.items():
            i=a.SLOTS.index(s);bb=[round(float(row[k])) for k in ['x1','y1','x2','y2']];cv2.rectangle(im,tuple(bb[:2]),tuple(bb[2:]),COLORS[i],3);cv2.putText(im,NAMES[i],(bb[0],max(30,bb[1]-10)),0,.8,COLORS[i],2)
        if slot in group:
            row=group[slot];point=(round(float(row['bottom_center_x'])),round(float(row['bottom_center_y'])));cv2.circle(im,point,9,(0,128,255),-1);cv2.circle(im,point,12,(0,0,0),2)
        if side and slot in matches:
            kp=np.asarray(matches[slot][0]['keypoints'])
            for j in [15,16]:
                if kp[j,2]>=.5:cv2.drawMarker(im,tuple(np.rint(kp[j,:2]).astype(int)),(0,0,255),cv2.MARKER_TILTED_CROSS,16,3)
        view=im[t:b,l:r];scale=min(800/(r-l),450/(b-t));small=cv2.resize(view,(round((r-l)*scale),round((b-t)*scale)));dx=side*800+(800-small.shape[1])//2;dy=90+(450-small.shape[0])//2;canvas[dy:dy+small.shape[0],dx:dx+small.shape[1]]=small
    cv2.putText(canvas,f'{diag["tag"]} F{diag["source_frame"]}: {label}',(8,25),0,.67,(255,255,255),1)
    cv2.putText(canvas,f'Native IDs unchanged | boxRisk={diag["geometry_box_overlap_risk"]} bodyRisk={diag["geometry_body_overlap_risk"]} | '+('proxy missing' if slot not in group else 'orange: native proxy'),(8,51),0,.56,(255,255,255),1)
    cv2.putText(canvas,'LEFT: original image | RIGHT: other-body approximation; red: model ankles, not ground truth',(8,77),0,.53,(255,255,255),1)
    return canvas

def main():
    p=a.load(OUT/'protocol.json');dest=OUT/'comparison_video';dest.mkdir();segments=[dict(tag='002',first=2269,last=2269,repeat=90,slot='far_front',label='Rectangle-only overlap warning; still image'),dict(tag='002',first=2303,last=2303,repeat=90,slot='far_front',label='Proxy on foreground body; still image'),dict(tag='019',first=46396,last=46396,repeat=90,slot='far_rear',label='Fixed input-window proxy warning; still image'),dict(tag='019',first=46500,last=46521,repeat=6,slot='far_rear',label='Mixed native box;1/6 speed'),dict(tag='019',first=46625,last=46641,repeat=6,slot='far_rear',label='Rear player hidden by teammate;1/6 speed')]
    tables={};poses={};diags={};selections=[]
    for tag in ['002','019']:
        e=next(e for e in p['cases'] if e['tag']==tag);tables[tag]=a.native_table(e);poses[tag]=a.all_pose_geometry(tag);diags[tag]={(int(r['source_frame']),r['player_slot']):r for r in read(OUT/f'case_{tag}/position_proxy_support.csv')}
    for segment in segments:
        tag,slot=segment['tag'],segment['slot'];focus=[]
        for f in range(segment['first'],segment['last']+1):
            group=tables[tag][f];d=diags[tag][f,slot]
            for s in {slot,*d['native_proxy_in_other_boxes'].split('|'),*d['native_proxy_in_other_pose_bodies'].split('|')}:
                if s in group:focus.append(a.rect(group[s]))
        bb=np.asarray(focus);crop=[max(0,int(bb[:,0].min())-150),max(0,int(bb[:,1].min())-120),min(3840,int(bb[:,2].max())+150),min(2160,int(bb[:,3].max())+160)];selections.append(dict(segment,crop_native_xyxy=crop))
    dump(dest/'selection.json',dict(segments=selections,source_clock=p['source_clock'],selection_after_diagnostic_outputs=True,aspect_preserved=True,dimensions=[1600,560],fps=60000/1001,interpretation='Warnings and body/ankle estimates only; no repaired coordinates or identity changes.',script_sha256=sha(Path(__file__))))
    cap=cv2.VideoCapture(p['source_video']);temp=dest/'proxy_support_mp4v.mp4';writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),60000/1001,(1600,560));assert writer.isOpened();count=0;QA=[]
    for segment in selections:
        tag,slot=segment['tag'],segment['slot'];cap.set(1,segment['first'])
        for f in range(segment['first'],segment['last']+1):
            ok,im=cap.read();assert ok and round(cap.get(1))==f+1;group=tables[tag][f];canvas=frame_panel(im,group,matches_for(f,group,poses[tag]),slot,diags[tag][f,slot],segment['crop_native_xyxy'],segment['label'])
            if segment['first']==segment['last'] or f in [46511,46634]:QA.append(cv2.resize(canvas,(800,280)))
            for _ in range(segment['repeat']):writer.write(canvas);count+=1
    cap.release();writer.release();assert count==504 and len(QA)==5;video=dest/'position_proxy_support_h264.mp4';subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-i',str(temp),'-an','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True);assert cv2.imwrite(str(dest/'video_QA.jpg'),np.vstack(QA));cap=cv2.VideoCapture(str(video));decoded=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(560,1600);decoded+=1
    cap.release();assert decoded==504;dump(dest/'verification.json',dict(status='passed',decoded_frames=decoded,aspect_preserved=True,QA_frames=5,video_sha256=sha(video),QA_sha256=sha(dest/'video_QA.jpg'),native_IDs_changed=False,model_coordinates_adopted=False));print(decoded,'decoded frames')

if __name__=='__main__':main()
