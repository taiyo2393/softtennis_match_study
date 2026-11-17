"""Explanatory temporal pose video; inherited native IDs are deliberately unchanged."""
import os,subprocess
from pathlib import Path
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,sha,dump
from run_body_part_localization_probe import OUT,P,CROPS,SLOTS,load,case_tables
from prepare_body_part_probe_review import EDGES,COLORS,NAMES

DEST=OUT/'temporal_demo'

def main():
    p=load(OUT/'protocol.json');cached={(r['tag'],r['source_frame']):r['modes']['far_context'] for r in load(OUT/'pose_predictions.json')['frames']};DEST.mkdir()
    segments=[dict(tag='006',first=12410,last=12432,label='Actual player beside official'),dict(tag='016',first=40615,last=40635,label='Actual server retained'),dict(tag='019',first=46485,last=46535,label='Mixed official/player rectangles'),dict(tag='019',first=46620,last=46645,label='Rear player hidden behind teammate'),dict(tag='019',first=46650,last=46680,label='Near player wrongly labelled far F2')]
    protected={str(n):sha(n) for n in [Path(__file__),OUT/'protocol.json',OUT/'pose_predictions.json']}
    plan=dict(segments=segments,selection_after_diagnostic_images=True,purpose='Explain body-part feasibility and remaining failures over continuous source frames; not an independent temporal tracking test.',parameters=P,crop_model_native=CROPS['far_context'],crop_display_native=[1200,970,2800,1900],output_dimensions=[1920,720],source_fps=60000/1001,repeat=3,aspect_preserved=True,source_video=p['source_video'],source_clock=p['source_clock'],model=p['model'],native_IDs_unchanged=True,pose_indices_are_per_frame_not_track_IDs=True,model_outputs_not_exported_as_ground_contact=True,protected_input_hashes=protected)
    dump(DEST/'protocol.json',plan);os.environ['YOLO_CONFIG_DIR']=str(OUT/'ultralytics_config')
    from ultralytics import YOLO
    net=YOLO(p['model']);cap=cv2.VideoCapture(p['source_video']);tables={e['tag']:case_tables(e) for e in p['cases'] if e['tag'] in ['006','016','019']};temp=DEST/'demo_mp4v.mp4';writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),60000/1001,(1920,720));assert writer.isOpened();records=[];QA=[];inferences=reused=0
    for segment in segments:
        tag=segment['tag'];cap.set(1,segment['first'])
        for f in range(segment['first'],segment['last']+1):
            ok,raw=cap.read();assert ok and round(cap.get(1))==f+1
            if (tag,f) in cached:people=cached[tag,f];reused+=1
            else:
                l,t,r,b=CROPS['far_context'];pred=net.predict(raw[t:b,l:r],imgsz=1920,conf=.25,iou=.7,device=0,verbose=False)[0];people=[];inferences+=1
                for j,(bb,kp) in enumerate(zip(pred.boxes.data.cpu().numpy(),pred.keypoints.data.cpu().numpy())):
                    bb=bb.copy();kp=kp.copy();bb[[0,2]]+=l;bb[[1,3]]+=t;kp[:,0]+=l;kp[:,1]+=t
                    people.append(dict(pose_index=j,box=bb[:4].tolist(),confidence=float(bb[4]),keypoints=kp.tolist(),eligible_not_crop_truncated=bool(min(bb[0]-l,bb[1]-t,r-bb[2],b-bb[3])>8)))
            records.append(dict(tag=tag,source_frame=f,reused_frozen_prediction=(tag,f) in cached,people=people));canvas=np.zeros((720,1920,3),np.uint8)
            cv2.putText(canvas,f'{tag} source F{f}: {segment["label"]}',(15,28),0,.72,(255,255,255),1)
            cv2.putText(canvas,'INHERITED NATIVE BOXES / IDs',(15,65),0,.70,(255,255,255),2);cv2.putText(canvas,'SAME IDs + POSE ESTIMATES (not a repaired path)',(975,65),0,.64,(255,255,255),2)
            for side in range(2):
                im=raw.copy()
                for slot,row in tables[tag][f].items():
                    i=SLOTS.index(slot);xy=[round(float(row[k])) for k in ['x1','y1','x2','y2']];cv2.rectangle(im,tuple(xy[:2]),tuple(xy[2:]),COLORS[i],4);cv2.putText(im,NAMES[i],(xy[0],max(40,xy[1]-10)),0,.9,COLORS[i],2)
                if side:
                    for person in people:
                        if not person['eligible_not_crop_truncated']:continue
                        kp=np.asarray(person['keypoints'])
                        for a,b in EDGES:
                            if min(kp[a,2],kp[b,2])>=.5:cv2.line(im,tuple(np.rint(kp[a,:2]).astype(int)),tuple(np.rint(kp[b,:2]).astype(int)),(255,255,0),3)
                        for j,k in enumerate(kp):
                            if k[2]>=.5:cv2.circle(im,tuple(np.rint(k[:2]).astype(int)),5,(0,0,255) if j in [15,16] else (255,255,0),-1)
                crop=cv2.resize(im[970:1900,1200:2800],(960,558),interpolation=cv2.INTER_AREA);canvas[100:658,side*960:(side+1)*960]=crop
            cv2.putText(canvas,'1/3 speed | cyan: estimated skeleton | red: estimated ankles, NOT verified ground contact',(15,699),0,.65,(255,255,255),1)
            for _ in range(3):writer.write(canvas)
            if (tag,f) in {('006',12421),('016',40625),('019',46501),('019',46511),('019',46634),('019',46665)}:QA.append(cv2.resize(canvas,(960,360)))
        print('Rendered',tag,segment['first'],segment['last'],flush=True)
    cap.release();writer.release();assert len(records)==152 and len(QA)==6;dump(DEST/'pose_predictions.json',dict(frames=records,temporal_identity_tracking_performed=False));out=DEST/'body_part_feasibility_h264.mp4'
    subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-i',str(temp),'-an','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(out)],check=True);assert cv2.imwrite(str(DEST/'video_QA.jpg'),np.vstack(QA));cap=cv2.VideoCapture(str(out));n=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(720,1920);n+=1
    cap.release();assert n==456 and all(sha(Path(k))==h for k,h in protected.items())
    dump(DEST/'verification.json',dict(status='passed',source_frames=152,inference_calls=inferences,reused_probe_frames=reused,decoded_video_frames=n,aspect_preserved=True,QA_frames=6,video_sha256=sha(out),QA_sha256=sha(DEST/'video_QA.jpg'),trajectory_repaired=False,new_verified_positions=False));print(load(DEST/'verification.json'))

if __name__=='__main__':main()
