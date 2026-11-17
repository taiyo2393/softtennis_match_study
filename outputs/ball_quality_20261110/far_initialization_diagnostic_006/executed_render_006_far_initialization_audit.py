"""Aspect-preserved candidate audit panels; colors encode gates, never truth."""
import subprocess
from collections import defaultdict
import cv2
import numpy as np
from run_fixed_cohort_player_case_v3 import BASE, FPS, read, dump
from run_near_prefix_initialization_trial import load

OUT=BASE/'far_initialization_diagnostic_006'
FAR_CROP=(690,900,3150,1515)  # 2460x615 ->1280x320
DETAIL_CROP=(1950,1050,2370,1386)  # 420x336 ->500x400


def panel(frame,rows,record,f):
    annotated=frame.copy()
    for r in rows:
        if r['broad_far_ROI']!='1':continue
        green=r['far_seed_eligible']=='1'
        color=(50,230,50) if green else (0,150,255)
        xy=[round(float(r[k])) for k in ['x1','y1','x2','y2']]
        cv2.rectangle(annotated,tuple(xy[:2]),tuple(xy[2:]),color,4)
        text=f"raw{r['track_id']}"
        cv2.putText(annotated,text,(xy[0],max(32,xy[1]-12)),cv2.FONT_HERSHEY_SIMPLEX,1.1,color,3)
    canvas=np.zeros((800,1280,3),np.uint8)
    title=[f'006 source F{f} | native detections only | diagnostic, not identity ground truth',
           'Green: passes seed gate; orange: other broad-ROI box. Color does NOT mean correct/incorrect.']
    for i,t in enumerate(title):cv2.putText(canvas,t,(12,26+32*i),cv2.FONT_HERSHEY_SIMPLEX,.62,(245,245,245),1,cv2.LINE_AA)
    l,t,r,b=FAR_CROP;canvas[80:400]=cv2.resize(annotated[t:b,l:r],(1280,320),interpolation=cv2.INTER_AREA)
    l,t,r,b=DETAIL_CROP
    canvas[400:,0:500]=cv2.resize(frame[t:b,l:r],(500,400),interpolation=cv2.INTER_CUBIC)
    canvas[400:,500:1000]=cv2.resize(annotated[t:b,l:r],(500,400),interpolation=cv2.INTER_CUBIC)
    for x,text in [(8,'Raw overlap detail'),(508,'Same crop with boxes')]:
        cv2.rectangle(canvas,(x,404),(x+290,435),(0,0,0),-1)
        cv2.putText(canvas,text,(x+3,426),cv2.FONT_HERSHEY_SIMPLEX,.58,(255,255,255),1,cv2.LINE_AA)
    lines=['Existing gates',f"Far seeds: {record['far_seed_candidates']}",
           f"Near output: {record['near_output_count']}/2",f"Far output: {record['far_output_count']}/2",'',
           'All boxes are native.','No extra far IDs.','No hidden-foot estimate.','',
           'Video playback 0.5x','Reference P011:','one far point unplaceable.']
    for i,line in enumerate(lines):cv2.putText(canvas,line,(1008,425+26*i),cv2.FONT_HERSHEY_SIMPLEX,.49,(235,235,235),1,cv2.LINE_AA)
    return canvas


def main():
    p=load(OUT/'protocol.json');result=load(OUT/'result.json');wanted=set(result['review_frames'])
    by=defaultdict(list)
    for r in read(OUT/'candidate_gate_audit.csv'):
        if r['tag']=='006':by[int(r['source_frame'])].append(r)
    records={int(r['source_frame']):r for r in read(OUT/'frame_gate_audit.csv') if r['tag']=='006'}
    first,last=10430,10596
    dump(OUT/'media_protocol.json',dict(source_frame_range=[first,last],review_frames=sorted(wanted),
        far_display_crop_native=FAR_CROP,detail_display_crop_native=DETAIL_CROP,
        aspect_preserved=True,selection='Frozen diagnostic frames plus a continuous illustrative interval, not an accuracy sample.',
        colors_are_gate_results_not_correctness=True,actual_positions_or_IDs_exported=False))
    video=OUT/'006_far_candidate_audit.mp4';assert not video.exists()
    command=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','1280x800',
        '-r',str(FPS/2),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    with (OUT/'encoding.log').open('wb') as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        cap=cv2.VideoCapture(p['source_video']);cap.set(1,first);mapping=[]
        for f in range(first,last+1):
            ok,im=cap.read();assert ok and round(cap.get(1))==f+1
            canvas=panel(im,by[f],records[f],f);process.stdin.write(canvas.tobytes())
            if f in wanted:assert cv2.imwrite(str(OUT/f'F{f}_candidate_audit.jpg'),canvas)
            mapping.append(dict(output_frame=f-first,source_frame=f))
        cap.release();process.stdin.close();assert process.wait()==0
    from run_fixed_cohort_player_case_v3 import write
    write(OUT/'video_frame_map.csv',mapping)
    cap=cv2.VideoCapture(str(video));decoded=0;fps=cap.get(5)
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(800,1280);decoded+=1
    cap.release();assert decoded==167 and abs(fps-FPS/2)<.001
    dump(OUT/'media_verification.json',dict(status='passed',decoded_frames=decoded,fps=fps,
        source_crop_aspects_unchanged=True,source_output_frame_map_complete=True))


if __name__=='__main__':main()
