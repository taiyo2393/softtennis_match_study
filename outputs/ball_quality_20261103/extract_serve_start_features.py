"""Label-blind pose/audio features for a minimal overhead-serve proposal pilot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import wave

import cv2
import numpy as np

SOURCE=Path(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study')
ROOT=Path(__file__).resolve().parent;OUT=ROOT/'serve_start_pilot'
os.environ['YOLO_CONFIG_DIR']=str(OUT/'ultralytics_config')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',required=True,choices=['016','002']);parser.add_argument('--tiled',action='store_true');a=parser.parse_args()
    feature_root=ROOT/'serve_start_pilot_tiled_v2' if a.tiled else OUT
    feature_root.mkdir(exist_ok=True);cid=f'match1_motion_pilot_{a.pilot}';dest=feature_root/cid;dest.mkdir(exist_ok=True)
    assert not (dest/'features.json').exists()
    base=SOURCE/'data/balltrack/racketvision/match1'/cid
    meta=json.loads((base/'runs/threshold030_median120/run_metadata.json').read_text());fps=meta['fps']
    model_path=SOURCE/'models/yolo11m-pose.pt'
    protocol=dict(pilot=cid,frames=meta['frame_count'],fps=fps,sampling_stride=2,
        image='cached source JPEG, court-oriented crop y320:1080, full width',
        far_court_tile=[480,360,1536,720] if a.tiled else None,
        model=str(model_path),model_sha256=sha(model_path),imgsz=1920,confidence=.25,iou=.7,
        geometry='person height>=55px, feet y>=0.48H, center x in0.12W..0.94W',
        audio='source clip mono16k PCM; 20ms RMS,10ms hop; positive log-energy rise over prior70ms median',
        labels_used=False,purpose='Development start candidates, not a validated contact/validity classifier.')
    (dest/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    from ultralytics import YOLO
    net=YOLO(str(model_path));output=[]
    for t in range(0,meta['frame_count'],2):
        im=cv2.imread(str(base/'frames'/f'{t:06d}.jpg'));assert im is not None
        regions=[(0,320,1920,1080,'court')]
        if a.tiled:regions=[(480,360,1536,720,'far_court'),*regions]
        people=[]
        for left,top,right,bottom,origin in regions:
            result=net.predict(im[top:bottom,left:right],imgsz=1920,conf=.25,iou=.7,device=0,verbose=False)[0]
            for box,kp in zip(result.boxes.data.cpu().numpy(),result.keypoints.data.cpu().numpy()):
                box=box.copy();box[[0,2]]+=left;box[[1,3]]+=top;kp=kp.copy();kp[:,0]+=left;kp[:,1]+=top
                x1,y1,x2,y2=box[:4];h=y2-y1;cx=(x1+x2)/2
                if h<55 or y2<.48*1080 or not .12*1920<=cx<=.94*1920:continue
                duplicate=False
                for p in people:
                    q=np.array(p['box']);iw=max(0,min(x2,q[2])-max(x1,q[0]));ih=max(0,min(y2,q[3])-max(y1,q[1]));inter=iw*ih
                    iou=inter/max(1,(x2-x1)*(y2-y1)+(q[2]-q[0])*(q[3]-q[1])-inter)
                    if iou>.5:duplicate=True;break
                if not duplicate:people.append(dict(box=box[:4].tolist(),confidence=float(box[4]),keypoints=kp.tolist(),origin=origin))
        output.append(dict(local_frame=t,people=people))
        if t%120==0:print(cid,t,'/',meta['frame_count'],flush=True)
    audio=dest/'audio16k.wav'
    subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-i',meta['clip'],
                    '-vn','-ac','1','-ar','16000','-c:a','pcm_s16le',str(audio)],check=True)
    with wave.open(str(audio),'rb') as wav:
        assert wav.getnchannels()==1 and wav.getsampwidth()==2 and wav.getframerate()==16000
        signal=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(np.float32)/32768.
    frames=np.lib.stride_tricks.sliding_window_view(signal,320)[::160]
    energy=20*np.log10(np.sqrt(np.mean(frames*frames,axis=1))+1e-7)
    times=(np.arange(len(energy))*160+160)/16000.
    rise=np.array([max(0.,float(energy[i]-np.median(energy[max(0,i-7):i]))) if i else 0. for i in range(len(energy))])
    np.savez_compressed(dest/'audio_features.npz',time_sec=times,energy_db=energy,rise_db=rise)
    (dest/'features.json').write_text(json.dumps(dict(frames=output,protocol_sha256=sha(dest/'protocol.json'),
        script_sha256=sha(Path(__file__)),labels_read=False)),encoding='utf-8')
    print(cid,'features saved',len(output),'pose samples',flush=True)


if __name__=='__main__':main()
