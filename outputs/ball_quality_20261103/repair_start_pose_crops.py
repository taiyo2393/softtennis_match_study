"""Recover complete court poses before tile/global deduplication, label blind."""
import argparse,json,os
from pathlib import Path
import cv2
import numpy as np
from extract_serve_start_features import SOURCE,ROOT,sha

V2=ROOT/'serve_start_pilot_tiled_v2';OUT=ROOT/'serve_phase_pilot_v3'
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'serve_start_pilot/ultralytics_config')

def contained(p):
    l,t,r,b=(480,360,1536,720) if p.get('origin','court')=='far_court' else (0,320,1920,1080)
    x1,y1,x2,y2=p['box']
    return min(x1-l,y1-t,r-x2,b-y2)>4

def overlap(a,b):
    x1,y1,x2,y2=a;u1,v1,u2,v2=b
    inter=max(0,min(x2,u2)-max(x1,u1))*max(0,min(y2,v2)-max(y1,v1))
    return inter/max(1,(x2-x1)*(y2-y1)+(u2-u1)*(v2-v1)-inter)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',choices=['016','002'],required=True)
    ap.add_argument('--rebuild-from-cache',action='store_true');a=ap.parse_args()
    cid=f'match1_motion_pilot_{a.pilot}';dest=OUT/cid;dest.mkdir(parents=True,exist_ok=True)
    assert a.rebuild_from_cache or not (dest/'features.json').exists()
    old=V2/cid;protocol=json.loads((old/'protocol.json').read_text())
    tile=json.loads((old/'features.json').read_text())
    oldfull=ROOT/'serve_start_pilot'/cid/'features.json'
    cache=dest/'full_court_features.json'
    if a.rebuild_from_cache:assert oldfull.exists() or cache.exists()
    if oldfull.exists():
        full=json.loads(oldfull.read_text());full_origin=str(oldfull);full_hash=sha(oldfull)
    elif cache.exists():
        full=json.loads(cache.read_text());full_origin=str(cache);full_hash=sha(cache)
    else:
        from ultralytics import YOLO
        net=YOLO(protocol['model']);frames=[]
        for t in range(0,protocol['frames'],2):
            im=cv2.imread(str(SOURCE/'data/balltrack/racketvision/match1'/cid/'frames'/f'{t:06d}.jpg'));assert im is not None
            result=net.predict(im[320:1080],imgsz=1920,conf=.25,iou=.7,device=0,verbose=False)[0]
            people=[]
            for box,kp in zip(result.boxes.data.cpu().numpy(),result.keypoints.data.cpu().numpy()):
                box=box.copy();kp=kp.copy();box[[1,3]]+=320;kp[:,1]+=320
                x1,y1,x2,y2=box[:4];h=y2-y1;cx=(x1+x2)/2
                if h<55 or y2<.48*1080 or not .12*1920<=cx<=.94*1920:continue
                people.append(dict(box=box[:4].tolist(),confidence=float(box[4]),keypoints=kp.tolist(),origin='court'))
            frames.append(dict(local_frame=t,people=people))
            if t%120==0:print(cid,t,'/',protocol['frames'],flush=True)
        full=dict(frames=frames,model_sha256=protocol['model_sha256'],labels_read=False)
        cache.write_text(json.dumps(full),encoding='utf-8');full_origin=str(cache);full_hash=sha(cache)
    output=[];removed=0
    for tf,ff in zip(tile['frames'],full['frames'],strict=True):
        assert tf['local_frame']==ff['local_frame']
        candidates=[p for p in tf['people'] if p['origin']=='far_court']+ff['people'];people=[]
        for p in candidates:
            p=dict(p,origin=p.get('origin','court'))
            if not contained(p):removed+=1;continue
            if any(overlap(p['box'],q['box'])>.5 for q in people):continue
            people.append(p)
        output.append(dict(local_frame=tf['local_frame'],people=people))
    protocol.update(repair='Reject boxes within4px of extraction-crop edge BEFORE deduplication; recover full-court detections suppressed by truncated tile boxes',
        tiled_feature_sha256=sha(old/'features.json'),full_court_input=full_origin,full_court_sha256=full_hash,
        rejected_crop_edge_boxes=removed,script_sha256=sha(Path(__file__)),labels_used=False)
    (dest/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    (dest/'features.json').write_text(json.dumps(dict(frames=output,labels_read=False,protocol_sha256=sha(dest/'protocol.json'))),encoding='utf-8')
    print(cid,'repaired',len(output),'samples; rejected',removed,flush=True)

if __name__=='__main__':main()
