"""Stage 2: continuity-aware distractor veto and far-court crop augmentation.

Preserve the old run. These are development candidates, not evaluated IDs.
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
sys.path.insert(0,str(OUT.parent/'tracking_quality_20261102'))
import four_player_association as assoc

original_cost = assoc.cost


def continuity_cost(state,row,feature,frame,use_appearance=True):
    # Reject severe shrinking into a head-only fragment. A per-frame size
    # ratio alone can drift gradually into a tiny box during occlusion.
    if assoc.height(row) < .50*state.get('reference_height',assoc.height(state['row'])):
        return 1e6
    gap = frame-state['frame']
    same_id = int(row['track_id']) == int(state['row']['track_id'])
    h = max(assoc.height(row),assoc.height(state['row']))
    predicted = assoc.foot(state['row'])+state['velocity']*min(gap,12)
    residual = float(np.linalg.norm(assoc.foot(row)-predicted))/max(1,h)
    # Keep all original appearance/scale/side/motion gates. Only the distractor
    # veto is bypassed for a short, spatially continuous SAME raw track.
    # New IDs and reappearances still need the negative-template check.
    if same_id and gap <= 2 and residual < .40:
        return original_cost(dict(state,negatives=[]),row,feature,frame,use_appearance)
    return original_cost(state,row,feature,frame,use_appearance)


def augment(tag):
    os.environ['YOLO_CONFIG_DIR'] = str(OUT/'ultralytics_config')
    from ultralytics import YOLO
    folder = assoc.BASE/f'full_rally_{tag}'
    raw = assoc.read(folder/'bytetrack.csv')
    config = json.loads((folder/'four_player_selection_summary.json').read_text(encoding='utf-8'))[0]
    start,end = config['seed_frame'],config['analysis_ready_end_frame']
    byframe = defaultdict(list)
    for r in raw:
        byframe[int(r['source_frame'])].append(r)
    result = []
    cap = cv2.VideoCapture(str(assoc.SOURCE/'data/raw/match1.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    fps = cap.get(cv2.CAP_PROP_FPS)
    model = YOLO(str(assoc.SOURCE/'models/yolo11m.pt'))
    counts = dict(frames=end-start+1,crop_boxes=0,added_boxes=0,overlap_rejected=0,border_rejected=0)
    for f in range(start,end+1):
        ok,im = cap.read()
        if not ok:
            raise RuntimeError(f'Failed decoding F{f}')
        h,w = im.shape[:2]
        xa,ya,xb,yb = round(.18*w),round(.43*h),round(.82*w),round(.68*h)
        crop = im[ya:yb,xa:xb]
        prediction = model.track(crop,persist=True,classes=[0],conf=.25,imgsz=1280,
                                 tracker='bytetrack.yaml',device=0,verbose=False)[0]
        old = byframe[f]
        result.extend(dict(r,candidate_origin='original_full_frame') for r in old)
        added = []
        for b in prediction.boxes:
            if b.id is None:
                continue
            counts['crop_boxes'] += 1
            x1,y1,x2,y2 = b.xyxy[0].cpu().tolist()
            # Truncated near players/officials should not become full boxes.
            if min(x1,y1,crop.shape[1]-x2,crop.shape[0]-y2) < 3:
                counts['border_rejected'] += 1
                continue
            row = dict(source_frame=f,timestamp_sec=f/fps,
                       track_id=1000000+int(b.id.item()),x1=x1+xa,y1=y1+ya,x2=x2+xa,y2=y2+ya,
                       bottom_center_x=(x1+x2)/2+xa,bottom_center_y=y2+ya,
                       confidence=float(b.conf.item()),in_initial_roi=int(cv2.pointPolygonTest(
                           np.array([[.02,.90],[.98,.90],[.68,.51],[.32,.51]],dtype=np.float32),
                           (((x1+x2)/2+xa)/w,(y2+ya)/h),False) >= 0),
                       candidate_origin='far_crop_extra')
            if assoc.height(row) < 50 or not .49 < float(row['bottom_center_y'])/h < .665:
                continue
            duplicate = any(assoc.box_iou(row,r) > .35 or (
                np.linalg.norm(assoc.foot(row)-assoc.foot(r)) < .22*min(assoc.height(row),assoc.height(r))
                and .5 < assoc.height(row)/max(1,assoc.height(r)) < 2.) for r in old+added)
            if duplicate:
                counts['overlap_rejected'] += 1
                continue
            added.append(row)
        counts['added_boxes'] += len(added)
        result.extend(added)
        if (f-start)%240 == 0:
            print(tag,'crop',f-start,'added',counts['added_boxes'],flush=True)
    cap.release()
    path = OUT/f'augmented_detections_{tag}.csv'
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(result[0]))
        writer.writeheader()
        writer.writerows(result)
    (OUT/f'crop_summary_{tag}.json').write_text(json.dumps(dict(counts,source_csv=str(folder/'bytetrack.csv'),
        crop_normalized=[.18,.43,.82,.68],model='existing yolo11m.pt',imgsz=1280,
        confidence=.25,warning='Additional candidates, not true-positive counts. Geometry is match1-specific.'),indent=2),encoding='utf-8')
    print(tag,counts,flush=True)
    return path


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--rallies',nargs='+',default=['044','056'])
    p.add_argument('--variant',choices=['continuity','crop'],default='continuity')
    p.add_argument('--no-render',action='store_true')
    args = p.parse_args()
    assoc.OUT = OUT
    assoc.cost = continuity_cost
    for tag in args.rallies:
        path = None
        if args.variant == 'crop':
            path = OUT/f'augmented_detections_{tag}.csv'
            if not path.exists():
                path = augment(tag)
        summary = assoc.process(tag,render=not args.no_render,detection_csv=path,
                                variant='continuity_size_guard'+('_far_crop' if path else ''))
        summary['parameters'].update(continuity_veto_bypass=dict(max_gap=2,max_residual_height=.40,
                                                                same_raw_id_required=True),
                                     min_initial_height_fraction=.50)
        dest = OUT/f'players_{tag}_{summary["method"]}'
        (dest/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
