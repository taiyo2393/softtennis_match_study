"""Apply frozen stage-2 settings to three additional, preselected intervals.

No threshold changes, no alternative seed search, and no GT label edits.
These intervals are new to full stage-2 tracking, not independent matches.
"""
import csv
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
STAGE2 = OUT.parent/'tracking_quality_20261102_stage2'
sys.path.insert(0,str(STAGE2))
import improve_players as stage2
assoc = stage2.assoc
SOURCE = assoc.SOURCE
os.environ['YOLO_CONFIG_DIR'] = str(OUT/'ultralytics_config')
from ultralytics import YOLO
import ultralytics


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_rows(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plan():
    ready = assoc.read(SOURCE/'data/segments/analysis_ready_main_camera_segments.csv')
    intervals = []
    # Deliberately include 005, where previous snapshots showed occlusion.
    # Limit each interval to 900 source frames; do not infer exact point ends.
    for tag,start in [('002',2223),('005',8607),('048',125859)]:
        matches = [r for r in ready if r['video_id'] == 'match1'
                   and r['usable_for_analysis'].lower() == 'true'
                   and int(r['start_source_frame']) <= start <= int(r['end_source_frame'])]
        assert len(matches) == 1
        end = min(int(matches[0]['end_source_frame']),start+899)
        intervals.append(dict(tag=tag,first_frame=start,last_frame=end,frames=end-start+1,
                              ready_segment=matches[0],boundary_status='candidate_interval_not_human_confirmed_point'))
    sources = [STAGE2/'improve_players.py',OUT.parent/'tracking_quality_20261102/four_player_association.py',
               SOURCE/'scripts/96_select_four_players_pilot.py',SOURCE/'models/yolo11m.pt',
               Path(ultralytics.__file__).parent/'cfg/trackers/bytetrack.yaml']
    metadata = dict(intervals=intervals,settings='stage2 continuity_size_guard_far_crop, frozen before new inference',
                    selection_reason='early game, post-court-change occluded interval, later game; all same match',
                    sha256={str(p):sha(p) for p in sources},ultralytics_version=ultralytics.__version__,
                    max_interval_frames=900,warning='No independent identity labels; earlier snapshots existed for these points.')
    path = OUT/'predeclared_protocol.json'
    if path.exists():
        old = json.loads(path.read_text(encoding='utf-8'))
        assert old == metadata,'Protocol or source changed; use a new experiment directory.'
    else:
        path.write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    return metadata


def detect(interval,selector):
    tag,start,end = interval['tag'],interval['first_frame'],interval['last_frame']
    folder = assoc.BASE/f'full_rally_{tag}'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'run_status.json').exists():
        return json.loads((folder/'run_status.json').read_text(encoding='utf-8'))
    model = YOLO(str(SOURCE/'models/yolo11m.pt'))
    tracker_path = folder/'bytetrack.yaml'
    tracker_path.write_bytes((Path(ultralytics.__file__).parent/'cfg/trackers/bytetrack.yaml').read_bytes())
    cap = cv2.VideoCapture(str(SOURCE/'data/raw/match1.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    fps = cap.get(cv2.CAP_PROP_FPS)
    polygon = np.array([[.02,.90],[.98,.90],[.68,.51],[.32,.51]],dtype=np.float32)
    rows,first_im = [],None
    for frame in range(start,end+1):
        ok,im = cap.read()
        if not ok:
            raise RuntimeError(f'Cannot decode F{frame}')
        if frame == start:
            first_im = cv2.resize(im,(1920,1080))
        h,w = im.shape[:2]
        pred = model.track(im,persist=True,classes=[0],conf=.25,imgsz=1280,
                           tracker=str(tracker_path),device=0,verbose=False)[0]
        for b in pred.boxes:
            x1,y1,x2,y2 = b.xyxy[0].cpu().tolist()
            x,y = (x1+x2)/2,y2
            rows.append(dict(source_frame=frame,timestamp_sec=frame/fps,
                             track_id=int(b.id.item()) if b.id is not None else -1,
                             x1=x1,y1=y1,x2=x2,y2=y2,bottom_center_x=x,bottom_center_y=y,
                             confidence=float(b.conf.item()),
                             in_initial_roi=int(cv2.pointPolygonTest(polygon,(x/w,y/h),False)>=0)))
        if (frame-start)%240 == 0:
            print(tag,'full-frame detection',frame-start,'/',interval['frames'],flush=True)
    cap.release()
    write_rows(folder/'bytetrack.csv',rows)
    status = dict(interval,source_fps=fps,source_width=w,source_height=h,initialization_status='unattempted')
    try:
        seeds = selector.seed_by_court_geometry(rows,start,w,h)
        status.update(initialization_status='four_seeds_proposed',seed_ids=seeds,
                      seed_review_status='not_human_confirmed')
        config = dict(mode='bytetrack',seed_frame=start,analysis_ready_end_frame=end,seed_ids=seeds,
                      seed_status='geometry_proposal_not_independent_identity_ground_truth')
        (folder/'four_player_selection_summary.json').write_text(json.dumps([config],indent=2),encoding='utf-8')
    except RuntimeError as exc:
        status.update(initialization_status='withheld_cannot_seed_four',reason=str(exc))
    for r in rows:
        if int(r['source_frame']) != start:
            continue
        box = [round(float(r[k])*.5) for k in ('x1','y1','x2','y2')]
        x1,y1,x2,y2 = box
        cv2.rectangle(first_im,(x1,y1),(x2,y2),(0,230,230),2)
        cv2.putText(first_im,str(r['track_id']),(x1,max(20,y1-5)),0,.65,(0,230,230),2)
    cv2.imwrite(str(folder/'first_frame_all_candidates.jpg'),first_im)
    (folder/'run_status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(tag,status['initialization_status'],flush=True)
    return status


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    protocol = plan()
    spec = importlib.util.spec_from_file_location('frozen_seed_selector',SOURCE/'scripts/96_select_four_players_pilot.py')
    selector = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(selector)
    assoc.BASE = OUT/'detector_runs'
    assoc.OUT = OUT/'results'
    assoc.OUT.mkdir(exist_ok=True)
    stage2.OUT = OUT/'crop_runs'
    stage2.OUT.mkdir(exist_ok=True)
    assoc.cost = stage2.continuity_cost
    statuses = []
    for interval in protocol['intervals']:
        status = detect(interval,selector)
        tag = interval['tag']
        if status['initialization_status'] == 'four_seeds_proposed':
            path = stage2.OUT/f'augmented_detections_{tag}.csv'
            if not path.exists():
                path = stage2.augment(tag)
            final_folder = assoc.OUT/f'players_{tag}_frozen_stage2'
            if not (final_folder/'summary.json').exists():
                summary = assoc.process(tag,detection_csv=path,variant='frozen_stage2')
                summary['parameters'].update(continuity_veto_bypass=dict(max_gap=2,max_residual_height=.40,
                    same_raw_id_required=True),min_initial_height_fraction=.50)
                (final_folder/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
            status['tracking_summary'] = json.loads((final_folder/'summary.json').read_text(encoding='utf-8'))
        statuses.append(status)
        (OUT/'validation_results.json').write_text(json.dumps(statuses,indent=2),encoding='utf-8')
    # Check the implementation/weights still match the pre-inference snapshot.
    for path,digest in protocol['sha256'].items():
        assert sha(Path(path)) == digest,path
    print(json.dumps([dict(tag=r['tag'],initialization=r['initialization_status'],
                           counts=r.get('tracking_summary',{}).get('counts')) for r in statuses],indent=2),flush=True)


if __name__ == '__main__':
    main()
