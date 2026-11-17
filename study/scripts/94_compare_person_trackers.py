"""Non-destructive, full-rate tracker pilot. Outputs are candidates, not ground truth."""
import argparse
import csv
import json
import os
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR'] = str(ROOT / '.ultralytics')
import cv2
import numpy as np
import yaml
from ultralytics import YOLO
import ultralytics


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--start-frame', type=int, default=428)
    p.add_argument('--frames', type=int, default=180)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode',choices=('both','bytetrack','botsort_reid'),default='both')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summaries = []
    modes=['bytetrack','botsort_reid'] if args.mode=='both' else [args.mode]
    for mode in modes:
        cfg = Path(ultralytics.__file__).parent / 'cfg/trackers' / ('bytetrack.yaml' if mode == 'bytetrack' else 'botsort.yaml')
        settings = yaml.safe_load(cfg.read_text())
        if mode == 'botsort_reid':
            settings.update(with_reid=True, model='auto', gmc_method='none')
        config = args.output / f'{mode}.yaml'
        config.write_text(yaml.safe_dump(settings), encoding='utf-8')
        model = YOLO(ROOT / 'models/yolo11m.pt')
        cap = cv2.VideoCapture(str(ROOT / 'data/raw/match1.mp4'))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        rows, counts, ids = [], [], set()
        writer = None
        started = time.perf_counter()
        for i in range(args.frames):
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            result = model.track(frame, persist=True, classes=[0], conf=0.25,
                                 imgsz=1280, tracker=str(config), device=0, verbose=False)[0]
            preview = cv2.resize(frame, (1280, round(h * 1280 / w)))
            polygon = np.array([[.02,.90],[.98,.90],[.68,.51],[.32,.51]], dtype=np.float32)
            count = 0
            for box in result.boxes:
                x1,y1,x2,y2 = box.xyxy[0].cpu().tolist()
                x, y = (x1+x2)/2, y2
                in_initial_roi = cv2.pointPolygonTest(polygon, (x/w,y/h), False) >= 0
                tid = int(box.id.item()) if box.id is not None else -1
                if tid >= 0:
                    ids.add(tid)
                count += int(in_initial_roi)
                rows.append([args.start_frame+i,(args.start_frame+i)/fps,tid,x1,y1,x2,y2,x,y,float(box.conf.item()),int(in_initial_roi)])
                scale = 1280/w
                if in_initial_roi or tid in (1,2,3,4):
                    color = (0,220,255) if in_initial_roi else (255,130,0)
                    cv2.rectangle(preview,(int(x1*scale),int(y1*scale)),(int(x2*scale),int(y2*scale)),color,2)
                    cv2.putText(preview,str(tid),(int(x1*scale),int(y1*scale)-5),cv2.FONT_HERSHEY_SIMPLEX,.7,color,2)
            counts.append(count)
            cv2.putText(preview,f'{mode} F{args.start_frame+i}',(20,30),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
            if writer is None:
                writer = cv2.VideoWriter(str(args.output / f'{mode}.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(preview.shape[1],preview.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError('Preview writer failed')
            writer.write(preview)
        cap.release()
        if writer:
            writer.release()
        with (args.output / f'{mode}.csv').open('w',newline='',encoding='utf-8-sig') as f:
            writer_csv=csv.writer(f)
            writer_csv.writerow(['source_frame','timestamp_sec','track_id','x1','y1','x2','y2','bottom_center_x','bottom_center_y','confidence','in_initial_roi'])
            writer_csv.writerows(rows)
        summaries.append(dict(mode=mode,frames=len(counts),source_fps=fps,source_width=w,source_height=h,
                              unique_detection_track_ids=len(ids),frames_with_at_least_four_roi_candidates=sum(n >= 4 for n in counts),
                              elapsed_sec=time.perf_counter()-started))
        print(json.dumps(summaries[-1]),flush=True)
    (args.output/'summary.json').write_text(json.dumps({'start_frame':args.start_frame,'warning':'Candidate counts are not tracking accuracy. Bounding-box bottom centers are approximate ground positions. Court ROI is match1-specific; no human labels modified.','results':summaries},indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
