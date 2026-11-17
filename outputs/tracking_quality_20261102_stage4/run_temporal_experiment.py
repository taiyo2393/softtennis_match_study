"""Compare temporal recovery against the retained frozen stage-2 run."""
import csv
import hashlib
import json
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from temporal_tracker import TemporalTracker, PARAMETERS, a, previous

OUT = Path(__file__).resolve().parent
OLD = OUT.parent/'tracking_quality_20261102_stage3'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_five(interval):
    dest = OUT/'crop_inputs/full_rally_005'
    dest.mkdir(parents=True,exist_ok=True)
    shutil.copyfile(OLD/'detector_runs/full_rally_005/bytetrack.csv',dest/'bytetrack.csv')
    config = [dict(seed_frame=interval['first_frame'],analysis_ready_end_frame=interval['last_frame'])]
    (dest/'four_player_selection_summary.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    a.BASE = OUT/'crop_inputs'
    previous.OUT = OUT/'crop_runs'
    previous.OUT.mkdir(exist_ok=True)
    path = previous.OUT/'augmented_detections_005.csv'
    if not path.exists():
        previous.augment('005')
    return path


def process(interval,path):
    tag,start,end = interval['tag'],interval['first_frame'],interval['last_frame']
    dest = OUT/f'players_{tag}_temporal'
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'summary.json').exists():
        return json.loads((dest/'summary.json').read_text(encoding='utf-8'))
    raw = a.read(path)
    byframe = defaultdict(list)
    for r in raw:
        if int(r['track_id']) >= 0 and float(r['confidence']) >= .25:
            byframe[int(r['source_frame'])].append(r)
    cap = cv2.VideoCapture(str(a.SOURCE/'data/raw/match1.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    fps = cap.get(cv2.CAP_PROP_FPS)
    temp = dest/'preview_intermediate.mp4'
    writer = cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1920,1080))
    assert writer.isOpened()
    tracker = TemporalTracker()
    selected,frame_status = [],[]
    counts = Counter()
    for frame in range(start,end+1):
        ok,im = cap.read()
        assert ok,frame
        im = cv2.resize(im,(1920,1080))
        rows = byframe[frame]
        features = [a.appearance(im,r) for r in rows]
        assignments = tracker.step(frame,rows,features)
        assert len({i for i,_,_,_,_ in assignments}) == len(assignments)
        assert len({j for _,j,_,_,_ in assignments}) == len(assignments)
        for i,j,margin,value,origin in assignments:
            r = rows[j]
            selected.append(dict(r,player_slot=a.SLOTS[i],position_source='detected_box',
                identity_status='candidate_not_ground_truth',identity_source=origin,
                identity_review_status='requires_identity_review',assignment_cost=round(value,5),
                alternative_cost_margin=round(margin,5)))
            counts[a.SLOTS[i]] += 1
            counts['total_observed_assignments'] += 1
            x1,y1,x2,y2 = [round(float(r[k])*.5) for k in ('x1','y1','x2','y2')]
            cv2.rectangle(im,(x1,y1),(x2,y2),a.COLORS[i],2)
            label = f'{a.NAMES[i]} / raw {r["track_id"]}'
            if origin == 'gallery_reacquisition_candidate':
                label += ' R?'
            cv2.putText(im,label,(x1,max(22,y1-8)),0,.65,a.COLORS[i],2)
        counts['four_simultaneous'] += len(assignments) == 4
        counts['pre_initialization_frames'] += tracker.initialized_frame is None
        frame_status.append(dict(source_frame=frame,initialized=tracker.initialized_frame is not None,
                                 assigned_players=len(assignments),pending_recovery=len(tracker.pending)))
        if frame == tracker.initialized_frame:
            cv2.imwrite(str(dest/'initialization_confirmed.jpg'),im)
        if frame in (start+180,start+360,start+600,end):
            cv2.imwrite(str(dest/f'preview_F{frame}.jpg'),im)
        cv2.rectangle(im,(0,1000),(1920,1080),(0,0,0),-1)
        cv2.putText(im,f'{tag} F{frame} | temporal initialization + gallery reacquisition | {len(assignments)}/4 candidates',
                    (15,1028),0,.60,(255,255,255),1)
        message = ('Waiting for 8 source frames with four separable seeds' if tracker.initialized_frame is None
                   else 'R? = unreviewed reacquisition proposal. Missing positions remain empty.')
        cv2.putText(im,message,(15,1054),0,.60,(255,255,255),1)
        cv2.putText(im,'Clothing appearance is a weak cue; no independent identity accuracy has been measured.',
                    (15,1079),0,.55,(255,255,255),1)
        writer.write(im)
        if (frame-start)%240 == 0:
            print(tag,'temporal',frame-start,'init',tracker.initialized_frame,
                  'recoveries',sum(e['event']=='reacquired_candidate' for e in tracker.events),flush=True)
    cap.release()
    writer.release()
    video = dest/'tracking_review_h264.mp4'
    subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y',
        '-i',str(temp),'-ss',str(start/fps),'-t',str((end-start+1)/fps),'-i',str(a.SOURCE/'data/raw/match1.mp4'),
        '-map','0:v:0','-map','1:a:0?','-c:v','libx264','-preset','fast','-crf','20',
        '-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',str(video)],check=True)
    for name,rs in [('selected_four.csv',selected),('frame_status.csv',frame_status)]:
        if rs:
            with (dest/name).open('w',encoding='utf-8-sig',newline='') as f:
                w = csv.DictWriter(f,fieldnames=list(rs[0]))
                w.writeheader()
                w.writerows(rs)
    (dest/'events.json').write_text(json.dumps(tracker.events,indent=2),encoding='utf-8')
    summary = dict(tag=tag,first_frame=start,last_frame=end,frames=end-start+1,
                   initialized_frame=tracker.initialized_frame,counts=dict(counts),
                   reacquisition_proposals=sum(e['event']=='reacquired_candidate' for e in tracker.events),
                   parameters=PARAMETERS,input_csv=str(path),input_sha256=sha(path),
                   warning='Development on formerly inspected same-match intervals; candidate availability is not identity accuracy. No retroactive output or interpolation.')
    (dest/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)
    return summary


def main():
    OUT.mkdir(exist_ok=True)
    baseline = json.loads((OLD/'predeclared_protocol.json').read_text(encoding='utf-8'))
    sources = [OUT/'temporal_tracker.py',OUT/'run_temporal_experiment.py',
               OUT.parent/'tracking_quality_20261102_stage2/improve_players.py',
               OUT.parent/'tracking_quality_20261102/four_player_association.py']
    protocol = dict(parameters=PARAMETERS,intervals=baseline['intervals'],
                    sources_sha256={str(p):sha(p) for p in sources},
                    baseline_sha256=sha(OLD/'predeclared_protocol.json'),
                    scope='New development version; stage3 intervals are now development data.')
    planpath = OUT/'experiment_protocol.json'
    if planpath.exists():
        assert json.loads(planpath.read_text(encoding='utf-8')) == protocol
    else:
        planpath.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    reports = []
    for interval in protocol['intervals']:
        tag = interval['tag']
        path = prepare_five(interval) if tag == '005' else OLD/'crop_runs'/f'augmented_detections_{tag}.csv'
        reports.append(process(interval,path))
        (OUT/'experiment_results.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    for path,digest in baseline['sha256'].items():
        assert sha(Path(path)) == digest,path
    for path,digest in protocol['sources_sha256'].items():
        assert sha(Path(path)) == digest,path


if __name__ == '__main__':
    main()
