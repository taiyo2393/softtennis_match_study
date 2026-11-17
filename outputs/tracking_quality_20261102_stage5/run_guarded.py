"""Run the identity veto and render anonymous far detections separately."""
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2

from identity_guard import IdentityGuardTracker,GUARD_PARAMETERS,old
import run_temporal_experiment as runner

OUT = Path(__file__).resolve().parent
STAGE4 = OUT.parent/'tracking_quality_20261102_stage4'


def write(path,rows):
    if not rows:
        return
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w = csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def render_review(summary,tracker):
    a = old.a
    tag = summary['tag']
    dest = OUT/f'players_{tag}_temporal'
    byframe,rawframe = defaultdict(list),defaultdict(list)
    for r in a.read(dest/'selected_four.csv'):
        byframe[int(r['source_frame'])].append(r)
    for r in a.read(Path(summary['input_csv'])):
        if int(r['track_id']) >= 0 and float(r['confidence']) >= .25:
            rawframe[int(r['source_frame'])].append(r)
    cap = cv2.VideoCapture(str(a.SOURCE/'data/raw/match1.mp4'))
    cap.set(cv2.CAP_PROP_POS_FRAMES,summary['first_frame'])
    fps = cap.get(cv2.CAP_PROP_FPS)
    temp = dest/'identity_review_intermediate.mp4'
    writer = cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1920,1080))
    assert writer.isOpened()
    anonymous = []
    far = ['far_rear','far_front']
    ref = [tracker.states[s]['reference_height'] for s in far]
    negatives = tracker.states[far[0]]['negatives']
    for frame in range(summary['first_frame'],summary['last_frame']+1):
        ok,im = cap.read()
        assert ok
        im = cv2.resize(im,(1920,1080))
        labeled = byframe[frame]
        unresolved = []
        for r in sorted(rawframe[frame],key=lambda r:float(r['confidence']),reverse=True):
            if frame < summary['initialized_frame']:
                continue
            x,y = a.foot(r)/[3840,2160]
            h = a.height(r)
            if not (.27<x<.73 and .54<y<.645 and .5*min(ref)<h<2.2*max(ref)):
                continue
            if any(old.overlap_fraction(r,t) > .45 or a.box_iou(r,t) > .35 for t in labeled+unresolved):
                continue
            feature = a.appearance(im,r)
            appearance = min(a.app_distance(anchor,feature) for s in far for anchor in tracker.identity_anchors[s])
            if appearance > .65:
                continue
            if negatives and min(a.app_distance(n,feature) for n in negatives)+.06 < appearance:
                continue
            unresolved.append(r)
            anonymous.append(dict(r,player_slot='',identity_status='unresolved',
                                  identity_source='anonymous_far_detection_not_reidentified'))
        for r in unresolved:
            x1,y1,x2,y2 = [round(float(r[k])*.5) for k in ('x1','y1','x2','y2')]
            cv2.rectangle(im,(x1,y1),(x2,y2),(200,200,200),2)
            cv2.putText(im,f'F? / raw {r["track_id"]}',(x1,max(22,y1-8)),0,.60,(220,220,220),2)
        for r in labeled:
            i = a.SLOTS.index(r['player_slot'])
            x1,y1,x2,y2 = [round(float(r[k])*.5) for k in ('x1','y1','x2','y2')]
            cv2.rectangle(im,(x1,y1),(x2,y2),a.COLORS[i],2)
            cv2.putText(im,f'{a.NAMES[i]} / raw {r["track_id"]}',(x1,max(22,y1-8)),0,.65,a.COLORS[i],2)
        cv2.rectangle(im,(0,1000),(1920,1080),(0,0,0),-1)
        cv2.putText(im,f'{tag} F{frame} | partner-aware handover veto | {len(labeled)} labeled candidates',
                    (15,1028),0,.63,(255,255,255),1)
        cv2.putText(im,'GRAY F? = detector candidate with UNKNOWN identity. It is not F1 or F2.',
                    (15,1054),0,.62,(255,255,255),1)
        cv2.putText(im,'No interpolation; ambiguous handovers withheld. Same raw ID alone does not prove identity.',
                    (15,1079),0,.55,(255,255,255),1)
        writer.write(im)
    writer.release()
    cap.release()
    video = dest/'identity_guard_review_h264.mp4'
    subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y',
        '-i',str(temp),'-ss',str(summary['first_frame']/fps),'-t',str(summary['frames']/fps),
        '-i',str(a.SOURCE/'data/raw/match1.mp4'),'-map','0:v:0','-map','1:a:0?',
        '-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p',
        '-c:a','aac','-movflags','+faststart',str(video)],check=True)
    write(dest/'unresolved_far_detections.csv',anonymous)
    (dest/'guard_events.json').write_text(json.dumps(tracker.guard_events,indent=2),encoding='utf-8')
    summary['identity_guard'] = dict(parameters=GUARD_PARAMETERS,
        withheld_handover_frames=len({e['frame'] for e in tracker.guard_events}),
        withheld_proposals=len(tracker.guard_events),anonymous_detection_rows=len(anonymous),
        caution='Anonymous boxes are candidates, not confirmed players; false positives remain possible.')
    (dest/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(tag,summary['identity_guard'],flush=True)


def main():
    runner.OUT = OUT
    runner.TemporalTracker = IdentityGuardTracker
    runner.PARAMETERS = dict(old.PARAMETERS,identity_guard=GUARD_PARAMETERS)
    baseline = json.loads((STAGE4/'experiment_protocol.json').read_text(encoding='utf-8'))
    files = [OUT/'identity_guard.py',OUT/'run_guarded.py',STAGE4/'temporal_tracker.py',STAGE4/'run_temporal_experiment.py']
    plan = dict(parameters=runner.PARAMETERS,intervals=baseline['intervals'],
        source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        motivation='User reported F1/F2 swaps; stage4 is not approved identity tracking.',
        scope='Development abstention guard, not solved player reidentification')
    protocol = OUT/'experiment_protocol.json'
    if protocol.exists():
        assert json.loads(protocol.read_text(encoding='utf-8')) == plan
    else:
        protocol.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    reports = []
    for interval in plan['intervals']:
        prior = json.loads((STAGE4/f'players_{interval["tag"]}_temporal/summary.json').read_text(encoding='utf-8'))
        summary = runner.process(interval,Path(prior['input_csv']))
        render_review(summary,IdentityGuardTracker.instances[-1])
        reports.append(summary)
        (OUT/'experiment_results.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    for path,digest in plan['source_sha256'].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest


if __name__ == '__main__':
    main()
