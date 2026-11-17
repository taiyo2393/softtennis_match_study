"""Four persistent identities using joint motion/appearance assignment.

Reuse all stored detector boxes, not just a seed's original ByteTrack ID.
Torso/shorts histograms are weak appearance cues, NOT trained player ReID.
Keep ambiguous/missing assignments empty; no interpolation, box smoothing,
or identity-accuracy claims. Original CSVs and labels are never modified.
"""
import argparse
import csv
import importlib.util
import json
import math
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

SOURCE = Path(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study')
BASE = Path(__file__).resolve().parent.parent/'frontcourt_handoff_20260927'
OUT = Path(__file__).resolve().parent
SLOTS = ('near_front', 'near_rear', 'far_rear', 'far_front')
COLORS = ((80,255,80), (255,190,40), (255,80,240), (30,180,255))
NAMES = ('N1', 'N2', 'F1', 'F2')


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def foot(r):
    return np.array([float(r['bottom_center_x']), float(r['bottom_center_y'])])


def height(r):
    return float(r['y2'])-float(r['y1'])


def appearance(im, row):
    # Decode at half source resolution; independently normalize clothing parts.
    x1,y1,x2,y2 = [float(row[k])*.5 for k in ('x1','y1','x2','y2')]
    w,h = x2-x1,y2-y1
    features = []
    for a,b in ((.18,.52), (.52,.78)):
        patch = im[max(0,round(y1+a*h)):min(im.shape[0],round(y1+b*h)),
                   max(0,round(x1+.15*w)):min(im.shape[1],round(x2-.15*w))]
        if patch.size == 0:
            return None
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0,1,2], None, [12,4,4], [0,180,0,256,0,256]).ravel()
        features.append(hist/max(float(hist.sum()), 1))
    return np.array(features, dtype=np.float32)


def app_distance(a, b):
    if a is None or b is None:
        return 1.
    return float(np.mean([cv2.compareHist(x, y, cv2.HISTCMP_BHATTACHARYYA)
                          for x,y in zip(a,b)]))


def cost(state, row, feature, frame, use_appearance=True):
    gap = frame-state['frame']
    old = state['row']
    h = max(height(old),height(row))
    ratio = height(row)/max(1,height(old))
    if not .45 <= ratio <= 2.2 or gap > 90:
        return 1e6
    p = foot(row)
    y = p[1]/2160
    if state['side'] == 'near' and y < .585:
        return 1e6
    if state['side'] == 'far' and not .49 < y < .665:
        return 1e6
    prediction = foot(old)+state['velocity']*min(gap, 12)
    distance = float(np.linalg.norm(p-prediction))/max(1,h)
    gate = min(2., .55+.025*gap)
    if distance > gate:
        return 1e6
    appearance_distance = app_distance(state['template'],feature)
    if use_appearance and appearance_distance > .86:
        return 1e6
    if use_appearance and state.get('negatives'):
        # Initial non-player people (officials etc.) are explicit distractors.
        # A weak clothing cue must not win merely because motion is plausible.
        negative_distance = min(app_distance(n,feature) for n in state['negatives'])
        if negative_distance + .04 < appearance_distance:
            return 1e6
    same_raw_id = int(old['track_id']) == int(row['track_id'])
    return (.62*distance + (.75*appearance_distance if use_appearance else 0.)
            + .15*abs(math.log(ratio)) - (.20 if same_raw_id else 0.))


def joint_assignment(costs, missing_cost=1.05):
    n,m = costs.shape
    augmented = np.full((n,m+n),missing_cost)
    augmented[:,:m] = costs
    ri,ci = linear_sum_assignment(augmented)
    total = float(augmented[ri,ci].sum())
    selected = []
    for i,j in zip(ri,ci):
        if j >= m or augmented[i,j] >= missing_cost:
            continue
        alternative = augmented.copy()
        alternative[i,j] = 1e6
        ar,ac = linear_sum_assignment(alternative)
        margin = float(alternative[ar,ac].sum()-total)
        # Margin is a cost gap, not a calibrated identity confidence.
        if margin >= .10:
            selected.append((int(i),int(j),margin,float(augmented[i,j])))
    return selected


def linear_sum_assignment(matrix):
    """Exact rectangular assignment for four rows, without a SciPy dependency.

    Scan columns while storing the best cost for each subset of assigned rows.
    Each update uses the preceding column's states, so a box is never reused.
    """
    n,m = matrix.shape
    states = {0:(0., ())}
    for j in range(m):
        updated = dict(states)
        for mask,(value,pairs) in states.items():
            for i in range(n):
                if mask & (1 << i):
                    continue
                next_mask = mask | (1 << i)
                candidate = value+float(matrix[i,j])
                if next_mask not in updated or candidate < updated[next_mask][0]:
                    updated[next_mask] = (candidate, pairs+((i,j),))
        states = updated
    pairs = sorted(states[(1 << n)-1][1])
    return np.array([p[0] for p in pairs]),np.array([p[1] for p in pairs])


def box_iou(a,b):
    ax1,ay1,ax2,ay2 = [float(a[k]) for k in ('x1','y1','x2','y2')]
    bx1,by1,bx2,by2 = [float(b[k]) for k in ('x1','y1','x2','y2')]
    inter = max(0,min(ax2,bx2)-max(ax1,bx1))*max(0,min(ay2,by2)-max(ay1,by1))
    return inter/max(1,(ax2-ax1)*(ay2-ay1)+(bx2-bx1)*(by2-by1)-inter)


def process(tag, use_appearance=True, render=True, negative_guard=True,
            detection_csv=None, variant=None):
    folder = BASE/f'full_rally_{tag}'
    raw = read(Path(detection_csv) if detection_csv else folder/'bytetrack.csv')
    config = json.loads((folder/'four_player_selection_summary.json').read_text(encoding='utf-8'))[0]
    start,end = config['seed_frame'],config['analysis_ready_end_frame']
    seeds = config['seed_ids']
    byframe = defaultdict(list)
    for r in raw:
        if int(r['track_id']) >= 0 and float(r['confidence']) >= .25:
            byframe[int(r['source_frame'])].append(r)
    name = ('appearance_motion_guarded' if negative_guard else 'appearance_motion') if use_appearance else 'motion_only'
    if variant:
        name = variant
    dest = OUT/f'players_{tag}_{name}'
    dest.mkdir(parents=True,exist_ok=True)
    cap = cv2.VideoCapture(str(SOURCE/'data/raw/match1.mp4'))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES,start)
    states,selected,decisions = {},[],[]
    writer = None
    if render:
        writer = cv2.VideoWriter(str(dest/'preview_intermediate.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1920,1080))
        if not writer.isOpened():
            raise RuntimeError('Cannot create preview')
    counts = Counter()
    first_sheet = []
    for frame in range(start,end+1):
        ok,im = cap.read()
        if not ok:
            raise RuntimeError(f'Failed decoding source frame {frame}')
        im = cv2.resize(im,(1920,1080))
        rows = byframe[frame]
        features = [appearance(im,r) for r in rows]
        if frame == start:
            assignments = []
            seed_rows = [r for r in rows if int(r['track_id']) in seeds.values()]
            negatives = [f for r,f in zip(rows,features)
                         if int(r['track_id']) not in seeds.values()
                         and max(box_iou(r,s) for s in seed_rows) < .10
                         and f is not None] if negative_guard else []
            for i,slot in enumerate(SLOTS):
                j = next(j for j,r in enumerate(rows) if int(r['track_id']) == seeds[slot])
                states[slot] = dict(row=rows[j], frame=frame, velocity=np.zeros(2),
                                    side=slot.split('_')[0], template=features[j],negatives=negatives,
                                    reference_height=height(rows[j]))
                assignments.append((i,j,0.,0.))
        else:
            costs = np.array([[cost(states[s],r,f,frame,use_appearance) for r,f in zip(rows,features)]
                              for s in SLOTS]).reshape(4,len(rows))
            assignments = joint_assignment(costs)
        used = set()
        for i,j,margin,value in assignments:
            slot = SLOTS[i]
            state = states[slot]
            row = rows[j]
            assert int(row['track_id']) not in used, 'Nonunique raw identity assignment'
            used.add(int(row['track_id']))
            gap = frame-state['frame']
            oldid = int(state['row']['track_id'])
            newid = int(row['track_id'])
            if gap > 0:
                velocity = (foot(row)-foot(state['row']))/gap
                # Limit the contribution of noisy box-bottom derivatives.
                speed = np.linalg.norm(velocity)
                if speed > .10*height(row):
                    velocity *= .10*height(row)/speed
                state['velocity'] = .75*state['velocity']+.25*velocity
            if oldid != newid or gap > 1:
                decisions.append(dict(source_frame=frame, player_slot=slot,
                                      old_raw_id=oldid,new_raw_id=newid,gap_frames=max(0,gap-1),
                                      assignment_cost=value,alternative_cost_margin=margin,
                                      review_status='unreviewed_candidate'))
            state.update(row=row,frame=frame)
            selected.append(dict(row,player_slot=slot,position_source='detected_box',
                                 identity_status='candidate_not_ground_truth',
                                 assignment_cost=round(value,5),alternative_cost_margin=round(margin,5)))
            counts[slot] += 1
            box = [round(float(row[k])*.5) for k in ('x1','y1','x2','y2')]
            x1,y1,x2,y2 = box
            cv2.rectangle(im,(x1,y1),(x2,y2),COLORS[i],2)
            cv2.putText(im,f'{NAMES[i]} / raw {newid}',(x1,max(22,y1-8)),0,.65,COLORS[i],2)
            if frame == start:
                first_sheet.append((NAMES[i],seeds[slot],box))
        counts['four_simultaneous'] += len(assignments) == 4
        counts['total_observed_assignments'] += len(assignments)
        if frame == start:
            cv2.imwrite(str(dest/'seed_frame.jpg'),im)
        if frame in (start+180,start+360,start+600,end):
            cv2.imwrite(str(dest/f'preview_F{frame}.jpg'),im)
        if writer is not None:
            cv2.rectangle(im,(0,1010),(1920,1080),(0,0,0),-1)
            cv2.putText(im,f'{tag} F{frame} | {name} | N1/N2/F1/F2 = identities seeded at clip start',
                        (15,1037),0,.62,(255,255,255),1)
            cv2.putText(im,'Candidate assignments; no interpolation. Empty = withheld/missing. Colors do not certify identity.',
                        (15,1067),0,.60,(255,255,255),1)
            writer.write(im)
        if (frame-start)%240 == 0:
            print(f'{tag} {name} {frame-start}/{end-start+1}',flush=True)
    cap.release()
    if writer is not None:
        writer.release()
        subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y',
                        '-i',str(dest/'preview_intermediate.mp4'),'-ss',str(start/fps),
                        '-t',str((end-start+1)/fps),'-i',str(SOURCE/'data/raw/match1.mp4'),
                        '-map','0:v:0','-map','1:a:0?','-c:v','libx264','-preset','fast',
                        '-crf','20','-pix_fmt','yuv420p','-c:a','aac','-movflags','+faststart',
                        str(dest/'tracking_review_h264.mp4')],check=True)
    for filename,rs in (('selected_four.csv',selected),('identity_decisions.csv',decisions)):
        if not rs:
            continue
        with (dest/filename).open('w',encoding='utf-8-sig',newline='') as f:
            w = csv.DictWriter(f,fieldnames=list(rs[0]))
            w.writeheader()
            w.writerows(rs)
    summary = dict(candidate_id=tag,method=name,source_video=str(SOURCE/'data/raw/match1.mp4'),
                   first_frame=start,last_frame=end,frames=end-start+1,seeds=seeds,
                   initial_seed_boxes=first_sheet,counts=dict(counts),decision_count=len(decisions),
                   detection_csv=str(detection_csv or folder/'bytetrack.csv'),
                   warning='Assignment availability only; no independent identity reference. Match-specific geometry, manually inspect seeds and junctions. Slots are initial identities, not changing front/rear roles.',
                   parameters=dict(max_missing_frames=90,missing_cost=1.05,min_global_margin=.10,
                                   appearance='fixed initial torso/shorts HSV histograms',
                                   negative_guard=negative_guard,interpolation=False))
    (dest/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary['counts']),flush=True)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--rallies',nargs='+',default=['044','056'])
    p.add_argument('--motion-only',action='store_true')
    p.add_argument('--no-render',action='store_true')
    p.add_argument('--no-negative-guard',action='store_true')
    args = p.parse_args()
    for tag in args.rallies:
        process(tag,not args.motion_only,not args.no_render,not args.no_negative_guard)
