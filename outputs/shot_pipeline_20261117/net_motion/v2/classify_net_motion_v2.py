"""Front-player motion v2. Inputs and v1 artifacts are read-only.

Run study/.venv/Scripts/python.exe -X utf8 -B -u <this file>.
The classifier below is adapted from v1; v1 helpers are imported without calling
its writers or main. All generated files are confined to this script's directory.
"""
from __future__ import annotations
import importlib.util
import json
import math
from fractions import Fraction
from pathlib import Path
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
V1 = OUT.parent
BASE = V1.parent
ROOT = BASE.parent.parent
spec = importlib.util.spec_from_file_location("net_motion_v1", V1 / "classify_net_motion.py")
v1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v1)
WIDTH, LENGTH, NET = v1.WIDTH, v1.LENGTH, v1.NET
SLOTS, NAMES = v1.SLOTS, v1.NAMES
DIR3, DIR8, JA, PAPER = v1.DIR3, v1.DIR8, v1.JA, v1.PAPER
WS, DS, MAIN_W, MAIN_D = v1.WS, v1.DS, 0.4, 0.3
PRE_FRAMES, MIN_PRE = v1.PRE_FRAMES, v1.MIN_PRE
MAX_MISSING, MAX_GAP = v1.MAX_MISSING, v1.MAX_GAP
FEINT_MIN, FEINT_MAX = v1.FEINT_MIN, v1.FEINT_MAX
ANKLE_CONF = 0.3
MEDIAN_FRAMES, MEAN_FRAMES = 5, 3
TURN_DEG, MIN_SEG_FRAMES = 75.0, 4
SEGMENT_DISTANCE_FACTOR = 1.0
PROJECTION_ATOL = 1e-10
SHEET_STRIDE = 6
MAX_EXTRA_REVIEW = 10
truth, finite_xy, long_gap = v1.truth, v1.finite_xy, v1.long_gap
angle, directions = v1.angle, v1.directions
CAL = ROOT / "outputs/ball_quality_20261110/match2_camera_transfer_pilot/protocol.json"


def save(df, name):
    df.to_csv(OUT / name, index=False, encoding="utf-8-sig", float_format="%.10g")


def smooth_positions(raw, segments):
    """Median5 then mean3; missing/identity boundaries are never crossed.

    Each median needs >=3 observations, each truncated mean >=2. Both passes
    preserve missing timestamps; all support is restricted to [c-6,end].
    """
    values = raw.copy()
    for width, reducer, minimum in [(MEDIAN_FRAMES, np.median, 3),
                                    (MEAN_FRAMES, np.mean, 2)]:
        valid = np.isfinite(values).all(axis=1)
        result = np.full_like(values, np.nan)
        for i in range(len(values)):
            if not valid[i]:
                continue
            lo, hi = i, i+1
            while lo > max(0, i-width//2) and valid[lo-1] and segments[lo-1] == segments[i]:
                lo -= 1
            while hi < min(len(values), i+width//2+1) and valid[hi] and segments[hi] == segments[i]:
                hi += 1
            if hi-lo >= minimum:
                result[i] = reducer(values[lo:hi], axis=0)
        values = result
    return values


def split_motion(p, frames, d):
    def rec(i, j):
        best_k, best_angle = None, TURN_DEG-1e-9
        for k in range(i+1, j):
            a, b = p[k]-p[i], p[j]-p[k]
            if (frames[k]-frames[i] < MIN_SEG_FRAMES or frames[j]-frames[k] < MIN_SEG_FRAMES
                    or np.linalg.norm(a) < d*SEGMENT_DISTANCE_FACTOR
                    or np.linalg.norm(b) < d*SEGMENT_DISTANCE_FACTOR):
                continue
            deg = angle(a, b)
            if deg > best_angle:
                best_k, best_angle = k, deg
        return [i,j] if best_k is None else rec(i,best_k)[:-1] + rec(best_k,j)
    breaks = rec(0,len(p)-1)
    # Audit final recursive leaves as well as the candidate split.
    if len(breaks) > 2:
        assert all(frames[j]-frames[i] >= MIN_SEG_FRAMES and
                   np.linalg.norm(p[j]-p[i]) >= d*SEGMENT_DISTANCE_FACTOR
                   for i,j in zip(breaks[:-1],breaks[1:]))
    return breaks


def classify(base, tracks, protocol, w, d):
    r = dict(base)
    c = int(r['contact_frame'])
    fps = float(Fraction(protocol['source_fps']))
    end = c+int(round(w*fps))
    rule = 'W_cap'
    if np.isfinite(r['next_contact_frame']):
        n = int(r['next_contact_frame'])
        if n <= end:
            end,rule = n,'next_label_cap'
    r.update(W_sec=w,D_m=d,fps=fps,end_frame=end,end_rule=rule,
             actual_duration_sec=(end-c)/fps, classification='unclassifiable',
             unclassifiable_reason='', motion_present=np.nan,
             start_x_m=np.nan,start_y_m=np.nan,end_x_m=np.nan,end_y_m=np.nan,
             pre_valid_frames=0,pre_max_displacement_m=np.nan,pre_path_m=np.nan,
             positioning_stationary=np.nan,trajectory_missing_fraction=np.nan,
             trajectory_max_missing_gap=0,track_ids='',track_id_changed=False,
             segment_changed=False,path_length_m=np.nan,max_displacement_m=np.nan,
             net_displacement_m=np.nan,max_frame_step_m=np.nan,steps_over_1m=0,
             delta_right_m=np.nan,delta_front_m=np.nan,
             n_segments=0,direction3='',direction8='',angle_deg=np.nan,
             preliminary_direction3='',preliminary_direction8='',preliminary_angle_deg=np.nan,
             preliminary_directions_json='[]',final_direction3='',final_direction8='',final_angle_deg=np.nan,
             feint=False,feint_angle_deg=np.nan,segments_json='[]')
    if r['role_issue']:
        r['unclassifiable_reason']=r['role_issue']
        return r,None
    t = tracks[r['front_slot']]
    # Limit smoothing support to this event (no observations beyond the endpoint).
    idx = np.arange(c-PRE_FRAMES,end+1)
    a = t.reindex(idx)
    raw = a[['x_m','y_m']].to_numpy(float)
    valid = np.isfinite(raw).all(axis=1)
    smooth = smooth_positions(raw, a.segment.to_numpy())
    pre = raw[:PRE_FRAMES]
    pv = valid[:PRE_FRAMES]
    r['pre_valid_frames']=int(pv.sum())
    reasons=[]
    if pv.sum()<MIN_PRE:
        reasons.append('insufficient_positioning_frames')
    if pv.any():
        start = np.median(pre[pv],axis=0)
        r['start_x_m'],r['start_y_m']=start
        r['pre_max_displacement_m']=float(np.linalg.norm(pre[pv]-pre[pv][0],axis=1).max())
        r['pre_path_m']=float(sum(np.linalg.norm(pre[i]-pre[i-1]) for i in range(1,PRE_FRAMES) if pv[i] and pv[i-1]))
        r['positioning_stationary']=int(r['pre_max_displacement_m']<d)
    else:
        start=np.array([np.nan,np.nan])
    post = smooth[PRE_FRAMES:]
    ok = np.isfinite(post).all(axis=1)
    r['trajectory_missing_fraction']=float(1-ok.mean())
    r['trajectory_max_missing_gap']=long_gap(~ok)
    if 1-ok.mean()>MAX_MISSING+1e-9:
        reasons.append('trajectory_missing_over_20pct')
    if long_gap(~ok)>MAX_GAP:
        reasons.append('trajectory_gap_over_2frames')
    if not ok[0] or not ok[-1]:
        reasons.append('trajectory_endpoint_missing')
    segments=a.segment.dropna().unique()
    r['segment_changed']=len(segments)>1
    if len(segments)>1:
        reasons.append('tracking_segment_boundary')
    ids=a.track_id.dropna().astype(int).unique()
    r.update(track_ids=';'.join(map(str,ids)),track_id_changed=len(ids)>1)
    if end<=c:
        reasons.append('nonpositive_duration')
    r['unclassifiable_reason']=';'.join(reasons)
    if reasons:
        return r,None
    endxy=post[-1]
    r['end_x_m'],r['end_y_m']=endxy
    observed=post[ok]
    frames=np.arange(c,end+1)[ok]
    # Anchor at virtual c-1 to avoid two distinct points with the same timestamp.
    p=np.vstack([start,observed]); frames=np.r_[c-1,frames]
    sign=1 if r['front_side']=='N' else -1
    local=(p-start)*sign
    r['path_length_m']=float(np.linalg.norm(post[0]-start)+sum(
        np.linalg.norm(post[i]-post[i-1]) for i in range(1,len(post)) if ok[i] and ok[i-1]))
    steps=[np.linalg.norm(post[i]-post[i-1]) for i in range(1,len(post)) if ok[i] and ok[i-1]]
    r['max_frame_step_m']=float(max(steps,default=0))
    r['steps_over_1m']=sum(step>1.0 for step in steps)
    r['max_displacement_m']=float(np.linalg.norm(p-start,axis=1).max())
    r['net_displacement_m']=float(np.linalg.norm(endxy-start))
    r['delta_right_m'],r['delta_front_m']=local[-1]
    geometry={'p':p,'frames':frames,'local':local}
    if r['max_displacement_m']<d:
        r.update(classification='none',motion_present=0)
        return r,geometry
    breaks=split_motion(local,frames,d)
    vectors=[local[j]-local[i] for i,j in zip(breaks[:-1],breaks[1:])]
    # A negligible final resultant after an excursion is explicitly unresolved.
    if any(np.linalg.norm(v)<1e-9 for v in vectors):
        r['unclassifiable_reason']='zero_resultant_direction'
        return r,geometry
    r.update(classification='single' if len(vectors)==1 else 'compound',motion_present=1,
             n_segments=len(vectors))
    r['direction3'],r['direction8'],r['angle_deg']=directions(local[-1])
    desc=[]
    for i,j,v in zip(breaks[:-1],breaks[1:],vectors):
        d3,d8,theta=directions(v)
        desc.append(dict(start_frame=int(frames[i]),end_frame=int(frames[j]),
                         right_m=float(v[0]),front_m=float(v[1]),length_m=float(np.linalg.norm(v)),
                         direction3=d3,direction8=d8,angle_deg=theta))
    r['segments_json']=json.dumps(desc,ensure_ascii=False)
    r['final_direction3'],r['final_direction8'],r['final_angle_deg']=directions(vectors[-1])
    r['preliminary_directions_json']=json.dumps(desc[:-1],ensure_ascii=False)
    if len(vectors)>1:
        r['preliminary_direction3'],r['preliminary_direction8'],r['preliminary_angle_deg']=directions(vectors[0])
        # The immediately preceding segment is the preparatory action for the final one.
        a8=desc[-2]['direction8'];b8=desc[-1]['direction8']
        turn=angle(vectors[-2],vectors[-1]);r['feint_angle_deg']=turn
        r['feint']=({a8,b8}=={'left_front','right_front'} and FEINT_MIN<=turn<=FEINT_MAX)
    return r,geometry


def project(H, xy):
    """Same homogeneous division as build_044_spatiotemporal_pilot.project."""
    p = np.c_[xy, np.ones(len(xy))] @ H.T
    assert np.all(np.abs(p[:, 2]) > 1e-9)
    return p[:, :2] / p[:, 2:]


def load_coordinates(tracks, protocols):
    calibration = json.loads(CAL.read_text(encoding='utf-8-sig'))
    H = np.asarray(calibration['image_to_court_homography_canonical'], float)
    canonical = np.asarray(calibration['canonical_dimensions'], float)
    assert tuple(canonical) == (3840, 2160)
    ankles, checks, inputs = {}, [], [CAL]
    for point, slots in tracks.items():
        cap = cv2.VideoCapture(protocols[point]['source_video'])
        assert cap.isOpened()
        native = np.array([cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)])
        cap.release()
        assert tuple(native) == (1920, 1080)
        scale = canonical / native
        # This assertion must precede any projection of pose coordinates.
        for slot, t in slots.items():
            xy = t[['bottom_center_x', 'bottom_center_y']].to_numpy(float)
            reproduced = project(H, xy * scale)
            expected = t[['court_proxy_x', 'court_proxy_y']].to_numpy(float)
            np.testing.assert_allclose(reproduced, expected, rtol=0, atol=PROJECTION_ATOL,
                                       err_msg=f'court_proxy round trip: {point} {slot}')
            checks.append(dict(point=point, slot=slot, n=len(t),
                               max_abs_error_normalized=float(np.abs(reproduced-expected).max()),
                               scale_x=scale[0], scale_y=scale[1], assertion_passed=True))
        ankles[point] = {}
        for path in sorted((BASE/'pose/pose_cache').glob(f'point_{point:02d}_rally_*.csv')):
            inputs.append(path)
            pose = pd.read_csv(path, keep_default_na=True)
            assert not pose.duplicated(['source_frame','player_slot']).any()
            rally = path.stem.split('_rally_')[1]
            lo, hi = int(pose.source_frame.min()), int(pose.source_frame.max())
            prepared = {}
            for slot, t in slots.items():
                a = t.copy()
                k = pose[pose.player_slot == slot].set_index('source_frame').reindex(t.index)
                inside = (a.index >= lo) & (a.index <= hi)
                left = k[['kp_15_x','kp_15_y']].to_numpy(float)
                right = k[['kp_16_x','kp_16_y']].to_numpy(float)
                lv = (k.kp_15_conf.to_numpy(float) >= ANKLE_CONF) & np.isfinite(left).all(axis=1)
                rv = (k.kp_16_conf.to_numpy(float) >= ANKLE_CONF) & np.isfinite(right).all(axis=1)
                foot = np.full_like(left, np.nan)
                foot[lv & rv] = (left[lv & rv] + right[lv & rv])/2
                foot[lv & ~rv] = left[lv & ~rv]
                foot[rv & ~lv] = right[rv & ~lv]
                sources = np.full(len(a), 'ankles_missing', dtype=object)
                sources[lv & rv] = 'ankle_midpoint'
                sources[lv & ~rv] = 'left_ankle_only'
                sources[rv & ~lv] = 'right_ankle_only'
                # Only cache absence/outside its frame range permits box fallback.
                # Low confidence or a missing slot inside the cache remains missing.
                foot[~inside] = a.loc[~inside, ['bottom_center_x','bottom_center_y']].to_numpy(float)
                sources[~inside] = 'bbox_cache_outside'
                valid = np.isfinite(foot).all(axis=1)
                court = np.full_like(foot, np.nan)
                court[valid] = project(H, foot[valid] * scale) * [WIDTH, LENGTH]
                a['x_m'], a['y_m'] = court.T
                a['foot_image_x'], a['foot_image_y'] = foot.T
                a['foot_source'] = sources
                prepared[slot] = a
            ankles[point][rally] = prepared
    return ankles, checks, inputs


def prepare(shot, nxt, tracks, protocol):
    r = v1.prepare(shot, nxt, tracks, protocol)
    r.update(role_selection_basis='nearest_net_at_contact', receiver_slot='', receiver_basis='not_serve')
    if not truth(shot.is_serve):
        return r
    side = r['front_side']
    inverse = {name: slot for slot, name in NAMES.items()}
    receiver = inverse.get(str(nxt.hitter_slot)) if nxt is not None else None
    if receiver in SLOTS[side]:
        basis = 'next_label_hitter_slot'
    else:
        receiver = None
        if nxt is not None:
            candidates = []
            for slot in SLOTS[side]:
                t = tracks[slot]
                f = int(nxt.contact_frame)
                p = t.loc[f] if f in t.index else None
                if finite_xy(p):
                    # Distance to the receiving side's baseline, never to the ball.
                    baseline = 0.0 if side == 'N' else LENGTH
                    candidates.append((abs(float(p.y_m)-baseline),slot))
            if len(candidates) == 2:
                receiver = min(candidates)[1]
        basis = 'nearest_baseline_at_next_contact' if receiver else 'receiver_unresolved'
    r['receiver_basis'] = basis
    r['role_selection_basis'] = 'serve_nonreceiver_' + basis
    if receiver is None:
        r.update(front_slot='', role_issue='serve_receiver_unresolved', touch_next=np.nan,
                 touch_reason='front_role_unknown')
        return r
    slot = next(s for s in SLOTS[side] if s != receiver)
    t = tracks[slot]
    c = int(shot.contact_frame)
    p = t.loc[c] if c in t.index else None
    r.update(front_slot=slot, receiver_slot=receiver, role_issue='', role_ambiguous=False,
             front_net_distance_m=abs(float(p.y_m)-NET) if finite_xy(p) else np.nan)
    # The serving receiver exclusion is a role rule, even when next-side labels conflict.
    if nxt is not None and nxt.hitter_side == side and inverse.get(str(nxt.hitter_slot)) in SLOTS[side]:
        r.update(touch_next=int(inverse[nxt.hitter_slot] == slot), touch_reason='slot_label_match')
    elif nxt is not None and nxt.hitter_side != side:
        r.update(touch_next=0, touch_reason='next_hitter_other_side_sequence_flag')
    else:
        r.update(touch_next=np.nan, touch_reason='next_hitter_slot_missing')
    return r


def event_tracks(point, rally, method, boxes, ankles):
    if method == 'ankle' and rally in ankles[point]:
        return ankles[point][rally]
    result = {}
    for slot, t in boxes[point].items():
        a = t.copy()
        a['foot_image_x'] = a.bottom_center_x
        a['foot_image_y'] = a.bottom_center_y
        a['foot_source'] = 'bbox' if method == 'bbox' else 'bbox_cache_absent'
        result[slot] = a
    return result


def classify_event(base, t, protocol, w, d, method):
    r, geometry = classify(base, t, protocol, w, d)
    r['coordinate_method'] = method
    if r['front_slot']:
        a = t[r['front_slot']].reindex(np.arange(int(r['contact_frame'])-PRE_FRAMES, r['end_frame']+1))
        sources = a.foot_source.fillna('tracking_missing')
        counts = sources.value_counts().to_dict()
        r['foot_source_counts_json'] = json.dumps({str(k): int(v) for k,v in counts.items()}, sort_keys=True)
        r['bbox_fallback_frames'] = int(sources.str.startswith('bbox_cache_').sum())
        r['ankle_valid_frames'] = int(sources.isin(['ankle_midpoint','left_ankle_only','right_ankle_only']).sum())
        r['foot_source_mode'] = ';'.join(sorted(counts))
        r['uses_bbox_fallback'] = r['bbox_fallback_frames'] > 0
        # A switch of geometric proxy can itself manufacture displacement. Keep the
        # exported raw positions but abstain from classifying mixed-proxy events.
        r['mixed_ankle_bbox'] = r['bbox_fallback_frames'] > 0 and r['ankle_valid_frames'] > 0
        if r['mixed_ankle_bbox']:
            r.update(classification='unclassifiable', motion_present=np.nan,
                     unclassifiable_reason=';'.join(filter(None,[r['unclassifiable_reason'],'mixed_ankle_bbox_proxy'])))
            geometry = None
    else:
        r.update(foot_source_counts_json='{}', bbox_fallback_frames=0, ankle_valid_frames=0,
                 foot_source_mode='role_unknown', uses_bbox_fallback=False, mixed_ankle_bbox=False)
    c = int(r['contact_frame'])
    assert r['end_frame'] <= c+round(w*r['fps'])
    if np.isfinite(r['next_contact_frame']):
        assert r['end_frame'] <= r['next_contact_frame']
    return r, geometry


def evaluate(main, old, reference):
    old = old.set_index('id')
    datasets = {'v1':old, **{f'v2_{method}':g.set_index('id') for method,g in main.groupby('coordinate_method')}}
    records, scores = [], []
    for version, data in datasets.items():
        for _, ref in reference.iterrows():
            r = data.loc[ref.id]
            oldslot = old.loc[ref.id, 'front_slot']
            wrong = 'wrong_player' in ref.claude_issue.split(';')
            side = old.loc[ref.id, 'front_side']
            expected = next(s for s in SLOTS[side] if s != oldslot) if wrong else oldslot
            motion = 'motion' if r.motion_present == 1 else ('none' if r.motion_present == 0 else 'unclassifiable')
            item = dict(version=version, review_id=ref.review_id, id=ref.id,
                        claude_motion=ref.claude_motion, claude_direction3=ref.claude_direction3,
                        claude_single_compound=ref.claude_single_compound, claude_issue=ref.claude_issue,
                        classification=r.classification, predicted_motion=motion, direction3=r.final_direction3,
                        front_slot=r.front_slot, expected_front_slot=expected,
                        role_reference_basis='other_of_two_from_wrong_player' if wrong else 'v1_slot_no_wrong_player_flag',
                        role_match=int(r.front_slot==expected),
                        motion_match=int(motion==ref.claude_motion) if ref.claude_motion else np.nan,
                        direction3_match=int(motion=='motion' and r.final_direction3==ref.claude_direction3)
                            if ref.claude_motion=='motion' and ref.claude_direction3 else np.nan,
                        complexity_match=int(r.classification==ref.claude_single_compound)
                            if ref.claude_single_compound else np.nan,
                        same_person_as_motion_reference=r.front_slot==oldslot,
                        unclassifiable_reason=r.unclassifiable_reason)
            records.append(item)
    detail = pd.DataFrame(records)
    for version, g in detail.groupby('version',sort=False):
        for metric, col in [('motion','motion_match'),('direction3','direction3_match'),
                            ('single_compound','complexity_match'),('front_selection','role_match')]:
            v = g[col].dropna()
            scores.append(dict(version=version, metric=metric, matches=int(v.sum()), denominator=len(v),
                               percent=100*v.mean() if len(v) else np.nan,
                               note='role reference partly inferred; missing predictions count as mismatch'))
    return detail, pd.DataFrame(scores)


def select_review(main, old, reference):
    ankle = main[main.coordinate_method=='ankle'].set_index('id')
    old = old.set_index('id')
    changes = []
    for ident, r in ankle.iterrows():
        a = old.loc[ident]
        fields = [k for k in ['classification','final_direction3','final_direction8','front_slot'] if r[k] != a[k]]
        if fields:
            changes.append(dict(id=ident, changed_fields=';'.join(fields),
                                v1_classification=a.classification, v2_classification=r.classification,
                                v1_direction3=a.final_direction3, v2_direction3=r.final_direction3,
                                v1_slot=a.front_slot, v2_slot=r.front_slot))
    changes = pd.DataFrame(changes)
    rows = []
    for _, ref in reference.iterrows():
        r = ankle.loc[ref.id].to_dict()
        rows.append(dict(r, id=ref.id, review_id=ref.review_id, selection_basis='same_v1_review20'))
    # Deterministic: movement/complexity changes first, then role, 3dir, 8dir;
    # then point/shot order. These extra cases have no reference labels yet.
    pool = changes[~changes.id.isin(reference.id)].copy()
    pool['priority'] = pool.changed_fields.map(lambda s: min(
        {'classification':0,'front_slot':1,'final_direction3':2,'final_direction8':3}[k] for k in s.split(';')))
    pool = pool.sort_values(['priority','id']).head(MAX_EXTRA_REVIEW)
    for n, ident in enumerate(pool.id, 21):
        rows.append(dict(ankle.loc[ident].to_dict(), id=ident, review_id=f'motion_{n:02d}',
                         selection_basis='v2_changed_deterministic'))
    return pd.DataFrame(rows), changes


def review_sheets(selected, boxes, ankles, protocols):
    folder = OUT/'review_sheets'
    folder.mkdir(exist_ok=True)
    schedules, wanted = {}, {}
    for _, r in selected.iterrows():
        frames = list(range(int(r.contact_frame),int(r.end_frame)+1,SHEET_STRIDE))
        if frames[-1] != int(r.end_frame): frames.append(int(r.end_frame))
        schedules[r.id] = frames
        wanted.setdefault(int(r.point),set()).update(frames)
    images = {}
    for point, requested in wanted.items():
        cap = cv2.VideoCapture(protocols[point]['source_video'])
        assert cap.isOpened()
        assert cap.set(cv2.CAP_PROP_POS_FRAMES,min(requested))
        for f in range(min(requested),max(requested)+1):
            assert cap.grab(), (point,f)
            if f in requested:
                ok, img = cap.retrieve()
                assert ok and round(cap.get(cv2.CAP_PROP_POS_FRAMES)) == f+1
                images[point,f] = img
        cap.release()
    records, paths = [], []
    for _, r in selected.iterrows():
        point, c, end = int(r.point), int(r.contact_frame), int(r.end_frame)
        frames = schedules[r.id]
        tracks = event_tracks(point, r.rally, 'ankle', boxes, ankles)
        slot = r.front_slot
        t = tracks[slot] if slot else None
        idx = np.arange(c-PRE_FRAMES,end+1)
        a = t.reindex(idx) if t is not None else None
        # Use the classification's smoothed court positions, inverse-project them
        # for overlay; also show the raw ankle observations to expose smoothing.
        path = np.full((len(idx),2),np.nan)
        raw = path.copy()
        if a is not None:
            raw = a[['foot_image_x','foot_image_y']].to_numpy(float)
            court = smooth_positions(a[['x_m','y_m']].to_numpy(float), a.segment.to_numpy())
            valid = np.isfinite(court).all(axis=1)
            H = np.asarray(json.loads(CAL.read_text(encoding='utf-8-sig'))['image_to_court_homography_canonical'])
            path[valid] = project(np.linalg.inv(H),court[valid]/[WIDTH,LENGTH])/2
        for i, f in enumerate(idx):
            paths.append(dict(review_id=r.review_id,id=r.id,source_frame=int(f),
                foot_source=a.foot_source.iloc[i] if a is not None else 'role_unknown',
                raw_image_x=raw[i,0],raw_image_y=raw[i,1],
                smoothed_image_x=path[i,0],smoothed_image_y=path[i,1]))
        canvas = np.full((390,320+400*len(frames),3),245,np.uint8)
        lines = [f'{r.review_id} {r.id}',f'slot={NAMES.get(slot,"unknown")}',
                 f'{r.classification} / {r.final_direction3}',f'8dir={r.final_direction8}',
                 f'Max={r.max_displacement_m:.2f}m',f'F{c}-{end}; W=0.4 D=0.3',
                 f'fallback frames={r.bbox_fallback_frames}',f'p8near={int(r.exclude_p8_near)}',
                 'cyan: smoothed foot path','gray: raw ankle/box path','yellow: raw current foot',
                 'magenta: smoothed current',f'{r.selection_basis}']
        for j,line in enumerate(lines):
            cv2.putText(canvas,line,(8,24+26*j),0,.48,(20,20,20),1,cv2.LINE_AA)
        for k,f in enumerate(frames):
            img = images[point,f]
            h, w = img.shape[:2]
            box = t.loc[f] if t is not None and f in t.index else None
            if box is None and t is not None:
                nearest = t.index[np.argmin(np.abs(t.index-f))]
                box = t.loc[nearest]
            cx, cy = ((box.x1+box.x2)/2,(box.y1+box.y2)/2) if box is not None else (w/2,h/2)
            left = max(0,min(w-400,int(round(cx-200))))
            top = max(0,min(h-300,int(round(cy-150))))
            tile = img[top:top+300,left:left+400].copy()
            for trajectory, color, thickness in [(raw,(155,155,155),1),(path,(255,220,0),2)]:
                for j in range(PRE_FRAMES+1,len(idx)):
                    if np.isfinite(trajectory[j-1:j+1]).all():
                        p0,p1 = np.rint(trajectory[j-1:j+1]-[left,top]).astype(int)
                        cv2.line(tile,tuple(p0),tuple(p1),color,thickness,cv2.LINE_AA)
            if box is not None:
                cv2.rectangle(tile,(int(box.x1)-left,int(box.y1)-top),
                              (int(box.x2)-left,int(box.y2)-top),(0,220,60),1)
            i = f-idx[0]
            for xy,color in [(raw[i],(0,255,255)),(path[i],(255,0,255))]:
                if np.isfinite(xy).all():
                    cv2.circle(tile,tuple(np.rint(xy-[left,top]).astype(int)),4,color,-1)
            if not np.isfinite(raw[i]).all(): cv2.putText(tile,'FOOT MISSING',(8,28),0,.65,(0,0,255),2)
            x = 320+400*k
            canvas[48:348,x:x+400] = tile
            cv2.putText(canvas,f'F{f} crop({left},{top})',(x+8,25),0,.5,(20,20,20),1,cv2.LINE_AA)
            source = str(a.foot_source.iloc[i]) if a is not None else 'role_unknown'
            cv2.putText(canvas,source,(x+8,375),0,.48,(20,20,20),1,cv2.LINE_AA)
        name = r.review_id+'.jpg'
        assert cv2.imwrite(str(folder/name),canvas,[cv2.IMWRITE_JPEG_QUALITY,88])
        records.append(dict(review_id=r.review_id,id=r.id,point=point,front_slot=slot,
            contact_frame=c,end_frame=end,auto_classification=r.classification,
            auto_direction3=r.final_direction3,auto_direction8=r.final_direction8,
            selection_basis=r.selection_basis,role_selection_basis=r.role_selection_basis,
            foot_source_mode=r.foot_source_mode,bbox_fallback_frames=r.bbox_fallback_frames,
            sheet='review_sheets/'+name,sampled_frames=';'.join(map(str,frames)),
            claude_motion='',claude_direction3='',claude_single_compound='',claude_issue='',claude_note=''))
    save(pd.DataFrame(paths),'review_screen_trajectories.csv')
    review = pd.DataFrame(records)
    save(review,'review_template.csv')
    links = '\n'.join(f'<h2>{r.review_id}: {r.id}</h2><p>{r.auto_classification} / {r.auto_direction3}</p>'
                      f'<a href="{r.sheet}"><img style="max-width:100%" src="{r.sheet}"></a>'
                      for _,r in review.iterrows())
    (OUT/'review_index.html').write_text('<!doctype html><meta charset="utf-8"><title>Net motion v2 review</title>'
        '<h1>前衛モーション v2：目視用30件</h1><p>既存20件＋分類等が変わった最大10件。'
        '水色＝平滑化後、灰＝生の足元。足首欠測時は線を結ばない。</p>'+links,encoding='utf-8')
    return review


def format_percent(n, den):
    return f'{100*n/den:.1f}% ({int(n)}/{int(den)})' if den else 'NA (0/0)'


def paper_tables(summary):
    main = summary[(summary.W_sec==MAIN_W)&(summary.D_m==MAIN_D)&(summary.serves=='include')]
    rows = []
    for method in ['bbox','ankle']:
        g = main[main.coordinate_method==method]
        for metric in PAPER:
            for _, a in g[(g.scope=='all')&(g.metric==metric)].iterrows():
                b = g[(g.scope=='exclude_p8_near')&(g.metric==metric)&(g.category==a.category)].iloc[0]
                rows.append(dict(coordinate_method=method, metric=metric, category=a.category,
                    category_ja=a.category_ja,paper_percent=a.paper_percent,
                    all_n=a.n,all_denominator=a.denominator,all_percent=a.percent,
                    exclude_p8_near_n=b.n,exclude_p8_near_denominator=b.denominator,
                    exclude_p8_near_percent=b.percent))
    return pd.DataFrame(rows)


def figures(paper, main):
    plt.rcParams['font.family'] = 'Meiryo'
    fig, axes = plt.subplots(2,3,figsize=(17,10))
    for ax, metric in zip(axes.flat, PAPER):
        a = paper[(paper.coordinate_method=='bbox')&(paper.metric==metric)]
        b = paper[(paper.coordinate_method=='ankle')&(paper.metric==metric)]
        x = np.arange(len(a))
        ax.bar(x-.27,a.paper_percent,.27,color='#888888',label='論文')
        ax.bar(x,a.all_percent,.27,color='#9b734b',label='v2 箱')
        ax.bar(x+.27,b.all_percent,.27,color='#287fb3',label='v2 足首')
        ax.set_xticks(x,a.category_ja,rotation=35,ha='right')
        ax.set_ylim(0,105)
        ax.set_ylabel('%')
        ax.set_title(f'{metric}：分母 箱{int(a.all_denominator.iloc[0])} / 足首{int(b.all_denominator.iloc[0])}')
        ax.legend(fontsize=8)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0,1,'W=0.4秒、D=0.3m；サーブ含む全件\n\n論文8方向は図5の値を転載：合計101.9%\n正規化していない\n\n終点はネット通過時刻そのものではない\n投影値は実測の移動量ではない\n5点内の打球は独立ではない',va='top')
    fig.tight_layout()
    fig.savefig(OUT/'paper_comparison.png',dpi=140)
    plt.close(fig)
    colors = {'none':'#999999','single':'#287fb3','compound':'#e48e23','unclassifiable':'#d34242'}
    fig,axes = plt.subplots(1,5,figsize=(17,8))
    for ax,(point,g) in zip(axes,main[main.coordinate_method=='ankle'].groupby('point')):
        ax.plot([0,WIDTH,WIDTH,0,0],[0,0,LENGTH,LENGTH,0],color='#333333')
        ax.axhline(NET,color='#444444')
        bounds = [np.array([[0,0],[WIDTH,LENGTH]])]
        for _,r in g.iterrows():
            if r.classification=='unclassifiable':continue
            p = np.array([[r.start_x_m,r.start_y_m],[r.end_x_m,r.end_y_m]])
            bounds.append(p)
            ax.plot(p[:,0],p[:,1],marker='o',ms=2,color=colors[r.classification],alpha=.7,
                    linestyle='--' if r.exclude_p8_near else '-')
            ax.text(*p[0],str(r.shot_index),fontsize=7)
        a = np.vstack(bounds)
        ax.set(xlim=(a[:,0].min()-1.5,a[:,0].max()+1.5),ylim=(a[:,1].min()-2,a[:,1].max()+2),
               xlabel='x (m)',title=f'第{point}点')
        ax.set_aspect('equal')
    axes[0].set_ylabel('y (m)')
    fig.suptitle('v2 足首：始点→終点（線は経路そのものではない） W=0.4秒 D=0.3m\n青=単、橙=複、灰=なし、破線=第8点手前；全経路はtrajectories_main.csv')
    fig.tight_layout()
    fig.savefig(OUT/'court_displacements.png',dpi=140)
    plt.close(fig)


def write_readme(main, summary, paper, scores, detail, checks, reviews, changes):
    table = v1.table
    evaluation = []
    labels = {'motion':'動きの有無','direction3':'3方向（参照が動きあり）',
              'single_compound':'単・複','front_selection':'前衛選択（推定参照を含む）'}
    for metric in labels:
        values = []
        for version in ['v1','v2_bbox','v2_ankle']:
            a = scores[(scores.version==version)&(scores.metric==metric)].iloc[0]
            values.append(format_percent(a.matches,a.denominator))
        evaluation.append([labels[metric],*values])
    texts = ['# 前衛モーション自動類型化 v2',
        '対象：第7〜11点・rally Aを除く86打球（5サーブ）。v1の出力・目視CSV・入力は変更しない。'
        '新規パッケージなし。全生成物はこのv2ディレクトリ内。',
        '## 実行',
        '```powershell\n& .\\study\\.venv\\Scripts\\python.exe -X utf8 -B -u '
        '.\\outputs\\shot_pipeline_20261117\\net_motion\\v2\\classify_net_motion_v2.py\n'
        '& .\\study\\.venv\\Scripts\\python.exe -X utf8 -B -u '
        '.\\outputs\\shot_pipeline_20261117\\net_motion\\v2\\validate_net_motion_v2.py\n```',
        '入力とv1全ファイル（v2を除く）のSHA-256を実行前後で照合する。実行前から存在するv1の'
        'review_claude_v1.csvも対象。再実行はv2生成物のみ更新する。手動レビューは別名保存する。',
        '## 座標と区間の監査',
        '- build_match2_player_ball_review.pyは追跡中に3840×2160座標でprojectし、CSVでは'
        'FrameCoordinates.row(to_canonical=False)で1920×1080へ戻す。従ってCSVのbottom_centerを'
        '2倍してprotocol.jsonのimage_to_court_homography_canonicalに入力し、同次除算する。'
        'build_044_spatiotemporal_pilot.pyのprojectと同じ式。コート座標は正規化x×10.97、y×23.77 m。',
        f'- 全{sum(c["n"] for c in checks):,}行の箱座標でcourt_proxy再現assertを先に実行。'
        f'許容絶対誤差={PROJECTION_ATOL:g}、実際の最大誤差='
        f'{max(c["max_abs_error_normalized"] for c in checks):.3g}（正規化座標）。'
        'projection_validation.csvにpoint/slot別の誤差を保存。これは既存変換の再現検証で、実位置精度の検証ではない。',
        '- 足首は元画像座標。左右ともconf>=0.3なら中点、片側だけならその足首、両側が低い・座標欠損なら欠測。'
        'キャッシュ時間範囲内の未検出・slot欠損を箱で埋めない。キャッシュファイルなし／範囲外だけ箱へfallback。'
        'foot_source_mode、foot_source_counts_json、bbox_fallback_frames、ankle_valid_framesに記録する。'
        '一つの区間[c-6,end]内で足首とfallback箱が混在すると代理点の差で移動が生じるため、'
        'mixed_ankle_bbox_proxyとして判定不能にする。',
        '- 終点=min(次の打球フレーム,c+round(W×fps))。次の打者が前衛でも例外なし。'
        'W=0.3/0.4/0.5秒（主0.4）、fps=60000/1001。roundにより主条件は24フレーム、約0.4004秒。'
        'ネット通過フレームそのものを測定していないため、論文と厳密に同じ区間ではない。',
        '## 前衛の選択',
        '- 両座標条件とも役割判定にはv1の箱コート位置を使い、足首条件との差を足元へ限定する。'
        '非サーブは接触時にネットに近い方、2人の片方でも欠測なら役割不明、距離差<0.5mは曖昧。',
        '- サーブは同じpoint/rallyのlabels次行hitter_slotをレシーバーとして相方を選ぶ。'
        'shot_tableとのラベル一致をassert済み。次slotが空欄・不正なら次接触フレームで'
        'レシーブ側ベースライン（N:y=0、F:y=23.77）に近い方をレシーバーとする。'
        '2人の位置が揃わない・次行がないなら判定不能。距離同率はslot名順。'
        'ボールへの近さは使わない。receiver_slot、receiver_basis、role_selection_basisを保存。',
        '## 分類の定数',
        table(['定数','値・規則'],[
            ['WS / DS','W=0.3/0.4/0.5秒、D=0.2/0.3/0.5m（主W=0.4,D=0.3）'],
            ['ANKLE_CONF','0.3（以上を採用）'],['MEDIAN_FRAMES / MEAN_FRAMES','中心中央値5→中心移動平均3'],
            ['平滑化端点','中央値は3観測以上、移動平均は2観測以上；端は窓を短縮'],
            ['TURN_DEG / MIN_SEG_FRAMES','75°以上／各分割区間の時間差4フレーム以上'],
            ['SEGMENT_DISTANCE_FACTOR','1.0：分割の両区間の変位>=D（主0.3m）'],
            ['PRE_FRAMES / MIN_PRE','接触前6フレームの生座標中央値；4観測以上必要'],
            ['MAX_MISSING / MAX_GAP','区間欠測20%以下、連続欠測2フレーム以下、両端観測必須'],
            ['CROSS_DEG / FEINT_MIN,MAX','8方向の十字軸±10°；誘い候補60〜120°'],
            ['ROLE_MARGIN','非サーブでネット距離差0.5m未満を曖昧として記録'],
            ['SHEET_STRIDE / MAX_EXTRA_REVIEW','6フレーム／変更例を最大10件追加']]),
        '平滑化は[c-6,end]内だけを使い、欠測・追跡segment境界を越えず、欠測時刻を埋めない。'
        '前6フレームの始点はv1同様、生座標中央値。最大変位>=Dを動きありとする。'
        '再帰分割は条件を満たす角度最大の点（同率は早い方）。分割後の各区間の距離・時間もassert。'
        '単の方向は始点→終点、複の評価方向は最終区間。direction3は全体変位、final_direction3は最終区間。'
        '局所座標はN=(Δx,Δy)、F=(-Δx,-Δy)で右・ネット方向。前成分>|右成分|を前、'
        '-前成分>|右成分|を後ろ、他を横とする。8方向、誘い候補、品質除外・集計スコープはv1と同じ。'
        '短い欠測をまたぐ分割ベクトルは観測点同士だけで計算し、経路長は欠測をまたいで加算しない。',
        '## Claude目視20件との一致',
        table(['評価項目','v1','v2 箱','v2 足首'],evaluation),
        '20件はv1の分類から選んだ非無作為標本であり、Claude一人が1回付けた参照である。'
        '改良の設計にも使った標本なので独立テストではなく、一般化性能や人間との一致率ではない。'
        'motion_03/13/20の動きは空欄で、有無は17件が分母。3方向と単・複は参照が動きありの9件が分母。'
        '予測なし・判定不能も分母から除かず不一致。方向はfinal_direction3。'
        '元CSVのauto_classificationではなく実際のv1 motions.csvを使う（motion_18のauto表記がmotionのため）。',
        '前衛選択の正解slotは目視CSVに明示されていない。wrong_playerの2件はv1の選択と反対の相方、'
        '残り18件は「誤選択指摘なし＝v1選択が正しい」と仮定した派生参照。'
        '従って20件の前衛選択一致は厳密な独立正解による精度ではない。'
        'evaluation_details.csvのrole_reference_basisで区別する。'
        '足元・前衛の変更後に参照が適用できるかはsame_person_as_motion_referenceも併せて確認が必要。'
        'wrong_playerの2件を直しても、新しい対象の動きラベルは空欄のままであり自動補完しない。',
        '「完全一致8/20=40%」というv1の総評は残すが、未記入項目を含むため本表では項目別の'
        '評価可能件数を使う。検定は行わない。',
        '### 今回の判定',
        '指定規則の実装は完了したが、足首版の精度改善は確認できない。有無はv1の13/17から12/17、'
        '3方向は6/9から4/9へ低下した。単・複は6/9のまま。前衛の誤選択2件は修正できたが、'
        'その20/20という一致には上記の派生参照の仮定がある。',
        '- false_motionのmotion_02/09/14/17は足首版でも全て動きあり。さらにmotion_01がなし→単へ変わり不一致。\n'
        '- compound_falseのうちmotion_05/15は単へ修正、motion_10は複のまま。'
        'motion_20は1.5秒の例外区間からW以内へ短縮され単となったが、元CSVに単・複の参照ラベルはない。\n'
        '- direction_wrongのmotion_19は横へ修正。motion_07/15は横にならない。'
        '一方、元は正しかったmotion_05/08/10の最終方向が横→前となる。\n'
        '- 足首版のmotion_02/07/08/09/10では、前6フレームを含め採用点は全て両足首中点。'
        'これらの新規・残存する複判定を片足首切替や箱fallbackのせいとは説明できない。'
        '足首推定の揺れ、足の上下動の平面投影、始点定義を再確認する候補であり、原因は未確定。']
    for method in ['bbox','ankle']:
        g = main[main.coordinate_method==method]
        texts += [f'## 論文との主結果：v2 {"箱" if method=="bbox" else "足首"}（サーブ含む）',
            f'判定可能={sum(g.classification!="unclassifiable")}/86、判定不能={sum(g.classification=="unclassifiable")}。'
            f'分類内訳={g.classification.value_counts().to_dict()}。'
            f'箱fallbackを含む区間={int(g.uses_bbox_fallback.sum())}、混在代理点={int(g.mixed_ankle_bbox.sum())}。',
            table(['指標','型','論文 %','今回全件 % (n/分母)','第8点手前除外 % (n/分母)'],[
                [r.metric,r.category_ja,f'{r.paper_percent:.1f}',format_percent(r.all_n,r.all_denominator),
                 format_percent(r.exclude_p8_near_n,r.exclude_p8_near_denominator)]
                for _,r in paper[paper.coordinate_method==method].iterrows()])]
    texts += ['論文の8方向は井箟ほかの図5の値をそのまま転載したもので、合計101.9%となる。'
        '論文側の丸めの不一致と考えられるとの指定に従い、値を正規化・修正していない。'
        'ただし小数第1位への通常の四捨五入だけで1.9ポイントの超過は説明できず、原因は未確定。'
        '原図・原集計の確認が必要である。本実装では論文本体の追加照合は行っていない。',
        '有無の分母＝判定可能、単・複と最終方向＝動きあり、単の方向＝単モーション。'
        'summary.csvにはW×D全9条件、3品質スコープ、サーブ含む／除外を箱・足首別に保存。'
        'paper_comparison.csvはv1と同じ主結果の列構成、論文との差はsummary.csvに保存。',
        '## 再レビュー',
        f'既存20件は同じreview_idと打球IDで保持し、新たな分類・方向・前衛の変更例から'
        f'{len(reviews)-20}件追加（合計{len(reviews)}件）。全変更は{len(changes)}件、classification_changes.csvに保存。'
        '追加順は分類変更→前衛変更→3方向変更→8方向変更、同順位はID順。'
        'review_index.htmlで閲覧。review_template.csvの参照欄は新しいレビュー用の空欄。',
        'シートはv2足首条件の選択slot・区間を使用する。灰線＝生の足首中点／片足首、'
        '水色＝分類に使うコート平滑化位置を同じ変換で画面へ逆投影した経路、'
        '黄点＝現在の生の足元、紫点＝現在の平滑化足元。fallback時は箱となり各コマに出所を表示。'
        '欠測時は線をつながない。背景cropは選手に追従し、各コマのcrop原点を表示する。'
        'review_screen_trajectories.csvに画面座標を記録し、trajectories_main.csvには86件×2条件の'
        '前6フレームを含む生座標・平滑化コート座標を保存する。',
        '## 解釈と残る確認',
        '箱→足首で支持脚・スプリットステップの影響が完全になくなるわけではない。'
        '足首は接地点そのものではなく、空中・隠れ・片側採用・透視投影に誤差が残る。'
        '第8点手前の追跡不良とswappedは元の品質フラグを引き継ぐ。'
        'この5点内の打球は独立ではなく、論文との比率差を戦術・競技特性の証拠とはしない。'
        '論文と比率を近づけることを改良の評価基準にはしない。'
        '次は今回の再レビューと別の未使用区間での複数評定者による確認が必要。',
        '## 主な成果物',
        '- motions.csv：86打球×W3×D3×座標2。役割・欠測・fallback・区間と分類の監査列。\n'
        '- summary.csv / outcome_by_motion.csv：v1と同じ集計（座標条件列を追加）。\n'
        '- evaluation_summary.csv / evaluation_details.csv：3手法×4項目、20件の個別一致。\n'
        '- paper_comparison.csv / paper_comparison.png：主結果と論文の記述比較。\n'
        '- input_manifest.json / projection_validation.csv：入力保持と座標再現の証跡。\n'
        '- validate_net_motion_v2.py / validation_report.json：元入力との出力整合検証。\n'
        '- review_index.html / review_sheets/ / review_template.csv：再レビュー。\n'
        '- .gitignore：実行後に256KiB超の生成物を列挙。']
    (OUT/'README.md').write_text('\n\n'.join(texts)+'\n',encoding='utf-8')


def main():
    OUT.mkdir(exist_ok=True)
    preserved = {p: v1.sha(p) for p in V1.rglob('*') if p.is_file() and not p.is_relative_to(OUT)}
    shots, boxes, protocols, inputs = v1.load()
    ankles, checks, coordinate_inputs = load_coordinates(boxes,protocols)
    inputs += coordinate_inputs + [V1/'review_claude_v1.csv',V1/'motions.csv',V1/'classify_net_motion.py',
        ROOT/'outputs/ball_quality_20261111/build_match2_player_ball_review.py',
        ROOT/'outputs/ball_quality_20261111/build_044_spatiotemporal_pilot.py']
    inputs = sorted(set(inputs))
    hashes = {p:v1.sha(p) for p in inputs}
    print('Projection assertions passed:',sum(c['n'] for c in checks),'rows',flush=True)
    rows, trajectory_rows = [], []
    for (point,rally), group in shots.groupby(['point','rally'],sort=True):
        group = group.sort_values('shot_index').reset_index(drop=True)
        assert np.all(np.diff(group.contact_frame.to_numpy(int))>0)
        for i,shot in group.iterrows():
            nxt = group.iloc[i+1] if i+1<len(group) else None
            base = prepare(shot,nxt,boxes[point],protocols[point])
            for method in ['bbox','ankle']:
                t = event_tracks(point,rally,method,boxes,ankles)
                for w in WS:
                    for d in DS:
                        r, _ = classify_event(base,t,protocols[point],w,d,method)
                        rows.append(r)
                        if w==MAIN_W and d==MAIN_D and r['front_slot']:
                            a = t[r['front_slot']].reindex(np.arange(int(r['contact_frame'])-PRE_FRAMES,r['end_frame']+1))
                            smooth = smooth_positions(a[['x_m','y_m']].to_numpy(float),a.segment.to_numpy())
                            for j,(f,p) in enumerate(a.iterrows()):
                                trajectory_rows.append(dict(id=r['id'],coordinate_method=method,source_frame=int(f),
                                    front_slot=r['front_slot'],foot_source=p.foot_source,
                                    raw_image_x=p.foot_image_x,raw_image_y=p.foot_image_y,
                                    raw_x_m=p.x_m,raw_y_m=p.y_m,smooth_x_m=smooth[j,0],smooth_y_m=smooth[j,1]))
    df = pd.DataFrame(rows)
    assert len(df)==1548 and not df.duplicated(['id','coordinate_method','W_sec','D_m']).any()
    save(df,'motions.csv')
    save(pd.DataFrame(trajectory_rows),'trajectories_main.csv')
    save(pd.DataFrame(checks),'projection_validation.csv')
    main_df = df[(df.W_sec==MAIN_W)&(df.D_m==MAIN_D)].copy()
    old = pd.read_csv(V1/'motions.csv',keep_default_na=False)
    old = old[(old.W_sec==MAIN_W)&(old.D_m==MAIN_D)]
    # Blank v1 numeric fields must stay unknown rather than becoming no-motion.
    old['motion_present'] = pd.to_numeric(old.motion_present,errors='coerce')
    reference = pd.read_csv(V1/'review_claude_v1.csv',keep_default_na=False)
    assert len(reference)==20 and reference.id.is_unique
    detail,scores = evaluate(main_df,old,reference)
    save(detail,'evaluation_details.csv')
    save(scores,'evaluation_summary.csv')
    summary, outcome = [], []
    for method,g in df.groupby('coordinate_method'):
        s = v1.summarize(g)
        s['coordinate_method'] = method
        s['paper_source'] = 'user_supplied_paper_Fig5_for_8dir'
        summary.append(s)
        o = v1.outcomes(g)
        o['coordinate_method'] = method
        outcome.append(o)
    summary = pd.concat(summary,ignore_index=True)
    save(summary,'summary.csv')
    save(pd.concat(outcome,ignore_index=True),'outcome_by_motion.csv')
    paper = paper_tables(summary)
    save(paper,'paper_comparison.csv')
    selected,changes = select_review(main_df,old,reference)
    save(changes,'classification_changes.csv')
    print(scores.to_string(index=False),flush=True)
    print('Rendering review sheets:',len(selected),flush=True)
    reviews = review_sheets(selected,boxes,ankles,protocols)
    figures(paper,main_df)
    write_readme(main_df,summary,paper,scores,detail,checks,reviews,changes)
    assert all(v1.sha(p)==h for p,h in hashes.items()), 'INPUT MUTATED'
    current_preserved = {p for p in V1.rglob('*') if p.is_file() and not p.is_relative_to(OUT)}
    assert current_preserved==set(preserved), 'V1 FILE SET CHANGED'
    assert all(v1.sha(p)==h for p,h in preserved.items()), 'V1 OUTPUT MUTATED'
    manifest = dict(input_sha256={p.relative_to(ROOT).as_posix():h for p,h in hashes.items()},
        v1_preserved_sha256={p.relative_to(V1).as_posix():h for p,h in preserved.items()},
        input_hashes_unchanged=True,v1_files_unchanged=True,projection_assertions_passed=True,
        projection_rows=sum(c['n'] for c in checks),
        projection_max_error=max(c['max_abs_error_normalized'] for c in checks),
        shots=len(shots),motion_rows=len(df),review_sheets=len(reviews),
        parameters=dict(WS=WS,DS=DS,MAIN_W=MAIN_W,MAIN_D=MAIN_D,ANKLE_CONF=ANKLE_CONF,
            MEDIAN_FRAMES=MEDIAN_FRAMES,MEAN_FRAMES=MEAN_FRAMES,TURN_DEG=TURN_DEG,
            MIN_SEG_FRAMES=MIN_SEG_FRAMES,SEGMENT_DISTANCE_FACTOR=SEGMENT_DISTANCE_FACTOR,
            MAX_MISSING=MAX_MISSING,MAX_GAP=MAX_GAP,PROJECTION_ATOL=PROJECTION_ATOL))
    (OUT/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    large = sorted(p for p in OUT.rglob('*') if p.is_file() and p.stat().st_size>256*1024)
    (OUT/'.gitignore').write_text('# Generated files exceeding 256 KiB; refreshed by classify_net_motion_v2.py\n'+
        '\n'.join('/'+p.relative_to(OUT).as_posix() for p in large)+'\n',encoding='utf-8')
    print('Completed; v1 and inputs unchanged. Ignored large files:',len(large),flush=True)
    print(main_df.groupby(['coordinate_method','classification']).size().to_string(),flush=True)
    print(paper.to_string(index=False),flush=True)


if __name__=='__main__':
    main()

