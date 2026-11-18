"""Read-only-input, CPU-only front-player motion classification and review sheets.

Run from any directory with study/.venv/Scripts/python.exe -X utf8 -B.
Definitions, denominators and all operational thresholds are in generated README.md.
"""
from __future__ import annotations

import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
WS = (0.3, 0.4, 0.5)
DS = (0.2, 0.3, 0.5)
MAIN_W, MAIN_D = 0.4, 0.3
WIDTH, LENGTH, NET = 10.97, 23.77, 11.885
SMOOTH = 5
PRE_FRAMES, MIN_PRE = 6, 4
MAX_MISSING, MAX_GAP = 0.20, 2
ROLE_MARGIN = 0.5
TURN_DEG, MIN_SEG_FRAMES = 60.0, 3
CROSS_DEG = 10.0
FEINT_MIN, FEINT_MAX = 60.0, 120.0
SHEET_COUNT, SHEET_STRIDE = 20, 6
SLOTS = {'N': ['near_front', 'near_rear'], 'F': ['far_rear', 'far_front']}
NAMES = dict(zip(['near_front', 'near_rear', 'far_rear', 'far_front'], ['N1','N2','F1','F2']))
DIR3 = ['front', 'lateral', 'back']
DIR8 = ['left_front', 'straight_front', 'right_front', 'left_lateral',
        'right_lateral', 'left_back', 'straight_back', 'right_back']
JA = {'none':'なし','single':'単','compound':'複','unclassifiable':'判定不能',
      'motion':'あり','front':'前','lateral':'横','back':'後ろ',
      'left_front':'左斜め前','straight_front':'まっすぐ前','right_front':'右斜め前',
      'left_lateral':'左横','right_lateral':'右横','left_back':'左斜め後ろ',
      'straight_back':'まっすぐ後ろ','right_back':'右斜め後ろ'}
PAPER = {'presence': {'motion':93.1,'none':6.9},
         'complexity': {'single':72.8,'compound':27.2},
         'single_3dir': dict(zip(DIR3,[60.3,21.1,18.6])),
         'single_8dir': dict(zip(DIR8,[31.6,2.7,27.1,9.4,12.1,6.1,1.8,11.1])),
         'final_3dir': dict(zip(DIR3,[58.9,21.3,19.8]))}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(df, name):
    df.to_csv(OUT / name, index=False, encoding='utf-8-sig', float_format='%.8g')


def truth(v):
    return str(v).lower() == 'true'


def finite_xy(row):
    return row is not None and np.isfinite(row[['x_m','y_m']].to_numpy(dtype=float)).all()


def long_gap(mask):
    best = run = 0
    for v in mask:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def angle(a, b):
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return math.degrees(math.acos(float(np.clip(np.dot(a,b)/den,-1,1)))) if den > 1e-10 else 0.0


def directions(v):
    """v=(player-relative right, netward); theta is clockwise from netward."""
    r, f = map(float, v)
    if math.hypot(r,f) < 1e-9:
        return '', '', float('nan')
    theta = math.degrees(math.atan2(r, f))
    d3 = 'front' if f > abs(r) else ('back' if -f > abs(r) else 'lateral')
    if abs(theta) <= CROSS_DEG:
        d8 = 'straight_front'
    elif abs(abs(theta)-180) <= CROSS_DEG:
        d8 = 'straight_back'
    elif abs(abs(theta)-90) <= CROSS_DEG:
        d8 = 'right_lateral' if r > 0 else 'left_lateral'
    else:
        d8 = ('right_' if r > 0 else 'left_') + ('front' if f > 0 else 'back')
    return d3, d8, theta


def split_motion(p, frames, d):
    """Recursive observed-point split: biggest eligible turn, earliest on ties.

    Both resultant displacements >= D/2 and both durations >=3 frames.
    Returns ordered break indices; never manufactures/interpolates positions.
    """
    def rec(i,j):
        best_k, best_angle = None, TURN_DEG-1e-9
        for k in range(i+1,j):
            a,b = p[k]-p[i], p[j]-p[k]
            if (frames[k]-frames[i] < MIN_SEG_FRAMES or frames[j]-frames[k] < MIN_SEG_FRAMES
                    or np.linalg.norm(a) < d/2 or np.linalg.norm(b) < d/2):
                continue
            deg = angle(a,b)
            if deg > best_angle:
                best_k,best_angle = k,deg
        return [i,j] if best_k is None else rec(i,best_k)[:-1]+rec(best_k,j)
    return rec(0,len(p)-1)


def load():
    inputs = [BASE/'shot_table.csv']
    shots = pd.read_csv(inputs[0], keep_default_na=False)
    shots = shots[shots.rally != 'A'].copy()
    assert len(shots)==86 and set(shots.point)==set(range(7,12))
    tracks, protocols = {},{}
    keys = ['point','rally','shot_index','contact_frame','hitter_side','hitter_slot','is_serve',
            'shot_outcome','confidence','tracking_issue','note']
    for point in range(7,12):
        label = BASE/f'labels/point_{point:02d}.csv'
        folder = BASE/f'tracking/point_{point:02d}'
        inputs += [label, folder/'player_court_proxies.csv', folder/'protocol.json']
        labels = pd.read_csv(label,keep_default_na=False)
        a = labels[labels.rally!='A'][keys].reset_index(drop=True).astype(str)
        b = shots[shots.point==point][keys].reset_index(drop=True).astype(str)
        assert a.equals(b), f'Label/shot table mismatch point {point}'
        protocol = json.loads((folder/'protocol.json').read_text(encoding='utf-8-sig'))
        protocols[point] = protocol
        t = pd.read_csv(folder/'player_court_proxies.csv')
        assert not t.duplicated(['source_frame','player_slot']).any()
        t['x_m'], t['y_m'] = t.court_proxy_x*WIDTH, t.court_proxy_y*LENGTH
        tracks[point] = {slot:group.set_index('source_frame').sort_index()
                        for slot,group in t.groupby('player_slot')}
    return shots,tracks,protocols,inputs


def prepare(shot, nxt, tracks, protocol):
    row = shot.to_dict()
    point, c = int(shot.point), int(shot.contact_frame)
    side = 'F' if shot.hitter_side=='N' else 'N'
    row.update(id=f'p{point:02d}_{shot.rally}_{int(shot.shot_index):02d}',
               front_side=side, front_slot='', front_net_distance_m=np.nan,
               role_distance_gap_m=np.nan, role_ambiguous=False, role_issue='',
               front_positions_trusted=truth(shot.near_positions_trusted if side=='N' else shot.far_positions_trusted),
               exclude_p8_near=(point==8 and side=='N'),
               next_contact_frame=int(nxt.contact_frame) if nxt is not None else np.nan,
               next_hitter_side=nxt.hitter_side if nxt is not None else '',
               next_hitter_slot=nxt.hitter_slot if nxt is not None else '',
               next_tracking_issue=nxt.tracking_issue if nxt is not None else '',
               next_confidence=nxt.confidence if nxt is not None else '',
               same_side_next=(nxt is not None and nxt.hitter_side==shot.hitter_side),
               touch_next=np.nan, touch_reason='', front_won_point=np.nan,
               swapped_current=('swapped' in str(shot.tracking_issue)),
               swapped_next=(nxt is not None and 'swapped' in str(nxt.tracking_issue)))
    row['swapped_either'] = row['swapped_current'] or row['swapped_next']
    pair = shot.near_pair if side=='N' else shot.far_pair
    if pair and shot.point_winner:
        row['front_won_point'] = int(pair==shot.point_winner)
    candidates = []
    for slot in SLOTS[side]:
        t = tracks[slot]
        r = t.loc[c] if c in t.index else None
        if finite_xy(r):
            candidates.append((abs(float(r.y_m)-NET),slot))
    if len(candidates)!=2:
        row['role_issue']='opponent_pair_missing_at_contact'
        row['touch_reason']='front_role_unknown'
        return row
    candidates.sort()
    dist,slot = candidates[0]
    gap = candidates[1][0]-dist
    row.update(front_slot=slot, front_net_distance_m=dist, role_distance_gap_m=gap,
               role_ambiguous=gap<ROLE_MARGIN)
    if nxt is None:
        if shot.shot_outcome and shot.shot_outcome!='continue':
            row.update(touch_next=0,touch_reason='terminal_label_no_next_shot')
        else:
            row['touch_reason']='no_next_label_unresolved'
    elif nxt.hitter_side!=side:
        row.update(touch_next=0,touch_reason='next_hitter_other_side_sequence_flag')
    elif nxt.hitter_slot not in NAMES.values():
        row['touch_reason']='next_hitter_slot_missing'
    else:
        next_slot = next(s for s,n in NAMES.items() if n==nxt.hitter_slot)
        row.update(touch_next=int(next_slot==slot),touch_reason='slot_label_match')
    return row


def classify(base, tracks, protocol, w, d):
    r = dict(base)
    c = int(r['contact_frame'])
    fps = float(Fraction(protocol['source_fps']))
    end = c+int(round(w*fps))
    rule = 'W_cap'
    if np.isfinite(r['next_contact_frame']):
        n = int(r['next_contact_frame'])
        if r['touch_next']==1:
            end,rule = n,'front_touch_override_W'
        elif n <= end:
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
    smooth = np.full_like(raw,np.nan)
    # Do not smooth over a missing frame or identity segment boundary.
    for i in range(len(raw)):
        if not valid[i]:
            continue
        lo,hi = i,i+1
        while lo>max(0,i-SMOOTH//2) and valid[lo-1] and a.segment.iloc[lo-1]==a.segment.iloc[i]:
            lo-=1
        while hi<min(len(raw),i+SMOOTH//2+1) and valid[hi] and a.segment.iloc[hi]==a.segment.iloc[i]:
            hi+=1
        if hi-lo>=3:
            smooth[i]=np.median(raw[lo:hi],axis=0)
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


def scope_groups(df):
    for scope, subset in [('all',df),('exclude_p8_near',df[~df.exclude_p8_near]),
                           ('exclude_p8_near_and_swapped',df[~df.exclude_p8_near & ~df.swapped_either])]:
        yield scope,'include',subset
        yield scope,'exclude',subset[~subset.is_serve.map(truth)]


def summarize(df):
    rows=[]
    for (w,d),grid in df.groupby(['W_sec','D_m'],sort=True):
        for scope,serve,s in scope_groups(grid):
            valid=s[s.classification!='unclassifiable']
            motion=valid[valid.motion_present==1];single=motion[motion.classification=='single']
            groups=[('availability',s,'classification',['unclassifiable']),
                    ('presence',valid,'motion_present',[1,0]),
                    ('complexity',motion,'classification',['single','compound']),
                    ('single_3dir',single,'direction3',DIR3),
                    ('single_8dir',single,'direction8',DIR8),
                    ('final_3dir',motion,'final_direction3',DIR3),
                    ('final_8dir',motion,'final_direction8',DIR8),
                    ('feint',motion,'feint',[True])]
            for metric,data,col,cats in groups:
                for cat in cats:
                    label=cat
                    if metric=='presence':label='motion' if cat else 'none'
                    elif metric=='feint':label='feint'
                    n=int((data[col]==cat).sum());den=len(data)
                    paper=PAPER.get(metric,{}).get(label,np.nan)
                    pct=100*n/den if den else np.nan
                    rows.append(dict(W_sec=w,D_m=d,scope=scope,serves=serve,metric=metric,category=label,
                        category_ja=JA.get(label,label),n=n,denominator=den,percent=pct,paper_percent=paper,
                        difference_pp=pct-paper,total_shots=len(s),classifiable=len(valid),
                        unclassifiable=len(s)-len(valid),paper_source='user_supplied_summary'))
    return pd.DataFrame(rows)


def outcomes(df):
    rows=[]
    for (w,d),grid in df.groupby(['W_sec','D_m'],sort=True):
        for scope,serve,s in scope_groups(grid):
            groups=[('classification',cat,s[s.classification==cat]) for cat in ['none','single','compound','unclassifiable']]
            groups += [('single_3dir',cat,s[(s.classification=='single') & (s.direction3==cat)]) for cat in DIR3]
            groups += [('compound_final_3dir',cat,s[(s.classification=='compound') & (s.final_direction3==cat)]) for cat in DIR3]
            groups += [('feint','feint',s[s.feint])]
            for metric,cat,g in groups:
                touch=g.touch_next.dropna();win=g.front_won_point.dropna()
                rows.append(dict(W_sec=w,D_m=d,scope=scope,serves=serve,metric=metric,category=cat,
                    n=len(g),n_points=g.point.nunique(),touch_n=int(touch.sum()),touch_denominator=len(touch),
                    touch_unknown=len(g)-len(touch),touch_percent=100*touch.mean() if len(touch) else np.nan,
                    won_n=int(win.sum()),win_denominator=len(win),win_percent=100*win.mean() if len(win) else np.nan,
                    swapped_n=int(g.swapped_either.sum()),role_ambiguous_n=int(g.role_ambiguous.sum())))
    return pd.DataFrame(rows)


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str,row))+' |' for row in rows])


def figures(main,summary,geometries):
    plt.rcParams['font.family']='Meiryo'
    colors={'none':'#9b9b9b','single':'#287fb3','compound':'#e48e23','unclassifiable':'#d34242'}
    fig,axes=plt.subplots(1,5,figsize=(17,8))
    for ax,(point,g) in zip(axes,main.groupby('point')):
        ax.plot([0,WIDTH,WIDTH,0,0],[0,0,LENGTH,LENGTH,0],color='#333333',lw=1)
        ax.axhline(NET,color='#444444',lw=2)
        for y in [5.485,LENGTH-5.485]:ax.hlines(y,1.37,WIDTH-1.37,colors='#bbbbbb',lw=.8)
        all_points=[np.array([[0.,0.],[WIDTH,LENGTH]])]
        for _,r in g.iterrows():
            geom=geometries.get(r.id)
            if geom is None:continue
            p=geom['p'];all_points.append(p)
            ax.plot(p[:,0],p[:,1],marker='o',ms=1.5,lw=1,color=colors[r.classification],alpha=.7,
                    linestyle='--' if r.exclude_p8_near else '-')
            ax.text(p[0,0],p[0,1],str(r.shot_index),fontsize=7,clip_on=True)
        bounds=np.vstack(all_points)
        ax.set(xlim=(bounds[:,0].min()-1.5,bounds[:,0].max()+1.5),
               ylim=(bounds[:,1].min()-2,bounds[:,1].max()+2),title=f'第{point}点',xlabel='x (m)')
        ax.set_aspect('equal')
        if point==8:ax.text(0,-1.2,'手前は追跡不良を含む',fontsize=8,color='red')
    axes[0].set_ylabel('y (m); ネット = 11.885')
    fig.suptitle('打球別の前衛軌跡  W=0.4秒・D=0.3m（青=単、橙=複、灰=なし、破線=第8点手前）\n投影値を図外に切り捨てず表示；各パネルの表示範囲は異なる')
    fig.tight_layout();fig.savefig(OUT/'court_trajectories.png',dpi=140);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(17,10))
    for ax,metric in zip(axes.flat,['presence','complexity','single_3dir','single_8dir','final_3dir']):
        s=summary[(summary.W_sec==MAIN_W)&(summary.D_m==MAIN_D)&(summary.serves=='include')&(summary.metric==metric)]
        a=s[s.scope=='all'];b=s[s.scope=='exclude_p8_near']
        x=np.arange(len(a))
        ax.bar(x-.26,a.paper_percent,.26,label='論文（提示値）',color='#777777')
        ax.bar(x,a.percent,.26,label='今回：全件',color='#287fb3')
        ax.bar(x+.26,b.percent,.26,label='今回：第8点手前除外',color='#e48e23')
        ax.set_xticks(x,a.category_ja,rotation=35,ha='right');ax.set_ylim(0,105);ax.set_ylabel('%')
        ax.set_title(metric+f'（今回の全件分母 {int(a.denominator.iloc[0])}）');ax.legend(fontsize=8)
    axes.flat[-1].axis('off')
    axes.flat[-1].text(0,1,'自動分類の暫定値\n\n論文8方向の提示値合計=101.9%\n3方向と8方向は区分の角度が異なる\nネット通過時刻・静止・意図は未検証\n5点内の打球は独立ではない',va='top')
    fig.tight_layout();fig.savefig(OUT/'paper_comparison.png',dpi=140);plt.close(fig)


def select_review(main):
    g=main[main.classification!='unclassifiable'].copy()
    g['review_category']=g.classification
    for k in DIR3:g.loc[(g.classification=='single')&(g.direction3==k),'review_category']='single_'+k
    g.loc[g.feint,'review_category']='feint'
    g['risk']=g.exclude_p8_near.astype(int)*100+g.swapped_either.astype(int)*10+g.role_ambiguous.astype(int)*5+g.track_id_changed.astype(int)
    categories=['none','single_front','single_lateral','single_back','compound','feint']
    pools={cat:list(g[g.review_category==cat].sort_values(['risk','point','shot_index']).index) for cat in categories}
    selected=[]
    while len(selected)<min(SHEET_COUNT,len(g)):
        changed=False
        for cat in categories:
            if pools[cat] and len(selected)<SHEET_COUNT:
                # Alternate points before taking a second case from the same point.
                counts=g.loc[selected].point.value_counts().to_dict() if selected else {}
                idx=min(pools[cat],key=lambda i:(g.loc[i,'risk'],counts.get(g.loc[i,'point'],0),int(g.loc[i,'shot_index'])))
                pools[cat].remove(idx);selected.append(idx);changed=True
        if not changed:break
    return g.loc[selected].copy()


def review_sheets(selected,tracks,protocols):
    folder=OUT/'review_sheets';folder.mkdir(exist_ok=True)
    records=[]
    # Decode once per point in chronological order; retained only needed frames.
    frames_by_point={}
    schedules={}
    for i,(_,r) in enumerate(selected.iterrows(),1):
        frames=list(range(int(r.contact_frame),int(r.end_frame)+1,SHEET_STRIDE))
        if frames[-1]!=int(r.end_frame):frames.append(int(r.end_frame))
        schedules[r.id]=(i,frames)
        frames_by_point.setdefault(int(r.point),set()).update(frames)
    images={}
    for point,wanted in frames_by_point.items():
        cap=cv2.VideoCapture(protocols[point]['source_video'])
        assert cap.isOpened()
        assert abs(cap.get(cv2.CAP_PROP_FPS)-float(Fraction(protocols[point]['source_fps'])))<.01
        lo,hi=min(wanted),max(wanted);cap.set(cv2.CAP_PROP_POS_FRAMES,lo)
        for frame in range(lo,hi+1):
            assert cap.grab(), f'Video decoding failed {point} {frame}'
            if frame in wanted:
                ok,img=cap.retrieve();assert ok
                images[(point,frame)]=img
        cap.release()
    for _,r in selected.iterrows():
        number,frames=schedules[r.id];t=tracks[int(r.point)][r.front_slot]
        c,end=int(r.contact_frame),int(r.end_frame)
        pix=t.reindex(np.arange(c,end+1))[['bottom_center_x','bottom_center_y']]
        canvas=np.full((350,300+400*len(frames),3),245,np.uint8)
        label=[f'{r.id}  {NAMES[r.front_slot]}',f'{r.review_category}',
               f'3dir={r.final_direction3}',f'8dir={r.final_direction8}',
               f'Max={r.max_displacement_m:.2f}m',f'W={r.W_sec}s D={r.D_m}m',
               f'F{c}-{end}',f'p8near={int(r.exclude_p8_near)}',
               f'swapped={int(r.swapped_either)}',f'role_gap={r.role_distance_gap_m:.2f}m',
               'cyan: screen foot path','yellow: current foot']
        for j,line in enumerate(label):cv2.putText(canvas,line,(8,25+25*j),cv2.FONT_HERSHEY_SIMPLEX,.52,(20,20,20),1,cv2.LINE_AA)
        for k,frame in enumerate(frames):
            img=images[(int(r.point),frame)];h,w=img.shape[:2]
            if frame in t.index:
                box=t.loc[frame]
                cx=(box.x1+box.x2)/2;cy=(box.y1+box.y2)/2
                cy=max(cy,box.bottom_center_y-110)
            else:
                # Keep missing tile visible; use contact box only for crop placement.
                box=t.loc[c];cx=(box.x1+box.x2)/2;cy=(box.y1+box.y2)/2
            left=max(0,min(w-400,int(round(cx-200))));top=max(0,min(h-300,int(round(cy-150))))
            tile=img[top:top+300,left:left+400].copy()
            for f in range(c+1,end+1):
                xy0=pix.loc[f-1].to_numpy(float);xy1=pix.loc[f].to_numpy(float)
                if np.isfinite(xy0).all() and np.isfinite(xy1).all():
                    a=tuple(np.rint(xy0-[left,top]).astype(int));b=tuple(np.rint(xy1-[left,top]).astype(int))
                    cv2.line(tile,a,b,(255,220,0),2,cv2.LINE_AA)
            if frame in t.index:
                box=t.loc[frame]
                cv2.rectangle(tile,(int(box.x1)-left,int(box.y1)-top),(int(box.x2)-left,int(box.y2)-top),(0,220,60),1)
                pt=tuple(np.rint([box.bottom_center_x-left,box.bottom_center_y-top]).astype(int))
                cv2.circle(tile,pt,5,(0,255,255),-1)
            else:cv2.putText(tile,'MISSING',(8,28),0,.7,(0,0,255),2)
            x=300+400*k;canvas[40:340,x:x+400]=tile
            cv2.putText(canvas,f'F{frame}  crop({left},{top})',(x+8,25),0,.5,(20,20,20),1,cv2.LINE_AA)
        name=f'motion_{number:02d}.jpg';assert cv2.imwrite(str(folder/name),canvas,[cv2.IMWRITE_JPEG_QUALITY,88])
        records.append(dict(review_id=f'motion_{number:02d}',id=r.id,point=r.point,shot_index=r.shot_index,
            front_slot=r.front_slot,contact_frame=c,end_frame=end,review_category=r.review_category,
            auto_classification=r.classification,auto_direction3=r.final_direction3,auto_direction8=r.final_direction8,
            auto_preliminary_direction8=r.preliminary_direction8,auto_feint=r.feint,
            exclude_p8_near=r.exclude_p8_near,swapped_either=r.swapped_either,
            sheet='review_sheets/'+name,sampled_frames=';'.join(map(str,frames)),
            claude_label='',claude_direction3='',claude_direction8='',claude_feint='',claude_note=''))
    reviews=pd.DataFrame(records);save(reviews,'review_template.csv')
    return reviews


def write_readme(main,summary,review):
    s=summary[(summary.W_sec==MAIN_W)&(summary.D_m==MAIN_D)&(summary.serves=='include')]
    compare=[]
    for metric in PAPER:
        for _,a in s[(s.scope=='all')&(s.metric==metric)].iterrows():
            b=s[(s.scope=='exclude_p8_near')&(s.metric==metric)&(s.category==a.category)].iloc[0]
            compare.append([metric,JA.get(a.category,a.category),f'{a.paper_percent:.1f}',
                f'{a.percent:.1f} ({a.n}/{a.denominator})',f'{b.percent:.1f} ({b.n}/{b.denominator})'])
    sensitivity=[]
    for (w,d,scope),g in summary[(summary.serves=='include')&(summary.scope.isin(['all','exclude_p8_near']))].groupby(['W_sec','D_m','scope']):
        get=lambda m,c:g[(g.metric==m)&(g.category==c)].iloc[0]
        sensitivity.append([w,d,scope,int(g.classifiable.iloc[0]),f"{get('presence','motion').percent:.1f}",
                            f"{get('complexity','compound').percent:.1f}",f"{get('single_3dir','front').percent:.1f}",
                            f"{get('final_3dir','front').percent:.1f}"])
    def p(metric,cat,scope='all',serve='include'):
        v=summary[(summary.W_sec==MAIN_W)&(summary.D_m==MAIN_D)&(summary.scope==scope)&(summary.serves==serve)&(summary.metric==metric)&(summary.category==cat)].iloc[0]
        return f'{v.percent:.1f}% ({v.n}/{v.denominator})'
    outcome_table=[]
    oc=outcomes(main)
    oc=oc[(oc.scope=='all')&(oc.serves=='include')]
    for _,r in oc.iterrows():
        if r.metric not in ['classification','single_3dir','feint']:continue
        touch=f'{r.touch_percent:.1f} ({r.touch_n}/{r.touch_denominator})' if r.touch_denominator else '不明'
        win=f'{r.win_percent:.1f} ({r.won_n}/{r.win_denominator})' if r.win_denominator else '不明'
        outcome_table.append([r.metric,JA.get(r.category,r.category),r.n,r.n_points,touch,r.touch_unknown,win])
    classifiable=main[main.classification!='unclassifiable']
    stationary=int(classifiable.positioning_stationary.sum())
    lines=['# match2 第7〜11点：前衛モーションの自動分類','',
        '## 結果の位置づけ','',
        '本結果は、指定された論文の概念を追跡位置に操作化した**自動候補分類**であり、論文と同じ観察手続による再現や、意図・フェイントの検証ではない。論文原文はこの工程では再確認していない。論文値・定義・書誌情報は依頼文のClaudeによる要約を出典とする。',
        '既存入力は読むだけで、新規成果物はこのフォルダ内に保存した。パッケージ追加・検定・git commitは行っていない。','',
        '## 実行方法','',
        '```powershell',
        '& .\\study\\.venv\\Scripts\\python.exe -X utf8 -B -u .\\outputs\\shot_pipeline_20261117\\net_motion\\classify_net_motion.py',
        '```','',
        'スクリプトの場所から入力を解決する。再実行は本フォルダの生成物のみ更新する。review_template.csvは空欄テンプレートとして再生成するため、レビュー記入版は別名で保持する。入力CSV・protocolのSHA-256を実行前後に照合する。','',
        '## 定義との対応と固定した判定規則','',
        '- 対象は第7〜11点のrally Aを除く86打球（サーブ5打球）。shot_tableの対応する全ラベル列をlabelsと照合した。各point/rally内で次のラベルを決め、点やラリーを越えて次打球を結ばない。',
        f'- 前衛は相手側2人のcontact_frameちょうどの位置のうち、ネットy={NET}mとの距離が小さいslot。2人の片方でも欠測なら役割を判定不能とする。距離差<{ROLE_MARGIN}mをrole_ambiguousとして記録し、同距離はslot名順で決める。slot名から前衛を固定しない。選んだslotを区間中は追跡し、役割の途中再選択はしない。',
        '- 正規化x×10.97、y×23.77でmに換算する。局所座標は（本人の右、ネットへ前）：手前=(Δx,Δy)、奥=(-Δx,-Δy)。従って手前の左=-x、奥の左=+x。',
        f'- ポジショニング始点は打球前6フレーム[c-6,c-1]の**未平滑化の観測位置**の座標別中央値。4フレーム以上必要。pre_max_displacement_mは最初の有効位置からの最大変位、pre_path_mは連続観測間の移動合計。前者<Dならpositioning_stationary=1とする。静止の有無は記録し、主集計の除外条件にはしない。静止未成立の軌跡も含むため論文のポジショニングと同一ではない。',
        '- 軌跡は[c,end]。座標ごとに中心5フレームの中央値を使い、端では短縮し3観測以上必要。平滑化の入力範囲は[c-6,end]で終点以後を使わない。欠測フレームを値で埋めず、欠測や追跡segment境界を越えて平滑化しない。',
        '- 区間の欠測率>20%、連続欠測>2フレーム、始端または終端欠測、始点の有効数不足、追跡segment跨ぎは判定不能。ここで欠測は平滑化後の利用不能を含む。短い欠測が残る場合、分割は観測点同士のベクトルで行うが、欠測時刻の位置を生成しない。移動合計は欠測をまたいで加算しない。track_id変化は記録するが、それだけでは除外しない。',
        '- 終点はmin(次のラベル, c+round(W×fps))。W=0.3/0.4/0.5秒、実fpsはprotocolの60000/1001。**次の打者が当該前衛のslotと分かる場合は、依頼文の接触例外を優先しWを超えても次の打球まで**とする。end_ruleとactual_duration_secで監査可能。slot不明・swappedでは接触と終点の信頼性も落ちる。最後の打球はWで打ち切るためポイント終了後の移動を含みうる。',
        '- モーション有無は始点から観測軌跡への**最大変位**で決める。D=0.3mを主、0.2/0.5mも実行する。最大変位<Dなら「なし」。移動合計・終点変位も保存するが、有無の条件には使わない。',
        '- 単・複は再帰分割。区間の観測候補点kについて、始点→k、k→終点の両変位がD/2以上、両時間が3フレーム以上で、両ベクトルのなす角が60°以上なら、角度最大のk（同率は早い方）で分割し、各区間で繰り返す。1区間=単、2以上=複。仮想始点時刻c-1に前6フレームの中央値を置く。曲線や検出の揺れも複と分類されうる。D以上の往復があるが最終合成ベクトルがゼロで分割も成立しない例は判定不能。',
        '- 3方向：前成分>|右成分|なら前、-前成分>|右成分|なら後ろ、残り（45°境界を含む）は横。依頼文の「>」を採用。8方向：前・後・左右の各軸±10°以内は十字方向、それ以外は象限による斜め。角度は前0°、本人の右が正、±180°が後ろ。',
        '- 単の方向は始点→終点。複の最終方向は最後の区間。preliminary_*は最初の予備区間、preliminary_directions_jsonは全予備区間、segments_jsonは全区間を収録する。direction3/8は全体の始点→終点で、複の決定方向とは別に保持する。',
        '- 誘い型は複の最後から2番目と最後が「左斜め前→右斜め前」または逆で、角度差60〜120°。幾何学的な候補印であり意図を推定した事実ではない。',
        '- 3方向の前は±45°、8方向の斜め前は象限（十字±10°を除く）なので、8方向の前3区分を足しても3方向の「前」と一致しない。論文の分類角度を厳密に再現したものではない。','',
        '## 主結果：W=0.4秒、D=0.3m、サーブあり','',
        table(['指標','型','論文 %','今回全件 % (n/分母)','第8点手前除外 % (n/分母)'],compare),'',
        '有無の分母=判定可能打球、単・複と最終方向の分母=モーションあり、単の方向の分母=単モーション。判定不能を「なし」に含めない。availability行の分母=対象全打球。summary.csvには各分母・対象数・判定不能数・論文との差（百分率ポイント）を保存。','',
        f'全件の判定可能={int((main.classification!="unclassifiable").sum())}/86、判定不能={int((main.classification=="unclassifiable").sum())}。第8点手前の対象={int(main.exclude_p8_near.sum())}打球。主条件の曖昧な前衛選択={int(main.role_ambiguous.sum())}、current/nextのswappedフラグ={int(main.swapped_either.sum())}、同じ側が連続するラベル={int(main.same_side_next.sum())}、接触によるW例外={int((main.end_rule=="front_touch_override_W").sum())}。','',
        f'判定可能例のうち前6フレームの静止目安を満たすもの={stationary}/{len(classifiable)}、満たさないもの={len(classifiable)-stationary}。平滑化後も1フレームで1m超の変化を含むもの={int((classifiable.steps_over_1m>0).sum())}。max_frame_step_mとsteps_over_1mは追跡飛びの確認用で、自動除外には使用していない。','',
        f'サーブ除外時：あり {p("presence","motion",serve="exclude")}、複 {p("complexity","compound",serve="exclude")}、単の前 {p("single_3dir","front",serve="exclude")}、全モーション最終の前 {p("final_3dir","front",serve="exclude")}。全方向のサーブ除外結果もsummary.csvに含む。','',
        '## 感度分析','',
        table(['W秒','D m','対象','判定可能','あり %','複 %','単の前 %','全最終の前 %'],sensitivity),'',
        'summary.csvはW×Dの9条件×3対象（all / exclude_p8_near / exclude_p8_near_and_swapped）×サーブ有無。第8点手前だけでなくcurrent/nextのswappedも除く対象は補足的な品質感度分析で、残った値の正しさを保証しない。','',
        '## 論文との比較・解釈','',
        f'単モーションの前は今回全件 {p("single_3dir","front")}、第8点手前除外 {p("single_3dir","front","exclude_p8_near")}、論文60.3%。全モーション最終方向の前は今回全件 {p("final_3dir","front")}、第8点手前除外 {p("final_3dir","front","exclude_p8_near")}、論文58.9%。「約6割が前」という知見は前後成分を分析すべき根拠にはなるが、今回の一致・不一致を戦術や男女差の証拠とは解釈できない。',
        '提示された論文の単8方向は合計101.9%（前3区分61.4%、横21.5%、後ろ19.0%）。提示された単3方向60.3/21.1/18.6%と一致せず、通常の丸め誤差だけでは説明できない。数値を正規化・修正せず転載した。中間発表等で引用する前に原表の数値・分母を照合する必要がある。','',
        '## 結果との探索的な結び付け','',
        'outcome_by_motion.csvに型ごとの件数、含まれる点数、次打者タッチ率、前衛側の得点率を記録。単の3方向、複の決定3方向、誘い候補も出す。誘い行は複と重なる。タッチは次ラベルの打者slotとの一致であり、映像の接触を独立に検証したものではない。次slot不明は欠測としtouch_unknownに保存する。最後の打球は終了ラベルがあればタッチ0、終了が未解決なら不明。同側連続ラベルはタッチ0とするが、打球の見落とし等の可能性をsame_side_nextに記録する。',
        '得点は前衛側のペア識別とpoint_winnerを照合する。同一点内の全打球に同じ点の結果が結び付くため、**打球は独立ではない**。成功率・因果効果・有効な戦術の証明ではなく、5点のラベルに対する記述であり検定は行わない。','',
        table(['分類軸','型','件数','点数','次打者タッチ % (n/既知)','タッチ不明','前衛側得点 % (n/分母)'],outcome_table),'',
        '## 目視確認用シート','',
        f'{len(review)}件を主条件の判定可能例から選択。型を巡回して、既知の追跡不良・swapped・役割曖昧を避け、点番号も分散させた便宜抽出（無作為抽出ではない）。型別枚数：'+', '.join(f'{k}={v}' for k,v in review.review_category.value_counts().items())+'。存在しない型は生成しない。',
        '400×300pxの追跡箱周辺を打球から終点まで6フレームおきに横並びし、終点が6の倍数でなければ最後に追加。左端に分類、方向、品質フラグを記載。水色の線は**元画像の箱下端中央**を画面座標のまま結んだ全区間の軌跡、黄色の点はそのコマの足元。各コマのcrop原点を引くだけで、コート投影から逆算しない。欠測は線を結ばない。切抜きの中心は箱で、足元が収まるよう必要時のみ下へずらす。',
        'review_template.csvのclaude_label等は空欄。Claudeによる正誤判定・一致率の算出は未実施であり、20件のシート作成は精度検証の完了を意味しない。','',
        '## 限界','',
        '- 論文は女子の室内4試合（1980年代）を手で書き写したもの。こちらは男子トップの1試合5点を自動で追跡したもの。標本・時代・観察法が異なる。',
        '- 足元は検出の箱の下端で、静止・移動の判定は位置の揺れの影響を受ける。固定ホモグラフィの投影値は位置の真値ではなく、遠近・カメラ変化・追跡誤同定にも影響される。',
        '- ネットを越える時刻は分からず、Wで代えている。接触例外以外では通過より前後にずれ、相手の打球に対する意図的反応以外の移動も入りうる。',
        '- 打球のフレームはClaudeが付けたラベルで、6〜12フレームずれることがある。この解析はWとDの感度だけで、接触フレームを±6/12ずらした頑健性は検証していない。',
        '- 第8点は手前の1人（ピンク）が追跡されず、N1の箱が線審に付く。near_positions_trusted=False。その側の前衛選択・変位・タッチ率を信頼できないため、除外値を併記した。swappedは現在と次のラベル両方を記録し、ID変化も別に記録する。ラベルにフラグがない区間の同定精度は未検証。',
        '- 論文の一時静止・意図的移動・予備動作と決定動作を、この短い軌跡だけで直接観測したわけではない。打球以前の予備動作を取りこぼす可能性もある。D・60°・±10°は本解析の操作的しきい値で、論文の確定した判定基準ではない。','',
        '## ファイル','',
        '- classify_net_motion.py：全生成工程。motions.csv：86打球×9条件=774行。元ラベル、前衛選択、品質、始終点、変位、方向、全分割を保持。',
        '- summary.csv：論文との比較と全感度条件。outcome_by_motion.csv：型別の探索的な結果。',
        '- court_trajectories.png：点別コート軌跡。paper_comparison.png：主条件と除外値の比較。',
        '- review_sheets/motion_XX.jpg・review_template.csv：目視確認用。input_manifest.json：読取入力ハッシュと実行確認。',
        '- .gitignore：このフォルダで256KiB（262144 bytes）を超える生成物を列挙して除外。既存の.gitignoreは変更しない。','']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')


def ignore_large():
    files=sorted(p for p in OUT.rglob('*') if p.is_file() and p.stat().st_size>256*1024)
    (OUT/'.gitignore').write_text('# Generated artifacts exceeding 256 KiB. Updated by classify_net_motion.py\n'+
        '\n'.join('/'+p.relative_to(OUT).as_posix() for p in files)+'\n',encoding='utf-8')


def main():
    shots,tracks,protocols,inputs=load()
    hashes={str(p.relative_to(BASE)):sha(p) for p in inputs}
    rows=[];geometries={}
    for _,group in shots.groupby(['point','rally'],sort=True):
        group=group.sort_values('shot_index').reset_index(drop=True)
        assert np.all(np.diff(group.contact_frame.to_numpy(int))>0)
        for i,shot in group.iterrows():
            nxt=group.iloc[i+1] if i+1<len(group) else None
            point=int(shot.point)
            base=prepare(shot,nxt,tracks[point],protocols[point])
            for w in WS:
                for d in DS:
                    r,geom=classify(base,tracks[point],protocols[point],w,d)
                    rows.append(r)
                    if w==MAIN_W and d==MAIN_D:geometries[r['id']]=geom
    df=pd.DataFrame(rows);assert len(df)==774
    assert not df.duplicated(['id','W_sec','D_m']).any()
    save(df,'motions.csv')
    summary=summarize(df);save(summary,'summary.csv');save(outcomes(df),'outcome_by_motion.csv')
    main_df=df[(df.W_sec==MAIN_W)&(df.D_m==MAIN_D)].copy()
    figures(main_df,summary,geometries)
    review=review_sheets(select_review(main_df),tracks,protocols)
    write_readme(main_df,summary,review)
    assert all(sha(BASE/path)==h for path,h in hashes.items()),'INPUT MUTATED'
    manifest=dict(input_sha256=hashes,input_hashes_unchanged=True,shots=len(shots),motion_rows=len(df),
                  review_sheets=len(review),parameters=dict(WS=WS,DS=DS,MAX_MISSING=MAX_MISSING,
                  MAX_GAP=MAX_GAP,TURN_DEG=TURN_DEG,CROSS_DEG=CROSS_DEG,MIN_SEG_FRAMES=MIN_SEG_FRAMES))
    (OUT/'input_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    ignore_large()
    print(main_df.classification.value_counts().to_string())
    print('review_sheets:',len(review),'input_hashes_unchanged: True')
    print(summary[(summary.W_sec==MAIN_W)&(summary.D_m==MAIN_D)&(summary.scope=='all')&
                  (summary.serves=='include')&summary.metric.isin(PAPER)][
                      ['metric','category','n','denominator','percent','paper_percent']].to_string(index=False))


if __name__=='__main__':
    main()
