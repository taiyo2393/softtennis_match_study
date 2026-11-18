"""Independent saved-output audit for review2, including full MP4 decoding."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

import build_court_view as b
import review2 as r2
import review2c as c
from speed_reset import STATUS as SPEED_RESET


def audit_rule_boundaries():
    """Counterexamples for rules not necessarily exercised by the 32 real shots."""
    def decide(boxes,ball,wrist=''):
        hit=dict(point=1,frame=100,side='N',hitter_slot=wrist,reason='test')
        tracks=pd.DataFrame(boxes,columns=['source_frame','player_slot','x1','y1','x2','y2'])
        candidates=pd.DataFrame(ball,columns=['point','source_frame','candidate_present','x_px','y_px','confidence'])
        return b.abc.combine_ball([hit],{1:tracks},candidates)[0]
    first=(100,'near_front',0,0,100,100)
    second=(100,'near_rear',110,0,210,100)
    inside=[(1,100,1,50,50,.8)]
    single=decide([first],inside)
    assert (single['hitter'],single['status'])==('N1','auto_ball_single_visible')
    # Expanded boxes both cover x105: proximity alone must not vote for either.
    both=[(1,100,1,105,50,.8)]
    fallback=decide([first,second],both,'near_front')
    assert fallback['status']=='auto_wrist' and fallback['hitter']=='N1'
    assert decide([first,second],both)['status']=='undecided'
    assert decide([first,second],inside,'near_rear')['status']=='undecided','decisive conflict must remain unresolved'
    context=decide([first],[])
    assert (context['hitter'],context['status'])==('N2','auto_context_offscreen')
    # Missing exactly one frame of the absence window invalidates offscreen.
    invalid=decide([first,(92,'near_rear',110,0,210,100)],[])
    assert invalid['status']=='undecided'
    for duration,expected in [(41,True),(42,True),(43,False)]:
        raw=[dict(source_frame=f,rally_no=1,y1=400.,y2=490. if 20<=f<20+duration else 500.,lower_occluded=False)
             for f in range(20+duration+10)]
        overrides,events=r2.find_lifts(raw)
        assert bool(events)==expected,(duration,events)
        if expected:assert events[0]['duration_source_frames']==duration
    # Inclusive frame window, minimum count, rally separation, one-sided clamp,
    # height outlier exclusion, and inclusion of occluded observations.
    raw=[dict(source_frame=f,rally_no=1,x1=900.,x2=950.,y1=460.,y2=y,lower_occluded=occ)
         for f,y,occ in [(0,520.,False),(15,535.,True),(16,540.,False),(30,530.,False)]]
    raw += [dict(source_frame=17,rally_no=2,x1=900.,x2=950.,y1=460.,y2=550.,lower_occluded=False)]
    one=r2.env.envelope_rows(raw);other=r2.env.envelope_rows(raw[-1:])
    assert one[0]['envelope_y2_px']==520. and one[0]['envelope_window_observations']==2
    assert np.isclose(one[15]['envelope_y2_px'],536.25)
    assert one[16]['envelope_y2_px']==540. and other[17]['envelope_y2_px']==550.
    tall=dict(source_frame=14,rally_no=1,x1=900.,x2=950.,y1=400.,y2=600.,lower_occluded=True)
    excluded=r2.env.envelope_rows(raw[:-1]+[tall])
    assert np.isclose(excluded[15]['envelope_y2_px'],one[15]['envelope_y2_px'])
    assert excluded[14]['envelope_y2_px']==600.
    # Neither a source-frame count nor an absent sample should move the boundary.
    for side in ('near_front','far_front'):
        raw=[dict(source_frame=f,rally_no=1,player_slot=side,point=1,track_id=1,
                  x1=900.,x2=950.,y1=490.,y2=550.,lower_occluded=False,
                  court_proxy_x=0. if f==0 else 1.,court_proxy_y=.5)
             for f in range(35) if f!=5]
        if side.startswith('near_'):
            rows,_=b.clean_player(side,raw,range(35),far_revision=False,speed_reset=True)
        else:
            rows,_=r2.far.clean_far(side,raw,range(35),x_reference={f:0. if f==0 else 1. for f in range(35)},speed_reset=True)
        at={q['source_frame']:q for q in rows}
        assert at[31]['filter_status']=='rejected_speed'
        assert at[32]['filter_status']==SPEED_RESET and at[32]['speed_reset_frame']==32
        assert at[32]['speed_reject_start_frame']==1 and at[33]['filter_status']!='rejected_speed'
    def link_case(gap,dist):
        raw=pd.DataFrame([dict(source_frame=f,point=1,track_id=tid,player_slot='near_front',
                              court_proxy_x=x,court_proxy_y=.5,head_court_x=x,head_court_y=.5,
                              lower_occluded=False,filter_status='observed')
                          for f,tid,x in [(0,1,0.),(gap,2,dist/10.97)]])
        def passthrough(slot,rows,frames):return rows,{}
        _,mapping,_=b.abc.relink(raw,[dict(start_frame=0,end_frame=100,rally_no=1)],passthrough,None,None,vacant_reentry=True)
        return mapping.iloc[-1]
    assert link_case(2,20).identity_status=='unlinked_missing'
    assert link_case(59,20).identity_status=='unlinked_missing'
    assert link_case(60,20).identity_status=='reentry_vacant_slot'
    assert link_case(2,.1).identity_status=='linked'
    from speed_reset import SpeedReset
    gate=SpeedReset()
    assert gate.update(1,'rejected_speed',1) is None
    assert gate.update(31,'rejected_speed',1) is None
    assert gate.update(32,'rejected_speed',1)==1
    gate.update(33,'rejected_speed',1);gate.update(40,'rejected_box_height',1)
    assert gate.update(64,'rejected_speed',1) is None
    assert gate.update(100,'rejected_speed',2) is None
    return dict(synthetic_ball_and_context_cases=6,lift_duration_boundary_cases=3,envelope_boundary_cases=6,
                speed_reset_side_cases=2,speed_streak_boundary_cases=4,relink_boundary_cases=4)


def audit_envelope(p,marked,positions,segments):
    """Recalculate quantiles, geometry, accepted anchors and speed from raw boxes."""
    project,H=b.abc.geometry(b.load_module)
    old=pd.read_csv(r2.env.BACKUP/f'cleaned_player_positions_p{p:02d}.csv')
    previous=old.set_index(['source_frame','player_slot']);now=positions.set_index(['source_frame','player_slot'])
    valid=previous.court_x.notna()&now.court_x.notna()&previous.index.get_level_values(1).str.startswith('far_')
    assert np.allclose(previous.loc[valid,'court_x'],now.loc[valid,'court_x'],rtol=0,atol=1e-12)
    checked=0
    for (slot,rally),raw in marked[marked.player_slot.str.startswith('far_')].groupby(['player_slot','rally_no']):
        g=positions[positions.player_slot.eq(slot)&positions.rally_no.eq(rally)].set_index('source_frame')
        prior=old[old.player_slot.eq(slot)].set_index('source_frame')
        # Relink can change raw depth anchors and therefore its pre-envelope x
        # reference. Displayed existing x is separately frozen to review2 above.
        baseline,_=r2.far.clean_far(slot,raw.to_dict('records'),g.index)
        base=pd.DataFrame(baseline).set_index('source_frame')
        bx=pd.to_numeric(base.effective_court_x,errors='coerce')
        comparable=g.effective_court_x.notna()&bx.notna()
        assert np.allclose(g.loc[comparable,'effective_court_x'],bx.loc[comparable],rtol=0,atol=1e-12)
        anchors=g[g.filter_status.isin(['observed',r2.far.ENVELOPE,r2.SERVE]) | (g.filter_status.eq(SPEED_RESET)&g.lower_occluded.eq(False))]
        accepted=[];reject_start=None
        for row in raw.sort_values('source_frame').itertuples():
            f=row.source_frame;q=g.loc[f];height_ok=row.y2-row.y1<=1.35*.73*(row.y2-415)
            eligible=raw[(raw.y2-raw.y1<=1.35*.73*(raw.y2-415))&raw.source_frame.between(f-15,f+15)]
            expected=max(row.y2,float(np.percentile(eligible.y2,75))) if height_ok and len(eligible)>=3 else row.y2
            assert np.isclose(q.envelope_y2_px,expected,rtol=0,atol=1e-10)
            assert q.envelope_window_observations==len(eligible) and bool(q.envelope_height_eligible)==height_ok
            shift=(project(H,(row.x1+row.x2)/2,row.y2)[1]-project(H,(row.x1+row.x2)/2,expected)[1])*23.77
            assert np.isclose(q.envelope_shift_m,shift,rtol=0,atol=1e-10) and shift>=-1e-10
            override=q.filter_status in (r2.SERVE,r2.LIFT)
            depth=expected
            if row.lower_occluded and not override:
                left=anchors[anchors.index<f];right=anchors[anchors.index>f]
                a=left.iloc[-1] if len(left) else None;z=right.iloc[0] if len(right) else None
                if a is not None and z is not None:
                    t=(f-a.name)/(z.name-a.name);depth=(1-t)*a.depth_y_px+t*z.depth_y_px
                elif a is not None or z is not None:depth=(a if a is not None else z).depth_y_px
                else:depth=None
                assert (pd.isna(q.depth_anchor_before_frame) if a is None else q.depth_anchor_before_frame==a.name)
                assert (pd.isna(q.depth_anchor_after_frame) if z is None else q.depth_anchor_after_frame==z.name)
            if override:depth=q.depth_y_px
            if depth is None:
                assert q.filter_status=='missing';continue
            assert np.isclose(q.depth_y_px,depth)
            court_y=project(H,(row.x1+row.x2)/2,depth)[1]
            assert np.isclose(q.effective_court_y,court_y)
            speed=bool(accepted) and b.distance((q.effective_court_x,court_y),accepted[-1][1])>8*(f-accepted[-1][0])/b.FPS+1
            status=(q.filter_status if override else 'rejected_box_height' if not row.lower_occluded and not height_ok else
                    'rejected_speed' if speed else r2.far.DEPTH if row.lower_occluded else r2.far.ENVELOPE if shift>=.5 else 'observed')
            reset=False
            if status=='rejected_speed':
                if reject_start is None:reject_start=f
                reset=f-reject_start>30
                if reset:
                    assert q.speed_reset_frame==f and q.speed_reject_start_frame==reject_start
                    status=SPEED_RESET;reject_start=None
            else:reject_start=None
            assert pd.notna(q.speed_reset_frame)==reset
            assert q.filter_status==status,(p,slot,f,q.filter_status,status)
            direct=status in ('observed',r2.far.ENVELOPE) or (status==SPEED_RESET and not row.lower_occluded)
            assert bool(q.envelope_applied)==(direct and shift>1e-9)
            if status in r2.far.ACCEPTED:accepted.append((f,(q.effective_court_x,court_y)))
            checked+=1
    stats,intervals=r2.env.summarize(p,positions.copy(),segments)
    if p==7:
        g=positions[positions.player_slot.eq('far_rear')].set_index('source_frame')
        for f in (22920,22930,22950,22960,22970):
            assert g.loc[f,'filter_status'] in ('observed',r2.far.ENVELOPE),('short-box speed regression',f)
        assert g.loc[22910,'filter_status']==r2.far.ENVELOPE and g.loc[22910,'envelope_shift_m']>6.
    summary=json.loads((b.OUT/'cleaning_summary.json').read_text(encoding='utf-8'))
    assert stats==[e for e in summary['review2c_envelope_stats'] if e['point']==p]
    # Floats undergo CSV round trips; compare interval keys separately.
    keys=['player_slot','rally_no','start_frame','end_frame','duration_source_frames','midpoint_frame']
    assert [[e[k] for k in keys] for e in intervals]==[[e[k] for k in keys] for e in summary['review2c_envelope_intervals'] if e['point']==p]
    return dict(raw_envelope_and_speed_rows=checked,existing_far_x_rows_unchanged=int(valid.sum()))


def audit_near_and_reentry(p,fixed,mapping,positions,s):
    """Near reconstruction plus independent source-time/distance relink checks."""
    raw_index=positions.set_index(['source_frame','player_slot'])
    checked=0
    for slot in b.SLOTS[:2]:
        raw=fixed[fixed.player_slot.eq(slot)].to_dict('records')
        output,_=b.clean_player(slot,raw,positions.source_frame.unique(),far_revision=False,speed_reset=True)
        expected=pd.DataFrame(output).set_index('source_frame')
        actual=raw_index.xs(slot,level='player_slot')
        obs=[int(r['source_frame']) for r in raw]
        for col in ('status','filter_status'):
            assert expected.loc[obs,col].tolist()==actual.loc[obs,col].tolist(),(p,slot,col)
        for col in ('court_x','court_y','effective_court_x','effective_court_y','speed_reset_frame'):
            assert np.allclose(pd.to_numeric(expected.loc[obs,col],errors='coerce'),actual.loc[obs,col],equal_nan=True),(p,slot,col)
        # Audit speed and reset decisions without calling the streak helper.
        for rally,group in actual.loc[obs].groupby('rally_no'):
            accepted=None;start=None
            for f,q in group.sort_index().iterrows():
                eligible=q.filter_status in b.OBS or q.filter_status=='rejected_speed'
                if not eligible:start=None;continue
                bad=accepted is not None and b.distance((q.effective_court_x,q.effective_court_y),accepted[1])>8*(f-accepted[0])/b.FPS+1
                if bad:
                    if start is None:start=f
                    reset=f-start>30
                    assert (q.filter_status==SPEED_RESET)==reset,(p,slot,f,'speed-reset boundary')
                    if reset:
                        assert q.speed_reset_frame==f and q.speed_reject_start_frame==start
                        start=None
                    else:assert q.filter_status=='rejected_speed',(p,slot,f,'speed gate')
                else:start=None
                if q.filter_status in b.OBS:accepted=(f,(q.effective_court_x,q.effective_court_y))
        checked+=len(obs)
    linked=0;vacant=0
    fixed_index=fixed.set_index(['source_frame','track_id'])
    for rally,group in mapping.groupby('rally_no'):
        last={};known={}
        for row in group.sort_values('source_frame',kind='stable').itertuples():
            if pd.isna(row.player_slot) or not row.player_slot:continue
            q=fixed_index.loc[row.source_frame,row.track_id]
            assert q.player_slot==row.player_slot
            if row.track_id not in known and row.player_slot in last:
                prev=last[row.player_slot];dt=(row.source_frame-prev.name[0])/b.FPS
                if dt>1:
                    assert row.identity_status=='reentry_vacant_slot';vacant+=1
                else:
                    assert row.identity_status=='linked'
                    dist=np.hypot((q.link_x-prev.link_x)*10.97,(q.link_y-prev.link_y)*23.77)
                    assert 0<dt<=1 and dist<=8*dt+1.5
                    linked+=1
            else:assert row.identity_status!='reentry_vacant_slot'
            known[row.track_id]=row.player_slot;last[row.player_slot]=q
    recomputed=r2.reentry_intervals(p,fixed,mapping)
    assert recomputed==[e for e in s['review2_reentries'] if e['point']==p]
    return dict(near_raw_rows_reconstructed=checked,short_gap_links_verified=linked,long_gap_reentries_verified=vacant)


def audit_ball_and_context(h,fixed,balls):
    """Re-measure evidence from untouched candidate coordinates and fixed boxes."""
    slots=[h.player_1,h.player_2];boxes=fixed.set_index(['source_frame','player_slot'])
    evidence=json.loads(h.ball_evidence_json);mins=[[],[]]
    source=balls[(balls.point==h.point)&balls.source_frame.between(h.frame-3,h.frame+3)&balls.candidate_present.eq(1)]
    assert len(source)==len(evidence)==h.ball_candidate_frames
    for saved,ball in zip(evidence,source.itertuples()):
        assert saved['frame']==ball.source_frame
        assert np.allclose([saved['x_px'],saved['y_px'],saved['confidence']],[ball.x_px,ball.y_px,ball.confidence])
        for i,slot in enumerate(slots):
            key=(ball.source_frame,slot)
            if key not in boxes.index:
                assert saved['distances_px'][i] is None;continue
            box=boxes.loc[key];w=box.x2-box.x1;ht=box.y2-box.y1
            dx=max(0.,box.x1-.3*w-ball.x_px,ball.x_px-box.x2-.3*w)
            dy=max(0.,box.y1-.3*ht-ball.y_px,ball.y_px-box.y2)
            d=np.hypot(dx,dy)
            assert np.isclose(saved['distances_px'][i],d)
            assert np.isclose(saved['box_widths_px'][i],w)
            mins[i].append((float(d),int(ball.source_frame),float(w)))
    minima=[min(v) if v else None for v in mins]
    for i in range(2):
        saved=getattr(h,f'ball_player_{i+1}_distance_px')
        assert (minima[i] is None and pd.isna(saved)) or (minima[i] is not None and np.isclose(saved,minima[i][0]))
    ball='';single=False
    both_close=all(v is not None and v[0]<.25*v[2] for v in minima)
    visible=[slot for slot in slots if (h.frame,slot) in boxes.index]
    if not both_close:
        if len(visible)==1 and minima[slots.index(visible[0])] is not None and minima[slots.index(visible[0])][0]==0:
            ball=visible[0];single=True
        elif all(v is not None for v in minima):
            lo=int(minima[1][0]<minima[0][0]);hi=1-lo
            d0,d1=minima[lo][0],minima[hi][0]
            if d0<d1 and ((d0==0 and d1>=minima[hi][2]) or (d1>0 and d0/d1<.5)):ball=slots[lo]
    assert (h.ball_hitter_slot if pd.notna(h.ball_hitter_slot) else '')==ball
    wrist='';speeds=[h.player_1_max,h.player_2_max]
    if np.isfinite(speeds).all() and max(speeds)>min(speeds) and max(speeds)>=1.3*min(speeds):wrist=slots[int(speeds[1]>speeds[0])]
    assert (h.wrist_hitter_slot if pd.notna(h.wrist_hitter_slot) else '')==wrist
    off=''
    context=balls[(balls.point==h.point)&balls.source_frame.between(h.frame-10,h.frame+5)&balls.candidate_present.eq(1)]
    for i,slot in enumerate(slots):
        other=slots[1-i]
        absent=not any((n,slot) in boxes.index for n in range(h.frame-8,h.frame+5))
        other_visible=any((n,other) in boxes.index for n in range(h.frame-8,h.frame+5))
        nearby=False
        for q in context.itertuples():
            if (q.source_frame,other) not in boxes.index:continue
            z=boxes.loc[q.source_frame,other]
            d=np.hypot(max(z.x1-q.x_px,0,q.x_px-z.x2),max(z.y1-q.y_px,0,q.y_px-z.y2))
            nearby |= d<=1.5*(z.x2-z.x1)
        if absent and other_visible and not nearby:off=slot
    winner='';status='undecided'
    if off:winner,status=off,'auto_context_offscreen'
    elif single:winner,status=ball,'auto_ball_single_visible'
    elif ball and wrist and ball==wrist:winner,status=ball,'auto_ball_wrist'
    elif ball and not wrist:winner,status=ball,'auto_ball'
    elif not ball and wrist:winner,status=wrist,'auto_wrist'
    assert h.status==status and (h.hitter_slot if pd.notna(h.hitter_slot) else '')==winner,(h.point,h.frame)


def main():
    out=b.OUT;s=json.loads((out/'cleaning_summary.json').read_text(encoding='utf-8'))
    boundaries=audit_rule_boundaries()
    before=json.loads((out/'review2_before.json').read_text(encoding='utf-8'))
    envelope_before=json.loads((out/'review2b_before.json').read_text(encoding='utf-8'))
    for name,digest in envelope_before['sha256'].items():assert b.sha(r2.env.BACKUP/name)==digest,name
    current_before=json.loads((out/'review2c_before.json').read_text(encoding='utf-8'))
    for name,digest in current_before['sha256'].items():assert b.sha(c.BACKUP/name)==digest,name
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=b.ROOT,text=True).strip()==current_before['git_head']
    for name in ('human_confirmed_hitters_p07_11.csv',):
        assert b.sha(out/name)==envelope_before['sha256'][name],('unexpected non-depth change',name)
    registered=json.loads(json.dumps(s));c.finalize(registered,write=False)
    assert registered['review2_depth_counts']==s['review2_depth_counts']
    assert registered['review2c_speed_resets']==s['review2c_speed_resets']
    assert registered['review2c_p07_F1_above_intervals']==s['review2c_p07_F1_above_intervals']
    for name,digest in before['input_sha256'].items():assert b.sha(Path(name))==digest,f'protected input: {name}'
    for name,digest in before['baseline_sha256'].items():assert b.sha(out/'review2_before'/name)==digest,f'baseline: {name}'
    assert b.sha(out/'hitter_auto_before_review2.csv')==before['baseline_sha256']['hitter_auto.csv']
    hits=pd.read_csv(out/'hitter_auto.csv');truth=pd.read_csv(out/'human_confirmed_hitters_p07_11.csv').fillna('')
    assert hits.hitter.tolist()==pd.read_csv(c.BACKUP/'hitter_auto.csv').hitter.tolist()
    assert hits.hitter.eq(truth.confirmed_hitter).sum()==32
    assert len(hits)==len(truth)==32 and hits.rule_version.eq(r2.VERSION).all()
    assert list(zip(hits.point,hits.frame))==list(zip(truth.point,truth.frame))
    assert truth.groupby('point',sort=False).shot_no.apply(list).to_dict()=={7:list(range(1,9)),8:list(range(1,9)),9:list(range(1,9)),10:list(range(1,5)),11:list(range(1,5))}
    old=pd.read_csv(out/'hitter_auto_before_review2.csv')
    assert old.hitter.tolist()==truth.auto_hitter.tolist() and old.status.tolist()==truth.auto_status.tolist()
    assert truth.confirm_source.value_counts().to_dict()=={'auto_accepted_by_user':25,'user_answered':6,'user_answer_reinterpreted':1}
    ambiguous=truth[(truth.point==7)&truth.shot_no.eq(6)].iloc[0]
    assert ambiguous.user_answer=='N1' and ambiguous.confirmed_hitter=='N2' and '要再確認' in ambiguous.note
    unamb=truth.confirm_source.ne('user_answer_reinterpreted')
    agreement=dict(before=int(old.hitter.eq(truth.confirmed_hitter).sum()),after=int(hits.hitter.eq(truth.confirmed_hitter).sum()),
                   undecided_before=int(old.status.eq('undecided').sum()),undecided_after=int(hits.status.eq('undecided').sum()),
                   excluding_reinterpreted=dict(total=int(unamb.sum()),before=int((old.hitter.eq(truth.confirmed_hitter)&unamb).sum()),after=int((hits.hitter.eq(truth.confirmed_hitter)&unamb).sum())))
    balls=pd.read_csv(out/'ball_candidates.csv');wrist=pd.read_csv(out/'wrist_identity.csv')
    official=json.loads((b.PIPE/'tracking_v2/verification.json').read_text(encoding='utf-8'))['excluded_candidates']
    project,H=b.abc.geometry(b.load_module);audits={}
    for p in range(7,12):
        fixed=pd.read_csv(out/f'identity_positions_p{p:02d}.csv');mapping=pd.read_csv(out/f'identity_map_p{p:02d}.csv')
        raw=pd.read_csv(b.PIPE/f'tracking_v2/point_{p:02d}/player_court_proxies.csv')
        positions=pd.read_csv(out/f'cleaned_player_positions_p{p:02d}.csv')
        assert not fixed.duplicated(['source_frame','player_slot']).any()
        assert not positions.duplicated(['source_frame','player_slot']).any()
        raw_boxes=raw.set_index(['source_frame','track_id'])
        for q in fixed.itertuples():
            z=raw_boxes.loc[q.source_frame,q.track_id]
            assert np.allclose([q.x1,q.y1,q.x2,q.y2],[z.x1,z.y1,z.x2,z.y2]),'invented box'
        for (_,tid),g in fixed.groupby(['rally_no','track_id']):assert g.player_slot.nunique()==1
        for row in fixed.itertuples():
            for e in official:
                if e['point']!=p or e['track_id']!=row.track_id:continue
                assert not e.get('full_track_excluded',False)
                assert not any(a<=row.source_frame<=z for a,z in e.get('exclusion_ranges',[]))
        for h in hits[hits.point.eq(p)].itertuples():
            audit_ball_and_context(h,fixed,balls)
            for i in (1,2):
                values=wrist.loc[wrist.point.eq(p)&wrist.player_slot.eq(getattr(h,f'player_{i}'))&wrist.source_frame.between(h.frame-8,h.frame+4),'wrist_speed']
                values=values[np.isfinite(values)&values.ge(0)]
                val=getattr(h,f'player_{i}_max')
                assert (not len(values) and pd.isna(val)) or (len(values) and np.isclose(values.max(),val))
        segments=s['points'][str(p)]['rendering']['segments'];pos=positions.set_index(['source_frame','player_slot'])
        held=0
        for seg in segments:
            for slot in b.SLOTS:
                last=None
                observed=set(fixed.loc[fixed.player_slot.eq(slot),'source_frame'])
                for f in range(seg['start_frame'],seg['end_frame']+1):
                    z=pos.loc[f,slot]
                    if f in observed:
                        assert z.filter_status!=r2.HOLD
                        if z.status in b.VALID and pd.notna(z.court_x):last=(f,z.court_x,z.court_y)
                    elif last is not None:
                        assert z.status==z.filter_status==r2.HOLD
                        assert z.offscreen_anchor_frame==last[0]
                        assert np.isclose(z.court_y,last[2])
                        if slot.startswith('near_'):assert np.isclose(z.court_x,last[1])
                        # far x is frozen to review2 even if newly accepted depth
                        # changes the frame of the last valid hold anchor.
                        held+=1
                    else:assert z.status!=r2.HOLD
        marked=r2.far.mark_occlusion(fixed)
        envelope_audit=audit_envelope(p,marked,positions,segments)
        near_audit=audit_near_and_reentry(p,fixed,mapping,positions,s)
        max_x_shift=0.;corrected=0
        for slot in b.SLOTS[2:]:
            ref=pd.read_csv(r2.env.BACKUP/f'cleaned_player_positions_p{p:02d}.csv')
            ref=ref[ref.player_slot.eq(slot)].set_index('source_frame')
            g=positions[positions.player_slot.eq(slot)].set_index('source_frame')
            valid=ref.court_x.ne('')&ref.court_x.notna()&g.court_x.notna()&g.status.ne(r2.HOLD)
            shift=np.abs(ref.loc[valid,'court_x'].astype(float)-g.loc[valid,'court_x'].astype(float))
            assert (shift<1e-12).all(),('horizontal changed',p,slot,float(shift.max()))
            max_x_shift=max(max_x_shift,float(shift.max()) if len(shift) else 0.)
            corrected+=int(g.filter_status.isin([r2.SERVE,r2.LIFT]).sum())
            for q in g[g.filter_status.isin([r2.SERVE,r2.LIFT])].itertuples():
                assert np.isclose(q.court_y,q.effective_court_y)
                observed_y=project(H,(q.x1+q.x2)/2,q.depth_y_px)[1]
                assert np.isclose(q.court_y,observed_y)
            for serve in s['review2_serves']:
                if serve['point']==p and serve['player']==b.NAMES[slot] and serve['applied']:
                    z=g.loc[serve['frame']]
                    assert z.filter_status==r2.SERVE and np.isclose(z.court_y*23.77,24.07,atol=1e-10)
        for event in s['review2_lifts']:
            if event['point']!=p:continue
            assert 0<event['duration_source_frames']<=42
            g=positions[positions.player_slot.eq(event['player_slot'])]
            g=g[g.source_frame.between(event['start_frame'],event['end_frame'])&g.filter_status.eq(r2.LIFT)]
            for q in g.itertuples():
                a,z=int(q.review2_anchor_before_frame),int(q.review2_anchor_after_frame)
                endpoints=pos.loc[[(a,event['player_slot']),(z,event['player_slot'])]]
                y0=float(endpoints.iloc[0].depth_y_px);y1=float(endpoints.iloc[1].depth_y_px)
                t=(q.source_frame-a)/(z-a)
                assert np.isclose(q.depth_y_px,(1-t)*y0+t*y1)
        frame_map=pd.read_csv(out/f'video_source_frame_map_p{p:02d}.csv')
        required=r2.b.REVIEW2_CHECKS.get(p,[])
        if p==7:required=required+[23865]
        expected=[]
        for seg in segments:
            previous={7:[23422,23548],8:[25470],10:[30633],11:[32999]}.get(p,[])
            expected+=sorted(set(range(seg['start_frame'],seg['end_frame']+1,2))|{f for f in required+previous if seg['start_frame']<=f<=seg['end_frame']})
        assert frame_map.source_frame.tolist()==expected
        for f in required:
            path=out/f'check_review2_p{p:02d}_F{f:05d}.jpg'
            assert path.exists() and cv_size(path)==(1920,1080)
        video=out/f'point_{p:02d}.mp4'
        subprocess.run([b.FFMPEG,'-v','error','-xerror','-i',str(video),'-f','null','-'],check=True,capture_output=True)
        probe=json.loads(subprocess.check_output([b.FFPROBE,'-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=width,height,r_frame_rate,nb_read_frames','-of','json',str(video)]))['streams'][0]
        assert int(probe['nb_read_frames'])==len(frame_map) and probe['r_frame_rate']=='15/1'
        assert (probe['width'],probe['height'])==(1920,1080)
        audits[str(p)]=dict(frames=len(frame_map),full_decode_verified=True,original_frame_map_verified=True,
                            envelope=envelope_audit,
                            near_and_reentry=near_audit,
                            offscreen_held_rows_verified=held,depth_corrected_rows=corrected,horizontal_max_shift_normalized=max_x_shift,
                            reentry_newly_assigned_frames=sum(x['newly_assigned_frames'] for x in s['review2_reentries'] if x['point']==p))
    # Confirm the source-labelled endpoint positions exist for offscreen shots.
    for p,f in [(7,23803),(8,25863),(9,28526)]:
        table=pd.read_csv(out/f'cleaned_player_positions_p{p:02d}.csv')
        shot=next(x for x in s['hitter_decisions'] if x['point']==p and x['frame']==f)
        z=table[table.source_frame.eq(f)&table.player_slot.eq(shot['hitter_slot'])].iloc[0]
        assert z.status==r2.HOLD and np.isfinite([z.court_x,z.court_y]).all()
    for auto,display,t in zip(hits.itertuples(),s['hitter_decisions'],truth.itertuples()):
        assert display['hitter']==t.confirmed_hitter and display['auto_hitter']==t.auto_hitter
        assert display['current_auto_hitter']==auto.hitter
    counts=pd.read_csv(out/'review2_far_depth_counts.csv')
    old_counts=pd.read_csv(c.BACKUP/'review2_far_depth_counts.csv')
    assert counts[old_counts.columns].equals(old_counts)
    for row in counts.itertuples():
        table=pd.read_csv(out/f'cleaned_player_positions_p{row.point:02d}.csv')
        selected=pd.Series(False,index=table.index)
        for seg in s['points'][str(row.point)]['rendering']['segments']:selected|=table.source_frame.between(seg['start_frame'],seg['end_frame'])
        above=selected&table.player_slot.eq(row.player_slot)&table.status.isin(b.VALID)&table.court_y.mul(23.77).gt(25.3)
        assert row.review2c_above_25_3m==int(above.sum())
        assert row.review2c_offscreen_held_above_25_3m==int((above&table.status.eq(r2.HOLD)).sum())
    recounted=json.loads(json.dumps(s))
    import review2c as r2c
    r2c.finalize(recounted,write=False)
    assert recounted['review2_depth_counts']==s['review2_depth_counts']
    assert len(s['review2c_checks'])==2
    longest=max(s['review2c_p07_F1_above_intervals'],key=lambda e:(e['duration_source_frames'],-e['start_frame']))
    assert next(q['source_frame'] for q in s['review2c_checks'] if q['point']==7)==longest['midpoint_frame']
    for check in s['review2c_checks']:
        assert cv_size(out/check['file'])==(1920,1080)
        frame_map=pd.read_csv(out/f'video_source_frame_map_p{check["point"]:02d}.csv')
        assert frame_map.iloc[check['output_frame']].source_frame==check['source_frame']
        import cv2
        cap=cv2.VideoCapture(str(out/f'point_{check["point"]:02d}.mp4'))
        cap.set(cv2.CAP_PROP_POS_FRAMES,check['output_frame']);ok,decoded=cap.read();cap.release()
        assert ok
        jpg=cv2.imread(str(out/check['file']))
        assert np.abs(decoded.astype(float)-jpg.astype(float)).mean()<4,'check image is not the mapped video frame'
    f2=pd.read_csv(out/'cleaned_player_positions_p11.csv')
    f2=f2[f2.player_slot.eq('far_front')].set_index('source_frame')
    assert f2.loc[32946:33001].status.isin(b.VALID).all()
    assert f2.loc[33001,'status']!='offscreen_hold','hitter must have accepted/interpolated observed position'
    r2.ignore_large()
    inventory=b.abc.inventory();r2.csv('file_inventory.csv',inventory)
    report=dict(version=r2.VERSION,depth_version=s['review2c_version'],envelope_snapshot_sha256_verified=len(envelope_before['sha256']),review2b_snapshot_sha256_verified=len(current_before['sha256']),rule_boundary_checks=boundaries,protected_sha256_verified=len(before['input_sha256']),
                baseline_sha256_verified=len(before['baseline_sha256']),human_reference_rows=32,
                agreement=agreement,points=audits,baseline_rows=s['review2_baselines'],
                speed_resets=s['review2c_speed_resets'],checks='box provenance, track identity and official exclusion, independent evidence arithmetic, wrist windows, human display, rally-only holds, near reconstruction and existing far x, independent 75th quantiles and depth anchors and reset speed gate, relink gap/distance audit, four-version counts, exact video checks, exact serve anchors, lift time interpolation, source frame maps, all-frame MP4 decode, ignore rules')
    b.dump(out/'review2_verification.json',report);b.dump(out/'verification.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)


def cv_size(path):
    from PIL import Image
    with Image.open(path) as image:return image.size


if __name__=='__main__':main()
