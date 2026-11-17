"""Two-sided local observed-ball branches; preserve the reviewed input.

This is a development proposal on two diagnosed in-play windows. No temporal
exclusion outside anchored connected branches and no inferred coordinates.
"""
import json,csv,math
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
from integrate_044_reviewed_tracks import ROOT,OUT as INTEGRATED,load,read,write
from extract_serve_start_features import sha

OUT=ROOT/'ball_044_branch_recovery'
P=dict(component_peak_min=.15,max_source_frame_gap=12,
    distance_gate_base_px=32.,distance_gate_per_source_frame_px=32.,
    closest_candidate_margin_px=12.,two_direction_agreement_px=1.,anchor_match_px=16.,
    preserve_raw_if_agreement_px=16.,windows=[[890,896],[959,974]],
    scope='Only the two diagnosed044 in-play windows; offline development, not a global ball filter',
    automatic_adoption=False)

def xy(r):return np.array([float(r['x_px']),float(r['y_px'])])

def trace(by,raw,start,end):
    direction=1 if end>start else -1
    seeds=sorted(by.get(start,[]),key=lambda c:np.linalg.norm(xy(c)-xy(raw[start])))
    if not seeds or np.linalg.norm(xy(seeds[0])-xy(raw[start]))>P['anchor_match_px']:return {},{},False
    selected={start:seeds[0]};contexts={};last=start;point=xy(seeds[0])
    for t in range(start+direction,end+direction,direction):
        gap=abs(t-last)
        if gap>P['max_source_frame_gap']:break
        gate=P['distance_gate_base_px']+P['distance_gate_per_source_frame_px']*gap
        options=sorted([(float(np.linalg.norm(xy(c)-point)),c) for c in by.get(t,[])],key=lambda q:q[0])
        valid=[q for q in options if q[0]<=gate]
        contexts[t]=dict(last_observed_frame=last,last_observed_xy=point.tolist(),distance_gate_px=gate,
            raw_distance_px=float(np.linalg.norm(xy(raw[t])-point)) if int(raw[t]['visible']) else None,
            decision='missing_or_outside_gate')
        if not valid:continue
        if len(valid)>1 and valid[1][0]-valid[0][0]<P['closest_candidate_margin_px']:
            contexts[t]['decision']='ambiguous';continue
        selected[t]=valid[0][1];last=t;point=xy(valid[0][1]);contexts[t]['decision']='observed_component_assigned'
    connected=end in selected and np.linalg.norm(xy(selected[end])-xy(raw[end]))<=P['anchor_match_px']
    return selected,contexts,bool(connected)

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'branch_proposals.csv').exists()
    manifest=load(ROOT/'reviewed_ball_overrides.json')['clips']['match1_motion_pilot_016']
    source=Path(manifest['reviewed_output']);assert sha(source)==manifest['reviewed_output_sha256']
    raw=read(source);cp=ROOT/'inplay_ball_044/candidates.csv';candidates=read(cp);by=defaultdict(list)
    for c in candidates:
        if float(c['peak'])>=P['component_peak_min']:by[int(c['local_frame'])].append(c)
    proposed={};diagnostics=[]
    for start,end in P['windows']:
        assert int(raw[start]['visible']) and int(raw[end]['visible'])
        forward,fc,fconnected=trace(by,raw,start,end);backward,bc,bconnected=trace(by,raw,end,start)
        detail=dict(window=[start,end],forward_connected=fconnected,backward_connected=bconnected,
            forward_selected={str(t):c for t,c in forward.items()},backward_selected={str(t):c for t,c in backward.items()},
            forward_context=fc,backward_context=bc)
        diagnostics.append(detail)
        if not (fconnected and bconnected):continue
        for t in range(start+1,end):
            agrees=t in forward and t in backward and np.linalg.norm(xy(forward[t])-xy(backward[t]))<=P['two_direction_agreement_px']
            if agrees:
                c=forward[t]
                if not int(raw[t]['visible']):proposed[t]=dict(action='add_observed_component',component=c)
                elif np.linalg.norm(xy(raw[t])-xy(c))>P['preserve_raw_if_agreement_px']:
                    proposed[t]=dict(action='replace_with_observed_component',component=c)
            elif int(raw[t]['visible']) and t in fc and t in bc:
                # Ambiguity or low score alone never withholds an existing output.
                too_far=all(q['decision']!='ambiguous' and q['raw_distance_px'] is not None and q['raw_distance_px']>q['distance_gate_px'] for q in [fc[t],bc[t]])
                if too_far:proposed[t]=dict(action='withhold_raw_outside_both_connected_branch_gates',component=None)
    rows=[]
    for t in range(739,1141):
        r=raw[t];change=proposed.get(t);c=change['component'] if change else None
        visible=int(r['visible']) if change is None else int(c is not None)
        rows.append(dict(r,branch_proposed_visible=visible,
            branch_proposed_x_px=c['x_px'] if c else (0 if change else r['x_px']),
            branch_proposed_y_px=c['y_px'] if c else (0 if change else r['y_px']),
            branch_proposal_action=change['action'] if change else 'unchanged_reviewed_input',
            branch_component_rank=c['rank'] if c else '',branch_component_peak=c['peak'] if c else '',
            branch_proposal_review='pending' if change else 'not_newly_reviewed',branch_interpolated=0))
    write(OUT/'branch_proposals.csv',rows)
    for r in rows:
        t=int(r['local_frame']);assert all(str(r[k])==raw[t][k] for k in raw[t])
        if t not in proposed:assert str(r['branch_proposed_x_px'])==raw[t]['x_px'] and str(r['branch_proposed_y_px'])==raw[t]['y_px']
        elif proposed[t]['component'] is not None:
            assert any(int(c['local_frame'])==t and c['x_px']==str(r['branch_proposed_x_px']) and c['y_px']==str(r['branch_proposed_y_px']) for c in candidates)
    assert not set(proposed).intersection(manifest['reviewed_changed_local_frames'])
    assert sha(source)==manifest['reviewed_output_sha256']
    report=dict(parameters=P,script_sha256=sha(Path(__file__)),reviewed_input=str(source),reviewed_input_sha256=sha(source),
        candidate_sha256=sha(cp),changes={str(t):c for t,c in proposed.items()},windows=diagnostics,
        change_counts=dict(Counter(c['action'] for c in proposed.values())),
        old_ball_output_frames=sum(int(raw[t]['visible']) for t in range(739,1141)),
        proposed_ball_output_frames=sum(r['branch_proposed_visible'] for r in rows),
        independent_accuracy_measured=False,human_review='pending',
        limitation='Model-chosen endpoint seeds and image-pixel gates are development assumptions, not verified ball labels or physical motion bounds. Withholding requires local review.')
    (OUT/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (OUT/'verification.json').write_text(json.dumps(dict(reviewed_input_and_all_original_columns_preserved=True,
        all_nonmissing_proposed_coordinates_from_same_frame_model_components=True,
        existing_three_user_confirmed_ball_coordinates_unchanged=True,
        no_interpolation=True,global_temporal_filter_not_reenabled=True,
        accuracy_established=False),indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ['windows','parameters']},indent=2))

if __name__=='__main__':main()
