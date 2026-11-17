"""Label-blind policy replay on frozen displayed candidates; development only.

No new detections, labels, interpolation or adopted outputs. Reviewed candidate
labels are used only after all policy choices have been generated.
"""
import json,math
from pathlib import Path
from collections import defaultdict
import numpy as np
from integrate_044_reviewed_tracks import ROOT,load,read,write,sha
from propose_044_ball_branches import trace,P as BRANCH_PARAMETERS

OUT=ROOT/'ball_044_temporal_replay'
CV_PARAMETERS=dict(gate_base_px=16.,gate_per_source_frame_px=4.,velocity_update_weight=.5,
    max_source_frame_gap=12,closest_candidate_margin_px=12.,anchor_match_px=16.)

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')
def point(c):return np.array([float(c['x_px']),float(c['y_px'])])
def key(c):return f"{int(c['local_frame'])}:{c['display_label']}" if c else ''

def velocity_replay(by,raw,start,end):
    options=sorted(by.get(start,[]),key=lambda c:float(np.linalg.norm(point(c)-point(raw[start]))))
    if not options or np.linalg.norm(point(options[0])-point(raw[start]))>CV_PARAMETERS['anchor_match_px']:return {},{}
    selected={start:options[0]};last=start;p=point(options[0]);v=np.zeros(2);contexts={}
    for t in range(start+1,end+1):
        gap=t-last
        if gap>CV_PARAMETERS['max_source_frame_gap']:
            contexts[t]=dict(decision='long_gap_stop');break
        prediction=p+v*gap
        gate=CV_PARAMETERS['gate_base_px']+CV_PARAMETERS['gate_per_source_frame_px']*gap
        options=sorted([(float(np.linalg.norm(point(c)-prediction)),c) for c in by.get(t,[])],key=lambda q:q[0])
        valid=[q for q in options if q[0]<=gate]
        contexts[t]=dict(last_observation_frame=last,prediction_xy=prediction.tolist(),gate_px=gate,
            nearest_residual_px=options[0][0] if options else None,decision='no_candidate_in_gate')
        if not valid:continue
        if len(valid)>1 and valid[1][0]-valid[0][0]<CV_PARAMETERS['closest_candidate_margin_px']:
            contexts[t]['decision']='ambiguous';continue
        c=valid[0][1];newp=point(c);observed_v=(newp-p)/gap
        w=CV_PARAMETERS['velocity_update_weight'];v=(1-w)*v+w*observed_v
        selected[t]=c;last=t;p=newp;contexts[t]['decision']='observed_candidate_selected'
    return selected,contexts

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists(),'Keep fixed replays; do not tune in place.'
    candidates_path=ROOT/'ball_044_scale_probe/displayed_candidates.csv'
    baseline=ROOT.parent/'tracking_quality_20261102/ball_aligned/match1_motion_pilot_016/balltrack_predictions.csv'
    labelled_path=ROOT/'ball_044_scale_probe/user_reviewed/review.json'
    protected_paths=[ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json']
    registry=load(protected_paths[0])['clips']['match1_motion_pilot_016'];protected_paths.append(Path(registry['reviewed_output']))
    protected={str(p):sha(p) for p in protected_paths}
    protocol=dict(status='specified_before_replay_not_independent_preregistration',
        goal='Check whether temporal association alone rejects newly reviewed tiled mistakes while retaining reviewed real candidates',
        candidate_input=str(candidates_path),candidate_sha256=sha(candidates_path),
        candidate_pool='Exactly the displayed, deduplicated tiled candidates; no labels filter this pool',
        candidate_peak_floor=.15,peak_floor_reason='Reuse unchanged existing branch gate; no sweep on these labels',
        anchor_frames=[1080,1096],anchor_source='Original uncorrected model outputs, not full-track ground truth',
        base_input=str(baseline),base_sha256=sha(baseline),scored_window=[1083,1093],
        policy_names=['highest_peak','forward_nearest','backward_nearest','two_direction_agreement','velocity_gate'],
        branch_parameters=dict(BRANCH_PARAMETERS),velocity_parameters=CV_PARAMETERS,
        velocity_parameter_status='New illustrative development policy chosen after seeing the failure cases; no validation or parameter search',
        script_sha256=sha(Path(__file__)),branch_script_sha256=sha(ROOT/'propose_044_ball_branches.py'),
        evaluation_label_source=str(labelled_path),evaluation_label_sha256=sha(labelled_path),
        evaluation_timing='Read labels only after generating policy selections; existing human knowledge makes this a retrospective development replay',
        field_meanings='Retained/rejected counts refer to exact displayed candidates, not physical visibility or frame-center accuracy',
        automatic_adoption=False,independent_accuracy_evaluation=False,protected_input_hashes=protected)
    dump(OUT/'protocol.json',protocol)
    raw=read(baseline);by=defaultdict(list)
    for c in read(candidates_path):
        if float(c['peak'])>=protocol['candidate_peak_floor']:by[int(c['local_frame'])].append(c)
    start,end=protocol['anchor_frames']
    forward,fc,fconnected=trace(by,raw,start,end)
    backward,bc,bconnected=trace(by,raw,end,start)
    agreement={t:c for t,c in forward.items() if t in backward and key(c)==key(backward[t])} if fconnected and bconnected else {}
    strongest={t:max(by[t],key=lambda c:float(c['peak'])) for t in range(start,end+1) if by[t]}
    velocity,vc=velocity_replay(by,raw,start,end)
    policies=dict(highest_peak=strongest,forward_nearest=forward,backward_nearest=backward,
        two_direction_agreement=agreement,velocity_gate=velocity)
    selections=[]
    for name,selected in policies.items():
        for t in range(start,end+1):
            c=selected.get(t)
            selections.append(dict(policy=name,local_frame=t,selected_candidate=key(c),
                x_px=c['x_px'] if c else '',y_px=c['y_px'] if c else '',
                candidate_peak=c['peak'] if c else '',interpolated=0))
    write(OUT/'policy_selections.csv',selections)
    # Decisions have been saved. Evaluate exact displayed candidate IDs now.
    review=load(labelled_path);positive={key(c):c for c in review['confirmed_observations']}
    negative={key(c):c for c in review['rejected_tiled_components']}
    assert len(positive)==8 and len(negative)==4 and not set(positive)&set(negative)
    annotations=[];summaries=[]
    for name,selected in policies.items():
        kept={key(c) for t,c in selected.items() if 1083<=t<=1093}
        for status,labelled in [('confirmed_ball',positive),('confirmed_false',negative)]:
            for identity,c in labelled.items():
                annotations.append(dict(policy=name,candidate=identity,local_frame=c['local_frame'],
                    label=status,selected=int(identity in kept),peak=c['peak'],
                    prediction_component_label=selected.get(int(c['local_frame']),{}).get('display_label','')))
        tp=sorted(kept&set(positive));fp=sorted(kept&set(negative))
        missed=sorted(set(positive)-kept)
        summaries.append(dict(policy=name,reviewed_ball_candidates_retained=len(tp),
            reviewed_ball_candidates_total=len(positive),reviewed_false_candidates_selected=len(fp),
            reviewed_false_candidates_total=len(negative),retained_ball_candidate_ids=';'.join(tp),
            selected_false_candidate_ids=';'.join(fp),missed_ball_candidate_ids=';'.join(missed),
            unreviewed_candidate_selections_in_scored_window=len(kept-set(positive)-set(negative)),
            qualifies_for_adoption=int(len(tp)==8 and len(fp)==0)))
    write(OUT/'reviewed_candidate_outcomes.csv',annotations);write(OUT/'policy_summary.csv',summaries)
    counterexample=dict(candidate='1092:A',review_label='user_confirmed_is_not_ball',
        preceding_confirmed_ball='1091:B',preceding_distance_px=float(np.linalg.norm(point(negative['1092:A'])-point(positive['1091:B']))),
        velocity_gate_selected=1092 in velocity and key(velocity[1092])=='1092:A',velocity_context=vc.get(1092),
        warning='Motion prediction is a candidate-association aid; passing its gate is not evidence that the object is a ball.')
    assert all(sha(Path(p))==h for p,h in protected.items())
    for p,h in load(ROOT/'play_start_audit/inventory_summary.json')['source_hashes'].items():assert sha(Path(p))==h
    decision=dict(status='temporal_only_policies_fail_existing_reviewed_counterexamples',
        automatic_adoption=False,reviewed_output_unchanged=True,source_annotations_unchanged=True,
        forward_endpoint_connected=fconnected,backward_endpoint_connected=bconnected,
        any_policy_retains_all8_and_rejects_all4=any(r['qualifies_for_adoption'] for r in summaries),
        policy_summary=summaries,counterexample1092=counterexample,
        conclusion='Do not globally filter, interpolate, or select solely by peak/position/velocity. Existing local reviewed corrections remain authoritative.',
        next='Before adding more local rules, compare object appearance and heatmap shape on the exact reviewed true/false cases. Separate detector evidence from trajectory association and retain an abstain state.',
        scope='Retrospective development failure check on one selected rally, not an independent accuracy evaluation')
    assert not decision['any_policy_retains_all8_and_rejects_all4']
    dump(OUT/'decision.json',decision)
    dump(OUT/'contexts.json',dict(forward=fc,backward=bc,velocity=vc))
    print(json.dumps(decision,indent=2))

if __name__=='__main__':main()
