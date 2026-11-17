"""Fixed presence/unknown DP after frozen temporal candidate paths.

Parameters chosen on123training labels only. These gates only keep or withhold
the already selected target observation; no new coordinates or absence GT.
"""
import argparse,json,math
from pathlib import Path
from itertools import product
import numpy as np
from integrate_044_reviewed_tracks import ROOT,load,read,write,sha
import compare_temporal_candidate_paths as temporal
from compare_ball_candidate_decoders import choose
from run_manual_ball_finetune import metrics
from run_hard_case_ball_finetune import good_ids,err
from evaluate_observed_ball_fallback import camera_mask

OUT=ROOT/'ball_abstention_comparison';BASE=ROOT/'ball_temporal_path_comparison'
CONFIGS=[dict(name=f'unknown{q:g}_switch{s:g}',unknown_cost=q,switch_cost=s) for q in [1.2,1.8,2.4] for s in [.25,.5,1.]]
TEMPORAL=dict(name='displacement_0.1',order='displacement',strength=.1)
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')

def binary_path(points,config):
    """State0=keep observed candidate,1=unknown. Missing frames only allow1."""
    emissions=np.array([[(-math.log(max(p['component_peak'],1e-8))) if p else np.inf,
        config['unknown_cost'] if p else 0.] for p in points],dtype=np.float64)
    costs=emissions[0];backs=[]
    for t in range(1,len(points)):
        d=math.hypot(points[t]['x_px']-points[t-1]['x_px'],points[t]['y_px']-points[t-1]['y_px']) if points[t] and points[t-1] else 0
        transition=np.array([[.1*min(d/64.,4.),config['switch_cost']],[config['switch_cost'],0.]])
        total=costs[:,None]+transition;back=np.argmin(total,axis=0)
        costs=total[back,np.arange(2)]+emissions[t];backs.append(back)
    states=[int(np.argmin(costs))]
    for back in reversed(backs):states.append(int(back[states[-1]]))
    return list(reversed(states))

def gate(context,t,mask,config):
    original,status=temporal.select(context,t,mask,TEMPORAL)
    if original is None:return None,'already_missing'
    if config is None:return original,'frozen_temporal_baseline'
    times=sorted(j for j in context if j in {t+d for d in temporal.OFFSETS})
    if mask[min(times):max(times)+1].any():return original,'camera_guard_baseline_retained'
    chosen={j:None for j in times};segments=[];segment=[]
    for j in times:
        if context[j]:
            if segment and j!=segment[-1]+2:segments.append(segment);segment=[]
            segment.append(j)
        elif segment:segments.append(segment);segment=[]
    if segment:segments.append(segment)
    for segment in segments:
        if len(segment)<3:
            for j in segment:chosen[j]=choose(context[j],'bbox_area')
        else:
            nodes=[sorted(context[j],key=lambda c:(-c['bbox_area'],c['component_index'])) for j in segment]
            path=temporal.best_path(nodes,'displacement',.1)
            for j,cs,k in zip(segment,nodes,path,strict=True):chosen[j]=cs[k]
    assert chosen[t]['component_index']==original['component_index']
    states=binary_path([chosen[j] for j in times],config)
    return (original,'kept_observed_candidate') if states[times.index(t)]==0 else (None,'withheld_position_unknown')

def self_test():
    c=dict(unknown_cost=1.2,switch_cost=.5)
    p=lambda x,peak:dict(x_px=x,y_px=100,component_peak=peak)
    assert binary_path([None,p(100,.65),None],c)==[1,1,1]
    assert binary_path([p(100,.8),p(110,.8),p(120,.8)],c)==[0,0,0]
    assert binary_path([None,p(100,.95),None],dict(unknown_cost=2.4,switch_cost=.5))==[1,0,1]
    points=[p(100,.8),None,p(120,.4),p(130,.8)]
    def cost(states):
        total=0.
        for i,(s,point) in enumerate(zip(states,points)):
            if s==0 and point is None:return np.inf
            total+=(-math.log(point['component_peak']) if s==0 else c['unknown_cost'] if point else 0)
            if i:
                if s!=states[i-1]:total+=c['switch_cost']
                elif s==0:total+=.1*min(abs(point['x_px']-points[i-1]['x_px'])/64,4)
        return total
    actual=binary_path(points,c);assert abs(cost(actual)-min(cost(s) for s in product([0,1],repeat=4)))<1e-10
    print('Passed: unknown option, retained strong/coherent observations, no gap filling, exact DP vs exhaustive search.')

def contexts(path):return {cid:{int(t):cs for t,cs in table.items()} for cid,table in load(path).items()}
def masks_for(cids,protected):
    result={}
    for cid in cids:
        p=ROOT.parent/'ball_quality_20261102/temporal_guard/camera_features'/f'{cid}.json';protected[str(p)]=sha(p)
        result[cid]=camera_mask(load(p)['changed_fraction'])
    return result

def predict(rows,cache,masks,config,role):
    result=[]
    for r in rows:
        t=int(r['local_frame']);c,status=gate(cache[r['candidate_id']],t,masks[r['candidate_id']],config)
        px,py=(c['x_px'],c['y_px']) if c else (0,0)
        result.append(dict(sample_id=r['sample_id'],candidate_id=r['candidate_id'],local_frame=r['local_frame'],source_frame=r['source_frame'],
            role=role,method=config['name'] if config else 'frozen_temporal_baseline',
            truth_visibility=r.get('truth_visibility',''),truth_x_px=r.get('truth_x_px',''),truth_y_px=r.get('truth_y_px',''),
            x_px=px,y_px=py,visible=int(px!=0 or py!=0),confidence=c['bbox_mean'] if c else 0.,abstention_status=status,
            physical_ball_absence_inferred=0,interpolated=0))
    return result

def compare(rows,cache,masks,config,role):
    baseline=predict(rows,cache,masks,None,role);proposed=predict(rows,cache,masks,config,role)
    for a,b in zip(baseline,proposed,strict=True):
        assert not int(b['visible']) or (b['x_px'],b['y_px'],b['visible'])==(a['x_px'],a['y_px'],a['visible'])
    write(OUT/f'{role}_temporal_baseline.csv',baseline);write(OUT/f'{role}_abstention.csv',proposed)
    a=metrics(baseline);b=metrics(proposed);lost=sorted(good_ids(baseline)-good_ids(proposed))
    return dict(baseline=a,proposed=b,original_correct_lost=lost,nonregression=not lost and err(b)<=err(a),
        outputs_withheld=sum(int(x['visible']) and not int(y['visible']) for x,y in zip(baseline,proposed))),baseline,proposed

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists(),'Preserve fixed trial; no post-result grid search.'
    data=ROOT/'balltrack_manual_finetune_pilot';train_path=ROOT/'balltrack_expanded_centers_pilot/expanded_train_labels.csv'
    review=ROOT/'ball_040_object_swap_review/round2/user_reviewed'
    protected={}
    paths=[ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json',
        train_path,data/'validation_labels.csv',data/'audit_labels.csv',ROOT/'ball_inplay_challenge/manual_center_labels.csv',
        BASE/'training_selection.json',review/'candidate_review.csv',review/'review.json',
        ROOT/'temporal_rally_reviews_v2/match1_motion_pilot_015/balltrack_predictions_proposed.csv',
        ROOT/'temporal_rally_reviews_v2/match1_motion_pilot_015/full_scope_component_cache.json']
    for role in ['train','validation','challenge','audit']:
        paths.extend([BASE/f'{role}_component_cache.json',BASE/f'{role}_context_image_hashes.json'])
    for p in paths:protected[str(p)]=sha(p)
    assert load(BASE/'training_selection.json')['chosen_configuration']==TEMPORAL
    # Existing cached data are used; inspect their original model checkpoint hash.
    model=Path(load(BASE/'protocol.json')['original_model']);assert sha(model)==load(BASE/'protocol.json')['protected_input_hashes'][str(model)]
    protected[str(model)]=sha(model)
    protocol=dict(status='fixed_before_training_gate_selection',base_selector=TEMPORAL,configurations=CONFIGS,
        costs='Selected candidate -log(peak); unknown q when a candidate exists,0 otherwise; changing keep/unknown costs switch; keep-to-keep uses previous displacement0.1cost',
        paths='Freeze candidate path from previous selector first, then2state DP keep/unknown across gaps. Never choose new coordinates, fill gaps or infer physical absence.',
        training='Only123labels on8training clips. Include no-abstention baseline; retain all66correct and errors<=4; minimise errors then maximise retained correct; ties baseline then listed order.',
        evaluation='ONE training-selected configuration on120validation and4challenge; evaluate6newfalse-candidate labels as a separate diagnostic regardless of gates. No tuning using those labels.',
        six_case_source='Before-manual-exclusion v2proposal and raw model components. Never score already removed zero-output rows as automatic rejections.',
        audit='Only if validation and challenge keep all previously correct points without adding errors, and there is measurable withheld-error benefit; one config only.',
        fallback='Camera guard retains previous selected output; no automatic removal of prior human-confirmed points/registry updates',
        scope='Known same-match developmental failures motivated design. Training-only numeric selection is not independent external validation.',
        automatic_adoption=False,protected_input_hashes=protected,script_sha256=sha(Path(__file__)))
    dump(OUT/'protocol.json',protocol)
    train=read(train_path);assert len(train)==123
    cache=contexts(BASE/'train_component_cache.json');masks=masks_for(cache,protected)
    baseline=predict(train,cache,masks,None,'train');bm=metrics(baseline);original=good_ids(baseline)
    assert len(original)==66 and err(bm)==4
    write(OUT/'train_temporal_baseline.csv',baseline);best=(4,-66);chosen=None;grid=[]
    for config in CONFIGS:
        rows=predict(train,cache,masks,config,'train');m=metrics(rows);lost=sorted(original-good_ids(rows));eligible=not lost and err(m)<=4
        grid.append(dict(config,correct=len(good_ids(rows)),errors=err(m),correct_lost=len(lost),lost_sample_ids=';'.join(lost),eligible=int(eligible),
            withheld=sum(int(a['visible']) and not int(b['visible']) for a,b in zip(baseline,rows))))
        write(OUT/f'train_{config["name"]}.csv',rows)
        if eligible and (err(m),-len(good_ids(rows)))<best:chosen=config;best=(err(m),-len(good_ids(rows)))
    write(OUT/'training_grid.csv',grid)
    selection=dict(chosen=chosen,training_baseline=bm,grid=grid,validation_labels_used=False,six_false_labels_used=False,challenge_labels_used=False,choice_frozen=True)
    dump(OUT/'training_selection.json',selection);print('TRAIN SELECT',json.dumps(selection),flush=True)
    result=dict(status='fixed_abstention_trial_complete',chosen_configuration=chosen,training=selection,validation=None,challenge=None,six_rejected_candidates=None,audit=None,
        operational_outputs_changed=False,automatic_adoption=False)
    if chosen:
        val=read(data/'validation_labels.csv');cache=contexts(BASE/'validation_component_cache.json');masks=masks_for(cache,protected)
        result['validation'],_,_=compare(val,cache,masks,chosen,'validation')
        assert result['validation']['baseline']['counts']['localized_tp_16px']==81 and err(result['validation']['baseline'])==2
        labels={r['sample_id']:r for r in read(ROOT/'ball_inplay_challenge/manual_center_labels.csv')}
        query=[dict(r,**{k:labels[r['sample_id']][k] for k in ['truth_visibility','truth_x_px','truth_y_px']}) for r in read(ROOT/'ball_inplay_challenge/query_model_inputs.csv')]
        cache=contexts(BASE/'challenge_component_cache.json');masks=masks_for(cache,protected)
        result['challenge'],_,_=compare(query,cache,masks,chosen,'challenge')
        assert result['challenge']['baseline']['counts']['localized_tp_16px']==4
        # Now open false labels, confirm exact displayed coordinates, and score only candidate retention.
        false_rows=read(review/'candidate_review.csv');source=read(ROOT/'temporal_rally_reviews_v2/match1_motion_pilot_015/balltrack_predictions_proposed.csv')
        cache=contexts(ROOT/'temporal_rally_reviews_v2/match1_motion_pilot_015/full_scope_component_cache.json');masks=masks_for(cache,protected);outcomes=[]
        for r in false_rows:
            t=int(r['local_frame']);p=source[t];assert int(p['visible']) and float(p['x_px'])==float(r['displayed_x_px']) and float(p['y_px'])==float(r['displayed_y_px'])
            assert temporal.select(cache['match1_motion_pilot_015'],t,masks['match1_motion_pilot_015'],TEMPORAL)[0] is not None
            c,status=gate(cache['match1_motion_pilot_015'],t,masks['match1_motion_pilot_015'],chosen)
            outcomes.append(dict(local_frame=t,source_frame=r['source_frame'],known_false_x_px=r['displayed_x_px'],known_false_y_px=r['displayed_y_px'],
                known_false_candidate_withheld=int(c is None),decision=status,true_ball_location='unknown',physical_visibility='not_labelled'))
        write(OUT/'six_false_candidate_outcomes.csv',outcomes)
        result['six_rejected_candidates']=dict(cases=6,withheld=sum(r['known_false_candidate_withheld'] for r in outcomes),retained=sum(not r['known_false_candidate_withheld'] for r in outcomes),
            true_center_recovery_not_evaluated=True,manual_zero_outputs_not_used=True)
        passes=result['validation']['nonregression'] and result['challenge']['nonregression']
        if passes and (err(result['validation']['proposed'])<2 or result['six_rejected_candidates']['withheld']>0):
            audit=read(data/'audit_labels.csv');cache=contexts(BASE/'audit_component_cache.json');masks=masks_for(cache,protected)
            result['audit'],_,_=compare(audit,cache,masks,chosen,'audit')
            assert result['audit']['baseline']['counts']['localized_tp_16px']==35
    assert all(sha(Path(p))==h for p,h in protected.items())
    dump(OUT/'verified_inputs.json',dict(protected_input_hashes=protected,all_unchanged=True))
    result.update(protected_inputs_unchanged=True,interpolation=0,physical_absence_labels_added=0,
        interpretation='Withheld output means unknown position, not absent ball. Lost correct observations and extra missing data must be reported with error reduction.')
    dump(OUT/'decision.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:self_test()
    else:main()
