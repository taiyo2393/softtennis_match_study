"""Evaluate manual image-position references; no labels means no accuracy result."""
import argparse,json,itertools,math
from collections import defaultdict
from pathlib import Path
from run_fixed_cohort_player_case_v3 import BASE,read,write,dump,sha

OUT=BASE/'identity_position_bundle'
PILOT=OUT/'reference_pilot'

def pair_result(reference,prediction):
    states=[r.get('state','pending') for r in reference];points=[(float(r['x']),float(r['y'])) for r in reference if r.get('state')=='point'];result=dict(reference_states='|'.join(states),reference_pair_reviewed=int(all(s!='pending' for s in states)),reference_position_points=len(points),prediction_points=len(prediction),position_pair_evaluable=int(len(points)==2),prediction_pair_available=int(len(prediction)==2),evaluation_state='pending_reference' if 'pending' in states else 'reference_positions_unavailable' if len(points)!=2 else 'prediction_pair_incomplete' if len(prediction)!=2 else 'scored',mean_point_error_px='',mean_point_error_over_image_height='',centroid_x_error_norm='',centroid_y_error_norm='',x_gap_error_norm='',y_gap_error_norm='')
    if result['evaluation_state']=='scored':
        a,b=min(itertools.permutations(prediction),key=lambda pair:sum(math.dist(x,y) for x,y in zip(points,pair)));errors=[math.dist(points[0],a),math.dist(points[1],b)];result['mean_point_error_px']=sum(errors)/2;result['mean_point_error_over_image_height']=result['mean_point_error_px']/2160
        xs=[p[0] for p in points];ys=[p[1] for p in points];px=[p[0] for p in prediction];py=[p[1] for p in prediction];result.update(centroid_x_error_norm=abs(sum(xs)-sum(px))/2/3840,centroid_y_error_norm=abs(sum(ys)-sum(py))/2/2160,x_gap_error_norm=abs(abs(xs[0]-xs[1])-abs(px[0]-px[1]))/3840,y_gap_error_norm=abs(abs(ys[0]-ys[1])-abs(py[0]-py[1]))/2160)
    return result

def self_test():
    ref=[dict(state='point',x=100.,y=200.),dict(state='point',x=300.,y=400.)]
    a=pair_result(ref,[(300.,400.),(100.,200.)]);assert a['evaluation_state']=='scored' and a['mean_point_error_px']==0
    b=pair_result(ref,[(100.,200.)]);assert b['evaluation_state']=='prediction_pair_incomplete' and b['mean_point_error_px']=='' and b['position_pair_evaluable']==1
    c=pair_result([dict(state='unplaceable'),ref[1]],[(100.,200.),(300.,400.)]);assert c['evaluation_state']=='reference_positions_unavailable' and c['mean_point_error_px']==''
    d=pair_result([dict(state='pending'),ref[1]],[]);assert d['evaluation_state']=='pending_reference'
    e=pair_result(ref,[(103.,204.),(303.,404.)]);assert e['mean_point_error_px']==5 and e['centroid_x_error_norm']==3/3840 and e['centroid_y_error_norm']==4/2160
    return dict(status='passed',synthetic_cases=5,real_human_labels_created=False)

def main():
    labels_path=PILOT/'human_reference_labels.json';dest=PILOT/'evaluation';dest.mkdir(exist_ok=True)
    if not labels_path.exists():dump(dest/'status.json',dict(status='awaiting_manual_reference_labels',accuracy_estimated=False,no_model_as_reference=True));print('Awaiting manual reference labels; no accuracy values generated.');return
    labels=json.loads(labels_path.read_text(encoding='utf-8'));assert labels['label_origin']=='manual_reference_ui';items=json.loads((PILOT/'private_sampling_manifest.json').read_text(encoding='utf-8'))['items'];pred=defaultdict(list)
    for r in read(OUT/'position_policy_observations.csv'):
        if r['analysis_proxy_present']=='1':pred[r['candidate_id'][-3:],int(r['source_frame']),r['camera_side'],r['position_policy']].append((float(r['analysis_proxy_x']),float(r['analysis_proxy_y'])))
    rows=[]
    for item in items:
        for group in item['sampling_groups']:
            sides=['near','far'] if group=='uniform_one_per_context' else [item['target_slot'].split('_')[0]]
            for side in sides:
                refs=[labels.get('labels',{}).get(item['image_id'],{}).get(f'{side}_{lr}',dict(state='pending')) for lr in ['left','right']]
                for policy in ['retain_native_proxy','withhold_body_warning_proxy']:rows.append(dict(sampling_group=group,image_id=item['image_id'],tag=item['tag'],source_frame=item['source_frame'],camera_side=side,position_policy=policy,**pair_result(refs,pred[item['tag'],item['source_frame'],side,policy])))
    write(dest/'per_context_side_results.csv',rows);summaries=[]
    for group in ['uniform_one_per_context','known_body_warning_challenge']:
        for policy in ['retain_native_proxy','withhold_body_warning_proxy']:
            selected=[r for r in rows if r['sampling_group']==group and r['position_policy']==policy];scored=[r for r in selected if r['evaluation_state']=='scored'];summaries.append(dict(sampling_group=group,position_policy=policy,reference_context_side_pairs=len(selected),reviewed_reference_pairs=sum(r['reference_pair_reviewed'] for r in selected),reference_position_pairs_available=sum(r['position_pair_evaluable'] for r in selected),scored_pairs=len(scored),prediction_incomplete_with_reference_available=sum(r['evaluation_state']=='prediction_pair_incomplete' for r in selected),reference_unplaceable_or_uncertain_pairs=sum(r['evaluation_state']=='reference_positions_unavailable' for r in selected),pending_reference_pairs=sum(r['evaluation_state']=='pending_reference' for r in selected),conditional_mean_point_error_px=sum(r['mean_point_error_px'] for r in scored)/len(scored) if scored else '',not_ID_accuracy=1,not_independent_match_test=1))
    write(dest/'coverage_and_conditional_error_summary.csv',summaries);dump(dest/'status.json',dict(status='reference_evaluation_generated',reference_labels_sha256=sha(labels_path),reference_revision=labels['revision'],score_scope='Same-match development image-position and arrangement error, unordered within-side correspondence; no identity continuity or ground-contact claim.',sampling_groups_never_pooled=True,missing_predictions_and_unplaceable_references_retained=True,no_error_pass_threshold=True,no_outcome_model=True));print(dest)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--self-test',action='store_true');args=ap.parse_args()
    if args.self_test:print(self_test())
    else:main()
