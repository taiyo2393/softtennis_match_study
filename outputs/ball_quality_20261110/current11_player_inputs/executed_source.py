"""Publish the next fixed case as a new, immutable comparison-input snapshot."""
import argparse,csv,hashlib,json,re
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'ball_quality_20261110';COHORT=BASE/'cohort_and_baselines'
INPUT_FILES=['B1_server_side.csv','B2_snapshot.csv','B2_window_means.csv','diagnostic_snapshot_masks_only.csv','diagnostic_window_masks_only.csv']
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def write(p,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--previous',required=True);ap.add_argument('--output',required=True);args=ap.parse_args()
    assert re.fullmatch(r'\d{3}',args.tag) and all(re.fullmatch(r'[A-Za-z0-9_]+',n) for n in [args.previous,args.output])
    previous=BASE/args.previous;out=BASE/args.output;case=BASE/'cohort_tracking'/f'case_{args.tag}'
    assert previous!=out and not out.exists();assert load(previous/'verification.json')['status']=='passed'
    verification=load(case/'verification.json');assert verification['status']=='passed'
    p=load(case/'protocol.json');result=load(case/'case_result.json');findings=load(case/'AI_visual_findings.json');cid=p['candidate_id']
    assert findings['input_phase_status']=='ai_visual_sequence_supported','Keep doubtful phase cases in tracking registry; resolve model-input scope separately.'
    files=[x for x in previous.iterdir() if x.is_file()]+[case/n for n in ['protocol.json','case_result.json','verification.json','AI_visual_findings.json','snapshot_player_features.csv','prefix_player_summary_features.csv']]
    files += [COHORT/n for n in ['fixed30_processing_queue.csv','all_candidate_group_registry.csv','basic_server_inputs_11.csv','human_outcomes_11.csv']]
    protected={str(x):sha(x) for x in files};out.mkdir()
    dump(out/'protocol.json',dict(date_jst='2026-11-10',previous_snapshot=str(previous),added_candidate=cid,
        purpose='Fixed chronological cohort input construction; not model fitting or independent accuracy evaluation.',
        unchanged='Retain prior rows, comparisons, fixed cohort order and missing-value definitions. Do not select an alternate time at missing endpoints.',
        phase='New candidate has AI chronological-image support only; exact contact/point end and identity/position GT remain unconfirmed.',
        ball='No missing ball datasets converted to no-ball observations; no new ball comparison.',
        protected_input_hashes=protected,script_sha256=sha(Path(__file__))))
    basic={r['rally_candidate_id']:r for r in read(COHORT/'basic_server_inputs_11.csv')}
    labels={r['rally_candidate_id']:r for r in read(COHORT/'human_outcomes_11.csv')}
    base=basic.get(cid,dict(rally_candidate_id=cid,server_side_near=''))
    label=labels.get(cid,dict(rally_candidate_id=cid,winner_camera_side='',human_side_review_id=''))
    snapshot=read(case/'snapshot_player_features.csv')[0];summary=read(case/'prefix_player_summary_features.csv')[0]
    assert snapshot['rally_candidate_id']==summary['rally_candidate_id']==cid
    geom=[side+'_'+f for side in ['near','far'] for f in ['center_x_norm','center_y_norm','x_gap_norm','y_gap_norm']]
    snap_mask=[side+'_pair_valid' for side in ['near','far']];window_mask=[side+'_pair_observed_fraction' for side in ['near','far']]
    new={'B1_server_side.csv':base,'B2_snapshot.csv':dict(base,**{k:snapshot[k] for k in geom+snap_mask}),
        'B2_window_means.csv':dict(base,**{k:summary[k] for k in [g+'_mean' for g in geom]+window_mask}),
        'diagnostic_snapshot_masks_only.csv':dict(base,**{k:snapshot[k] for k in snap_mask}),
        'diagnostic_window_masks_only.csv':dict(base,**{k:summary[k] for k in window_mask})}
    allowed=set(geom+[g+'_mean' for g in geom]+snap_mask+window_mask+['rally_candidate_id','server_side_near'])
    allids=None
    for name in INPUT_FILES:
        old=read(previous/name);oldmap={r['rally_candidate_id']:r for r in old};assert cid not in oldmap
        assert set(new[name])==set(old[0]) and set(new[name])<=allowed
        rows=sorted(old+[new[name]],key=lambda r:r['rally_candidate_id']);write(out/name,rows)
        assert all(r==next(x for x in rows if x['rally_candidate_id']==r['rally_candidate_id']) for r in old)
        ids=[r['rally_candidate_id'] for r in rows]
        if allids is None:allids=ids
        else:assert ids==allids
    for name,newrow in [('human_outcomes.csv',label),('source_snapshot_features.csv',snapshot),('source_window_summary_features.csv',summary)]:
        old=read(previous/name);assert cid not in {r['rally_candidate_id'] for r in old}
        assert set(newrow)==set(old[0]);write(out/name,sorted(old+[newrow],key=lambda r:r['rally_candidate_id']))
    registry={r['candidate_id']:r for r in read(COHORT/'all_candidate_group_registry.csv')};meta=read(previous/'rally_metadata.csv')
    meta.append(dict(rally_candidate_id=cid,point_group_id=registry[cid]['point_group_id'],in_fixed30=1,
        input_first_source_frame=p['requested_input_frames'][0],input_end_source_frame=p['requested_input_frames'][-1],
        four_observation_times=result['input_times_with_four_observations'],requested_times=31,
        phase_evidence='AI_chronological_images_support_play;exact_point_end_not_confirmed',independent_identity_position_accuracy_established=0,
        tracking_source=str(case/'four_player_path.csv')))
    meta.sort(key=lambda r:r['rally_candidate_id']);write(out/'rally_metadata.csv',meta)
    progress=read(previous/'fixed30_processing_progress.csv');pending=[r for r in progress if r['fixed_window_features_constructed']=='0']
    assert pending[0]['candidate_id']==cid,'Preserve chronological order; retain failures rather than skip.'
    for r in progress:
        if r['candidate_id']==cid:r.update(fixed_window_features_constructed='1',status=f'new_{args.tag}_candidate_constructed_AI_image_check')
    write(out/'fixed30_processing_progress.csv',progress)
    assert all(sha(Path(n))==h for n,h in protected.items())
    schema=load(previous/'feature_schema.json');dump(out/'feature_schema.json',schema)
    v=dict(status='passed',unique_candidate_rows=len(allids),point_groups=len({r['point_group_id'] for r in meta}),
        previous_model_input_rows_exactly_retained=True,candidate_order_identical_across_all_comparisons=True,
        input_columns_allowlisted=True,no_outcome_or_absolute_time_columns_in_inputs=True,
        protected_sources_unchanged=True,missing_endpoint_retained=snapshot['far_pair_valid']=='0',
        independent_identity_or_position_accuracy_established=False)
    dump(out/'verification.json',v)
    counts=Counter(r['winner_camera_side'] or 'unknown' for r in read(out/'human_outcomes.csv'))
    done=sum(int(r['fixed_window_features_constructed']) for r in progress);pending=[r for r in progress if r['fixed_window_features_constructed']=='0']
    dump(out/'result.json',dict(status='expanded_comparison_inputs_constructed',candidates=len(allids),added_candidate=cid,outcome_counts=dict(counts),
        fixed30_constructed_inputs=done,fixed30_pending_inputs=len(pending),next_candidate=pending[0]['candidate_id'] if pending else None,
        new_case_four_observation_times=result['input_times_with_four_observations'],new_case_requested_times=31,
        model_fitting_performed=False,performance_estimated=False,ball_comparison_ready=False,new_human_annotation_requested=False,
        canonical_tracking_changed=False,artifact_hashes={x.name:sha(x) for x in out.glob('*.csv')}))
    print(json.dumps(load(out/'result.json'),ensure_ascii=False))

if __name__=='__main__':main()
