"""Join the first chronological expansion case without replacing the frozen eight."""
import csv,hashlib,json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'ball_quality_20261110'
OLD=ROOT.parent/'ball_quality_20261109/rally_player_prefix_expansion'
CASE=BASE/'cohort_tracking/case_001';COHORT=BASE/'cohort_and_baselines'
OUT=BASE/'current9_player_inputs'

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def write(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'protocol.json').exists()
    assert load(CASE/'verification.json')['status']=='passed'
    files=[OLD/n for n in ['snapshot_player_features.csv','prefix_player_summary_features.csv','rally_metadata.csv']]
    files += [CASE/n for n in ['snapshot_player_features.csv','prefix_player_summary_features.csv','case_result.json','verification.json','candidate_sequence_grid.jpg','far_pair_detail_grid.jpg']]
    files += [COHORT/n for n in ['protocol.json','result.json','fixed30_processing_queue.csv','basic_server_inputs_11.csv','human_outcomes_11.csv']]
    hashes={str(p):sha(p) for p in files}
    dump(OUT/'protocol.json',dict(date_jst='2026-11-10',purpose='Nine-case input construction after deterministic001 expansion; no classifier fitting or new accuracy claim.',
        preserve='Frozen current8 retained; first30 cohort membership and processing order unchanged.',
        primary_window_comparison='Server side plus eight within-window mean pair-image-configuration features and two pair-observation fractions. No outcome-based model/feature selection.',
        snapshot_comparison='Server side plus eight last-requested-time configuration features and two pair-valid flags.',
        mask_controls='For each comparison, same server metadata plus its own two availability fields only.',
        basic_metadata='Serving side is a provisional AI attribute, not human-confirmed. Missingness and offline future-assisted tracking remain limitations.',
        ball='No nine-case ball features synthesized; existing two assisted ball cases do not support a matched-class outcome comparison.',
        protected_input_hashes=hashes))
    snapshots=read(OLD/'snapshot_player_features.csv')+read(CASE/'snapshot_player_features.csv')
    summaries=read(OLD/'prefix_player_summary_features.csv')+read(CASE/'prefix_player_summary_features.csv')
    for rs in [snapshots,summaries]:rs.sort(key=lambda r:r['rally_candidate_id'])
    assert len(snapshots)==len(summaries)==9
    ids=[r['rally_candidate_id'] for r in snapshots];assert len(set(ids))==9 and ids==[r['rally_candidate_id'] for r in summaries]
    basic={r['rally_candidate_id']:r for r in read(COHORT/'basic_server_inputs_11.csv')}
    outcomes={r['rally_candidate_id']:r for r in read(COHORT/'human_outcomes_11.csv')}
    queue=read(COHORT/'fixed30_processing_queue.csv');groups={r['candidate_id']:r['point_group_id'] for r in read(COHORT/'all_candidate_group_registry.csv')}
    geometry=[side+'_'+f for side in ['near','far'] for f in ['center_x_norm','center_y_norm','x_gap_norm','y_gap_norm']]
    snapshot_mask=[side+'_pair_valid' for side in ['near','far']]
    window_mask=[side+'_pair_observed_fraction' for side in ['near','far']]
    outputs={'B1_server_side.csv':[], 'B2_snapshot.csv':[], 'B2_window_means.csv':[],
             'diagnostic_snapshot_masks_only.csv':[], 'diagnostic_window_masks_only.csv':[]}
    for snap,summ in zip(snapshots,summaries):
        cid=snap['rally_candidate_id'];base=basic[cid]
        outputs['B1_server_side.csv'].append(base)
        for name,source,keys in [('B2_snapshot.csv',snap,geometry+snapshot_mask),
            ('B2_window_means.csv',summ,[k+'_mean' for k in geometry]+window_mask),
            ('diagnostic_snapshot_masks_only.csv',snap,snapshot_mask),
            ('diagnostic_window_masks_only.csv',summ,window_mask)]:
            outputs[name].append(dict(base,**{k:source[k] for k in keys}))
    expected_allowed=set(geometry+[k+'_mean' for k in geometry]+snapshot_mask+window_mask+['rally_candidate_id','server_side_near'])
    for name,rows in outputs.items():
        assert [r['rally_candidate_id'] for r in rows]==ids
        assert set(rows[0])<=expected_allowed
        assert not any('winner' in k or 'source_frame' in k or 'point_group' in k or 'path' in k for k in rows[0])
        write(OUT/name,rows)
    write(OUT/'human_outcomes.csv',[outcomes[cid] for cid in ids])
    write(OUT/'source_snapshot_features.csv',snapshots);write(OUT/'source_window_summary_features.csv',summaries)
    metadata=[]
    old={r['rally_candidate_id']:r for r in read(OLD/'rally_metadata.csv')}
    for cid in ids:
        isnew=cid.endswith('_001');r=old.get(cid)
        metadata.append(dict(rally_candidate_id=cid,point_group_id=groups[cid],
            in_fixed30=int(any(q['candidate_id']==cid for q in queue)),
            input_first_source_frame=117 if isnew else int(r['input_first_source_frame']),
            input_end_source_frame=177 if isnew else int(r['input_end_source_frame']),
            four_observation_times=31 if isnew else int(r['four_observed_times']),requested_times=31,
            phase_evidence='AI_fixed_image_sequence_supports_play;exact_timing_unknown' if isnew else 'see_prior_phase_evidence_table',
            independent_identity_position_accuracy_established=0,
            tracking_source=str(CASE/'four_player_path.csv') if isnew else r['player_source_parts_json']))
    write(OUT/'rally_metadata.csv',metadata)
    progress=[]
    for q in queue:
        cid=q['candidate_id'];progress.append(dict(candidate_id=cid,point_group_id=q['point_group_id'],
            cohort_group_rank=q['chronological_group_rank'],fixed_window_features_constructed=int(cid in ids),
            status='new_001_candidate_constructed_AI_image_check' if cid.endswith('_001') else 'prior_fixed_window_reused' if cid in ids else 'pending',
            independent_tracking_accuracy_established=0))
    write(OUT/'fixed30_processing_progress.csv',progress)
    assert sum(r['fixed_window_features_constructed'] for r in progress)==6
    assert all(sha(Path(p))==h for p,h in hashes.items())
    verification=dict(status='passed',unique_candidate_rows=9,point_groups=len({groups[cid] for cid in ids}),
        candidate_order_identical_across_all_comparisons=True,same_source_values_preserved=True,
        input_columns_allowlisted=True,no_outcome_or_absolute_time_columns_in_inputs=True,
        previous_outputs_and_frozen_cohort_unchanged=True,cohort_constructed_candidate_inputs=6,
        source_tracking_accuracy_not_independently_verified=True)
    dump(OUT/'verification.json',verification)
    dump(OUT/'feature_schema.json',dict(inputs={name:[k for k in rows[0] if k!='rally_candidate_id'] for name,rows in outputs.items()},
        join_key_not_model_input='rally_candidate_id',labels_file='human_outcomes.csv',
        metadata_not_model_inputs=['rally_metadata.csv','fixed30_processing_progress.csv'],
        empty_coordinates='Missing; preserve as blank, never zero; fit any future imputation only on training groups'))
    dump(OUT/'result.json',dict(status='nine_candidate_comparison_inputs_constructed',candidates=9,
        outcome_counts=dict(Counter(outcomes[cid]['winner_camera_side'] for cid in ids)),
        fixed30_constructed_inputs=6,fixed30_pending_inputs=24,supplementary_development_candidates=3,
        B1_inputs=1,B2_snapshot_inputs=11,B2_window_inputs=11,
        model_fitting_performed=False,performance_estimated=False,ball_comparison_ready=False,
        new_human_annotation_requested=False,canonical_tracking_changed=False,
        hashes={p.name:sha(p) for p in OUT.glob('*.csv')}))
    print(json.dumps(load(OUT/'result.json'),ensure_ascii=False))

if __name__=='__main__':main()
