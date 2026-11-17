"""Fix a chronological feasibility cohort and reuse pre-outcome serving-side metadata."""
import csv,hashlib,json,shutil
from collections import Counter,defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'ball_quality_20261110'
OUT=BASE/'cohort_and_baselines'
HANDOFF=ROOT.parent/'frontcourt_handoff_20260927'
OLD=ROOT.parent/'ball_quality_20261109/rally_player_prefix_expansion'

def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def write(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def ordered_groups(records,group_map):
    # Selection reads only timestamps, IDs and the pre-existing point-group map.
    grouped=defaultdict(list)
    for r in records:grouped[group_map[r['candidate_id']]['point_group_id']].append(r)
    return sorted(grouped.items(),key=lambda x:(min(float(r['candidate_start_sec']) for r in x[1]),min(r['candidate_id'] for r in x[1])))

def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'protocol.json').exists()
    paths=[HANDOFF/n for n in ['rally_label_review.csv','human_review_log.csv','candidate_point_group_map.csv',
        'exploratory_position_outcome_features.csv','compute_exploratory_position_outcome.py','score_suggestion_validation_sample.csv']]
    paths += [OLD/n for n in ['rally_metadata.csv','human_outcomes.csv','prefix_player_summary_features.csv','snapshot_player_features.csv']]
    paths += [BASE/'prefix_phase_audit/eight_candidate_phase_evidence.csv']
    protected={str(p):sha(p) for p in paths}
    records=read(paths[0]);group_map={r['candidate_id']:r for r in read(HANDOFF/'candidate_point_group_map.csv')}
    ordered=ordered_groups(records,group_map);chosen=[g for g,_ in ordered[:30]]
    assert len(records)==63 and len(ordered)==62 and len(set(chosen))==30
    assert group_map['match1_rally_window_050']['point_group_id']==group_map['match1_rally_window_051']['point_group_id']
    # A meaningful bias guard: changing outcome, visibility and legacy quality statuses cannot alter selection/order.
    perturbed=[dict(r,winner_camera_side='far' if i%2 else 'near',analysis_eligible='true' if i%2 else 'false',
        legacy_window_status='changed',position_count=str(i%5),tracking_available=str(i%2)) for i,r in enumerate(reversed(records))]
    assert [g for g,_ in ordered_groups(perturbed,group_map)]==[g for g,_ in ordered]
    # The large registry also exercises the known050/051 merge, even outside the first30.
    merged=[rs for _,rs in ordered if len(rs)>1];assert len(merged)==1 and {r['candidate_id'][-3:] for r in merged[0]}=={'050','051'}
    dump(OUT/'protocol.json',dict(date_jst='2026-11-10',selection_fixed_before_new_tracking=True,
        sampling='First30 existing candidate-point groups in source timestamp order; include every candidate status, outcome and observed/missing pattern. This is a chronological one-match feasibility cohort, not30 verified independent rallies.',
        already_exposed='Existing footage, labels and prior development have been viewed. Do not call this an untouched holdout. Current044/048/056 are supplementary development examples outside the first30.',
        no_replacement='No replacement for occlusion, double fault, missing serve, edit, ambiguous outcome, invalid interval or tracking failure. Record each reason and stage separately. New group discoveries must be versioned and propagated before fitting.',
        existing_score_validation='Retain the previous frozen16-case score-suggestion validation sample unchanged; this technical cohort does not replace it or certify its labels.',
        primary_construction='Reuse the fixed contact-last+30 through+90 window; legacy point contacts retain uncertainty flags. No quality-based time shifts or last-observed-frame snapshot.',
        basic_inputs='Server camera side is a provisional AI attribute. Current metadata permit this one basic covariate. Do not import old winner-relative features, winner team IDs, score differences, final score or post-point reactions.',
        comparisons=['B0 training-fold majority/probability only','B1 B0 plus pre-outcome serving side',
            'B2 same base plus early-window player arrangement','B3 B2 plus ball data on exactly the same rally groups/time window',
            'Diagnostic: base plus missingness masks only, to test whether observation patterns explain apparent gains'],
        additional_pre_outcome_conditions='Serving court half, first/second attempt, pre-point score and pre-point team mapping are desired but incomplete; leave unknown and record provenance. Winner_team_id is never a source for input team identity. Stale score overlays and game resets require separate checks.',
        role='Technical-quality and data-feasibility expansion for the existing research question; not a final thesis-topic decision or a prespecified confirmatory study.',
        evaluation='Do not fit a performance model to the current8 cases. First30 is one-match development/feasibility. Later fix additional matches before inspecting outcomes/tracking success, split by match and preserve point groups. Fit imputation/scaling/model selection on training folds only. Number of points/matches must be justified by observed variability, not declared sufficient by a fixed30 or50 rule.',
        scope='No causal loss explanation or tactical recommendation. Offline tracking may use future frames; selecting pre-score coordinates does not establish a causal online pipeline.',
        ball='Existing two assisted ball cases remain supplementary development; same-outcome data do not establish added predictive value or independent detector precision.',
        protected_input_hashes=protected,script_sha256=sha(Path(__file__))))
    shutil.copy2(Path(__file__),OUT/'executed_source.py')
    # Enrich only AFTER the timestamp/group selection is fixed.
    exports={r['rally_candidate_id']:r for r in read(OLD/'rally_metadata.csv')}
    frozen_score_sample={r['candidate_id'] for r in read(HANDOFF/'score_suggestion_validation_sample.csv')}
    registry=[];queue=[]
    for index,(group,rs) in enumerate(ordered,1):
        for r in sorted(rs,key=lambda r:r['candidate_id']):
            cid=r['candidate_id'];ready=cid in exports;in_cohort=group in chosen
            row=dict(chronological_group_rank=index,point_group_id=group,candidate_id=cid,
                group_status=group_map[cid]['group_status'],candidate_start_sec=r['candidate_start_sec'],candidate_end_sec=r['candidate_end_sec'],
                in_fixed30_feasibility_cohort=int(in_cohort),legacy_window_status=r['legacy_window_status'],
                existing_human_winner_side=int(r['winner_camera_side_review_status']=='human_confirmed'),
                existing_fixed_window_player_features=int(ready),existing_frozen_score_validation_member=int(cid in frozen_score_sample),
                contact_record_status=r['serve_review_status'],new_processing_status='reuse_existing_fixed_window' if ready else 'not_yet_processed',
                exclusion_or_replacement_permitted=0)
            registry.append(row)
            if in_cohort:queue.append(row)
    assert len(queue)==30 and [r['candidate_id'][-3:] for r in queue]==[f'{i:03d}' for i in range(1,31)]
    write(OUT/'all_candidate_group_registry.csv',registry);write(OUT/'fixed30_processing_queue.csv',queue)
    legacy={r['candidate_id']:r for r in read(HANDOFF/'exploratory_position_outcome_features.csv')}
    current_labels={r['candidate_id']:r for r in records if r['winner_camera_side_review_status']=='human_confirmed'}
    latest={}
    for r in read(HANDOFF/'human_review_log.csv'):
        if r['field']=='winner_camera_side':latest[r['candidate_id']]=r
    basic=[];provenance=[];labels=[];composition=[]
    for cid,r in sorted(current_labels.items()):
        tag=cid[-3:];old=legacy[tag];side=old['server_side_candidate'];assert side in {'near','far'}
        assert latest[cid]['value']==r['winner_camera_side']
        # Keep the provisional server indicator separate from evidence and from outcomes.
        basic.append(dict(rally_candidate_id=cid,server_side_near=int(side=='near')))
        provenance.append(dict(rally_candidate_id=cid,server_camera_side_candidate=side,server_status=old['server_review'],
            source=str(HANDOFF/'exploratory_position_outcome_features.csv'),source_sha256=sha(HANDOFF/'exploratory_position_outcome_features.csv'),
            origin='Legacy explicit AI side map from compute_exploratory_position_outcome.py; not an independent human label',
            server_court_half='',serve_attempt_number='',pre_point_score_top='',pre_point_score_bottom='',
            input_team_mapping='',additional_conditions_status='not_imported_pending_pre_outcome_evidence',
            in_current8=int(cid in exports),in_fixed30=int(group_map[cid]['point_group_id'] in chosen)))
        labels.append(dict(rally_candidate_id=cid,winner_camera_side=r['winner_camera_side'],human_side_review_id=latest[cid]['review_id']))
        composition.append(dict(rally_candidate_id=cid,server_won_given_provisional_side=int(side==r['winner_camera_side']),
            old_snapshot_position_count=int(old['position_count']),in_current8=int(cid in exports)))
    write(OUT/'basic_server_inputs_11.csv',basic)
    write(OUT/'basic_server_inputs_current8.csv',[r for r in basic if r['rally_candidate_id'] in exports])
    write(OUT/'basic_covariate_provenance.csv',provenance);write(OUT/'human_outcomes_11.csv',labels)
    write(OUT/'descriptive_selection_composition.csv',composition)
    complete=[r for r in composition if r['old_snapshot_position_count']==4]
    partial=[r for r in composition if r['old_snapshot_position_count']!=4]
    assert len(complete)==9 and len(partial)==2 and {r['rally_candidate_id'][-3:] for r in partial}=={'005','006'}
    assert all(not r['server_won_given_provisional_side'] for r in partial)
    assert all(sha(Path(p))==h for p,h in protected.items())
    result=dict(status='feasibility_cohort_and_basic_covariates_frozen',candidate_rows=63,provisional_point_groups=62,
        chronological_cohort_groups=30,cohort_existing_human_side_labels=sum(r['existing_human_winner_side'] for r in queue),
        cohort_existing_player_feature_candidates=sum(r['existing_fixed_window_player_features'] for r in queue),
        cohort_not_yet_processed=25,next_candidate='match1_rally_window_001',
        preserved_score_validation_cases=len(frozen_score_sample),
        descriptive_server_wins_current8=sum(r['server_won_given_provisional_side'] for r in composition if r['in_current8']),
        descriptive_server_wins_all11=sum(r['server_won_given_provisional_side'] for r in composition),
        descriptive_server_wins_old_complete9=sum(r['server_won_given_provisional_side'] for r in complete),
        selection_guard_outcome_and_quality_perturbation='passed',group050_051_merge='preserved',
        model_fit=False,performance_estimated=False,new_human_annotation_requested=False,
        caution='Composition counts condition on provisional AI serving-side attributes and an old convenience sample; not serve-win rates for the match or classifier performance.',
        protected_inputs_unchanged=True,artifact_hashes={p.name:sha(p) for p in OUT.glob('*.csv')})
    dump(OUT/'result.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
