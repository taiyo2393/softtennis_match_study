"""Describe two assisted-review cases separately;no population accuracy or time-savings."""
import datetime,json
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,read,load,write,sha

BASE=ROOT.parent/'ball_quality_20261109'
OUT=BASE/'two_rally_repair_assessment'
CASES=[('044',BASE/'rally_repair_pilot_044_v2/completed_review','previously_repaired_input_with_6_existing_human_centers'),
       ('056',BASE/'rally_repair_056/completed_review','unpatched_original_model_output')]


def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')


def verify_case(tag,folder):
    result=load(folder/'result.json');verification=load(folder/'verification.json')
    assert verification['status']=='passed' and result['rally_candidate_id']==f'match1_rally_window_{tag}'
    assert all(sha(folder/n)==h for n,h in result['source_snapshot_hashes'].items())
    assert all(sha(folder/n)==h for n,h in result['export_hashes'].items())
    assert sha(folder/'executed_finalize_source.py')==result['script_sha256']
    assert all(sha(Path(p))==h for p,h in load(folder/'source_image_hashes.json').items())
    labels=read(folder/'review_labels.csv');states=read(folder/'assisted_states_sampled.csv');positions=read(folder/'assisted_positions_sampled.csv')
    cases={c['case_id']:c for c in load(folder/'cases.json')}
    assert len(labels)==len(states)==len(cases)==result['sampled_times']
    assert len(positions)==result['final_sampled_position_count'] and len({r['source_frame'] for r in positions})==len(positions)
    assert all(r['independent_new_center_gt']=='0' for r in positions)
    pairs=[('accept_prediction','accepted_predictions'),('correct_center','new_clicked_centers'),('not_visible','not_visible'),
           ('uncertain','uncertain'),('nonplay','nonplay'),('existing_manual_center','reused_manual_centers')]
    assert all(sum(r['decision']==decision for r in labels)==result[key] for decision,key in pairs)
    assert sum(int(r['observation_present']) for r in states)==len(positions)
    events=[json.loads(s) for s in (folder/'review_events.jsonl').read_text(encoding='utf-8').splitlines()]
    initial=read(folder/'initial_labels_snapshot.csv');state={r['case_id']:r for r in initial}
    for e in events:
        assert state[e['case_id']]==e['before'] and e['predictions_visible'] is True;state[e['case_id']]=e['after']
    assert [state[r['case_id']] for r in initial]==labels
    times=[json.loads(s) for s in (folder/'timing_events.jsonl').read_text(encoding='utf-8').splitlines()]
    assert sum(t['active_ms'] for t in times)==result['captured_active_ms']
    assert sum(t['gap_ms'] for t in times)==result['unmeasured_gap_ms']
    assert sum(t['kind']=='save' for t in times)==len(events)==result['journal_events']
    sessions=read(folder/'timing_sessions.csv');paused=0.
    for a,b in zip(sessions,sessions[1:]):
        paused+=(datetime.datetime.fromisoformat(b['start_utc'])-datetime.datetime.fromisoformat(a['end_utc'])).total_seconds()
    # Cross-classify observable input output versus the final assisted decisions.
    cross=[]
    for presence in [0,1]:
        for decision,_ in pairs:
            n=sum(cases[r['case_id']]['prediction_visible']==presence and r['decision']==decision for r in labels)
            if n:cross.append(dict(rally=tag,input_output_present=presence,decision=decision,times=n))
    summary=dict(rally=tag,candidate_scope_seconds=result['candidate_scope_seconds'],review_sample_times=result['sampled_times'],
        new_review_targets=result['newly_reviewed_times'],reused_centers=result['reused_manual_centers'],
        captured_active_seconds=result['captured_active_seconds'],unmeasured_gap_seconds=result['unmeasured_gap_ms']/1000,
        recorded_pause_wall_seconds=paused,initial_position_outputs=result['initial_sampled_output_count'],
        accepted_predictions=result['accepted_predictions'],new_clicked_centers=result['new_clicked_centers'],
        filled_missing=result['manual_filled_missing'],replaced_existing=result['manual_replaced_existing'],
        input_outputs_marked_unobservable=result['existing_outputs_marked_unobservable'],
        final_position_times=result['final_sampled_position_count'],unobservable_times=result['not_visible'],uncertain_times=result['uncertain'],nonplay_times=result['nonplay'],
        position_availability_only=result['final_sampled_position_count']/result['sampled_times'],new_independent_center_gt=0)
    check=dict(status='passed',label_history_replayed=len(events),timing_events_recomputed=len(times),source_images_verified=len(load(folder/'source_image_hashes.json')),
        exported_positions_reloaded=len(positions),snapshot_and_export_hashes_exact=True,assisted_groundtruth_not_claimed=True)
    return summary,cross,check


def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'assessment.json').exists(),'Keep the fixed two-case assessment.'
    summaries=[];cross=[];checks={};sources={}
    for tag,folder,input_status in CASES:
        row,table,check=verify_case(tag,folder);row['input_status']=input_status;summaries.append(row);cross+=table;checks[tag]=check
        sources[str(folder/'result.json')]=sha(folder/'result.json')
        sources[str(folder/'review_labels.csv')]=sha(folder/'review_labels.csv');sources[str(folder/'timing_events.jsonl')]=sha(folder/'timing_events.jsonl')
        if tag=='056':dump(folder/'export_consistency_check.json',check)
    write(OUT/'case_summary.csv',summaries);write(OUT/'input_output_dispositions.csv',cross)
    scenarios=[]
    # Arithmetic planning scenarios,not estimated population costs or measured productivity.
    seconds=summaries[1]['captured_active_seconds']
    for n in [30,50]:
        scenarios.append(dict(rallies=n,assumed_active_seconds_per_rally=seconds,active_hours=n*seconds/3600,
            assumption='Every rally has same operator burden as this056candidate interior;not validated. Excludes processing/setup,player review,score labels,and independent evaluation.'))
    dump(OUT/'planning_scenarios.json',scenarios)
    assessment=dict(status='two_case_feasibility_assessment_complete',cases=summaries,
        verification=checks,case_count=2,source_result_hashes=sources,
        conclusions=['Both scoped samples were processed with explicit accept/edit/unobservable decisions and closed time logs.',
            'A concrete semi-automatic preparation route exists for these cases;unobservable positions still remain.',
            'Unpatched056needed109new centers and683.001capturedseconds;ball preparation is not unattended automation.',
            '044and056have different prior repairs,durations,andcontent. Do not infer method superiority from the time difference.',
            'Do not compute independent detector accuracy from prediction-assisted acceptance or a time-savings rate without a from-scratch comparator.'],
        decision=dict(stop_additional_timed_review_requests_for_now=True,keep_original_detector_weights=True,
            prioritize='Define the rally-level analysis and minimum needed ball observations before expanding annotation;retain four-player positions and point outcome as the research core.',
            data_use='Keep explicit provenance and missingness. Assisted coordinates may support development/preparation but are not an untouched independent evaluation set.',
            no_new_model_training=True,no_canonical_tracking_update=True),
        limitations=['Known same-match cases,not random or external samples.','Candidate interiors and stride2sampling,not exact full-rally/all-frame labels.',
            'Operator times exclude historical044repairs,instructions,setup,and model processing.','Four-player identity/ground-point quality and causal loss explanations were not evaluated.'],
        export_hashes={n:sha(OUT/n) for n in ['case_summary.csv','input_output_dispositions.csv','planning_scenarios.json']},script_sha256=sha(Path(__file__)))
    dump(OUT/'assessment.json',assessment);dump(OUT/'verification.json',dict(status='passed',case_checks=checks,input_snapshots_unchanged=True,no_new_annotation_or_timing_created=True,script_sha256=sha(Path(__file__))))
    print(json.dumps(dict(status=assessment['status'],cases=summaries,decision=assessment['decision']),indent=2),flush=True)


if __name__=='__main__':main()
