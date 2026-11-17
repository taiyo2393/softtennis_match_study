"""Seal assisted-review labels and timing; never call acceptance independent accuracy."""
import collections,datetime,json,math,shutil
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,read,load,write,sha

OUT=ROOT.parent/'ball_quality_20261109/rally_repair_pilot_044_v2'
DONE=OUT/'completed_review'


def stamp(value):return datetime.datetime.fromisoformat(value)
def dump(path,value):path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def main():
    assert not DONE.exists(),'Preserve completed review. Corrections require a separate revision.'
    manifest=load(OUT/'manifest.json')
    assert all(sha(Path(p))==h for p,h in manifest['protected_input_hashes'].items())
    assert all(sha(OUT/n)==h for n,h in manifest['frozen_artifact_hashes'].items())
    assert load(OUT/'http_verification.json')['status']=='passed'
    paths=[OUT/n for n in ['review_labels.csv','initial_labels_snapshot.csv','review_events.jsonl','timing_events.jsonl','cases.json',
        'baseline_predictions.csv','protocol.json','manifest.json','initial_readiness.json','http_verification.json','app_verification.json','source_image_hashes.json','serve_review.py','review.html']]
    before={str(p):sha(p) for p in paths}
    initial,labels=read(OUT/'initial_labels_snapshot.csv'),read(OUT/'review_labels.csv')
    cases={c['case_id']:c for c in load(OUT/'cases.json')}
    assert len(labels)==len(initial)==len(cases)==201
    state={r['case_id']:r for r in initial}
    events=[json.loads(s) for s in (OUT/'review_events.jsonl').read_text(encoding='utf-8').splitlines()]
    allowed={'review_status','decision','x_px','y_px','coordinate_source','independent_new_center_gt','reviewer_id','reviewed_at_utc','notes'}
    for e in events:
        cid=e['case_id'];assert cid in state and state[cid]==e['before'] and e['predictions_visible'] is True
        assert set(e['after'])==set(e['before'])
        assert all(e['after'][k]==e['before'][k] for k in e['before'] if k not in allowed)
        assert not cases[cid]['reused_manual_center']
        assert e['after']['reviewer_id']=='human_user' and e['after']['independent_new_center_gt']=='0'
        stamp(e['after']['reviewed_at_utc']);state[cid]=e['after']
    assert [state[r['case_id']] for r in initial]==labels
    assert all(r['review_status']!='unreviewed' for r in labels)
    changed={e['case_id'] for e in events}
    assert changed=={r['case_id'] for r in initial if r['review_status']=='unreviewed'} and len(changed)==195
    for r in labels:
        c=cases[r['case_id']]
        assert str(c['local_frame'])==r['local_frame'] and str(c['source_frame'])==r['source_frame']
        if c['reused_manual_center']:
            assert r==next(s for s in initial if s['case_id']==r['case_id'])
            assert sha(Path(r['prior_center_source_file']))==r['prior_center_source_sha256']
        elif r['decision']=='accept_prediction':
            assert c['prediction_visible'] and (float(r['x_px']),float(r['y_px']))==(c['prediction_x_px'],c['prediction_y_px'])
            assert r['coordinate_source']=='prediction_explicitly_accepted_by_human'
        elif r['decision']=='correct_center':
            x,y=float(r['x_px']),float(r['y_px']);assert math.isfinite(x) and math.isfinite(y) and 0<=x<1920 and 0<=y<1080
            assert r['coordinate_source']=='human_click_with_prediction_visible'
        else:
            assert r['decision'] in {'not_visible','uncertain','nonplay'} and r['x_px']==r['y_px']==r['coordinate_source']==''
    times=[json.loads(s) for s in (OUT/'timing_events.jsonl').read_text(encoding='utf-8').splitlines()]
    active=False;client=None;previous_utc=None;segments=[];start=None;segment_ms=0;save_times=[]
    for i,e in enumerate(times):
        assert e['kind'] in {'start','tick','save','pause','expired'}
        assert isinstance(e['active_ms'],int) and isinstance(e['gap_ms'],int) and 0<=e['active_ms']<=15000 and e['gap_ms']>=0
        now=stamp(e['server_utc'])
        assert previous_utc is None or now>=previous_utc
        if e['kind']=='start':
            assert not active and e['active_ms']==e['gap_ms']==0
            active=True;client=e['client_id'];start=e['server_utc'];segment_ms=0
        else:
            assert active and client==e['client_id']
            segment_ms+=e['active_ms']
            if e['kind']=='save':save_times.append(e)
            if e['kind'] in {'pause','expired'} or e['gap_ms']>0:
                segments.append(dict(start_utc=start,end_utc=e['server_utc'],captured_ms=segment_ms,wall_ms=round((now-stamp(start)).total_seconds()*1000),close_kind=e['kind'],reason=e['reason']))
                active=False
        previous_utc=now
    assert not active,'Timing session remains open; no final timing claim yet.'
    assert len(save_times)==len(events)
    for e,t in zip(events,save_times,strict=True):
        # Validation timestamps and capture timestamps use the same local server clock.
        assert abs((stamp(e['after']['reviewed_at_utc'])-stamp(t['server_utc'])).total_seconds())<1
    assert segments and times[-1]['kind']=='pause' and times[-1]['reason']=='all_targets_completed'
    active_ms=sum(e['active_ms'] for e in times);gap_ms=sum(e['gap_ms'] for e in times)
    assert active_ms==sum(s['captured_ms'] for s in segments)
    for s in segments:
        assert abs(s['wall_ms']-s['captured_ms'])<=gap_ms+len(times)+1000
    decisions=collections.Counter(r['decision'] for r in labels)
    cross=collections.Counter((cases[r['case_id']]['prediction_visible'],r['decision']) for r in labels)
    initial_outputs=sum(c['prediction_visible'] for c in cases.values())
    exported=[];states=[];correction_details=[]
    for r in labels:
        c=cases[r['case_id']];present=r['decision'] in {'existing_manual_center','accept_prediction','correct_center'}
        common=dict(case_id=r['case_id'],local_frame=r['local_frame'],source_frame=r['source_frame'],clock='legacy_edited_match1',
            decision=r['decision'],review_status=r['review_status'],coordinate_source=r['coordinate_source'],
            x_px=r['x_px'],y_px=r['y_px'],observation_present=int(present),independent_new_center_gt=0,
            prediction_was_visible_to_reviewer=int(not c['reused_manual_center']),no_interpolation=1,
            annotation_source_file=str(DONE/'review_labels.csv'),annotation_source_sha256=before[str(OUT/'review_labels.csv')])
        states.append(dict(common,baseline_output_present=c['prediction_visible'],baseline_x_px=c['prediction_x_px'],baseline_y_px=c['prediction_y_px']))
        if present:exported.append(common)
        if r['decision']=='correct_center':
            distance=math.hypot(float(r['x_px'])-c['prediction_x_px'],float(r['y_px'])-c['prediction_y_px']) if c['prediction_visible'] else ''
            correction_details.append(dict(case_id=r['case_id'],local_frame=r['local_frame'],source_frame=r['source_frame'],
                kind='replace_existing_output' if c['prediction_visible'] else 'fill_missing_output',displacement_from_input_px=distance,
                x_px=r['x_px'],y_px=r['y_px'],interpretation='prediction_assisted_manual_edit_not_blind_accuracy_test'))
    added=cross[0,'correct_center'];removed=cross[1,'not_visible']+cross[1,'uncertain']+cross[1,'nonplay']
    assert len(exported)==initial_outputs+added-removed
    # Report sampled unavailable runs without inventing intermediate-frame truth or positions.
    runs=[];run=[]
    for r in states:
        if not r['observation_present']:run.append(r)
        elif run:
            runs.append(run);run=[]
    if run:runs.append(run)
    longest=max(runs,key=len) if runs else []
    assert all(sha(Path(p))==h for p,h in before.items()),'Input changed during sealing.'
    DONE.mkdir()
    for p in paths:shutil.copy2(p,DONE/p.name)
    shutil.copy2(Path(__file__),DONE/'executed_finalize_source.py')
    write(DONE/'assisted_positions_sampled.csv',exported);write(DONE/'assisted_states_sampled.csv',states)
    write(DONE/'manual_edit_details.csv',correction_details);write(DONE/'timing_sessions.csv',segments)
    result=dict(status='assisted_review_complete_and_sealed',completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        rally_candidate_id='match1_rally_window_044',sampled_times=201,newly_reviewed_times=195,reused_manual_centers=decisions['existing_manual_center'],
        accepted_predictions=decisions['accept_prediction'],new_clicked_centers=decisions['correct_center'],
        not_visible=decisions['not_visible'],uncertain=decisions['uncertain'],nonplay=decisions['nonplay'],
        journal_events=len(events),unique_journal_targets=len(changed),resaves=len(events)-len(changed),
        initial_sampled_output_count=initial_outputs,manual_filled_missing=added,manual_replaced_existing=cross[1,'correct_center'],
        existing_outputs_marked_unobservable=cross[1,'not_visible'],final_sampled_position_count=len(exported),remaining_sampled_unavailable=201-len(exported),
        visible_input_prediction_disposition=dict(accepted=cross[1,'accept_prediction'],prior_manual_center=cross[1,'existing_manual_center'],
            manually_replaced=cross[1,'correct_center'],not_observable=cross[1,'not_visible'],uncertain=cross[1,'uncertain'],nonplay=cross[1,'nonplay']),
        missing_input_disposition=dict(manually_added=cross[0,'correct_center'],not_observable=cross[0,'not_visible'],uncertain=cross[0,'uncertain'],nonplay=cross[0,'nonplay']),
        captured_active_ms=active_ms,captured_active_seconds=active_ms/1000,unmeasured_gap_ms=gap_ms,timing_sessions=len(segments),
        first_start_utc=segments[0]['start_utc'],final_pause_utc=segments[-1]['end_utc'],wall_span_seconds=(stamp(segments[-1]['end_utc'])-stamp(segments[0]['start_utc'])).total_seconds(),
        candidate_scope_source_frames=402,candidate_scope_seconds=402/(60000/1001),sampled_stride=2,
        longest_sampled_unavailable_run=dict(samples=len(longest),first_local_frame=longest[0]['local_frame'] if longest else '',last_local_frame=longest[-1]['local_frame'] if longest else ''),
        time_scope='Remaining review/edit work after historical corrections;instructions/setup excluded;not total raw-automation-to-final cost.',
        no_from_scratch_timing_comparator=True,independent_detector_accuracy_computed=False,assisted_positions_are_not_independent_gt=True,
        physical_ball_absence_labels_created=False,positions_interpolated=False,unsampled_frames_reviewed=False,exact_rally_boundaries_confirmed=False,
        new_training=False,canonical_tracking_changed=False,independent_external_validation=False,
        limitation='One previously inspected,partly repaired same-match candidate interior;not representative labor or accuracy benchmark.',
        source_snapshot_hashes={Path(p).name:h for p,h in before.items()},
        export_hashes={n:sha(DONE/n) for n in ['assisted_positions_sampled.csv','assisted_states_sampled.csv','manual_edit_details.csv','timing_sessions.csv']},
        script_sha256=sha(Path(__file__)))
    dump(DONE/'result.json',result)
    assert all(sha(DONE/Path(p).name)==h and sha(Path(p))==h for p,h in before.items())
    assert all(sha(Path(p))==h for p,h in manifest['protected_input_hashes'].items())
    assert len(read(DONE/'assisted_positions_sampled.csv'))==len(exported) and len(read(DONE/'assisted_states_sampled.csv'))==201
    dump(DONE/'verification.json',dict(status='passed',history_events_replayed=len(events),new_targets_verified=len(changed),reused_centers_unchanged=6,
        timing_events_verified=len(times),timing_sessions_closed=True,label_and_timing_save_event_order_consistent=True,
        source_clock_and_positions_checked=True,snapshot_hashes_exact=True,protected_inputs_unchanged=True,independent_gt_not_claimed=True,
        script_sha256=sha(Path(__file__))))
    print(json.dumps({k:v for k,v in result.items() if k not in {'source_snapshot_hashes','export_hashes'}},indent=2,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
