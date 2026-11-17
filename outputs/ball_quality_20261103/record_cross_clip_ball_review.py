"""Record008 A/C real, B/D false; score untouched pre-answer predictions."""
import json
from pathlib import Path
from collections import Counter,defaultdict
from integrate_044_reviewed_tracks import ROOT,load,read,write,sha

OUT=ROOT/'ball_candidate_cross_rally'
ANSWERS={'A':True,'B':False,'C':True,'D':False}

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    dest=OUT/'user_reviewed';dest.mkdir(exist_ok=True);assert not (dest/'review.json').exists()
    cases=read(OUT/'cases.csv');pending=[r for r in cases if r['role']=='new_cross_clip_candidate']
    assert {r['display_label'] for r in pending}==set(ANSWERS)
    render=load(OUT/'review_video_report.json');video=OUT/'four_candidate_review_h264.mp4'
    assert sha(video)==render['video_sha256'] and sha(OUT/'cases.csv')==render['candidate_table_sha256']
    probe=OUT/'appearance_probe_v2';fixed=load(probe/'report.json')
    pp=probe/'frozen_predictions.csv';model=probe/'model.npz'
    assert sha(pp)==fixed['frozen_predictions_sha256'] and sha(model)==fixed['model_sha256']
    manifest=load(OUT/'case_manifest.json');assert all(sha(Path(p))==h for p,h in manifest['input_hashes'].items())
    before_registry=load(ROOT/'reviewed_ball_overrides.json');old044=before_registry['clips']['match1_motion_pilot_016']
    immutable={str(pp):sha(pp),str(model):sha(model),str(OUT/'cases.csv'):sha(OUT/'cases.csv'),
        str(Path(old044['reviewed_output'])):old044['reviewed_output_sha256'],
        str(ROOT/'reviewed_person_overrides.json'):sha(ROOT/'reviewed_person_overrides.json'),
        str(ROOT/'reviewed_tracking_bundles.json'):sha(ROOT/'reviewed_tracking_bundles.json')}
    cid='match1_motion_pilot_008';source=ROOT.parent/'tracking_quality_20261102/ball_aligned'/cid/'balltrack_predictions.csv'
    immutable[str(source)]=sha(source);baseline=read(source)
    review_id='20261103_008_four_candidates_AC_real_BD_false'
    review=dict(review_id=review_id,recorded_date='2026-11-03',reviewer='human_user',
        answer='ACは実球，他は誤検出',ball_pilot_id=cid,rally_candidate_id='match1_rally_window_026',
        scope='Exactly the four displayed blue raw detector candidates; no neighbouring-frame or whole-track confirmation.',
        confirmed_ball_candidate_local_frames=[391,615],confirmed_false_raw_local_frames=[503,1623],
        candidates=[dict(display_label=r['display_label'],local_frame=int(r['local_frame']),source_frame=int(r['source_frame']),
            xy_px=[float(r['x_px']),float(r['y_px'])],candidate_is_ball=ANSWERS[r['display_label']],case_id=r['case_id']) for r in pending],
        confirmed_observations=[dict(local_frame=int(r['local_frame']),source_frame=int(r['source_frame']),
            xy_px=[float(r['x_px']),float(r['y_px'])]) for r in pending if ANSWERS[r['display_label']]],
        rejected_raw_observations=[dict(local_frame=int(r['local_frame']),source_frame=int(r['source_frame']),
            rejected_xy_px=[float(r['x_px']),float(r['y_px'])]) for r in pending if not ANSWERS[r['display_label']]],
        precise_center_groundtruth_created=False,physical_ball_visibility_at_rejected_frames='not_established',
        exact_rally_boundaries_confirmed=False,independent_accuracy_evaluation=False,
        video_sha256=sha(video),candidate_table_sha256=sha(OUT/'cases.csv'),
        frozen_model_sha256=sha(model),frozen_predictions_sha256=sha(pp),model_refitted_after_answer=False)
    dump(dest/'review.json',review)
    reviewed=[]
    for r in cases:
        q=dict(r)
        if r['role']=='new_cross_clip_candidate':
            q['label']='confirmed_ball_enclosed' if ANSWERS[r['display_label']] else 'confirmed_not_ball_enclosed'
            q['evidence_record']=str((dest/'review.json').resolve())
        reviewed.append(q)
    write(dest/'cases.csv',reviewed)
    predictions={r['case_id']:r for r in read(pp)};outcomes=[]
    for c in pending:
        r=predictions[c['case_id']];assert r['label_at_prediction']=='pending_user_confirmation'
        outcomes.append(dict(case_id=c['case_id'],display_label=c['display_label'],local_frame=c['local_frame'],
            candidate_is_ball=int(ANSWERS[c['display_label']]),appearance_keep=int(r['appearance_keep']),
            no_added_filter_keep=int(r['no_added_filter_keep']),uncalibrated_ball_score=r['uncalibrated_ball_score'],
            frozen_model_sha256=sha(model)))
    write(dest/'frozen_prediction_outcomes.csv',outcomes)
    real=[r for r in outcomes if r['candidate_is_ball']];false=[r for r in outcomes if not r['candidate_is_ball']]
    comparison=dict(new_reviewed_candidates=4,real_candidates=2,false_candidates=2,
        appearance_real_retained=sum(r['appearance_keep'] for r in real),appearance_false_retained=sum(r['appearance_keep'] for r in false),
        no_added_filter_real_retained=sum(r['no_added_filter_keep'] for r in real),no_added_filter_false_retained=sum(r['no_added_filter_keep'] for r in false),
        prior_safety_real_retained=fixed['prior_real_candidates_retained'],prior_safety_real_total=fixed['prior_real_total'],
        prior_safety_false_retained=fixed['prior_false_candidates_retained'],prior_safety_false_total=fixed['prior_false_total'],
        new_candidates_not_used_for_training_or_threshold_selection=True,model_and_predictions_unchanged=True,
        conclusion='No false candidates removed on008; previous real-ball loss remains. Keep the appearance trial rejected.',
        scope='Selected four same-match candidates, not representative accuracy or full tracking validation')
    dump(dest/'comparison.json',comparison)
    # Sparse per-clip correction: raw history retained, only explicit candidates changed.
    answers_by_frame={int(r['local_frame']):ANSWERS[r['display_label']] for r in pending};result=[]
    for t,r in enumerate(baseline):
        q=dict(r,raw_x_px=r['x_px'],raw_y_px=r['y_px'],raw_confidence=r['confidence'],raw_position_source=r['position_source'],
            ball_candidate_review='not_reviewed_by_this_confirmation',human_review_id='',precise_center_reviewed_here='0',interpolated_here='0')
        if t in answers_by_frame:
            assert int(r['visible'])==1
            q['human_review_id']=review_id
            if answers_by_frame[t]:
                q.update(position_source='model_observed_user_confirmed_ball_candidate',ball_candidate_review='user_confirmed_is_ball')
            else:
                q.update(x_px='0',y_px='0',visible='0',confidence='',
                    position_source='no_ball_observation_user_rejected_raw_candidate',ball_candidate_review='user_confirmed_raw_candidate_is_not_ball')
        result.append(q)
    target=dest/'predictions'/cid/'balltrack_predictions.csv';target.parent.mkdir(parents=True,exist_ok=True);write(target,result)
    saved=read(target)
    original_column_changes=[t for t,(a,b) in enumerate(zip(baseline,saved,strict=True)) if any(a[k]!=b[k] for k in a)]
    coordinate_changes=[t for t,(a,b) in enumerate(zip(baseline,saved,strict=True)) if any(a[k]!=b[k] for k in ['x_px','y_px','visible'])]
    assert len(saved)==len(baseline)==1679 and original_column_changes==[391,503,615,1623] and coordinate_changes==[503,1623]
    for t in (391,615):assert all(saved[t][k]==baseline[t][k] for k in ['x_px','y_px','visible','confidence'])
    for t in (503,1623):assert saved[t]['visible']=='0'
    assert all(b['raw_x_px']==a['x_px'] and b['raw_y_px']==a['y_px'] for a,b in zip(baseline,saved,strict=True))
    # A unified, provenance-preserving candidate index. All four source clips stay grouped.
    corpus=[]
    for r in read(ROOT/'ball_044_reviewed_appearance_cases/cases.csv'):
        corpus.append(dict(case_id='match1_motion_pilot_016:'+r['case_id'],ball_pilot_id='match1_motion_pilot_016',
            local_frame=r['local_frame'],source_frame=r['source_frame'],label=r['label'],x_px=r['x_px'],y_px=r['y_px'],
            split_group='legacy_match1:match1_motion_pilot_016',inference_view=r['inference_view'],
            review_record=r['review_record'],original_image=r['original_image'],original_image_sha256=r['original_image_sha256'],
            patch96_path=r['patch96_path'],patch96_sha256=r['patch96_sha256'],context224_path=r['context224_path'],context224_sha256=r['context224_sha256']))
    for r in reviewed:
        corpus.append(dict(case_id=r['ball_pilot_id']+':'+r['case_id'],ball_pilot_id=r['ball_pilot_id'],
            local_frame=r['local_frame'],source_frame=r['source_frame'],label=r['label'],x_px=r['x_px'],y_px=r['y_px'],
            split_group=r['split_group'],inference_view=r['inference_view'],review_record=r['evidence_record'],
            original_image=r['original_image'],original_image_sha256=r['original_image_sha256'],patch96_path=r['patch96_path'],
            patch96_sha256=r['patch96_sha256'],context224_path=r['context224_path'],context224_sha256=r['context224_sha256']))
    assert len(corpus)==len({r['case_id'] for r in corpus})==36
    assert Counter(r['label'] for r in corpus)=={'confirmed_ball_enclosed':21,'confirmed_not_ball_enclosed':15}
    for r in corpus:
        assert sha(Path(r['patch96_path']))==r['patch96_sha256'] and sha(Path(r['context224_path']))==r['context224_sha256']
    write(dest/'all_reviewed_candidate_cases.csv',corpus)
    groups=defaultdict(Counter)
    for r in corpus:groups[r['split_group']][r['label']]+=1
    dump(dest/'corpus_manifest.json',dict(total_cases=36,real=21,false=15,source_clips=4,match_count=1,
        groups={g:dict(v) for g,v in groups.items()},all_cases_csv_sha256=sha(dest/'all_reviewed_candidate_cases.csv'),
        data_origin='Explicit existing user reviews only; no missing/unreviewed negative labels',
        caution='Clip IDs are conservative split groups, not confirmed one-rally boundaries. 002 is preparation;018 includes transitions. Adjacent frames are correlated; no random row split.'))
    # Check the protected histories before publishing the new008 pointer.
    assert all(sha(Path(p))==h for p,h in immutable.items())
    for p,h in load(ROOT/'play_start_audit/inventory_summary.json')['source_hashes'].items():assert sha(Path(p))==h
    dump(dest/'previous_reviewed_ball_registry.json',before_registry)
    assert cid not in before_registry['clips']
    before_registry['clips'][cid]=dict(rally_candidate_id='match1_rally_window_026',reviewed_output=str(target.resolve()),
        reviewed_output_sha256=sha(target),original_baseline=str(source.resolve()),
        reviewed_changed_local_frames=[391,503,615,1623],confirmed_ball_candidate_local_frames=[391,615],
        confirmed_false_raw_local_frames=[503,1623],review_id=review_id,review_record=str((dest/'review.json').resolve()),
        review_records=[str((dest/'review.json').resolve())],automatic_filter_adopted=False,
        precise_center_or_full_track_accuracy_established=False,physical_ball_visibility_at_rejected_frames='not_established')
    assert before_registry['clips']['match1_motion_pilot_016']==old044
    dump(ROOT/'reviewed_ball_overrides.json',before_registry)
    decision=load(probe/'decision.json');decision.update(status='rejected_after_prior_cases_and_new008_confirmation',
        new_008_review_status='AC_real_BD_false_confirmed',comparison=str((dest/'comparison.json').resolve()),
        review_record=str((dest/'review.json').resolve()),new_008_result='Real2/2 retained; false2/2 retained. No improvement over no additional filter.',
        combined_safety_and008_result='Real2/3 retained; false5/5 retained. No model or threshold updated.')
    dump(probe/'decision.json',decision)
    verification=dict(recorded_answer='AC real,BD false',new_confirmations=4,source_clip_rows=1679,
        coordinate_presence_changes=coordinate_changes,all_other_original_columns_unchanged_outside_four_candidates=True,
        positive_coordinates_and_raw_history_preserved=True,frozen_model_and_predictions_unchanged=True,
        original_case_table_with_pending_labels_preserved=True,person_inputs_and_044_bundle_unchanged=True,
        source_rally_annotations_unchanged=True,corpus_real_cases=21,corpus_false_cases=15,
        new008_reviewed_output_sha256=sha(target),independent_accuracy_established=False)
    dump(dest/'verification.json',verification);print(json.dumps(dict(comparison=comparison,verification=verification),indent=2))

if __name__=='__main__':main()
