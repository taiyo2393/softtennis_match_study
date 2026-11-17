"""Verify016 matched transfer and input expansion without GT promotion."""
import json,shutil
from pathlib import Path
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,sha,dump
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    c=BASE/'cohort_tracking/case_016';t=BASE/'court_initialization_transfer_016';old=BASE/'current19_player_inputs';out=BASE/'current20_player_inputs';anchor=BASE/'cohort_anchor_proposals/016';assert not (out/'additional_verification.json').exists()
    for folder in [c,out]:
        p=load(folder/'protocol.json');assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items());assert load(folder/'verification.json')['status']=='passed'
        r=load(folder/('case_result.json' if folder==c else 'result.json'));assert all(sha(folder/n)==h for n,h in r['artifact_hashes'].items())
    p=load(c/'protocol.json');tp=load(t/'protocol.json');assert all(sha(Path(n))==h for n,h in tp['protected_input_hashes'].items());frozen=load(Path(tp['frozen_plan']));assert frozen['fixed_before_016_inspection'] and not frozen['threshold_tuning_on_016'];assert all(sha(Path(n))==h for n,h in frozen['frozen_code_and_calibration_hashes'].items())
    proposal=load(Path(p['contact_proposal_file']));assert proposal['fixed_before_new_person_tracking'] and proposal['context_boundary_modified'] is False;assert all(sha(Path(n))==h for n,h in proposal['evidence_hashes'].items())
    times=p['requested_input_frames'];assert times==list(range(40669,40730,2));assert p['contact_candidate_first_source_frame']==40633 and p['contact_candidate_last_source_frame']==40639
    names=['B1_server_side.csv','B2_snapshot.csv','B2_window_means.csv','diagnostic_snapshot_masks_only.csv','diagnostic_window_masks_only.csv','human_outcomes.csv','source_snapshot_features.csv','source_window_summary_features.csv','rally_metadata.csv','processing_interventions.csv','outcome_label_status.csv']
    for name in names:
        a=read(old/name);b=read(out/name);assert len(a)==19 and len(b)==20;bm={r['rally_candidate_id']:r for r in b};assert all(bm[r['rally_candidate_id']]==r for r in a)
    assert (old/'confirmed_human_outcomes.csv').read_bytes()==(out/'confirmed_human_outcomes.csv').read_bytes();assert len(read(out/'confirmed_human_outcomes.csv'))==11
    pending=lambda folder:{r['rally_candidate_id'] for r in read(folder/'human_outcomes.csv') if not r['winner_camera_side']};assert pending(out)==pending(old)|{'match1_rally_window_016'}
    a=read(old/'fixed30_processing_progress.csv');b=read(out/'fixed30_processing_progress.csv');assert [r for r in a if not r['candidate_id'].endswith('_016')]==[r for r in b if not r['candidate_id'].endswith('_016')]
    meta=next(r for r in read(out/'rally_metadata.csv') if r['rally_candidate_id'].endswith('_016'));assert meta['tracking_source']==str(c/'four_player_path.csv')
    keys=['source_frame','player_slot','track_id','x1','y1','x2','y2','bottom_center_x','bottom_center_y'];key=lambda r:tuple(r[k] for k in keys);baseline=read(c/'four_player_path.csv');checked=load(c/'verification.json')['recalculated_feature_values'];matched=len(baseline);decoded=load(c/'verification.json')['video_frames_decoded'];result=load(t/'result.json');assert result['baseline_temporal_replay_exact']
    modes=['court_seed_only','old_guard','court_seed_plus_old_guard'];diagnostic=load(t/'temporal_metadata_diagnostic.json')
    for mode in modes:
        folder=t/mode;q=load(folder/'protocol.json');assert all(sha(Path(n))==h for n,h in q['protected_input_hashes'].items());assert load(folder/'verification.json')['status']=='passed';rr=load(folder/'case_result.json');assert all(sha(folder/n)==h for n,h in rr['artifact_hashes'].items())
        assert [key(r) for r in read(folder/'four_player_path.csv')]==[key(r) for r in baseline]
        for name in ['prefix_frame_features.csv','snapshot_player_features.csv','prefix_player_summary_features.csv']:assert read(folder/name)==read(c/name)
        v=load(folder/'verification.json');assert v['initialization_delay_from_baseline']==0 and v['four_observation_requested_times']==31 and not v['numeric_thresholds_changed'];assert diagnostic[mode]['temporal_box_ID_rows_equal']
        assert not load(folder/'AI_visual_findings.json')['known_wrong_person_assignment'];checked+=v['recalculated_feature_values'];matched+=v['matched_native_boxes'];decoded+=v['video_frames_decoded']
    assert diagnostic['court_seed_only']['metadata_differing_fields']['alternative_cost_margin']==0
    assert all(diagnostic[m]['metadata_differing_fields']['alternative_cost_margin']==1 for m in modes[1:])
    phase=load(c/'phase_evidence/manifest.json');assert phase['input_grid_source_frames']==times;assert all(sha(c/'phase_evidence'/n)==h for n,h in phase['images'].items())
    counts=load(c/'case_result.json');assert counts['input_times_with_four_observations']==31
    cp=load(BASE/'cohort_and_baselines/protocol.json');score={n:h for n,h in cp['protected_input_hashes'].items() if Path(n).name=='score_suggestion_validation_sample.csv'};assert len(score)==1 and all(sha(Path(n))==h for n,h in score.items())
    video=t/'service_prefix_review_h264.mp4';cap=cv2.VideoCapture(str(video));n=0;cells=[]
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(1080,1920)
        if n in [0,87,180,300]:
            head=np.zeros((32,960,3),np.uint8);cv2.putText(head,f'016 F{40549+n}: frozen combined candidate',(10,24),0,.7,(255,255,255),1);cells.append(np.vstack([head,cv2.resize(im,(960,540))]))
        n+=1
    cap.release();assert n==301 and len(cells)==4;assert cv2.imwrite(str(t/'video_QA.jpg'),np.vstack([np.hstack(cells[:2]),np.hstack(cells[2:])]))
    dump(t/'excerpt_manifest.json',dict(source_video=str(t/'court_seed_plus_old_guard/tracking_quality_20261102_stage7/players_016_global/global_review_h264.mp4'),first_source_frame=40549,last_source_frame=40849,frames=n,dimensions=[1920,1080],aspect_preserved=True,video_sha256=sha(video),scope='Entire inherited context, includes initial waiting and late broadcast dissolve; not certified exact point boundaries.'))
    dump(t/'initialization_overlay_manifest.json',dict(source_frame=40556,source_video_sha256=p['source_video_sha256'],seed_ids=result['modes']['court_seed_plus_old_guard']['seed_ids'],calibration=frozen['calibration'],calibration_sha256=sha(Path(frozen['calibration'])),image_sha256=sha(t/'initialization_overlay.jpg'),assessment='AI visual line consistency only; no independent camera-calibration transfer accuracy.'))
    r=load(out/'result.json');assert r['fixed30_constructed_inputs']==17 and r['fixed30_pending_inputs']==13 and r['next_candidate']=='match1_rally_window_017'
    nextplan=dict(frozen,next_candidate=r['next_candidate'],previous_transfer_result=str(t/'result.json'),previous_transfer_result_sha256=sha(t/'result.json'),previous_transfer_visual_findings=str(t/'AI_visual_findings.json'),previous_transfer_visual_findings_sha256=sha(t/'AI_visual_findings.json'),previous_transfer_interpretation='016 final paths and31timefeatures unchanged acrossfour modes; no extra initialization delay or nonplayer rescue opportunity. Retain24frame late F2 gap. Stable-output comparison only.')
    nextplan['fixed_before_017_inspection']=True;nextplan['threshold_tuning_on_017']=False;dump(t/'next_transfer_protocol.json',nextplan)
    report=dict(status='passed',candidate_id=p['candidate_id'],previous_input_rows_exactly_retained=19,total_input_rows=20,confirmed_human_outcomes_preserved=11,pending_outcomes=len(pending(out)),all_four_modes_final_slot_ID_box_paths_equal=True,all_four_modes_fixed31_features_equal=True,all_four_temporal_box_ID_paths_equal=True,guard_alternative_cost_margin_only_difference_rows=1,requested_four_candidate_times=31,requested_times=31,additional_initialization_wait_frames=0,individual_ID_binding_added=False,threshold_tuning_on_016=False,matched_native_observation_rows_across_four_outputs=matched,recalculated_feature_values_across_four_outputs=checked,decoded_frames_before_excerpt=decoded,review_video_frames=n,source_contact_AI_bracket_frozen_before_tracking=True,existing_score_validation_sample_unchanged=True,late_F2_gap_frames_retained=24,canonical_pipeline_replaced=False,independent_identity_or_position_accuracy_established=False,evidence_for_new_error_prevention=False)
    dump(out/'additional_verification.json',report);dump(t/'final_verification.json',report)
    for name in ['run_fixed_cohort_player_case_v3.py','verify_fixed_cohort_player_case.py','prepare_cohort_phase_images.py','prepare_fixed_input_identity_sheet.py','prepare_player_gap_diagnostics.py','prepare_post_contact_gap_images.py','record_case016_visual_review.py','finalize_case016.py']:shutil.copy2(ROOT/name,c/('executed_'+name))
    for name in ['run_court_initialization_transfer.py','court_initialization_guard.py','confirmed_continuity_guard.py','automatic_reacquisition_guard.py']:shutil.copy2(ROOT/name,t/('executed_'+name))
    shutil.copy2(ROOT/'append_fixed_cohort_player_inputs_v4.py',out/'executed_source.py')
    for name in ['prepare_cohort_anchor_search.py','prepare_contact_detail_v2.py']:shutil.copy2(ROOT/name,anchor/('executed_'+name))
    day=BASE/'README.md';marker='016で固定済みの初期化対策を追加比較';history='\n\n'+marker+'。カメラ復帰時刻をサーブ接触と混同せず、原画像で接触候補F40633〜40639を追跡前に固定。指定F40669〜40729の31時点は全て4候補で、AI点検では対応が整合して見える。原処理・初期化領域制限のみ・旧再取得ガード・併用の4通りで最終1123観測と配置値は同一、追加の初期化待ちなし。手前の非選手混入の防止を新たに確認した事例ではない。旧ガードの時間方向メタデータの差はalternative_cost_margin1行のみ。終盤F2のF40813〜40836の24フレーム欠測・画面切替は保持。4出力の4492元枠・1516配置値・計1505動画フレーム等を照合し、公開用301フレーム動画も確認。current20_player_inputsに従来処理由来を追加し、旧19件・確認済み11ラベルを保持。固定30群内は17件作成・13件未処理、得点側未確認9件。正規処理未置換・独立精度未評価。次の017もコードと閾値を固定して比較する。\n'
    assert marker not in day.read_text(encoding='utf-8');day.write_text(day.read_text(encoding='utf-8')+history,encoding='utf-8')
    for folder in [anchor,c,t,out]:dump(folder/'publication_hashes.json',{str(n.relative_to(folder)):sha(n) for n in sorted(folder.rglob('*')) if n.is_file() and n.name!='publication_hashes.json'})
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
