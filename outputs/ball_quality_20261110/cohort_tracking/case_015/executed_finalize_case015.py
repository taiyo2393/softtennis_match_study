"""Verify015 assisted publication while rejecting both known-wrong automatic branches."""
import csv,json,hashlib,shutil,subprocess,sys
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'ball_quality_20261110'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    c=BASE/'cohort_tracking/case_015';v=c/'AI_native_binding';t=BASE/'automatic_reacquisition_transfer_015';old=BASE/'current18_player_inputs';out=BASE/'current19_player_inputs';anchor=BASE/'cohort_anchor_proposals/015';assert not (out/'additional_verification.json').exists()
    for folder in [c,v,t,out]:
        p=load(folder/'protocol.json');assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items());assert load(folder/'verification.json')['status']=='passed'
        if folder!=t:
            r=load(folder/('result.json' if folder==out else 'case_result.json'));assert all(sha(folder/n)==h for n,h in r['artifact_hashes'].items())
    assert load(c/'AI_visual_findings.json')['known_wrong_person_assignment'] and load(t/'AI_visual_findings.json')['known_wrong_person_assignment']
    assert not load(v/'AI_visual_findings.json')['known_wrong_person_assignment'] and load(v/'protocol.json')['AI_identity_binding_added']
    names=['B1_server_side.csv','B2_snapshot.csv','B2_window_means.csv','diagnostic_snapshot_masks_only.csv','diagnostic_window_masks_only.csv','human_outcomes.csv','source_snapshot_features.csv','source_window_summary_features.csv','rally_metadata.csv','processing_interventions.csv','outcome_label_status.csv']
    for name in names:
        before=read(old/name);after=read(out/name);assert len(before)==18 and len(after)==19;amap={r['rally_candidate_id']:r for r in after};assert all(amap[r['rally_candidate_id']]==r for r in before),name
    assert (old/'confirmed_human_outcomes.csv').read_bytes()==(out/'confirmed_human_outcomes.csv').read_bytes();assert len(read(out/'confirmed_human_outcomes.csv'))==11
    previous={r['rally_candidate_id'] for r in read(old/'human_outcomes.csv') if not r['winner_camera_side']};pending={r['rally_candidate_id'] for r in read(out/'human_outcomes.csv') if not r['winner_camera_side']};assert pending==previous|{'match1_rally_window_015'} and len(pending)==8
    before=read(old/'fixed30_processing_progress.csv');after=read(out/'fixed30_processing_progress.csv');assert [r for r in before if not r['candidate_id'].endswith('_015')]==[r for r in after if not r['candidate_id'].endswith('_015')]
    p=load(v/'protocol.json');times=p['requested_input_frames'];assert times==list(range(38846,38907,2))==load(c/'protocol.json')['requested_input_frames']
    a=read(c/'four_player_path.csv');b=read(v/'four_player_path.csv');key=lambda r:tuple(r[k] for k in ['source_frame','player_slot','track_id','x1','y1','x2','y2','bottom_center_x','bottom_center_y'])
    assert [key(r) for r in a if r['player_slot']!='near_front']==[key(r) for r in b if r['player_slot']!='near_front']
    aa=[r for r in a if r['player_slot']=='near_front'];bb=[r for r in b if r['player_slot']=='near_front'];assert len(aa)==len(bb)==653 and all(r['track_id']=='3' for r in bb)
    assert {r['source_frame'] for r in aa}=={r['source_frame'] for r in bb};assert all(r['track_id']=='1' for r in aa if int(r['source_frame']) in times)
    assert sum(key(x)!=key(y) for x,y in zip(aa,bb))==653
    for name in ['source_snapshot_features.csv','source_window_summary_features.csv']:
        chosen=next(r for r in read(out/name) if r['rally_candidate_id'].endswith('_015'));native=read(v/('snapshot_player_features.csv' if 'snapshot' in name else 'prefix_player_summary_features.csv'))[0];assert chosen==native
    orig=read(c/'snapshot_player_features.csv')[0];new=read(v/'snapshot_player_features.csv')[0];assert orig['near_x_gap_norm']!=new['near_x_gap_norm']
    meta=next(r for r in read(out/'rally_metadata.csv') if r['rally_candidate_id'].endswith('_015'));assert meta['tracking_source']==str(v/'four_player_path.csv')
    i=next(r for r in read(out/'processing_interventions.csv') if r['rally_candidate_id'].endswith('_015'));assert i['intervention']=='AI_identity_seed_binding'
    rejected=BASE/'rejected_auto015_gate_check';assert not rejected.exists()
    proc=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'append_fixed_cohort_player_inputs_v4.py'),'--tag','015','--previous',old.name,'--output',rejected.name],capture_output=True,text=True,encoding='utf-8');assert proc.returncode!=0 and 'Known wrong-person output' in proc.stderr and not rejected.exists()
    tp=load(t/'protocol.json');frozen=load(Path(tp['frozen_plan']));assert frozen['fixed_before_015_inspection'];assert frozen['source_code_sha256']==tp['frozen_guard_code_sha256']==sha(Path(frozen['source_code']))
    trial=load(t/'verification.json');assert trial['baseline_temporal_replay_exact'] and trial['fixed31_features_identical'] and trial['removed_near_observations']==25 and not trial['threshold_or_confirmation_length_tuned_on_transfer']
    cp=load(BASE/'cohort_and_baselines/protocol.json');score={n:h for n,h in cp['protected_input_hashes'].items() if Path(n).name=='score_suggestion_validation_sample.csv'};assert len(score)==1 and all(sha(Path(n))==h for n,h in score.items())
    cap=cv2.VideoCapture(str(v/'service_prefix_review_h264.mp4'));frames=0;fps=cap.get(5)
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(1080,1920);frames+=1
    cap.release();assert frames==539 and abs(fps-60000/1001)<.01
    dump(v/'excerpt_manifest.json',dict(source_video=str(v/'tracking_quality_20261102_stage7/players_015_global/global_review_h264.mp4'),first_source_frame=38662,last_source_frame=39200,frames=frames,aspect_preserved=True,AI_assisted_overlay=True,video_sha256=sha(v/'service_prefix_review_h264.mp4'),scope='Includes late broadcast dissolve; not formal point-end ground truth.'))
    result=load(out/'result.json');assert result['fixed30_constructed_inputs']==16 and result['fixed30_pending_inputs']==14 and result['next_candidate']=='match1_rally_window_016'
    nextplan=dict(frozen,next_candidate=result['next_candidate'],previous_transfer_result=str(t/'result.json'),previous_transfer_result_sha256=sha(t/'result.json'),previous_transfer_visual_findings=str(t/'AI_visual_findings.json'),previous_transfer_visual_findings_sha256=sha(t/'AI_visual_findings.json'),previous_transfer_known_wrong_initialization=True,previous_case_interpretation='Both automatic outputs assign the official as N1 at all31 requested times. No successful automatic rescue. Assisted variant is kept separately.',selection='Next chronological pending group retained; initial-player selection failure must be addressed before treating stable tracking as success.',priority_before_more_expansion='Develop explicit initial-player selection/exclusion; any new rule is development on known failures and must be separately frozen before subsequent transfer.')
    nextplan['fixed_before_016_inspection']=True;nextplan['threshold_tuning_on_016']=False;dump(t/'next_transfer_protocol.json',nextplan)
    report=dict(status='passed',previous_input_rows_exactly_retained=18,candidate_rows=19,confirmed_human_outcomes_preserved=11,unreviewed_outcomes=8,known_wrong_automatic_output_rejected_before_writing=True,automatic_and_frozen_guard_initialization_failure_retained=True,model_input_uses_explicit_AI_assisted_variant=True,AI_bound_N1_observations=653,unchanged_other_slot_observations=len(b)-653,far_search_rerun_with_retained_numeric_parameters=True,far_assignments_unchanged=True,requested_input_frames_unchanged=True,fixed_prefix_N1_assignments_changed=31,baseline_near_x_gap_at_endpoint=float(orig['near_x_gap_norm']),assisted_near_x_gap_at_endpoint=float(new['near_x_gap_norm']),fixed_prefix_four_observation_times=31,existing_score_validation_sample_unchanged=True,baseline_temporal_replay_exact=True,frozen_guard_code_unchanged=True,assisted_excerpt_frames_decoded=frames,fully_automatic_success_claimed=False,independent_identity_or_position_accuracy_established=False,performance_evaluation_performed=False,canonical_tracking_replaced=False)
    dump(out/'additional_verification.json',report);dump(t/'final_verification.json',report);dump(v/'publication_verification.json',report)
    for name in ['run_fixed_cohort_player_case_v3.py','verify_fixed_cohort_player_case.py','prepare_cohort_phase_images.py','prepare_player_gap_diagnostics.py','prepare_fixed_input_identity_sheet.py','prepare_case015_identity_diagnosis.py','record_case015_visual_review.py','finalize_case015.py']:shutil.copy2(ROOT/name,c/('executed_'+name))
    shutil.copy2(ROOT/'repair_case015_native_binding.py',v/'executed_repair_source.py');shutil.copy2(ROOT/'prepare_fixed_input_identity_sheet_v2.py',v/'executed_prepare_fixed_input_identity_sheet_v2.py');shutil.copy2(ROOT/'append_fixed_cohort_player_inputs_v4.py',out/'executed_source.py')
    for name in ['run_frozen_guard_transfer.py','confirmed_continuity_guard.py','automatic_reacquisition_guard.py']:shutil.copy2(ROOT/name,t/('executed_'+name))
    for name in ['prepare_contact_detail_v2.py','prepare_cohort_anchor_search.py']:shutil.copy2(ROOT/name,anchor/('executed_'+name))
    day=BASE/'README.md';marker='015は初期化で審判を選ぶ失敗';history=f'\n\n{marker}を検出。従来と固定済みガードの両方が全31指定時点でN1を右端の審判へ割り当てたため、自動出力を配置入力から不採用として保持。実選手raw3の653個の元枠をAI個別指定でN1に結び、同じ数値条件で奥を再探索した別出力をcase_015/AI_native_bindingに保存。残る1944観測は同じで、序盤31時点の4候補と配置値を保存。指定末尾の手前横幅は誤出力{report["baseline_near_x_gap_at_endpoint"]:.3f}から修正版{report["assisted_near_x_gap_at_endpoint"]:.3f}に変わる。current19_player_inputsは旧18件を保持し、個別指定の由来を明記して追加。固定30群の16件作成・14件未処理、得点側の人手確認11件・未確認8件。固定スコア検証標本も保持。自動成功や独立精度とはせず、次は審判等を除外する初期化対策を優先する。次の未処理016も残す。\n'
    assert marker not in day.read_text(encoding='utf-8');day.write_text(day.read_text(encoding='utf-8')+history,encoding='utf-8')
    for folder in [anchor,v,c,t,out]:dump(folder/'publication_hashes.json',{str(n.relative_to(folder)):sha(n) for n in sorted(folder.rglob('*')) if n.is_file() and n.name!='publication_hashes.json'})
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
