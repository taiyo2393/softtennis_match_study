"""Verify012 input expansion and frozen transfer, retaining stable-case limitations."""
import csv,hashlib,json,shutil
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'ball_quality_20261110'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    case=BASE/'cohort_tracking/case_012';transfer=BASE/'automatic_reacquisition_transfer_012';old=BASE/'current15_player_inputs';out=BASE/'current16_player_inputs';anchor=BASE/'cohort_anchor_proposals/012'
    assert not (out/'additional_verification.json').exists()
    for folder in [case,transfer,out]:
        p=load(folder/'protocol.json');assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
        assert load(folder/'verification.json')['status']=='passed'
    for folder,name in [(case,'case_result.json'),(out,'result.json')]:assert all(sha(folder/n)==h for n,h in load(folder/name)['artifact_hashes'].items())
    files=['B1_server_side.csv','B2_snapshot.csv','B2_window_means.csv','diagnostic_snapshot_masks_only.csv','diagnostic_window_masks_only.csv','human_outcomes.csv','source_snapshot_features.csv','source_window_summary_features.csv','rally_metadata.csv','processing_interventions.csv','outcome_label_status.csv']
    for name in files:
        before=read(old/name);after=read(out/name);assert len(before)==15 and len(after)==16
        amap={r['rally_candidate_id']:r for r in after};assert all(amap[r['rally_candidate_id']]==r for r in before),name
    assert (old/'confirmed_human_outcomes.csv').read_bytes()==(out/'confirmed_human_outcomes.csv').read_bytes()
    assert len(read(out/'confirmed_human_outcomes.csv'))==11
    assert {r['rally_candidate_id'][-3:] for r in read(out/'human_outcomes.csv') if not r['winner_camera_side']}=={'008','009','010','011','012'}
    before=read(old/'fixed30_processing_progress.csv');after=read(out/'fixed30_processing_progress.csv');assert [r for r in before if not r['candidate_id'].endswith('_012')]==[r for r in after if not r['candidate_id'].endswith('_012')]
    cols=['source_frame','player_slot','track_id','x1','y1','x2','y2','bottom_center_x','bottom_center_y'];baseline=read(case/'four_player_path.csv');guard=read(transfer/'four_player_path.csv');assert [{k:r[k] for k in cols} for r in baseline]==[{k:r[k] for k in cols} for r in guard]
    for name in ['prefix_frame_features.csv','snapshot_player_features.csv','prefix_player_summary_features.csv']:assert read(case/name)==read(transfer/name)
    p=load(case/'protocol.json');assert p['requested_input_frames']==list(range(32056,32117,2))
    frame_rows=read(case/'prefix_frame_features.csv');assert len(frame_rows)==31 and all(r['near_pair_valid']==r['far_pair_valid']=='1' for r in frame_rows)
    v=load(transfer/'verification.json');assert v['near_raw_ID_transitions']=={'baseline_replay':0,'frozen_guard':0} and not v['guard_benefit_exercised_by_actual_ID_switch']
    frozen_path=BASE/'automatic_reacquisition_followup/next_transfer_protocol.json';frozen=load(frozen_path);assert sha(Path(frozen['source_code']))==frozen['source_code_sha256']==load(transfer/'protocol.json')['frozen_guard_code_sha256']
    cp=load(BASE/'cohort_and_baselines/protocol.json');score={n:h for n,h in cp['protected_input_hashes'].items() if Path(n).name=='score_suggestion_validation_sample.csv'};assert len(score)==1 and all(sha(Path(n))==h for n,h in score.items())
    video=transfer/'service_prefix_review_h264.mp4';cap=cv2.VideoCapture(str(video));count=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(1080,1920);count+=1
    cap.release();assert count==191
    dump(transfer/'excerpt_manifest.json',dict(source_video=str(transfer/'tracking_quality_20261102_stage7/players_012_global/global_review_h264.mp4'),first_source_frame=31990,last_source_frame=32180,frames=191,video_sha256=sha(video),aspect_preserved=True,scope='Single guard-output excerpt; assignments are identical to baseline, not an improvement comparison.'))
    nextplan=dict(frozen,next_candidate='match1_rally_window_013',fixed_before_013_inspection=True,case012_result=str(transfer/'result.json'),case012_result_sha256=sha(transfer/'result.json'),previous_case_interpretation='No actual near-ID switch; stable-input nonregression only, not a new prevention success.',selection='Next chronological pending group. Keep stable, ambiguous and failed cases; no selection only for guard success.');dump(transfer/'next_transfer_protocol.json',nextplan)
    report=dict(status='passed',previous15_input_rows_exactly_retained=True,confirmed_human_outcomes_preserved=11,unreviewed_outcomes=5,baseline_and_guard_full_assignments_identical=True,native_observation_rows_each=1225,requested_input_times_each=31,four_observation_requested_times_each=31,baseline_and_guard_fixed_features_identical=True,near_ID_switches_observed_each=0,evidence_for_new_guard_benefit=False,stable_case_nonregression_only=True,frozen_code_hash_matched=True,parameters_tuned_on_012=False,manual_ID_binding=False,existing_score_validation_sample_unchanged=True,excerpt_video_frames_decoded=count,independent_identity_or_position_accuracy_established=False,model_fitting_or_performance_estimation=False,canonical_pipeline_replaced=False,new_case013_inspected=False)
    dump(out/'additional_verification.json',report);dump(transfer/'final_verification.json',report)
    for name in ['run_fixed_cohort_player_case_v3.py','verify_fixed_cohort_player_case.py','prepare_cohort_phase_images.py','prepare_player_gap_diagnostics.py','prepare_fixed_input_identity_sheet.py','finalize_case012_transfer.py']:shutil.copy2(ROOT/name,case/('executed_'+name))
    for name in ['run_frozen_guard_transfer012.py','confirmed_continuity_guard.py','automatic_reacquisition_guard.py']:shutil.copy2(ROOT/name,transfer/('executed_'+name))
    shutil.copy2(ROOT/'append_fixed_cohort_player_inputs_v4.py',out/'executed_source.py')
    for name in ['prepare_contact_detail_v2.py','prepare_cohort_anchor_search.py']:shutil.copy2(ROOT/name,anchor/('executed_'+name))
    day=BASE/'README.md';history='\n\n012へ前回固定した自動ID変更確認を、ガードコード・閾値・確認長を変えず適用。接触候補F32020〜32026をAIが追跡前に固定し、指定F32056〜32116の31時点で4候補を処理。従来の割り当て・由来・コスト等を完全再現し、試作の4人全1225枠の対応と序盤配置値は従来と同じだった。手前のID変更が0回のため、変更で出力が悪化しなかった確認に限り、誤再取得防止の新たな成功例とは数えない。原画像・全31時点のID付き拡大画像・奥F2の欠測前後をAI点検し、元枠・各379値・動画等を照合。`automatic_reacquisition_transfer_012` に別保存し、元処理を由来として `current16_player_inputs` に追加。旧15件・確認済み11ラベル・固定16件のスコア検証標本を保持。固定30群では13件作成・17件未処理、次の未処理013。008〜012の5件は得点側未確認。同じガードで013を比較する条件を未点検のまま固定した。\n'
    assert '012へ前回固定した自動ID変更確認' not in day.read_text(encoding='utf-8');day.write_text(day.read_text(encoding='utf-8')+history,encoding='utf-8')
    for folder in [anchor,case,transfer,out]:dump(folder/'publication_hashes.json',{str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='publication_hashes.json'})
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
