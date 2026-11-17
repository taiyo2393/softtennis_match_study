"""Verify assisted011 publication without promoting it to automatic tracking GT."""
import csv,hashlib,json,shutil,subprocess,sys
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'ball_quality_20261110'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    case=BASE/'cohort_tracking/case_011';variant=case/'AI_native_binding';old=BASE/'current14_player_inputs';out=BASE/'current15_player_inputs';anchor=BASE/'cohort_anchor_proposals/011'
    assert not (out/'additional_verification.json').exists()
    for folder in [case,variant,out]:
        p=load(folder/'protocol.json');assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
        result=load(folder/('result.json' if folder==out else 'case_result.json'));assert all(sha(folder/n)==h for n,h in result['artifact_hashes'].items())
        assert load(folder/'verification.json')['status']=='passed'
    cp=load(BASE/'cohort_and_baselines/protocol.json');score_files={n:h for n,h in cp['protected_input_hashes'].items() if Path(n).name=='score_suggestion_validation_sample.csv'}
    assert len(score_files)==1 and all(sha(Path(n))==h for n,h in score_files.items())
    files=['B1_server_side.csv','B2_snapshot.csv','B2_window_means.csv','diagnostic_snapshot_masks_only.csv','diagnostic_window_masks_only.csv','human_outcomes.csv','source_snapshot_features.csv','source_window_summary_features.csv','rally_metadata.csv','processing_interventions.csv','outcome_label_status.csv']
    for name in files:
        before=read(old/name);after=read(out/name);assert len(before)==14 and len(after)==15
        amap={r['rally_candidate_id']:r for r in after};assert all(amap[r['rally_candidate_id']]==r for r in before),name
    assert (old/'confirmed_human_outcomes.csv').read_bytes()==(out/'confirmed_human_outcomes.csv').read_bytes()
    assert len(read(out/'confirmed_human_outcomes.csv'))==11
    assert {r['rally_candidate_id'][-3:] for r in read(out/'human_outcomes.csv') if not r['winner_camera_side']}=={'008','009','010','011'}
    before=read(old/'fixed30_processing_progress.csv');after=read(out/'fixed30_processing_progress.csv');assert [r for r in before if not r['candidate_id'].endswith('_011')]==[r for r in after if not r['candidate_id'].endswith('_011')]
    p=load(variant/'protocol.json');times=p['requested_input_frames'];assert times==list(range(26933,26994,2))
    keys=['source_frame','player_slot','track_id','x1','y1','x2','y2','bottom_center_x','bottom_center_y']
    original=read(case/'four_player_path.csv');repaired=read(variant/'four_player_path.csv')
    aa=[{k:r[k] for k in keys} for r in original if int(r['source_frame']) in times];bb=[{k:r[k] for k in keys} for r in repaired if int(r['source_frame']) in times];assert aa==bb and len(aa)==124
    n1={r['source_frame']:r for r in repaired if r['player_slot']=='near_front'}
    assert all({k:n1[r['source_frame']][k] for k in r}==r for r in original if r['player_slot']=='near_front')
    assert all(r['player_slot']=='near_rear' for r in repaired if r['track_id']=='1')
    for name in ['prefix_frame_features.csv','snapshot_player_features.csv','prefix_player_summary_features.csv']:assert read(case/name)==read(variant/name)
    meta=next(r for r in read(out/'rally_metadata.csv') if r['rally_candidate_id'].endswith('_011'));assert 'AI_native_binding' in meta['tracking_source']
    intervention=next(r for r in read(out/'processing_interventions.csv') if r['rally_candidate_id'].endswith('_011'));assert intervention['intervention']=='AI_identity_seed_binding'
    rejected=BASE/'rejected_auto011_gate_check';assert not rejected.exists()
    proc=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'append_fixed_cohort_player_inputs_v4.py'),'--tag','011','--previous','current14_player_inputs','--output',rejected.name],capture_output=True,text=True,encoding='utf-8')
    assert proc.returncode!=0 and 'Known wrong-person output' in proc.stderr and not rejected.exists()
    video=variant/'before_after_N2_h264.mp4';cap=cv2.VideoCapture(str(video));frames=0;fps=cap.get(cv2.CAP_PROP_FPS)
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(540,1920);frames+=1
    cap.release();assert frames==241 and abs(fps-60000/1001)<.01
    dump(variant/'before_after_manifest.json',dict(left_source=str(case/'tracking_quality_20261102_stage7/players_011_global/global_review_h264.mp4'),right_source=str(variant/'tracking_quality_20261102_stage7/players_011_global/global_review_h264.mp4'),first_source_frame=27500,last_source_frame=27740,frames=frames,width=1920,height=540,each_panel=[960,540],aspect_preserved=True,video_sha256=sha(video),scope='Left automatic, right AI assisted; diagnostic selected failure segment, not representative performance evaluation.'))
    report=dict(status='passed',previous14_input_rows_exactly_retained=True,confirmed_human_outcomes_preserved=11,unreviewed_outcomes=4,known_wrong_automatic_output_rejected_before_writing=True,model_input_uses_explicit_AI_assisted_variant=True,all124_fixed_prefix_slot_raw_box_assignments_unchanged=True,prefix_feature_values_unchanged=True,unchanged_N1_observations=2900,native_N2_observations_bound=2882,native_N2_missing_frames_retained=18,far_search_rerun_with_retained_numeric_parameters=True,existing_score_validation_sample_unchanged=True,comparison_video_frames_decoded=frames,fully_automatic_success_claimed=False,independent_identity_or_position_accuracy_established=False,performance_evaluation_performed=False)
    dump(out/'additional_verification.json',report)
    for name in ['run_fixed_cohort_player_case_v3.py','verify_fixed_cohort_player_case.py','prepare_cohort_phase_images.py','prepare_player_gap_diagnostics.py','prepare_fixed_input_identity_sheet.py','finalize_case011.py']:shutil.copy2(ROOT/name,case/('executed_'+name))
    shutil.copy2(ROOT/'repair_case011_native_binding.py',variant/'executed_repair_source.py');shutil.copy2(ROOT/'append_fixed_cohort_player_inputs_v4.py',out/'executed_source.py')
    for name in ['prepare_contact_detail_v2.py','prepare_cohort_anchor_search.py']:shutil.copy2(ROOT/name,anchor/('executed_'+name))
    day=BASE/'README.md';history='\n\n011は最初の試技後の停止・再試技を分け、後のサーブ接触候補F26897〜26903をAIが追跡前に固定。固定窓F26933〜26993の31時点は4候補で、全31画像では対応が整合した。一方、ラリー途中でN2が相手の橙色raw409を追い、奥側探索にも影響する誤りを発見。元の全区間自動出力を不採用として保持し、実際のサーバーraw1をAIが指定、元の2882観測のみをN2へ結び付け、18欠測・N1の2900観測を保持、同じ数値設定で奥側を再探索した別出力を `cohort_tracking/case_011/AI_native_binding` に保存。序盤124枠の対応と3特徴CSVは修正前後で同一。元10030観測・5814動画フレーム、修正版10950観測・2907動画フレーム、各379値等を照合。`current15_player_inputs` にAI指定の由来を明記して追加し、旧14件・確認済み11ラベル・固定16件のスコア検証標本は保持。固定30群では12件作成・18件未処理、次の未処理は012。得点側は008〜011の4件が未確認。個別修正であり、自動再取得の根本対策・独立精度・有効性の達成とはしていない。\n'
    assert '011は最初の試技後' not in day.read_text(encoding='utf-8');day.write_text(day.read_text(encoding='utf-8')+history,encoding='utf-8')
    for folder in [anchor,variant,case,out]:dump(folder/'publication_hashes.json',{str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='publication_hashes.json'})
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
