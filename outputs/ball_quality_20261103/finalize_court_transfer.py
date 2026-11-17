"""Close a frozen same-match transfer; preserve missing observations and label provenance."""
import argparse,json,shutil
from pathlib import Path
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,sha,dump

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def protected(p):
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
def verified(folder,result_name):
    protected(load(folder/'protocol.json'))
    v=load(folder/'verification.json');assert v['status']=='passed'
    r=load(folder/result_name)
    assert all(sha(folder/n)==h for n,h in r['artifact_hashes'].items())
    return v,r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--previous',required=True);ap.add_argument('--output',required=True);ap.add_argument('--excerpt-first',type=int,required=True);ap.add_argument('--excerpt-last',type=int,required=True);a=ap.parse_args()
    tag=a.tag;c=BASE/f'cohort_tracking/case_{tag}';t=BASE/f'court_initialization_transfer_{tag}';old=BASE/a.previous;out=BASE/a.output;anchor=BASE/f'cohort_anchor_proposals/{tag}'
    assert not (out/'additional_verification.json').exists()
    cv,cr=verified(c,'case_result.json');ov,r=verified(out,'result.json');p=load(c/'protocol.json');tp=load(t/'protocol.json');protected(tp)
    frozen=load(Path(tp['frozen_plan']));assert frozen[f'fixed_before_{tag}_inspection'] and not frozen[f'threshold_tuning_on_{tag}']
    assert all(sha(Path(n))==h for n,h in frozen['frozen_code_and_calibration_hashes'].items())
    assert not tp['individual_ID_binding_added'] and not tp['numeric_thresholds_changed'] and not tp['transfer_results_used_to_tune'] and not tp['canonical_pipeline_replaced']
    times=p['requested_input_frames'];assert times==list(range(p['contact_candidate_last_source_frame']+30,p['contact_candidate_last_source_frame']+91,2))
    if p['contact_proposal_file']:
        proposal=load(Path(p['contact_proposal_file']));assert proposal['fixed_before_new_person_tracking'] and not proposal['context_boundary_modified'];assert all(sha(Path(n))==h for n,h in proposal['evidence_hashes'].items())
    else:assert p['contact_status']=='legacy_record_precision_unknown_not_new_GT'
    names=['B1_server_side.csv','B2_snapshot.csv','B2_window_means.csv','diagnostic_snapshot_masks_only.csv','diagnostic_window_masks_only.csv','human_outcomes.csv','source_snapshot_features.csv','source_window_summary_features.csv','rally_metadata.csv','processing_interventions.csv','outcome_label_status.csv']
    oldn=len(read(old/'rally_metadata.csv'))
    for name in names:
        before=read(old/name);after=read(out/name);assert len(before)==oldn and len(after)==oldn+1
        assert [row for row in after if row['rally_candidate_id']!=p['candidate_id']]==before
    assert (old/'confirmed_human_outcomes.csv').read_bytes()==(out/'confirmed_human_outcomes.csv').read_bytes()
    pending=lambda folder:{row['rally_candidate_id'] for row in read(folder/'human_outcomes.csv') if not row['winner_camera_side']}
    assert pending(out)==pending(old)|{p['candidate_id']}
    assert [row for row in read(old/'fixed30_processing_progress.csv') if row['candidate_id']!=p['candidate_id']]==[row for row in read(out/'fixed30_processing_progress.csv') if row['candidate_id']!=p['candidate_id']]
    meta=next(row for row in read(out/'rally_metadata.csv') if row['rally_candidate_id']==p['candidate_id']);assert meta['tracking_source']==str(c/'four_player_path.csv')
    keys=['source_frame','player_slot','track_id','x1','y1','x2','y2','bottom_center_x','bottom_center_y'];key=lambda row:tuple(row[k] for k in keys)
    baseline={key(row) for row in read(c/'four_player_path.csv')};assert len(baseline)==cr['selected_slot_counts']['near_front']+sum(cr['selected_slot_counts'][k] for k in ['near_rear','far_front','far_rear'])
    result=load(t/'result.json');assert result['baseline_temporal_replay_exact'];changes={};matched=len(baseline);checked=cv['recalculated_feature_values'];decoded=cv['video_frames_decoded']
    for mode in ['court_seed_only','old_guard','court_seed_plus_old_guard']:
        folder=t/mode;v,rr=verified(folder,'case_result.json');actual={key(row) for row in read(folder/'four_player_path.csv')};added=actual-baseline;removed=baseline-actual
        for name in ['prefix_frame_features.csv','snapshot_player_features.csv','prefix_player_summary_features.csv']:assert read(folder/name)==read(c/name)
        assert len(added)==v['added_full_assignments'] and len(removed)==v['removed_full_assignments']
        assert not v['numeric_thresholds_changed'] and v['four_observation_requested_times']==cr['input_times_with_four_observations']
        assert not load(folder/'AI_visual_findings.json')['known_wrong_person_assignment']
        changes[mode]=dict(added_rows=len(added),removed_rows=len(removed),added_source_frames=sorted({int(row[0]) for row in added}),removed_source_frames=sorted({int(row[0]) for row in removed}),initialization_delta_frames=v['initialization_delay_from_baseline'])
        matched+=v['matched_native_boxes'];checked+=v['recalculated_feature_values'];decoded+=v['video_frames_decoded']
    features=read(c/'prefix_frame_features.csv');assert len(features)==31
    summary=read(c/'prefix_player_summary_features.csv')[0]
    for side in ['near','far']:
        observed=sum(int(row[f'{side}_pair_valid']) for row in features);assert abs(float(summary[f'{side}_pair_observed_fraction'])-observed/31)<1e-12
    visual=load(c/'AI_visual_findings.json');assert visual['input_phase_status']=='ai_visual_sequence_supported' and not visual['known_wrong_person_assignment']
    assert not visual['AI_identity_binding_added'] and not visual['independent_identity_or_position_accuracy_established']
    phase=load(c/'phase_evidence/manifest.json');assert phase['input_grid_source_frames']==times;assert all(sha(c/'phase_evidence'/n)==h for n,h in phase['images'].items())
    init=load(t/'initialization_evidence/manifest.json');assert all(sha(t/'initialization_evidence'/row['image_file'])==row['sha256'] for row in init['images'])
    cp=load(BASE/'cohort_and_baselines/protocol.json');score={n:h for n,h in cp['protected_input_hashes'].items() if Path(n).name=='score_suggestion_validation_sample.csv'};assert len(score)==1 and all(sha(Path(n))==h for n,h in score.items())
    video=t/'service_prefix_review_h264.mp4';cap=cv2.VideoCapture(str(video));n=0;cells=[]
    selected=[0,p['contact_candidate_last_source_frame']-a.excerpt_first,times[-1]-a.excerpt_first,a.excerpt_last-a.excerpt_first];assert len(set(selected))==4
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(1080,1920)
        if n in selected:
            head=np.zeros((32,960,3),np.uint8);cv2.putText(head,f'{tag} source F{a.excerpt_first+n}: frozen combined candidate',(10,24),0,.65,(255,255,255),1);cells.append(np.vstack([head,cv2.resize(im,(960,540))]))
        n+=1
    cap.release();assert n==a.excerpt_last-a.excerpt_first+1 and len(cells)==4;assert cv2.imwrite(str(t/'video_QA.jpg'),np.vstack([np.hstack(cells[:2]),np.hstack(cells[2:])]))
    dump(t/'excerpt_manifest.json',dict(source_video=str(t/f'court_seed_plus_old_guard/tracking_quality_20261102_stage7/players_{tag}_global/global_review_h264.mp4'),first_source_frame=a.excerpt_first,last_source_frame=a.excerpt_last,frames=n,dimensions=[1920,1080],aspect_preserved=True,video_sha256=sha(video),scope='Service vicinity through inherited context end; not independently certified exact rally bounds.'))
    nexttag=r['next_candidate'].rsplit('_',1)[-1];nextplan=dict(frozen,next_candidate=r['next_candidate'],scope='Additional same-match transfer after rule freeze; not independent other-match validation.',previous_transfer_result=str(t/'result.json'),previous_transfer_result_sha256=sha(t/'result.json'),previous_transfer_visual_findings=str(t/'AI_visual_findings.json'),previous_transfer_visual_findings_sha256=sha(t/'AI_visual_findings.json'),previous_transfer_interpretation=f'{tag}: four fixed31 input features equal; retain missing observations. Initialization changes in final_verification.json. No new independent accuracy or error-prevention evidence.')
    nextplan[f'fixed_before_{nexttag}_inspection']=True;nextplan[f'threshold_tuning_on_{nexttag}']=False;dump(t/'next_transfer_protocol.json',nextplan)
    report=dict(status='passed',candidate_id=p['candidate_id'],previous_input_rows_exactly_retained=oldn,total_input_rows=oldn+1,confirmed_human_outcomes_preserved=len(read(out/'confirmed_human_outcomes.csv')),pending_outcomes=len(pending(out)),assignment_changes=changes,all_four_modes_fixed31_features_equal=True,requested_four_candidate_times=cr['input_times_with_four_observations'],requested_times=31,missing_requested_times=cv['missing_requested_times'],individual_ID_binding_added=False,threshold_tuning=False,matched_native_observation_rows_across_four_outputs=matched,recalculated_feature_values_across_four_outputs=checked,decoded_frames_before_excerpt=decoded,review_video_frames=n,contact_status=p['contact_status'],existing_score_validation_sample_unchanged=True,canonical_pipeline_replaced=False,independent_identity_or_position_accuracy_established=False,evidence_for_new_error_prevention=False)
    dump(out/'additional_verification.json',report);dump(t/'final_verification.json',report)
    for name in ['run_fixed_cohort_player_case_v3.py','verify_fixed_cohort_player_case.py','prepare_cohort_phase_images.py','prepare_fixed_input_identity_sheet.py','prepare_player_gap_diagnostics.py',f'record_case{tag}_visual_review.py','finalize_court_transfer.py']:shutil.copy2(ROOT/name,c/('executed_'+name))
    for name in ['run_court_initialization_transfer.py','court_initialization_guard.py','confirmed_continuity_guard.py','automatic_reacquisition_guard.py','prepare_court_transfer_diagnostics.py','finalize_court_transfer.py']:shutil.copy2(ROOT/name,t/('executed_'+name))
    shutil.copy2(ROOT/'append_fixed_cohort_player_inputs_v4.py',out/'executed_source.py')
    for name in ['prepare_cohort_anchor_search.py','prepare_contact_detail_v2.py']:shutil.copy2(ROOT/name,anchor/('executed_'+name))
    gap_path=c/'gap_diagnostics/all_final_path_gap_runs.csv'
    gaps=read(gap_path) if gap_path.exists() else []
    missing_text=json.dumps(cv['missing_requested_times'],ensure_ascii=False)
    change_text=json.dumps(changes,ensure_ascii=False)
    fractions={side:float(summary[f'{side}_pair_observed_fraction']) for side in ['near','far']}
    text=f'''# {tag}：固定序盤の配置入力と初期化比較

接触候補F{p['contact_candidate_first_source_frame']}〜F{p['contact_candidate_last_source_frame']}を基準に、指定{times[0]}〜{times[-1]}の31時点を変更せず処理した。接触記録の状態は {p['contact_status']}。原画像のサービス動作と入力窓をAI点検した。独立した正確な接触時刻やサービス有効性の正解ではない。

4候補が揃うのは{cr['input_times_with_four_observations']}/31時点。指定時点の欠測は {missing_text}。補間も別選手への置換も行っていない。平均値は各組の観測時点から算出し、観測割合 {fractions} を併記した。終点は指定終点の観測のみを使う。全経路の連続欠測はgap_diagnostics/all_final_path_gap_runs.csvに保存（{len(gaps)}区間）。

従来処理・初期化領域制限のみ・旧再取得ガード・併用で指定31時点の配置値は一致した。従来との全経路の差は {change_text}。指定窓の一致と全経路の一致は区別する。新たな誤初期化防止の成功例には数えない。数値コストの変化は本人同定精度ではない。

AI点検所見：{visual['observation']}

全指定時点と欠測前後の画像で選手への対応は整合して見えるが、AI点検であり独立した本人同定・接地点の正解は追加していない。部分的な選手枠の下端は接地点を表さない場合がある。奥側対応付けには後続映像を使用する。終盤の放送切替と未確定のラリー終端は保持し、得点側は未確認のままにした。同一試合での開発比較であり、別試合での検証ではない。
'''
    (c/'README.md').write_text(text,encoding='utf-8');(t/'README.md').write_text(text+'\n実験処理は正規パイプラインへ置換していない。次の候補でもコード・閾値を固定し、失敗や欠測を残す。検証記録はfinal_verification.json、動画はservice_prefix_review_h264.mp4。\n',encoding='utf-8')
    (out/'README.md').write_text(f'''# 配置入力{oldn+1}候補：{tag}を追加

旧{oldn}件の全入力値・欠測・介入履歴と、確認済み11得点ラベル（手前7・奥4）を保持した。全{oldn+1}候補、固定30群内では{r['fixed30_constructed_inputs']}件作成・{r['fixed30_pending_inputs']}件未処理、次は{nexttag}。これは自動追跡成功数や、正解付き分析適格ラリー数ではない。得点側未確認は{len(pending(out))}件で空欄。human_outcomes.csvは未確認も含む台帳、confirmed_human_outcomes.csvは確認済み11件である。

{tag}の入力は従来処理由来。4候補{cr['input_times_with_four_observations']}/31時点、欠測は {missing_text}、観測割合は {fractions}。平均は観測時点、終点の配置は指定終点の観測を使用する。4条件で指定窓の配置値は同じだが、全経路の差は別途final_verification.jsonに記録した。AI点検では示した対応と整合するものの、独立精度は未評価。新しい本人IDの個別指定はない。009・011・015の過去の個別修正由来は残している。

既存接触記録の時間精度、部分枠の接地点誤差、後続映像を使う奥側対応付け、同一試合内の開発という制約は残る。今回、学習・性能推定・球の追加価値比較は行っていない。固定16件の得点検証標本と原プロジェクトも変更していない。
''',encoding='utf-8')
    day=BASE/'README.md';marker=f'{tag}の固定序盤入力を追加';assert marker not in day.read_text(encoding='utf-8')
    day.write_text(day.read_text(encoding='utf-8')+f'\n\n{marker}。4候補{cr["input_times_with_four_observations"]}/31、欠測を保持。4条件の指定窓配置値は同じ。全経路の差は当該final_verification.jsonを参照。新たな本人同定精度や誤初期化防止成功を示したものではない。{a.output}に従来処理由来で追加、旧{oldn}件と11得点ラベルを保持。固定30群内{r["fixed30_constructed_inputs"]}件作成・{r["fixed30_pending_inputs"]}件未処理、次{nexttag}。詳細と照合記録は当該README・final_verification.json。\n',encoding='utf-8')
    for folder in [anchor,c,t,out]:dump(folder/'publication_hashes.json',{str(n.relative_to(folder)):sha(n) for n in sorted(folder.rglob('*')) if n.is_file() and n.name!='publication_hashes.json'})
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
