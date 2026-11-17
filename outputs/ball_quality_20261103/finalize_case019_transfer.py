"""Close019 as a retained far-association failure; do not promote structural checks."""
import json,shutil
from pathlib import Path
from collections import defaultdict
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,write,sha,dump

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,value):assert not p.exists();dump(p,value)
def verify(folder,result_name):
    p=load(folder/'protocol.json');assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    v=load(folder/'verification.json');assert v['status']=='passed'
    r=load(folder/result_name);assert all(sha(folder/n)==h for n,h in r['artifact_hashes'].items())
    return p,v,r

def main():
    tag='019';c=BASE/f'cohort_tracking/case_{tag}';t=BASE/f'pair_initialization_transfer_{tag}';anchor=BASE/f'cohort_anchor_proposals/{tag}';current=BASE/'current22_player_inputs'
    p,v,r=verify(c,'case_result.json');tp=load(t/'protocol.json');assert all(sha(Path(n))==h for n,h in tp['protected_input_hashes'].items());plan=load(Path(tp['frozen_plan']));assert plan['fixed_before_019_inspection'] and not plan['threshold_tuning_on_019'];assert all(sha(Path(n))==h for n,h in plan['frozen_code_and_calibration_hashes'].items())
    assert tp['modes']==['baseline','court_seed_only','pair_rank'] and not tp['numeric_thresholds_changed'] and not tp['individual_ID_binding_added'] and not tp['transfer_results_used_to_tune']
    proposal=load(Path(p['contact_proposal_file']));assert proposal['fixed_before_new_person_tracking'] and not proposal['context_boundary_modified'];assert all(sha(Path(n))==h for n,h in proposal['evidence_hashes'].items())
    assert proposal['earlier_service_motion_present'] and proposal['serve_attempt_number']=='' and not proposal['phase_selection_uses_outcome_label'];assert p['requested_input_frames']==list(range(46376,46437,2))
    keys=['source_frame','player_slot','track_id','x1','y1','x2','y2','bottom_center_x','bottom_center_y'];sig=lambda rows:[tuple(row[k] for k in keys) for row in rows];baseline=read(c/'four_player_path.csv');matched=len(baseline);checked=v['recalculated_feature_values'];decoded=v['video_frames_decoded'];result=load(t/'result.json');assert result['baseline_temporal_replay_exact']
    for mode in ['court_seed_only','pair_rank']:
        mp,mv,mr=verify(t/mode,'case_result.json');assert sig(read(t/mode/'four_player_path.csv'))==sig(baseline)
        assert mv['temporal_assignments_equal_baseline'] and mv['initialization_delay_from_baseline']==0 and mv['four_observation_requested_times']==21 and not mv['numeric_thresholds_changed']
        for n in ['prefix_frame_features.csv','snapshot_player_features.csv','prefix_player_summary_features.csv']:assert read(t/mode/n)==read(c/n)
        matched+=mv['matched_native_boxes'];checked+=mv['recalculated_feature_values'];decoded+=mv['video_frames_decoded']
    diagnostic=load(t/'temporal_metadata_diagnostic.json');assert all(e['temporal_box_ID_rows_equal'] and all(value==0 for value in e['metadata_differing_fields_on_matching_boxes'].values()) for e in diagnostic.values())
    times=p['requested_input_frames'];assert [row['source_frame'] for row in v['missing_requested_times']]==list(range(46376,46395,2))
    summary=read(c/'prefix_player_summary_features.csv')[0];assert abs(float(summary['far_pair_observed_fraction'])-21/31)<1e-12 and float(summary['near_pair_observed_fraction'])==1
    assert read(c/'snapshot_player_features.csv')[0]['far_pair_valid']=='1'
    people=defaultdict(dict)
    for row in baseline:people[int(row['source_frame'])][row['player_slot']]=row
    probes=[]
    for f,slot,assessment in [(46501,'far_rear','AI: box includes background cyan official; invalid single-player localization'),(46503,'far_rear','AI: assigned box is on background cyan official'),(46505,'far_rear','AI: assigned box remains on background cyan official'),(46511,'far_rear','AI: one box includes cyan official and dark player'),(46634,'far_rear','AI: overlapping dark/orange players; single-person localization unresolved'),(46665,'far_front','AI: far F2 assigned to near white/red player at crossing'),(46666,'far_front','AI: far F2 still assigned to near white/red player at crossing')]:
        assert f>times[-1];row=people[f][slot];probes.append(dict(row,AI_assessment=assessment,in_requested_input_window=0,independent_human_GT=0))
    write(t/'AI_failure_probes.csv',probes)
    phase=load(c/'phase_evidence/manifest.json');assert phase['input_grid_source_frames']==times;assert all(sha(c/'phase_evidence'/n)==h for n,h in phase['images'].items())
    identity=load(c/'fixed_input_identity/manifest.json');assert all(sha(c/'fixed_input_identity'/n)==h for n,h in identity['images'].items())
    gaps=load(c/'post_contact_gap_evidence/manifest.json');assert len(gaps['gaps'])==20;assert all(sha(c/'post_contact_gap_evidence'/row['image_file'])==row['sha256'] for row in gaps['gaps'])
    init=load(t/'initialization_evidence/manifest.json');assert len(init['images'])==1 and set(init['images'][0]['modes'])==set(tp['modes']);assert sha(t/'initialization_evidence'/init['images'][0]['image_file'])==init['images'][0]['sha256']
    findings=dict(candidate_id=p['candidate_id'],reviewer='Codex AI saved chronological image inspection, not independent human GT',input_phase_status='ai_visual_sequence_supported',known_wrong_person_assignment=True,AI_identity_binding_added=False,input_accepted_for_analysis=False,tracking_identity_success=False,scope_of_known_wrong_persons='Later active-play context, outside fixed input window. This does not prove all observed prefix positions wrong.',contact_candidate_frames=[46340,46346],earlier_service_motion_present=True,earlier_motion_termination_reason='unresolved',formal_serve_attempt_number='',service_validity_human_confirmed=False,initialized_frame=45831,initialization_warning='Broadcast dissolve still present; shown boxes surround actual four players, not an automatic clean-view gate.',observation='Requested boxes appear consistent with N1 dark-capped near-right receiver, N2 white-capped near-left teammate, F1 dark far-left server and F2 orange far player. F1 is absent for first10 requested times and often represented only by a visible fragment after returning. Later F1 selects cyan official/non-single-person boxes aroundF46501..46517. F2 raw218 selects near white/red player aroundF46665..46666. Allthree modes have identical final paths, so new initial-pair rule does not solve these far-association failures.',requested_input_frames=times,four_candidate_requested_times=21,missing_requested_frames=list(range(46376,46395,2)),actual_overlapping_prefix_gap=dict(slot='far_rear',first=46353,last=46394,source_frames=42),known_wrong_probe_frames=[row['source_frame'] for row in probes],input_prefix_identity_accuracy_established=False,partial_box_lower_edge_is_ground_contact=False,all_three_modes_paths_and_prefix_equal=True,evidence_for_new_error_prevention=False,independent_identity_or_position_accuracy_established=False,formal_point_end_frame_confirmed=False,winner_camera_side='',winner_label_status='unreviewed',offline_future_context_used_for_far_association=True,inspected_saved_images=['phase_evidence/context_grid.jpg','fixed_input_identity/near_all31.jpg','fixed_input_identity/far_all31.jpg','tracking_quality_20261102_stage4/players_019_temporal/initialization_confirmed.jpg','gap_diagnostics/gap_01_near_rear.jpg']+[str(Path('post_contact_gap_evidence')/row['image_file']) for row in gaps['gaps']],inspected_raw_service_evidence=['coarse_page_01.jpg','coarse_page_02.jpg','coarse_page_03.jpg','later_service_detail/page_01.jpg','later_service_detail/page_02.jpg','contact_focus/page_01.jpg','contact_followthrough/page_01.jpg'])
    save(c/'AI_visual_findings.json',findings);save(t/'AI_visual_findings.json',dict(findings,shared_evidence='All source-frame/slot/ID/box paths and initial seeds are exact across three modes.'))
    for mode in ['court_seed_only','pair_rank']:save(t/mode/'AI_visual_findings.json',dict(findings,mode=mode))
    _,_,current_result=verify(current,'result.json');assert current_result['fixed30_constructed_inputs']==19 and current_result['next_candidate']==p['candidate_id'];assert len(read(current/'confirmed_human_outcomes.csv'))==11;assert p['candidate_id'] not in {row['rally_candidate_id'] for row in read(current/'rally_metadata.csv')}
    current_hashes={str(n):sha(n) for n in current.iterdir() if n.is_file()};assert all(sha(current/n)==h for n,h in load(current/'publication_hashes.json').items())
    progress=read(current/'fixed30_processing_progress.csv')
    for row in progress:
        if row['candidate_id']==p['candidate_id']:assert row['fixed_window_features_constructed']=='0';row['status']='attempted_three_modes_far_wrong_person_context_retained_input_adoption_pending'
    write(t/'fixed30_processing_progress_after019.csv',progress)
    decision=dict(candidate_id=p['candidate_id'],status='known_far_wrong_person_assignments_retained_adoption_pending',tracking_attempt_completed=True,prefix_features_calculated_and_retained=True,prefix_four_candidate_times=21,prefix_requested_times=31,known_wrong_person_probes_outside_prefix=True,policy='Preserve prefix calculation and full-path failure separately. No claim that later errors invalidate every prefix observation. Do not silently relax current known-wrong-person adoption gate or delete candidate from fixed cohort.',new_candidate_appended=False,current_valid_input_snapshot=str(current),current_input_snapshot_hashes=current_hashes,fixed30_constructed_inputs=19,fixed30_pending_inputs=11,retained_failed_candidate=p['candidate_id'],next_unresolved_candidate=p['candidate_id'],no_visibility_driven_substitution=True,next_action='Address far association after occlusion: official/non-single-person boxes and near/far duplicated person boxes. Use known019 probes for development and retained controls for regression; keep missing rather than fabricate coordinates.',threshold_tuning_on019=False,individual_ID_binding_added=False,canonical_pipeline_replaced=False,outcomes_unchanged=True,independent_GT_added=False)
    save(c/'processing_decision.json',decision);save(t/'processing_decision.json',decision)
    cp=load(BASE/'cohort_and_baselines/protocol.json');score={n:h for n,h in cp['protected_input_hashes'].items() if Path(n).name=='score_suggestion_validation_sample.csv'};assert len(score)==1 and all(sha(Path(n))==h for n,h in score.items())
    video=load(t/'failure_review_manifest.json');assert video['decoded_frames']==426 and sha(t/'failure_review_h264.mp4')==video['video_sha256'] and sha(t/'failure_review_QA.jpg')==video['QA_image_sha256']
    report=dict(status='passed',verification_scope='Calculation, source-box provenance, exact comparison and failure retention; not tracking identity success',tracking_identity_success=False,baseline_temporal_replay_exact=True,all_three_modes_final_paths_equal=True,all_three_modes_fixed31_features_equal=True,matched_native_rows_across_three_outputs=matched,recalculated_feature_values_across_three_outputs=checked,decoded_original_and_branch_frames=decoded,known_wrong_probe_rows=len(probes),post_contact_gap_sheets_AI_inspected=20,four_candidate_requested_times=21,requested_times=31,review_video_decoded_frames=426,current22_inputs_unchanged=all(sha(Path(n))==h for n,h in current_hashes.items()),confirmed11_outcomes_preserved=True,fixed30_constructed_inputs=19,fixed30_pending_inputs=11,case019_retained_not_skipped=True,individual_ID_binding_added=False,threshold_tuning=False,canonical_pipeline_replaced=False,independent_identity_or_position_accuracy_established=False)
    save(t/'final_verification.json',report)
    text='''# 019：初期化3条件は一致、奥側の誤対応を発見して入力採用保留

018で開発した服装によるペア選択を、事前固定したコード・閾値で019へ適用した。原処理・コート領域制限のみ・ペア順位の3条件で、初期化F45831、4人の初期対象、最終3268観測、指定序盤の配置値は完全に一致した。新しい誤初期化防止の成功例ではなく、初期化を変えても残る奥側の対応付け問題を示す。同一試合の追加比較であり、別試合の独立精度評価ではない。

候補区間にはサーブ動作が2回あり、最初の動作の後はいったん仕切り直している。原画像の時系列から、続くプレーの接触候補をF46340〜46346に追跡前固定し、F46376〜46436を2フレーム間隔の31時点として計算した。カメラ復帰や追跡初期化を接触として使わない。最初がフォルトかレットか、正式な何本目か、有効性、厳密なラリー終端、得点側は人手未確認のまま。

指定31時点の最初10時点で奥F1が欠測。実際の連続欠測はF46353〜46394の42元フレームであり、手前N2との重なりを含む。4候補が揃うのは21/31。奥側平均は観測21時点からの値で、観測割合21/31を保存し、欠測を補間しない。復帰後にも身体の一部だけの枠があり、その下端は接地点として不正確になり得る。観測枠数を本人同定精度や測位精度に読み替えない。

接触後の全20欠測区間の前後画像をAI点検すると、F46501・46503・46505付近でF1の枠が背景の水色の審判へ移り、F46511などでは審判と選手を含む混合枠になっていた。F46665・46666付近では、奥F2が手前の白赤の選手へ割り当てられる。AI_failure_probes.csvに元枠とAI判定を保存した。候補を一意のrawIDで割り当て、計算と元枠を照合しても、この人物対応の失敗は検出できなかった。3条件に同じ失敗があり、初期化対策の効果とはいえない。

これらの明白な誤対応の例は指定入力窓の後であり、序盤の観測がすべて誤りだと証明したわけではない。序盤特徴の計算結果と、ラリー文脈全体の追跡品質は分けて保存する。既知の誤人物を含む出力は現在の採用条件を満たさないため、019の入力表への追加は保留した。候補を固定30群の分母から削除せず、失敗を飛ばして020へ進めず、未来の追跡失敗を理由に標本を黙って選別しない。将来、序盤だけを採用する条件を設ける場合には、条件と適用範囲を明記して再検討する。

current22_player_inputsの全ファイル・確認済み11得点ラベルを保持した。固定30群の入力作成は19件、未完了11件のまま。これは分析適格ラリー数や追跡成功数ではない。設定調整、個別ID指定、正規処理の置換、学習や性能推定は行っていない。

次は、遮蔽後の奥側対応で審判・混合枠・手前選手の重複検出を選ばない処理を開発する。019の発見済み事例は開発例として扱い、過去の対照区間で正しい観測を過剰に落とさないかも比較する。初期化の改善と、遮蔽中の観測の信頼性を別々に評価する。

failure_review_h264.mp4は失敗付近2区間の拡大動画。縦横比を保ち、元映像の1/3速度で表示している。修正後の成功動画ではない。
'''
    (c/'README.md').write_text(text,encoding='utf-8');(t/'README.md').write_text(text,encoding='utf-8')
    save(t/'far_failure_followup_plan.json',dict(status='planned_not_implemented',reason='Initial pair selection left far official/near-person assignment failures unchanged',development_case='019',known_probe_file=str(t/'AI_failure_probes.csv'),known_probe_sha256=sha(t/'AI_failure_probes.csv'),preserve_cases_and_missing=True,do_not_hardcode_frame_or_ID=True,comparators='Retained original far association versus proposed rejection/association rule; unchanged detections, near path and fixed requested times.',evaluation='Record wrong-object rejection and lost true observations separately; new rule needs controls before a frozen additional transfer. AI-probe review remains development, not independent identity/position GT.',cohort_candidate_remains_pending='match1_rally_window_019',canonical_pipeline_replaced=False))
    for n in ['run_fixed_cohort_player_case_v3.py','verify_fixed_cohort_player_case.py','prepare_cohort_phase_images.py','prepare_fixed_input_identity_sheet_v3.py','prepare_player_gap_diagnostics_v2.py','prepare_post_contact_gap_images.py','finalize_case019_transfer.py']:shutil.copy2(ROOT/n,c/('executed_'+n))
    for n in ['run_pair_initialization_transfer.py','pair_initialization_guard.py','court_initialization_guard.py','prepare_initialization_transfer_diagnostics.py','prepare_pair_transfer_failure_review.py','finalize_case019_transfer.py']:shutil.copy2(ROOT/n,t/('executed_'+n))
    for n in ['prepare_cohort_anchor_search.py','prepare_contact_detail_v2.py']:shutil.copy2(ROOT/n,anchor/('executed_'+n))
    day=BASE/'README.md';marker='019の固定3条件比較で奥側の誤対応を発見';assert marker not in day.read_text(encoding='utf-8');day.write_text(day.read_text(encoding='utf-8')+'\n\n'+marker+'。2回のサービス動作のうち続くプレー側の接触候補F46340〜46346を追跡前固定。指定31時点は4候補21・F1欠測10。3条件の最終3268観測と配置値は同じだが、入力窓後にF1の審判・混合枠、F2の手前選手への割当てが残る。全20接触後欠測前後をAI点検。既知誤対応を含むため019の入力採用は保留、序盤特徴は診断用に保存。入力窓後の失敗と序盤値の正誤は区別する。固定群から除外せず019を未解決に残す。current22の全ファイル・11ラベル保持、19/30作成のまま。次は奥側の遮蔽後対応付けを改善する。\n',encoding='utf-8')
    for folder in [anchor,c,t]:dump(folder/'publication_hashes.json',{str(n.relative_to(folder)):sha(n) for n in sorted(folder.rglob('*')) if n.is_file() and n.name!='publication_hashes.json'})
    assert all(sha(Path(n))==h for n,h in current_hashes.items());print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
