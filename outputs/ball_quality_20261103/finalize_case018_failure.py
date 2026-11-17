"""Preserve a known wrong-person transfer; structural checks are not identity success."""
import json,shutil
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,write,sha,dump
from court_initialization_guard import court_point,eligible

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(p,x):assert not p.exists();dump(p,x)
def check(folder,name):
    p=load(folder/'protocol.json');assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    v=load(folder/'verification.json');assert v['status']=='passed'
    result=load(folder/name);assert all(sha(folder/n)==h for n,h in result['artifact_hashes'].items())
    return p,v,result

def main():
    c=BASE/'cohort_tracking/case_018';t=BASE/'court_initialization_transfer_018';anchor=BASE/'cohort_anchor_proposals/018';current=BASE/'current21_player_inputs'
    p,v,r=check(c,'case_result.json');tp=load(t/'protocol.json');assert all(sha(Path(n))==h for n,h in tp['protected_input_hashes'].items())
    plan=load(Path(tp['frozen_plan']));assert plan['fixed_before_018_inspection'] and not plan['threshold_tuning_on_018'];assert all(sha(Path(n))==h for n,h in plan['frozen_code_and_calibration_hashes'].items())
    proposal=load(Path(p['contact_proposal_file']));assert proposal['fixed_before_new_person_tracking'];assert all(sha(Path(n))==h for n,h in proposal['evidence_hashes'].items())
    assert p['requested_input_frames']==list(range(43838,43899,2)) and proposal['contact_candidate_first_frame']==43804 and proposal['contact_candidate_last_frame']==43808
    tr=load(t/'result.json');assert tr['baseline_temporal_replay_exact'] and not tp['individual_ID_binding_added'] and not tp['numeric_thresholds_changed']
    signatures=['source_frame','player_slot','track_id','x1','y1','x2','y2'];key=lambda row:tuple(row[k] for k in signatures)
    baseline={key(row) for row in read(c/'four_player_path.csv')};counts={};matched=v['selected_boxes_matched_to_augmented_detection_source'];checked=v['recalculated_feature_values'];decoded=v['video_frames_decoded']
    for mode in ['court_seed_only','old_guard','court_seed_plus_old_guard']:
        mp,mv,mr=check(t/mode,'case_result.json');actual={key(row) for row in read(t/mode/'four_player_path.csv')}
        assert len(actual-baseline)==mv['added_full_assignments'] and len(baseline-actual)==mv['removed_full_assignments']
        for name in ['prefix_frame_features.csv','snapshot_player_features.csv','prefix_player_summary_features.csv']:assert read(t/mode/name)==read(c/name)
        assert mv['seed_ids']=={'near_front':3,'near_rear':1,'far_rear':4,'far_front':5} and mv['initialized_frame']==43646
        assert not mv['numeric_thresholds_changed'] and not mv['individual_ID_binding_added']
        counts[mode]=dict(added_rows=mv['added_full_assignments'],removed_rows=mv['removed_full_assignments'],known_wrong_initial_identity=True,four_candidate_times=mv['four_observation_requested_times'])
        matched+=mv['matched_native_boxes'];checked+=mv['recalculated_feature_values'];decoded+=mv['video_frames_decoded']
    byframe=defaultdict(dict)
    for row in read(c/'four_player_path.csv'):byframe[int(row['source_frame'])][row['player_slot']]=row
    assert all(byframe[f]['near_front']['track_id']=='3' and len(byframe[f])==4 for f in p['requested_input_frames'])
    h=np.array(load(Path(plan['calibration']))['image_to_court_homography']);candidates=[]
    raw=read(c/'crop_runs/augmented_detections_018.csv')
    for row in raw:
        if int(row['source_frame'])!=43646:continue
        xy=court_point(row,h)
        candidates.append(dict(row,court_x='' if xy is None else xy[0],court_y='' if xy is None else xy[1],allowed_by_frozen_court_rule=int(eligible(row,h)),visual_identity_note='AI visual: official, wrongly seeded' if row['track_id']=='3' else 'AI visual: near-right actual player, not seeded' if row['track_id']=='2' else 'not individually annotated here'))
    write(t/'initialization_candidate_geometry.csv',candidates)
    official=next(row for row in candidates if row['track_id']=='3');player=next(row for row in candidates if row['track_id']=='2')
    assert official['allowed_by_frozen_court_rule']==1 and player['allowed_by_frozen_court_rule']==1 and float(official['confidence'])>float(player['confidence'])
    phase=load(c/'phase_evidence/manifest.json');assert all(sha(c/'phase_evidence'/n)==value for n,value in phase['images'].items())
    init=load(t/'initialization_evidence/manifest.json');assert len(init['images'])==1 and set(init['images'][0]['modes'])==set(tr['modes']);assert all(sha(t/'initialization_evidence'/row['image_file'])==row['sha256'] for row in init['images'])
    findings=dict(candidate_id=p['candidate_id'],reviewer='Codex AI saved image inspection, not independent human GT',input_phase_status='ai_visual_sequence_supported',known_wrong_person_assignment=True,AI_identity_binding_added=False,input_accepted_for_analysis=False,wrong_slot='near_front',wrong_raw_id=3,actual_unseeded_near_player_raw_id_candidate=2,wrong_requested_frames=p['requested_input_frames'],four_candidate_times=31,correct_four_player_identity_not_established=True,contact_candidate_frames=[43804,43808],contact_review='AI bracket from raw overhead service sequence frozen before person tracking; legal-service/attempt-number/exact-frame GT unknown',inspected_images=['fixed_input_identity/near_all31.jpg','fixed_input_identity/far_all31.jpg','phase_evidence/context_grid.jpg','candidate_sequence_grid.jpg','far_pair_detail_grid.jpg']+[str(n.relative_to(c)) for n in sorted((c/'gap_diagnostics').glob('*.jpg'))]+[str(t/'initialization_evidence'/init['images'][0]['image_file'])],inspected_raw_contact_images=['coarse_page_01.jpg','coarse_page_02.jpg','service_detail/page_01.jpg','service_detail/page_02.jpg','service_contact_refined/page_01.jpg'],observation='N1 raw3 follows the cyan-clad near-right official throughout all31 requested times. The actual dark-capped near-right player raw2 is unassigned. N2 raw1 is the white-capped near-left receiver; F1 dark far-left and F2 orange far-right server appear consistent in the requested images. Allfour automatic modes inherit the same wrong initial N1 identity.',cause='Expanded spatial eligibility admits both official and player; initializer ranks near candidates by detector confidence, not player identity. Eight-frame persistence confirms a stable official as readily as a stable player. Reacquisition guard cannot correct a wrong seed.',initialization_frame=43646,official_court_x=official['court_x'],official_confidence=float(official['confidence']),actual_near_player_confidence=float(player['confidence']),all_four_modes_fixed31_features_equal=True,equality_is_identity_success=False,threshold_tuning_on_018=False,independent_identity_or_position_accuracy_established=False,winner_camera_side='',winner_label_status='unreviewed',formal_point_end_frame_confirmed=False,late_broadcast_dissolve_present=True,interpolation=False,individual_ID_repair_performed=False)
    save(c/'AI_visual_findings.json',findings);save(t/'AI_visual_findings.json',findings)
    for mode in ['court_seed_only','old_guard','court_seed_plus_old_guard']:save(t/mode/'AI_visual_findings.json',dict(findings,mode=mode,common_requested_box_evidence_exact=True))
    _,_,current_result=check(current,'result.json');assert current_result['fixed30_constructed_inputs']==18 and current_result['next_candidate']==p['candidate_id']
    current_hashes={str(n):sha(n) for n in current.iterdir() if n.is_file()};assert all(sha(current/n)==value for n,value in load(current/'publication_hashes.json').items())
    assert p['candidate_id'] not in {row['rally_candidate_id'] for row in read(current/'rally_metadata.csv')} and len(read(current/'confirmed_human_outcomes.csv'))==11
    progress=read(current/'fixed30_processing_progress.csv')
    for row in progress:
        if row['candidate_id']==p['candidate_id']:assert row['fixed_window_features_constructed']=='0';row['status']='attempted_all4_modes_wrong_initial_person_rejected_pending_redesign'
    write(t/'fixed30_processing_progress_after018.csv',progress)
    decision=dict(candidate_id=p['candidate_id'],status='tracking_attempt_completed_known_wrong_person_input_rejected',structural_verification='passed, validates calculations/provenance only',fixed30_constructed_inputs=18,fixed30_pending_inputs=12,current_valid_input_snapshot=str(current),current_input_snapshot_hashes=current_hashes,new_candidate_appended=False,failed_candidate_retained=True,next_unresolved_candidate=p['candidate_id'],next_action='Develop general initial-person selection and abstention; compare against all retained cases including009/015/018 and nonfailure controls before freeze. Do not skip018 or hardcode raw2. Strict court boundary alone is an untested hypothesis and can exclude legitimate out-of-court players.',canonical_pipeline_replaced=False,outcomes_unchanged=True,independent_GT_added=False)
    save(c/'processing_decision.json',decision);save(t/'processing_decision.json',decision)
    video=t/'known_wrong_initialization_review_h264.mp4';cap=cv2.VideoCapture(str(video));n=0;cells=[]
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(1080,1920)
        if n in [0,60,150,517]:
            head=np.zeros((32,960,3),np.uint8);cv2.putText(head,f'018 F{43748+n}: WRONG green N1 follows official',(8,24),0,.65,(255,255,255),1);cells.append(np.vstack([head,cv2.resize(im,(960,540))]))
        n+=1
    cap.release();assert n==518 and len(cells)==4;assert cv2.imwrite(str(t/'video_QA.jpg'),np.vstack([np.hstack(cells[:2]),np.hstack(cells[2:])]))
    save(t/'excerpt_manifest.json',dict(first_source_frame=43748,last_source_frame=44265,frames=518,dimensions=[1920,1080],aspect_preserved=True,source_video=str(t/'court_seed_plus_old_guard/tracking_quality_20261102_stage7/players_018_global/global_review_h264.mp4'),sha256=sha(video),scope='Known wrong-person result retained for review, not a corrected player-tracking demonstration.'))
    cp=load(BASE/'cohort_and_baselines/protocol.json');score={name:value for name,value in cp['protected_input_hashes'].items() if Path(name).name=='score_suggestion_validation_sample.csv'};assert len(score)==1 and all(sha(Path(name))==value for name,value in score.items())
    report=dict(status='passed',verification_scope='Structural preservation/recalculation and rejection gate, not identity success',tracking_identity_success=False,known_wrong_person_assignment_all_four_modes=True,candidate_rows_not_appended=True,current21_snapshot_unchanged=all(sha(Path(name))==value for name,value in current_hashes.items()),fixed31_features_equal_but_wrong=True,branch_assignment_changes=counts,matched_native_observation_rows_across_four_outputs=matched,recalculated_feature_values_across_four_outputs=checked,decoded_frames_before_excerpt=decoded,review_video_frames=n,confirmed_human_outcomes_preserved=11,fixed30_constructed_inputs=18,fixed30_pending_inputs=12,threshold_tuning=False,canonical_pipeline_replaced=False,independent_identity_or_position_accuracy_established=False)
    save(t/'final_verification.json',report)
    text='''# 018：全4条件で審判を誤初期化したため入力不採用

原処理・コート領域による初期化制限のみ・旧再取得ガード・併用の全条件で、N1が手前右の水色の審判raw3を追った。実際の手前右選手raw2は初期対象に入らない。指定F43838〜43898の31時点すべてで4枠が存在するが、4選手の正しい配置ではない。4条件の序盤配置値の一致も成功ではない。独立した精度正解を追加したという意味ではなく、AI画像点検で明白な誤対応を発見した事例である。

追跡前の原画像で奥右の橙色選手のサーブ接触候補F43804〜43808を固定した。カメラ復帰の728秒を接触として使わず、入力窓は動かしていない。接触時刻・有効性・何本目のサーブかは人手正解未確定。得点側も未確認のまま。

原因は初期対象の選別。審判の下端はコート正規化x=1.0276付近で、現在の拡張許容範囲[-0.15,1.15]に入る。初期化時のperson検出確信度は審判0.8966817、実選手0.8966296であり、初期候補の確信度順で審判を選ぶ。8フレームの連続確認は、同じ審判を安定して選ぶことも確認してしまう。コート線の視覚的一致は独立した校正精度の検証ではない。候補全体の座標・判定はinitialization_candidate_geometry.csvに保存した。

領域制限のみの全経路は原処理と同一。旧ガード・併用は入力窓外で20枠除去・1枠追加となるが、誤った初期人物は変わらない。終盤には奥選手の11・17・14・27フレームの欠測と放送切替があり、欠測は補間していない。計算・元枠照合の検証合格と、人物対応の失敗を区別した。

018は比較入力へ追加しない。current21_player_inputsの旧21候補・確認済み11ラベルは全ファイル不変。固定30群は入力作成18件、未完了12件のままで、018を未解決として残す。失敗を飛ばして019へ進めない。個別IDの手動指定、閾値調整、正規処理の置換も行っていない。

次は初期選手の組合せの評価と、曖昧な場合に初期化を保留する方法を検討する。009・015・018の失敗と従来の対照区間を同じ条件で再比較してから設定を固定する。境界だけを狭める方法は、コート外にいる正当な選手を落とし得るため未検証の案に留める。
'''
    (c/'README.md').write_text(text,encoding='utf-8');(t/'README.md').write_text(text,encoding='utf-8')
    for name in ['run_fixed_cohort_player_case_v3.py','verify_fixed_cohort_player_case.py','prepare_cohort_phase_images.py','prepare_fixed_input_identity_sheet.py','prepare_player_gap_diagnostics.py','finalize_case018_failure.py']:shutil.copy2(ROOT/name,c/('executed_'+name))
    for name in ['run_court_initialization_transfer.py','court_initialization_guard.py','confirmed_continuity_guard.py','automatic_reacquisition_guard.py','prepare_court_transfer_diagnostics.py','finalize_case018_failure.py']:shutil.copy2(ROOT/name,t/('executed_'+name))
    for name in ['prepare_cohort_anchor_search.py','prepare_contact_detail_v2.py']:shutil.copy2(ROOT/name,anchor/('executed_'+name))
    day=BASE/'README.md';marker='018は4条件とも審判を誤初期化';assert marker not in day.read_text(encoding='utf-8');day.write_text(day.read_text(encoding='utf-8')+'\n\n'+marker+'。31/31時点で4候補があっても本人対応が誤っているため配置入力へ不採用。拡張許容領域に審判が入り、検出確信度順の選別で実選手を上回った。コード・閾値の事後調整なし。current21_player_inputsを全ファイル保持、入力作成18/30のまま、018を未解決として残す。個別ID指定や失敗のスキップをせず初期化規則の再検討が必要。詳細はcourt_initialization_transfer_018/README.mdとprocessing_decision.json。\n',encoding='utf-8')
    for folder in [anchor,c,t]:dump(folder/'publication_hashes.json',{str(n.relative_to(folder)):sha(n) for n in sorted(folder.rglob('*')) if n.is_file() and n.name!='publication_hashes.json'})
    assert all(sha(Path(name))==value for name,value in current_hashes.items());print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
