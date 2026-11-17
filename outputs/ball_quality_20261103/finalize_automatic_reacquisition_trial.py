"""Check immutable matched trials, record AI review and package experimental outputs."""
import csv,hashlib,json,shutil
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'ball_quality_20261110';TRIAL=BASE/'automatic_reacquisition_trial_v2';FOLLOW=BASE/'automatic_reacquisition_followup'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def key(r):return tuple(str(r[k]) for k in ['source_frame','player_slot','track_id','x1','y1','x2','y2'])
def main():
    assert not (FOLLOW/'final_verification.json').exists();protocol=load(TRIAL/'protocol.json');matched=0
    for folder in [TRIAL,FOLLOW,FOLLOW/'case_011/four_player_experiment']:
        p=load(folder/'protocol.json');assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    replay_values=0
    for plan in protocol['cases']:
        folder=TRIAL/f'case_{plan["tag"]}';raw=read(folder/'cached_detections.csv');cache=load(folder/'cache_manifest.json');z=np.load(folder/'appearance_cache.npz');assert z['features'].shape==(len(raw),2,192) and z['frame_offsets'][-1]==len(raw) and len(z['frame_offsets'])==plan['last']-plan['first']+2
        native=defaultdict(list)
        for r in raw:native[int(r['source_frame']),r['track_id']].append(r)
        baseline=read(folder/'baseline/selected_four.csv');old=read(Path(plan['baseline_folder'])/'selected_four.csv');cols=['source_frame','player_slot','track_id','x1','y1','x2','y2','identity_source','assignment_cost','alternative_cost_margin']
        assert [[r[k] for k in cols] for r in baseline]==[[r[k] for k in cols] for r in old];replay_values+=len(baseline)*len(cols)
        variants=[folder/mode/'selected_four.csv' for mode in protocol['variants']]+[FOLLOW/f'case_{plan["tag"]}/selected_four.csv']
        for path in variants:
            rows=read(path);perframe=defaultdict(list)
            for r in rows:
                f=int(r['source_frame']);assert plan['first']<=f<=plan['last'];assert r['position_source']=='detected_box'
                assert any(all(r[k]==x[k] for k in ['x1','y1','x2','y2','bottom_center_x','bottom_center_y','confidence']) for x in native[f,r['track_id']]);matched+=1;perframe[f].append(r)
            for rs in perframe.values():assert len({r['player_slot'] for r in rs})==len(rs) and len({r['track_id'] for r in rs})==len(rs)
        b={key(r) for r in read(folder/'confirm_new_id/selected_four.csv') if r['player_slot'].startswith('near_')};c={key(r) for r in read(folder/'confirm_new_id_opponent_negatives/selected_four.csv') if r['player_slot'].startswith('near_')};assert b==c
    comparison=load(FOLLOW/'event_comparison.json');assert [r['additional_delay_frames'] for r in comparison['controls']]==[7,5,None,0];assert comparison['stable005_near_assignments_exactly_unchanged']
    probes=comparison['known011_failure_probes'];assert probes[0]['N2_raw409_assignment_rows']==821
    for r in probes[1:]:assert r['N2_raw409_assignment_rows']==0 and all(n['near_rear_raw_id']=='1' for n in r['known_failure_probes'])
    full=FOLLOW/'case_011/four_player_experiment';assert load(full/'verification.json')['status']=='passed'
    near1={key(r) for r in read(FOLLOW/'case_011/selected_four.csv') if r['player_slot'].startswith('near_')};near2={key(r) for r in read(full/'four_player_path.csv') if r['player_slot'].startswith('near_')};assert near1==near2
    current=BASE/'current15_player_inputs';p=load(current/'protocol.json');assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items());assert all(sha(current/n)==h for n,h in load(current/'result.json')['artifact_hashes'].items())
    video=FOLLOW/'automatic_before_after_011_h264.mp4';cap=cv2.VideoCapture(str(video));frames=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(540,1920);frames+=1
    cap.release();assert frames==241
    dump(FOLLOW/'comparison_video_manifest.json',dict(first_source_frame=27500,last_source_frame=27740,frames=241,each_panel=[960,540],left='Original automatic output',right='Confirmed-continuity experimental automatic output, no AI raw-ID binding',aspect_preserved=True,video_sha256=sha(video)))
    review=dict(reviewer='Codex AI chronological image inspection',review_complete=True,independent_human_GT=False,control_findings=[dict(tag='002',finding='Shown before/after frames are consistent with the same dark near-rear player; follow-up output resumes7 source frames later.'),dict(tag='003',finding='Shown before/after frames are consistent with the same orange near-front player; B/C lose all44 remaining observations, follow-up resumes5 frames after baseline change.'),dict(tag='025',finding='N2 dark near-rear player reappears consistently at the existing recovery time. N1 one-frame edge fragment is not emitted by confirmation guard; later baseline also lacks that person.'),dict(tag='005',finding='Near rows exactly unchanged; no new control visual relabelling or independent GT.')],control_evidence='control_image_review/manifest.json and earlier baseline sheets in automatic_reacquisition_trial_v2/control_image_review',case011_finding='candidate_sequence_grid.jpg in case_011/four_player_experiment: shown N2 corresponds to capped white/red server and F1 to orange far player. Sparse images, not whole-frame GT.',decision='Retain automatic follow-up as experimental candidate. Reject original B/C. Do not replace canonical/current15 due single-frame loss, same-match adaptive development and absent independent identity/position evaluation.',opponent_negative_added_value='Not isolated. B and C have identical near outputs in five cases.',new_case012_used=False)
    dump(FOLLOW/'AI_visual_findings.json',review)
    dump(TRIAL/'final_review.json',dict(computational_status=load(TRIAL/'run_result.json'),AI_control_review_complete=True,review_record=str(FOLLOW/'AI_visual_findings.json'),decision='B/C rejected for legitimate sustained003 switch regression.'))
    guard=ROOT/'confirmed_continuity_guard.py';dump(FOLLOW/'next_transfer_protocol.json',dict(next_candidate='match1_rally_window_012',fixed_before_012_inspection=True,source_code=str(guard),source_code_sha256=sha(guard),comparison='Retained baseline and follow-up use exactly same detections and fixed early time window. Determine contact proposal from raw video before tracking; do not move feature window for visibility.',manual_ID_binding=False,threshold_tuning_on_012=False,scope='Next same-match transfer probe, not independent other-match accuracy validation.',retain_failures=True,canonical_adoption='Not authorized by successful development probes alone; report individual known regressions and limits.'))
    dump(full/'scope_clarification.json',dict(near_assignment_policy_changed=True,existing_numeric_thresholds_retained=True,manual_ID_binding=False,canonical_pipeline_changed=False,original_inherited_parameter_flag='algorithm_parameters_changed:false refers retained numeric thresholds; this is a new association policy.',whole_pipeline_online=False))
    report=dict(status='passed',baseline_exact_replay_cases=5,baseline_core_serialized_values_compared=replay_values,all_variant_observations_matched_to_native_boxes=matched,duplicate_slot_or_raw_ID_assignments=False,interpolation=False,guardBC_near_outputs_identical_in_five_cases=True,known_failure_probe_repaired_by_automatic_candidate=True,sustained_control_added_delay_frames=[7,5,0],one_frame_edge_observation_sacrificed=True,stable005_near_rows_preserved=854,comparison_video_frames_decoded=frames,full_four_player_video_frames_decoded=2907,independent_identity_or_position_accuracy_established=False,current15_inputs_and_label_hashes_unchanged=True,manual_raw_ID_binding_in_candidate=False,canonical_pipeline_changed=False,new_case012_inspected=False)
    dump(FOLLOW/'final_verification.json',report)
    for name in ['automatic_reacquisition_guard.py','run_automatic_reacquisition_trial_v2.py','prepare_reacquisition_control_review.py']:shutil.copy2(ROOT/name,TRIAL/('executed_'+name))
    for name in ['confirmed_continuity_guard.py','run_confirmed_continuity_followup.py','review_automatic_guard_trial.py','build_automatic_four_player_trial.py','finalize_automatic_reacquisition_trial.py']:shutil.copy2(ROOT/name,FOLLOW/('executed_'+name))
    day=BASE/'README.md';history='\n\n011のN2取り違えについて、通常継続経路では新IDでも8フレーム再取得確認を通らず採用されることを再現。002・003・005・025・011の5区間で保存済み検出と外観を共有し、元割り当て・由来・コスト等を完全再現した。新IDを厳しい再取得へ回すB/C案は011誤連結を防ぐが、003の正しいID変更後44フレームを失うため不採用。通常継続に妥当な候補も8フレーム確認する追加案を探索的に作り、011の点検済み取り違え場面を個別ID指定なしで回避、持続する対照変更は追加待ち7・5・0フレームで再取得、005の手前854観測を保持した。ただし025の画面端1フレームの観測を失う。`automatic_reacquisition_trial_v2` と `automatic_reacquisition_followup` に比較・コード・点検・制約を保存。追加案の手前経路で011の奥側を同設定で再計算し、全10944枠・379値・2907動画フレームを検査した。正規追跡・current15・人手ラベルは保持。012は未点検のまま次の同一試合適用比較を固定し、他試合の独立評価とは区別した。\n'
    assert '011のN2取り違えについて、通常継続経路' not in day.read_text(encoding='utf-8');day.write_text(day.read_text(encoding='utf-8')+history,encoding='utf-8')
    for folder in [TRIAL,FOLLOW]:dump(folder/'publication_hashes.json',{str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='publication_hashes.json'})
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
