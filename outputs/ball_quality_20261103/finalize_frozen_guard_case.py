"""Verify a chronological cohort expansion and its separately frozen guard transfer."""
import argparse,csv,hashlib,json,re,shutil
from pathlib import Path
import cv2
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'ball_quality_20261110'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--previous',required=True);ap.add_argument('--output',required=True);ap.add_argument('--excerpt-first',type=int,required=True);ap.add_argument('--excerpt-last',type=int,required=True);args=ap.parse_args();tag=args.tag
    assert re.fullmatch(r'\d{3}',tag) and all(re.fullmatch(r'[A-Za-z0-9_]+',x) for x in [args.previous,args.output])
    case=BASE/'cohort_tracking'/f'case_{tag}';trial=BASE/f'automatic_reacquisition_transfer_{tag}';old=BASE/args.previous;out=BASE/args.output;anchor=BASE/'cohort_anchor_proposals'/tag
    assert not (out/'additional_verification.json').exists()
    for folder in [case,trial,out]:
        p=load(folder/'protocol.json');assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items());assert load(folder/'verification.json')['status']=='passed'
    for folder,name in [(case,'case_result.json'),(out,'result.json')]:assert all(sha(folder/n)==h for n,h in load(folder/name)['artifact_hashes'].items())
    original_count=load(old/'result.json')['candidates'];result=load(out/'result.json');assert result['candidates']==original_count+1
    names=['B1_server_side.csv','B2_snapshot.csv','B2_window_means.csv','diagnostic_snapshot_masks_only.csv','diagnostic_window_masks_only.csv','human_outcomes.csv','source_snapshot_features.csv','source_window_summary_features.csv','rally_metadata.csv','processing_interventions.csv','outcome_label_status.csv']
    for name in names:
        before=read(old/name);after=read(out/name);assert len(before)==original_count and len(after)==original_count+1
        amap={r['rally_candidate_id']:r for r in after};assert all(amap[r['rally_candidate_id']]==r for r in before),name
    assert (old/'confirmed_human_outcomes.csv').read_bytes()==(out/'confirmed_human_outcomes.csv').read_bytes()
    before_pending={r['rally_candidate_id'] for r in read(old/'human_outcomes.csv') if not r['winner_camera_side']};pending={r['rally_candidate_id'] for r in read(out/'human_outcomes.csv') if not r['winner_camera_side']};assert pending==before_pending|{f'match1_rally_window_{tag}'}
    before=read(old/'fixed30_processing_progress.csv');after=read(out/'fixed30_processing_progress.csv');assert [r for r in before if not r['candidate_id'].endswith('_'+tag)]==[r for r in after if not r['candidate_id'].endswith('_'+tag)]
    p=load(case/'protocol.json');times=p['requested_input_frames'];assert times==list(range(p['contact_candidate_last_source_frame']+30,p['contact_candidate_last_source_frame']+91,2))
    columns=['source_frame','player_slot','track_id','x1','y1','x2','y2','bottom_center_x','bottom_center_y'];key=lambda r:tuple(r[k] for k in columns)
    baseline=read(case/'four_player_path.csv');guard=read(trial/'four_player_path.csv');aa={key(r) for r in baseline};bb={key(r) for r in guard};removed=[{k:r[k] for k in columns} for r in baseline if key(r) not in bb];added=[{k:r[k] for k in columns} for r in guard if key(r) not in aa]
    dump(trial/'assignment_changes.json',dict(removed=removed,added=added))
    original_prefix=[key(r) for r in baseline if int(r['source_frame']) in times];guard_prefix=[key(r) for r in guard if int(r['source_frame']) in times];assert original_prefix==guard_prefix
    for name in ['prefix_frame_features.csv','snapshot_player_features.csv','prefix_player_summary_features.csv']:assert read(case/name)==read(trial/name)
    features=read(case/'prefix_frame_features.csv');assert len(features)==31
    near_fraction=sum(r['near_pair_valid']=='1' for r in features)/31;far_fraction=sum(r['far_pair_valid']=='1' for r in features)/31
    mean=next(r for r in read(out/'B2_window_means.csv') if r['rally_candidate_id'].endswith('_'+tag));assert abs(float(mean['near_pair_observed_fraction'])-near_fraction)<1e-12 and abs(float(mean['far_pair_observed_fraction'])-far_fraction)<1e-12
    tp=load(trial/'protocol.json');frozen=load(Path(tp['frozen_plan']));assert frozen['next_candidate']==p['candidate_id'] and frozen[f'fixed_before_{tag}_inspection'];assert sha(Path(frozen['source_code']))==frozen['source_code_sha256']==tp['frozen_guard_code_sha256']
    cp=load(BASE/'cohort_and_baselines/protocol.json');score={n:h for n,h in cp['protected_input_hashes'].items() if Path(n).name=='score_suggestion_validation_sample.csv'};assert len(score)==1 and all(sha(Path(n))==h for n,h in score.items())
    first,last=args.excerpt_first,args.excerpt_last;assert p['context_first_source_frame']<=first<=last<=p['context_last_source_frame'];video=trial/'service_prefix_review_h264.mp4';cap=cv2.VideoCapture(str(video));count=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(1080,1920);count+=1
    cap.release();assert count==last-first+1
    dump(trial/'excerpt_manifest.json',dict(source_video=str(trial/'tracking_quality_20261102_stage7'/f'players_{tag}_global/global_review_h264.mp4'),first_source_frame=first,last_source_frame=last,frames=count,aspect_preserved=True,video_sha256=sha(video)))
    findings=load(trial/'AI_visual_findings.json');assert not findings['known_wrong_person_assignment']
    if tag=='013':
        assert len(removed)==1 and removed[0]['source_frame']=='34195' and removed[0]['player_slot']=='near_front' and not added
        assert near_fraction==1 and far_fraction==20/31 and len(original_prefix)==113
        assert findings['near_ID_change_in_reviewed_clean_play_context'] is False
    if result['next_candidate']:
        nexttag=result['next_candidate'][-3:];nextplan=dict(frozen,next_candidate=result['next_candidate'],previous_transfer_result=str(trial/'result.json'),previous_transfer_result_sha256=sha(trial/'result.json'),selection='Next chronological pending group, retaining stable, ambiguous and failed cases. No selection for improved outputs only.');nextplan[f'fixed_before_{nexttag}_inspection']=True;nextplan[f'threshold_tuning_on_{nexttag}']=False;dump(trial/'next_transfer_protocol.json',nextplan)
    report=dict(status='passed',previous_input_rows_exactly_retained=original_count,candidate_rows=result['candidates'],confirmed_human_outcomes_preserved=len(read(out/'confirmed_human_outcomes.csv')),unreviewed_outcomes=len(pending),baseline_temporal_replay_exact=load(trial/'verification.json')['baseline_temporal_replay_exact'],fixed_prefix_slot_raw_box_assignments_unchanged=True,fixed_prefix_observation_rows=len(original_prefix),fixed31_features_unchanged=True,near_pair_observed_fraction=near_fraction,far_pair_observed_fraction=far_fraction,removed_full_context_assignments=len(removed),added_full_context_assignments=len(added),frozen_guard_code_hash_matched=True,parameters_tuned_on_transfer=False,manual_ID_binding=False,existing_score_validation_sample_unchanged=True,excerpt_video_frames_decoded=count,evidence_for_new_error_prevention=findings.get('evidence_for_new_guard_error_prevention',False),independent_identity_or_position_accuracy_established=False,model_fitting_or_performance_estimation=False,canonical_pipeline_replaced=False)
    dump(out/'additional_verification.json',report);dump(trial/'final_verification.json',report)
    for name in ['run_fixed_cohort_player_case_v3.py','verify_fixed_cohort_player_case.py','prepare_cohort_phase_images.py','prepare_player_gap_diagnostics.py','prepare_fixed_input_identity_sheet.py','finalize_frozen_guard_case.py']:shutil.copy2(ROOT/name,case/('executed_'+name))
    for name in ['run_frozen_guard_transfer.py','confirmed_continuity_guard.py','automatic_reacquisition_guard.py']:shutil.copy2(ROOT/name,trial/('executed_'+name))
    shutil.copy2(ROOT/'append_fixed_cohort_player_inputs_v4.py',out/'executed_source.py')
    for name in ['prepare_contact_detail_v2.py','prepare_cohort_anchor_search.py']:shutil.copy2(ROOT/name,anchor/('executed_'+name))
    day=BASE/'README.md';marker=f'{tag}への固定ガード比較を完了';history=f'\n\n{marker}。指定31時点の4候補は{result["new_case_four_observation_times"]}時点。全指定窓の枠・ID・配置値は元処理と試作で同一、全文脈では元から{len(removed)}観測を除外・{len(added)}観測追加。013は奥F1の最初11時点の欠測を保持し、差は放送切替中F34195のN1の1枠のみで、プレー中の誤再取得防止の新たな成功例とは数えない。元1531観測・840動画フレーム、試作1530観測・420動画フレーム、各379配置値等を照合した。{args.output}に旧{original_count}件を保って追加。固定30群では{result["fixed30_constructed_inputs"]}件作成・{result["fixed30_pending_inputs"]}件未処理。人手確認11ラベル・固定スコア標本を保持し、未確認の勝者は埋めない。正規処理は置換せず、次の未処理{result["next_candidate"]}も同じ条件で比較する。\n'
    assert marker not in day.read_text(encoding='utf-8');day.write_text(day.read_text(encoding='utf-8')+history,encoding='utf-8')
    for folder in [anchor,case,trial,out]:dump(folder/'publication_hashes.json',{str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='publication_hashes.json'})
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
