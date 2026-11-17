"""Check case010 packaging, archive executed code and hash published outputs."""
import csv,hashlib,json,shutil
from pathlib import Path
import cv2

ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'ball_quality_20261110'
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    case=BASE/'cohort_tracking/case_010';old=BASE/'current13_player_inputs';out=BASE/'current14_player_inputs';anchor=BASE/'cohort_anchor_proposals/010'
    assert not (out/'additional_verification.json').exists()
    for folder in [case,out]:
        p=load(folder/'protocol.json');assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
        result=load(folder/'case_result.json' if folder==case else folder/'result.json')
        assert all(sha(folder/n)==h for n,h in result['artifact_hashes'].items())
        assert load(folder/'verification.json')['status']=='passed'
    files=['B1_server_side.csv','B2_snapshot.csv','B2_window_means.csv','diagnostic_snapshot_masks_only.csv','diagnostic_window_masks_only.csv','human_outcomes.csv','source_snapshot_features.csv','source_window_summary_features.csv','rally_metadata.csv','processing_interventions.csv','outcome_label_status.csv']
    for name in files:
        before=read(old/name);after=read(out/name);assert len(before)==13 and len(after)==14
        amap={r['rally_candidate_id']:r for r in after}
        assert all(amap[r['rally_candidate_id']]==r for r in before),name
    assert (old/'confirmed_human_outcomes.csv').read_bytes()==(out/'confirmed_human_outcomes.csv').read_bytes()
    assert len(read(out/'confirmed_human_outcomes.csv'))==11
    registry=read(out/'human_outcomes.csv');assert {r['rally_candidate_id'][-3:] for r in registry if not r['winner_camera_side']}=={'008','009','010'}
    before=read(old/'fixed30_processing_progress.csv');after=read(out/'fixed30_processing_progress.csv')
    assert [r for r in before if not r['candidate_id'].endswith('_010')]==[r for r in after if not r['candidate_id'].endswith('_010')]
    prefix=read(case/'prefix_frame_features.csv');assert len(prefix)==31
    p=load(case/'protocol.json');assert p['requested_input_frames']==list(range(23728,23789,2))
    assert [f for f,r in zip(p['requested_input_frames'],prefix) if r['far_pair_valid']=='0']==[23770,23772,23774,23776]
    assert all(r['near_pair_valid']=='1' for r in prefix) and prefix[-1]['far_pair_valid']=='1'
    gates=load(case/'input_gap_prefilter_diagnosis.json')['findings']
    missing=[r for r in gates if r['source_frame'] in [23770,23772,23774,23776]]
    assert len(missing)==4 and all(r['native_detection_exists'] and not r['prefilter_passes']['negative_appearance'] and all(v for k,v in r['prefilter_passes'].items() if k!='negative_appearance') for r in missing)
    cap=cv2.VideoCapture(str(case/'prefix_review_h264.mp4'));count=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(1080,1920);count+=1
    cap.release();assert count==189
    dump(case/'prefix_review_manifest.json',dict(source_video=str(case/'tracking_quality_20261102_stage7/players_010_global/global_review_h264.mp4'),first_source_frame=23660,last_source_frame=23848,frames=count,video_sha256=sha(case/'prefix_review_h264.mp4'),scope='Short excerpt of existing offline path, not a new tracker or point-boundary GT.'))
    report=dict(status='passed',previous13_input_rows_exactly_retained=True,confirmed_human_outcomes_preserved=11,unreviewed_outcomes=3,new_case_requested_times=31,new_case_four_observation_times=27,all_four_missing_requested_times_have_native_raw41_detections=True,retained_negative_appearance_veto_reproduced_at_all_four_missing_times=True,threshold_or_path_changed=False,AI_identity_seed_binding_added=False,AI_contact_proposal_frozen_before_tracking=True,independent_identity_or_position_accuracy_established=False,offline_future_context_in_association_documented=True,prediction_evaluation_performed=False,excerpt_video_frames_decoded=count)
    dump(out/'additional_verification.json',report)
    for name in ['run_fixed_cohort_player_case_v3.py','verify_fixed_cohort_player_case.py','prepare_cohort_phase_images.py','prepare_player_gap_diagnostics.py','diagnose_case010_input_gap.py','finalize_case010.py']:
        shutil.copy2(ROOT/name,case/('executed_'+name))
    shutil.copy2(ROOT/'append_fixed_cohort_player_inputs_v4.py',out/'executed_source.py')
    shutil.copy2(ROOT/'prepare_contact_detail.py',anchor/'executed_prepare_contact_detail.py')
    shutil.copy2(ROOT/'prepare_cohort_anchor_search.py',anchor/'executed_prepare_cohort_anchor_search.py')
    for folder in [anchor,case,out]:
        dump(folder/'publication_hashes.json',{str(p.relative_to(folder)):sha(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name!='publication_hashes.json'})
    print(json.dumps(report,ensure_ascii=False))
if __name__=='__main__':main()
