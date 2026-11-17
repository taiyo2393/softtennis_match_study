"""Record explicit video-level F1/F2 confirmation without inventing coordinates."""
import csv,json
from pathlib import Path
from extract_serve_start_features import ROOT,sha

OUT=ROOT/'person_044_reinit'

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    dest=OUT/'user_reviewed';dest.mkdir(exist_ok=True)
    assert not (dest/'review.json').exists()
    render=json.loads((OUT/'render_report.json').read_text())
    original=OUT/'observed_player_paths.csv';far=OUT/'far_pair_path.csv'
    assert sha(original)==render['paths_sha256']
    review_id='20261103_044_far_pair_video_identity'
    review=dict(review_id=review_id,recorded_date='2026-11-03',reviewer='human_user',
        question='F1（紫）・F2（黄）は、途中や欠落後の再取得時に入れ替わらず、それぞれ同じ選手を示していますか？',
        answer='それぞれ同じ選手を示している',
        status='human_video_level_identity_confirmation',
        rally_candidate_id='match1_rally_window_044',ball_pilot_id='match1_motion_pilot_016',
        reviewed_slots=['far_rear','far_front'],reviewed_local_frames=render['source_local_frames'],
        identity_names=dict(F1='far_rear, initial far server',F2='far_front, initial far partner'),
        scope='Displayed F1/F2 correspondence, including reacquisition; applies only to the displayed clip.',
        video=str((OUT/'person_review_h264.mp4').resolve()),video_sha256=sha(OUT/'person_review_h264.mp4'),
        original_paths_sha256=sha(original),original_far_paths_sha256=sha(far),
        independent_per_frame_identity_groundtruth=False,position_accuracy_measured=False,
        near_player_identity_reviewed_here=False,missing_positions_confirmed_or_filled=False)
    (dest/'review.json').write_text(json.dumps(review,indent=2,ensure_ascii=False),encoding='utf-8')
    checked={}
    for source in [original,far]:
        before=read(source);after=[]
        for row in before:
            reviewed=row['player_slot'].startswith('far_')
            q=dict(row,original_identity_status=row['identity_status'],
                human_video_review_id=review_id if reviewed else '',
                human_video_identity_review='confirmed_same_player_in_displayed_clip' if reviewed else 'not_reviewed_by_this_confirmation',
                independent_frame_identity_groundtruth=0)
            if reviewed:q['identity_status']='human_video_correspondence_confirmed_not_independent_frame_groundtruth'
            after.append(q)
        target=dest/source.name;write(target,after);saved=read(target)
        assert len(before)==len(saved)
        for a,b in zip(before,saved,strict=True):
            assert all(a[k]==b[k] for k in a if k!='identity_status')
            assert b['original_identity_status']==a['identity_status']
            assert bool(b['human_video_review_id'])==a['player_slot'].startswith('far_')
        checked[source.name]=dict(rows=len(saved),positions_and_missing_frames_unchanged=True,sha256=sha(target))
    assert sha(original)==review['original_paths_sha256'] and sha(far)==review['original_far_paths_sha256']
    proto=json.loads((OUT/'protocol.json').read_text())
    assert sha(Path(proto['raw_detector_source']))==proto['raw_source_sha256']
    source_hashes=json.loads((ROOT/'play_start_audit/inventory_summary.json').read_text())['source_hashes']
    for path,expected in source_hashes.items():assert sha(Path(path))==expected
    ball=json.loads((ROOT/'reviewed_ball_overrides.json').read_text())['clips']['match1_motion_pilot_016']
    assert sha(Path(ball['reviewed_output']))==ball['reviewed_output_sha256']
    (dest/'verification.json').write_text(json.dumps(dict(review_id=review_id,outputs=checked,
        original_detection_and_path_files_unchanged=True,source_review_tables_unchanged=True,
        reviewed_ball_output_unchanged=True,interpolated_positions_added=0,
        independent_accuracy_established=False),indent=2),encoding='utf-8')
    override_path=ROOT/'reviewed_person_overrides.json'
    registry=json.loads(override_path.read_text()) if override_path.exists() else dict(scope='Per-clip human video correspondence; distinct from independent ID/position groundtruth',rallies={})
    registry['rallies']['match1_rally_window_044']=dict(ball_pilot_id='match1_motion_pilot_016',
        reviewed_output=str((dest/'observed_player_paths.csv').resolve()),
        reviewed_output_sha256=sha(dest/'observed_player_paths.csv'),
        far_pair_output=str((dest/'far_pair_path.csv').resolve()),
        review_record=str((dest/'review.json').resolve()),review_id=review_id,
        reviewed_slots=['far_rear','far_front'],output_local_frames=render['source_local_frames'],
        coordinate_system='3840x2160 source pixels; multiply0.5 for cached1920x1080',
        original_six_clip_baseline_unchanged=True,independent_accuracy_established=False)
    override_path.write_text(json.dumps(registry,indent=2),encoding='utf-8')
    decision=json.loads((OUT/'decision.json').read_text())
    decision.update(status='far_pair_displayed_identity_human_confirmed_registered_as_reviewed_clip',
        human_identity_review='human_video_level_F1_F2_same_player_correspondence_confirmed',
        review_record=str((dest/'review.json').resolve()),reviewed_output=str((dest/'observed_player_paths.csv').resolve()),
        next='Diagnose visible-player misses in remaining gaps; keep missing positions empty and position accuracy unmeasured.')
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
    print(json.dumps(dict(review_id=review_id,outputs=checked,reviewed_slots=review['reviewed_slots'],
        original_and_ball_outputs_unchanged=True),indent=2))

if __name__=='__main__':main()
