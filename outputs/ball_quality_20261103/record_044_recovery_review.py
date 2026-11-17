"""Apply explicit confirmation to the11 newly shown recovery observations."""
import csv,json
from collections import Counter
from pathlib import Path
from extract_serve_start_features import ROOT,sha

OUT=ROOT/'person_044_pose_recovery'
FRAMES=[796,797,905,906,907,912,984,985,1112,1113,1114]

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main():
    dest=OUT/'user_reviewed';dest.mkdir(exist_ok=True);assert not (dest/'review.json').exists()
    regpath=ROOT/'reviewed_person_overrides.json';registry=load(regpath);key='match1_rally_window_044'
    previous=registry['rallies'][key];base=Path(previous['reviewed_output'])
    assert sha(base)==previous['reviewed_output_sha256']
    proposed=OUT/'proposed_paths.csv';render=load(OUT/'render_report.json')
    assert sha(proposed)==render['paths_sha256']
    rows=read(proposed);old=read(base)
    oldlookup={(r['source_frame'],r['player_slot']):r for r in old}
    added=[r for r in rows if (r['source_frame'],r['player_slot']) not in oldlookup]
    assert sorted(int(r['local_frame']) for r in added)==FRAMES and len(added)==11
    review_id='20261103_044_eleven_recovered_boxes_identity'
    review=dict(review_id=review_id,recorded_date='2026-11-03',reviewer='human_user',
        question='水色の追加枠は、それぞれ対応するF1・F2を囲めていますか？',
        answer='それぞれ対応するF1・F2を囲めている',
        status='human_video_confirmation_of_eleven_added_boxes',rally_candidate_id=key,
        ball_pilot_id='match1_motion_pilot_016',reviewed_local_frames=FRAMES,
        reviewed_observations=[dict(local_frame=int(r['local_frame']),source_frame=int(r['source_frame']),
            slot=r['player_slot'],raw_track_id=r['track_id']) for r in added],
        scope='Only the eleven cyan recovery boxes correspond to their F1/F2 targets in the displayed video.',
        previous_review_record=previous['review_record'],video=str((OUT/'recovery_review_h264.mp4').resolve()),
        video_sha256=sha(OUT/'recovery_review_h264.mp4'),proposed_paths_sha256=sha(proposed),
        independent_frame_identity_groundtruth=False,precise_box_or_footpoint_reviewed=False,
        remaining_gap_visibility_reviewed=False,near_player_identity_reviewed_here=False)
    (dest/'review.json').write_text(json.dumps(review,indent=2,ensure_ascii=False),encoding='utf-8')
    (dest/'previous_reviewed_reference.json').write_text(json.dumps(previous,indent=2),encoding='utf-8')
    output=[]
    for r in rows:
        q=dict(r);k=(r['source_frame'],r['player_slot'])
        if k in oldlookup:assert r==oldlookup[k]
        else:
            assert r['human_video_review_id']==''
            q.update(identity_status='human_video_recovery_correspondence_confirmed_not_independent_frame_groundtruth',
                human_video_review_id=review_id,
                human_video_identity_review='confirmed_corresponding_F1_F2_in_displayed_recovery_clip')
        output.append(q)
    full=dest/'observed_player_paths.csv';far=dest/'far_pair_path.csv'
    write(full,output);write(far,[r for r in output if r['player_slot'].startswith('far_')])
    saved=read(full);lookup={(r['source_frame'],r['player_slot']):r for r in saved}
    assert len(saved)==1588 and len(lookup)==len(saved)
    for k,r in oldlookup.items():assert lookup[k]==r
    for r in added:
        now=lookup[(r['source_frame'],r['player_slot'])]
        assert all(now[k]==r[k] for k in r if k not in ['identity_status','human_video_review_id','human_video_identity_review'])
        assert now['human_video_review_id']==review_id
    assert sha(base)==previous['reviewed_output_sha256'] and sha(proposed)==render['paths_sha256']
    for name,h in load(ROOT/'play_start_audit/inventory_summary.json')['source_hashes'].items():assert sha(Path(name))==h
    ball=load(ROOT/'reviewed_ball_overrides.json')['clips']['match1_motion_pilot_016']
    assert sha(Path(ball['reviewed_output']))==ball['reviewed_output_sha256']
    counts=Counter(int(r['source_frame']) for r in saved if 115374<=int(r['source_frame'])<=115775)
    assert sum(n==4 for n in counts.values())==326
    previous_records=previous.get('review_records',[previous['review_record']])
    updated=dict(previous,reviewed_output=str(full.resolve()),reviewed_output_sha256=sha(full),
        far_pair_output=str(far.resolve()),review_record=str((dest/'review.json').resolve()),review_id=review_id,
        review_records=previous_records+[str((dest/'review.json').resolve())],
        reviewed_recovery_local_frames=FRAMES,previous_reviewed_output=str(base),
        remaining_missing_player_times=76,independent_accuracy_established=False)
    registry['rallies'][key]=updated;regpath.write_text(json.dumps(registry,indent=2),encoding='utf-8')
    routepath=ROOT/'serve_phase_fixed_application/tracking_entry_routes.json';routes=load(routepath)
    primary=routes['primary_tracking_development_input']
    primary['preferred_person_input']='../person_044_pose_recovery/user_reviewed/observed_player_paths.csv'
    primary['person_identity_review_scope']='Previous F1/F2 clip correspondence plus11 recovered boxes confirmed; exact positions and remaining gaps unvalidated'
    routepath.write_text(json.dumps(routes,indent=2,ensure_ascii=False),encoding='utf-8')
    decision=load(OUT/'decision.json');decision.update(status='eleven_observed_recovery_boxes_human_confirmed_and_applied',
        new_rows_human_identity_review='user_confirmed_corresponding_F1_F2',
        existing_reviewed_output_replaced=True,previous_reviewed_file_preserved=True,
        reviewed_output=str(full.resolve()),review_record=str((dest/'review.json').resolve()),
        next='Use reviewed player and ball outputs for clip-level integration; preserve unresolved gaps and unvalidated ground positions.')
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
    report=dict(review_id=review_id,confirmed_added_boxes=11,previous1577_rows_preserved_exactly=True,
        all_coordinates_unchanged_from_proposed_observations=True,interpolated_positions_added=0,
        remaining_missing_player_times=76,frames_with_four_observed_boxes=326,
        original_proposed_and_previous_reviewed_files_preserved=True,reviewed_ball_output_unchanged=True,
        source_review_tables_unchanged=True,reviewed_output_sha256=sha(full),independent_accuracy_established=False)
    (dest/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
