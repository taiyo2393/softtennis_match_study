"""Append-only auxiliary hypotheses; native coordinates always take precedence."""
import sys,json,math,shutil
from pathlib import Path
from collections import defaultdict,Counter
import cv2
from run_fixed_cohort_player_case_v3 import ROOT,BASE,SOURCE,read,write,sha,dump
from run_near_prefix_initialization_trial import load
from run_body_part_localization_probe import overlap

OUT=BASE/'native_body_auxiliary_bundle'
NATIVE=BASE/'camera_valid_near_prefix_trial'
BODY=BASE/'partial_body_identity_trial'
SLOTS=['near_front','near_rear','far_rear','far_front']
MODES=['unmasked_torso','exclude_near_rectangles']


def rect(r):return [float(r[k]) for k in ['x1','y1','x2','y2']]


def resolve(native,body,other_native):
    """Return explicit evidence state; body geometry can never become foot xy."""
    proposed=body is not None and str(body['body_ID_candidate'])=='1'
    conflicts=[]
    if proposed:
        bb=json.loads(body['body_box_json'])
        conflicts=[s for s,r in other_native.items() if overlap(bb,rect(r),union=True)>=.5]
    if native is not None:
        return dict(evidence_state='native_box_candidate',identity_candidate_present=1,
            native_position_proxy_present=1,analysis_x=native['bottom_center_x'],analysis_y=native['bottom_center_y'],
            native_raw_id=native['track_id'],body_candidate_available=int(proposed),
            body_conflicting_other_native_slots=';'.join(conflicts),position_reason='native_bbox_bottom_image_proxy_not_ground_truth')
    if proposed and not conflicts:
        return dict(evidence_state='body_only_identity_hypothesis',identity_candidate_present=1,
            native_position_proxy_present=0,analysis_x='',analysis_y='',native_raw_id='',
            body_candidate_available=1,body_conflicting_other_native_slots='',position_reason='no_native_position_do_not_substitute_body_or_pose_coordinates')
    return dict(evidence_state='unsupported_body_conflict' if conflicts else 'unsupported',identity_candidate_present=0,
        native_position_proxy_present=0,analysis_x='',analysis_y='',native_raw_id='',body_candidate_available=int(proposed),
        body_conflicting_other_native_slots=';'.join(conflicts),position_reason='native_position_missing')


def prepare():
    assert not OUT.exists();p=load(BODY/'protocol.json')
    files=[Path(__file__),NATIVE/'protocol.json',NATIVE/'publication_hashes.json',BODY/'protocol.json',
           BODY/'publication_hashes.json',BODY/'all_fixed_body_hypotheses.csv',ROOT/'run_body_part_localization_probe.py',
           SOURCE/'data/metadata/videos.csv',Path.cwd()/'README.md',Path.cwd()/'DATASET_IMPORT_README.md']
    files.extend(NATIVE/f'case_{e["tag"]}/candidate_native_observations.csv' for e in p['cases'])
    files.extend(n for n in (BASE/'current22_player_inputs').iterdir() if n.is_file())
    protected={str(n):sha(n) for n in files}
    OUT.mkdir()
    dump(OUT/'protocol.json',dict(cases=p['cases'],modes=MODES,
        combination='Keep every existing native observation, ID, coordinate and missing-position definition. Only where native is missing, retain an auxiliary body ID hypothesis if its pose box does not overlap another existing native slot at IoU>=.5. All proposals and conflicts remain in output.',
        conflict_threshold_source='Existing0.5 native/pose correspondence gate reused, no calibration or sweep.',
        geometry_features='Image-space near/far pair distance only when both native proxies exist; body-only states never create pair geometry. Not a court/ground metric.',
        selection='All310 frozen times and4 slots, both already-completed body variants; no per-case choice.',
        candidate_bundle_not_canonical=True,offline_future_006_body_evidence=True,
        independent_accuracy_established=False,
        inventory_scope='Source data/raw, registered videos.csv and cwd dataset/videos only. No claim of searching every PC folder. Original/edited copies are not automatically independent matches.',
        inventory_sample='Sorted dataset/videos files: first,middle,last representative frame only. Illustrative scope check, not full validation.',
        policy_fixed_before_combined_outputs=True,protected_input_hashes=protected))
    print('Frozen combined output rules; existing native rows always retained.',flush=True)


def build():
    p=load(OUT/'protocol.json');assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    bodies={(r['tag'],r['policy'],int(r['source_frame']),r['player_slot']):r for r in read(BODY/'all_fixed_body_hypotheses.csv')}
    rows=[];frames=[];pairs=[];reports=[]
    for e in p['cases']:
        tag=e['tag'];native=defaultdict(dict)
        for r in read(NATIVE/f'case_{tag}/candidate_native_observations.csv'):native[int(r['source_frame'])][r['player_slot']]=r
        for mode in MODES:
            local=[]
            for f in e['requested_input_frames']:
                group=native[f];merged={}
                for slot in SLOTS:
                    body=bodies.get((tag,mode,f,slot));n=group.get(slot)
                    evidence=resolve(n,body,{s:r for s,r in group.items() if s!=slot})
                    row=dict(tag=tag,policy=mode,source_frame=f,player_slot=slot,**evidence,
                        native_source=str(NATIVE/f'case_{tag}/candidate_native_observations.csv') if n else '',
                        body_source=str(BODY/f'case_{tag}/{mode}_fixed_body_hypotheses.csv') if body else '',
                        body_pose_index=body['pose_index'] if body else '',
                        body_box_json=body['body_box_json'] if body else '',
                        uses_future_body_anchor=int(evidence['evidence_state']=='body_only_identity_hypothesis' and body['uses_future_common_anchor']=='1'),
                        independent_identity_verified=0,independent_position_verified=0)
                    rows.append(row);local.append(row);merged[slot]=row
                    if n:assert row['analysis_x']==n['bottom_center_x'] and row['analysis_y']==n['bottom_center_y'] and row['native_raw_id']==n['track_id']
                    else:assert row['analysis_x']==row['analysis_y']==''
                native_count=sum(r['native_position_proxy_present'] for r in merged.values())
                body_count=sum(r['evidence_state']=='body_only_identity_hypothesis' for r in merged.values())
                frames.append(dict(tag=tag,policy=mode,source_frame=f,native_position_proxies=native_count,
                    supplemental_body_ID_hypotheses=body_count,total_identity_candidates=native_count+body_count,
                    four_native_position_proxies=int(native_count==4),four_identity_candidates=int(native_count+body_count==4)))
                for side in ['near','far']:
                    side_rows=[r for s,r in merged.items() if s.startswith(side+'_')]
                    available=all(r['native_position_proxy_present'] for r in side_rows)
                    distance=math.hypot(float(side_rows[0]['analysis_x'])-float(side_rows[1]['analysis_x']),float(side_rows[0]['analysis_y'])-float(side_rows[1]['analysis_y'])) if available else ''
                    pairs.append(dict(tag=tag,policy=mode,source_frame=f,side=side,native_pair_geometry_available=int(available),
                        native_pair_distance_image_px=distance,identity_candidates=sum(r['identity_candidate_present'] for r in side_rows),
                        missing_geometry_reason='' if available else 'one_or_both_native_position_proxies_missing_body_is_not_a_position'))
            report=dict(tag=tag,policy=mode,states=dict(Counter(r['evidence_state'] for r in local)))
            reports.append(report);print(report,flush=True)
    write(OUT/'combined_evidence_states.csv',rows);write(OUT/'frame_availability_not_accuracy.csv',frames);write(OUT/'native_pair_geometry.csv',pairs)
    # Both feature tables must be exactly equal except the body support count.
    numeric={(r['tag'],r['source_frame'],r['side'],r['policy']):r for r in pairs}
    for tag,f,side,mode in list(numeric):
        if mode!=MODES[0]:continue
        a=numeric[tag,f,side,MODES[0]];b=numeric[tag,f,side,MODES[1]]
        assert a['native_pair_geometry_available']==b['native_pair_geometry_available'] and a['native_pair_distance_image_px']==b['native_pair_distance_image_px']
    dump(OUT/'result.json',dict(cases=reports,combined_slot_rows=len(rows),frame_policy_rows=len(frames),pair_geometry_rows=len(pairs),
        existing_native_IDs_and_coordinates_unchanged=True,body_created_position_coordinates=0,
        body_created_geometry_features=0,current22_changed=False,independent_accuracy_established=False))
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())


def inventory():
    records=[]
    for r in read(SOURCE/'data/metadata/videos.csv'):
        records.append(dict(kind='registered_research_video',video_id=r['video_id'],path=str(SOURCE/'data/raw'/r['local_filename']),
            declared_match=r['video_id'],independent_match_established=0,notes='Current development match; not independent validation.'))
    registered={r['path'] for r in records}
    for n in sorted((SOURCE/'data/raw').glob('*.mp4')):
        if str(n) not in registered:records.append(dict(kind='unregistered_source_file',video_id='',path=str(n),declared_match='',
            independent_match_established=0,notes='Filename indicates original/alternate match1 file; independence is not established by filename, encoding or a different hash.'))
    videos=sorted((Path.cwd()/'dataset/videos').glob('*.mp4'))
    indexes=sorted(set([0,len(videos)//2,len(videos)-1])) if videos else []
    for i in indexes:
        n=videos[i];cap=cv2.VideoCapture(str(n));count=round(cap.get(7));fps=cap.get(5);w=round(cap.get(3));h=round(cap.get(4))
        sample=max(0,count//2);cap.set(1,sample);ok,im=cap.read();cap.release();assert ok
        scale=min(640/w,640/h);out=(round(w*scale),round(h*scale));name=f'dataset_sample_{i:04}.jpg'
        assert cv2.imwrite(str(OUT/name),cv2.resize(im,out,interpolation=cv2.INTER_AREA))
        records.append(dict(kind='dataset_video_representative_only',video_id=n.stem,path=str(n),declared_match='',independent_match_established=0,
            notes='Shot/subject collection. Representative image reviewed separately; do not assume four-player match applicability.',
            frames=count,fps=fps,width=w,height=h,sample_source_frame=sample,sample_image=name))
    write(OUT/'local_video_inventory.csv',records)
    dump(OUT/'inventory_scope.json',dict(registered_match_count=len(read(SOURCE/'data/metadata/videos.csv')),
        source_raw_mp4_files=len(list((SOURCE/'data/raw').glob('*.mp4'))),dataset_video_files=len(videos),
        representative_dataset_files_sampled=len(indexes),confirmed_additional_independent_matches=0,
        every_PC_directory_searched=False,match_identity_or_player_independence_inferred_from_file_count=False,
        next_data_needed='A separate doubles match source path or URL; ideally different players. Do not automatically download or relabel an alternate copy as a new match.'))


if __name__=='__main__':{'prepare':prepare,'build':build,'inventory':inventory}[sys.argv[1]]()
