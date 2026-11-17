"""Publish a separated native-track/position-policy bundle and blinded reference pilot."""
import random,secrets
from pathlib import Path
from collections import defaultdict
import cv2
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,write,sha,dump,FPS
import run_position_proxy_support_head_independent as support
from build_rally_analysis_prototype import pair_features,summarize_prefix
from verify_player_prefix_expansion import calculate_frame,calculate_summary,close

OUT=BASE/'identity_position_bundle'
POLICIES=['retain_native_proxy','withhold_body_warning_proxy']
REF_SLOTS=['near_left','near_right','far_left','far_right']

def main():
    previous=BASE/'position_proxy_support_trial';assert support.load(previous/'final_verification.json')['status']=='passed';p=support.load(support.OUT/'protocol.json');OUT.mkdir();protected=dict(p['protected_input_hashes'])
    for stage in [previous,BASE/'torso_temporal_association_trial',BASE/'current22_player_inputs']:
        for n in ['publication_hashes.json','final_verification.json','protocol.json']:
            if (stage/n).exists():protected[str(stage/n)]=sha(stage/n)
    for n in [Path(__file__),ROOT/'build_rally_analysis_prototype.py',ROOT/'verify_player_prefix_expansion.py']:protected[str(n)]=sha(n)
    for e in p['cases']:
        tag=e['tag']
        for n in ['position_proxy_support.csv','fixed_prefix_position_support_complete.csv']:protected[str(support.OUT/f'case_{tag}'/n)]=sha(support.OUT/f'case_{tag}'/n)
    dump(OUT/'protocol.json',dict(cases=p['cases'],source_video=p['source_video'],source_video_sha256=p['source_video_sha256'],source_clock=p['source_clock'],policies=POLICIES,purpose='Separate native candidate identity observations from image-position inclusion decisions. Preserve all fixed times and unknowns; no canonical replacement.',position='Native bbox-bottom image proxy only, not measured ground contact or model ankle coordinates.',withheld_rows='Body-warning variant blanks only analysis proxy coordinates. Candidate slot/raw ID/native rectangle remain present. No-warning and unassessed rows are not certified correct.',unassessed='Explicit pose_evaluated/body evidence and warning states retained. Never equate flag0 with correctness.',fixed_windows_unchanged=True,canonical_pipeline_replaced=False,independent_GT=False,protected_input_hashes=protected))
    observations=[];policies=[];frame_features=[];snapshots=[];summaries=[];comparison=[];checked=0
    for e in p['cases']:
        tag=e['tag'];native=support.native_table(e);quality=read(support.OUT/f'case_{tag}/position_proxy_support.csv');quality+=read(support.OUT/f'case_{tag}/fixed_prefix_position_support_complete.csv');q={(int(r['source_frame']),r['player_slot']):r for r in quality};times=sorted(set(range(e['initialization']['frame'],e['last']+1))|set(e['requested_input_frames']));kept_by_policy={policy:defaultdict(dict) for policy in POLICIES}
        for f in times:
            for slot in support.SLOTS:
                r=native[f].get(slot);warning=q[f,slot];present=int(r is not None)
                observation=dict(candidate_id=f'match1_rally_window_{tag}',source_frame=f,timestamp_sec=f/FPS,player_slot=slot,camera_side=slot.split('_')[0],candidate_box_present=present,native_track_id=r['track_id'] if r else '',x1=r['x1'] if r else '',y1=r['y1'] if r else '',x2=r['x2'] if r else '',y2=r['y2'] if r else '',native_proxy_x=r['bottom_center_x'] if r else '',native_proxy_y=r['bottom_center_y'] if r else '',identity_status='tracked_candidate_not_independent_GT' if r else 'no_native_ID_observation',independent_identity_certified=0)
                observations.append(observation)
                for policy in POLICIES:
                    held=bool(present and policy=='withhold_body_warning_proxy' and warning['geometry_body_overlap_risk']=='1');available=bool(present and not held)
                    row=dict(observation,position_policy=policy,analysis_proxy_present=int(available),analysis_proxy_x=r['bottom_center_x'] if available else '',analysis_proxy_y=r['bottom_center_y'] if available else '',position_disposition='native_missing' if not present else 'withheld_by_experimental_body_warning' if held else 'native_proxy_retained_unvalidated',position_warning_state=warning['position_support_state'],body_overlap_warning=int(warning['geometry_body_overlap_risk']),rectangle_overlap_warning=int(warning['geometry_box_overlap_risk']),pose_frame_evaluated=int(warning['pose_frame_evaluated']),own_pose_matched=int(warning['own_pose_matched']),other_pose_bodies_evaluated=warning['other_pose_bodies_evaluated'],position_certified=0)
                    policies.append(row)
                    if available:kept_by_policy[policy][f][slot]=r
        for policy in POLICIES:
            frames=[]
            for f in e['requested_input_frames']:
                kept=kept_by_policy[policy][f];frame=dict(rally_candidate_id=f'match1_rally_window_{tag}',relative_source_frame=f-e['requested_input_frames'][0],**pair_features(kept));expected=calculate_frame(frame['rally_candidate_id'],kept_by_policy[policy],f,frame['relative_source_frame'])
                for k,value in expected.items():
                    if k!='rally_candidate_id':close(frame[k],value);checked+=1
                frames.append(frame);frame_features.append(dict(position_policy=policy,source_frame=f,**frame))
            snapshot,summary,_=summarize_prefix(tag,[dict(f,ball_observation_present=0,ball_x_norm='',ball_y_norm='') for f in frames]);es,em=calculate_summary(frames)
            for actual,expected in [(snapshot,es),(summary,em)]:
                for k,v in expected.items():close(actual[k],v);checked+=1
            snapshots.append(dict(position_policy=policy,**snapshot));summaries.append(dict(position_policy=policy,**summary));comparison.append(dict(tag=tag,position_policy=policy,requested_times=31,four_candidate_proxy_times=sum(f['near_pair_valid'] and f['far_pair_valid'] for f in frames),identity_paths_changed=0,not_accuracy=1))
    write(OUT/'native_identity_observations.csv',observations);write(OUT/'position_policy_observations.csv',policies);write(OUT/'fixed_prefix_frame_features.csv',frame_features);write(OUT/'fixed_prefix_snapshot_features.csv',snapshots);write(OUT/'fixed_prefix_summary_features.csv',summaries);write(OUT/'policy_output_availability_not_accuracy.csv',comparison)
    # Freeze a tiny reference pilot before exposing any new reference coordinates.
    pilot=OUT/'reference_pilot';pilot.mkdir();rng=random.Random(20261110);selected={};uniform=[]
    for e in sorted(p['cases'],key=lambda e:e['tag']):
        f=rng.choice(e['requested_input_frames']);k=e['tag'],f;selected[k]=dict(tag=e['tag'],source_frame=f,sampling_groups=['uniform_one_per_context'],target_slot=None);uniform.append(dict(tag=e['tag'],source_frame=f,selection='Single rng.choice of the original31 requested times; no replacement for missing predictions or difficult images.'))
    for tag,f,slot in [('002',2303,'far_front'),('017',42132,'far_front'),('017',42138,'far_front'),('019',46396,'far_rear')]:
        k=tag,f
        if k in selected:selected[k]['sampling_groups'].append('known_body_warning_challenge');selected[k]['target_slot']=slot
        else:selected[k]=dict(tag=tag,source_frame=f,sampling_groups=['known_body_warning_challenge'],target_slot=slot)
    entries=list(selected.values());rng.shuffle(entries)
    for i,e in enumerate(entries,1):e['image_id']=f'P{i:03d}'
    dump(pilot/'private_sampling_protocol.json',dict(selection_fixed_before_new_reference_labels=True,rng='Python random.Random(20261110),one requested time per sorted context,then fixed4 warning examples,then deterministic shuffle',uniform_samples=uniform,items=entries,reference_slots=REF_SLOTS,reference_definition='Approximate image point at midpoint x between the two shoes and the lower shoe-bottom y, if those locations can be placed from the image. Jumping does not itself invalidate a visible shoe. Mark unplaceable/absent/uncertain instead of guessing hidden feet.',prediction_blinding='Viewer serves original images and manual labels only; no model IDs, keypoints, quality flags, predictions or sampling groups exposed.',same_match_development=True,reviewer_may_have_seen_prior_examples=True,independent_reference_coordinates_pending=True,not_independent_match_test=True,primary_reporting='Uniform10 contexts only, grouped by context and near/far side. Known-warning4 examples reported separately and never pooled into representative accuracy.',evaluation='Within-side unordered point-set matching; report localization and pair arrangement error only when both reference positions and both predicted proxies exist. Report all incomplete/unplaceable/reference-missing denominators separately. No ID-continuity score, contact-time task, coordinate-error pass threshold, model fitting, outcome prediction or tactical-cause inference.',new_manual_labels_created=False))
    images=pilot/'images';images.mkdir();public=[];cap=cv2.VideoCapture(p['source_video'])
    for e in entries:
        cap.set(1,e['source_frame']);ok,im=cap.read();assert ok and im.shape[:2]==(2160,3840) and round(cap.get(1))==e['source_frame']+1;path=images/(e['image_id']+'.png');assert cv2.imwrite(str(path),im);e['image_sha256']=sha(path);public.append(dict(image_id=e['image_id'],width=3840,height=2160,image_file=path.name))
    cap.release();dump(pilot/'private_sampling_manifest.json',dict(items=entries,source_video_sha256=p['source_video_sha256'],original_frames_no_predictions=True));dump(pilot/'public_manifest.json',dict(title='画像上の選手位置・小規模確認',slots=REF_SLOTS,items=public,instruction='足元が分かる選手だけ、左右の靴の中央x・低い方の靴底yを目安に点を置きます。厳密な接地判定は不要です。隠れている位置は推測せず「位置不明」を選んでください。'))
    assert not (pilot/'human_reference_labels.json').exists();assert all(sha(Path(n))==h for n,h in protected.items());report=dict(status='separated_bundle_constructed_reference_pilot_pending',contexts=10,frame_slot_rows=len(observations),position_policy_rows=len(policies),fixed_prefix_frame_feature_rows=len(frame_features),feature_values_independently_checked=checked,reference_images=len(entries),uniform_context_samples=10,known_warning_challenge_samples=4,human_reference_labels_created=False,current22_changed=False,position_filter_adopted=False,accuracy_established=False);dump(OUT/'construction_result.json',report);print(report,flush=True)

if __name__=='__main__':main()
