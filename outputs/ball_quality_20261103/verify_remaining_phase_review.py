"""Verify context clips and record phase coverage without guessing labels."""
import json
from pathlib import Path
from collections import Counter
from integrate_044_reviewed_tracks import ROOT,load,read,write,sha,cv2

OUT=ROOT.parent/'ball_quality_20261109/remaining_error_phase_review'
manifest=load(OUT/'manifest.json');contexts=load(OUT/'contexts.json');protocol=load(OUT/'protocol.json')
assert sha(OUT/'executed_source.py')==protocol['script_sha256']
assert sha(OUT/'contexts.json')==manifest['contexts_sha256']
assert sha(OUT/'phase_review_labels.csv')==manifest['initial_review_labels_sha256']
assert all(r['phase_status']=='unreviewed' and not r['reviewer_id'] for r in read(OUT/'phase_review_labels.csv'))
matched=decoded=0
for c in contexts:
    assert sha(Path(c['video']))==c['video_sha256']
    frames=[Path(p) for p in load(OUT/'source_image_hashes.json') if c['candidate_id'] in p]
    fd=frames[0].parent;mp=fd.parent/'runs/threshold030_median120/run_metadata.json';meta=load(mp)
    targets={t['local_frame'] for t in c['targets']};cap=cv2.VideoCapture(meta['clip'])
    for t in range(max(targets)+1):
        ok,im=cap.read();assert ok
        if t in targets:
            ok,encoded=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok
            assert encoded.tobytes()==(fd/f'{t:06d}.jpg').read_bytes();matched+=1
    cap.release();cap=cv2.VideoCapture(c['video']);n=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape==(800,1280,3);n+=1
    cap.release();assert n==c['display_frames'];decoded+=n
assert matched==3 and decoded==1500
images=load(OUT/'source_image_hashes.json');assert all(sha(Path(p))==h for p,h in images.items())
assert all(sha(Path(p))==h for p,h in manifest['protected_input_hashes'].items())

# Include ALL123training sample times, independent of model success/failure.
train=read(ROOT/'balltrack_expanded_centers_pilot/expanded_train_labels.csv')
ready_path=next(Path(p) for p in manifest['protected_input_hashes'] if p.endswith('analysis_ready_main_camera_segments.csv'))
ready=read(ready_path);audit=[];group_counts=[]
for g in train:
    source=int(g['source_frame']);matches=[r for r in ready if r['video_id']=='match1' and r['usable_for_analysis'].lower()=='true' and int(r['start_source_frame'])<=source<=int(r['end_source_frame'])]
    phase=g.get('phase_status','')
    # The source field is blank or unknown for this dataset. Do not fill it
    # from ready-camera masks or from the fact a ball center was clicked.
    assert phase in ['', 'unknown']
    audit.append(dict(sample_id=g['sample_id'],candidate_id=g['candidate_id'],local_frame=g['local_frame'],source_frame=g['source_frame'],
        source_phase_status=phase,human_phase_confirmed=False,automated_ready_main_view=int(bool(matches)),
        camera_is_not_phase_ground_truth=True))
for cid in sorted({g['candidate_id'] for g in train}):
    group=[r for r in audit if r['candidate_id']==cid]
    group_counts.append(dict(candidate_id=cid,all_target_times=len(group),human_phase_recorded=0,
        automated_ready_main_view=sum(r['automated_ready_main_view'] for r in group)))
write(OUT/'all_training_phase_coverage.csv',audit);write(OUT/'phase_coverage_by_clip.csv',group_counts)
verification=dict(status='passed_phase_answers_pending',target_jpegs_matching_original_clip=3,
    full_context_video_frames_decoded=1500,source_images_verified=len(images),all_training_times_audited=123,
    confirmed_human_phase_labels_in_existing_training_table=0,
    source_phase_field_counts=dict(Counter(g.get('phase_status','') for g in train)),
    old_model_metrics_unchanged=True,no_error_based_exclusion_applied=True,no_new_phase_labels=True,
    needs='Human phase responses for highlighted moments. A later in-play benchmark must classify phases independently across all sampled successes and failures.',
    script_sha256=sha(Path(__file__)))
(OUT/'verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
print(json.dumps(verification,indent=2))
