"""Preserve completed genuine labels and verify the frozen method comparison."""
import json,shutil,math
from collections import Counter
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,load,read,write,sha

OUT=ROOT.parent/'ball_quality_20261104'
SNAP=OUT/'completed_review'
assert not SNAP.exists(),'Keep completed snapshots; revisions need a new folder.'
m=load(OUT/'manifest.json');report=load(OUT/'evaluation_report.json')
labels_path=OUT/'manual_sequence_labels.csv';labels=read(labels_path)
pred=read(OUT/'frozen_predictions.csv');cache=load(OUT/'frozen_component_cache.json')
assert len(labels)==52 and len({g['sample_id'] for g in labels})==52
assert all(g['review_status']=='human_frame_review' and g['reviewer_id']=='human_user' and g['annotated_at_utc'] for g in labels)
assert sha(labels_path)==report['labels_sha256']
assert sha(OUT/'annotation_events.jsonl')==report['events_sha256']
assert sha(OUT/'frozen_predictions.csv')==report['frozen_predictions_sha256']==m['predictions_sha256']
assert all(sha(Path(p))==h for p,h in m['frozen_artifact_hashes'].items())
assert all(sha(Path(p))==h for p,h in m['protected_input_hashes'].items())
initial=read(OUT/'initial_labels_snapshot.csv');replay={r['case_id']:r for r in initial}
events=[json.loads(s) for s in (OUT/'annotation_events.jsonl').read_text(encoding='utf-8').splitlines() if s]
seen=set()
for e in events:
    assert replay[e['case_id']]==e['before'];replay[e['case_id']]=e['after'];seen.add(e['case_id'])
assert seen=={g['case_id'] for g in labels} and all(replay[g['case_id']]==g for g in labels)
by={(p['sample_id'],p['method']):p for p in pred}
assert len(by)==104
different=[];missing=[];counts=Counter()
for g in labels:
    a,b=[by[g['sample_id'],method] for method in ['bbox_area','displacement_0.1']]
    # Compare actual outputs, rather than just equal correctness categories.
    if any(a[k]!=b[k] for k in ['visible','x_px','y_px','confidence']):different.append(g['sample_id'])
    counts[g['truth_visibility']]+=1
    if g['phase_status']=='play' and g['truth_visibility']=='visible':
        if int(a['visible']):
            assert math.hypot(float(a['x_px'])-float(g['truth_x_px']),float(a['y_px'])-float(g['truth_y_px']))<=16
        else:
            cs=cache[g['candidate_id']][g['local_frame']]
            assert not cs,'Do not call a selection failure a detection absence.'
            missing.append(dict(case_id=g['case_id'],candidate_id=g['candidate_id'],local_frame=g['local_frame'],source_frame=g['source_frame'],
                truth_x_px=g['truth_x_px'],truth_y_px=g['truth_y_px'],candidate_components_over03=len(cs),
                direct_failure_stage='no_thresholded_candidate_at_target',physical_ball_absence=False))
assert not different and counts==Counter(visible=45,not_visible=4,uncertain=3) and len(missing)==12
assert Counter(r['candidate_id'] for r in missing)==Counter(match1_motion_expand_01_003=2,match1_motion_expand_01_008=10)
changes=read(OUT/'paired_method_changes.csv');assert all(int(r['gain'])==int(r['loss'])==0 for r in changes)
SNAP.mkdir()
names=['manual_sequence_labels.csv','annotation_events.jsonl','initial_labels_snapshot.csv','frozen_predictions.csv',
    'evaluation_report.json','per_frame_evaluation.csv','per_window_summary.csv','adjacent_sampled_pairs.csv',
    'paired_method_changes.csv','manifest.json','sequence_protocol.json','reserved_windows.json','annotation_cases.json',
    'query_model_inputs.csv','frozen_component_cache.json','camera_features.json','source_image_hashes.json','overnight_handoff.json']
hashes={}
for name in names:
    source=OUT/name;target=SNAP/name;shutil.copy2(source,target);assert sha(source)==sha(target);hashes[name]=sha(target)
shutil.copy2(OUT/'README.md',SNAP/'README_before_review.md')
assert sha(SNAP/'README_before_review.md')==load(OUT/'overnight_handoff.json')['final_artifact_hashes'][str(OUT/'README.md')]
write(SNAP/'missing_visible_cases.csv',missing)
completion=dict(status='completed_review_verified',completed_date_jst='2026-11-09',reviewed_frames=52,review_events=len(events),
    label_states=dict(counts),all_phase_labels='play',methods_have_identical_outputs_on_all52=True,
    visible_center_evaluable=45,correct16px_each=33,missing_visible_each=12,wrong_position_on_visible_each=0,
    pooled_visible_recall16px=33/45,macro4window_visible_recall16px=report['macro'][0]['macro_window_recall16px'],
    gains=0,losses=0,missing_candidates_over_threshold_all12=0,
    no_output_on4_human_not_visible_times=True,
    visibility_caveat='A05/A06/A08 are uncertain near net/lines; D10-D13 are marked not_visible with occlusion notes. Neither group gives a measured ball center or establishes physical absence.',
    conclusion='No added benefit of the fixed temporal selector in these4sequences; missed visible balls have no candidate above0.3, so selection cannot recover them.',
    scope='Four deliberately small same-match development sequences,2frame stride. Pooling is descriptive only, not45independent samples or whole-match accuracy.',
    next_step='Diagnose fixed detector responses on training data before any new comparison; do not tune on B or fill unseen truth by interpolation.',
    source_files_unchanged=True,model_or_canonical_tracking_changes=False,
    historical_pending_handoff_retained=True,snapshot_hashes=hashes,script_sha256=sha(Path(__file__)))
(SNAP/'verification.json').write_text(json.dumps(completion,indent=2,ensure_ascii=False),encoding='utf-8')
(OUT/'completed_review.json').write_text(json.dumps(completion,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:v for k,v in completion.items() if k!='snapshot_hashes'},indent=2,ensure_ascii=False))
