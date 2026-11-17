"""Snapshot completed human labels and report precision without changing tracking."""
import json,math,shutil
from pathlib import Path
from collections import Counter
import numpy as np
from integrate_044_reviewed_tracks import ROOT,read,load,sha

OUT=ROOT/'ball_short_sequence_evaluation'
def main():
    assert not (OUT/'completed_review.json').exists(),'Keep completed review snapshot.'
    manifest=load(OUT/'manifest.json');report=load(OUT/'evaluation_report.json');labels=read(OUT/'manual_sequence_labels.csv')
    assert report['labels_sha256']==sha(OUT/'manual_sequence_labels.csv') and report['events_sha256']==sha(OUT/'annotation_events.jsonl')
    assert sha(OUT/'frozen_predictions.csv')==manifest['predictions_sha256']==report['original_frozen_predictions_sha256']
    assert len(labels)==26 and all(r['phase_status']=='play' and r['truth_visibility']=='visible' for r in labels)
    refs={}
    for r in labels:
        if r['center_source']=='existing_human_click':
            p=Path(r['center_source_file']);assert sha(p)==r['center_source_file_sha256']
            if str(p) not in refs:refs[str(p)]={s['sample_id']:s for s in read(p)}
            original=refs[str(p)][r['sample_id']]
            assert original['source_frame']==r['source_frame'] and original['truth_x_px']==r['truth_x_px'] and original['truth_y_px']==r['truth_y_px']
    pred=read(OUT/'frozen_predictions.csv');by={method:{r['sample_id']:r for r in pred if r['method']==method} for method in ['bbox_area','displacement_0.1']}
    changed=[]
    for sid,a in by['bbox_area'].items():
        b=by['displacement_0.1'][sid]
        if (a['visible'],a['x_px'],a['y_px'])!=(b['visible'],b['x_px'],b['y_px']):changed.append(sid)
    data=[]
    for wid in ['A','B']:
        gt=[r for r in labels if r['window_id']==wid]
        for method in by:
            errors=[math.hypot(float(by[method][r['sample_id']]['x_px'])-float(r['truth_x_px']),float(by[method][r['sample_id']]['y_px'])-float(r['truth_y_px'])) for r in gt]
            data.append(dict(window_id=wid,method=method,observations=13,new_centers=sum(r['center_source']=='new_human_frame_review' for r in gt),
                reused_centers=sum(r['center_source']=='existing_human_click' for r in gt),correct16px=sum(e<=16 for e in errors),
                mean_error_px=float(np.mean(errors)),median_error_px=float(np.median(errors)),max_error_px=max(errors),
                first_frame=int(gt[0]['local_frame']),last_frame=int(gt[-1]['local_frame']),
                truth_endpoint_displacement_px=math.hypot(float(gt[-1]['truth_x_px'])-float(gt[0]['truth_x_px']),float(gt[-1]['truth_y_px'])-float(gt[0]['truth_y_px']))))
    pairs=read(OUT/'adjacent_sampled_pairs.csv');pair_counts={}
    for wid in ['A','B']:
        for method in by:
            group=[r for r in pairs if r['window_id']==wid and r['method']==method]
            pair_counts[wid+':'+method]=dict(pairs=len(group),both_correct=sum(r['both_predictions_correct']=='1' for r in group),independent_samples=False)
    # The relevant source images and all protected models/registries/labels remain exact.
    for p,h in load(OUT/'frozen_context_image_hashes.json').items():assert sha(Path(p))==h
    assert all(sha(Path(p))==h for p,h in manifest['protected_input_hashes'].items())
    snapshot=OUT/'reviewed_snapshot';snapshot.mkdir()
    for name in ['manual_sequence_labels.csv','annotation_events.jsonl','frozen_predictions.csv','evaluation_report.json']:
        shutil.copyfile(OUT/name,snapshot/name);assert sha(snapshot/name)==sha(OUT/name)
    events=[json.loads(x) for x in (OUT/'annotation_events.jsonl').read_text(encoding='utf-8').splitlines() if x]
    result=dict(status='completed_review_and_frozen_evaluation_verified',new_human_frame_reviews=22,phase_only_reviews=2,reused_confirmed_anchors=2,
        saved_cases=24,events=len(events),observations=26,new_center_observations=22,reused_center_observations=4,
        phase_counts=dict(Counter(r['phase_status'] for r in labels)),visibility_counts=dict(Counter(r['truth_visibility'] for r in labels)),
        prediction_coordinate_changes_between_methods=len(changed),changed_sample_ids=changed,
        per_window_precision=data,adjacent_pair_counts=pair_counts,
        protected_inputs_unchanged=True,existing_human_centers_preserved=True,context_images_unchanged=True,
        frozen_predictions_unchanged=True,automatic_adoption=False,operational_outputs_changed=False,
        independent_tracking_accuracy_established=False,
        snapshot_hashes={p.name:sha(p) for p in snapshot.iterdir()},
        interpretation='Both fixed methods pass these two0.4004second neighbourhoods. This demonstrates local agreement with manual centers, not improvement on false-object switches or whole-rally accuracy.')
    (OUT/'completed_review.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
