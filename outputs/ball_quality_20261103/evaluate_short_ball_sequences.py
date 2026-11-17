"""Evaluate already frozen predictions only after24 frame reviews are saved."""
import argparse,json,math
from pathlib import Path
from collections import Counter
from integrate_044_reviewed_tracks import ROOT,load,read,write,sha
OUT=ROOT/'ball_short_sequence_evaluation'
def distance(a,b):return math.hypot(float(a['x_px'])-float(b['truth_x_px']),float(a['y_px'])-float(b['truth_y_px']))
def main(status_only=False):
    m=load(OUT/'manifest.json');labels=read(OUT/'manual_sequence_labels.csv');cases=load(OUT/'annotation_cases.json')
    assert len(labels)==26 and sha(OUT/'frozen_predictions.csv')==m['predictions_sha256']
    pending=[r['case_id'] for r in labels if r['review_status']=='unreviewed']
    if status_only:print(json.dumps(dict(pending=len(pending),saved=24-len(pending),reused_confirmed_anchors=2)));return
    assert not pending,'Human review is incomplete; do not infer missing labels.'
    assert not (OUT/'evaluation_report.json').exists(),'Keep original evaluation. Review revisions require a separate snapshot.'
    assert all(sha(Path(p))==h for p,h in m['protected_input_hashes'].items())
    assert sha(OUT/'annotation_cases.json')==m['cases_sha256']
    events=[json.loads(x) for x in (OUT/'annotation_events.jsonl').read_text(encoding='utf-8').splitlines() if x];replay={};case_by={c['case_id']:c for c in cases}
    for e in events:
        cid=e['case_id'];assert cid in case_by
        if cid in replay:assert replay[cid]==e['before']
        else:assert e['before']['review_status']=='unreviewed'
        replay[cid]=e['after'];assert e['after']['reviewer_id']=='human_user' and e['after']['annotated_at_utc']
    lookup={r['case_id']:r for r in labels}
    assert set(replay)==set(case_by) and all(lookup[k]==v for k,v in replay.items())
    for c in cases:
        assert sha(Path(c['source_image']))==c['source_image_sha256']
        if c['center_reused']:assert lookup[c['case_id']]['truth_x_px']==c['reused_x'] and lookup[c['case_id']]['truth_y_px']==c['reused_y']
    windows=load(OUT/'windows.json');pred=read(OUT/'frozen_predictions.csv');point_results=[];summaries=[];pairs=[]
    for w in windows:
        gt=sorted([r for r in labels if r['window_id']==w['window_id']],key=lambda r:int(r['local_frame']))
        for method in sorted({r['method'] for r in pred}):
            predictions={r['sample_id']:r for r in pred if r['method']==method};classified=[];counts=Counter()
            for r in gt:
                p=predictions[r['sample_id']];visible=int(p['visible']);error='';category='excluded_phase'
                if r['phase_status']=='play':
                    if r['truth_visibility']=='visible':
                        if visible:error=distance(p,r);category='correct16px' if error<=16 else 'wrong_position'
                        else:category='missing_visible_ball'
                    elif r['truth_visibility']=='not_visible':category='output_when_not_observable' if visible else 'no_output_when_not_observable'
                    else:category='excluded_ball_uncertain'
                counts[category]+=1;classified.append(category)
                point_results.append(dict(window_id=w['window_id'],method=method,sample_id=r['sample_id'],local_frame=r['local_frame'],source_frame=r['source_frame'],
                    phase_status=r['phase_status'],truth_visibility=r['truth_visibility'],truth_x_px=r['truth_x_px'],truth_y_px=r['truth_y_px'],
                    predicted_visible=visible,x_px=p['x_px'],y_px=p['y_px'],error_px=error,outcome=category,center_source=r['center_source']))
            longest=run=0
            for category in classified:
                run=run+1 if category in ['wrong_position','output_when_not_observable'] else 0;longest=max(longest,run)
            summaries.append(dict(window_id=w['window_id'],candidate_id=w['candidate_id'],method=method,span_sec=w['span_sec'],stride=2,
                **{k:counts[k] for k in ['correct16px','wrong_position','missing_visible_ball','output_when_not_observable','no_output_when_not_observable','excluded_phase','excluded_ball_uncertain']},
                longest_sampled_wrong_run=longest,run_span_sec=max(0,longest-1)*2/w['fps']))
            for i,(a,b) in enumerate(zip(gt,gt[1:])):
                assert int(b['local_frame'])-int(a['local_frame'])==2
                known=all(r['phase_status']=='play' and r['truth_visibility']=='visible' for r in [a,b]);pa,pb=predictions[a['sample_id']],predictions[b['sample_id']]
                pairs.append(dict(window_id=w['window_id'],method=method,first_frame=a['local_frame'],last_frame=b['local_frame'],
                    both_visible_gt_and_play=int(known),both_predictions_correct=int(known and classified[i]=='correct16px' and classified[i+1]=='correct16px'),
                    both_outputs_present=int(int(pa['visible']) and int(pb['visible'])),
                    interpretation='Overlapping stride2 pair, not independent; no interpolation across uncertainty.'))
    write(OUT/'per_frame_evaluation.csv',point_results);write(OUT/'per_window_summary.csv',summaries);write(OUT/'adjacent_sampled_pairs.csv',pairs)
    report=dict(status='human_labelled_short_sequences_evaluated',windows=windows,summary=summaries,
        labels_sha256=sha(OUT/'manual_sequence_labels.csv'),events_sha256=sha(OUT/'annotation_events.jsonl'),
        original_frozen_predictions_sha256=m['predictions_sha256'],protected_inputs_unchanged=True,
        new_training=False,automatic_adoption=False,interpolation=0,
        limitations='Two deliberately small neighbourhoods in known same-match validation clips, including4reused centers; not independent external testing or full-rally accuracy. Adjacent pairs overlap; missing is unknown position, not physical absence.')
    (OUT/'evaluation_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(summaries,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--status',action='store_true');a=p.parse_args();main(a.status)
