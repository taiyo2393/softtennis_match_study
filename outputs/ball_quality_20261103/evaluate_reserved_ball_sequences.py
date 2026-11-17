"""Evaluate frozen52time comparison only after genuine human annotations exist."""
import argparse, json, math
from pathlib import Path
from collections import Counter
from integrate_044_reviewed_tracks import ROOT, read, load, write, sha

OUT=ROOT.parent/'ball_quality_20261104'

def classify(g,p):
    if g['phase_status']=='preparation_or_ended':return 'excluded_nonplay',None
    if g['phase_status']!='play':return 'excluded_phase_uncertain',None
    visible=int(p['visible'])
    if g['truth_visibility']=='not_visible':
        return ('output_when_not_observable' if visible else 'no_output_when_not_observable'),None
    if g['truth_visibility']!='visible':return 'excluded_ball_uncertain',None
    if not visible:return 'missing_visible_ball',None
    e=math.hypot(float(p['x_px'])-float(g['truth_x_px']),float(p['y_px'])-float(g['truth_y_px']))
    return ('correct16px' if e<=16 else 'wrong_position'),e

def ratio(n,d):return n/d if d else None

def summarize(labels,predictions,windows):
    points,summary,pairs,paired=[],[],[],[]
    methods=sorted({p['method'] for p in predictions})
    assert methods==['bbox_area','displacement_0.1']
    by={(p['sample_id'],p['method']):p for p in predictions}
    assert len(by)==len(predictions)==len(labels)*2
    for w in windows:
        gt=sorted([g for g in labels if g['window_id']==w['window_id']],key=lambda g:int(g['local_frame']))
        assert len(gt)==13
        for method in methods:
            categories=[];errors=[];counts=Counter()
            for g in gt:
                p=by[g['sample_id'],method];kind,e=classify(g,p);counts[kind]+=1;categories.append(kind)
                if e is not None:errors.append(e)
                points.append(dict(window_id=w['window_id'],method=method,sample_id=g['sample_id'],local_frame=g['local_frame'],
                    source_frame=g['source_frame'],phase_status=g['phase_status'],truth_visibility=g['truth_visibility'],
                    truth_x_px=g['truth_x_px'],truth_y_px=g['truth_y_px'],predicted_visible=p['visible'],
                    x_px=p['x_px'],y_px=p['y_px'],outcome=kind,error_px=e if e is not None else ''))
            runs={}
            for name,states in [('correct',{'correct16px'}),('missing',{'missing_visible_ball'}),
                                ('wrong',{'wrong_position','output_when_not_observable'})]:
                longest=run=0
                for kind in categories:run=run+1 if kind in states else 0;longest=max(longest,run)
                runs['longest_'+name+'_sampled_run']=longest
                runs['longest_'+name+'_sampled_span_sec']=max(0,longest-1)*2/w['fps']
            true_visible=counts['correct16px']+counts['wrong_position']+counts['missing_visible_ball']
            output_eligible=counts['correct16px']+counts['wrong_position']+counts['output_when_not_observable']
            keys=['correct16px','wrong_position','missing_visible_ball','output_when_not_observable',
                  'no_output_when_not_observable','excluded_nonplay','excluded_phase_uncertain','excluded_ball_uncertain']
            summary.append(dict(window_id=w['window_id'],candidate_id=w['candidate_id'],method=method,
                sampled_span_sec=24/w['fps'],**{k:counts[k] for k in keys},
                visible_ball_recall16px=ratio(counts['correct16px'],true_visible),
                localized_output_fraction16px=ratio(counts['correct16px'],output_eligible),
                mean_error_both_visible_px=sum(errors)/len(errors) if errors else None,**runs))
            for i,(a,b) in enumerate(zip(gt,gt[1:])):
                assert int(b['local_frame'])-int(a['local_frame'])==2
                known=all(g['phase_status']=='play' and g['truth_visibility']=='visible' for g in [a,b])
                pa,pb=by[a['sample_id'],method],by[b['sample_id'],method]
                present=all(int(p['visible']) for p in [pa,pb]);displacement=''
                if known and present:
                    dx=(float(pb['x_px'])-float(pa['x_px']))-(float(b['truth_x_px'])-float(a['truth_x_px']))
                    dy=(float(pb['y_px'])-float(pa['y_px']))-(float(b['truth_y_px'])-float(a['truth_y_px']))
                    displacement=math.hypot(dx,dy)
                pairs.append(dict(window_id=w['window_id'],method=method,first_local_frame=a['local_frame'],last_local_frame=b['local_frame'],
                    verified_visible_play_pair=int(known),both_outputs_present=int(present),
                    both_correct16px=int(known and categories[i]==categories[i+1]=='correct16px'),displacement_error_px=displacement))
        for g in gt:
            kinds={m:classify(g,by[g['sample_id'],m])[0] for m in methods}
            a,b=kinds['bbox_area'],kinds['displacement_0.1']
            paired.append(dict(window_id=w['window_id'],sample_id=g['sample_id'],local_frame=g['local_frame'],
                bbox_area_outcome=a,temporal_outcome=b,gain=int(b=='correct16px' and a!='correct16px'),
                loss=int(a=='correct16px' and b!='correct16px')))
    return points,summary,pairs,paired

def self_test():
    g=dict(phase_status='play',truth_visibility='visible',truth_x_px='100',truth_y_px='100')
    p=dict(visible='1',x_px='116',y_px='100')
    assert classify(g,p)==('correct16px',16)
    assert classify(g,dict(p,x_px='117'))==('wrong_position',17)
    assert classify(g,dict(p,visible='0',x_px='',y_px=''))==('missing_visible_ball',None)
    assert classify(dict(g,phase_status='uncertain'),p)==('excluded_phase_uncertain',None)
    assert classify(dict(g,phase_status='preparation_or_ended'),p)==('excluded_nonplay',None)
    assert classify(dict(g,truth_visibility='uncertain'),p)==('excluded_ball_uncertain',None)
    assert classify(dict(g,truth_visibility='not_visible'),p)==('output_when_not_observable',None)
    assert ratio(0,0) is None
    # Already completed older human-labelled sequence is a regression fixture only.
    old=ROOT/'ball_short_sequence_evaluation'
    outputs=summarize(read(old/'manual_sequence_labels.csv'),read(old/'frozen_predictions.csv'),load(old/'windows.json'))
    assert len(outputs[0])==52 and len(outputs[2])==48
    assert all(r['correct16px']==13 and r['longest_correct_sampled_run']==13 for r in outputs[1])
    assert not any(r['gain'] or r['loss'] for r in outputs[3])
    print('Passed synthetic phase/visibility/missing/radius cases and completed older26time regression. No research labels written.')

def main(status_only=False):
    m=load(OUT/'manifest.json');labels=read(OUT/'manual_sequence_labels.csv')
    assert len(labels)==52
    pending=[g['case_id'] for g in labels if g['review_status']=='unreviewed']
    if status_only or pending:
        print(json.dumps(dict(status='human_annotation_pending' if pending else 'ready_to_evaluate',pending=len(pending),saved=52-len(pending),accuracy=None if pending else 'not_computed')))
        return
    assert not (OUT/'evaluation_report.json').exists(), 'Keep completed evaluation; revisions need a new snapshot.'
    assert all(sha(Path(p))==h for p,h in m['frozen_artifact_hashes'].items())
    assert all(sha(Path(p))==h for p,h in m['protected_input_hashes'].items())
    initial=read(OUT/'initial_labels_snapshot.csv')
    assert sha(OUT/'initial_labels_snapshot.csv')==m['initial_labels_sha256']
    replay={g['case_id']:dict(g) for g in initial};events_path=OUT/'annotation_events.jsonl'
    events=[json.loads(s) for s in events_path.read_text(encoding='utf-8').splitlines() if s]
    for e in events:
        assert replay[e['case_id']]==e['before']
        assert e['after']['reviewer_id']=='human_user' and e['after']['annotated_at_utc']
        for k in ['sample_id','candidate_id','local_frame','source_frame','window_id','case_id']:
            assert e['before'][k]==e['after'][k]
        replay[e['case_id']]=e['after']
    assert all(replay[g['case_id']]==g for g in labels)
    for c in load(OUT/'annotation_cases.json'):
        assert sha(Path(c['source_image']))==c['source_image_sha256']
    results=summarize(labels,read(OUT/'frozen_predictions.csv'),load(OUT/'reserved_windows.json'))
    for name,rows in zip(['per_frame_evaluation.csv','per_window_summary.csv','adjacent_sampled_pairs.csv','paired_method_changes.csv'],results):write(OUT/name,rows)
    macro=[]
    for method in ['bbox_area','displacement_0.1']:
        values=[r['visible_ball_recall16px'] for r in results[1] if r['method']==method and r['visible_ball_recall16px'] is not None]
        macro.append(dict(method=method,windows_with_visible_gt=len(values),macro_window_recall16px=sum(values)/len(values) if values else None))
    report=dict(status='completed_human_review_evaluation',summary=results[1],macro=macro,
        labels_sha256=sha(OUT/'manual_sequence_labels.csv'),events_sha256=sha(events_path),
        frozen_predictions_sha256=m['predictions_sha256'],new_training=False,operational_changes=False,
        scope='Four reserved same-match development sequences, not independent match validation. Stride2 spans0.4004sec; pairs overlap. No full rally, physical absence, 3D or causal claim.')
    (OUT/'evaluation_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--status',action='store_true');p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.self_test:self_test()
    else:main(args.status)
