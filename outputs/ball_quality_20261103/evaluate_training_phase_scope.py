"""Apply one human phase scope to all four already-frozen methods."""
import argparse,json,math,shutil
from pathlib import Path
from collections import Counter
from integrate_044_reviewed_tracks import ROOT,read,load,write,sha
from serve_training_phase_scope import verify_history
from run_manual_ball_finetune import metrics

OUT=ROOT.parent/'ball_quality_20261109/training_phase_scope_review'

def classify(g,p):
    if g['truth_visibility']!='visible':return 'output_when_not_observable' if int(p['visible']) else 'no_output_when_not_observable',None
    if not int(p['visible']):return 'missing_visible_ball',None
    error=math.hypot(float(p['x_px'])-float(g['truth_x_px']),float(p['y_px'])-float(g['truth_y_px']))
    return ('correct16px' if error<=16 else 'wrong_position'),error

def main(status=False):
    m=load(OUT/'manifest.json');phase=read(OUT/'phase_labels.csv');assert len(phase)==123
    pending=sum(r['review_status']=='unreviewed' for r in phase)
    if status or pending:
        print(json.dumps(dict(status='phase_review_pending' if pending else 'ready_to_evaluate',pending=pending,saved=123-pending,
            prior_confirmed=3,phase_conditional_accuracy=None)));return
    assert not (OUT/'evaluation_report.json').exists(),'Keep completed evaluation; revisions need a new snapshot.'
    for table in [m['protected_input_hashes'],m['immutable_artifact_hashes']]:assert all(sha(Path(p))==h for p,h in table.items())
    verify_history(OUT)
    initial={g['sample_id']:g for g in read(OUT/'initial_phase_labels.csv')}
    for p in phase:
        assert p['phase_status'] in ['play','nonplay','uncertain'] and p['reviewer_id']=='human_user' and p['reviewed_at_utc']
        if int(p['locked']):assert p==initial[p['sample_id']]
    gt={g['sample_id']:g for g in read(OUT/'frozen_ball_groundtruth.csv')};scope={p['sample_id']:p for p in phase}
    expected={'current_temporal':(66,4),'native_missing_only':(77,14),'crossscale_agreement16':(72,5),'logistic10_oof':(77,7)}
    points=[];summary=[];pairs=[];mixed=[];predictions={}
    for method in m['frozen_methods']:
        name=method['method'];path=Path(method['path']);assert sha(path)==method['sha256'];rs=read(path)
        assert len(rs)==123 and {r['sample_id'] for r in rs}==set(gt)
        old=metrics(rs)['counts'];assert (old['localized_tp_16px'],old['wrong_localization']+old['presence_fp'])==expected[name]
        mixed.append(dict(method=name,correct16px=old['localized_tp_16px'],errors=old['wrong_localization']+old['presence_fp']))
        predictions[name]={r['sample_id']:r for r in rs}
        for clip in sorted({p['clip_id'] for p in phase}):
            counts=Counter();visible=not_visible=0;errors=[]
            samples=[p for p in phase if p['clip_id']==clip]
            for p in samples:
                sid=p['sample_id'];g=gt[sid];r=predictions[name][sid];error=None
                assert all(r[k]==g[k] for k in ['candidate_id','local_frame','source_frame','truth_visibility','truth_x_px','truth_y_px'])
                if p['phase_status']=='nonplay':kind='excluded_nonplay'
                elif p['phase_status']=='uncertain':kind='excluded_phase_uncertain'
                else:
                    kind,error=classify(g,r);visible+=g['truth_visibility']=='visible';not_visible+=g['truth_visibility']=='not_visible'
                counts[kind]+=1
                if error is not None:errors.append(error)
                points.append(dict(method=name,sample_id=sid,clip_id=clip,local_frame=g['local_frame'],source_frame=g['source_frame'],
                    phase_status=p['phase_status'],truth_visibility=g['truth_visibility'],predicted_visible=r['visible'],x_px=r['x_px'],y_px=r['y_px'],outcome=kind,error_px=error if error is not None else ''))
            fields=['correct16px','wrong_position','missing_visible_ball','output_when_not_observable','no_output_when_not_observable','excluded_nonplay','excluded_phase_uncertain']
            summary.append(dict(method=name,clip_id=clip,target_times=len(samples),play_visible=visible,play_not_observable=not_visible,
                **{k:counts[k] for k in fields},visible_ball_recall16px=counts['correct16px']/visible if visible else None,
                mean_error_both_visible_px=sum(errors)/len(errors) if errors else None))
    for method in m['frozen_methods']:
        name=method['method']
        for p in phase:
            g=gt[p['sample_id']];base=predictions['current_temporal'][p['sample_id']];r=predictions[name][p['sample_id']]
            a,_=classify(g,base);b,_=classify(g,r)
            pairs.append(dict(method=name,sample_id=p['sample_id'],clip_id=p['clip_id'],phase_status=p['phase_status'],
                current_outcome=a,method_outcome=b,gain=int(p['phase_status']=='play' and b=='correct16px' and a!='correct16px'),
                loss=int(p['phase_status']=='play' and a=='correct16px' and b!='correct16px')))
    write(OUT/'phase_conditional_per_frame.csv',points);write(OUT/'phase_conditional_per_clip.csv',summary);write(OUT/'paired_changes.csv',pairs)
    totals=[]
    for method in m['frozen_methods']:
        name=method['method'];ss=[r for r in summary if r['method']==name]
        values={k:sum(r[k] for r in ss) for k in ['target_times','play_visible','play_not_observable','correct16px','wrong_position','missing_visible_ball','output_when_not_observable','no_output_when_not_observable','excluded_nonplay','excluded_phase_uncertain']}
        recalls=[r['visible_ball_recall16px'] for r in ss if r['visible_ball_recall16px'] is not None]
        totals.append(dict(method=name,**values,pooled_visible_recall16px=values['correct16px']/values['play_visible'] if values['play_visible'] else None,
            macro_clip_recall16px=sum(recalls)/len(recalls) if recalls else None,clips_with_visible_play=len(recalls)))
    snapshot=OUT/'completed_snapshot';snapshot.mkdir()
    for name in ['phase_labels.csv','annotation_events.jsonl','initial_phase_labels.csv','frozen_ball_groundtruth.csv','frozen_methods.json']:
        shutil.copy2(OUT/name,snapshot/name)
    for method in m['frozen_methods']:shutil.copy2(Path(method['path']),snapshot/Path(method['path']).name)
    for name in ['protocol.json','phase_conditional_per_frame.csv','phase_conditional_per_clip.csv','paired_changes.csv']:
        shutil.copy2(OUT/name,snapshot/name)
    report=dict(status='all_training_phases_human_reviewed_comparison_complete',phase_counts=dict(Counter(p['phase_status'] for p in phase)),
        mixed_phase_results_retained=mixed,phase_conditional_totals=totals,
        phase_labels_sha256=sha(OUT/'phase_labels.csv'),events_sha256=sha(OUT/'annotation_events.jsonl'),
        comparison_is_same_match_development=True,phase_scope_is_human_not_automatic=True,
        no_retraining_or_threshold_selection=True,no_automatic_adoption=True,canonical_tracking_changed=False,
        limitations='Phase choices cover all fixed targets equally,not error-only exclusions. Same8training clips; no independent match,full-track or automatic start/end performance claim.')
    (OUT/'evaluation_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    shutil.copy2(OUT/'evaluation_report.json',snapshot/'evaluation_report.json')
    print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--status',action='store_true');main(p.parse_args().status)
