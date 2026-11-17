"""Score only frozen methods on explicit human main-play center/visibility labels.

Report is separate from old validation120 and contains no automatic rule adoption.
Never train on these four queries or interpret skipped nonplay as absent ball.
"""
import argparse,json,math
from pathlib import Path
from collections import Counter
from integrate_044_reviewed_tracks import ROOT,load,read,write,sha
from run_manual_ball_finetune import metrics

OUT=ROOT/'ball_inplay_challenge'
def main(status_only=False):
    labels=read(OUT/'manual_center_labels.csv');manifest=load(OUT/'annotation_manifest.json')
    pending=[r['case_id'] for r in labels if r['truth_visibility']=='unreviewed' or r['phase_status']=='unknown']
    if status_only or pending:
        print(json.dumps(dict(cases=4,pending_cases=pending,complete=not pending,phase_counts=dict(Counter(r['phase_status'] for r in labels))),ensure_ascii=False));return
    assert not (OUT/'evaluation_report.json').exists(),'Preserve prior evaluation; do not retune in place.'
    assert all(r['reviewer_id']=='human_user' and r['annotated_at_utc'] for r in labels)
    assert sha(OUT/'frozen_method_predictions.csv')==manifest['frozen_predictions_sha256']
    assert all(sha(Path(p))==h for p,h in manifest['protected_input_hashes'].items())
    events=[json.loads(line) for line in (OUT/'annotation_events.jsonl').read_text(encoding='utf-8').splitlines() if line]
    latest={e['sample_id']:e['after'] for e in events};assert all(latest[r['sample_id']]==r for r in labels)
    usable={r['sample_id']:r for r in labels if r['phase_status']=='play' and r['phase_review_status']=='human_confirmed'
        and r['truth_visibility'] in ['visible','not_visible']}
    for r in usable.values():
        if r['truth_visibility']=='visible':
            assert all(math.isfinite(float(r[k])) for k in ['truth_x_px','truth_y_px'])
            assert 0<=float(r['truth_x_px'])<1920 and 0<=float(r['truth_y_px'])<1080
        else:assert r['truth_x_px']==r['truth_y_px']==''
    predictions=read(OUT/'frozen_method_predictions.csv');methods=list(dict.fromkeys(r['method'] for r in predictions))
    summaries=[];details=[]
    for method in methods:
        rows=[]
        for p in predictions:
            if p['method']!=method or p['sample_id'] not in usable:continue
            r=usable[p['sample_id']];rows.append(dict(p,truth_visibility=r['truth_visibility'],truth_x_px=r['truth_x_px'],truth_y_px=r['truth_y_px']))
        details.extend(rows)
        summaries.append(dict(method=method,evaluable_main_play_frames=len(rows),metrics=metrics(rows) if rows else None))
    if details:write(OUT/'main_play_fixed_method_outcomes.csv',details)
    report=dict(status='frozen_methods_evaluated_no_adoption',human_reviewed_cases=4,
        phase_counts=dict(Counter(r['phase_status'] for r in labels)),evaluable_main_play_frames=len(usable),
        excluded=[dict(case_id=r['case_id'],phase=r['phase_status'],visibility=r['truth_visibility'],ball_annotation_scope=r['ball_annotation_scope']) for r in labels if r['sample_id'] not in usable],
        methods=summaries,source_labels_sha256=sha(OUT/'manual_center_labels.csv'),frozen_predictions_sha256=sha(OUT/'frozen_method_predictions.csv'),
        target_radius_source_px=16,training_use_forbidden=True,old_validation_not_pooled=True,automatic_adoption=False,
        independent_test=False,scope='Actively selected same-match challenge, near older validation frames; no population or external accuracy claim')
    (OUT/'evaluation_report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(report,indent=2,ensure_ascii=False))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--status',action='store_true');args=parser.parse_args();main(args.status)
