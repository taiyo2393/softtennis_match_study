"""Immutable case-level diagnostic of already frozen predictions, no tuning."""
import json,math
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,load,read,write,sha

OUT=ROOT/'ball_inplay_challenge'
def main():
    assert not (OUT/'post_review_decision.json').exists(),'Preserve completed diagnostics.'
    report=load(OUT/'evaluation_report.json');manifest=load(OUT/'annotation_manifest.json')
    labels=read(OUT/'manual_center_labels.csv');cases={r['sample_id']:r for r in load(OUT/'annotation_cases.json')}
    assert sha(OUT/'manual_center_labels.csv')==report['source_labels_sha256']
    assert sha(OUT/'annotation_cases.json')==manifest['cases_sha256']
    assert report['evaluable_main_play_frames']==4 and all(r['phase_status']=='play' and r['truth_visibility']=='visible' for r in labels)
    predictions=read(OUT/'frozen_method_predictions.csv');lookup={(r['sample_id'],r['method']):r for r in predictions}
    methods=list(dict.fromkeys(r['method'] for r in predictions));rows=[];correct_by_method={m:set() for m in methods}
    for r in labels:
        c=cases[r['sample_id']]
        assert sha(Path(c['source_image']))==c['source_image_sha256']
        assert int(r['source_frame'])==c['source_frame']==c['clip_first_source_frame']+int(r['local_frame'])
        assert r['candidate_id']==c['candidate_id'] and int(r['local_frame'])==c['local_frame']
        result=dict(case_id=r['case_id'],sample_id=r['sample_id'],candidate_id=r['candidate_id'],local_frame=r['local_frame'],
            source_frame=r['source_frame'],phase_status='human_confirmed_play',truth_x_px=r['truth_x_px'],truth_y_px=r['truth_y_px'])
        for method in methods:
            p=lookup[r['sample_id'],method];distance=math.hypot(float(p['x_px'])-float(r['truth_x_px']),float(p['y_px'])-float(r['truth_y_px']))
            good=distance<=16
            result.update({f'{method}_x_px':p['x_px'],f'{method}_y_px':p['y_px'],f'{method}_error_px':distance,f'{method}_correct16px':int(good)})
            if good:correct_by_method[method].add(r['case_id'])
        rows.append(result)
    write(OUT/'reviewed_case_comparison.csv',rows)
    base=correct_by_method['bbox_area'];comparisons={}
    for method in methods[1:]:
        comparisons[method]=dict(correct=len(correct_by_method[method]),new_correct_cases=sorted(correct_by_method[method]-base),
            original_correct_cases_lost=sorted(base-correct_by_method[method]),
            errors_px={r['case_id']:r[f'{method}_error_px'] for r in rows})
    assert base=={'A','D'} and all(c['new_correct_cases']==['B','C'] and c['original_correct_cases_lost']==['A'] for c in comparisons.values())
    assert all(sha(Path(p))==h for p,h in manifest['protected_input_hashes'].items())
    decision=dict(status='four_main_play_queries_evaluated_original_retained',human_confirmed_play_targets=4,
        source_labels_sha256=report['source_labels_sha256'],frozen_predictions_sha256=manifest['frozen_predictions_sha256'],
        baseline_correct_cases=sorted(base),baseline_correct_count=len(base),alternative_comparisons=comparisons,
        mean_source_pixel_error= {m['method']:m['metrics']['mean_error_both_visible_px'] for m in report['methods']},
        operational_method='bbox_area',operational_threshold=.3,operational_model_and_outputs_changed=False,
        reason='Each score rule recovers B/C but loses previously correct A with a573px displacement. Single-frame score replacement does not satisfy nonregression.',
        new_audit_evaluation=False,audit_reason='Challenge already reveals original-correct loss; no audit-based rule selection',
        training_use_forbidden=True,post_label_parameter_tuning=False,
        interpretation='This is4 actively selected human-reviewed frames, not general tracking accuracy or trajectory correctness. All4 balls visible; no evaluation of absent-ball false output.',
        next='Use training clips to design any temporal candidate selector, fixing rules before further evaluation. Preserve these4 labels for evaluation only; no new human input needed for the completed comparison.',
        protected_original_labels_weights_reviewed_outputs_unchanged=True)
    (OUT/'post_review_decision.json').write_text(json.dumps(decision,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(decision,indent=2,ensure_ascii=False))

if __name__=='__main__':main()
