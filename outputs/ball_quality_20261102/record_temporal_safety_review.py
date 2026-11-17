"""Persist explicit user review separately from model and frozen labels."""
import csv
import json

from protect_visible_ball_candidates import DEST
from temporal_consensus_guard import OUT,BASE,SOURCE,LABEL,read,sha
from render_temporal_safety_review import render


def main():
    audit=OUT/'suppression_audit_20261103';review_dir=audit/'user_reviewed';review_dir.mkdir(exist_ok=True)
    cid='match1_motion_pilot_002';target=318
    source=DEST/'predictions'/cid/'balltrack_predictions.csv'
    raw=read(BASE/cid/'balltrack_predictions.csv');rows=read(source)
    review=dict(review_id='20261103_match1_motion_pilot_002_frame318',candidate_id=cid,local_frame=target,
        recorded_date='2026-11-03',reviewer='user',review_scope='raw_detector_candidate_is_a_real_ball',
        question='上の002の動画で、frame318のオレンジの丸は、バウンド中の本物のボールを囲んでいますか？',
        answer='本物のボール',candidate_is_ball=True,precise_pixel_center_reviewed=False,
        coordinate_source='unchanged raw model output',raw_model_x_px=raw[target]['x_px'],raw_model_y_px=raw[target]['y_px'],
        implication='Human-confirmed false withholding by temporal consensus. Not an independent 16px localization label.',
        original_groundtruth_csv_modified=False)
    (review_dir/'review.json').write_text(json.dumps(review,indent=2,ensure_ascii=False),encoding='utf-8')
    output=[]
    for t,row in enumerate(rows):
        p=dict(row,user_candidate_review='',user_review_id='',precise_center_reviewed='')
        if t==target:
            for key in ('x_px','y_px','visible','confidence'):p[key]=raw[t][key]
            p.update(position_source='model_observed_user_confirmed_ball_candidate',user_candidate_review='is_ball',
                user_review_id=review['review_id'],precise_center_reviewed=0,
                temporal_guard_flag=0,temporal_reason='',temporal_warning='temporal_exclusion_overridden_by_user_ball_confirmation')
        else:assert all(p[k]==row[k] for k in row)
        output.append(p)
    folder=review_dir/'predictions'/cid;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'balltrack_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    assert all(str(output[target][k])==str(raw[target][k]) for k in ('x_px','y_px','visible','confidence'))
    # The reviewed candidate is a development regression, not an edit to frozen evaluation labels.
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert sha(LABEL)==protocol['label_sha256']
    rendered=render(cid,298,338,[318],prediction_root=review_dir/'predictions',output_dir=review_dir/'videos',reviewed=True)
    report=dict(review=review,video=rendered,source_experimental_sha256=sha(source),
        raw_baseline_sha256=sha(BASE/cid/'balltrack_predictions.csv'),source_labels_sha256=sha(LABEL),
        changed_frames=[318],all_other_rows_unchanged=True,
        raw_coordinate_restored=True,supplemental_review_not_used_as_independent_accuracy=True,
        blanket_temporal_filter_adoption='rejected_pending_further_design_due_confirmed_real_ball_withholding')
    (review_dir/'report.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='review'},indent=2))


if __name__=='__main__':main()
