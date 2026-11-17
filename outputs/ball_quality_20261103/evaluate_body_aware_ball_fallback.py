"""One conservative appearance abstention on NEW fallback additions only.

Original detections are never filtered. Person/racket overlap means abstain
from filling a gap, not a semantic label that the candidate is a false ball.
This refinement follows a known validation failure and remains development.
"""
import os,json
from pathlib import Path
from collections import defaultdict
import numpy as np
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2
from evaluate_observed_ball_fallback import P,decide,camera_mask,evaluate_samples
from run_manual_ball_finetune import metrics

OUT=ROOT/'balltrack_observed_fallback/body_aware'
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'serve_start_pilot/ultralytics_config')

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def point_overlaps_boxes(candidate,boxes):
    x,y=float(candidate['x_px']),float(candidate['y_px'])
    return [i for i,b in enumerate(boxes) if b['xyxy'][0]<=x<=b['xyxy'][2] and b['xyxy'][1]<=y<=b['xyxy'][3]]

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists(),'Keep each diagnostic refinement.'
    parent=OUT.parent;training=ROOT/'balltrack_manual_finetune_pilot'
    earlier=load(parent/'protocol.json');manifest=load(training/'data_manifest.json')
    labels=read(training/'validation_labels.csv');base=ROOT.parent/'tracking_quality_20261102/ball_aligned'
    cameras=ROOT.parent/'ball_quality_20261102/temporal_guard/camera_features'
    model=SOURCE/'models/yolo11m.pt';frozen=earlier['protected_input_hashes']
    assert all(sha(Path(p))==h for p,h in frozen.items())
    protocol=dict(status='fixed_before_detection_and_refined_selection',
        refinement_reason='The previously evaluated short-gap policy added a wrong racket-area candidate at019/325. This is a validation-informed development refinement, not preregistered external testing.',
        inheritance='All earlier temporal/camera/threshold rules unchanged; original detections preserved',
        detector='existing local YOLO11m COCO',weights_sha256=sha(model),classes=[0,38],expected_names=['person','tennis racket'],
        imgsz=1920,confidence=.25,nms_iou=.7,rectangle_expansion_px=0,
        action='If a NEW fallback candidate center lies inside any detected person/racket rectangle, leave the original gap blank. No original observation is removed.',
        interpretation='Overlap is uncertainty, not proof of false ball. Real balls at contact/occlusion may also remain missing.',
        selection='Same baseline-correct preservation and baseline error-count<=3; max validationTP,then fewer errors/fills; original wins exact tie',
        validation_only=True,audit_used_for_selection=False,hyperparameter_sweep_on_detector=False,automatic_adoption=False,
        script_sha256=sha(Path(__file__)),source_policy_protocol_sha256=sha(parent/'protocol.json'),protected_hashes=frozen)
    dump(OUT/'protocol.json',protocol)
    raw_tables={cid:read(base/cid/'balltrack_predictions.csv') for cid in sorted({r['candidate_id'] for r in labels})}
    masks={cid:camera_mask(load(cameras/(cid+'.json'))['changed_fraction']) for cid in raw_tables}
    all_candidates={}
    jobs=set()
    for threshold in P['thresholds']:
        table={(r['candidate_id'],int(r['local_frame'])):r for r in read(training/'calibration'/f'validation_threshold{int(threshold*100):02d}.csv')}
        all_candidates[threshold]=table
        for r in labels:
            cid=r['candidate_id'];t=int(r['local_frame'])
            if decide(raw_tables[cid],t,table[cid,t],masks[cid])['decision']=='add_observed_finetuned_candidate':jobs.add((cid,t))
    from ultralytics import YOLO
    detector=YOLO(str(model));assert detector.names[0]=='person' and detector.names[38]=='tennis racket'
    boxes_by={};records=[]
    for cid,t in sorted(jobs):
        path=SOURCE/'data/balltrack/racketvision/match1'/cid/'frames'/f'{t:06d}.jpg'
        im=cv2.imread(str(path));assert im is not None
        pred=detector.predict(im,imgsz=1920,conf=.25,iou=.7,classes=[0,38],device=0,verbose=False,save=False)[0]
        boxes=[dict(xyxy=[float(q) for q in b[:4]],confidence=float(b[4]),class_id=int(b[5]),class_name=detector.names[int(b[5])]) for b in pred.boxes.data.cpu().numpy()]
        boxes_by[cid,t]=boxes;records.append(dict(candidate_id=cid,local_frame=t,source_image_sha256=sha(path),boxes=boxes))
    dump(OUT/'validation_object_boxes.json',records)
    baseline=[]
    for r in labels:
        p=raw_tables[r['candidate_id']][int(r['local_frame'])]
        baseline.append(dict(sample_id=r['sample_id'],truth_visibility=r['truth_visibility'],truth_x_px=r['truth_x_px'],truth_y_px=r['truth_y_px'],
            x_px=p['x_px'],y_px=p['y_px'],visible=p['visible']))
    bm=metrics(baseline);original_correct={r['sample_id'] for r in baseline if r['truth_visibility']=='visible' and int(r['visible']) and
        np.hypot(float(r['x_px'])-float(r['truth_x_px']),float(r['y_px'])-float(r['truth_y_px']))<=16}
    best=None;best_key=(80,-3,0);summary=[];diagnostics=[]
    for threshold,table in all_candidates.items():
        _,rows,ds=evaluate_samples(labels,raw_tables,table,masks,threshold)
        for r,d in zip(rows,ds,strict=True):
            cid=r['candidate_id'];t=int(r['local_frame']);hits=[]
            if d['decision']=='add_observed_finetuned_candidate':
                hits=point_overlaps_boxes(table[cid,t],boxes_by[cid,t])
                if hits:
                    original=raw_tables[cid][t]
                    r.update(x_px=original['x_px'],y_px=original['y_px'],visible=original['visible'],confidence=original['confidence'],output_source='unchanged_original')
                    d['decision']='body_or_racket_overlap_abstain'
            d['overlapping_box_ids']=';'.join(map(str,hits));diagnostics.append(d)
        m=metrics(rows);ids={r['sample_id'] for r in rows if r['truth_visibility']=='visible' and int(r['visible']) and
            np.hypot(float(r['x_px'])-float(r['truth_x_px']),float(r['y_px'])-float(r['truth_y_px']))<=16}
        error=m['counts']['wrong_localization']+m['counts']['presence_fp'];added=[r for r in rows if r['output_source']=='observed_finetuned_missing_fallback']
        feasible=not(original_correct-ids) and error<=3;scores=(m['counts']['localized_tp_16px'],-error,-len(added))
        if feasible and scores>best_key:best_key=scores;best=dict(threshold=threshold,metrics=m,rows=rows,added=added)
        summary.append(dict(threshold=threshold,localized_tp_16px=m['counts']['localized_tp_16px'],errors=error,additions=len(added),
            original_correct_frames_lost=len(original_correct-ids),feasible=feasible,added_samples=[r['sample_id'] for r in added]))
        write(OUT/f'validation_threshold{int(threshold*100):02d}.csv',rows)
    write(OUT/'validation_decisions.csv',diagnostics)
    result=dict(status='body_aware_validation_complete',selected_policy='body_aware_observed_fallback' if best else 'original',
        selected_threshold=best['threshold'] if best else None,baseline=bm,selected_validation=best['metrics'] if best else bm,
        threshold_results=summary,validation_added_samples=[r['sample_id'] for r in best['added']] if best else [],
        automatic_adoption=False,audit_used_for_selection=False,original_observations_preserved=True,
        warning='Selected same-match validation examples; this is a retrospective refined policy. It can miss true balls near bodies and rackets. No global removal rule is adopted.')
    assert all(sha(Path(p))==h for p,h in frozen.items()) and sha(model)==protocol['weights_sha256']
    dump(OUT/'selection.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
