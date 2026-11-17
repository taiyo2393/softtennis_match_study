"""One fixed training-free YOLO sports-ball check on reviewed candidate patches."""
import json,os
from pathlib import Path
import numpy as np,torch
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2

OUT=ROOT/'ball_pretrained_sports_ball_probe'
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'serve_start_pilot/ultralytics_config')

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists(),'No tuning in-place.'
    cp=ROOT/'ball_candidate_cross_rally/user_reviewed/all_reviewed_candidate_cases.csv'
    cases=read(cp);assert len(cases)==36
    weight=SOURCE/'models/yolo11m.pt'
    protected={str(p):sha(p) for p in [cp,ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json']}
    protocol=dict(status='specified_before_inference',model='existing local YOLO11m COCO sports-ball class',
        weights=str(weight),weights_sha256=sha(weight),training_on_study_cases=False,
        inputs='Native96x96 candidate patch, OpenCV BGR, network imgsz640, candidate at48/48; no generated detail',
        class_id=32,expected_class_name='sports ball',confidence_min=.1,nms_iou=.7,
        candidate_match='Predicted sports-ball box center at most8 native source-image pixels from candidate patch center',
        candidate_match_radius_native_px=8.,
        abstention_meaning='A missing nearby sports-ball detection is not proof that the original candidate is false; evaluated as an experimental keep/drop gate only',
        score_semantics='Detector confidence is not a calibrated correctness probability',
        selection='All36 already-reviewed candidates, selected same-match development data, not independent accuracy',
        no_parameter_search=True,no_fine_tuning=True,no_download=True,automatic_adoption=False,
        corpus_sha256=sha(cp),script_sha256=sha(Path(__file__)),protected_input_hashes=protected)
    dump(OUT/'protocol.json',protocol)
    from ultralytics import YOLO
    net=YOLO(str(weight));assert net.names[32]=='sports ball'
    rows=[];all_boxes=[]
    for first in range(0,len(cases),8):
        batch=cases[first:first+8];images=[]
        for r in batch:
            assert sha(Path(r['patch96_path']))==r['patch96_sha256']
            im=cv2.imread(r['patch96_path']);assert im.shape[:2]==(96,96);images.append(im)
        results=net.predict(images,imgsz=640,conf=.1,iou=.7,classes=[32],device=0,verbose=False,save=False)
        for r,pred in zip(batch,results):
            nearby=[]
            for k,box in enumerate(pred.boxes.data.cpu().numpy()):
                x1,y1,x2,y2,score,cls=map(float,box)
                assert int(cls)==32
                cx=(x1+x2)/2;cy=(y1+y2)/2;distance=float(np.hypot(cx-48,cy-48))
                all_boxes.append(dict(case_id=r['case_id'],box_index=k,x1_patch=x1,y1_patch=y1,x2_patch=x2,y2_patch=y2,
                    sports_ball_confidence=score,distance_from_candidate_native_px=distance,within8px=int(distance<=8)))
                if distance<=8:nearby.append((score,distance))
            rows.append(dict(case_id=r['case_id'],ball_pilot_id=r['ball_pilot_id'],local_frame=r['local_frame'],
                candidate_is_ball=int(r['label']=='confirmed_ball_enclosed'),sports_ball_nearby_found=int(bool(nearby)),
                nearby_confidence=max(q[0] for q in nearby) if nearby else '',
                sports_ball_boxes_anywhere_in_patch=len(pred.boxes),physical_ball_visibility_confirmed=0))
    write(OUT/'predictions.csv',rows)
    if all_boxes:write(OUT/'sports_ball_boxes.csv',all_boxes)
    real=[r for r in rows if r['candidate_is_ball']];false=[r for r in rows if not r['candidate_is_ball']]
    report=dict(status='training_free_candidate_check_complete_not_adopted',
        reviewed_real_total=len(real),reviewed_real_nearby_detections=sum(r['sports_ball_nearby_found'] for r in real),
        reviewed_false_total=len(false),reviewed_false_nearby_detections=sum(r['sports_ball_nearby_found'] for r in false),
        missed_real_candidate_ids=[r['case_id'] for r in real if not r['sports_ball_nearby_found']],
        accepted_false_candidate_ids=[r['case_id'] for r in false if r['sports_ball_nearby_found']],
        reviewed_inputs_and_source_annotations_unchanged=True,weights_unchanged=True,
        automatic_adoption=False,independent_accuracy_established=False,
        warning='No detection cannot be used as an automatic false label. Selected36 candidate checks do not establish sports-ball tracking precision or recall.')
    assert all(sha(Path(p))==h for p,h in protected.items()) and sha(weight)==protocol['weights_sha256']
    for p,h in load(ROOT/'play_start_audit/inventory_summary.json')['source_hashes'].items():assert sha(Path(p))==h
    dump(OUT/'decision.json',report);print(json.dumps(report,indent=2))

if __name__=='__main__':main()
