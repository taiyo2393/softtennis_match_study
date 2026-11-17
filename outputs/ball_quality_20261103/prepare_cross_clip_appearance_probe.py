"""Prepare existing safety cases and four unlabelled cross-clip candidates.

Existing300-row labels are NOT copied as new confirmed labels. The new four
cases were selected to include apparently successful/failed model outputs;
they are a selected development check, not representative test sampling.
"""
import json
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2,np

OUT=ROOT/'ball_candidate_cross_rally'
NEW=[('A',391),('B',503),('C',615),('D',1623)]

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    assert not (OUT/'case_manifest.json').exists();OUT.mkdir(exist_ok=True)
    base=ROOT.parent/'tracking_quality_20261102/ball_aligned'
    prior=ROOT.parent/'ball_quality_20261102/temporal_guard'
    rp=prior/'suppression_audit_20261103/user_reviewed/review.json';real=load(rp)
    hp=prior/'user_reported_cases.json';hands=load(hp)
    assert real['candidate_is_ball'] is True and real['local_frame']==318
    assert hands['candidate_id']=='match1_motion_pilot_018' and hands['local_frames']==[208,229,230]
    inventory={r['ball_pilot_id']:r for r in read(ROOT/'play_start_audit/ball_pilot_start_inventory.csv')}
    inputs={str(p):sha(p) for p in [rp,hp,ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json']}
    cases=[]
    for suffix,t,label,role,display,evidence,scope in [
        ('002',318,'confirmed_ball_enclosed','prior_regression','',rp,'preparation_or_between_attempts_not_main_rally'),
        *[('018',t,'confirmed_not_ball_enclosed','prior_regression','',hp,'hand_false_detection_camera_transition_nearby') for t in hands['local_frames']],
        *[('008',t,'pending_user_confirmation','new_cross_clip_candidate',name,None,'phase_not_human_confirmed_at_target') for name,t in NEW]]:
        cid=f'match1_motion_pilot_{suffix}';input_path=base/cid/'balltrack_predictions.csv'
        inputs[str(input_path)]=sha(input_path);b=read(input_path)[t]
        assert int(b['local_frame'])==t and int(b['visible'])==1
        if suffix=='002':assert b['x_px']==real['raw_model_x_px'] and b['y_px']==real['raw_model_y_px']
        row=dict(case_id=f'{suffix}_{t}',display_label=display,ball_pilot_id=cid,local_frame=t,
            source_frame=int(b['source_frame']),rally_candidate_id=inventory[cid]['rally_window_candidate_id'],
            label=label,role=role,x_px=b['x_px'],y_px=b['y_px'],split_group=f'legacy_match1:{cid}',
            phase_status=scope,evidence_record=str(evidence) if evidence else '',
            label_precision='candidate_encloses_ball_not_precise_center',inference_view='unchanged_full_frame_raw')
        image=SOURCE/'data/balltrack/racketvision/match1'/cid/'frames'/f'{t:06d}.jpg'
        im=cv2.imread(str(image));assert im is not None and im.shape[:2]==(1080,1920)
        row['original_image']=str(image);row['original_image_sha256']=sha(image)
        for size,dirname in [(96,'patch96'),(224,'context224')]:
            (OUT/dirname).mkdir(exist_ok=True);left=round(float(b['x_px']))-size//2;top=round(float(b['y_px']))-size//2
            assert left>=0 and top>=0 and left+size<=1920 and top+size<=1080
            patch=im[top:top+size,left:left+size];p=OUT/dirname/(row['case_id']+'.png')
            assert cv2.imwrite(str(p),patch) and np.array_equal(cv2.imread(str(p)),patch)
            row[dirname+'_path']=str(p.resolve());row[dirname+'_sha256']=sha(p)
        cases.append(row)
    write(OUT/'cases.csv',cases)
    assert len(cases)==8 and sum(r['label']=='pending_user_confirmation' for r in cases)==4
    assert all(sha(Path(p))==h for p,h in inputs.items())
    manifest=dict(status='four_prior_regressions_and_four_new_candidates_prepared',
        prior_confirmed_real=1,prior_confirmed_false=3,new_candidates_pending=4,
        training_data='../ball_044_reviewed_appearance_cases/cases.csv',training_source_group='legacy_match1:rally044',
        prospective_check_group='legacy_match1:match1_motion_pilot_008',
        selection='008:391,503,615,1623 selected after looking at old annotation/model disagreement and image overview; old truth coordinates were not used as new labels',
        split_unit='whole source clip, not individual frames; a clip may contain edits/multiple points',
        warning='Selected same-match development/safety cases, not an untouched representative test. 002/318 is before play;018 includes camera changes. Serve validity and exact rally bounds not newly confirmed.',
        candidate_table_sha256=sha(OUT/'cases.csv'),input_hashes=inputs,
        adopted_tracking_inputs_unchanged=True,source_labels_modified=False)
    dump(OUT/'case_manifest.json',manifest);print(json.dumps({k:v for k,v in manifest.items() if k!='input_hashes'},indent=2))

if __name__=='__main__':main()
