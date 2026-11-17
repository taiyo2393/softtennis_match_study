"""One fixed temporal-input sensitivity comparison: t-3..t vs t-6,t-4,t-2,t.

No fitting, threshold sweep, body mask, score tuning or inferred new labels.
"""
import json,sys,math,argparse
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2
from run_manual_ball_finetune import metrics
from run_hard_case_ball_finetune import good_ids,err
from compare_ball_candidate_decoders import components,choose

OUT=ROOT/'ball_input_stride_comparison';BASE=ROOT/'ball_temporal_path_comparison';DATA=ROOT/'balltrack_manual_finetune_pilot'
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def ids(t,stride):return [max(0,t-j*stride) for j in [3,2,1,0]]
def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists(),'Preserve fixed trial. No result-driven alternate strides.'
    modelroot=SOURCE/'third_party/RacketVision/source/BallTrack';weight=modelroot/'checkpoints/balltrack_best.pth'
    train_path=ROOT/'balltrack_expanded_centers_pilot/expanded_train_labels.csv';short=ROOT/'ball_short_sequence_evaluation'
    paths=[weight,modelroot/'dataset/uball.py',modelroot/'configs/tracknetv3_base.py',modelroot/'inference.py',
        train_path,DATA/'validation_labels.csv',DATA/'audit_labels.csv',ROOT/'ball_inplay_challenge/query_model_inputs.csv',ROOT/'ball_inplay_challenge/manual_center_labels.csv',
        short/'manual_sequence_labels.csv',short/'evaluation_report.json',short/'frozen_predictions.csv',short/'query_model_inputs.csv',
        ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json',
        SOURCE/'annotations/balltrack_groundtruth_match1_pilot.csv',ROOT/'ball_040_object_swap_review/round2/user_reviewed/candidate_review.csv',
        ROOT/'ball_appearance_training_review/candidate_review_labels.csv',ROOT/'ball_training_only_appearance_comparison/decision.json',
        BASE/'train_bbox_area.csv',BASE/'train_displacement_0.1.csv',BASE/'validation_bbox_area.csv',BASE/'validation_selected.csv',BASE/'challenge_selected.csv',BASE/'audit_selected.csv']
    protected={str(p):sha(p) for p in paths}
    protocol=dict(status='fixed_before_new_inference',hypothesis='More distinct motion history may help on approximately30Hz motion encoded at59.94Hz; NOT a demonstrated cause or a correction to the model.',
        source_training_code='Uball supports sliding_step and config uses1; original training-video frame rates not established. Stride2 changes inference distribution.',
        input_strides=[1,2],input_frame_ids='causal max(0,t-3*s),max(0,t-2*s),max(0,t-s),t; both end at same target t',
        image='Same original BGR JPEGs, median background,512x288,/255; seq_len4; no crops/augmentation',
        exposure_span_sec=dict(stride1=3/59.94005994005994,stride2=6/59.94005994005994),
        model=str(weight),threshold=.3,decoder='Same external contours and maximum bounding-rectangle area; no temporal reranking applied to either fresh-input arm',
        baseline='Fresh stride1 isolates input effect; also compare against prior stronger frozen temporal outputs to check safety.',
        numerics='CUDA AMP,eval; fixed batch4. Re-inference jitter versus old saved predictions reported, not called improvement.',
        roles='train123,validation120,challenge4,short_sequence26; same-match previously seen development data. Short includes4reused centers,2of them challenge anchors; never combine counts.',
        fitting=False,parameter_search=False,validation_selection=False,
        six_false='Score old actual6false-candidate images before manual removal. Exact same point is known false; any shifted/new output stays unverified without true center; no output means position unknown.',
        audit='Only if stride2 loses no prior temporal-correct point and does not increase error counts in train/validation/challenge/short, and improves validation or withholds a known false candidate; one fixed stride only.',
        adoption=False,interpolation=False,physical_absence_labels_added=0,
        protected_input_hashes=protected,script_sha256=sha(Path(__file__)))
    dump(OUT/'protocol.json',protocol)
    sys.path.insert(0,str(modelroot));from inference import BallInferencer
    infer=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(weight),device='cuda',thre=.3,batchsize=4)
    assert infer.seq_len==4
    medians={};image_hashes={};source_metas={};input_rows=[];changes=[];cadence=[]
    def run(rows,role):
        result={1:[],2:[]};maps={1:[],2:[]}
        for stride in [1,2]:
            for start in range(0,len(rows),4):
                batch=[];indices=[]
                for r in rows[start:start+4]:
                    cid=r['candidate_id'];t=int(r['local_frame']);base=SOURCE/'data/balltrack/racketvision/match1'/cid
                    if cid not in source_metas:
                        p=base/'runs/threshold030_median120/run_metadata.json';source_metas[cid]=load(p);protected[str(p)]=sha(p)
                    mp=base/'runs/threshold030_median120/median.npz'
                    if str(mp) not in medians:
                        protected[str(mp)]=sha(mp);medians[str(mp)]=np.moveaxis(cv2.resize(np.load(mp)['median'],(512,288)).astype(np.float32)/255.,-1,0)
                    js=ids(t,stride);ims=[];hashes=[]
                    for j in js:
                        p=base/'frames'/f'{j:06d}.jpg'
                        if str(p) not in image_hashes:image_hashes[str(p)]=sha(p)
                        im=cv2.imread(str(p));assert im.shape==(1080,1920,3);ims.append(cv2.resize(im,(512,288)));hashes.append(image_hashes[str(p)])
                    assert js[-1]==t
                    if stride==1 and 'input_frames_json' in r:
                        assert [Path(p).stem for p in json.loads(r['input_frames_json'])]==[f'{j:06d}' for j in js]
                        assert hashes==json.loads(r['input_frame_sha256_json'])
                    input_rows.append(dict(role=role,sample_id=r['sample_id'],candidate_id=cid,local_frame=t,source_frame=r['source_frame'],input_stride=stride,
                        input_local_frames=json.dumps(js),input_image_sha256=json.dumps(hashes),median_path=str(mp),median_sha256=protected[str(mp)],
                        source_fps=source_metas[cid]['fps'],span_sec=(js[-1]-js[0])/source_metas[cid]['fps']))
                    # Label-independent image-change proxy, not unique-motion-frame GT.
                    mad=[float(np.abs(a[96:].astype(np.float32)-b[96:].astype(np.float32)).mean()) for a,b in zip(ims,ims[1:])]
                    cadence.append(dict(role=role,sample_id=r['sample_id'],stride=stride,court_rgb_mad_steps_json=json.dumps(mad),mean_court_mad=float(np.mean(mad))))
                    seq=np.moveaxis(np.stack(ims).astype(np.float32)/255.,-1,1)
                    batch.append(np.concatenate([medians[str(mp)][None],seq],axis=0).reshape(15,288,512));indices.append(r)
                with torch.inference_mode(),torch.amp.autocast('cuda'):
                    prediction,_,_=infer.model.forward(frames=torch.from_numpy(np.stack(batch)).to(infer.device))
                for r,hm in zip(indices,prediction[:,0].float().cpu().numpy(),strict=True):
                    cs=components(hm);c=choose(cs,'bbox_area');x,y=(c['x_px'],c['y_px']) if c else (0,0)
                    ox,oy,ow,oh,_=infer._predict_location(hm);assert (x,y)==(int((ox+ow/2)*1920/512),int((oy+oh/2)*1080/288))
                    result[stride].append(dict(sample_id=r['sample_id'],candidate_id=r['candidate_id'],local_frame=r['local_frame'],source_frame=r['source_frame'],role=role,input_stride=stride,
                        truth_visibility=r.get('truth_visibility',''),truth_x_px=r.get('truth_x_px',''),truth_y_px=r.get('truth_y_px',''),
                        x_px=x,y_px=y,visible=int(c is not None),confidence=c['bbox_mean'] if c else 0.,component_count=len(cs),interpolated=0))
                    maps[stride].append(hm.astype(np.float16))
            write(OUT/f'{role}_stride{stride}.csv',result[stride])
            np.savez_compressed(OUT/f'{role}_stride{stride}_heatmaps.npz',sample_ids=np.array([r['sample_id'] for r in rows]),heatmaps=np.stack(maps[stride]))
        print('INFER',role,len(rows),flush=True);return result
    train=read(train_path);validation=read(DATA/'validation_labels.csv')
    gt={r['sample_id']:r for r in read(ROOT/'ball_inplay_challenge/manual_center_labels.csv')}
    challenge=[dict(r,**{k:gt[r['sample_id']][k] for k in ['truth_visibility','truth_x_px','truth_y_px']}) for r in read(ROOT/'ball_inplay_challenge/query_model_inputs.csv')]
    gt={r['sample_id']:r for r in read(short/'manual_sequence_labels.csv')}
    assert all(r['phase_status']=='play' and r['truth_visibility']=='visible' for r in gt.values())
    short_rows=[dict(r,**{k:gt[r['sample_id']][k] for k in ['truth_visibility','truth_x_px','truth_y_px']}) for r in read(short/'query_model_inputs.csv')]
    references=dict(train=read(BASE/'train_displacement_0.1.csv'),validation=read(BASE/'validation_selected.csv'),challenge=read(BASE/'challenge_selected.csv'),
        short_sequence=[dict(r,**{k:gt[r['sample_id']][k] for k in ['truth_visibility','truth_x_px','truth_y_px']}) for r in read(short/'frozen_predictions.csv') if r['method']=='displacement_0.1'])
    results={};comparisons=[]
    def compare(rows,role,reference):
        arms=run(rows,role);a,b=arms[1],arms[2];baseline=metrics(a);proposed=metrics(b);ref=metrics(reference)
        lost=sorted(good_ids(reference)-good_ids(b));lost_fresh=sorted(good_ids(a)-good_ids(b));gained=sorted(good_ids(b)-good_ids(a))
        for x,y in zip(a,b,strict=True):
            if (x['x_px'],x['y_px'],x['visible'])!=(y['x_px'],y['y_px'],y['visible']):
                changes.append(dict(role=role,sample_id=x['sample_id'],candidate_id=x['candidate_id'],local_frame=x['local_frame'],source_frame=x['source_frame'],
                    truth_visibility=x['truth_visibility'],truth_x_px=x['truth_x_px'],truth_y_px=x['truth_y_px'],
                    stride1_x=x['x_px'],stride1_y=x['y_px'],stride1_visible=x['visible'],stride2_x=y['x_px'],stride2_y=y['y_px'],stride2_visible=y['visible']))
        r=dict(fresh_stride1=baseline,stride2=proposed,prior_temporal=ref,prior_correct_lost=lost,fresh_correct_lost=lost_fresh,fresh_correct_gained=gained,
            nonregression=not lost and err(proposed)<=err(ref),visible_position_target='same frame, no shift')
        results[role]=r
        for method,m in [('fresh_stride1',baseline),('stride2',proposed),('prior_temporal',ref)]:
            comparisons.append(dict(role=role,method=method,frames=len(rows),correct=m['counts']['localized_tp_16px'],errors=err(m),missing_visible=m['counts']['presence_fn'],outputs=m['counts']['pred_visible']))
        return arms
    old_area_paths=dict(train=BASE/'train_bbox_area.csv',validation=BASE/'validation_bbox_area.csv')
    reproduction=[]
    for role,rows in [('train',train),('validation',validation),('challenge',challenge),('short_sequence',short_rows)]:
        arms=compare(rows,role,references[role])
        if role in old_area_paths:
            old={r['sample_id']:r for r in read(old_area_paths[role])}
            for r in arms[1]:
                b=old[r['sample_id']];same=(int(r['visible']),float(r['x_px']),float(r['y_px']))==(int(b['visible']),float(b['x_px']),float(b['y_px']))
                reproduction.append(dict(role=role,sample_id=r['sample_id'],old_x=b['x_px'],old_y=b['y_px'],fresh_x=r['x_px'],fresh_y=r['y_px'],same=int(same),
                    distance_if_both_output=math.hypot(float(r['x_px'])-float(b['x_px']),float(r['y_px'])-float(b['y_px'])) if int(r['visible']) and int(b['visible']) else ''))
    false=read(ROOT/'ball_040_object_swap_review/round2/user_reviewed/candidate_review.csv');six=[]
    for r in false:
        six.append(dict(sample_id=f'match1_motion_pilot_015_frame_{int(r["local_frame"]):06d}',candidate_id='match1_motion_pilot_015',local_frame=r['local_frame'],source_frame=r['source_frame'],
            known_false_x=r['displayed_x_px'],known_false_y=r['displayed_y_px']))
    arms=run(six,'six_false');false_results=[]
    for query,a,b in zip(six,arms[1],arms[2],strict=True):
        dist=math.hypot(b['x_px']-float(query['known_false_x']),b['y_px']-float(query['known_false_y'])) if b['visible'] else None
        state='no_output_position_unknown' if not b['visible'] else 'exact_same_known_false_point' if dist==0 else 'near_known_false_point_unverified' if dist<=16 else 'different_unverified_candidate'
        false_results.append(dict(query,stride1_x=a['x_px'],stride1_y=a['y_px'],stride1_visible=a['visible'],stride2_x=b['x_px'],stride2_y=b['y_px'],stride2_visible=b['visible'],
            distance_to_old_false_point=dist,status=state,true_ball_center='unknown',physical_visibility='not_labelled'))
    write(OUT/'six_false_outcomes.csv',false_results)
    benefit=err(results['validation']['stride2'])<err(results['validation']['prior_temporal']) or results['validation']['stride2']['counts']['localized_tp_16px']>81 or any(r['status']=='no_output_position_unknown' for r in false_results)
    eligible=all(r['nonregression'] for r in results.values()) and benefit
    if eligible:compare(read(DATA/'audit_labels.csv'),'audit',read(BASE/'audit_selected.csv'))
    write(OUT/'comparison_summary.csv',comparisons);write(OUT/'changed_observations.csv',changes) if changes else None
    write(OUT/'stride1_reproduction.csv',reproduction);write(OUT/'input_manifest.csv',input_rows);write(OUT/'input_image_change_proxy.csv',cadence)
    dump(OUT/'input_image_hashes.json',image_hashes)
    assert all(sha(Path(p))==h for p,h in protected.items()) and all(sha(Path(p))==h for p,h in image_hashes.items())
    decision=dict(status='fixed_input_stride_comparison_complete',results=results,six_false_status_counts=dict(Counter(r['status'] for r in false_results)),
        audit_performed=bool(eligible),automatic_adoption=False,operational_outputs_changed=False,models_and_source_labels_unchanged=True,
        stride1_reproduction=dict(rows=len(reproduction),identical=sum(r['same'] for r in reproduction),changed=sum(not r['same'] for r in reproduction)),
        no_parameter_tuning=True,independent_generalization_established=False,protected_input_hashes=protected,
        limitation='Stride2 is a fixed model input sensitivity test, not a verified30fps correction. New or shifted six-case outputs have no true centers and remain unverified.')
    dump(OUT/'decision.json',decision);print(json.dumps(comparisons,indent=2),flush=True);print('SIX',decision['six_false_status_counts'],flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:
        assert ids(10,1)==[7,8,9,10] and ids(10,2)==[4,6,8,10] and ids(2,2)==[0,0,0,2]
        for t in range(20):
            for s in [1,2]:assert len(ids(t,s))==4 and ids(t,s)[-1]==t and all(0<=j<=t for j in ids(t,s))
        print('Passed causal four-image order, target alignment and boundary clamp; no labels written.')
    else:main()
