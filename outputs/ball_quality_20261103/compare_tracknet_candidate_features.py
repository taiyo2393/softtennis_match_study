"""Candidate-only verifier on frozen TrackNet's pre-predictor temporal features.

No Gaussian targets or empty-frame negatives are created. A separate small
linear head uses36 explicit candidate labels, with whole source clips held out.
"""
import json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np,torch
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2
from compare_frozen_ball_backbones import fit,predict

OUT=ROOT/'ball_tracknet_candidate_feature_comparison'
BASE_GROUP='legacy_match1:match1_motion_pilot_016'

def dump(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8')

def main(view_mode='source_view'):
    global OUT
    assert view_mode in ('source_view','common_full_frame')
    if view_mode=='common_full_frame':OUT=ROOT/'ball_tracknet_common_view_comparison'
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists(),'Preserve completed experiments; no in-place tuning.'
    source=ROOT/'ball_sequence_candidate_supervision';manifest=load(source/'manifest.json')
    sp=source/'sequence_cases.json';assert sha(sp)==manifest['sequence_cases_sha256']
    cases=load(sp);assert len(cases)==36
    if view_mode=='common_full_frame':
        cases=[dict(c,annotation_inference_view_name=c['view_name'],annotation_inference_view_xywh=c['view_xywh'],
            view_name='common_full_frame',view_xywh=[0,0,1920,1080],
            query_network_xy=[c['candidate_source_xy'][0]*512/1920,c['candidate_source_xy'][1]*288/1080]) for c in cases]
    groups=np.array([r['split_group'] for r in cases]);labels=np.array([r['candidate_is_ball'] for r in cases],dtype=np.float64)
    modelroot=SOURCE/'third_party/RacketVision/source/BallTrack';weight=modelroot/'checkpoints/balltrack_best.pth'
    protected={str(p):sha(p) for p in [sp,weight,ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json',
        ROOT/'ball_pretrained_feature_comparison/grouped_predictions.csv']}
    config=modelroot/'configs/tracknetv3_base.py'
    protocol=dict(status='specified_before_feature_extraction_or_head_fitting',
        model='Existing RacketVision MS-TrackNetV3,15 input channels,4 temporal frames plusmedian',
        source_weights_sha256=sha(weight),source_config_sha256=sha(config),
        backbone_training=False,predictor_training=False,new_head='Separate64-feature binary candidate verifier only',
        view_mode=view_mode,
        view_policy='Exact original detector views' if view_mode=='source_view' else 'All cases use the full1920x1080 source frame, including those originally reviewed with tiled inference. Same candidate coordinates and labels.',
        view_policy_reason='The original12 tiled cases all belong to016, so held-out016 leaves no tiled cases in training. Common-view comparison removes that view/clip confound, without changing labels or choosing a threshold.',
        input='Original clocks; BGR median+t-3..t,512x288, /255; view specified byview_policy',
        features='64 channels immediately before predictor1x1conv; bilinear samples on3x3 output-grid queries centered at candidate; average of9 samples',
        query_geometry='Original view-relative source coordinates scaled512/width,288/height; align_corners=True for grid_sample; no predicted/interpolated ball positions',
        candidate_center_is_not_precise_gt=True,
        fit='Same training-only standardisation and class-balanced meanBCE+0.5L2,CPUfloat64,LBFGS300,strong_wolfe,zero initialisation,threshold0.5',
        regimes=['train016_only_check_other8','leave_one_source_clip_out'],split_unit='whole source clip',
        comparator='Previously saved grouped hand-feature and frozenResNet feature predictions, identical36 case IDs/groups',
        additional_reference='Max original heatmap in query3x3 >=0.3, a candidate-local score only, NOT the original largest-component decoder',
        warning='Selected same-match retrospective development cases. Model domain, temporal context and input views differ fromResNet; any difference cannot be assigned to temporal information alone.',
        legacy300_rows_used_for_training=False,dense_heatmap_targets_created=False,hyperparameter_search=False,
        automatic_adoption=False,independent_accuracy_evaluation=False,
        sequence_cases_sha256=sha(sp),script_sha256=sha(Path(__file__)),protected_input_hashes=protected)
    dump(OUT/'protocol.json',protocol)
    dump(OUT/'effective_sequence_cases.json',cases)
    sys.path.insert(0,str(modelroot));from inference import BallInferencer
    infer=BallInferencer(str(config),str(weight),device='cuda',thre=.3,batchsize=4)
    infer.model.eval().requires_grad_(False)
    assert infer.model.predictor.in_channels==64 and infer.model.last_only
    captured={}
    def capture(module,args):captured['features']=args[0]
    hook=infer.model.predictor.register_forward_pre_hook(capture)
    by=defaultdict(list)
    for index,c in enumerate(cases):by[(c['ball_pilot_id'],c['local_frame'],tuple(c['view_xywh']))].append((index,c))
    jobs=list(by);features=np.zeros((36,64),dtype=np.float64);references=np.zeros(36,dtype=np.float64)
    source_images={};medians={};checks=[]
    cached_full=np.load(ROOT/'inplay_ball_044/heatmaps.npz')
    cached_index={int(t):i for i,t in enumerate(cached_full['local_frames'])}
    for first in range(0,len(jobs),4):
        batch=jobs[first:first+4];inputs=[]
        for identity in batch:
            c=by[identity][0][1];left,top,width,height=c['view_xywh'];patches=[]
            mp=c['median_path']
            assert sha(Path(mp))==c['median_sha256']
            if mp not in medians:medians[mp]=np.load(mp)['median']
            assert medians[mp].shape[:2]==(1080,1920)
            patches.append(medians[mp][top:top+height,left:left+width])
            for path,h in zip(c['frame_paths'],c['frame_sha256'],strict=True):
                assert sha(Path(path))==h
                if path not in source_images:source_images[path]=cv2.imread(path)
                im=source_images[path];assert im is not None and im.shape[:2]==(1080,1920)
                patches.append(im[top:top+height,left:left+width])
            arr=np.stack([np.moveaxis(cv2.resize(p,(512,288)).astype(np.float32)/255.,-1,0) for p in patches]).reshape(15,288,512)
            inputs.append(arr)
        with torch.inference_mode(),torch.amp.autocast('cuda'):
            heatmaps,_,_=infer.model.forward(frames=torch.from_numpy(np.stack(inputs)).to(infer.device))
        hidden=captured.pop('features').float();heatmaps=heatmaps.float()
        assert hidden.shape[1:]==(64,288,512) and heatmaps.shape[1:]==(1,288,512)
        for j,identity in enumerate(batch):
            for index,c in by[identity]:
                qx,qy=c['query_network_xy']
                grid=torch.tensor([[[[2*(qx+dx)/511-1,2*(qy+dy)/287-1] for dx in (-1,0,1)] for dy in (-1,0,1)]],device=infer.device,dtype=torch.float32)
                sampled=torch.nn.functional.grid_sample(hidden[j:j+1],grid,mode='bilinear',align_corners=True)
                local=torch.nn.functional.grid_sample(heatmaps[j:j+1],grid,mode='bilinear',align_corners=True)
                features[index]=sampled.mean((2,3))[0].cpu().numpy().astype(np.float64)
                references[index]=float(local.max())
            cid,t,rect=identity
            if cid=='match1_motion_pilot_016' and rect==(0,0,1920,1080):
                hm=heatmaps[j,0].cpu().numpy();old=cached_full['heatmaps'][cached_index[t]].astype(np.float32)
                difference=float(np.max(np.abs(hm-old)));assert difference<.01,(t,difference)
                checks.append(dict(local_frame=t,max_heatmap_difference=difference))
        del hidden,heatmaps
        print('frozen TrackNet inputs',min(first+4,len(jobs)),'/',len(jobs),flush=True)
    hook.remove();assert features.shape==(36,64) and np.isfinite(features).all()
    np.savez_compressed(OUT/'features.npz',features=features,case_ids=np.array([r['case_id'] for r in cases]),groups=groups,reference_local_peaks=references)
    folds=[('train016_only_check_other8','other_clips',groups==BASE_GROUP,groups!=BASE_GROUP)]
    folds += [('leave_one_source_clip_out',g,groups!=g,groups==g) for g in sorted(set(groups))]
    predictions=[];fold_meta=[]
    for regime,heldout,train,test in folds:
        assert not set(groups[train])&set(groups[test])
        # Check input frames as well as clip IDs: no temporal window crosses the split.
        train_hashes={h for c,m in zip(cases,train) if m for h in c['frame_sha256']}
        test_hashes={h for c,m in zip(cases,test) if m for h in c['frame_sha256']}
        assert not train_hashes&test_hashes
        fitted=fit(features[train],labels[train]);scores=predict(fitted,features[test])
        filename=f'{regime}__{heldout.replace(":","_")}.npz';np.savez_compressed(OUT/filename,**fitted)
        fold_meta.append(dict(regime=regime,heldout=heldout,train_cases=int(train.sum()),test_cases=int(test.sum()),
            train_groups=sorted(set(groups[train])),test_groups=sorted(set(groups[test])),head_sha256=sha(OUT/filename),head_file=filename))
        for index,score in zip(np.where(test)[0],scores):
            c=cases[index]
            predictions.append(dict(regime=regime,heldout_group=heldout,method='frozen_tracknet_temporal64',case_id=c['case_id'],
                local_frame=c['local_frame'],ball_pilot_id=c['ball_pilot_id'],group=c['split_group'],candidate_is_ball=int(labels[index]),
                uncalibrated_ball_score=float(score),keep=int(score>=.5),original_local_peak=references[index]))
    write(OUT/'grouped_predictions.csv',predictions)
    old=read(ROOT/'ball_pretrained_feature_comparison/grouped_predictions.csv')
    assert {r['case_id'] for r in predictions if r['regime']=='leave_one_source_clip_out'}=={r['case_id'] for r in old if r['regime']=='leave_one_source_clip_out'}
    all_rows=old+predictions;summaries=[]
    for regime in protocol['regimes']:
        for method in ('handcrafted672','frozen_resnet18_512','frozen_tracknet_temporal64'):
            rows=[r for r in all_rows if r['regime']==regime and r['method']==method]
            real=[r for r in rows if int(r['candidate_is_ball'])];false=[r for r in rows if not int(r['candidate_is_ball'])]
            summaries.append(dict(regime=regime,method=method,reviewed_real_total=len(real),reviewed_real_kept=sum(int(r['keep']) for r in real),
                reviewed_false_total=len(false),reviewed_false_kept=sum(int(r['keep']) for r in false),
                missed_real_candidates=';'.join(r['case_id'] for r in real if not int(r['keep'])),
                kept_false_candidates=';'.join(r['case_id'] for r in false if int(r['keep']))))
    write(OUT/'summary.csv',summaries)
    per_group=[]
    for g in sorted(set(groups)):
        rr=[r for r in predictions if r['regime']=='leave_one_source_clip_out' and r['group']==g]
        real=[r for r in rr if r['candidate_is_ball']];false=[r for r in rr if not r['candidate_is_ball']]
        per_group.append(dict(group=g,real_total=len(real),real_kept=sum(r['keep'] for r in real),false_total=len(false),false_kept=sum(r['keep'] for r in false)))
    write(OUT/'per_clip_summary.csv',per_group);dump(OUT/'folds.json',fold_meta)
    assert all(sha(Path(p))==h for p,h in protected.items())
    assert sha(weight)==protocol['source_weights_sha256']
    for p,h in load(ROOT/'play_start_audit/inventory_summary.json')['source_hashes'].items():assert sha(Path(p))==h
    new_summary=[r for r in summaries if r['method']=='frozen_tracknet_temporal64']
    decision=dict(status='candidate_only_temporal_feature_trial_complete_not_adopted',automatic_adoption=False,
        candidate_annotation_reuse='Valid as binary candidate labels; not promoted to dense localization truth',
        new_method_summary=new_summary,comparison_summary=summaries,source_model_reproduction_checks=checks,
        old_detector_and_reviewed_outputs_unchanged=True,sequence_frame_hashes_disjoint_across_folds=True,
        source_annotations_unchanged=True,backbone_training=False,old300_labels_used_for_training=False,
        independent_accuracy_established=False,scope='Small repeatedly inspected same-match candidate cases, not full-rally tracking accuracy')
    dump(OUT/'decision.json',decision);print(json.dumps(decision,indent=2))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--view-mode',choices=['source_view','common_full_frame'],default='source_view')
    main(parser.parse_args().view_mode)
