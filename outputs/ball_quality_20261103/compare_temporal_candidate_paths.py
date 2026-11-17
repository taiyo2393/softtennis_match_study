"""Fixed short-window candidate paths; parameters chosen on training clips only.

All outputs are actual original-model component bbox centers. Empty observations,
camera transitions and short gap-delimited segments retain original behavior.
Future8frames used: this is offline local selection, not causal live tracking.
"""
import argparse,json,math,sys
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2
from run_manual_ball_finetune import tensors,metrics
from run_hard_case_ball_finetune import good_ids,err
from compare_ball_candidate_decoders import components,choose
from evaluate_observed_ball_fallback import camera_mask

OUT=ROOT/'ball_temporal_path_comparison'
DATA=ROOT/'balltrack_manual_finetune_pilot'
OFFSETS=list(range(-8,9,2))
CONFIGS=[dict(name=f'{order}_{strength:g}',order=order,strength=strength)
    for order in ['displacement','acceleration'] for strength in [.1,.3,1.]]
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def xy(cs):return np.array([[c['x_px'],c['y_px']] for c in cs],dtype=np.float64)

def best_path(nodes,order,strength):
    """Exact DP for the stated additive costs; returns one node index per time."""
    emission=[-np.log(np.maximum([c['component_peak'] for c in cs],1e-8)) for cs in nodes]
    positions=[xy(cs) for cs in nodes]
    if order=='displacement':
        costs=emission[0];backs=[]
        for j in range(1,len(nodes)):
            distances=np.linalg.norm(positions[j-1][:,None]-positions[j][None],axis=2)
            total=costs[:,None]+strength*np.minimum(distances/64.,4.)
            back=np.argmin(total,axis=0);costs=total[back,np.arange(len(nodes[j]))]+emission[j];backs.append(back)
        indexes=[int(np.argmin(costs))]
        for back in reversed(backs):indexes.append(int(back[indexes[-1]]))
        return list(reversed(indexes))
    assert order=='acceleration' and len(nodes)>=3
    costs=emission[0][:,None]+emission[1][None,:];backs=[]
    for j in range(2,len(nodes)):
        residual=positions[j-2][:,None,None]-2*positions[j-1][None,:,None]+positions[j][None,None,:]
        total=costs[:,:,None]+strength*np.minimum(np.linalg.norm(residual,axis=3)/16.,4.)
        back=np.argmin(total,axis=0)
        costs=np.take_along_axis(total,back[None,:,:],axis=0)[0]+emission[j][None,:];backs.append(back)
    left,right=np.unravel_index(int(np.argmin(costs)),costs.shape);indexes=[int(right),int(left)]
    for back in reversed(backs):indexes.append(int(back[indexes[-1],indexes[-2]]))
    return list(reversed(indexes))

def select(context,t,mask,config):
    cs=context[t];base=choose(cs,'bbox_area')
    if base is None:return None,'original_missing'
    if config is None:return base,'original_area'
    times=sorted(j for j in context if j in {t+d for d in OFFSETS})
    if mask[max(0,min(times)):max(times)+1].any():return base,'camera_transition_original'
    pos=times.index(t);lo=hi=pos
    while lo>0 and times[lo]-times[lo-1]==2 and context[times[lo-1]]:lo-=1
    while hi+1<len(times) and times[hi+1]-times[hi]==2 and context[times[hi+1]]:hi+=1
    times=times[lo:hi+1]
    if len(times)<3:return base,'short_segment_original'
    # Sorting only for deterministic ties. Component indices are not temporal IDs.
    nodes=[sorted(context[j],key=lambda c:(-c['bbox_area'],c['component_index'])) for j in times]
    path=best_path(nodes,config['order'],config['strength'])
    return nodes[times.index(t)][path[times.index(t)]],'observed_path_selection'

def fake(x,peak=.8,area=9,index=0):return dict(x_px=x,y_px=100,component_peak=peak,bbox_area=area,component_index=index,bbox_mean=peak)
def self_test():
    from itertools import product
    nodes=[[fake(100),fake(800,.9,index=1)],[fake(110),fake(1300,.99,index=1)],[fake(120),fake(900,.9,index=1)], [fake(130),fake(1400,.9,index=1)]]
    for order in ['displacement','acceleration']:
        def cost(path):
            value=sum(-math.log(nodes[j][k]['component_peak']) for j,k in enumerate(path))
            if order=='displacement':value+=.3*sum(min(abs(nodes[j][path[j]]['x_px']-nodes[j-1][path[j-1]]['x_px'])/64,4) for j in range(1,len(path)))
            else:value+=.3*sum(min(abs(nodes[j][path[j]]['x_px']-2*nodes[j-1][path[j-1]]['x_px']+nodes[j-2][path[j-2]]['x_px'])/16,4) for j in range(2,len(path)))
            return value
        actual=best_path(nodes,order,.3);optimum=min(cost(p) for p in product(range(2),repeat=4))
        assert abs(cost(actual)-optimum)<1e-10 and actual==[0,0,0,0]
    mask=np.zeros(30,bool);context={8:[fake(100)],10:[fake(110),fake(1200,.99,100,index=1)],12:[fake(120)]}
    config=dict(order='displacement',strength=.3)
    assert select(context,10,mask,config)[0]['x_px']==110
    mask[9]=True;assert select(context,10,mask,config)[0]['x_px']==1200
    mask[:]=False;context[8]=[];assert select(context,10,mask,config)[1]=='short_segment_original'
    context[10]=[];assert select(context,10,mask,config)==(None,'original_missing')
    print('Passed: DP agrees with exhaustive enumeration, moving candidate support, cut/gap/missing fallbacks. No research labels written.')

def get_contexts(infer,rows,role,protected):
    contexts={};masks={};image_hashes={};medians={}
    groups={cid:[r for r in rows if r['candidate_id']==cid] for cid in sorted({r['candidate_id'] for r in rows})}
    for cid,group in groups.items():
        base=SOURCE/'data/balltrack/racketvision/match1'/cid;meta_path=base/'runs/threshold030_median120/run_metadata.json'
        meta=load(meta_path);protected[str(meta_path)]=sha(meta_path)
        cam_path=ROOT.parent/'ball_quality_20261102/temporal_guard/camera_features'/f'{cid}.json'
        camera=load(cam_path);protected[str(cam_path)]=sha(cam_path);masks[cid]=camera_mask(camera['changed_fraction'])
        mp=Path(group[0]['median_path']);assert sha(mp)==group[0]['median_sha256'];protected[str(mp)]=sha(mp)
        medians[str(mp)]=np.moveaxis(cv2.resize(np.load(mp)['median'],(512,288)).astype(np.float32)/255.,-1,0)
        # Existing input clocks and target images are verified before inference.
        for r in group:
            assert int(r['local_frame'])==int(Path(json.loads(r['input_frames_json'])[-1]).stem)
            for path,h in zip(json.loads(r['input_frames_json']),json.loads(r['input_frame_sha256_json']),strict=True):assert sha(Path(path))==h
        times=sorted({int(r['local_frame'])+d for r in group for d in OFFSETS if 3<=int(r['local_frame'])+d<meta['frame_count']})
        contexts[cid]={}
        for start in range(0,len(times),4):
            batch=[]
            for t in times[start:start+4]:
                ims=[]
                for j in range(t-3,t+1):
                    path=base/'frames'/f'{j:06d}.jpg'
                    if str(path) not in image_hashes:image_hashes[str(path)]=sha(path)
                    im=cv2.imread(str(path));assert im.shape[:2]==(1080,1920);ims.append(cv2.resize(im,(512,288)))
                batch.append(dict(frames=np.stack(ims),median_key=str(mp),row={}))
            with torch.inference_mode(),torch.amp.autocast('cuda'):maps,_,_=infer.model.forward(frames=tensors(batch,medians,infer.device))
            for t,hm in zip(times[start:start+4],maps[:,0].float().cpu().numpy(),strict=True):contexts[cid][t]=components(hm)
        print(role,cid,'context frames',len(times),flush=True)
    dump(OUT/f'{role}_component_cache.json',contexts);dump(OUT/f'{role}_context_image_hashes.json',image_hashes)
    return contexts,masks

def predictions(rows,contexts,masks,config,role):
    result=[]
    for r in rows:
        c,status=select(contexts[r['candidate_id']],int(r['local_frame']),masks[r['candidate_id']],config)
        px,py=(c['x_px'],c['y_px']) if c else (0,0)
        result.append(dict(sample_id=r['sample_id'],candidate_id=r['candidate_id'],local_frame=r['local_frame'],source_frame=r['source_frame'],
            role=role,method=config['name'] if config else 'bbox_area',truth_visibility=r['truth_visibility'],truth_x_px=r['truth_x_px'],truth_y_px=r['truth_y_px'],
            x_px=px,y_px=py,visible=int(px!=0 or py!=0),confidence=c['bbox_mean'] if c else 0.,selection_status=status,
            selected_component_index=c['component_index'] if c else -1))
    return result

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists(),'Preserve trial; no post-result retuning.'
    manifest=load(DATA/'data_manifest.json');protected=dict(manifest['input_hashes'])
    train_path=ROOT/'balltrack_expanded_centers_pilot/expanded_train_labels.csv'
    challenge=ROOT/'ball_inplay_challenge'
    for path in [train_path,DATA/'validation_labels.csv',DATA/'audit_labels.csv',challenge/'manual_center_labels.csv',challenge/'frozen_method_predictions.csv']:
        protected[str(path)]=sha(path)
    modelroot=SOURCE/'third_party/RacketVision/source/BallTrack';weight=modelroot/'checkpoints/balltrack_best.pth';protected[str(weight)]=sha(weight)
    assert all(sha(Path(p))==h for p,h in protected.items())
    protocol=dict(status='fixed_before_temporal_inference_and_training_parameter_selection',training='Existing120 + new A/C/D3; B postplay excluded. Training8clips only.',
        original_model=str(weight),threshold=.3,context_offsets=OFFSETS,context='9 observations over16source frames; offline future8frames (~0.133s at59.94fps)',
        candidates='Unchanged external contours and bbox centers; output only actual target-frame observed centers. No interpolation or new presence.',
        emission='-log(component_peak), uncalibrated heuristic cost, not calibrated probability',
        displacement_cost='strength*min(source_pixel_displacement/64,4) per2source frames',
        acceleration_cost='strength*min(norm(p[t]-2*p[t-2]+p[t-4])/16,4)',
        parameter_grid=CONFIGS,parameters_fitted_on='Training labels123 only; validation/challenge coordinates not read for fitting',
        fallback='Original area if cut-mask anywhere in window, nonempty segment<3 observations, or target missing; reset at empty contexts. Contour indices are never identity labels.',
        selection='Include original. Require every original correct TRAINING frame retained, errors<=training original. Max correct16px, then fewer wrong_position+presenceFP; ties original then grid order. No iterative search.',
        validation='Evaluate ONE frozen chosen configuration on fixed120; if it loses any original correct frame or adds errors, stop before challenge/audit.',
        challenge='Only after validation nonregression: frozen configuration on4 confirmed-play centers, never training/tuning. Previously seen same-match developmental challenge, not independent.',
        audit='Only if chosen config passes validation and challenge nonregression and improves either: original vs selected on60, no alternative choices.',
        limitations='Sparse labels include preparation/transitions; temporal candidates can be consistent but false; no full-track accuracy or external validation claim',
        automatic_adoption=False,protected_input_hashes=protected,script_sha256=sha(Path(__file__)))
    dump(OUT/'protocol.json',protocol)
    sys.path.insert(0,str(modelroot));from inference import BallInferencer
    torch.backends.cudnn.benchmark=False
    infer=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(weight),device='cuda',thre=.3,batchsize=4)
    train=read(train_path);assert len(train)==123 and {int(r['candidate_id'][-3:]) for r in train}==set(manifest['train_clips'])
    contexts,masks=get_contexts(infer,train,'train',protected)
    original=predictions(train,contexts,masks,None,'train');baseline=metrics(original);base_ids=good_ids(original)
    write(OUT/'train_bbox_area.csv',original);summaries=[];chosen=None;best=(len(base_ids),-err(baseline))
    for config in CONFIGS:
        rows=predictions(train,contexts,masks,config,'train');m=metrics(rows);lost=sorted(base_ids-good_ids(rows));eligible=not lost and err(m)<=err(baseline)
        assert [r['visible'] for r in rows]==[r['visible'] for r in original]
        summaries.append(dict(config,correct=m['counts']['localized_tp_16px'],errors=err(m),original_correct_lost=len(lost),lost_sample_ids=';'.join(lost),eligible=int(eligible)))
        write(OUT/f'train_{config["name"]}.csv',rows)
        if eligible and (m['counts']['localized_tp_16px'],-err(m))>best:chosen=config;best=(m['counts']['localized_tp_16px'],-err(m))
    write(OUT/'training_parameter_comparison.csv',summaries)
    selection=dict(chosen_configuration=chosen,training_original=baseline,selected_training_key=best,validation_labels_used=False,challenge_labels_used=False,
        training_results=summaries,selection_frozen=True,automatic_adoption=False)
    dump(OUT/'training_selection.json',selection);print('TRAINING SELECTION',json.dumps(selection),flush=True)
    report=dict(status='training_parameter_comparison_complete',chosen_configuration=chosen,original_training=baseline,
        validation=None,challenge=None,audit=None,operational_outputs_changed=False,automatic_adoption=False)
    if chosen is not None:
        validation=read(DATA/'validation_labels.csv');contexts,masks=get_contexts(infer,validation,'validation',protected)
        vb=predictions(validation,contexts,masks,None,'validation');vp=predictions(validation,contexts,masks,chosen,'validation')
        write(OUT/'validation_bbox_area.csv',vb);write(OUT/'validation_selected.csv',vp)
        bm=metrics(vb);pm=metrics(vp);lost=sorted(good_ids(vb)-good_ids(vp));passes=not lost and err(pm)<=err(bm)
        assert len(good_ids(vb))==80 and err(bm)==3
        report['validation']=dict(original=bm,selected=pm,original_correct_lost=lost,nonregression=passes)
        dump(OUT/'validation_decision.json',report['validation']);print('VALIDATION',json.dumps(report['validation']),flush=True)
        if passes:
            labels=read(challenge/'manual_center_labels.csv');inputs=read(challenge/'query_model_inputs.csv');by_id={r['sample_id']:r for r in labels}
            assert all(r['phase_status']=='play' and r['phase_review_status']=='human_confirmed' for r in labels)
            query=[dict(r,**{k:by_id[r['sample_id']][k] for k in ['truth_visibility','truth_x_px','truth_y_px']}) for r in inputs]
            contexts,masks=get_contexts(infer,query,'challenge',protected)
            cb=predictions(query,contexts,masks,None,'challenge');cp=predictions(query,contexts,masks,chosen,'challenge')
            write(OUT/'challenge_bbox_area.csv',cb);write(OUT/'challenge_selected.csv',cp)
            cm=metrics(cb);tm=metrics(cp);clost=sorted(good_ids(cb)-good_ids(cp));cpasses=not clost and err(tm)<=err(cm)
            assert len(good_ids(cb))==2 and err(cm)==2
            report['challenge']=dict(original=cm,selected=tm,original_correct_lost=clost,nonregression=cpasses)
            dump(OUT/'challenge_decision.json',report['challenge']);print('CHALLENGE',json.dumps(report['challenge']),flush=True)
            if cpasses and (len(good_ids(vp))>80 or len(good_ids(cp))>2 or err(pm)<3 or err(tm)<2):
                audit=read(DATA/'audit_labels.csv');contexts,masks=get_contexts(infer,audit,'audit',protected)
                ab=predictions(audit,contexts,masks,None,'audit');ap=predictions(audit,contexts,masks,chosen,'audit')
                write(OUT/'audit_bbox_area.csv',ab);write(OUT/'audit_selected.csv',ap)
                am=metrics(ab);sm=metrics(ap);alost=sorted(good_ids(ab)-good_ids(ap))
                assert len(good_ids(ab))==34 and err(am)==5
                report['audit']=dict(original=am,selected=sm,original_correct_lost=alost,nonregression=not alost and err(sm)<=err(am))
    assert all(sha(Path(p))==h for p,h in protected.items())
    dump(OUT/'verified_inputs.json',dict(protected_input_hashes=protected,all_unchanged=True))
    report.update(status='fixed_temporal_path_trial_complete',protected_inputs_unchanged=True,
        next='Keep original unless a reviewed broader tracking comparison supports replacement. Do not tune parameters on challenge/audit or infer continuous accuracy from sparse centers.')
    dump(OUT/'decision.json',report);print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--self-test',action='store_true');args=parser.parse_args()
    if args.self_test:self_test()
    else:main()
