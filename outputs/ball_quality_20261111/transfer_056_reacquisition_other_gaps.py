"""Apply the single frozen weak-component decoder to all eligible other 056 gaps."""
from pathlib import Path
import json,sys,shutil
from collections import Counter
import cv2,numpy as np,torch
import build_044_spatiotemporal_pilot as b
from prepare_056_route_transfer import FRAMES,OFFSET

OUT=b.HERE/'056_reacquisition_other_gaps'
BASE=b.PRIOR/'rally_repair_056'
FROZEN=b.HERE/'056_temporal_reacquisition_trial'
CONFIRMED=b.HERE/'056_temporal_reacquisition_confirmed'

def main():
    OUT.mkdir(exist_ok=False)
    method=b.load(FROZEN/'protocol.json')
    threshold=method['candidate_threshold'];radius=method['native_search_radius_px'];maxgap=method['max_gap_source_frames']
    assert (threshold,radius,maxgap)==(.1,32.,12)
    observed={int(r['source_frame']):r for r in b.read(BASE/'completed_review/assisted_states_sampled.csv')}
    baseline={int(r['source_frame']):r for r in b.read(BASE/'baseline_all_frames.csv')}
    phase={int(r['source_frame']):r['phase'] for r in b.read(b.HERE/'056_post_hit_trial/event_segmented_model_path.csv')}
    anchors=sorted(f for f,r in observed.items() if r['observation_present']=='1' and f!=154780)
    audit=[];jobs={}
    for a,z in zip(anchors,anchors[1:]):
        if z-a<=2:continue
        missing=[f for f in range(a+2,z,2) if observed[f]['observation_present']=='0' or f==154780]
        reason='eligible_same_flight_short_gap'
        if a<=154780<=z:reason='known_wrong_candidate_no_true_location'
        elif z-a>maxgap:reason='gap_exceeds_frozen_maximum'
        elif not(phase[a].startswith('flight_') and all(phase[f]==phase[a] for f in range(a,z+1))):
            reason='crosses_AI_contact_or_bounce_interval'
        for f in missing:
            why=reason
            if why=='eligible_same_flight_short_gap' and baseline[f]['visible']=='1':why='original_output_already_present'
            row=dict(source_frame=f,anchor_first=a,anchor_last=z,anchor_first_source=observed[a]['coordinate_source'],
                     anchor_last_source=observed[z]['coordinate_source'],eligibility=why,
                     source_phase=phase[f],baseline_visible=baseline[f]['visible'])
            audit.append(row)
            if why=='eligible_same_flight_short_gap':
                t=(f-a)/(z-a)
                guide=np.array([float(observed[a][k])*(1-t)+float(observed[z][k])*t for k in ['x_px','y_px']])
                jobs[f]=(a,z,guide)
    assert len(audit)==43 and len(jobs)==10
    b.write(OUT/'all_missing_frame_eligibility.csv',audit)
    maps={int(r['source_frame']):r for r in b.read(BASE/'map_records.csv')}
    groups=sorted({tuple(json.loads(maps[f]['batch_frames_json'])) for f in jobs})
    source_times=sorted({OFFSET+t for g in groups for t in g})
    images=sorted({f for t in source_times for f in range(t-3,t+1)})
    modelroot=b.SOURCE/'third_party/RacketVision/source/BallTrack';weight=modelroot/'checkpoints/balltrack_best.pth'
    protected={}
    for folder in [FROZEN,CONFIRMED]:
        for name,digest in b.load(folder/'publication_hashes.json').items():
            assert b.sha(folder/name)==digest
            protected[str(folder/name)]=digest
    for p in [weight,modelroot/'configs/tracknetv3_base.py',BASE/'baseline_all_frames.csv',BASE/'median.npz',
              BASE/'completed_review/assisted_states_sampled.csv',b.HERE/'056_post_hit_trial/event_segmented_model_path.csv',Path(__file__)]:
        protected[str(p)]=b.sha(p)
    b.dump(OUT/'source_images.json',[dict(source_frame=f,path=str(FRAMES/f'{f-OFFSET:06d}.jpg'),sha256=b.sha(FRAMES/f'{f-OFFSET:06d}.jpg')) for f in images])
    b.dump(OUT/'protocol.json',dict(
        fixed_before_new_heatmaps=True,role='same-rally applicability development transfer, not independent validation',
        inherited_setting=dict(candidate_threshold=threshold,native_search_radius_px=radius,max_gap_source_frames=maxgap),
        selection='ALL annotated missing frames before154839 plus known rejected154780; eligible iff short gap bracketed by existing ball observations in one unchanged AI flight phase and no original output. No visually favorable subset selection.',
        event_timing='AI provisional intervals; stroke count and hitter order human-confirmed, not precise contact timing.',
        input_anchors='Previous assisted observations, including human-clicked corrections. Unlike the end-only trial some anchors are clicked. This is a human-assisted repair pipeline, not raw automatic tracking or independent ground truth.',
        decoder='Exactly the original trial: full-image >0.1 components, bbox centers within32px of linear endpoint reference, return only when exactly1eligible component exists. No direct interpolation output, retraining or parameter search.',
        exclusions=dict(Counter(r['eligibility'] for r in audit)),eligible_source_frames=sorted(jobs),
        original_batch_groups_local_frames=[list(g) for g in groups],protected_input_hashes=protected,
        adoption='New candidates pending human review, old publications and annotations untouched. No accuracy claim from candidate counts.',
    ))
    sys.path.insert(0,str(modelroot));from inference import BallInferencer
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    engine=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(weight),device='cuda',thre=.3,batchsize=4)
    assert engine.device.type=='cuda'
    engine.model.eval().requires_grad_(False)
    median=np.moveaxis(cv2.resize(np.load(BASE/'median.npz')['median'],(512,288)).astype(np.float32)/255.,-1,0)
    heatmaps={};reproductions=[]
    for group in groups:
        times=[OFFSET+t for t in group];inputs=[]
        for f in times:
            frames=np.stack([cv2.resize(cv2.imread(str(FRAMES/f'{j-OFFSET:06d}.jpg')),(512,288)) for j in range(f-3,f+1)])
            norm=np.moveaxis(frames.astype(np.float32)/255.,-1,1)
            inputs.append(np.concatenate([median[None],norm],axis=0).reshape(15,288,512))
        tensor=torch.from_numpy(np.stack(inputs)).to(engine.device)
        with torch.inference_mode(),torch.amp.autocast('cuda'):output,_,_=engine.model.forward(frames=tensor)
        for f,hm in zip(times,output[:,0].float().cpu().numpy(),strict=True):
            x,y,w,h,conf=engine._predict_location(hm);px,py=int((x+w/2)*1920/512),int((y+h/2)*1080/288)
            old=baseline[f]
            exact=(px==int(old['x_px']) and py==int(old['y_px']) and float(conf)==float(old['confidence']))
            assert exact,(f,old,px,py,conf)
            reproductions.append(dict(source_frame=f,coordinates_and_confidence_exact=int(exact)))
            heatmaps[f]=hm
    rows=[];components=[]
    for f,(a,z,guide) in sorted(jobs.items()):
        hm=heatmaps[f];binary=(hm>threshold).astype(np.uint8)
        contours,_=cv2.findContours(binary*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);eligible=[]
        for i,c in enumerate(contours):
            x,y,w,h=cv2.boundingRect(c);cx,cy=int((x+w/2)*1920/512),int((y+h/2)*1080/288)
            distance=float(np.linalg.norm(np.array([cx,cy])-guide))
            mask=np.zeros_like(binary);cv2.drawContours(mask,[c],-1,1,-1);pixels=hm[(mask!=0)&(binary!=0)]
            component=dict(source_frame=f,component_index=i,x_px=cx,y_px=cy,bbox_width_px=w*1920/512,
                bbox_height_px=h*1080/288,peak=float(pixels.max()),mean=float(pixels.mean()),
                reference_distance_px=distance,passes_search_gate=int(distance<=radius))
            components.append(component)
            if distance<=radius:eligible.append(component)
        r=dict(source_frame=f,anchor_first=a,anchor_last=z,search_reference_x=float(guide[0]),search_reference_y=float(guide[1]),
               eligible_components=len(eligible),proposed_x_px='',proposed_y_px='',component_peak='',component_mean='',
               kind='missing_ambiguous_components' if len(eligible)>1 else 'missing',review_status='unreviewed')
        if len(eligible)==1:
            c=eligible[0];r.update(proposed_x_px=c['x_px'],proposed_y_px=c['y_px'],component_peak=c['peak'],
                                   component_mean=c['mean'],kind='weak_heatmap_candidate_unconfirmed')
        rows.append(r)
    b.write(OUT/'proposed_candidates.csv',rows);b.write(OUT/'all_weak_components.csv',components)
    b.write(OUT/'baseline_reproduction.csv',reproductions)
    times=sorted(heatmaps);np.savez_compressed(OUT/'full_heatmaps.npz',source_frames=np.array(times),heatmaps=np.stack([heatmaps[f] for f in times]))
    assert all(b.sha(Path(p))==digest for p,digest in protected.items())
    result=dict(audited_missing_times=len(audit),eligibility_counts=dict(Counter(r['eligibility'] for r in audit)),
        eligible_times=len(jobs),new_unconfirmed_candidates=sum(r['kind']=='weak_heatmap_candidate_unconfirmed' for r in rows),
        unresolved_eligible_times=sum(r['kind'].startswith('missing') for r in rows),
        proposed_frames=[r['source_frame'] for r in rows if r['kind']=='weak_heatmap_candidate_unconfirmed'],
        original_model_outputs_reproduced_exactly=len(reproductions),new_human_ball_labels=0,
        new_interpolated_output_positions=0,old_publications_unchanged=True,independent_accuracy_established=False)
    b.dump(OUT/'result.json',result);shutil.copy2(__file__,OUT/'executed_trial.py')
    print(result,flush=True)

if __name__=='__main__':main()
