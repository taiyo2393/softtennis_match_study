"""Single fixed 2x image-scale crop trial, retaining observations and failures."""
from pathlib import Path
import sys,shutil
from collections import Counter
import cv2,numpy as np,torch
import build_044_spatiotemporal_pilot as b
from prepare_056_route_transfer import FRAMES,OFFSET

OUT=b.HERE/'056_crop_reacquisition_trial'
BASE=b.PRIOR/'rally_repair_056'
GAPS=b.HERE/'056_reacquisition_other_gaps'
END=b.HERE/'056_temporal_reacquisition_confirmed'

def components(hm,threshold,x0,y0,width=960,height=540):
    binary=(hm>threshold).astype(np.uint8)
    contours,_=cv2.findContours(binary*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);out=[]
    for i,c in enumerate(contours):
        x,y,w,h=cv2.boundingRect(c);mask=np.zeros_like(binary);cv2.drawContours(mask,[c],-1,1,-1)
        values=hm[(mask!=0)&(binary!=0)]
        out.append(dict(component_index=i,x_px=x0+(x+w/2)*width/512,y_px=y0+(y+h/2)*height/288,
                        width_px=w*width/512,height_px=h*height/288,
                        peak=float(values.max()),mean=float(values.mean())))
    return out

def main():
    OUT.mkdir(exist_ok=False)
    frozen=b.load(b.HERE/'056_temporal_reacquisition_trial/protocol.json')
    threshold=frozen['candidate_threshold'];radius=frozen['native_search_radius_px']
    assert (threshold,radius)==(.1,32.)
    jobs=[]
    for r in b.read(GAPS/'proposed_candidates.csv'):
        jobs.append(dict(source_frame=int(r['source_frame']),role='previously_unresolved_gap',
            guide_x=float(r['search_reference_x']),guide_y=float(r['search_reference_y']),
            guide_source='existing assisted endpoint linear reference',
            comparison_x='',comparison_y='',guide_frames=[int(r['anchor_first']),int(r['anchor_last'])]))
    end=[r for r in b.read(END/'post_H13_ball_path_reviewed.csv') if r['observation_present']=='1']
    known={int(r['source_frame']):r for r in end}
    for f,r in sorted(known.items()):
        # Exclude the target itself when constructing control search guidance.
        neighbours=sorted((g for g in known if g!=f),key=lambda g:(abs(g-f),g))[:2]
        a,z=sorted(neighbours);t=(f-a)/(z-a)
        gx=float(known[a]['x_px'])*(1-t)+float(known[z]['x_px'])*t
        gy=float(known[a]['y_px'])*(1-t)+float(known[z]['y_px'])*t
        jobs.append(dict(source_frame=f,role='previously_ball_enclosing_candidate',guide_x=gx,guide_y=gy,
            guide_source='two nearest other reviewed end points; target coordinate excluded; not independent evaluation',
            comparison_x=float(r['x_px']),comparison_y=float(r['y_px']),guide_frames=[a,z]))
    assisted={int(r['source_frame']):r for r in b.read(BASE/'completed_review/assisted_states_sampled.csv')}
    a,z=154778,154782
    jobs.append(dict(source_frame=154780,role='known_other_object_coordinate_diagnostic',
        guide_x=(float(assisted[a]['x_px'])+float(assisted[z]['x_px']))/2,
        guide_y=(float(assisted[a]['y_px'])+float(assisted[z]['y_px']))/2,
        guide_source='adjacent assisted positions excluding rejected target',comparison_x=1102.,comparison_y=579.,guide_frames=[a,z]))
    jobs.sort(key=lambda r:r['source_frame'])
    assert len(jobs)==30
    for j in jobs:
        j['crop_x0']=max(0,min(960,round(j['guide_x']-480)))
        j['crop_y0']=max(0,min(540,round(j['guide_y']-270)))
        j['crop_width']=960;j['crop_height']=540
    modelroot=b.SOURCE/'third_party/RacketVision/source/BallTrack';weight=modelroot/'checkpoints/balltrack_best.pth'
    protected={}
    for folder in [GAPS,END]:
        for name,digest in b.load(folder/'publication_hashes.json').items():
            assert b.sha(folder/name)==digest;protected[str(folder/name)]=digest
    for p in [weight,modelroot/'configs/tracknetv3_base.py',BASE/'median.npz',BASE/'baseline_all_frames.csv',
              BASE/'completed_review/assisted_states_sampled.csv',Path(__file__)]:protected[str(p)]=b.sha(p)
    b.dump(OUT/'protocol.json',dict(
        fixed_before_crop_heatmaps=True,crop_native_size=[960,540],cached_image_size=[1920,1080],model_input_size=[512,288],
        image_scale_change='2x linear ball scale relative to full-image input. Crop context also changes; not a pure isolated resolution manipulation.',
        coordinates='All returned centers mapped from crop512x288 to cached1920x1080 coordinates. Not original4K coordinates.',
        crop_center='Search guide from other frames; fixed crop across all4frames and background for each target, bounded in image.',
        guides='Gap endpoints include previous human clicks. Control guide excludes target coordinates but uses other reviewed frames, including future frames. Human-assisted offline development, not an autonomous independent test.',
        original_weight_inputs='Same checkpoint, BGR, /255,4consecutiveframes plus same whole-context median, CUDA AMP. No training or threshold sweep.',
        primary_decoder=dict(candidate_threshold=threshold,search_radius_native_px=radius,
            rule='Same weak-component rule as prior trial: exactly1component center within fixed32px guide radius; otherwise abstain.'),
        diagnostic_decoder='Original0.3 largest-bbox output shown only in diagnostic columns; not an alternative selected after results.',
        target_roles=dict(Counter(j['role'] for j in jobs)),
        controls='19 previously ball-enclosing points: report coordinate displacement only, not pixel accuracy or new identity truth. One known other-object coordinate is not a full-frame negative label.',
        review_selection='If candidates on missing targets exist, review at most4, uniformly spaced in chronological candidate list; retain ALL other candidates and failed targets.',
        protected_input_hashes=protected,new_human_truth_generated=False,automatic_adoption=False,
    ))
    b.dump(OUT/'query_jobs.json',jobs)
    times=sorted({g for j in jobs for g in range(j['source_frame']-3,j['source_frame']+1)})
    b.dump(OUT/'source_images.json',[dict(source_frame=f,path=str(FRAMES/f'{f-OFFSET:06d}.jpg'),sha256=b.sha(FRAMES/f'{f-OFFSET:06d}.jpg')) for f in times])
    sys.path.insert(0,str(modelroot));from inference import BallInferencer
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    engine=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(weight),device='cuda',thre=.3,batchsize=4)
    assert engine.device.type=='cuda'
    engine.model.eval().requires_grad_(False)
    median=np.load(BASE/'median.npz')['median'];rows=[];allcomponents=[];heatmaps=[]
    for start in range(0,len(jobs),4):
        batch=jobs[start:start+4];inputs=[]
        for j in batch:
            f=j['source_frame'];x0,y0=j['crop_x0'],j['crop_y0']
            crop=lambda im:cv2.resize(im[y0:y0+540,x0:x0+960],(512,288))
            frames=np.stack([crop(cv2.imread(str(FRAMES/f'{g-OFFSET:06d}.jpg'))) for g in range(f-3,f+1)])
            images=np.moveaxis(frames.astype(np.float32)/255.,-1,1)
            med=np.moveaxis(crop(median).astype(np.float32)/255.,-1,0)
            inputs.append(np.concatenate([med[None],images],axis=0).reshape(15,288,512))
        tensor=torch.from_numpy(np.stack(inputs)).to(engine.device)
        with torch.inference_mode(),torch.amp.autocast('cuda'):output,_,_=engine.model.forward(frames=tensor)
        for j,hm in zip(batch,output[:,0].float().cpu().numpy(),strict=True):
            heatmaps.append(hm);r=dict(j);x0,y0=j['crop_x0'],j['crop_y0']
            cs=components(hm,threshold,x0,y0);eligible=[]
            for c in cs:
                distance=float(np.hypot(c['x_px']-j['guide_x'],c['y_px']-j['guide_y']))
                q=dict(source_frame=j['source_frame'],role=j['role'],**c,reference_distance_px=distance,passes_gate=int(distance<=radius))
                allcomponents.append(q)
                if distance<=radius:eligible.append(c)
            r.update(eligible_components=len(eligible),candidate_x_px='',candidate_y_px='',component_peak='',component_mean='',
                     kind='missing_ambiguous_components' if len(eligible)>1 else 'missing',comparison_distance_px='',
                     review_status='unreviewed',primary_threshold=threshold)
            if len(eligible)==1:
                c=eligible[0];r.update(candidate_x_px=c['x_px'],candidate_y_px=c['y_px'],component_peak=c['peak'],
                    component_mean=c['mean'],kind='crop_heatmap_candidate_unconfirmed')
                if j['comparison_x']!='':r['comparison_distance_px']=float(np.hypot(c['x_px']-j['comparison_x'],c['y_px']-j['comparison_y']))
            x,y,w,h,conf=engine._predict_location(hm)
            r['diagnostic_original_threshold_confidence']=float(conf)
            r['diagnostic_original_threshold_x']=x0+(x+w/2)*960/512 if x or y else ''
            r['diagnostic_original_threshold_y']=y0+(y+h/2)*540/288 if x or y else ''
            r['diagnostic_nearest_component_to_known_wrong_px']=(min(np.hypot(c['x_px']-1102,c['y_px']-579) for c in cs) if cs else '') if j['role']=='known_other_object_coordinate_diagnostic' else ''
            rows.append(r)
    b.write(OUT/'candidate_rows.csv',rows);b.write(OUT/'all_components.csv',allcomponents)
    np.savez_compressed(OUT/'full_crop_heatmaps.npz',source_frames=np.array([j['source_frame'] for j in jobs]),heatmaps=np.stack(heatmaps))
    new=[r for r in rows if r['role']=='previously_unresolved_gap' and r['kind']=='crop_heatmap_candidate_unconfirmed']
    indices=sorted(set(np.rint(np.linspace(0,len(new)-1,min(4,len(new)))).astype(int))) if new else []
    review=[new[i] for i in indices];b.dump(OUT/'selected_review_frames.json',[r['source_frame'] for r in review])
    byrole={role:dict(Counter(r['kind'] for r in rows if r['role']==role)) for role in sorted({r['role'] for r in rows})}
    assert all(b.sha(Path(p))==digest for p,digest in protected.items())
    result=dict(roles=byrole,new_unconfirmed_gap_candidates=len(new),review_frames=[r['source_frame'] for r in review],
        all_proposed_gap_frames=[r['source_frame'] for r in new],new_human_ball_labels=0,new_interpolated_output_positions=0,
        original_positions_or_labels_changed=0,old_publications_unchanged=True,independent_accuracy_established=False)
    b.dump(OUT/'result.json',result);shutil.copy2(__file__,OUT/'executed_trial.py');print(result,flush=True)

if __name__=='__main__':main()
