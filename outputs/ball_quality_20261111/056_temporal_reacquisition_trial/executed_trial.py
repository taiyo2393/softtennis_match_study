"""Fixed exploratory heatmap-candidate recovery inside observed tracklet gaps."""
from pathlib import Path
import sys, shutil
import cv2
import numpy as np
import torch
import build_044_spatiotemporal_pilot as b
from prepare_056_route_transfer import FRAMES, OFFSET

OUT=b.HERE/'056_temporal_reacquisition_trial'
BASE=b.PRIOR/'rally_repair_056'
REVIEW=b.HERE/'056_post_hit_confirmation'
FIRST,LAST=154853,154880
THRESHOLD=.1
RADIUS=32.0

def main():
    OUT.mkdir(exist_ok=False)
    modelroot=b.SOURCE/'third_party/RacketVision/source/BallTrack'
    weight=modelroot/'checkpoints/balltrack_best.pth'
    protected={}
    for folder in [REVIEW,b.HERE/'056_post_hit_trial']:
        for name,digest in b.load(folder/'publication_hashes.json').items():
            assert b.sha(folder/name)==digest
            protected[str(folder/name)]=digest
    for p in [weight,modelroot/'configs/tracknetv3_base.py',BASE/'baseline_all_frames.csv',BASE/'median.npz',Path(__file__)]:
        protected[str(p)]=b.sha(p)
    images=[dict(source_frame=f,path=str(FRAMES/f'{f-OFFSET:06d}.jpg'),sha256=b.sha(FRAMES/f'{f-OFFSET:06d}.jpg'))
            for f in range(FIRST-3,LAST+1)]
    b.dump(OUT/'source_images.json',images)
    b.dump(OUT/'protocol.json',dict(
        fixed_before_new_heatmaps=True,original_weight_and_inputs_unchanged=True,
        protected_input_hashes=protected,source_range=[FIRST,LAST],original_batch_alignment_preserved=True,
        hypothesis='A weak same-frame heatmap component may exist near a path bracketed by user-confirmed ball tracklets even when the original threshold0.3 decoder has no output.',
        candidate_threshold=THRESHOLD,native_search_radius_px=RADIUS,max_gap_source_frames=12,
        candidate_rule='On original missing frames only, extract ALL full-image contours above0.1; retain centers within32native pixels of endpoint linear reference. Select only if exactly ONE component passes. Do not rasterize a predicted ball when no component passes.',
        original_outputs='All15 post-interval original outputs retained exactly; do not reselect or filter them.',
        motion='Linear reference is search guidance, not a returned observation. Both endpoint anchors come from one user-reviewed window. No contact interval crossed. No claim of true trajectory linearity.',
        scope='Offline future-assisted, same-rally exploratory development. Hyperparameters are a single proposed setting, not independently validated or automatically adopted.',
        adoption='Keep original publications unchanged. Every new component requires review; no new human centers or true-ball labels generated.',
    ))
    original={int(r['source_frame']):r for r in b.read(BASE/'baseline_all_frames.csv')}
    confirmed=b.load(REVIEW/'human_video_review.json')['derived_confirmed_candidate_frames']
    anchors=sorted(confirmed)
    brackets={}
    for a,z in zip(anchors,anchors[1:]):
        if 1<z-a<=12:
            for f in range(a+1,z):
                assert original[f]['visible']=='0'
                t=(f-a)/(z-a)
                guide=np.array([float(original[a][k])*(1-t)+float(original[z][k])*t for k in ['x_px','y_px']])
                brackets[f]=(a,z,guide)
    assert sorted(brackets)==list(range(154861,154869))+list(range(154873,154877))+[154878]
    sys.path.insert(0,str(modelroot))
    from inference import BallInferencer
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    engine=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(weight),device='cuda',thre=.3,batchsize=4)
    assert engine.device.type=='cuda', 'Reproduce existing CUDA AMP baseline before testing decoder.'
    engine.model.eval().requires_grad_(False)
    median=np.load(BASE/'median.npz')['median']
    median=np.moveaxis(cv2.resize(median,(512,288)).astype(np.float32)/255.,-1,0)
    heatmaps=[];reproductions=[];rows=[];components=[]
    for lo in range(FIRST,LAST+1,4):
        times=list(range(lo,min(lo+4,LAST+1)));inputs=[]
        for f in times:
            frames=np.stack([cv2.resize(cv2.imread(str(FRAMES/f'{j-OFFSET:06d}.jpg')),(512,288)) for j in range(f-3,f+1)])
            normalized=np.moveaxis(frames.astype(np.float32)/255.,-1,1)
            inputs.append(np.concatenate([median[None],normalized],axis=0).reshape(15,288,512))
        tensor=torch.from_numpy(np.stack(inputs)).to(engine.device)
        with torch.inference_mode(),torch.amp.autocast('cuda'):
            output,_,_=engine.model.forward(frames=tensor)
        for f,hm in zip(times,output[:,0].float().cpu().numpy(),strict=True):
            heatmaps.append(hm)
            x,y,w,h,conf=engine._predict_location(hm)
            px,py=int((x+w/2)*1920/512),int((y+h/2)*1080/288)
            old=original[f]
            same=px==int(old['x_px']) and py==int(old['y_px']) and float(conf)==float(old['confidence'])
            reproductions.append(dict(source_frame=f,coordinates_and_confidence_exact=int(same)))
            assert same,(f,px,py,conf,old)
            r=dict(source_frame=f,baseline_x_px=old['x_px'],baseline_y_px=old['y_px'],baseline_visible=old['visible'],
                   proposed_x_px=old['x_px'] if old['visible']=='1' else '',proposed_y_px=old['y_px'] if old['visible']=='1' else '',
                   kind='retained_original_user_window_confirmed' if old['visible']=='1' else 'missing',
                   anchor_first='',anchor_last='',search_reference_x='',search_reference_y='',
                   eligible_components=0,component_peak='',component_mean='',review_status='no_new_human_review')
            if f in brackets:
                a,z,guide=brackets[f]
                r.update(anchor_first=a,anchor_last=z,search_reference_x=float(guide[0]),search_reference_y=float(guide[1]))
                binary=(hm>THRESHOLD).astype(np.uint8)
                contours,_=cv2.findContours(binary*255,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                eligible=[]
                for i,c in enumerate(contours):
                    x,y,w,h=cv2.boundingRect(c)
                    cx,cy=int((x+w/2)*1920/512),int((y+h/2)*1080/288)
                    distance=float(np.linalg.norm(np.array([cx,cy])-guide))
                    mask=np.zeros_like(binary);cv2.drawContours(mask,[c],-1,1,-1)
                    pixels=hm[(mask!=0)&(binary!=0)]
                    component=dict(source_frame=f,component_index=i,x_px=cx,y_px=cy,bbox_width_px=w*1920/512,
                                   bbox_height_px=h*1080/288,peak=float(pixels.max()),mean=float(pixels.mean()),
                                   reference_distance_px=distance,passes_search_gate=int(distance<=RADIUS))
                    components.append(component)
                    if distance<=RADIUS:eligible.append(component)
                r['eligible_components']=len(eligible)
                if len(eligible)==1:
                    c=eligible[0]
                    r.update(proposed_x_px=c['x_px'],proposed_y_px=c['y_px'],kind='weak_heatmap_candidate_unconfirmed',
                             component_peak=c['peak'],component_mean=c['mean'])
                elif len(eligible)>1:r['kind']='missing_ambiguous_components'
            rows.append(r)
        print('reproduced and decoded',times[0],times[-1],flush=True)
    np.savez_compressed(OUT/'full_heatmaps.npz',source_frames=np.arange(FIRST,LAST+1),heatmaps=np.stack(heatmaps))
    b.write(OUT/'proposed_path.csv',rows);b.write(OUT/'all_weak_components.csv',components)
    b.write(OUT/'baseline_reproduction.csv',reproductions)
    assert all(b.sha(Path(p))==digest for p,digest in protected.items())
    for im in images:assert b.sha(Path(im['path']))==im['sha256']
    result=dict(original_outputs_retained=sum(r['baseline_visible']=='1' for r in rows),
        original_missing_frames=len(brackets),new_weak_candidates=sum(r['kind']=='weak_heatmap_candidate_unconfirmed' for r in rows),
        unresolved_frames=sum(r['kind'].startswith('missing') for r in rows),
        original_model_reproduction_exact_frames=len(reproductions),old_publications_unchanged=True,
        new_confirmed_ball_locations=0,new_interpolated_returned_positions=0,
        proposed_candidate_frames=[r['source_frame'] for r in rows if r['kind']=='weak_heatmap_candidate_unconfirmed'])
    b.dump(OUT/'result.json',result)
    shutil.copy2(__file__,OUT/'executed_trial.py')
    print(result,flush=True)

if __name__=='__main__':main()
