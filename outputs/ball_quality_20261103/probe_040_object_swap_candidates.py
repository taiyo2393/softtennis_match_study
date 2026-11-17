"""Reapply existing fixed weak/full and9tile recipe; no replacement selection."""
import json,sys
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np,torch
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha

sys.path.insert(0,str(ROOT.parent/'ball_quality_20261102'))
from extract_weak_candidates import components,P
OUT=ROOT/'ball_040_object_swap_review/candidate_probe'
TARGETS=[464,467,470,554]
TIMES=sorted({t+d for t in TARGETS for d in [-2,0,2]})
VIEWS=[(0,0,1920,1080)]+[(x,y,960,540) for y in (0,270,540) for x in (0,480,960)]
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'protocol.json').exists()
    cid='match1_motion_pilot_015';base=SOURCE/'data/balltrack/racketvision/match1'/cid
    modelroot=SOURCE/'third_party/RacketVision/source/BallTrack';weight=modelroot/'checkpoints/balltrack_best.pth'
    protected={str(p):sha(p) for p in [ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json',weight,
        ROOT/'ball_040_object_swap_review/user_reviewed/review.json',ROOT/'ball_040_object_swap_review/user_reviewed/baseline_balltrack_predictions.csv',
        ROOT/'ball_040_object_swap_review/user_reviewed/temporal_balltrack_predictions.csv',ROOT/'balltrack_manual_finetune_pilot/validation_labels.csv',
        ROOT/'ball_inplay_challenge/manual_center_labels.csv']}
    assert P['heatmap_floor']==.1 and P['components_per_frame']==6
    protocol=dict(status='fixed_before_inference',targets=TARGETS,context_frames=TIMES,views_xywh=VIEWS,
        model_sha256=sha(weight),recipe='Reuse044scale probe: fixedfull image+all9uniform2x overlapping crops, unchanged .10floor/top6/halfpeak-weighted centers',
        no_view_or_component_chosen_using_true_center=True,known_labels='4false displayed candidates only, no true center or absence label',
        output='Diagnostic candidates only. No ball identity, no automatic filling, no re-training or threshold search.',
        protected_input_hashes=protected,script_sha256=sha(Path(__file__)))
    dump(OUT/'protocol.json',protocol)
    sys.path.insert(0,str(modelroot));from inference import BallInferencer
    infer=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(weight),device='cuda',thre=.3,batchsize=8)
    median=np.load(base/'runs/threshold030_median120/median.npz')['median']
    images={j:cv2.imread(str(base/'frames'/f'{j:06d}.jpg')) for j in sorted({j for t in TIMES for j in range(t-3,t+1)})}
    assert all(im is not None and im.shape[:2]==(1080,1920) for im in images.values())
    medians=[np.moveaxis(cv2.resize(median[y:y+h,x:x+w],(512,288)).astype(np.float32)/255.,-1,0) for x,y,w,h in VIEWS]
    jobs=[(t,k) for t in TIMES for k in range(len(VIEWS))];rows=[];maps=[]
    for start in range(0,len(jobs),8):
        batch=jobs[start:start+8];data=[]
        for t,k in batch:
            x,y,w,h=VIEWS[k]
            frames=np.stack([np.moveaxis(cv2.resize(images[j][y:y+h,x:x+w],(512,288)).astype(np.float32)/255.,-1,0) for j in range(t-3,t+1)])
            data.append(np.concatenate([medians[k][None],frames]).reshape(15,288,512))
        tensor=torch.from_numpy(np.stack(data)).to(infer.device)
        with torch.inference_mode(),torch.amp.autocast('cuda'):pred,_,_=infer.model.forward(frames=tensor)
        for (t,k),hm in zip(batch,pred[:,0].float().cpu().numpy(),strict=True):
            x,y,w,h=VIEWS[k];maps.append(hm.astype(np.float16))
            for rank,c in enumerate(components(hm,w,h),1):rows.append(dict(local_frame=t,view_index=k,view='full' if k==0 else 'fixed_2x_tile',tile_x=x,tile_y=y,
                rank=rank,x_px=c['x_px']+x,y_px=c['y_px']+y,peak=c['peak'],area=c['area'],mean=c['mean'],candidate_identity='unreviewed'))
    write(OUT/'all_diagnostic_candidates.csv',rows)
    np.savez_compressed(OUT/'heatmaps.npz',heatmaps=np.stack(maps),local_frames=np.array([t for t,k in jobs]),view_indices=np.array([k for t,k in jobs]))
    # Visual audit of possible alternatives, never automatic identity/GT assignment.
    reviews={int(r['local_frame']):r for r in read(ROOT/'ball_040_object_swap_review/user_reviewed/candidate_review.csv')}
    counts=[];displayed=[]
    for t in TARGETS:
        pool=[r for r in rows if r['local_frame']==t];unique=[]
        for r in sorted(pool,key=lambda c:-c['peak']):
            if not any(np.hypot(r['x_px']-c['x_px'],r['y_px']-c['y_px'])<=12 for c in unique):unique.append(r)
        counts.append(dict(local_frame=t,full_weak_candidates=sum(r['view_index']==0 for r in pool),tiled_candidates=sum(r['view_index']!=0 for r in pool),
            deduplicated_all_view_candidates=len(unique),verified_alternative_ball_candidates=0))
        image=images[t].copy();rejected=reviews[t]
        cv2.drawMarker(image,(round(float(rejected['displayed_x_px'])),round(float(rejected['displayed_y_px']))),(0,0,255),cv2.MARKER_TILTED_CROSS,25,2,cv2.LINE_AA)
        # Display6 strongest alternatives away from rejected location for readability only.
        alternatives=[r for r in unique if np.hypot(r['x_px']-float(rejected['displayed_x_px']),r['y_px']-float(rejected['displayed_y_px']))>16][:6]
        for k,c in enumerate(alternatives):
            label=chr(65+k);p=(round(c['x_px']),round(c['y_px']));cv2.circle(image,p,13,(255,220,0),2,cv2.LINE_AA)
            cv2.putText(image,label,(p[0]+16,p[1]-7),0,.7,(255,220,0),2,cv2.LINE_AA)
            displayed.append(dict(c,display_label=label,known_ball_identity='not_established'))
        canvas=np.full((1180,1920,3),18,np.uint8);canvas[100:]=image
        cv2.putText(canvas,f'040 local{t} | RED X=user rejected | BLUE A-F=unreviewed alternative candidates',(12,33),0,.8,(240,240,240),2,cv2.LINE_AA)
        cv2.putText(canvas,'Strongest candidates from fixed full+9tiles; omitted candidates remain in CSV. No recovered ball claimed.',(12,72),0,.70,(240,240,240),1,cv2.LINE_AA)
        assert cv2.imwrite(str(OUT/f'candidate_overview_{t}.jpg'),canvas)
    write(OUT/'candidate_counts.csv',counts)
    if displayed:write(OUT/'displayed_alternatives.csv',displayed)
    assert all(sha(Path(p))==h for p,h in protected.items())
    dump(OUT/'report.json',dict(status='fixed_scale_diagnostic_only',frames=len(TIMES),inference_jobs=len(jobs),target_counts=counts,
        no_alternative_identity_confirmed_by_this_script=True,outputs_filled=0,protected_inputs_unchanged=True))
    print(json.dumps(counts,indent=2))

if __name__=='__main__':main()
