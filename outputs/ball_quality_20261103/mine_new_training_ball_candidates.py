"""Mine unlabelled training-only jump candidates for precise center annotation.

Deviation is a sampling cue, never a false-ball label. Do not propagate old
center labels into neighbouring frames or copy validation/audit clips.
"""
import json
from pathlib import Path
import numpy as np
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2
from evaluate_observed_ball_fallback import camera_mask

OUT=ROOT/'ball_training_case_expansion'

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'mining_report.json').exists()
    data=ROOT/'balltrack_manual_finetune_pilot';manifest=load(data/'data_manifest.json')
    labels=read(data/'train_labels.csv');hard=read(data/'training_hard_cases/far_wrong_target_predictions.csv')
    ids=sorted({r['candidate_id'] for r in labels});annotated={cid:{int(r['local_frame']) for r in labels if r['candidate_id']==cid} for cid in ids}
    old_hard={cid:[int(r['local_frame']) for r in hard if r['candidate_id']==cid] for cid in ids}
    protected=dict(manifest['input_hashes']);records=[];best=[]
    base=ROOT.parent/'tracking_quality_20261102/ball_aligned';cams=ROOT.parent/'ball_quality_20261102/temporal_guard/camera_features'
    for cid in ids:
        p=base/cid/'balltrack_predictions.csv';rows=read(p);camera=load(cams/(cid+'.json'));assert camera['baseline_sha256']==sha(p)
        mask=camera_mask(camera['changed_fraction']);protected[str(p)]=sha(p);protected[str(cams/(cid+'.json'))]=sha(cams/(cid+'.json'))
        candidates=[]
        for t in range(90,len(rows)-24):
            if mask[t-12:t+13].any() or min(abs(t-v) for v in annotated[cid])<=8:continue
            if any(abs(t-v)<=60 for v in old_hard[cid]):continue
            if not all(int(rows[j]['visible']) for j in [t-2,t,t+2]):continue
            xy=lambda j:np.array([float(rows[j]['x_px']),float(rows[j]['y_px'])])
            first,last=xy(t-2),xy(t+2);point=xy(t)
            span=float(np.linalg.norm(last-first));residual=float(np.linalg.norm(point-(first+last)/2))
            if span>100 or residual<64:continue
            if not 270<=point[1]<=1058:continue
            r=dict(candidate_id=cid,local_frame=t,source_frame=rows[t]['source_frame'],source_timestamp_sec=rows[t]['source_timestamp_sec'],
                model_x_px=rows[t]['x_px'],model_y_px=rows[t]['y_px'],model_confidence=rows[t]['confidence'],
                reference_first=t-2,reference_last=t+2,reference_endpoint_span_px=span,reference_residual_px=residual,
                role='training_expansion_candidate',label='unreviewed',phase='unreviewed',sampling_semantics='temporal_deviation_is_not_false_detection',
                original_image=str(SOURCE/'data/balltrack/racketvision/match1'/cid/'frames'/f'{t:06d}.jpg'))
            candidates.append(r);records.append(r)
        candidates.sort(key=lambda r:r['reference_residual_px'],reverse=True)
        if candidates:best.append(candidates[0])
    assert records and best
    write(OUT/'all_mined_candidates.csv',records);write(OUT/'top_candidate_per_clip.csv',best)
    canvas=np.full((((len(best)+1)//2)*390,1280,3),20,np.uint8)
    for k,r in enumerate(best):
        im=cv2.imread(r['original_image']);assert im is not None
        tile=cv2.resize(im,(640,360),interpolation=cv2.INTER_AREA)
        cv2.circle(tile,(round(float(r['model_x_px'])/3),round(float(r['model_y_px'])/3)),6,(255,180,30),2)
        panel=np.full((390,640,3),20,np.uint8);panel[30:]=tile
        cv2.putText(panel,f"{r['candidate_id'][-3:]} local{r['local_frame']} - unreviewed candidate",(8,23),0,.62,(245,245,245),1,cv2.LINE_AA)
        canvas[k//2*390:(k//2+1)*390,k%2*640:(k%2+1)*640]=panel
    cv2.imwrite(str(OUT/'candidate_overview.jpg'),canvas)
    assert all(sha(Path(p))==h for p,h in protected.items())
    report=dict(status='training_only_unreviewed_candidates_mined',train_clips=manifest['train_clips'],
        candidate_rows=len(records),clips_with_candidates=len(best),
        parameters=dict(neighbour_offset=2,endpoint_max_span_px=100,query_min_residual_px=64,
            minimum_local_frame=90,candidate_y_range=[270,1058],exclude_manual_label_distance_frames=8,exclude_old_hard_case_distance_frames=60,
            exclude_camera_transition_margin_frames=12),
        warning='Temporal jumps can be true fast balls/bounces. No labels assigned, no training performed, no validation/audit data mined.',
        top_candidates=[dict(candidate_id=r['candidate_id'],local_frame=r['local_frame'],residual_px=r['reference_residual_px']) for r in best],
        protected_input_hashes=protected)
    (OUT/'mining_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='protected_input_hashes'},indent=2))

if __name__=='__main__':main()
