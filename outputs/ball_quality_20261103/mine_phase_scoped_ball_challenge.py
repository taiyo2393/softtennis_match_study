"""Mine new validation-only query frames; no new annotation or score selection.

Use4 validation clips with stored manually reviewed valid-service timestamps.
After-service/end-of-clip margins are query guards, NOT confirmed rally bounds.
"""
import json,math,sys
from pathlib import Path
import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2
from compare_ball_candidate_decoders import components,choose,METHODS
from run_manual_ball_finetune import tensors
from evaluate_observed_ball_fallback import camera_mask

OUT=ROOT/'ball_inplay_challenge'
DATA=ROOT/'balltrack_manual_finetune_pilot'
CLIPS=[12,15,19,20]
STRIDE=2
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')

def pick_query(rows):
    # Predeclared sequence: sustained method disagreement first; competing but
    # agreeing candidates are retained as fallback, without looking at any truth.
    for key in ['method_disagreement','spatial_competition']:
        runs=[];current=[]
        for r in rows:
            if r[key] and not r['near_existing_label'] and not r['camera_transition']:
                if current and r['local_frame']!=current[-1]['local_frame']+STRIDE:
                    runs.append(current);current=[]
                current.append(r)
            elif current:runs.append(current);current=[]
        if current:runs.append(current)
        runs=[run for run in runs if len(run)>=3]
        if runs:
            run=max(runs,key=lambda q:(len(q),-q[0]['local_frame']))
            return run[(len(run)-1)//2],dict(mode=key,run_first=run[0]['local_frame'],run_last=run[-1]['local_frame'],sampled_run_frames=len(run))
    return None,None

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'mining_protocol.json').exists()
    inventory_file=ROOT/'play_start_audit/ball_pilot_start_inventory.csv'
    inventory={r['ball_pilot_id']:r for r in read(inventory_file)}
    manifest=load(DATA/'data_manifest.json');protected=dict(manifest['input_hashes'])
    for role,h in manifest['partition_hashes'].items():
        path=DATA/f'{role}_labels.csv';assert sha(path)==h;protected[str(path)]=h
    protected[str(inventory_file)]=sha(inventory_file)
    for path in [ROOT.parent/'frontcourt_handoff_20260927/rally_label_review.csv',ROOT.parent/'frontcourt_handoff_20260927/human_review_log.csv',
        ROOT.parent/'frontcourt_handoff_20260927/044_serve_attempt_review.csv',ROOT.parent/'frontcourt_handoff_20260927/event_review_ranges.csv']:
        protected[str(path)]=sha(path)
    modelroot=SOURCE/'third_party/RacketVision/source/BallTrack';weight=modelroot/'checkpoints/balltrack_best.pth';protected[str(weight)]=sha(weight)
    for suffix in CLIPS:
        cid=f'match1_motion_pilot_{suffix:03d}';base=SOURCE/'data/balltrack/racketvision/match1'/cid
        for path in [ROOT.parent/'ball_quality_20261102/temporal_guard/camera_features'/f'{cid}.json',
            ROOT.parent/'tracking_quality_20261102/ball_aligned'/cid/'balltrack_predictions.csv',
            base/'runs/threshold030_median120/run_metadata.json',base/'runs/threshold030_median120/median.npz']:
            protected[str(path)]=sha(path)
    assert all(sha(Path(p))==h for p,h in protected.items())
    protocol=dict(status='fixed_before_new_clip_inference',clips=CLIPS,role='validation_only_challenge_never_training',
        source='Stored impact_confirmed valid-serve timestamps, unchanged legacy edited-video clock',
        query_scope='Start1second after stored service timestamp, at least local3; stop1second before clip end',
        scope_caveat='Margins are query guards only; actual play phase unknown until human review. Camera departure is not point end.',
        stride=STRIDE,model=str(weight),threshold=.3,methods=METHODS,
        query_selection='One per clip: midpoint of longest >=3 sampled-frame sustained disagreement run; fallback longest spatial competition run. Ties earliest run.',
        disagreement='Area vs peak centers >=64source pixels apart',spatial_competition='At least2 component centers >=64source pixels apart',
        exclusions='Existing300label frames +/-8; cached camera-transition mask +/-12 source frames',
        no_groundtruth_used_in_query_ranking=True,exact_existing_label_frames_not_queried=True,
        evaluation='Freeze all4 predictions before human labels. Score fixed methods on human-confirmed play only; keep other phase labels as diagnostics.',
        phase_review='Human only needs phase of target, not exact service/end frame. No automatic phase confirmation.',
        independence='Actively selected same-match validation challenge, not representative random testing or unseen match',
        retraining=False,score_parameter_tuning=False,protected_input_hashes=protected,script_sha256=sha(Path(__file__)))
    dump(OUT/'mining_protocol.json',protocol)
    sys.path.insert(0,str(modelroot));from inference import BallInferencer
    infer=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(weight),device='cuda',thre=.3,batchsize=4)
    oldlabels=read(DATA/'train_labels.csv')+read(DATA/'validation_labels.csv')+read(DATA/'audit_labels.csv')
    validation_clips={r['candidate_id'] for r in read(DATA/'validation_labels.csv')}
    known_hashes={h for r in oldlabels for h in json.loads(r['input_frame_sha256_json'])}
    all_rows=[];selected=[];locked=[];scope_rows=[];medians={}
    for suffix in CLIPS:
        cid=f'match1_motion_pilot_{suffix:03d}';assert cid in validation_clips
        entry=inventory[cid];assert entry['legacy_serve_review_status']=='impact_confirmed' and entry['legacy_start_method']=='manual_valid_serve_impact'
        base=SOURCE/'data/balltrack/racketvision/match1'/cid;meta=load(base/'runs/threshold030_median120/run_metadata.json');fps=meta['fps']
        start=max(3,math.ceil((float(entry['legacy_serve_timestamp_sec'])-float(entry['clip_start_sec'])+1)*fps));stop=meta['frame_count']-1-math.ceil(fps)
        old_frames=[int(r['local_frame']) for r in oldlabels if r['candidate_id']==cid]
        camera_path=ROOT.parent/'ball_quality_20261102/temporal_guard/camera_features'/f'{cid}.json';camera=load(camera_path)
        protected[str(camera_path)]=sha(camera_path)
        # Same cached transition semantics used during earlier mining.
        assert camera['baseline_sha256']==sha(ROOT.parent/'tracking_quality_20261102/ball_aligned'/cid/'balltrack_predictions.csv')
        mask=camera_mask(camera['changed_fraction'])
        median_path=base/'runs/threshold030_median120/median.npz';median=np.load(median_path)['median']
        mp=str(median_path);medians[mp]=np.moveaxis(cv2.resize(median,(512,288)).astype(np.float32)/255.,-1,0)
        candidates=[];tables={}
        times=list(range(start,stop+1,STRIDE))
        for first in range(0,len(times),4):
            batch=[]
            for t in times[first:first+4]:
                images=[]
                for j in range(t-3,t+1):
                    im=cv2.imread(str(base/'frames'/f'{j:06d}.jpg'));assert im is not None and im.shape[:2]==(1080,1920)
                    images.append(cv2.resize(im,(512,288)))
                batch.append(dict(frames=np.stack(images),median_key=mp,row={}))
            with torch.inference_mode(),torch.amp.autocast('cuda'):
                maps,_,_=infer.model.forward(frames=tensors(batch,medians,infer.device))
            for t,hm in zip(times[first:first+4],maps[:,0].float().cpu().numpy(),strict=True):
                cs=components(hm);picks={m:choose(cs,m) for m in METHODS};a=picks['bbox_area'];p=picks['component_peak']
                distance=math.hypot(a['x_px']-p['x_px'],a['y_px']-p['y_px']) if a and p else 0
                span=max((math.hypot(x['x_px']-y['x_px'],x['y_px']-y['y_px']) for x in cs for y in cs),default=0)
                transition=any(bool(mask[j]) for j in range(max(0,t-12),min(len(mask),t+13)))
                row=dict(candidate_id=cid,local_frame=t,source_frame=int(entry['clip_first_source_frame'])+t,
                    component_count=len(cs),method_disagreement=int(distance>=64),spatial_competition=int(span>=64),
                    area_peak_distance_px=distance,max_candidate_separation_px=span,
                    near_existing_label=int(any(abs(t-j)<=8 for j in old_frames)),camera_transition=int(transition))
                candidates.append(row);tables[t]=picks
            if first%120==0:print(cid,'sampled',first,'/',len(times),flush=True)
        q,why=pick_query(candidates);all_rows.extend(candidates)
        scope_rows.append(dict(candidate_id=cid,provisional_first_local_frame=start,provisional_last_local_frame=stop,
            source_valid_serve_time_sec=entry['legacy_serve_timestamp_sec'],sampled_frames=len(candidates),
            selected=int(q is not None),phase_status='human_pending',exact_end_not_inferred=True))
        if q:
            t=q['local_frame'];paths=[base/'frames'/f'{j:06d}.jpg' for j in range(t-3,t+1)];hashes=[sha(p) for p in paths]
            assert not set(hashes)&known_hashes
            case_id=chr(ord('A')+len(selected));sample_id=f'{cid}_frame_{t:06d}'
            selected.append(dict(q,case_id=case_id,sample_id=sample_id,play_window_id=entry['play_window_id'],
                rally_window_candidate_id=entry['rally_window_candidate_id'],selection_reason=json.dumps(why),source_image=str(paths[-1]),
                source_image_sha256=hashes[-1],clip=meta['clip'],clip_first_source_frame=int(entry['clip_first_source_frame']),
                provisional_scope_start=start,provisional_scope_end=stop,legacy_serve_time=entry['legacy_serve_timestamp_sec'],
                median_path=mp,median_sha256=sha(median_path),input_frames_json=json.dumps([str(p) for p in paths]),input_frame_sha256_json=json.dumps(hashes)))
            for method,c in tables[t].items():
                locked.append(dict(case_id=case_id,sample_id=sample_id,candidate_id=cid,local_frame=t,method=method,x_px=c['x_px'],y_px=c['y_px'],
                    visible=1,ranking_score=c[method],bbox_mean=c['bbox_mean'],component_index=c['component_index']))
    write(OUT/'mined_frame_inventory.csv',all_rows);write(OUT/'provisional_search_scopes.csv',scope_rows)
    assert selected,'No suitable queries; do not fabricate cases.'
    write(OUT/'selected_query_frames.csv',selected);write(OUT/'frozen_method_predictions.csv',locked)
    assert all(sha(Path(p))==h for p,h in protected.items())
    dump(OUT/'mining_report.json',dict(selected_cases=len(selected),selected=[{k:r[k] for k in ['case_id','candidate_id','local_frame','selection_reason']} for r in selected],
        frozen_predictions_sha256=sha(OUT/'frozen_method_predictions.csv'),validation_only=True,phase_not_yet_confirmed=True,
        existing_labels_models_reviewed_outputs_unchanged=True,protected_input_hashes=protected))
    print(json.dumps(load(OUT/'mining_report.json')['selected'],indent=2))

if __name__=='__main__':main()
