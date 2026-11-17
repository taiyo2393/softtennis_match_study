"""Replay the frozen52time experiment and describe missing responses without tuning."""
import json,sys,math
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from PIL import Image,ImageDraw,ImageFont
from integrate_044_reviewed_tracks import ROOT,SOURCE,read,load,write,sha,cv2
from run_manual_ball_finetune import tensors
from compare_ball_candidate_decoders import components

BASE=ROOT.parent/'ball_quality_20261104'
OUT=ROOT.parent/'ball_quality_20261109'/'missing_response_diagnosis'

def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')

def measure(hm,g):
    x,y=float(g['truth_x_px']),float(g['truth_y_px'])
    yy,xx=np.mgrid[:288,:512]
    mask=(xx*3.75-x)**2+(yy*3.75-y)**2<=16**2
    assert mask.any()
    masked=np.where(mask,hm,-1);ly,lx=np.unravel_index(np.argmax(masked),hm.shape)
    gy,gx=np.unravel_index(np.argmax(hm),hm.shape)
    fx,fy=x/3.75,y/3.75;ix,iy=int(fx),int(fy);dx,dy=fx-ix,fy-iy
    sampled=(hm[iy,ix]*(1-dx)*(1-dy)+hm[iy,ix+1]*dx*(1-dy)+hm[iy+1,ix]*(1-dx)*dy+hm[iy+1,ix+1]*dx*dy)
    return dict(peak_within16px=float(hm[ly,lx]),bilinear_response_at_manual_center=float(sampled),
        global_peak=float(hm[gy,gx]),global_peak_source_x=gx*3.75,global_peak_source_y=gy*3.75,
        global_peak_distance_from_manual_px=math.hypot(gx*3.75-x,gy*3.75-y),
        pixels_above03_near16=int((hm[mask]>.3).sum()),full_components_above03=len(components(hm)),
        local_peak_gap_to03=float(.3-hm[ly,lx]),
        response_note='Uncalibrated heatmap value; near16 uses gridindex*3.75, no subpixel/physical interpretation.')

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'protocol.json').exists(),'Preserve completed diagnosis; no alternative settings.'
    manifest=load(BASE/'manifest.json');review=load(BASE/'completed_review.json')
    assert review['correct16px_each']==33 and review['missing_visible_each']==12
    labels_path=BASE/'completed_review/manual_sequence_labels.csv';labels=read(labels_path)
    assert sha(labels_path)==load(BASE/'evaluation_report.json')['labels_sha256']
    visible=[g for g in labels if g['phase_status']=='play' and g['truth_visibility']=='visible'];assert len(visible)==45
    modelroot=SOURCE/'third_party/RacketVision/source/BallTrack';weights=modelroot/'checkpoints/balltrack_best.pth'
    protected=dict(manifest['protected_input_hashes'])
    for name in ['manifest.json','sequence_protocol.json','reserved_windows.json','query_model_inputs.csv','frozen_predictions.csv',
                 'inference_batches.csv','frozen_component_cache.json','source_image_hashes.json','manual_sequence_labels.csv',
                 'annotation_events.jsonl','completed_review.json','evaluation_report.json','completed_review/manual_sequence_labels.csv']:
        p=BASE/name;protected[str(p)]=sha(p)
    protected[str(weights)]=sha(weights)
    assert all(sha(Path(p))==h for p,h in protected.items())
    protocol=dict(status='fixed_before_new_heatmaps',date_jst='2026-11-09',
        scope='Describe ALL45 human-visible targets including33correct controls and12missing targets. Exclude7 unknown centers; no guessed ground truth.',
        model='Only unchanged original TrackNet',threshold=.3,training=False,threshold_search=False,
        replay='Exactly the saved ascending per-clip batch4 from the original84context outputs; median+BGRt-3..t,512x288,/255,CUDA AMP,cudnn.benchmark=False.',
        validation='Every replayed contour/component dictionary must equal the original cache, not merely match presence.',
        measures='Near16px peak,manual-center bilinear response,global peak and distance,threshold gap,above0.3pixels/components. Values are not probabilities.',
        figure='Lowest and highest near16response among ALL12missing cases; deterministic caseID tie breaker. Descriptive extrema, not new evaluation samples. Source128pxcrop x3,actual512input nearest-expanded,shared0..1heatmap.',
        claims='Direct model-output mechanism only. Does not isolate resolution,occlusion,training-data or network root cause. Existing evaluation remains frozen.',
        forbidden='No adopted lower threshold,updated predictions,individual frame patch,training or new annotations.',
        protected_input_hashes=protected,script_sha256=sha(Path(__file__)))
    dump(OUT/'protocol.json',protocol)
    sys.path.insert(0,str(modelroot));from inference import BallInferencer
    torch.backends.cudnn.benchmark=False
    infer=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(weights),device='cuda',thre=.3,batchsize=4)
    infer.model.eval().requires_grad_(False)
    windows=load(BASE/'reserved_windows.json');cache=load(BASE/'frozen_component_cache.json')
    hashes=load(BASE/'source_image_hashes.json');batches=read(BASE/'inference_batches.csv')
    maps={};input_hashes={};medians={};batch_log=[]
    for w in windows:
        cid=w['candidate_id'];fd=Path(w['frame_dir']);mp=Path(w['median_path']);assert sha(mp)==protected[str(mp)]
        medians[str(mp)]=np.moveaxis(cv2.resize(np.load(mp)['median'],(512,288)).astype(np.float32)/255.,-1,0)
        keys=sorted(map(int,cache[cid]))
        for start in range(0,len(keys),4):
            times=keys[start:start+4];batch=[]
            for t in times:
                original=next(r for r in batches if r['candidate_id']==cid and int(r['local_frame'])==t)
                assert json.loads(original['batch_frames_json'])==times
                ims=[]
                for j in range(t-3,t+1):
                    p=fd/f'{j:06d}.jpg';h=sha(p);assert hashes[str(p)]==h;input_hashes[str(p)]=h
                    im=cv2.imread(str(p));assert im.shape==(1080,1920,3);ims.append(cv2.resize(im,(512,288)))
                batch.append(dict(frames=np.stack(ims),median_key=str(mp),row={}))
            with torch.inference_mode(),torch.amp.autocast('cuda'):
                pred,_,_=infer.model.forward(frames=tensors(batch,medians,infer.device))
            for t,hm in zip(times,pred[:,0].float().cpu().numpy(),strict=True):
                assert components(hm)==cache[cid][str(t)],(cid,t,'Replay mismatch')
                maps[cid,t]=hm
                batch_log.append(dict(candidate_id=cid,local_frame=t,batch_frames_json=json.dumps(times),saved_components_match=True))
        print('Matched original context:',cid,len(keys),flush=True)
    assert len(maps)==84
    np.savez_compressed(OUT/'replayed_heatmaps.npz',candidate_ids=np.array([k[0] for k in maps]),local_frames=np.array([k[1] for k in maps]),heatmaps=np.stack(list(maps.values())).astype(np.float16))
    old={(r['sample_id'],r['method']):r for r in read(BASE/'frozen_predictions.csv')}
    records=[]
    for g in visible:
        p=old[g['sample_id'],'bbox_area'];rec=measure(maps[g['candidate_id'],int(g['local_frame'])],g)
        status='correct_control' if int(p['visible']) else 'missing_visible'
        if status=='correct_control':assert math.hypot(float(p['x_px'])-float(g['truth_x_px']),float(p['y_px'])-float(g['truth_y_px']))<=16
        else:assert rec['full_components_above03']==0 and rec['global_peak']<=.3
        records.append(dict(**{k:g[k] for k in ['case_id','window_id','candidate_id','local_frame','source_frame','truth_x_px','truth_y_px']},outcome=status,**rec))
    write(OUT/'visible_response_records.csv',records);write(OUT/'replayed_batches.csv',batch_log);dump(OUT/'input_image_hashes.json',input_hashes)
    missing=sorted([r for r in records if r['outcome']=='missing_visible'],key=lambda r:(r['peak_within16px'],r['case_id']))
    assert len(missing)==12
    write(OUT/'missing_response_records.csv',missing)
    groups=[]
    for wid in ['A','B','C','D']:
        for state in ['correct_control','missing_visible']:
            group=[r for r in records if r['window_id']==wid and r['outcome']==state]
            if not group:continue
            peaks=[r['peak_within16px'] for r in group]
            groups.append(dict(window_id=wid,outcome=state,count=len(group),local_peak_min=min(peaks),local_peak_median=float(np.median(peaks)),local_peak_max=max(peaks)))
    write(OUT/'window_response_summary.csv',groups)
    examples=[missing[0],missing[-1]];dump(OUT/'figure_cases.json',examples)
    # Common source window and scales; do not brighten/normalise weak maps per case.
    font=ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf',20);big=ImageFont.truetype(r'C:\Windows\Fonts\arial.ttf',24)
    canvas=Image.new('RGB',(1200,1120),(18,26,34));draw=ImageDraw.Draw(canvas)
    draw.text((18,10),'Missing cases: lowest / highest local response | shared heatmap scale 0..1',font=big,fill='white')
    for index,r in enumerate(examples):
        w=next(w for w in windows if w['candidate_id']==r['candidate_id']);t=int(r['local_frame'])
        im=cv2.imread(str(Path(w['frame_dir'])/f'{t:06d}.jpg'));tx,ty=float(r['truth_x_px']),float(r['truth_y_px']);x0,y0=round(tx)-64,round(ty)-64
        actual_input=cv2.resize(im,(512,288));expanded=cv2.resize(actual_input,(1920,1080),interpolation=cv2.INTER_NEAREST)
        heat=maps[r['candidate_id'],t];colored=cv2.applyColorMap(np.clip(heat*255,0,255).astype(np.uint8),cv2.COLORMAP_TURBO)
        expanded_heat=cv2.resize(colored,(1920,1080),interpolation=cv2.INTER_NEAREST)
        top=55+index*490;detail=f'{r["case_id"]} / local{t} | peak16={r["peak_within16px"]:.6f} | global={r["global_peak"]:.6f}'
        draw.text((18,top),detail,font=big,fill='white')
        for col,(image,title) in enumerate([(im,'Source image crop'),(expanded,'Actual 512x288 input, expanded'),(expanded_heat,'Model response (no normalization)')]):
            left=16+col*396;panel=image[y0:y0+128,x0:x0+128].copy();assert panel.shape==(128,128,3)
            # Draw a thin cross after enlargement, keeping the actual ball pixels readable.
            tile=cv2.resize(panel,(384,384),interpolation=cv2.INTER_NEAREST)
            px,py=round((tx-x0)*3),round((ty-y0)*3)
            cv2.drawMarker(tile,(px,py),(0,0,0),cv2.MARKER_CROSS,18,3,cv2.LINE_AA)
            cv2.drawMarker(tile,(px,py),(255,255,255),cv2.MARKER_CROSS,18,1,cv2.LINE_AA)
            draw.text((left,top+34),title,font=font,fill='white')
            draw.text((left,top+59),'White cross: existing human center',font=font,fill=(205,220,230))
            canvas.paste(Image.fromarray(cv2.cvtColor(tile,cv2.COLOR_BGR2RGB)),(left,top+88))
    gradient=np.tile(np.arange(256,dtype=np.uint8)[None,:],(12,1));bar=cv2.applyColorMap(gradient,cv2.COLORMAP_TURBO)
    bar=cv2.resize(bar,(1152,12),interpolation=cv2.INTER_NEAREST)
    canvas.paste(Image.fromarray(cv2.cvtColor(bar,cv2.COLOR_BGR2RGB)),(18,1045))
    draw.text((18,1061),'0.0',font=font,fill='white');draw.text((330,1061),'0.3 fixed threshold',font=font,fill='white');draw.text((1110,1061),'1.0',font=font,fill='white')
    draw.line((364,1041,364,1060),fill='white',width=2);canvas.save(OUT/'response_extremes.png')
    assert all(sha(Path(p))==h for p,h in protected.items()) and all(sha(Path(p))==h for p,h in input_hashes.items())
    result=dict(status='matched_original_detector_response_diagnosis_complete',replayed_contexts=84,visible_targets=45,missing_targets=12,correct_controls=33,
        missing_peak_range=[missing[0]['peak_within16px'],missing[-1]['peak_within16px']],window_summaries=groups,
        missing_global_peak_near_manual16px=sum(r['global_peak_distance_from_manual_px']<=16 for r in missing),
        examples=[r['case_id'] for r in examples],no_threshold_change=True,no_new_training=True,no_new_labels=True,
        frozen_evaluation_unchanged=True,operational_changes=False,protected_input_hashes=protected,
        figure_sha256=sha(OUT/'response_extremes.png'))
    dump(OUT/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='protected_input_hashes'},indent=2))

if __name__=='__main__':main()
