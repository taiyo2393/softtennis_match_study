"""Apply the frozen v3 heuristic on predeclared clips without event labels."""
import argparse,json,subprocess,wave
from datetime import datetime,timezone
from pathlib import Path
import cv2
import numpy as np
from extract_serve_start_features import ROOT,SOURCE,sha
from repair_start_pose_crops import contained,overlap
from segment_serve_phases import PARAMS,court_frames,context_series,phase_filter
from propose_serve_starts import tracks,propose

OUT=ROOT/'serve_phase_fixed_application'
FROZEN=ROOT/'serve_phase_pilot_v3/frozen_development_protocol.json'

def preflight():
    frozen=json.loads(FROZEN.read_text());p=frozen['protocol']
    for name,key in [('segment_serve_phases.py','script_sha256'),('repair_start_pose_crops.py','pose_repair_script_sha256'),('propose_serve_starts.py','gesture_code_sha256')]:
        assert sha(ROOT/name)==p[key],f'Frozen code changed: {name}'
    assert PARAMS==p['parameters']
    cal=SOURCE/'config/match1_court_plane_calibration.json'
    assert sha(cal)==p['calibration_sha256']
    return np.array(json.loads(cal.read_text(encoding='utf-8'))['image_to_court_homography'])

def poses_at(net,cid,t):
    im=cv2.imread(str(SOURCE/'data/balltrack/racketvision/match1'/cid/'frames'/f'{t:06d}.jpg'))
    assert im is not None and im.shape[:2]==(1080,1920)
    people=[]
    for l,top,r,b,origin in [(480,360,1536,720,'far_court'),(0,320,1920,1080,'court')]:
        result=net.predict(im[top:b,l:r],imgsz=1920,conf=.25,iou=.7,device=0,verbose=False)[0]
        for box,kp in zip(result.boxes.data.cpu().numpy(),result.keypoints.data.cpu().numpy()):
            box=box.copy();kp=kp.copy();box[[0,2]]+=l;box[[1,3]]+=top;kp[:,0]+=l;kp[:,1]+=top
            x1,y1,x2,y2=box[:4];height=y2-y1;cx=(x1+x2)/2
            if height<55 or y2<.48*1080 or not .12*1920<=cx<=.94*1920:continue
            p=dict(box=box[:4].tolist(),confidence=float(box[4]),keypoints=kp.tolist(),origin=origin)
            if not contained(p):continue
            if any(overlap(p['box'],q['box'])>.5 for q in people):continue
            people.append(p)
    return dict(local_frame=t,people=people)

def check_equivalence(net):
    # Check the reimplemented extractor against saved repaired development
    # features before applying it elsewhere. No start/outcome labels are read.
    checks=[]
    for pilot,t in [('016',60),('016',700),('002',700),('002',740)]:
        cid=f'match1_motion_pilot_{pilot}'
        old=json.loads((ROOT/'serve_phase_pilot_v3'/cid/'features.json').read_text())['frames'][t//2]
        new=poses_at(net,cid,t);assert len(old['people'])==len(new['people'])
        for a,b in zip(old['people'],new['people'],strict=True):
            assert a['origin']==b['origin']
            assert np.allclose(a['box'],b['box'],atol=.05,rtol=0)
            assert np.allclose(a['keypoints'],b['keypoints'],atol=.05,rtol=0)
        checks.append(dict(pilot=pilot,frame=t,people=len(new['people']),match=True))
    (OUT/'extractor_equivalence.json').write_text(json.dumps(dict(checks=checks,tolerance_px=.05,
        limitation='Four development-frame checks of software equivalence, not a tracking/start accuracy test.'),indent=2),encoding='utf-8')

def extract_audio(meta,dest):
    audio=dest/'audio16k.wav'
    subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-i',meta['clip'],'-vn','-ac','1','-ar','16000','-c:a','pcm_s16le',str(audio)],check=True)
    with wave.open(str(audio),'rb') as wav:
        assert wav.getnchannels()==1 and wav.getsampwidth()==2 and wav.getframerate()==16000
        signal=np.frombuffer(wav.readframes(wav.getnframes()),dtype='<i2').astype(np.float32)/32768.
    frames=np.lib.stride_tricks.sliding_window_view(signal,320)[::160]
    energy=20*np.log10(np.sqrt(np.mean(frames*frames,axis=1))+1e-7)
    times=(np.arange(len(energy))*160+160)/16000.
    rise=np.array([max(0.,float(energy[i]-np.median(energy[max(0,i-7):i]))) if i else 0. for i in range(len(energy))])
    np.savez_compressed(dest/'audio_features.npz',time_sec=times,energy_db=energy,rise_db=rise)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',choices=['003','008'],required=True);a=ap.parse_args()
    H=preflight();OUT.mkdir(exist_ok=True)
    plan=OUT/'application_plan.json'
    if not plan.exists():
        plan.write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),pilots=['003','008'],
            selection='003 is next unused clip;008 has a legacy start timestamp near clip entry. Both chosen before new outputs.',
            thresholds_will_not_be_retuned_on_these_outputs=True,frozen_protocol_sha256=sha(FROZEN),
            independent_accuracy_evaluation=False,scope='Previously viewed same-match clips, new applications of this fixed heuristic.'),indent=2),encoding='utf-8')
    cid=f'match1_motion_pilot_{a.pilot}';dest=OUT/cid;dest.mkdir(exist_ok=True)
    assert not (dest/'phases.json').exists()
    base=SOURCE/'data/balltrack/racketvision/match1'/cid;mp=base/'runs/threshold030_median120/run_metadata.json'
    meta=json.loads(mp.read_text());model=SOURCE/'models/yolo11m-pose.pt'
    saved=json.loads((ROOT/'serve_phase_pilot_v3/match1_motion_pilot_016/protocol.json').read_text())
    assert sha(model)==saved['model_sha256']
    # Pose/geometry/audio settings match the frozen development inputs.
    protocol=dict(pilot=cid,frames=meta['frame_count'],fps=meta['fps'],sampling_stride=2,
        metadata=str(mp),metadata_sha256=sha(mp),clip_sha256=sha(Path(meta['clip'])),
        clock='legacy_edited_match1_cached_clip_local_frames_not_match1_original',
        frozen_protocol_file_sha256=sha(FROZEN),runner_sha256=sha(Path(__file__)),model_sha256=sha(model),
        imgsz=1920,confidence=.25,iou=.7,regions=[[480,360,1536,720],[0,320,1920,1080]],
        crop_edge_margin_px=4,dedup_iou=.5,labels_read=False)
    (dest/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    if not (dest/'features.json').exists():
        from ultralytics import YOLO
        net=YOLO(str(model))
        if not (OUT/'extractor_equivalence.json').exists():check_equivalence(net)
        rows=[]
        for t in range(0,meta['frame_count'],2):
            rows.append(poses_at(net,cid,t))
            if t%120==0:print(cid,t,'/',meta['frame_count'],flush=True)
        (dest/'features.json').write_text(json.dumps(dict(frames=rows,protocol_sha256=sha(dest/'protocol.json'),labels_read=False)),encoding='utf-8')
    extract_audio(meta,dest);audio=np.load(dest/'audio_features.npz')
    data=json.loads((dest/'features.json').read_text());fps=meta['fps']
    repaired,_=propose(tracks(data['frames']),audio,fps)
    selected=court_frames(data['frames'],H);series,tdata=context_series(selected,fps)
    proposed,_=propose(tdata,audio,fps);result=phase_filter(proposed,series,fps,H,tdata)
    result.update(labels_read=False,serve_outcome_inferred=False,feature_sha256=sha(dest/'features.json'),
        frozen_protocol_file_sha256=sha(FROZEN),runner_sha256=sha(Path(__file__)),
        repaired_only_candidates=[{k:v for k,v in p.items() if k!='evidence_person_boxes'} for p in repaired])
    (dest/'phases.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (dest/'motion_context.json').write_text(json.dumps(dict(frames=series,selection=selected)),encoding='utf-8')
    summary=dict(pilot=cid,repaired_only=len(repaired),court_filtered=len(proposed),
        retained=[[p['contact_candidate_first_local_frame'],p['contact_candidate_last_local_frame']] for p in result['accepted_candidates']],
        suppressed=sum(p['review_action']=='suppress_during_activity' for p in result['candidates']),
        deferred=sum(p['review_action']=='defer_insufficient_start_evidence' for p in result['candidates']),
        complete_four_samples=sum(p['four_observed'] for p in series),quiet_samples=sum(p['four_player_quiet'] for p in series),samples=len(series))
    (dest/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()
