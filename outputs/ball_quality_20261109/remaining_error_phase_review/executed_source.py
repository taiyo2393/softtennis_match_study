"""Prepare source-only phase context for the three remaining training additions."""
import json,subprocess,shutil
from pathlib import Path
from collections import Counter
import numpy as np
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2

OUT=ROOT.parent/'ball_quality_20261109/remaining_error_phase_review'
GATE=ROOT.parent/'ball_quality_20261109/training_tile_gate_grouped'

def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'protocol.json').exists()
    residual=[r for r in read(GATE/'remaining_added_errors.csv') if r['method']=='logistic10_oof']
    assert len(residual)==3
    plans=[dict(review_id='A',candidate_id='match1_motion_pilot_001',target_local_frames=[464]),
           dict(review_id='B',candidate_id='match1_motion_pilot_006',target_local_frames=[279,341])]
    scope=ROOT/'play_start_audit/ball_pilot_start_inventory.csv'
    ready=SOURCE/'data/segments/analysis_ready_main_camera_segments.csv'
    rally_labels=ROOT.parent/'frontcourt_handoff_20260927/rally_label_review.csv'
    protected={str(p):sha(p) for p in [GATE/'result.json',GATE/'verification.json',GATE/'remaining_added_errors.csv',scope,ready,rally_labels,
        ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json',
        ROOT/'balltrack_expanded_centers_pilot/expanded_train_labels.csv',
        ROOT.parent/'ball_quality_20261104/completed_review/manual_sequence_labels.csv',ROOT.parent/'ball_quality_20261104/evaluation_report.json']}
    protocol=dict(status='fixed_before_context_render',date_jst='2026-11-09',plans=plans,
        selection='All three remaining added errors of the frozen grouped learned gate; failure-enriched diagnostic review,NOT new validation.',
        question='At each highlighted target frame, is the rally still in play,preparation/ended,or uncertain? No precise boundary or foot-contact request.',
        media='Entire original cached clip,1280x720 image plus80px header,1x,extra30displayframes hold at each target. No ball/player prediction markers,score inference,or guessed GT shown.',
        scope='Only review these3target moments. Do not infer complete phase intervals or retroactively remove them from performance counts.',
        camera_evidence='Existing automated main-view mask is not independent phase truth; known001target is outside its ready interval,006targets inside.',
        unchanged='Keep all previous training/OOF metrics and adoption decisions. Any main-play metric later needs phase review sampled independently of errors across methods.',
        no_new_training=True,no_automatic_phase_labels=True,protected_input_hashes=protected,script_sha256=sha(Path(__file__)))
    dump(OUT/'protocol.json',protocol);shutil.copy2(Path(__file__),OUT/'executed_source.py')
    inventory={r['ball_pilot_id']:r for r in read(scope)};view=read(ready);labels=read(rally_labels)
    inputs={};cases=[];contexts=[]
    for plan in plans:
        cid=plan['candidate_id'];base=SOURCE/'data/balltrack/racketvision/match1'/cid;mp=base/'runs/threshold030_median120/run_metadata.json'
        meta=load(mp);protected[str(mp)]=sha(mp);fps=meta['fps'];n=meta['frame_count'];offset=int(inventory[cid]['clip_first_source_frame'])
        output=OUT/('phase_'+plan['review_id']+'_source_context_h264.mp4')
        cmd=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1280x800','-r',str(fps),'-i','pipe:0',
            '-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
        expected=0;targets=[]
        with (OUT/('phase_'+plan['review_id']+'_render.log')).open('wb') as log:
            proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
            for t in range(n):
                p=base/'frames'/f'{t:06d}.jpg';inputs[str(p)]=sha(p);im=cv2.imread(str(p));assert im.shape==(1080,1920,3)
                frame=np.full((800,1280,3),18,np.uint8);frame[80:]=cv2.resize(im,(1280,720),interpolation=cv2.INTER_AREA)
                cv2.putText(frame,f'Phase {plan["review_id"]} | pilot{cid[-3:]} local{t} / source{offset+t} | 1x',(12,29),0,.75,(240,240,240),1,cv2.LINE_AA)
                cv2.putText(frame,'Yellow border = target moment. Judge phase only; no model positions are shown.',(12,61),0,.7,(220,220,220),1,cv2.LINE_AA)
                if t in plan['target_local_frames']:
                    cv2.rectangle(frame,(2,81),(1277,797),(0,225,255),3)
                    cv2.imwrite(str(OUT/(plan['review_id']+'_target_'+str(t)+'.jpg')),frame)
                    targets.append(dict(local_frame=t,source_frame=offset+t,first_display_frame=expected,last_display_frame=expected+30))
                for repeat in range(31 if t in plan['target_local_frames'] else 1):proc.stdin.write(frame.tobytes());expected+=1
            proc.stdin.close();assert proc.wait()==0
        cap=cv2.VideoCapture(str(output));decoded=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape==(800,1280,3);decoded+=1
        cap.release();assert decoded==expected
        contexts.append(dict(review_id=plan['review_id'],candidate_id=cid,video=str(output),video_sha256=sha(output),fps=fps,
            source_clip_frames=n,display_frames=decoded,duration_sec=decoded/fps,targets=targets))
        rally=next(r for r in labels if r['candidate_id']==inventory[cid]['rally_window_candidate_id'])
        for target in targets:
            source=target['source_frame'];ready_matches=[r for r in view if r['video_id']=='match1' and r['usable_for_analysis'].lower()=='true' and int(r['start_source_frame'])<=source<=int(r['end_source_frame'])]
            cases.append(dict(review_id=plan['review_id'],candidate_id=cid,local_frame=target['local_frame'],source_frame=source,
                phase_status='unreviewed',reviewer_id='',reviewed_at_utc='',notes='',
                automated_ready_main_view=int(bool(ready_matches)),ready_human_checked=';'.join(r['human_checked'] for r in ready_matches),
                prior_boundary_review_status=rally['boundary_review_status'],scope='target_moment_only_not_rally_boundary'))
    write(OUT/'phase_review_labels.csv',cases);dump(OUT/'contexts.json',contexts);dump(OUT/'source_image_hashes.json',inputs)
    assert all(sha(Path(p))==h for p,h in protected.items()) and all(sha(Path(p))==h for p,h in inputs.items())
    manifest=dict(status='three_phase_judgments_pending',review_videos=2,target_frames=3,new_labels=0,
        initial_review_labels_sha256=sha(OUT/'phase_review_labels.csv'),contexts_sha256=sha(OUT/'contexts.json'),
        total_display_frames=sum(c['display_frames'] for c in contexts),protected_input_hashes=protected,
        no_performance_reclassification=True,no_canonical_updates=True)
    dump(OUT/'manifest.json',manifest);print(json.dumps({k:v for k,v in manifest.items() if k!='protected_input_hashes'},indent=2))

if __name__=='__main__':main()
