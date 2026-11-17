"""Prepare one reviewed-rally usability/time pilot; no training or new predictions."""
import json
import shutil
import subprocess
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,SOURCE,read,load,write,sha,cv2,np

OUT=ROOT.parent/'ball_quality_20261109/rally_repair_pilot_044_v2'


def dump(path,value): path.write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    assert not (OUT/'protocol.json').exists(),'Never replace a pilot or human answers.'
    registry=load(ROOT/'reviewed_ball_overrides.json')['clips']['match1_motion_pilot_016']
    ball_path=Path(registry['reviewed_output']); assert sha(ball_path)==registry['reviewed_output_sha256']
    ball={int(r['local_frame']):r for r in read(ball_path)}
    gt_path=ROOT/'balltrack_manual_finetune_pilot/audit_labels.csv'
    known={int(r['local_frame']):r for r in read(gt_path) if r['candidate_id']=='match1_motion_pilot_016' and r['truth_visibility']=='visible'}
    scope=load(ROOT/'play_start_audit/044_scope_report.json')
    assert scope['candidate_main_play_local_frames']==[739,1140]
    frame_dir=SOURCE/'data/balltrack/racketvision/match1/match1_motion_pilot_016/frames'
    meta_path=frame_dir.parent/'runs/threshold030_median120/run_metadata.json'; meta=load(meta_path)
    fps=float(meta['fps']); offset=int(ball[0]['source_frame'])
    protected_paths=[ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json',
        ball_path,gt_path,meta_path,ROOT/'play_start_audit/044_scope_report.json',
        ROOT.parent/'frontcourt_handoff_20260927/rally_label_review.csv',ROOT.parent/'frontcourt_handoff_20260927/human_review_log.csv',
        ROOT/'integrated_044_v4/frame_observation_status.csv',ROOT/'integrated_044_v4/protocol.json',
        ROOT/'serve_rally_repair_pilot.py',ROOT/'rally_repair_pilot.html']
    protected={str(p):sha(p) for p in protected_paths}
    protocol=dict(status='fixed_before_new_review',date_jst='2026-11-09',purpose='Operator-assisted remaining repair burden and UI calibration;not blind detector accuracy or model training.',
        rally='match1_rally_window_044',ball_clip='match1_motion_pilot_016',clock='legacy_edited_match1',
        interval='Candidate interior739..1140;review even frames740..1140 step2(201times). Context730..1155includes provisional contact/end brackets. Boundary precision is not newly established;not a full-source-frame rally benchmark.',
        baseline='Frozen CURRENT human-modified044ball output,not raw automatic detector. Prior correction time unknown. Measure only remaining new review/correction time.',
        reuse='Six exact-frame prior human centers reused without new clicks. Earlier candidate-identity reviews remain provenance only;not converted to precise-center truth.',
        annotation='Predictions SHOWN. Explicit accept_prediction,click correction,not_visible,uncertain,nonplay. Accepted predicted centers are assisted usability labels,NOT independent pixel-center GT. New clicked coordinates are marked separately.',
        time='Explicit user start/pause;server monotonic intervals logged on heartbeat/save/pause. Hidden/unfocused page auto-pauses. Heartbeat gap capped15s with excess recorded as unmeasured gap;browser/process closure may lose up to one unflushed segment. Report captured active time AND gaps,not unqualified stopwatch precision.',
        timing_scope='Instruction/setup and historical work excluded;context viewing after start included. Per-frame accepted vsedited decisions describe this task;no general full-automation labor estimate.',
        frame_sampling='About29.97reviewtimes/sec;unsampled source frames remain unreviewed. No interpolation or all-frame success claim.',
        analysis='Keep pre-existing6,accepted model positions,manual corrections,unobservable,uncertain,nonplay separate. Remaining missing/changes are usability descriptions;not independent recall/precision. Per-rally case,not independent201trials.',
        progression='First044workflow calibration;then apply fixed workflow to further rallies with reviewed scope. Do not demand exact foot contact or repeat known serve validity/score-side questions.',
        protected_input_hashes=protected,script_sha256=sha(Path(__file__)),new_inference=False,new_training=False,canonical_tracking_changed=False)
    dump(OUT/'protocol.json',protocol);shutil.copy2(Path(__file__),OUT/'executed_prepare_source.py')
    cases,labels,baseline=[],[],[];image_hashes={}
    for number,t in enumerate(range(740,1141,2),1):
        b=ball[t];case_id=f'R{number:03d}';path=frame_dir/f'{t:06d}.jpg';image_hashes[str(path)]=sha(path)
        old=known.get(t);reused=old is not None
        cases.append(dict(case_id=case_id,local_frame=t,source_frame=int(b['source_frame']),source_image=str(path),source_image_sha256=sha(path),
            prediction_visible=int(b['visible']),prediction_x_px=float(b['x_px']),prediction_y_px=float(b['y_px']),
            prior_candidate_review=b['ball_candidate_review'],prior_review_id=b['human_review_id'],reused_manual_center=reused,
            video_target_time_sec=(t-730+.25)/fps))
        labels.append(dict(case_id=case_id,local_frame=t,source_frame=b['source_frame'],review_status='reused_manual_center' if reused else 'unreviewed',
            decision='existing_manual_center' if reused else '',x_px=old['truth_x_px'] if reused else '',y_px=old['truth_y_px'] if reused else '',
            coordinate_source='prior_human_click' if reused else '',independent_new_center_gt=0,
            prior_center_source_file=str(gt_path) if reused else '',prior_center_source_sha256=sha(gt_path) if reused else '',
            reviewer_id='human_user_prior' if reused else '',reviewed_at_utc='',notes='scope is provisional interior;not individually re-labelled' if reused else ''))
        baseline.append(b)
    assert len(cases)==201 and sum(c['reused_manual_center'] for c in cases)==6
    write(OUT/'review_labels.csv',labels);shutil.copy2(OUT/'review_labels.csv',OUT/'initial_labels_snapshot.csv')
    write(OUT/'baseline_predictions.csv',baseline);dump(OUT/'cases.json',cases)
    video=OUT/'context_silent.mp4';cmd=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1280x800','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',str(video)]
    with (OUT/'render.log').open('wb') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        for t in range(730,1156):
            p=frame_dir/f'{t:06d}.jpg';image_hashes[str(p)]=sha(p);im=cv2.imread(str(p));assert im.shape==(1080,1920,3)
            frame=np.full((800,1280,3),18,np.uint8);frame[80:]=cv2.resize(im,(1280,720),interpolation=cv2.INTER_AREA)
            b=ball[t]
            if int(b['visible']):cv2.circle(frame,(round(float(b['x_px'])*2/3),80+round(float(b['y_px'])*2/3)),8,(255,220,60),2,cv2.LINE_AA)
            cv2.putText(frame,f'044 | local {t} | current reviewed-input ball output',(12,29),0,.71,(240,240,240),1,cv2.LINE_AA)
            cv2.putText(frame,'Cyan: input candidate, not independent ground truth | no interpolation',(12,61),0,.59,(210,220,230),1,cv2.LINE_AA)
            proc.stdin.write(frame.tobytes())
        proc.stdin.close();assert proc.wait()==0
    final=OUT/'context.mp4'
    subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-i',str(video),'-ss',str(730/fps),'-i',meta['clip'],'-map','0:v:0','-map','1:a:0?','-t',str(426/fps),'-c:v','copy','-c:a','aac','-movflags','+faststart',str(final)],check=True)
    dump(OUT/'source_image_hashes.json',image_hashes)
    shutil.copy2(ROOT/'rally_repair_pilot.html',OUT/'review.html')
    source=(ROOT/'serve_rally_repair_pilot.py').read_text(encoding='utf-8')
    (OUT/'serve_review.py').write_text('import sys\nsys.path.insert(0,'+repr(str(ROOT))+')\n'+source,encoding='utf-8')
    py=SOURCE/'third_party/racketvision-env/Scripts/python.exe'
    starter="$taskFolder = Split-Path -Parent $MyInvocation.MyCommand.Path\n$taskStatePath = Join-Path $taskFolder 'server_state.json'\n$taskReady = $false\nif (Test-Path -LiteralPath $taskStatePath) {\n  $taskState = Get-Content -LiteralPath $taskStatePath -Raw | ConvertFrom-Json\n  try { $taskReady = (Invoke-WebRequest -Uri ($taskState.url + 'api/cases') -TimeoutSec 3).StatusCode -eq 200 } catch {}\n}\nif (-not $taskReady) {\n  Start-Process -FilePath '"+str(py)+"' -ArgumentList @('-X','utf8', ('\"' + (Join-Path $taskFolder 'serve_review.py') + '\"')) -WorkingDirectory $taskFolder -WindowStyle Hidden\n  for ($taskAttempt=0; $taskAttempt -lt 60; $taskAttempt++) {\n    Start-Sleep -Milliseconds 500\n    if (Test-Path -LiteralPath $taskStatePath) {\n      $taskState = Get-Content -LiteralPath $taskStatePath -Raw | ConvertFrom-Json\n      try { $taskReady = (Invoke-WebRequest -Uri ($taskState.url + 'api/cases') -TimeoutSec 2).StatusCode -eq 200; if ($taskReady) { break } } catch {}\n    }\n  }\n}\nif (-not $taskReady) { throw 'Review server did not start' }\nStart-Process $taskState.url\nWrite-Output $taskState.url\n"
    (OUT/'start_review.ps1').write_text(starter,encoding='utf-8-sig')
    # Readiness inventory preserves unresolved scope instead of treating camera departure as point end.
    readiness=[]
    for r in read(ROOT.parent/'frontcourt_handoff_20260927/rally_label_review.csv'):
        readiness.append(dict(rally_candidate_id=r['candidate_id'],serve_review_status=r['serve_review_status'],boundary_review_status=r['boundary_review_status'],
            winner_camera_side=r['winner_camera_side'],winner_side_review_status=r['winner_camera_side_review_status'],end_event=r['point_end_event_type'],end_event_review_status=r['point_end_event_review_status'],
            registered_player_ball_bundle=int(r['candidate_id']=='match1_rally_window_044'),exact_boundaries_confirmed=0,selected_for_current_usability_calibration=int(r['candidate_id']=='match1_rally_window_044')))
    write(OUT/'rally_readiness_inventory.csv',readiness)
    frames=read(ROOT/'integrated_044_v4/frame_observation_status.csv')
    readiness_summary=dict(candidate_interior_frames=402,frames_with_4_candidate_players=sum(int(r['observed_player_count'])==4 for r in frames),
        ball_output_frames=sum(int(r['ball_output_present']) for r in frames),review_sample_times=201,review_sample_output_times=sum(c['prediction_visible'] for c in cases),
        reused_precise_centers=6,new_review_targets=195,output_counts_are_not_accuracy=True,prior_repair_time_available=False)
    dump(OUT/'initial_readiness.json',readiness_summary)
    assert all(sha(Path(p))==h for p,h in protected.items())
    frozen={n:sha(OUT/n) for n in ['protocol.json','executed_prepare_source.py','initial_labels_snapshot.csv','baseline_predictions.csv','cases.json',
        'source_image_hashes.json','context.mp4','review.html','serve_review.py','start_review.ps1','rally_readiness_inventory.csv','initial_readiness.json']}
    dump(OUT/'manifest.json',dict(status='review_ready_human_not_started',protected_input_hashes=protected,frozen_artifact_hashes=frozen))
    print(json.dumps(readiness_summary,indent=2),flush=True)


if __name__=='__main__':main()
