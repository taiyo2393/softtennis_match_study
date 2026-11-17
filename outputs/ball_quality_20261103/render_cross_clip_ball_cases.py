"""Four candidate-only reviews; no old labels or classifier outputs displayed."""
import json,subprocess
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2,np

OUT=ROOT/'ball_candidate_cross_rally'

def main():
    assert not (OUT/'review_video_report.json').exists()
    manifest=load(OUT/'case_manifest.json');assert sha(OUT/'cases.csv')==manifest['candidate_table_sha256']
    cases=[r for r in read(OUT/'cases.csv') if r['label']=='pending_user_confirmation']
    assert [(r['display_label'],int(r['local_frame'])) for r in cases]==[('A',391),('B',503),('C',615),('D',1623)]
    predictions=OUT/'appearance_probe_v2/frozen_predictions.csv';before_prediction_hash=sha(predictions)
    assert before_prediction_hash==load(OUT/'appearance_probe_v2/report.json')['frozen_predictions_sha256']
    fps=59.94005994005994;final=OUT/'four_candidate_review_h264.mp4';expected=0;details=[]
    cmd=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y',
        '-f','rawvideo','-pix_fmt','bgr24','-s','1920x660','-r',str(fps),'-i','pipe:0',
        '-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(final)]
    with (OUT/'review_render.log').open('wb') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        for c in cases:
            target=int(c['local_frame']);x=float(c['x_px']);y=float(c['y_px']);name=c['display_label']
            left=max(0,min(1440,round(x)-240));top=max(0,min(810,round(y)-135))
            details.append(dict(case_id=c['case_id'],display_label=name,local_frame=target,source_frame=int(c['source_frame']),
                candidate_xy=[x,y],crop_xywh=[left,top,480,270],crop_uniform_scale=2,
                displayed_source_local_frames=[target-12,target+12],marker_shown_only_at_target=True,
                label='pending_user_confirmation',video_start_frame=expected))
            for t in range(target-12,target+13):
                image=SOURCE/'data/balltrack/racketvision/match1'/c['ball_pilot_id']/'frames'/f'{t:06d}.jpg'
                im=cv2.imread(str(image));assert im is not None
                frame=np.full((660,1920,3),22,np.uint8)
                frame[120:,:960]=cv2.resize(im,(960,540),interpolation=cv2.INTER_AREA)
                frame[120:,960:]=cv2.resize(im[top:top+270,left:left+480],(960,540),interpolation=cv2.INTER_LINEAR)
                if t==target:
                    color=(255,170,50)
                    cv2.circle(frame,(round(x*.5),120+round(y*.5)),8,color,2,cv2.LINE_AA)
                    cv2.circle(frame,(960+round((x-left)*2),120+round((y-top)*2)),16,color,2,cv2.LINE_AA)
                    cv2.putText(frame,name,(960+round((x-left)*2)+22,120+round((y-top)*2)-12),0,.8,color,2,cv2.LINE_AA)
                cv2.putText(frame,f'Candidate {name} | ball pilot008 | local {t} | target {target} | 0.25x + target hold',(16,31),0,.8,(245,245,245),1,cv2.LINE_AA)
                cv2.putText(frame,'Full frame,0.5x display scale',(16,75),0,.73,(230,230,230),1,cv2.LINE_AA)
                cv2.putText(frame,'Same frame, fixed crop, uniform2x enlargement',(980,75),0,.73,(230,230,230),1,cv2.LINE_AA)
                cv2.putText(frame,'Does the BLUE ring enclose a real ball? No predicted label is shown.',(980,108),0,.56,(230,230,230),1,cv2.LINE_AA)
                repeats=4+(42 if t==target else 0)
                if t==target:assert cv2.imwrite(str(OUT/f'candidate_{name}_local{target}.jpg'),frame)
                for _ in range(repeats):proc.stdin.write(frame.tobytes())
                expected+=repeats
        proc.stdin.close();assert proc.wait()==0
    cap=cv2.VideoCapture(str(final));n=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(660,1920);n+=1
    cap.release();assert n==expected==568
    assert sha(predictions)==before_prediction_hash
    assert all(sha(Path(p))==h for p,h in manifest['input_hashes'].items())
    report=dict(video=str(final.resolve()),video_sha256=sha(final),decoded_frames=n,duration_sec=n/fps,
        details=details,candidate_table_sha256=sha(OUT/'cases.csv'),
        frozen_predictions_sha256=before_prediction_hash,model_predictions_not_displayed=True,
        adopted_inputs_and_source_annotations_unchanged=True,review_status='pending')
    (OUT/'review_video_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='details'},indent=2))

if __name__=='__main__':main()
