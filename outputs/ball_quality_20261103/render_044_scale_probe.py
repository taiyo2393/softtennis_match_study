"""Render unconfirmed tiled components beside the current reviewed observations."""
import json,subprocess
from pathlib import Path
from collections import defaultdict
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2,np

OUT=ROOT/'ball_044_scale_probe'

def main():
    assert not (OUT/'render_report.json').exists()
    by=defaultdict(list)
    for c in read(OUT/'tiled_components.csv'):by[int(c['local_frame'])].append(c)
    manifest=load(ROOT/'reviewed_ball_overrides.json')['clips']['match1_motion_pilot_016']
    bp=Path(manifest['reviewed_output']);ball=read(bp);assert sha(bp)==manifest['reviewed_output_sha256']
    displayed=[];chosen={}
    # Same12px duplicate-merging rule fixed in inference protocol, no trajectory interpolation.
    for t in range(1078,1107):
        unique=[]
        for c in sorted(by[t],key=lambda c:float(c['peak']),reverse=True):
            if not any(np.hypot(float(c['x_px'])-float(q['x_px']),float(c['y_px'])-float(q['y_px']))<=12 for q in unique):unique.append(c)
        visible=sorted([c for c in unique if 620<=float(c['x_px'])<1100 and 460<=float(c['y_px'])<730],key=lambda c:float(c['x_px']))
        chosen[t]=visible
        for k,c in enumerate(visible):displayed.append(dict(c,display_label=chr(65+k),review='pending_new_candidate_confirmation'))
    write(OUT/'displayed_candidates.csv',displayed)
    final=OUT/'scale_probe_review_h264.mp4';fps=59.94005994005994;expected=0
    cmd=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y',
        '-f','rawvideo','-pix_fmt','bgr24','-s','1920x660','-r',str(fps),'-i','pipe:0',
        '-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(final)]
    with (OUT/'render.log').open('wb') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        for t in range(1078,1107):
            im=cv2.imread(str(SOURCE/'data/balltrack/racketvision/match1/match1_motion_pilot_016/frames'/f'{t:06d}.jpg'));assert im is not None
            crop=cv2.resize(im[460:730,620:1100],(960,540),interpolation=cv2.INTER_LINEAR)
            frame=np.full((660,1920,3),22,np.uint8);frame[120:,:960]=crop;frame[120:,960:]=crop
            b=ball[t]
            if int(b['visible']):
                color=(80,255,100) if b['human_review_id'] else (255,255,0)
                cv2.circle(frame,(round((float(b['x_px'])-620)*2),120+round((float(b['y_px'])-460)*2)),14,color,2,cv2.LINE_AA)
            for k,c in enumerate(chosen[t]):
                x=960+round((float(c['x_px'])-620)*2);y=120+round((float(c['y_px'])-460)*2)
                cv2.circle(frame,(x,y),13,(255,140,60),2,cv2.LINE_AA)
                cv2.putText(frame,chr(65+k),(x+17,y-10),0,.65,(255,180,90),1,cv2.LINE_AA)
            cv2.putText(frame,f'044 local {t} | 0.125x + brief holds | same fixed2x display crop',(16,30),0,.78,(245,245,245),1,cv2.LINE_AA)
            cv2.putText(frame,'Current reviewed outputs (missing stays blank)',(16,70),0,.72,(235,235,235),1,cv2.LINE_AA)
            cv2.putText(frame,'Tiled inference: BLUE candidates, NOT confirmed',(980,70),0,.72,(235,235,235),1,cv2.LINE_AA)
            cv2.putText(frame,'Letters identify candidates only within this frame',(980,103),0,.61,(220,220,220),1,cv2.LINE_AA)
            if t in (1084,1088,1090,1091,1092,1093):cv2.imwrite(str(OUT/f'comparison_local{t}.jpg'),frame)
            repeats=8+(24 if t in (1084,1088,1090,1091,1092,1093) else 0)
            for _ in range(repeats):proc.stdin.write(frame.tobytes())
            expected+=repeats
        proc.stdin.close();assert proc.wait()==0
    cap=cv2.VideoCapture(str(final));n=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(660,1920);n+=1
    cap.release();assert n==expected and sha(bp)==manifest['reviewed_output_sha256']
    p=load(OUT/'protocol.json')
    assert all(sha(Path(path))==h for path,h in p['input_hashes'].items())
    report=dict(video=str(final.resolve()),video_sha256=sha(final),decoded_frames=n,
        local_frames=[1078,1106],display_crop_xywh=[620,460,480,270],uniform_display_scale=2,
        displayed_candidates_sha256=sha(OUT/'displayed_candidates.csv'),displayed_candidates=len(displayed),
        review='pending',reviewed_inputs_and_registries_unchanged=True,
        caveat='The two candidates at1091 are both displayed, no identity assignment. Highest-peak deduplication is a display convention, not validation.')
    (OUT/'render_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
