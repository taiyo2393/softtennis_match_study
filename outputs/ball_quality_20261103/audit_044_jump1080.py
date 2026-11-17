"""Inspect the user-reported1080 instability without changing reviewed tracks."""
from pathlib import Path
import math
import subprocess
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2,np

OUT=ROOT/'ball_044_jump1080_audit'
SUSPECT={1082,1084,1086,1087,1088}

def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'diagnosis.json').exists(), 'Keep completed audit snapshots.'
    registry=ROOT/'reviewed_ball_overrides.json'
    manifest=load(registry)['clips']['match1_motion_pilot_016']
    source=Path(manifest['reviewed_output'])
    protected=[source,registry,ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json']
    before={str(p):sha(p) for p in protected}
    assert before[str(source)]==manifest['reviewed_output_sha256']
    ball=read(source);components=read(ROOT/'inplay_ball_044/candidates.csv')
    rows=[];last=None
    for t in range(1056,1107):
        b=ball[t];visible=int(b['visible']);point=(float(b['x_px']),float(b['y_px']))
        distance=math.dist(point,last[1]) if visible and last else None
        rows.append(dict(local_frame=t,source_frame=int(b['source_frame']),output_present=visible,
            x_px=b['x_px'],y_px=b['y_px'],previous_observation_frame=last[0] if last else '',
            previous_observation_distance_px=distance if distance is not None else '',
            frame_gap=t-last[0] if last else '',
            review_status='suspected_not_confirmed' if t in SUSPECT else 'not_newly_reviewed'))
        if visible:last=(t,point)
    write(OUT/'observations.csv',rows)
    write(OUT/'candidate_components.csv',[c for c in components if 1068<=int(c['local_frame'])<=1106])
    final=OUT/'jump1080_review_h264.mp4'
    # Fixed source crop, uniform2x enlargement. Left stays unmarked.
    x0,y0,x1,y1=620,480,1100,720
    fps=59.94005994005994;expected=0
    cmd=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y',
         '-f','rawvideo','-pix_fmt','bgr24','-s','1920x600','-r',str(fps),'-i','pipe:0',
         '-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(final)]
    with (OUT/'render.log').open('wb') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        for t in range(1068,1107):
            im=cv2.imread(str(SOURCE/'data/balltrack/racketvision/match1/match1_motion_pilot_016/frames'/f'{t:06d}.jpg'))
            assert im is not None
            crop=cv2.resize(im[y0:y1,x0:x1],(960,480),interpolation=cv2.INTER_LINEAR)
            canvas=np.full((600,1920,3),22,np.uint8);canvas[120:,:960]=crop;canvas[120:,960:]=crop
            b=ball[t]
            if int(b['visible']):
                cx=round((float(b['x_px'])-x0)*2)+960;cy=round((float(b['y_px'])-y0)*2)+120
                color=(0,165,255) if t in SUSPECT else (255,255,0)
                cv2.circle(canvas,(cx,cy),15,color,2,cv2.LINE_AA)
            # Alternative weak components are evidence only, never exported as track coordinates.
            if t in (1081,1082):
                for c in components:
                    if int(c['local_frame'])==t and int(c['rank'])==2:
                        cx=round((float(c['x_px'])-x0)*2)+960;cy=round((float(c['y_px'])-y0)*2)+120
                        cv2.circle(canvas,(cx,cy),12,(255,140,60),2,cv2.LINE_AA)
            cv2.putText(canvas,f'044 local {t} | source {114635+t} | 0.125x, brief holds',(20,30),0,.8,(245,245,245),1,cv2.LINE_AA)
            cv2.putText(canvas,'Original, fixed crop, uniform2x enlargement',(20,72),0,.7,(235,235,235),1,cv2.LINE_AA)
            cv2.putText(canvas,'Orange=suspect output; cyan=other output',(980,68),0,.7,(235,235,235),1,cv2.LINE_AA)
            cv2.putText(canvas,'Blue=weak alternative, unconfirmed',(980,100),0,.65,(235,235,235),1,cv2.LINE_AA)
            if t in (1080,1082,1084,1086,1087,1088):cv2.imwrite(str(OUT/f'review_local{t}.jpg'),canvas)
            repeats=8+(21 if t in (1080,1082,1084,1086,1087,1088) else 0)
            for _ in range(repeats):proc.stdin.write(canvas.tobytes())
            expected+=repeats
        proc.stdin.close();assert proc.wait()==0
    cap=cv2.VideoCapture(str(final));decoded=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(600,1920);decoded+=1
    cap.release();assert decoded==expected
    assert all(sha(Path(p))==s for p,s in before.items())
    diagnosis=dict(user_report='1080 frame vicinity is unstable',status='localized_review_pending',
        input_path=str(source),input_sha256=sha(source),source_frame_offset=114635,
        displayed_local_frames=[1068,1106],suspect_output_frames=sorted(SUSPECT),
        output1080_to1082_distance_px=math.dist((787,568),(714,605)),
        interval_before1080_without_output=[1057,1079],
        weak_alternative_frames=[1081,1082],weak_alternatives_confirmed_ball=False,
        note='The23-frame output gap prevents a secure incoming track anchor. Pixel jumps and proximity to player/racket alone do not prove false detection. No blanket smoothing, extrapolation or suppression applied.',
        video=str(final.resolve()),video_sha256=sha(final),decoded_frames=decoded,
        authoritative_inputs_unchanged=True,independent_accuracy_established=False)
    (OUT/'diagnosis.json').write_text(__import__('json').dumps(diagnosis,indent=2),encoding='utf-8')
    print(__import__('json').dumps(diagnosis,indent=2))

if __name__=='__main__':main()
