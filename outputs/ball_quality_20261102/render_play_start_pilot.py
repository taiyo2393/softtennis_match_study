"""Show the existing serve-start bracket on the actual ball-pilot clip."""
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np

from audit_play_starts import SOURCE,OUT,BASE,read


def render_pilot(cid,start,end,prefix):
    inventory=read(OUT/'ball_pilot_start_inventory.csv')
    r=next(r for r in inventory if r['ball_pilot_id']==cid)
    cid=r['ball_pilot_id'];first=int(r['first_contact_candidate_local_frame']);last=int(r['last_contact_candidate_local_frame'])
    base=SOURCE/'data/balltrack/racketvision/match1'/cid
    meta=json.loads((base/'runs/threshold030_median120/run_metadata.json').read_text());fps=meta['fps']
    temporary=OUT/f'{prefix}_context_silent.mp4';final=OUT/f'{prefix}_context_h264.mp4'
    command=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1280x800',
             '-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(temporary)]
    baseline=read(BASE/cid/'balltrack_predictions.csv')
    with (OUT/f'{prefix}_render.log').open('wb') as err:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=err)
        for t in range(start,end+1):
            im=cv2.imread(str(base/'frames'/f'{t:06d}.jpg'));assert im is not None
            canvas=np.full((800,1280,3),22,np.uint8);canvas[80:800]=cv2.resize(im,(1280,720),interpolation=cv2.INTER_AREA)
            if t<first:state='Preparation: excluded from position-tracking analysis';color=(170,170,170)
            elif t<=last:state='Serve-contact candidate interval: awaiting review';color=(0,190,255)
            else:state='After candidate start: eligibility still provisional';color=(80,255,100)
            cv2.putText(canvas,f'Ball pilot{cid[-3:]} / rally-window{r["rally_window_candidate_id"][-3:]} | local {t} | legacy source F{baseline[t]["source_frame"]}',(12,28),cv2.FONT_HERSHEY_SIMPLEX,.62,(235,235,235),1,cv2.LINE_AA)
            cv2.putText(canvas,state,(12,60),cv2.FONT_HERSHEY_SIMPLEX,.7,color,1,cv2.LINE_AA)
            cv2.rectangle(canvas,(0,78),(1279,799),color,2)
            process.stdin.write(canvas.tobytes())
        process.stdin.close();assert process.wait()==0
    # Restore the matching source audio, without changing the video timeline.
    mux=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-i',str(temporary),
         '-ss',f'{start/fps:.9f}','-t',f'{(end-start+1)/fps:.9f}','-i',meta['clip'],
         '-map','0:v:0','-map','1:a:0?','-c:v','copy','-c:a','aac','-b:a','128k','-movflags','+faststart',str(final)]
    subprocess.run(mux,check=True)
    cap=cv2.VideoCapture(str(final));decoded=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(800,1280)
        if decoded==first-start:cv2.imwrite(str(OUT/f'{prefix}_bracket_preview.png'),im)
        decoded+=1
    cap.release();assert decoded==end-start+1
    info=json.loads(subprocess.check_output([r'C:\ffmpeg\bin\ffprobe.exe','-v','error','-show_streams','-of','json',str(final)],text=True))
    # Verify the contact image's clock belongs to this cached clip, not original4K's clock.
    cap=cv2.VideoCapture(meta['clip']);source_im=None
    for t in range(last+1):
        ok,source_im=cap.read();assert ok
    cap.release();cache=cv2.imread(str(base/'frames'/f'{last:06d}.jpg'))
    ok,jpg=cv2.imencode('.jpg',source_im,[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok
    assert np.array_equal(cv2.imdecode(jpg,cv2.IMREAD_COLOR),cache)
    report=dict(candidate_id=cid,rally_window_candidate_id=r['rally_window_candidate_id'],context_local_frames=[start,end],
        contact_candidate_local_frames=[first,last],contact_candidate_source_frames=[int(r['effective_contact_candidate_first_source_frame']),int(r['effective_contact_candidate_last_source_frame'])],
        fps=fps,decoded_frames=decoded,audio_present=any(s['codec_type']=='audio' for s in info['streams']),
        cache_match_verified_at_local_frame=last,
        status='existing_contact_bracket_provisional',serve_validity_status=r['serve_validity_status'],video=str(final.resolve()),
        warning='No serve-validity, exact contact-frame, point-end or original4K-clock labels are newly confirmed.')
    (OUT/f'{prefix}_video_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':render_pilot('match1_motion_pilot_002',650,900,'002_start')
