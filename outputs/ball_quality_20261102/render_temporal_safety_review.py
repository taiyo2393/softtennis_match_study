"""Short continuous videos for the counterexample and partial appearance fix."""
import json
import subprocess

import cv2
import numpy as np

from protect_visible_ball_candidates import DEST
from temporal_consensus_guard import OUT,SOURCE,BASE,read


def render(cid,start,end,targets,prediction_root=None,output_dir=None,reviewed=False):
    folder=output_dir or DEST/'review_videos';folder.mkdir(parents=True,exist_ok=True)
    raw=read(BASE/cid/'balltrack_predictions.csv')
    proposed=read((prediction_root or DEST/'predictions')/cid/'balltrack_predictions.csv')
    center=raw[targets[0]];left=round(float(center['x_px']))-240;top=round(float(center['y_px']))-135
    left=max(0,min(1920-480,left));top=max(0,min(1080-270,top))
    fps=59.94005994005994;destination=folder/f'{cid[-3:]}_safety_slow.mp4'
    command=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1440x640',
             '-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(destination)]
    count=0
    with destination.with_suffix('.log').open('wb') as err:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=err)
        for t in range(start,end+1):
            im=cv2.imread(str(SOURCE/'data/balltrack/racketvision/match1'/cid/'frames'/f'{t:06d}.jpg'));assert im is not None
            canvas=np.full((640,1440,3),22,np.uint8)
            canvas[74:614,:960]=cv2.resize(im,(960,540),interpolation=cv2.INTER_AREA)
            canvas[74:344,960:1440]=im[top:top+270,left:left+480]
            text=lambda s,x,y,scale=.62:cv2.putText(canvas,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,(240,240,240),1,cv2.LINE_AA)
            text(f'{cid[-3:]} | local frame {t} | raw continuous frames, 0.25x',12,27)
            text('Marks only at reviewed target frame(s). Crop is native scale. No ground-truth positions added.',12,55)
            if reviewed:
                status='User confirms: this candidate is a ball.'
                details=['Green ring: original detection retained.','Center coordinate is from the model.','No pixel-center accuracy label added.']
            elif cid.endswith('002'):
                status='Unresolved: trajectory filter withheld this candidate.'
                details=['Orange ring at frame318.','Is it the actual ball?','Original detector coordinate retained.']
            else:
                status='Held-ball appearance protection example.'
                details=['Green ring: input detection retained.','At frames626 and630.','No interpolated coordinates.']
            text(status,973,380,.45)
            for j,s in enumerate(details):text(s,973,418+j*34,.5)
            if t in targets:
                p=raw[t];x=round(float(p['x_px']));y=round(float(p['y_px']))
                color=(80,255,100) if int(proposed[t]['visible']) else (0,175,255)
                cv2.circle(canvas,(x//2,74+y//2),10,color,1)
                cv2.circle(canvas,(960+x-left,74+y-top),12,color,1)
                cv2.rectangle(canvas,(0,72),(1439,616),color,2)
            for _ in range(4+(30 if t in targets else 0)):
                process.stdin.write(canvas.tobytes());count+=1
        process.stdin.close();assert process.wait()==0
    cap=cv2.VideoCapture(str(destination));decoded=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(640,1440)
        decoded+=1
    cap.release();assert decoded==count
    return dict(candidate_id=cid,start_frame=start,end_frame=end,targets=targets,decoded_frames=decoded,
                speed=.25,video=str(destination.resolve()),full_decode_verified=True)


def main():
    result=[render('match1_motion_pilot_002',298,338,[318]),render('match1_motion_pilot_014',610,646,[626,630])]
    (DEST/'safety_video_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
