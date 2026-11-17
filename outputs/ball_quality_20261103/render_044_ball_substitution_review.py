"""Short silent review of three proposed component substitutions."""
import json,subprocess
import cv2
import numpy as np
from extract_044_inplay_ball_candidates import OUT,SOURCE,read,sha

def main():
    rows={int(r['local_frame']):r for r in read(OUT/'substitution_proposals.csv')}
    fps=json.loads((OUT/'protocol.json').read_text())['fps'];changes=[t for t,r in rows.items() if int(r['substitution_proposed'])]
    assert changes==[773,784,1103]
    segments=[(766,792),(1094,1112)];video=OUT/'substitution_review_h264.mp4'
    cmd=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1600x720','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    encoded=0
    with (OUT/'substitution_render.log').open('wb') as log:
        process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        for first,last in segments:
            centers=[]
            for t in changes:
                if first<=t<=last:
                    r=rows[t];centers.extend([[float(r['x_px']),float(r['y_px'])],[float(r['proposed_x_px']),float(r['proposed_y_px'])]])
            lo=np.min(centers,axis=0);hi=np.max(centers,axis=0);cx,cy=(lo+hi)/2
            left=int(np.clip(round(cx-160),0,1600));top=int(np.clip(round(cy-120),0,840))
            for t in range(first,last+1):
                im=cv2.imread(str(SOURCE/'data/balltrack/racketvision/match1/match1_motion_pilot_016/frames'/f'{t:06d}.jpg'));assert im is not None
                canvas=np.full((720,1600,3),22,np.uint8)
                canvas[96:636,:960]=cv2.resize(im,(960,540),interpolation=cv2.INTER_AREA)
                canvas[96:576,960:]=cv2.resize(im[top:top+240,left:left+320],(640,480),interpolation=cv2.INTER_NEAREST)
                r=rows[t];changed=bool(int(r['substitution_proposed']))
                def circle(x,y,color,radius):
                    cv2.circle(canvas,(round(x/2),96+round(y/2)),radius,color,1,cv2.LINE_AA)
                    if left<=x<left+320 and top<=y<top+240:
                        cv2.circle(canvas,(960+round((x-left)*2),96+round((y-top)*2)),radius*2,color,2,cv2.LINE_AA)
                if int(r['visible']):circle(float(r['x_px']),float(r['y_px']),(0,170,255),5)
                if int(r['proposed_visible']):circle(float(r['proposed_x_px']),float(r['proposed_y_px']),(80,255,100),8)
                text=lambda s,x,y,size=.63,c=(235,235,235):cv2.putText(canvas,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,c,1,cv2.LINE_AA)
                text(f'044 / ball pilot016 | local frame {t} | 0.25x + 1s pauses | SILENT',12,27)
                text('Orange = original output | Green = proposed path | proposals only at773,784,1103',12,57)
                text('Full view (same aspect ratio)',12,85,.5);text('Fixed320x240 crop at2x (same aspect ratio)',972,85,.49)
                text('Alternative component proposed; original retained' if changed else 'Unchanged raw output here',12,672,.67,(80,255,100) if changed else (230,230,230))
                text('No interpolation; no new detections added. Three changed centers need visual review.',12,704,.57)
                text('Rank2 component' if changed else 'Original and proposed coincide',972,610,.57)
                text('Ball identity not independently confirmed',972,642,.5)
                if changed:
                    cv2.imwrite(str(OUT/f'substitution_frame{t}.png'),canvas)
                repeat=4+(60 if changed else 0)
                for _ in range(repeat):process.stdin.write(canvas.tobytes());encoded+=1
        process.stdin.close();assert process.wait()==0
    cap=cv2.VideoCapture(str(video));decoded=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape[:2]==(720,1600);decoded+=1
    cap.release();assert decoded==encoded
    (OUT/'render_report.json').write_text(json.dumps(dict(video=str(video.resolve()),decoded_frames=decoded,
        source_segments=segments,source_clock='legacy edited match1 local frames',speed=.25,
        silent=True,extra_pause_frames={str(t):60 for t in changes},crop_enlargement=2,
        source_aspect_preserved=True,proposal_csv_sha256=sha(OUT/'substitution_proposals.csv'),accuracy_established=False),indent=2),encoding='utf-8')
    print(video,'decoded',decoded)

if __name__=='__main__':main()
