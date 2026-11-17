"""Two short windows showing substitutions, one recovery and one withholding."""
import json,subprocess
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from integrate_044_reviewed_tracks import ROOT,SOURCE,read,load,COLORS,NAMES
from propose_044_ball_branches import OUT
from extract_serve_start_features import sha

def main():
    rows={int(r['local_frame']):r for r in read(OUT/'branch_proposals.csv')}
    pm=load(ROOT/'reviewed_person_overrides.json')['rallies']['match1_rally_window_044'];persons=defaultdict(list)
    assert sha(Path(pm['reviewed_output']))==pm['reviewed_output_sha256']
    for r in read(Path(pm['reviewed_output'])):persons[int(r['local_frame'])].append(r)
    segments=[(883,902),(952,978)];holds=[891,892,960,968,970]
    fps=59.94005994005994;video=OUT/'branch_review_h264.mp4';encoded=0
    cmd=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1920x720','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    with (OUT/'render.log').open('wb') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        for i,(first,last) in enumerate(segments,1):
            pts=[]
            for t in range(first,last+1):
                r=rows[t]
                if r['branch_proposal_action']!='unchanged_reviewed_input':
                    if int(r['visible']):pts.append([float(r['x_px']),float(r['y_px'])])
                    if int(r['branch_proposed_visible']):pts.append([float(r['branch_proposed_x_px']),float(r['branch_proposed_y_px'])])
            points=np.asarray(pts);center=(points.min(axis=0)+points.max(axis=0))/2
            left=int(np.clip(round(center[0]-320),0,1280));top=int(np.clip(round(center[1]-180),0,720))
            for t in range(first,last+1):
                im=cv2.imread(str(SOURCE/'data/balltrack/racketvision/match1/match1_motion_pilot_016/frames'/f'{t:06d}.jpg'));assert im is not None
                frame=np.full((720,1920,3),22,np.uint8);frame[96:636,:960]=cv2.resize(im,(960,540),interpolation=cv2.INTER_AREA)
                frame[96:636,960:]=cv2.resize(im[top:top+360,left:left+640],(960,540),interpolation=cv2.INTER_LINEAR)
                text=lambda s,x,y,size=.63,c=(235,235,235):cv2.putText(frame,s,(x,y),0,size,c,1,cv2.LINE_AA)
                text(f'044 | window {i}/2 | local frame {t} | 0.25x + pauses | SILENT',12,27)
                text('Orange = current output | Green = new observed component | Player paths are unchanged',12,57)
                text('Full view',12,85,.5);text('Fixed640x360 crop at1.5x; same aspect ratio',972,85,.5)
                # Person positions provide context, rather than contributing new labels.
                for p in persons[t]:
                    x1,y1,x2,y2=[float(p[k])*.5 for k in ['x1','y1','x2','y2']]
                    cv2.rectangle(frame,(round(x1*.5),96+round(y1*.5)),(round(x2*.5),96+round(y2*.5)),(115,115,115),1)
                    text(NAMES[p['player_slot']],max(0,min(910,round(x1*.5))),max(110,96+round(y1*.5)-3),.5,(160,160,160))
                r=rows[t];action=r['branch_proposal_action']
                def circle(x,y,color,radius):
                    cv2.circle(frame,(round(x*.5),96+round(y*.5)),radius,color,2,cv2.LINE_AA)
                    if left<=x<left+640 and top<=y<top+360:
                        cv2.circle(frame,(960+round((x-left)*1.5),96+round((y-top)*1.5)),round(radius*1.5),color,2,cv2.LINE_AA)
                if int(r['visible']):circle(float(r['x_px']),float(r['y_px']),(0,170,255),6)
                if action!='unchanged_reviewed_input' and int(r['branch_proposed_visible']):circle(float(r['branch_proposed_x_px']),float(r['branch_proposed_y_px']),(80,255,100),9)
                if action.startswith('withhold'):
                    detail='968: ORANGE proposed as false output; no replacement observation here.';color=(100,150,255)
                elif action=='add_observed_component':detail='New observed candidate proposed in a missing frame; NOT interpolation.';color=(80,255,100)
                elif action=='replace_with_observed_component':detail='Alternative observed component proposed; original output retained for comparison.';color=(80,255,100)
                else:detail='Unchanged current output at this frame.';color=(235,235,235)
                text(detail,12,674,.67,color)
                text('Candidate correspondence and proposed withholding require review; no independent accuracy measured.',12,708,.58)
                if t in holds:cv2.imwrite(str(OUT/f'branch_local{t}.jpg'),frame)
                for _ in range(4+(45 if t in holds else 0)):proc.stdin.write(frame.tobytes());encoded+=1
        proc.stdin.close();assert proc.wait()==0
    cap=cv2.VideoCapture(str(video));n=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape[:2]==(720,1920);n+=1
    cap.release();assert n==encoded
    (OUT/'render_report.json').write_text(json.dumps(dict(video=str(video.resolve()),decoded_frames=n,
        silent=True,speed=.25,pause_local_frames=holds,source_segments=segments,crop_enlargement=1.5,
        source_aspect_preserved=True,proposals_sha256=sha(OUT/'branch_proposals.csv'),
        person_input_sha256=pm['reviewed_output_sha256'],independent_accuracy_established=False),indent=2),encoding='utf-8')
    print(video,'decoded',n)

if __name__=='__main__':main()
