"""Review only missing-run recovery proposals, keeping confirmed paths intact."""
import json,subprocess
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from diagnose_044_person_gaps import GAPS
from track_044_far_pair import ROOT,FIRST,a,image_at
from extract_serve_start_features import sha

OUT=ROOT/'person_044_pose_recovery'
NAMES=dict(zip(a.SLOTS,['N1','N2','F1','F2']))
COLORS=dict(near_front=(140,140,140),near_rear=(140,140,140),far_rear=(255,80,240),far_front=(0,230,255))

def main():
    source=OUT/'proposed_paths.csv';rows=a.read(source);by=defaultdict(list)
    for r in rows:by[int(r['local_frame'])].append(r)
    original=json.loads((ROOT/'reviewed_person_overrides.json').read_text())['rallies']['match1_rally_window_044']
    old={(int(r['local_frame']),r['player_slot']):r for r in a.read(Path(original['reviewed_output']))}
    new=[r for r in rows if r['position_source']=='detected_box_visible_torso_recovery_candidate']
    new_times=sorted({int(r['local_frame']) for r in new});holds=[]
    for t in new_times:
        if t-1 not in new_times:holds.append(t)
    fps=59.94005994005994;video=OUT/'recovery_review_h264.mp4';encoded=0
    command=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1800x840','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    with (OUT/'render.log').open('wb') as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        for number,(slot,start,end) in enumerate(GAPS,1):
            before=old[(start-1,slot)];after=old[(end+1,slot)]
            center=(a.foot(before)+a.foot(after))*.25
            left=int(np.clip(round(center[0]-210),0,1500));top=int(np.clip(round(center[1]-230),0,720))
            for t in range(start-3,end+4):
                im=image_at(FIRST+t);canvas=np.full((840,1800,3),22,np.uint8)
                canvas[96:636,:960]=cv2.resize(im,(960,540),interpolation=cv2.INTER_AREA)
                canvas[96:816,960:]=cv2.resize(im[top:top+360,left:left+420],(840,720),interpolation=cv2.INTER_LINEAR)
                text=lambda s,x,y,size=.62,c=(235,235,235):cv2.putText(canvas,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,c,1,cv2.LINE_AA)
                text(f'044 | missing run {number}/4 | {NAMES[slot]} | local frame {t} | 0.25x + pauses | SILENT',12,27)
                text('CYAN = new observed recovery proposal | Purple/yellow = previously reviewed far paths',12,57)
                text('Full view',12,85,.5);text('Fixed crop per run:420x360 at2x; same aspect ratio',972,85,.5)
                added=False
                for r in by[t]:
                    new_row=r['position_source']=='detected_box_visible_torso_recovery_candidate';added|=new_row
                    color=(255,235,40) if new_row else COLORS[r['player_slot']]
                    label=NAMES[r['player_slot']]+('+' if new_row else '')
                    x1,y1,x2,y2=[float(r[k])*.5 for k in ['x1','y1','x2','y2']]
                    cv2.rectangle(canvas,(round(x1*.5),96+round(y1*.5)),(round(x2*.5),96+round(y2*.5)),color,2 if new_row else 1)
                    text(label,max(0,min(910,round(x1*.5))),max(110,96+round(y1*.5)-4),.59,color)
                    if r['player_slot'].startswith('far_'):
                        # Crop drawing stays inside the same fixed rectangle.
                        l=960+round((x1-left)*2);rr=960+round((x2-left)*2)
                        top_y=96+round((y1-top)*2);bottom=96+round((y2-top)*2)
                        if rr>=960 and l<1800 and bottom>=96 and top_y<816:
                            cv2.rectangle(canvas,(max(960,l),max(96,top_y)),(min(1799,rr),min(815,bottom)),color,2)
                            text(label,max(960,min(1740,l)),max(115,top_y-5),.8,color)
                text('NEW BOX: target-player correspondence needs review' if added else 'No new box at this frame; unresolved gaps remain empty',12,685,.62,(255,235,40) if added else (230,230,230))
                text('Confirmed paths and all coordinates preserved.',12,724,.58)
                text('Every added box comes from the original detector.',12,760,.58)
                text('No interpolation or reconstructed hidden body.',12,796,.58)
                if t in holds:cv2.imwrite(str(OUT/f'new_box_local{t}.jpg'),canvas)
                repeat=4+(30 if t in holds else 0)
                for _ in range(repeat):process.stdin.write(canvas.tobytes());encoded+=1
        process.stdin.close();assert process.wait()==0
    cap=cv2.VideoCapture(str(video));decoded=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape[:2]==(840,1800);decoded+=1
    cap.release();assert decoded==encoded
    (OUT/'render_report.json').write_text(json.dumps(dict(video=str(video.resolve()),decoded_frames=decoded,silent=True,
        speed=.25,extra_pause_source_local_frames=holds,new_box_local_frames=new_times,
        paths_sha256=sha(source),source_aspect_preserved=True,crop_enlargement=2,
        fixed_crop_each_run=True,identity_accuracy_established=False),indent=2),encoding='utf-8')
    print(video,'frames',decoded,'new boxes',new_times)

if __name__=='__main__':main()
