"""Track one seeded branch without deleting competing ball candidates."""
import csv
import json
import subprocess
from collections import defaultdict

import cv2
import numpy as np

from extract_multiball_candidates import DEST,ROOT,SOURCE,read,sha


def associate(candidates,seed_frame,seed_xy,start,end):
    by=defaultdict(list)
    for c in candidates:by[int(c['local_frame'])].append(c)
    distance=lambda c,p:float(np.hypot(float(c['x_px'])-p[0],float(c['y_px'])-p[1]))
    first=min(by[seed_frame],key=lambda c:distance(c,seed_xy))
    assert distance(first,seed_xy)<=16
    selected={seed_frame:first};events=[]
    for direction in (-1,1):
        last=seed_frame;xy=(float(first['x_px']),float(first['y_px']))
        for t in range(seed_frame+direction,start-1 if direction<0 else end+1,direction):
            gap=abs(t-last)
            if gap>8:
                events.append(dict(direction=direction,local_frame=t,event='stopped_after_eight_frame_gap'))
                break
            # Local displacement bound, not a global one-ball majority vote.
            options=sorted(((distance(c,xy),c) for c in by[t]),key=lambda item:item[0]) if by[t] else []
            gate=32+8*gap
            possible=[(d,c) for d,c in options if d<=gate]
            if not possible:continue
            if len(possible)>1 and possible[1][0]-possible[0][0]<8:
                events.append(dict(direction=direction,local_frame=t,event='ambiguous_candidates_not_assigned'))
                continue
            d,chosen=possible[0]
            selected[t]=chosen;last=t;xy=(float(chosen['x_px']),float(chosen['y_px']))
    return selected,events


def render(selected,candidates,baseline,seed,start,end,second_selected=None):
    by=defaultdict(list)
    for c in candidates:by[int(c['local_frame'])].append(c)
    dual=second_selected is not None
    fps=59.94005994005994;video=DEST/('dual_branch_slow_h264.mp4' if dual else 'seeded_branch_slow_h264.mp4')
    command=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1600x640',
             '-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    encoded=0
    with (DEST/'ffmpeg.log').open('wb') as err:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=err)
        for t in range(start,end+1):
            im=cv2.imread(str(SOURCE/'data/balltrack/racketvision/match1/match1_motion_pilot_002/frames'/f'{t:06d}.jpg'));assert im is not None
            canvas=np.full((640,1600,3),22,np.uint8);canvas[74:614,:960]=cv2.resize(im,(960,540),interpolation=cv2.INTER_AREA)
            rois=[(405,560),(1130,620)]
            for i,(left,top) in enumerate(rois):canvas[98:338,960+320*i:1280+320*i]=im[top:top+240,left:left+320]
            text=lambda s,x,y,size=.62:cv2.putText(canvas,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,(235,235,235),1,cv2.LINE_AA)
            text(f'002 frame {t} | '+('two candidate branches; A seed318(user), B seed310(model)' if dual else 'selected branch initialized at user-reviewed frame318')+' | 0.25x',12,27)
            text('Green=branch A | cyan=branch B | gray=other heatmap components' if dual else 'Green=seeded candidate branch | orange=old single output | gray=other heatmap components',12,55,.59)
            text('Left-ball region (native)',972,90,.53);text('Right-ball region (native)',1292,90,.53)
            def circles(x,y,color,radius):
                cv2.circle(canvas,(round(x/2),74+round(y/2)),radius,color,1)
                for i,(left,top) in enumerate(rois):
                    if left<=x<left+320 and top<=y<top+240:cv2.circle(canvas,(960+i*320+round(x-left),98+round(y-top)),radius+2,color,1)
            for c in by[t]:circles(float(c['x_px']),float(c['y_px']),(145,145,145),4)
            old=baseline[t]
            if not dual and int(old['visible']):circles(float(old['x_px']),float(old['y_px']),(0,160,255),5)
            if dual and t in second_selected:
                b=second_selected[t];circles(float(b['x_px']),float(b['y_px']),(255,220,0),7)
            if t in selected:
                c=selected[t];circles(float(c['x_px']),float(c['y_px']),(80,255,100),9)
                state=f"Selected candidate observed; heatmap peak {float(c['peak']):.2f}"
            else:state='Selected ball has no assigned candidate here.'
            if dual:state=f"Observed candidates: A={'yes' if t in selected else 'missing'}, B={'yes' if t in second_selected else 'missing'}"
            for j,s in enumerate((state,'Seeded association; full identity accuracy unverified.','Missing frames are left missing.','Candidate centers come from heatmap components.','Other candidates are retained, not suppressed.')):
                text(s,972,380+j*36,.49)
            repeat=4+(30 if t in (310,318) else 0)
            for _ in range(repeat):process.stdin.write(canvas.tobytes());encoded+=1
        process.stdin.close();assert process.wait()==0
    cap=cv2.VideoCapture(str(video));decoded=0
    target_frame=4*(310-start)
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(640,1600)
        if decoded==target_frame:cv2.imwrite(str(DEST/('dual_ball_candidates_frame310.png' if dual else 'both_ball_candidates_frame310.png')),im)
        decoded+=1
    cap.release();assert decoded==encoded
    return dict(video=str(video.resolve()),decoded_frames=decoded,full_decode=True,nominal_speed=.25,
                additional_target_freezes_frames=[310,318])


def main():
    assert not (DEST/'seeded_branch.csv').exists()
    protocol=json.loads((DEST/'protocol.json').read_text());start=protocol['start_frame'];end=protocol['end_frame_inclusive']
    review=ROOT/'temporal_guard/suppression_audit_20261103/user_reviewed/review.json'
    reviewed=json.loads(review.read_text(encoding='utf-8'));seed=reviewed['local_frame']
    assert reviewed['candidate_is_ball'] and seed==318
    seed_xy=(float(reviewed['raw_model_x_px']),float(reviewed['raw_model_y_px']))
    candidates=read(DEST/'candidates.csv');selected,events=associate(candidates,seed,seed_xy,start,end)
    baseline=read(ROOT.parent/'tracking_quality_20261102/ball_aligned/match1_motion_pilot_002/balltrack_predictions.csv')
    output=[]
    for t in range(start,end+1):
        c=selected.get(t)
        output.append(dict(candidate_id=protocol['candidate_id'],local_frame=t,branch='user_seeded_A',
            visible=int(c is not None),x_px=c['x_px'] if c else 0,y_px=c['y_px'] if c else 0,
            peak=c['peak'] if c else 0,component_rank=c['rank'] if c else '',
            position_source='observed_heatmap_component' if c else 'missing_no_supported_candidate',
            identity_status='seeded_association_candidate_not_full_track_reviewed',
            seeded_by_user_review_frame=seed,interpolated=0))
    with (DEST/'seeded_branch.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
    # Compare observation availability near the proposed branch, never accuracy.
    compatible=sum(int(baseline[t]['visible']) and np.hypot(float(baseline[t]['x_px'])-float(c['x_px']),
                   float(baseline[t]['y_px'])-float(c['y_px']))<=16 for t,c in selected.items())
    video=render(selected,candidates,baseline,seed,start,end)
    report=dict(protocol='seeded local association: gate32+8*gap px, max gap8 frames, competing candidates within8px distance margin stay unresolved',
        seed_review_sha256=sha(review),seed_frame=seed,seed_xy=seed_xy,selected_component_frames=len(selected),
        selected_frames=sorted(selected),old_output_agrees_with_selected_component_frames=int(compatible),
        frames_without_assignment=len(output)-len(selected),association_events=events,
        video=video,interpolated_positions=0,
        warning='16 assigned observed components are not 16 independently confirmed correct ball positions. No held-out identity or localization accuracy evaluation.')
    (DEST/'branch_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
