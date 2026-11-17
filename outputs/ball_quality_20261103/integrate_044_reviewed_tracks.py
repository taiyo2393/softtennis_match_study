"""Join reviewed player/ball inputs by exact legacy source-frame IDs."""
import csv,json,subprocess,re
from collections import defaultdict,Counter
from pathlib import Path
import cv2
import numpy as np
from extract_serve_start_features import ROOT,SOURCE,sha

OUT=ROOT/'integrated_044'
SLOTS=('near_front','near_rear','far_rear','far_front')
NAMES=dict(zip(SLOTS,['N1','N2','F1','F2']))
COLORS=dict(zip(SLOTS,[(80,255,80),(255,190,40),(255,80,240),(0,230,255)]))

def load(path):return json.loads(path.read_text(encoding='utf-8'))
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def main(output_name='integrated_044'):
    global OUT
    assert re.fullmatch(r'[A-Za-z0-9_-]+',output_name)
    OUT=ROOT/output_name
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists(),'Choose a new snapshot folder; preserve existing integration.'
    pm=load(ROOT/'reviewed_person_overrides.json')['rallies']['match1_rally_window_044']
    bm=load(ROOT/'reviewed_ball_overrides.json')['clips']['match1_motion_pilot_016']
    pp=Path(pm['reviewed_output']);bp=Path(bm['reviewed_output'])
    assert sha(pp)==pm['reviewed_output_sha256'] and sha(bp)==bm['reviewed_output_sha256']
    scope=load(ROOT/'play_start_audit/044_scope_report.json');start,end=scope['candidate_main_play_local_frames']
    ball=read(bp);people=read(pp);by=defaultdict(list)
    for r in people:by[int(r['source_frame'])].append(r)
    states=[];entities=[];timestamp_differences=[]
    for t in range(start,end+1):
        b=ball[t];f=int(b['source_frame']);assert f==114635+t
        players=by[f];assert len({r['player_slot'] for r in players})==len(players)
        present={r['player_slot'] for r in players}
        states.append(dict(local_frame=t,source_frame=f,ball_clip_timestamp_sec=b['source_timestamp_sec'],
            observed_player_count=len(players),missing_player_slots=';'.join(s for s in SLOTS if s not in present),
            ball_output_present=int(b['visible']),ball_review_id=b['human_review_id'],ball_candidate_review=b['ball_candidate_review'],
            player_identity_review_ids=';'.join(sorted({r['human_video_review_id'] for r in players if r['human_video_review_id']})),
            independent_position_groundtruth=0))
        common=dict(local_frame=t,source_frame=f,entity='',observation_present=1,
            x_px='',y_px='',x1_px='',y1_px='',x2_px='',y2_px='',point_meaning='',
            observation_source='',review_id='',identity_or_candidate_review='',position_precision_validated=0)
        for r in players:
            assert int(r['local_frame'])==t
            timestamp_differences.append(abs(float(r['timestamp_sec'])-float(b['source_timestamp_sec'])))
            q=dict(common,entity=NAMES[r['player_slot']],x_px=float(r['bottom_center_x'])*.5,
                y_px=float(r['bottom_center_y'])*.5,
                x1_px=float(r['x1'])*.5,y1_px=float(r['y1'])*.5,x2_px=float(r['x2'])*.5,y2_px=float(r['y2'])*.5,
                point_meaning='bbox_bottom_proxy_not_confirmed_foot_contact',observation_source=r['position_source'],
                review_id=r['human_video_review_id'],identity_or_candidate_review=r['human_video_identity_review'])
            entities.append(q)
        if int(b['visible']):
            entities.append(dict(common,entity='ball',x_px=b['x_px'],y_px=b['y_px'],
                point_meaning='model_ball_center_candidate',observation_source=b['position_source'],
                review_id=b['human_review_id'],identity_or_candidate_review=b['ball_candidate_review']))
    write(OUT/'frame_observation_status.csv',states);write(OUT/'observed_entities_1920x1080.csv',entities)
    protocol=dict(rally_candidate_id='match1_rally_window_044',ball_pilot_id='match1_motion_pilot_016',
        local_frames=[start,end],source_frame_offset=114635,clock='legacy edited match1',
        alignment_key='exact integer source_frame, never nearest timestamp',
        player_input=str(pp),player_input_sha256=sha(pp),ball_input=str(bp),ball_input_sha256=sha(bp),
        image_coordinate_system=[1920,1080],player_scale_from_original=.5,
        max_original_timestamp_difference_sec=max(timestamp_differences),
        timestamp_note='Player times derive from global frame/fps; ball times from clip start+local/fps. Source-frame IDs are the join key.',
        missing_positions_filled=False,score_labels_joined=False,output_presence_is_not_physical_visibility_groundtruth=True,script_sha256=sha(Path(__file__)))
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    # Integrated diagnostic video: unchanged reviewed inputs, not later proposals.
    fps=59.94005994005994;silent=OUT/'integrated_review_silent.mp4';final=OUT/'integrated_review_h264.mp4'
    cmd=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1280x800','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',str(silent)]
    with (OUT/'render.log').open('wb') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        for t in range(start,end+1):
            b=ball[t];f=int(b['source_frame'])
            im=cv2.imread(str(SOURCE/'data/balltrack/racketvision/match1/match1_motion_pilot_016/frames'/f'{t:06d}.jpg'));assert im is not None
            frame=np.full((800,1280,3),22,np.uint8);frame[80:]=cv2.resize(im,(1280,720),interpolation=cv2.INTER_AREA)
            for r in by[f]:
                x1,y1,x2,y2=[round(float(r[k])/3) for k in ['x1','y1','x2','y2']]
                color=COLORS[r['player_slot']];cv2.rectangle(frame,(x1,80+y1),(x2,80+y2),color,2)
                cv2.putText(frame,NAMES[r['player_slot']],(max(0,min(1230,x1)),max(98,80+y1-5)),0,.65,color,1,cv2.LINE_AA)
            if int(b['visible']):
                color=(80,255,100) if b['human_review_id'] else (255,255,0)
                cv2.circle(frame,(round(float(b['x_px'])*2/3),80+round(float(b['y_px'])*2/3)),7,color,2,cv2.LINE_AA)
            cv2.putText(frame,f'044 | local {t} | reviewed player/ball inputs | 0.5x',(12,26),0,.66,(235,235,235),1,cv2.LINE_AA)
            cv2.putText(frame,'Ball: cyan=model output; green=user-reviewed candidates | Missing observations stay blank',(12,56),0,.59,(235,235,235),1,cv2.LINE_AA)
            if t in [892,960,968]:cv2.imwrite(str(OUT/f'integrated_local{t}.jpg'),frame)
            for _ in range(2):proc.stdin.write(frame.tobytes())
        proc.stdin.close();assert proc.wait()==0
    meta=load(SOURCE/'data/balltrack/racketvision/match1/match1_motion_pilot_016/runs/threshold030_median120/run_metadata.json')
    graph=f'[1:a]atrim=start={start/fps:.9f}:end={(end+1)/fps:.9f},asetpts=PTS-STARTPTS,atempo=0.5[a]'
    subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-i',str(silent),'-i',meta['clip'],'-filter_complex',graph,'-map','0:v:0','-map','[a]','-c:v','copy','-c:a','aac','-movflags','+faststart',str(final)],check=True)
    cap=cv2.VideoCapture(str(final));n=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(800,1280);n+=1
    cap.release();assert n==2*(end-start+1)
    assert sha(pp)==pm['reviewed_output_sha256'] and sha(bp)==bm['reviewed_output_sha256']
    summary=dict(frames=len(states),frames_with_four_player_observations=sum(s['observed_player_count']==4 for s in states),
        ball_output_frames=sum(s['ball_output_present'] for s in states),
        entity_rows=len(entities),video_decoded_frames=n,video=str(final.resolve()),
        inputs_unchanged=True,independent_tracking_accuracy_measured=False,
        warning='Observed output counts only; bbox bottom is not validated ground contact. Boundary intervals remain provisional.')
    (OUT/'verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output-name',default='integrated_044')
    main(parser.parse_args().output_name)
