"""Synchronized 044 court-player proxies, assisted image-ball path and event proposals.

No new model inference, label edits, missing-coordinate filling or ball homography.
"""
import sys, json, csv, hashlib, math, subprocess, shutil
from pathlib import Path
from collections import defaultdict, Counter
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE=Path(__file__).resolve().parent
WORK=HERE.parent.parent
OLD=HERE.parent/'ball_quality_20261103'
PRIOR=HERE.parent/'ball_quality_20261109'
SOURCE=Path(r'C:\Users\ta239\Research\softtennis_match_study\study')
OUT=HERE/'044_spatiotemporal_pilot'
BALL=PRIOR/'rally_repair_pilot_044_v2/completed_review'
PERSON=OLD/'person_044_pose_recovery/user_reviewed/observed_player_paths.csv'
FRAMES=SOURCE/'data/balltrack/racketvision/match1/match1_motion_pilot_016/frames'
CAL=SOURCE/'config/match1_court_plane_calibration.json'
LABEL=HERE.parent/'frontcourt_handoff_20260927/rally_label_review.csv'
FPS=60000/1001
SLOTS=['near_front','near_rear','far_rear','far_front']
NAMES=dict(zip(SLOTS,['N1','N2','F1','F2']))
RGB=dict(zip(SLOTS,[(65,220,120),(70,170,255),(220,90,230),(250,210,65)]))

def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def dump(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(p,rows):
    if not rows:return
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def grouped(rows):
    out=defaultdict(list)
    for r in rows:out[int(r['source_frame'])].append(r)
    return out
def project(H,x,y):
    p=np.asarray(H)@np.array([x,y,1.]);assert abs(p[2])>1e-9
    return (p[:2]/p[2]).tolist()

def prepare():
    assert not OUT.exists()
    scope=load(OLD/'play_start_audit/044_scope_report.json')
    result=load(BALL/'result.json');assert all(sha(BALL/n)==h for n,h in result['export_hashes'].items())
    balls=read(BALL/'assisted_states_sampled.csv');assert len(balls)==201
    times=list(range(115365,115791,2))
    paths=[Path(__file__),PERSON,CAL,LABEL,OLD/'play_start_audit/044_scope_report.json',
        HERE.parent/'frontcourt_handoff_20260927/044_serve_attempt_review.csv',
        PRIOR/'rally_analysis_prototype/protocol.json',PRIOR/'rally_analysis_prototype/prototype_rally_metadata.csv',
        HERE.parent/'ball_quality_20261110/current22_player_inputs/publication_hashes.json']
    paths.extend(BALL/n for n in ['assisted_states_sampled.csv','result.json','verification.json','cases.json'])
    paths.extend(FRAMES/f'{f-114635:06d}.jpg' for f in times)
    OUT.mkdir()
    dump(OUT/'protocol.json',dict(purpose='One already reviewed same-match development rally, a synchronized inspection prototype; not an automated tactical system or accuracy result.',
        date_jst='2026-11-11',rally='match1_rally_window_044',source_clock='legacy_edited_match1',
        raw_video_replacement_warning='Use registered cached1920x1080 JPEGs and legacy source_frame keys. Do not seek the re-downloaded match1.mp4 with these old frame IDs.',
        source_frame_offset=114635,display_source_frames=times,source_fps=FPS,
        display='Every second source frame, half speed. Exact actual source image each display frame. No interpolation of images, player positions or ball positions.',
        ball_scope='Exactly201 human-assisted reviewed times115375..115775 stride2; preserve37 unavailable points. Earlier/later display times have no ball annotation, not a claim of invisible ball.',
        players='Original3840x2160 detection/pose-recovery boxes and bbox-bottom proxy points, converted to1920x1080 for overlay. Existing human far-pair clip correspondence retained; not independent all-frame identity or foot GT.',
        court='Retained match1 homography on original3840x2160 point coordinates. Normalized court plane: x0left..1right,y0near..1far. Project only person bbox-bottom proxies. No calibrated metres or verified ground-contact claim. Do not clamp out-of-court points.',
        ball_geometry='Image coordinates and image trajectory only. NEVER apply court-plane homography to airborne ball. No3D height, speed, predicted landing or true bounce coordinate.',
        event_candidate_rule=dict(source='new unvalidated engineering proposal, frozen before outputs',
            neighborhood='5 consecutive reviewed visible samples at stride2: source f-4,f-2,f,f+2,f+4',
            direction='incoming and outgoing least-squares image velocity fitted to3samples each sharing center',
            minimum_velocity_px_per_source_frame=.5,minimum_turn_degrees=45,minimum_separation_source_frames=12,
            suppression='Sort candidates by turn angle descending, retain at least12frames apart; report both near-player and other turns. No claim that a direction turn is a hit.',
            player_context='Distance from ball to person bbox rectangle divided by bbox height. Near-player candidate if <=0.5. Context only, no forced hitter identity.'),
        service_first_fault='Earlier first fault is the same point, separate attempt; retained source label, not rendered as main rally.',
        contact_and_end='Existing AI provisional serve bracket115365..115373, end bracket115776..115790; effective serve/out/near winner human-confirmed, exact event frames/hitter/landing not confirmed.',
        protected_input_hashes={str(p):sha(p) for p in paths},independent_accuracy_established=False,
        no_new_human_labels=True,no_training=True,no_canonical_outputs_changed=True))
    print('Frozen044 synchronized display, retained201assisted ball times and provisional event scope.',flush=True)

def build():
    p=load(OUT/'protocol.json');assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    H=load(CAL)['image_to_court_homography'];by=grouped(read(PERSON));balls={int(r['source_frame']):r for r in read(BALL/'assisted_states_sampled.csv')}
    people=[];ballrows=[];states=[]
    for f in p['display_source_frames']:
        for r in by[f]:
            u,v=project(H,float(r['bottom_center_x']),float(r['bottom_center_y']))
            people.append(dict(source_frame=f,local_frame=f-114635,player_slot=r['player_slot'],track_id=r['track_id'],
                image_x_px=float(r['bottom_center_x'])/2,image_y_px=float(r['bottom_center_y'])/2,
                x1_px=float(r['x1'])/2,y1_px=float(r['y1'])/2,x2_px=float(r['x2'])/2,y2_px=float(r['y2'])/2,
                court_proxy_x=u,court_proxy_y=v,position_source=r['position_source'],
                proxy_quality='bbox_bottom_projected_not_verified_foot',independent_position_validated=0,
                identity_status=r.get('identity_status',''),identity_review=r.get('human_video_review_id','')))
        b=balls.get(f);present=b is not None and b['observation_present']=='1'
        ballrows.append(dict(source_frame=f,local_frame=f-114635,annotation_available=int(b is not None),observation_present=int(present),
            image_x_px=b['x_px'] if present else '',image_y_px=b['y_px'] if present else '',
            coordinate_source=b['coordinate_source'] if b else 'not_annotated',review_decision=b['decision'] if b else '',
            court_x='',court_y='',height='',speed='',independent_position_validated=0))
        states.append(dict(source_frame=f,relative_to_serve_bracket_end_sec=(f-115373)/FPS,
            phase='serve_contact_provisional_interval' if f<=115373 else 'candidate_main_play' if f<=115775 else 'point_end_provisional_interval',
            person_candidates=len(by[f]),ball_annotation_available=int(b is not None),ball_observation_present=int(present)))
    write(OUT/'player_court_proxies.csv',people);write(OUT/'ball_image_observations.csv',ballrows);write(OUT/'frame_states.csv',states)
    points={r['source_frame']:np.array([float(r['image_x_px']),float(r['image_y_px'])]) for r in ballrows if r['observation_present']}
    candidates=[];trace=[]
    for f in sorted(points):
        if not all(f+d in points for d in [-4,-2,0,2,4]):continue
        before=np.array([points[f+d] for d in [-4,-2,0]]);after=np.array([points[f+d] for d in [0,2,4]])
        vin=(before[-1]-before[0])/4;vout=(after[-1]-after[0])/4
        norms=[float(np.linalg.norm(v)) for v in [vin,vout]]
        if min(norms)<.5:continue
        angle=math.degrees(math.acos(float(np.clip(np.dot(vin,vout)/np.prod(norms),-1,1))))
        context=[]
        for r in by[f]:
            x,y=points[f];l,t,rgt,b=[float(r[k])/2 for k in ['x1','y1','x2','y2']]
            dist=math.hypot(max(l-x,0,x-rgt),max(t-y,0,y-b))/max(b-t,1)
            context.append((dist,r['player_slot']))
        nearest=min(context) if context else (float('inf'),'')
        rr=dict(source_frame=f,relative_sec=(f-115373)/FPS,turn_degrees=angle,
            incoming_image_velocity_px_per_source_frame=norms[0],outgoing_image_velocity_px_per_source_frame=norms[1],
            nearest_player_slot=nearest[1],ball_to_box_distance_in_heights=nearest[0],
            event_type='near_player_direction_change_candidate' if nearest[0]<=.5 else 'image_direction_change_candidate',
            event_confirmation='unreviewed_proposal_not_hit_or_bounce_GT',hitter_confirmed='',ball_x_px=float(points[f][0]),ball_y_px=float(points[f][1]))
        trace.append(dict(rr,passes_turn_rule=int(angle>=45)))
        if angle>=45:candidates.append(rr)
    kept=[]
    for r in sorted(candidates,key=lambda r:(-r['turn_degrees'],r['source_frame'])):
        if all(abs(r['source_frame']-v['source_frame'])>=12 for v in kept):kept.append(r)
    kept.sort(key=lambda r:r['source_frame'])
    for i,r in enumerate(kept,1):r['event_id']=f'C{i:02}'
    write(OUT/'all_velocity_turn_diagnostics.csv',trace);write(OUT/'event_candidates.csv',kept)
    label=next(r for r in read(LABEL) if r['candidate_id']=='match1_rally_window_044')
    dump(OUT/'inherited_outcome_and_event_scope.json',dict(winner_side=label['winner_camera_side'],winner_status=label['winner_camera_side_review_status'],
        end_type=label['point_end_event_type'],end_status=label['point_end_event_review_status'],serve_bracket=[115365,115373],
        end_bracket=[115776,115790],exact_event_times_confirmed=False,scope='Existing human event types and side; provisional AI temporal intervals. No new labels.',first_fault_same_point_preserved=True))
    dump(OUT/'build_result.json',dict(display_times=len(states),ball_annotated_times=sum(r['annotation_available'] for r in ballrows),
        ball_positions=sum(r['observation_present'] for r in ballrows),person_observations=len(people),
        four_person_times=sum(r['person_candidates']==4 for r in states),direction_change_candidates=len(kept),
        confirmed_new_hits=0,ball_court_positions=0,missing_filled=False))
    # Image evidence for every proposal, including low-context turns.
    tiles=[]
    for r in kept:
        f=r['source_frame'];im=cv2.imread(str(FRAMES/f'{f-114635:06d}.jpg'));assert im is not None
        x,y=r['ball_x_px'],r['ball_y_px'];l=int(np.clip(round(x-240),0,1440));t=int(np.clip(round(y-135),0,810))
        crop=im[t:t+270,l:l+480].copy();cv2.circle(crop,(round(x-l),round(y-t)),7,(70,255,110),2)
        header=np.zeros((50,480,3),np.uint8)
        cv2.putText(header,f'{r["event_id"]} F{f} turn={r["turn_degrees"]:.0f}deg',(8,20),0,.55,(255,255,255),1,cv2.LINE_AA)
        cv2.putText(header,f'Near {NAMES.get(r["nearest_player_slot"],"?")} / candidate only',(8,42),0,.5,(255,255,255),1,cv2.LINE_AA)
        tile=np.vstack([header,crop]);cv2.imwrite(str(OUT/f'{r["event_id"]}_F{f}.jpg'),tile);tiles.append(tile)
    if tiles:
        if len(tiles)%2:tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(str(OUT/'all_direction_change_candidates.jpg'),np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)]))
    print(load(OUT/'build_result.json'),flush=True)

FONT=Path('C:/Windows/Fonts/meiryo.ttc')
def render():
    p=load(OUT/'protocol.json');people=grouped(read(OUT/'player_court_proxies.csv'));ball={int(r['source_frame']):r for r in read(OUT/'ball_image_observations.csv')}
    events=read(OUT/'event_candidates.csv') if (OUT/'event_candidates.csv').exists() else []
    fonts={s:ImageFont.truetype(str(FONT),s) for s in [17,19,21,24,28,34]}
    times=p['display_source_frames'];mapping=[];fps=FPS/4
    proc=subprocess.Popen(['C:/ffmpeg/bin/ffmpeg.exe','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'044_synchronized_half_speed.mp4')],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    snapshots=[115375,115575,115775]
    for i,f in enumerate(times):
        raw=cv2.imread(str(FRAMES/f'{f-114635:06d}.jpg'));assert raw is not None
        img=Image.new('RGB',(1920,1080),(12,22,32));draw=ImageDraw.Draw(img)
        def text(x,y,s,size=21,c=(225,235,245)):draw.text((x,y),s,font=fonts[size],fill=c)
        text(24,12,'044：選手配置・球軌跡・事象候補の同期表示',34)
        text(25,59,'半分の速度 ｜ 球は人手整備済み ｜ コート図は人物枠下端の投影（足元精度は未検証）',21)
        view=Image.fromarray(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)).resize((1280,720),Image.Resampling.LANCZOS);img.paste(view,(16,103));draw=ImageDraw.Draw(img)
        # Actual boxes and current image points only.
        for r in people[f]:
            col=RGB[r['player_slot']];x1,y1,x2,y2=[float(r[k])*2/3 for k in ['x1_px','y1_px','x2_px','y2_px']]
            draw.rectangle((16+x1,103+y1,16+x2,103+y2),outline=col,width=2)
            text(int(16+x1),int(76+y1),NAMES[r['player_slot']],21,col)
        # Short image trail; never bridge any missing annotation or missing observation.
        for g in range(max(times[0]+2,f-36),f+1,2):
            a,b=ball[g-2],ball[g]
            if a['observation_present']=='1' and b['observation_present']=='1':
                xy=[(16+float(r['image_x_px'])*2/3,103+float(r['image_y_px'])*2/3) for r in [a,b]]
                draw.line(xy,fill=(65,250,160),width=3)
        b=ball[f]
        if b['observation_present']=='1':
            x=16+float(b['image_x_px'])*2/3;y=103+float(b['image_y_px'])*2/3
            draw.ellipse((x-7,y-7,x+7,y+7),outline=(100,255,130),width=3)
        else:text(34,775,'球位置：観測不能' if b['annotation_available']=='1' else '球位置：この時点は未注釈',21,(255,210,105))
        text(1330,108,'4選手の配置（暫定）',28)
        text(1330,151,'球は左の映像上に表示・高さ未推定',19)
        # Court aspect10.97:23.77, approximated only for schematic layout; no metre statistics.
        left,top,w,h=1470,208,260,560
        def cp(u,v):return left+w*u,top+h*(1-v)
        draw.rectangle((left,top,left+w,top+h),outline=(160,195,210),width=2)
        draw.line([cp(0,.5),cp(1,.5)],fill=(220,230,240),width=3)
        for u in [.125,.875]:draw.line([cp(u,0),cp(u,1)],fill=(100,140,160),width=1)
        # Schematic service lines and center service line; proportional existing court convention.
        for v in [.231,.769]:draw.line([cp(0,v),cp(1,v)],fill=(100,140,160),width=1)
        draw.line([cp(.5,0),cp(.5,.231)],fill=(100,140,160),width=1);draw.line([cp(.5,.769),cp(.5,1)],fill=(100,140,160),width=1)
        text(left+95,175,'奥側',21);text(left+95,775,'手前側',21)
        seen=set()
        for r in people[f]:
            slot=r['player_slot'];seen.add(slot);u,v=float(r['court_proxy_x']),float(r['court_proxy_y']);x,y=cp(u,v)
            if 1330<x<1890 and 180<y<815:
                col=RGB[slot];draw.ellipse((x-8,y-8,x+8,y+8),fill=col);text(int(x+11),int(y-15),NAMES[slot],21,col)
        absent=[NAMES[s] for s in SLOTS if s not in seen]
        text(1330,824,'人物枠の欠測：'+('・'.join(absent) if absent else 'なし'),21,(255,210,105) if absent else (210,230,235))
        text(24,836,f'元フレーム {f} ｜ サーブ候補区間末尾から {(f-115373)/FPS:+.2f}秒',21)
        text(24,869,'緑線＝画像上の球軌跡（欠測をまたいで線を結ばない）',21,(100,255,165))
        text(24,907,'事象の時間軸：方向変化は打球・バウンドの候補。自動判定の正解率は未評価。',21)
        # Known interval labels are retained as intervals, never a fabricated exact contact time.
        lx,rx=50,1850;ly=993
        def tx(g):return lx+(g-times[0])/(times[-1]-times[0])*(rx-lx)
        draw.line((lx,ly,rx,ly),fill=(110,140,160),width=3)
        draw.rectangle((tx(115365),ly-9,tx(115373),ly+9),fill=(60,150,220));text(40,1020,'サーブ候補区間',19)
        draw.rectangle((tx(115776),ly-9,tx(115789),ly+9),fill=(190,120,50));text(1675,1020,'終端候補区間',19)
        for j,r in enumerate(events):
            x=tx(int(r['source_frame']));col=(250,205,80)
            draw.line((x,ly-18,x,ly+18),fill=col,width=2)
            text(int(x-18),int(ly-58-(j%2)*27),r['event_id'],19,col)
        x=tx(f);draw.ellipse((x-7,ly-7,x+7,ly+7),fill=(255,255,255))
        # Human-confirmed outcome is context, not an inferred causal explanation.
        text(1330,864,'既存確認：サーブ有効／手前得点／アウト',17)
        near=[r for r in events if abs(int(r['source_frame'])-f)<=6]
        if near:text(32,115,'方向変化候補 '+near[0]['event_id']+'（打球確定ではない）',21,(255,220,90))
        data=np.asarray(img);proc.stdin.write(data.tobytes())
        mapping.append(dict(display_frame=i,display_time_sec=i/fps,source_frame=f,local_frame=f-114635,legacy_source_time_sec=f/FPS))
        if f in snapshots:img.save(OUT/f'synchronized_F{f}.jpg',quality=95)
    proc.stdin.close();err=proc.stderr.read();assert proc.wait()==0,err
    write(OUT/'video_source_frame_map.csv',mapping)
    print('Rendered',len(times),'actual-image frames at half speed.',flush=True)

def verify():
    p=load(OUT/'protocol.json');assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    people=read(OUT/'player_court_proxies.csv');ball=read(OUT/'ball_image_observations.csv');times=p['display_source_frames']
    original={(int(r['source_frame']),r['player_slot']):r for r in read(PERSON)};H=np.array(load(CAL)['image_to_court_homography']);roundtrip=0.
    for r in people:
        old=original[int(r['source_frame']),r['player_slot']]
        assert r['track_id']==old['track_id']
        for a,b in [('image_x_px','bottom_center_x'),('image_y_px','bottom_center_y'),('x1_px','x1'),('y1_px','y1'),('x2_px','x2'),('y2_px','y2')]:assert abs(float(r[a])*2-float(old[b]))<1e-9
        x,y=project(np.linalg.inv(H),float(r['court_proxy_x']),float(r['court_proxy_y']));roundtrip=max(roundtrip,abs(x-float(old['bottom_center_x'])),abs(y-float(old['bottom_center_y'])))
    assert roundtrip<1e-6
    original_ball={int(r['source_frame']):r for r in read(BALL/'assisted_states_sampled.csv')}
    for r in ball:
        old=original_ball.get(int(r['source_frame']))
        assert r['court_x']==r['court_y']==r['height']==r['speed']==''
        assert r['observation_present']==(old['observation_present'] if old else '0')
        for a,b in [('image_x_px','x_px'),('image_y_px','y_px')]:
            assert r[a]==(old[b] if old and old['observation_present']=='1' else '')
    cap=cv2.VideoCapture(str(OUT/'044_synchronized_half_speed.mp4'));n=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape==(1080,1920,3);n+=1
    cap.release();assert n==len(times)==213
    mapping=read(OUT/'video_source_frame_map.csv');assert [int(r['source_frame']) for r in mapping]==times
    dump(OUT/'verification.json',dict(protected_inputs_and_cached_images_unchanged=True,person_rows_exact_actual_source_boxes=len(people),
        projected_person_proxy_inverse_error_max_canonical_px=roundtrip,ball_annotated_rows_exact_original=201,
        ball_positions_retained=164,ball_unavailable_retained=37,extra_display_times_without_ball_annotation=12,
        ball_homography_or_3D_positions_created=False,decoded_video_frames=n,source_frame_alignment='exact integer legacy IDs',
        source_images_not_retimed_to_replaced_raw_video=True,new_human_labels_created=False,independent_accuracy_established=False))
    shutil.copy2(Path(__file__),OUT/'executed_source.py')
    dump(OUT/'publication_hashes.json',{str(n.relative_to(OUT)):sha(n) for n in sorted(OUT.rglob('*')) if n.is_file() and n.name!='publication_hashes.json'})
    print('Verified exact provenance,213video frames,164ball points,37missing; no ball-plane projection.',flush=True)

if __name__=='__main__':{'prepare':prepare,'build':build,'render':render,'verify':verify}[sys.argv[1]]()
