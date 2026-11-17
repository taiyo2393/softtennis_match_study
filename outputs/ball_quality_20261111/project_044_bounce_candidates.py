"""Conditional ground-plane projection of two existing AI bounce candidates.

Keep airborne ball points in image space and all human confirmations scoped.
"""
from pathlib import Path
import shutil,sys,math,subprocess
from PIL import Image,ImageDraw,ImageFont
import cv2,numpy as np
import build_044_spatiotemporal_pilot as b

OUT=b.HERE/'044_bounce_projection'
STROKE=b.HERE/'044_stroke_review'
CONFIRMED=b.HERE/'044_sequence_confirmed'
CANDIDATES=[dict(id='B01',old_direction_candidate='C01',source_frame=115415,interval_first=115413,interval_last=115419,
    after_stroke='H01',before_stroke='H02',hitter_slot='far_rear',hitter_reference_frame=115369),
    dict(id='B02',old_direction_candidate='C03',source_frame=115571,interval_first=115569,interval_last=115575,
    after_stroke='H03',before_stroke='H04',hitter_slot='far_rear',hitter_reference_frame=115521)]

def prepare():
    assert not OUT.exists()
    files=[Path(__file__),b.CAL,b.OUT/'publication_hashes.json',STROKE/'publication_hashes.json',
        STROKE/'AI_event_proposals.json',CONFIRMED/'publication_hashes.json',CONFIRMED/'human_sequence_confirmation.json',
        b.OUT/'ball_image_observations.csv',b.OUT/'player_court_proxies.csv',b.OUT/'protocol.json']
    OUT.mkdir()
    b.dump(OUT/'protocol.json',dict(date_jst='2026-11-11',purpose='Add tentative bounce points to court diagram in same reviewed044 development rally; not validated landing measurements.',
        candidates=CANDIDATES,selection='ExistingC01/C03 frames selected in previous image-turn proposal, not selected or adjusted after seeing court projection. No new frame search or threshold tuning.',
        geometry='Native1920x1080 human-assisted ball center -> multiplyxy by2 -> retained3840x2160 image-to-court H. Conditional assumption: point is at ground level. No ball radius/height correction; exact contact time and pixel center error unvalidated.',
        coordinate_definition='Normalized court:x0screen-left doubles sideline..1right; y0near baseline..1far. No clipping/clamping or in/out decision.',
        identity_and_review_scope='User confirmed5stroke actor sequence only. AI bounce events/timing/positions are not human-confirmed. Accepted ball positions are human-assisted, not independent pixel-center truth.',
        map_display='Only2fixed bounce-candidate points. They appear at their representative source frame and remain as historical hollow rings; never shown as current airborne ball position. Current image-ball tracking stays on original video.',
        no_new_positions_for=['airborne_ground_projection','hidden_far_bounces','last_out_landing','missing_ball_frames'],
        no_interpolation=True,independent_accuracy_established=False,
        protected_input_hashes={str(p):b.sha(p) for p in files}))

def build():
    p=b.load(OUT/'protocol.json');assert all(b.sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    H=b.load(b.CAL)['image_to_court_homography'];balls={int(r['source_frame']):r for r in b.read(b.OUT/'ball_image_observations.csv')}
    people=b.grouped(b.read(b.OUT/'player_court_proxies.csv'));rows=[];evidence=[]
    for c in CANDIDATES:
        r=balls[c['source_frame']];assert r['observation_present']=='1'
        x,y=float(r['image_x_px']),float(r['image_y_px']);u,v=b.project(H,x*2,y*2)
        actor=next(r for r in people[c['hitter_reference_frame']] if r['player_slot']==c['hitter_slot'])
        rows.append(dict(candidate_id=c['id'],source_frame=c['source_frame'],interval_first=c['interval_first'],interval_last=c['interval_last'],
            after_stroke=c['after_stroke'],before_stroke=c['before_stroke'],image_x_px=x,image_y_px=y,
            assumed_ground_court_x=u,assumed_ground_court_y=v,coordinate_source=r['coordinate_source'],
            event_status='AI_bounce_candidate_not_human_confirmed',projection_status='conditional_on_ground_level_not_validated_landing',
            exact_bounce_time_confirmed=0,landing_position_confirmed=0,independent_position_GT=0,
            prior_hitter_slot=c['hitter_slot'],prior_hitter_reference_frame=c['hitter_reference_frame'],
            prior_hitter_court_proxy_x=actor['court_proxy_x'],prior_hitter_court_proxy_y=actor['court_proxy_y'],
            prior_hitter_position_is_not_ball_contact_point=1))
        path=b.FRAMES/f'{c["source_frame"]-114635:06d}.jpg';evidence.append(dict(candidate_id=c['id'],source_image=str(path),sha256=b.sha(path)))
    b.write(OUT/'conditional_bounce_court_points.csv',rows);b.dump(OUT/'source_image_manifest.json',evidence)
    b.dump(OUT/'result.json',dict(conditional_bounce_points=2,all_frame_ball_projection=False,
        confirmed_stroke_sequence_reused=True,new_human_bounce_confirmations=0,new_landing_truth=0,
        physical_ball_height_or_speed_estimated=False,missing_coordinates_filled=0))

def render():
    points=b.read(OUT/'conditional_bounce_court_points.csv');times=b.load(b.OUT/'protocol.json')['display_source_frames']
    players=b.grouped(b.read(b.OUT/'player_court_proxies.csv'));balls={int(r['source_frame']):r for r in b.read(b.OUT/'ball_image_observations.csv')}
    fonts={s:ImageFont.truetype(str(b.FONT),s) for s in [17,19,21,24,28,34]};color=(255,185,65)
    fps=b.FPS/4;mapping=[]
    proc=subprocess.Popen(['C:/ffmpeg/bin/ffmpeg.exe','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'044_bounce_candidates_half_speed.mp4')],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    snapshots=[115415,115571,115575,115729]
    for i,f in enumerate(times):
        raw=cv2.imread(str(b.FRAMES/f'{f-114635:06d}.jpg'));assert raw is not None
        image=Image.new('RGB',(1920,1080),(12,22,32));d=ImageDraw.Draw(image)
        def txt(x,y,s,size=21,c=(225,235,245)):d.text((x,y),s,font=fonts[size],fill=c)
        txt(24,12,'044：コート上にバウンド候補を追加',34)
        txt(25,62,'橙の輪＝地面上と仮定した暫定点 ｜ 打球順5回は確認済み、バウンドは未確認',21)
        image.paste(Image.fromarray(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)).resize((1280,720),Image.Resampling.LANCZOS),(16,105));d=ImageDraw.Draw(image)
        for r in players[f]:
            c=b.RGB[r['player_slot']];x1,y1,x2,y2=[float(r[k])*2/3 for k in ['x1_px','y1_px','x2_px','y2_px']]
            d.rectangle((16+x1,105+y1,16+x2,105+y2),outline=c,width=2);txt(int(16+x1),int(79+y1),b.NAMES[r['player_slot']],21,c)
        # Keep image-space ball path and all its gaps as before.
        for g in range(max(times[0]+2,f-36),f+1,2):
            a,z=balls[g-2],balls[g]
            if a['observation_present']=='1' and z['observation_present']=='1':d.line([(16+float(v['image_x_px'])*2/3,105+float(v['image_y_px'])*2/3) for v in [a,z]],fill=(70,245,155),width=3)
        if balls[f]['observation_present']=='1':
            x=16+float(balls[f]['image_x_px'])*2/3;y=105+float(balls[f]['image_y_px'])*2/3;d.ellipse((x-7,y-7,x+7,y+7),outline=(100,255,130),width=3)
        else:txt(32,780,'球位置：観測不能' if balls[f]['annotation_available']=='1' else '球位置：この時点は未注釈',21,(250,210,100))
        txt(1335,111,'選手配置とバウンド候補',28)
        txt(1335,158,'選手＝枠下端の投影（暫定）',19)
        left,top,w,h=1465,235,240,520
        def cp(u,v):return left+w*u,top+h*(1-v)
        d.rectangle((left,top,left+w,top+h),outline=(150,185,205),width=2)
        for v in [.231,.5,.769]:d.line([cp(0,v),cp(1,v)],fill=(205,220,235) if v==.5 else (90,130,155),width=3 if v==.5 else 1)
        for u in [.125,.875]:d.line([cp(u,0),cp(u,1)],fill=(90,130,155),width=1)
        d.line([cp(.5,0),cp(.5,.231)],fill=(90,130,155));d.line([cp(.5,.769),cp(.5,1)],fill=(90,130,155))
        txt(left+90,192,'奥側');txt(left+90,762,'手前側')
        present=set()
        for r in players[f]:
            present.add(r['player_slot']);x,y=cp(float(r['court_proxy_x']),float(r['court_proxy_y']))
            if 1335<x<1895 and 190<y<807:
                c=b.RGB[r['player_slot']];d.ellipse((x-7,y-7,x+7,y+7),fill=c);txt(int(x+10),int(y-14),b.NAMES[r['player_slot']],19,c)
        for point in points:
            pf=int(point['source_frame'])
            if f<pf:continue
            x,y=cp(float(point['assumed_ground_court_x']),float(point['assumed_ground_court_y']))
            active=f<=pf+12;rad=12 if active else 9
            d.ellipse((x-rad,y-rad,x+rad,y+rad),outline=color,width=3 if active else 2)
            txt(int(x+15),int(y-14),point['candidate_id'],19,color)
            if active:
                # Mark the same fixed reference image location briefly; not a moving ground-ball estimate.
                ix=16+float(point['image_x_px'])*2/3;iy=105+float(point['image_y_px'])*2/3
                d.ellipse((ix-13,iy-13,ix+13,iy+13),outline=color,width=2)
        txt(1335,812,'○ B01・B02＝既出の暫定バウンド点',19,color)
        txt(1335,847,'現在の空中球の位置ではありません。',19)
        txt(24,841,f'元フレーム {f} ｜ 半分の速度',21)
        txt(24,878,'緑＝その時点の画像上の球 ／ 橙＝固定したバウンド候補の代表位置',21)
        d.rounded_rectangle((24,934,1895,1047),radius=8,fill=(20,35,48),outline=(60,85,100))
        txt(43,947,'B01：1打目のサーブ後 → 2打目の返球前',24,color)
        txt(986,947,'B02：3打目の返球後 → 4打目の返球前',24,color)
        txt(43,991,'バウンドの時刻・座標はAI暫定。隠れたバウンドや最後のアウト落点は補っていません。',21)
        proc.stdin.write(np.asarray(image).tobytes());mapping.append(dict(display_frame=i,source_frame=f,display_time_sec=i/fps))
        if f in snapshots:image.save(OUT/f'bounce_projection_F{f}.jpg',quality=95)
    proc.stdin.close();err=proc.stderr.read();assert proc.wait()==0,err;b.write(OUT/'video_source_frame_map.csv',mapping)
    # A summary with two raw-image snippets and one common court; no fictitious ball-path segments.
    image=Image.new('RGB',(1600,1000),(12,22,32));d=ImageDraw.Draw(image)
    def txt(x,y,s,size=21,c=(225,235,245)):d.text((x,y),s,font=fonts[size],fill=c)
    txt(26,15,'044：2つのバウンド候補をコート上に置く',34)
    txt(27,69,'地面上の球位置と仮定した暫定投影。実測落点・現在の空中球の位置ではありません。',21)
    for i,point in enumerate(points):
        f=int(point['source_frame']);raw=cv2.imread(str(b.FRAMES/f'{f-114635:06d}.jpg'));x,y=float(point['image_x_px']),float(point['image_y_px'])
        l=round(x-250);t=round(y-150);crop=raw[t:t+300,l:l+500].copy()
        cv2.circle(crop,(250,150),10,(65,185,255),2);tile=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)).resize((750,450),Image.Resampling.LANCZOS)
        yy=153+i*392;image.paste(tile.resize((600,360)),(30,yy));txt(45,yy-35,f'{point["candidate_id"]}：元フレーム {f}',24,color)
    left,top,w,h=970,156,315,683
    def cp(u,v):return left+w*u,top+h*(1-v)
    d.rectangle((left,top,left+w,top+h),outline=(150,185,205),width=2)
    for v in [.231,.5,.769]:d.line([cp(0,v),cp(1,v)],fill=(205,220,235) if v==.5 else (90,130,155),width=3 if v==.5 else 1)
    for u in [.125,.875]:d.line([cp(u,0),cp(u,1)],fill=(90,130,155))
    txt(left+120,116,'奥側');txt(left+120,849,'手前側')
    for point in points:
        x,y=cp(float(point['assumed_ground_court_x']),float(point['assumed_ground_court_y']));d.ellipse((x-12,y-12,x+12,y+12),outline=color,width=3);txt(int(x+18),int(y-15),point['candidate_id'],24,color)
    txt(30,934,'点は以前の候補フレームの球中心から計算。正確な接触時刻・位置誤差は未検証です。',21)
    image.save(OUT/'two_bounce_candidates_on_court.jpg',quality=95)

def verify():
    p=b.load(OUT/'protocol.json');assert all(b.sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    parents={b.OUT:b.load(b.OUT/'publication_hashes.json'),STROKE:b.load(STROKE/'publication_hashes.json'),CONFIRMED:b.load(CONFIRMED/'publication_hashes.json')}
    for parent,entries in parents.items():assert all(b.sha(parent/n)==h for n,h in entries.items())
    points=b.read(OUT/'conditional_bounce_court_points.csv');H=np.array(b.load(b.CAL)['image_to_court_homography']);native=H@np.diag([2.,2.,1.])
    # Independent OpenCV route uses native-to-canonical scale composed into H.
    xy=np.array([[[float(r['image_x_px']),float(r['image_y_px'])] for r in points]],np.float64)
    test=cv2.perspectiveTransform(xy,native)[0];error=0.
    for r,z in zip(points,test):
        error=max(error,abs(float(r['assumed_ground_court_x'])-z[0]),abs(float(r['assumed_ground_court_y'])-z[1]))
        assert r['event_status']=='AI_bounce_candidate_not_human_confirmed' and r['landing_position_confirmed']=='0'
        old=next(v for v in b.read(b.OUT/'ball_image_observations.csv') if v['source_frame']==r['source_frame'])
        assert float(old['image_x_px'])==float(r['image_x_px']) and float(old['image_y_px'])==float(r['image_y_px'])
    assert error<1e-12 and len(points)==2
    mapping=b.read(OUT/'video_source_frame_map.csv');assert [int(r['source_frame']) for r in mapping]==b.load(b.OUT/'protocol.json')['display_source_frames']
    cap=cv2.VideoCapture(str(OUT/'044_bounce_candidates_half_speed.mp4'));n=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape==(1080,1920,3);n+=1
    cap.release();assert n==len(mapping)==213
    for r in b.load(OUT/'source_image_manifest.json'):assert b.sha(Path(r['source_image']))==r['sha256']
    b.dump(OUT/'verification.json',dict(parent_publication_files_unchanged={str(k):len(v) for k,v in parents.items()},
        conditional_projected_bounce_points=2,independent_OpenCV_arithmetic_max_difference=float(error),
        source_ball_centers_exactly_retained=True,airborne_ball_ground_positions_created=0,new_independent_landing_GT=0,
        hidden_bounces_or_out_landing_filled=0,decoded_video_frames=n,source_frame_map_exact=True))
    shutil.copy2(Path(__file__),OUT/'executed_source.py')
    b.dump(OUT/'publication_hashes.json',{str(n.relative_to(OUT)):b.sha(n) for n in sorted(OUT.rglob('*')) if n.is_file() and n.name!='publication_hashes.json'})
    print('Verified2conditional bounce projections,213video frames,unchanged parents. Not measured landing accuracy.',flush=True)

if __name__=='__main__':{'prepare':prepare,'build':build,'render':render,'verify':verify}[sys.argv[1]]()
