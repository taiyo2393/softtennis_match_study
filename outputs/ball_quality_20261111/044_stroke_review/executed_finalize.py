"""Build AI-provisional stroke records, unchanged proxy/ball inputs and review media."""
from pathlib import Path
from collections import Counter
import shutil,sys,subprocess,math
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import build_044_spatiotemporal_pilot as b
from review_044_stroke_events import OUT,WINDOWS

def chronology():
    protocol=b.load(OUT/'protocol.json');ball={int(r['source_frame']):r for r in b.read(b.OUT/'ball_image_observations.csv')};panels=[];images=[]
    for f in protocol['review_source_frames']:
        local=f-114635;path=b.FRAMES/f'{local:06d}.jpg';im=cv2.imread(str(path));assert im is not None
        h=np.zeros((36,775,3),np.uint8);cv2.putText(h,f'F{f} / local{local} / {(f-115373)/b.FPS:+.2f}s',(8,25),0,.63,(255,255,255),1,cv2.LINE_AA)
        r=ball.get(f)
        if r and r['observation_present']=='1':cv2.circle(im,(round(float(r['image_x_px'])),round(float(r['image_y_px']))),9,(80,255,100),2)
        panels.append(np.vstack([h,cv2.resize(im[450:950,300:1850],(775,250))]));images.append(dict(source_frame=f,image=str(path),sha256=b.sha(path)))
    for i in range(0,len(panels),12):
        batch=panels[i:i+12]
        while len(batch)<12:batch.append(np.zeros_like(panels[0]))
        cv2.imwrite(str(OUT/f'chronology_{i//12+1:02}.jpg'),np.vstack([np.hstack(batch[j:j+2]) for j in range(0,12,2)]))
    b.dump(OUT/'coarse_image_manifest.json',images)

def build():
    proposals=b.load(OUT/'AI_event_proposals.json');strokes=proposals['strokes']
    people=b.grouped(b.read(b.OUT/'player_court_proxies.csv'));balls={int(r['source_frame']):r for r in b.read(b.OUT/'ball_image_observations.csv')}
    records=[];snapshots=[];flights=[]
    for i,e in enumerate(strokes):
        f=e['reference'];actual={r['player_slot']:r for r in people[f]};actor=actual.get(e['hitter_candidate']);ball=balls[f]
        row=dict(stroke_id=e['stroke_id'],ordinal=e['ordinal'],event_type_candidate=e['kind'],
            source_frame_first=e['first'],source_frame_last=e['last'],reference_source_frame=f,
            approximate_reference_sec=(f-115373)/b.FPS,hitter_candidate=e['hitter_candidate'],
            next_hitter_candidate=e['next_hitter_candidate'],next_hitter_is_not_landing_point=1,
            all_four_position_proxies_present=int(len(actual)==4),hitter_proxy_present=int(actor is not None),
            hitter_court_proxy_x=actor['court_proxy_x'] if actor else '',hitter_court_proxy_y=actor['court_proxy_y'] if actor else '',
            hitter_coordinate_meaning='projected_bbox_bottom_at_reference_time_not_true_ball_contact',
            ball_annotation_available=ball['annotation_available'],ball_observation_present=ball['observation_present'],
            ball_image_x=ball['image_x_px'],ball_image_y=ball['image_y_px'],ball_court_x='',ball_court_y='',
            contact_x='',contact_y='',landing_x='',landing_y='',shot_type_confirmed='',
            event_type_status=e['event_type_status'],timing_status=e['timing_status'],hitter_status=e['hitter_status'],
            independent_event_GT=0,evidence=e['evidence'])
        records.append(row)
        for s in b.SLOTS:
            r=actual.get(s);snapshots.append(dict(stroke_id=e['stroke_id'],reference_source_frame=f,player_slot=s,observed=int(r is not None),
                image_x_px=r['image_x_px'] if r else '',image_y_px=r['image_y_px'] if r else '',
                court_proxy_x=r['court_proxy_x'] if r else '',court_proxy_y=r['court_proxy_y'] if r else '',
                raw_track_id=r['track_id'] if r else '',position_source=r['position_source'] if r else 'missing',independent_position_validated=0))
        end=strokes[i+1]['first']-1 if i+1<len(strokes) else 115789
        first=e['last']+1;sequence=[r for g,r in sorted(balls.items()) if first<=g<=end]
        visible=[r for r in sequence if r['observation_present']=='1']
        dx=float(visible[-1]['image_x_px'])-float(visible[0]['image_x_px']) if len(visible)>=2 else None
        # A descriptive overall image trend; no smoothing/interpolation or physical flight inference.
        row['sampled_after_hit_ball_positions']=len(visible)
        row['observed_image_x_trend']='screen_right' if dx is not None and dx>0 else 'screen_left' if dx is not None and dx<0 else 'unknown'
        for r in sequence:
            flights.append(dict(stroke_id=e['stroke_id'],source_frame=r['source_frame'],observation_present=r['observation_present'],
                annotation_available=r['annotation_available'],image_x_px=r['image_x_px'],image_y_px=r['image_y_px'],
                coordinate_source=r['coordinate_source'],court_x='',court_y='',segment_scope='after_AI_contact_bracket_before_next_AI_contact_bracket'))
    b.write(OUT/'AI_stroke_records.csv',records);b.write(OUT/'stroke_player_snapshot_proxies.csv',snapshots);b.write(OUT/'sampled_image_flights.csv',flights)
    # Ball missing observations are retained separately from inferred event labels.
    missing=[r for r in balls.values() if r['annotation_available']=='1' and r['observation_present']=='0'];runs=[]
    for r in missing:
        f=int(r['source_frame'])
        if not runs or f!=runs[-1][-1]+2:runs.append([f])
        else:runs[-1].append(f)
    b.dump(OUT/'ball_missing_interval_inventory.json',[dict(first_sampled_frame=r[0],last_sampled_frame=r[-1],missing_samples=len(r),
        physical_ball_visibility_or_event_type='not inferred from missing center') for r in runs])
    b.dump(OUT/'table_summary.json',dict(AI_proposed_strokes=len(records),new_human_confirmed_strokes=0,
        independent_event_accuracy_established=False,representative_player_slots=20,observed_position_proxies=sum(r['observed'] for r in snapshots),
        missing_position_proxies=sum(not r['observed'] for r in snapshots),ball_reference_times_with_position=sum(r['ball_observation_present']=='1' for r in records),
        statement='Representative frames are within intervals, not exact impact timestamps. Retained outcome facts are separate from AI actor proposals.'))
    print(b.load(OUT/'table_summary.json'),flush=True)

def render():
    proposal=b.load(OUT/'AI_event_proposals.json');strokes=proposal['strokes'];times=b.load(b.OUT/'protocol.json')['display_source_frames']
    people=b.grouped(b.read(b.OUT/'player_court_proxies.csv'));balls={int(r['source_frame']):r for r in b.read(b.OUT/'ball_image_observations.csv')}
    font={s:ImageFont.truetype(str(b.FONT),s) for s in [16,18,20,22,25,30,34]}
    fps=b.FPS/4;mapping=[]
    proc=subprocess.Popen(['C:/ffmpeg/bin/ffmpeg.exe','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'044_AI_stroke_sequence_half_speed.mp4')],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    for i,f in enumerate(times):
        raw=cv2.imread(str(b.FRAMES/f'{f-114635:06d}.jpg'));assert raw is not None
        img=Image.new('RGB',(1920,1080),(12,22,32));dr=ImageDraw.Draw(img)
        def txt(x,y,s,size=20,col=(225,235,245)):dr.text((x,y),s,font=font[size],fill=col)
        txt(24,12,'044：一打ごとの展開を整理する試作',34)
        txt(25,60,'AI暫定：サーブを含む5打球候補 ｜ 球は人手整備済み ｜ 半分の速度',22)
        img.paste(Image.fromarray(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)).resize((1280,720),Image.Resampling.LANCZOS),(16,103));dr=ImageDraw.Draw(img)
        for r in people[f]:
            col=b.RGB[r['player_slot']];x1,y1,x2,y2=[float(r[k])*2/3 for k in ['x1_px','y1_px','x2_px','y2_px']]
            dr.rectangle((16+x1,103+y1,16+x2,103+y2),outline=col,width=2);txt(int(16+x1),int(77+y1),b.NAMES[r['player_slot']],20,col)
        for g in range(max(times[0]+2,f-36),f+1,2):
            a,z=balls[g-2],balls[g]
            if a['observation_present']=='1' and z['observation_present']=='1':dr.line([(16+float(v['image_x_px'])*2/3,103+float(v['image_y_px'])*2/3) for v in [a,z]],fill=(70,245,155),width=3)
        if balls[f]['observation_present']=='1':
            x=16+float(balls[f]['image_x_px'])*2/3;y=103+float(balls[f]['image_y_px'])*2/3;dr.ellipse((x-7,y-7,x+7,y+7),outline=(100,255,130),width=3)
        else:txt(30,778,'球位置：観測不能' if balls[f]['annotation_available']=='1' else '球位置：この時点は未注釈',20,(255,210,105))
        txt(1330,108,'4選手の配置（暫定）',30)
        txt(1330,157,'枠下端の投影。足元精度は未検証。',18)
        left,top,w,h=1470,213,250,535
        def cp(u,v):return left+w*u,top+h*(1-v)
        dr.rectangle((left,top,left+w,top+h),outline=(150,185,205),width=2)
        for y in [.231,.5,.769]:dr.line([cp(0,y),cp(1,y)],fill=(205,220,235) if y==.5 else (90,130,155),width=3 if y==.5 else 1)
        for x in [.125,.875]:dr.line([cp(x,0),cp(x,1)],fill=(90,130,155),width=1)
        dr.line([cp(.5,0),cp(.5,.231)],fill=(90,130,155));dr.line([cp(.5,.769),cp(.5,1)],fill=(90,130,155))
        txt(left+95,182,'奥側');txt(left+95,754,'手前側')
        observed=set()
        for r in people[f]:
            observed.add(r['player_slot']);x,y=cp(float(r['court_proxy_x']),float(r['court_proxy_y']))
            if 1330<x<1890 and 180<y<800:
                col=b.RGB[r['player_slot']];dr.ellipse((x-8,y-8,x+8,y+8),fill=col);txt(int(x+11),int(y-14),b.NAMES[r['player_slot']],20,col)
        missing=[b.NAMES[s] for s in b.SLOTS if s not in observed];txt(1330,804,'人物枠の欠測：'+('・'.join(missing) if missing else 'なし'),20)
        txt(1330,838,'球の3D位置・落点は表示していません。',18)
        txt(24,835,f'元フレーム {f} ｜ 開始候補末尾から {(f-115373)/b.FPS:+.2f}秒',20)
        active=next((e for e in strokes if e['first']<=f<=e['last']),None)
        txt(24,869,('打球候補 '+active['stroke_id']+'：'+b.NAMES[active['hitter_candidate']]+'（AI暫定）') if active else '打球区間・打球者はAIによる暫定所見です。',22,(250,215,100))
        for j,e in enumerate(strokes):
            x=24+j*376;actor=b.NAMES[e['hitter_candidate']];nxt=b.NAMES.get(e['next_hitter_candidate'],'未確認')
            color=b.RGB[e['hitter_candidate']];highlight=e['first']<=f<=(strokes[j+1]['first']-1 if j+1<len(strokes) else times[-1])
            dr.rounded_rectangle((x,916,x+358,1007),radius=7,fill=(29,49,66) if highlight else (18,31,43),outline=color if highlight else (50,70,85),width=2)
            txt(x+12,923,f'{j+1}  '+('サーブ' if j==0 else '返球')+f'候補：{actor}',22,color)
            txt(x+12,958,f'次打球候補：{nxt}',18)
            txt(x+12,983,f'F{e["first"]}～{e["last"]}',16)
        txt(24,1032,'バウンド候補：B01＝F115413～419 ／ B02＝F115569～575。C02は打球と扱わず不明のまま。',18)
        proc.stdin.write(np.asarray(img).tobytes());mapping.append(dict(display_frame=i,source_frame=f,display_time_sec=i/fps))
        if f in [115441,115521,115729]:img.save(OUT/f'stroke_sequence_F{f}.jpg',quality=95)
    proc.stdin.close();err=proc.stderr.read();assert proc.wait()==0,err;b.write(OUT/'stroke_video_source_map.csv',mapping)

def verify():
    initial=b.load(OUT/'protocol.json');assert all(b.sha(Path(n))==h for n,h in initial['protected_input_hashes'].items())
    previous=b.load(b.OUT/'publication_hashes.json');assert all(b.sha(b.OUT/n)==h for n,h in previous.items())
    assert b.sha(b.HERE/'review_044_stroke_events.py')==b.load(OUT/'fine_review_plan.json')['script_sha256']
    for r in b.load(OUT/'coarse_image_manifest.json'):assert b.sha(Path(r['image']))==r['sha256']
    for r in b.read(OUT/'fine_source_image_manifest.csv'):assert b.sha(Path(r['source_image']))==r['sha256']
    strokes=b.read(OUT/'AI_stroke_records.csv');snaps=b.read(OUT/'stroke_player_snapshot_proxies.csv')
    by=b.grouped(b.read(b.OUT/'player_court_proxies.csv'));orig={(int(r['source_frame']),r['player_slot']):r for vv in by.values() for r in vv}
    for r in snaps:
        q=orig.get((int(r['reference_source_frame']),r['player_slot']))
        assert r['observed']==str(int(q is not None))
        for k in ['image_x_px','image_y_px','court_proxy_x','court_proxy_y']:assert r[k]==(q[k] if q else '')
    assert len(strokes)==5 and len(snaps)==20 and sum(r['observed']=='0' for r in snaps)==1
    assert all(int(r['source_frame_first'])<=int(r['reference_source_frame'])<=int(r['source_frame_last']) for r in strokes)
    assert all(r['independent_event_GT']=='0' and r['contact_x']==r['contact_y']==r['landing_x']==r['landing_y']==r['ball_court_x']==r['ball_court_y']=='' for r in strokes)
    videos=[]
    for w in WINDOWS:
        path=OUT/(w['id']+'_raw_quarter_speed.mp4');cap=cv2.VideoCapture(str(path));n=round(cap.get(7));assert n==w['last']-w['first']+1;ok,im=cap.read();assert ok;cap.release();videos.append(dict(name=path.name,frames=n))
    cap=cv2.VideoCapture(str(OUT/'044_AI_stroke_sequence_half_speed.mp4'));n=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape==(1080,1920,3);n+=1
    cap.release();assert n==213
    b.dump(OUT/'verification.json',dict(prior_synchronized_publication_files_unchanged=len(previous),all_source_hashes_unchanged=True,
        AI_stroke_rows=5,reference_player_rows=20,observed_position_proxies=19,missing_position_proxies_retained=1,
        all_reference_positions_exact_prior_proxies=True,new_interpolated_positions=0,new_human_event_GT=0,
        no_ball_court_positions_or_landing_points_created=True,raw_evidence_videos=videos,overview_video_decoded_frames=n,
        independent_precision_recall_not_computed=True))
    shutil.copy2(Path(__file__),OUT/'executed_finalize.py');shutil.copy2(b.HERE/'review_044_stroke_events.py',OUT/'executed_evidence.py')
    b.dump(OUT/'publication_hashes.json',{str(n.relative_to(OUT)):b.sha(n) for n in sorted(OUT.rglob('*')) if n.is_file() and n.name!='publication_hashes.json'})
    print('Verified5AI-provisional stroke rows,19original proxies and1missing; no independent hit labels.',flush=True)

if __name__=='__main__':{'chronology':chronology,'build':build,'render':render,'verify':verify}[sys.argv[1]]()
