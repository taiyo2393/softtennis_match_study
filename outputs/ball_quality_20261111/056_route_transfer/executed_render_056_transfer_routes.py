"""Complete fixed-scope056 transfer media and provenance; no human accuracy claims."""
from pathlib import Path
from collections import Counter
import shutil,subprocess,sys
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import build_044_spatiotemporal_pilot as b
from prepare_056_route_transfer import OUT,FRAMES,OFFSET,TIMES,PERSON,BALL
from render_044_estimated_routes import dashed,diamond,GREEN,YELLOW,CYAN,ORANGE

def render():
    path={int(r['source_frame']):r for r in b.read(OUT/'observed_and_estimated_image_path.csv')};people=b.grouped(b.read(OUT/'player_court_proxies.csv'))
    strokes=b.read(OUT/'AI_stroke_records.csv');bounces=b.read(OUT/'bounce_court_points.csv');route=b.read(OUT/'schematic_court_routes.csv')
    routes={s['stroke_id']:[r for r in route if r['stroke_id']==s['stroke_id']] for s in strokes};land={r['after_stroke']:r for r in bounces}
    fonts={s:ImageFont.truetype(str(b.FONT),s) for s in [16,18,20,22,25,30,34]};fps=b.FPS/4;mapping=[]
    proc=subprocess.Popen(['C:/ffmpeg/bin/ffmpeg.exe','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'056_transfer_bounces_and_routes.mp4')],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    for i,f in enumerate(TIMES):
        raw=cv2.imread(str(FRAMES/f'{f-OFFSET:06d}.jpg'));assert raw is not None
        img=Image.new('RGB',(1920,1080),(12,22,32));d=ImageDraw.Draw(img)
        def txt(x,y,s,size=20,c=(225,235,245)):d.text((x,y),s,font=fonts[size],fill=c)
        txt(24,12,'056：同じ方法で選手配置・配球・球軌跡を表示',34)
        txt(25,60,'13打球はAI候補・未確認 ｜ 球は既存の人手整備データ ｜ 半分の速度',22)
        img.paste(Image.fromarray(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)).resize((1280,720),Image.Resampling.LANCZOS),(16,104));d=ImageDraw.Draw(img)
        for r in people[f]:
            color=b.RGB[r['player_slot']];x1,y1,x2,y2=[float(r[k])*2/3 for k in ['x1_px','y1_px','x2_px','y2_px']]
            d.rectangle((16+x1,104+y1,16+x2,104+y2),outline=color,width=2);txt(int(16+x1),int(77+y1),b.NAMES[r['player_slot']],20,color)
        for g in range(max(TIMES[0]+2,f-48),f+1,2):
            a,z=path[g-2],path[g]
            if a['kind'] in ['missing','observed_warning'] or z['kind'] in ['missing','observed_warning']:continue
            q=[(16+float(r['display_image_x_px'])*2/3,104+float(r['display_image_y_px'])*2/3) for r in [a,z]]
            if a['kind']==z['kind']=='observed_assisted':d.line(q,fill=GREEN,width=3)
            else:dashed(d,q[0],q[1],YELLOW,3,5,4)
        z=path[f]
        if z['kind']!='missing':
            x=16+float(z['display_image_x_px'])*2/3;y=104+float(z['display_image_y_px'])*2/3
            if z['kind']=='estimated':diamond(d,x,y,YELLOW,9,3);txt(32,778,'球位置：欠測からの推定',20,YELLOW)
            else:d.ellipse((x-8,y-8,x+8,y+8),outline=(255,90,90) if z['kind']=='observed_warning' else GREEN,width=3)
        else:txt(32,778,'球位置：未取得' if 154088<=f<=154838 else '球位置：この時点は注釈範囲外',20,YELLOW)
        current=max((j for j,r in enumerate(strokes) if int(r['source_frame_first'])<=f),default=0);sid=strokes[current]['stroke_id']
        txt(1335,109,'選手配置と配球模式線',30)
        txt(1335,157,'○ 観測由来の候補   ◇ 欠測からの推定',18,ORANGE)
        left,top,w,h=1455,228,250,540
        def cp(u,v):return left+w*u,top+h*(1-v)
        d.rectangle((left,top,left+w,top+h),outline=(145,180,200),width=2)
        for v in [.231,.5,.769]:d.line([cp(0,v),cp(1,v)],fill=(205,220,235) if v==.5 else (80,120,145),width=3 if v==.5 else 1)
        for u in [.125,.875]:d.line([cp(u,0),cp(u,1)],fill=(80,120,145))
        d.line([cp(.5,0),cp(.5,.231)],fill=(80,120,145));d.line([cp(.5,.769),cp(.5,1)],fill=(80,120,145))
        txt(left+95,186,'奥側');txt(left+95,776,'手前側')
        for j in range(max(0,current-1),current+1):
            rs=routes[strokes[j]['stroke_id']];q=[cp(float(r['court_x']),float(r['court_y'])) for r in rs if r['court_x']!='' and r['court_y']!='']
            for a,z in zip(q,q[1:]):
                ok,aa,zz=cv2.clipLine((1330,180,575,625),tuple(round(v) for v in a),tuple(round(v) for v in z))
                if ok:dashed(d,aa,zz,CYAN if j==current else (65,100,125),2,7,5)
        for point in bounces:
            if f<int(point['source_frame']):continue
            x,y=cp(float(point['court_x']),float(point['court_y']));is_current=point['after_stroke']==sid
            if point['status'].startswith('AI_visible'):d.ellipse((x-8,y-8,x+8,y+8),outline=ORANGE,width=3 if is_current else 1)
            else:diamond(d,x,y,YELLOW,9,3 if is_current else 1)
            if is_current:txt(int(x+12),int(y-14),point['id'],18,ORANGE if point['id'].startswith('B') else YELLOW)
        outside=[]
        for r in people[f]:
            x,y=cp(float(r['court_proxy_x']),float(r['court_proxy_y']));color=b.RGB[r['player_slot']]
            if 1320<x<1905 and 180<y<804:d.ellipse((x-6,y-6,x+6,y+6),fill=color);txt(int(x+8),int(y-13),b.NAMES[r['player_slot']],18,color)
            else:outside.append(b.NAMES[r['player_slot']])
        if f<154086 or f>154855:txt(1335,812,'人物位置：既存入力の範囲外',18,YELLOW)
        else:
            missing=[b.NAMES[s] for s in b.SLOTS if s not in {r['player_slot'] for r in people[f]}]
            txt(1335,812,'欠測：'+('・'.join(missing) if missing else 'なし')+(' ／ 図外：'+'・'.join(outside) if outside else ''),18)
        txt(1335,847,'点は暫定。破線は実3D軌道ではありません。',16)
        txt(24,837,f'元フレーム {f} ｜ 打球候補 {current+1}/13：'+b.NAMES[strokes[current]['hitter_candidate']],22)
        if abs(f-154780)<=8:txt(24,874,'警告：F154780の球位置に単独の飛び。元データを残し、線は結ばない。',20,(255,110,105))
        else:txt(24,874,'緑＝整備済み観測 ／ 黄破線＝補完推定 ／ 赤丸＝位置の警告',20)
        start=max(0,min(len(strokes)-5,current-2))
        for k,r in enumerate(strokes[start:start+5]):
            j=start+k;x=24+k*376;color=b.RGB[r['hitter_candidate']]
            d.rounded_rectangle((x,921,x+358,1006),radius=7,fill=(29,49,66) if j==current else (18,31,43),outline=color if j==current else (50,70,85),width=2)
            txt(x+12,929,f'{j+1}：'+b.NAMES[r['hitter_candidate']]+'（AI候補）',22,color)
            txt(x+12,970,f'F{r["source_frame_first"]}〜{r["source_frame_last"]}',18)
        txt(24,1030,'ボレー候補の区間にバウンドを作っていません。最終2バウンドの位置は未取得。',20)
        proc.stdin.write(np.asarray(img).tobytes());mapping.append(dict(display_frame=i,source_frame=f,display_time_sec=i/fps))
        if f in [154230,154470,154780,154846]:img.save(OUT/f'transfer_F{f}.jpg',quality=95)
    proc.stdin.close();err=proc.stderr.read();assert proc.wait()==0,err;b.write(OUT/'video_source_frame_map.csv',mapping)

def verify():
    protocol=b.load(OUT/'protocol.json');assert all(b.sha(Path(n))==h for n,h in protocol['protected_input_hashes'].items())
    fit=b.load(OUT/'fit_application_protocol.json');assert all(b.sha(Path(n))==h for n,h in fit['source_hashes'].items())
    prior=b.HERE/'044_estimated_ball_routes';hh=b.load(prior/'publication_hashes.json');assert all(b.sha(prior/n)==h for n,h in hh.items())
    for r in b.load(OUT/'source_image_manifest.json'):assert b.sha(Path(r['path']))==r['sha256']
    source_ball={int(r['source_frame']):r for r in b.read(BALL/'assisted_states_sampled.csv')};observed=b.read(OUT/'ball_image_observations.csv');paths={int(r['source_frame']):r for r in b.read(OUT/'observed_and_estimated_image_path.csv')}
    for r in observed:
        f=int(r['source_frame']);a=source_ball.get(f);z=paths[f]
        assert r['observation_present']==(a['observation_present'] if a else '0')
        for field,source in [('image_x_px','x_px'),('image_y_px','y_px')]:assert r[field]==(a[source] if a and a['observation_present']=='1' else '')
        assert z['original_observation_present']==r['observation_present'] and z['original_image_x_px']==r['image_x_px'] and z['original_image_y_px']==r['image_y_px']
        if r['observation_present']=='1':assert z['display_image_x_px']==r['image_x_px'] and z['display_image_y_px']==r['image_y_px'] and z['estimate_source']==''
    native={(int(r['source_frame']),r['player_slot']):r for r in b.read(PERSON)};people=b.read(OUT/'player_court_proxies.csv');H=np.array(b.load(b.CAL)['image_to_court_homography'])
    for r in people:
        source=native[int(r['source_frame']),r['player_slot']];assert source['track_id']==r['track_id']
        assert abs(float(source['bottom_center_x'])/2-float(r['image_x_px']))<1e-9 and abs(float(source['bottom_center_y'])/2-float(r['image_y_px']))<1e-9
        u,v=b.project(H,float(source['bottom_center_x']),float(source['bottom_center_y']));assert abs(u-float(r['court_proxy_x']))<1e-10 and abs(v-float(r['court_proxy_y']))<1e-10
    models=b.load(OUT/'bounce_fit_models.json');assert len(models)==3 and sum(m['boundary_active'] for m in models)==0
    cap=cv2.VideoCapture(str(OUT/'056_transfer_bounces_and_routes.mp4'));n=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape==(1080,1920,3);n+=1
    cap.release();assert n==401
    assert [int(r['source_frame']) for r in b.read(OUT/'video_source_frame_map.csv')]==TIMES
    result=b.load(OUT/'result.json');b.dump(OUT/'verification.json',dict(prior044publication_files_unchanged=len(hh),protected_inputs_and401source_images_unchanged=True,
        original334ball_positions_and42missing_labels_exact=True,original_person_observations_preserved=len(people),
        estimated_image_positions=result['path_counts']['estimated'],suspicious_assisted_point_preserved=154780,
        new_human_event_or_bounce_labels=0,new_exact_impact_or_landing_GT=0,full_video_frames_decoded=n,
        same_frozen_curve_solver_and_numeric_parameters=True,independent_accuracy_or_transfer_success_not_established=True))
    for name in ['prepare_056_route_transfer.py','review_056_fine_events.py','build_056_transfer_routes.py','render_056_transfer_routes.py']:
        shutil.copy2(b.HERE/name,OUT/('executed_'+name))
    b.dump(OUT/'publication_hashes.json',{str(n.relative_to(OUT)):b.sha(n) for n in sorted(OUT.rglob('*')) if n.is_file() and n.name!='publication_hashes.json'})
    print('Verified056 transfer:334unchanged ball points,21separate estimates,3predicted bounces,401video frames;13stroke proposals remain AI-only.',flush=True)

if __name__=='__main__':{'render':render,'verify':verify}[sys.argv[1]]()
