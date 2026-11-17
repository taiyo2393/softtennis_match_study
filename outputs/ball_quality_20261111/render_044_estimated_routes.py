"""Show estimates openly: observed image tracks, predicted gaps, schematic court routes."""
from pathlib import Path
import math,shutil,subprocess,sys
from collections import Counter
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import build_044_spatiotemporal_pilot as b
import estimate_044_far_bounces as e
from constrain_044_far_bounce_fit import OUT as FIT

OUT=e.OUT
GREEN=(70,245,155);YELLOW=(255,210,70);CYAN=(75,175,245);ORANGE=(255,175,65)

def dashed(draw,a,z,color,width=2,dash=8,gap=6):
    a=np.array(a,dtype=float);z=np.array(z,dtype=float);length=float(np.linalg.norm(z-a))
    if length<1e-6:return
    for t in np.arange(0,length,dash+gap):
        q=a+(z-a)*t/length;r=a+(z-a)*min(length,t+dash)/length;draw.line([tuple(q),tuple(r)],fill=color,width=width)

def diamond(draw,x,y,color,r=9,width=2):draw.line([(x,y-r),(x+r,y),(x,y+r),(x-r,y),(x,y-r)],fill=color,width=width)

def render():
    path={int(r['source_frame']):r for r in b.read(FIT/'observed_and_estimated_image_path.csv')}
    people=b.grouped(b.read(b.OUT/'player_court_proxies.csv'));times=b.load(b.OUT/'protocol.json')['display_source_frames']
    near=b.read(e.NEAR/'conditional_bounce_court_points.csv');far=b.load(FIT/'far_fit_models.json')
    route=b.read(FIT/'schematic_court_routes.csv');groups={s:[r for r in route if r['stroke_id']==s] for s in ['H01','H02','H03','H04']}
    strokes=b.read(e.STROKE/'AI_stroke_records.csv');fonts={s:ImageFont.truetype(str(b.FONT),s) for s in [17,19,21,24,28,34]}
    fps=b.FPS/4;mapping=[]
    proc=subprocess.Popen(['C:/ffmpeg/bin/ffmpeg.exe','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'044_estimated_bounces_and_routes.mp4')],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    shots=[115479,115575,115701,115729]
    for i,f in enumerate(times):
        raw=cv2.imread(str(b.FRAMES/f'{f-114635:06d}.jpg'));assert raw is not None
        image=Image.new('RGB',(1920,1080),(12,22,32));d=ImageDraw.Draw(image)
        def txt(x,y,s,size=21,col=(225,235,245)):d.text((x,y),s,font=fonts[size],fill=col)
        txt(24,12,'044：奥の推定バウンド点と球の軌跡',34)
        txt(25,61,'手前2点＝おおむね妥当と確認 ／ 奥2点＝欠測からの推定（低信頼）',21)
        image.paste(Image.fromarray(cv2.cvtColor(raw,cv2.COLOR_BGR2RGB)).resize((1280,720),Image.Resampling.LANCZOS),(16,105));d=ImageDraw.Draw(image)
        for r in people[f]:
            color=b.RGB[r['player_slot']];x1,y1,x2,y2=[float(r[k])*2/3 for k in ['x1_px','y1_px','x2_px','y2_px']]
            d.rectangle((16+x1,105+y1,16+x2,105+y2),outline=color,width=2);txt(int(16+x1),int(79+y1),b.NAMES[r['player_slot']],21,color)
        for g in range(max(times[0]+2,f-48),f+1,2):
            a,z=path[g-2],path[g]
            if a['display_position_kind']=='missing' or z['display_position_kind']=='missing':continue
            q=[(16+float(r['display_image_x_px'])*2/3,105+float(r['display_image_y_px'])*2/3) for r in [a,z]]
            if a['display_position_kind']==z['display_position_kind']=='observed_assisted':d.line(q,fill=GREEN,width=3)
            else:dashed(d,q[0],q[1],YELLOW,3,5,4)
        r=path[f]
        if r['display_position_kind']!='missing':
            x=16+float(r['display_image_x_px'])*2/3;y=105+float(r['display_image_y_px'])*2/3
            if r['display_position_kind']=='estimated':diamond(d,x,y,YELLOW,9,3);txt(32,779,'球位置：欠測からの推定',21,YELLOW)
            else:d.ellipse((x-7,y-7,x+7,y+7),outline=GREEN,width=3)
        else:txt(32,779,'球位置：未取得（打球付近の欠測は残す）',21,YELLOW)
        txt(1335,110,'選手配置と推定配球',28)
        txt(1335,155,'奥の2点：ベースライン付近の粗い推定',19,YELLOW)
        left,top,w,h=1455,235,250,540
        def cp(u,v):return left+w*u,top+h*(1-v)
        d.rectangle((left,top,left+w,top+h),outline=(145,180,200),width=2)
        for v in [.231,.5,.769]:d.line([cp(0,v),cp(1,v)],fill=(205,220,235) if v==.5 else (80,120,145),width=3 if v==.5 else 1)
        for u in [.125,.875]:d.line([cp(u,0),cp(u,1)],fill=(80,120,145))
        d.line([cp(.5,0),cp(.5,.231)],fill=(80,120,145));d.line([cp(.5,.769),cp(.5,1)],fill=(80,120,145))
        txt(left+95,190,'奥側');txt(left+95,780,'手前側')
        current=max((j for j,r in enumerate(strokes) if int(r['source_frame_first'])<=f),default=0)
        for j,s in enumerate(['H01','H02','H03','H04']):
            if j>current:continue
            points=[cp(float(r['court_x']),float(r['court_y'])) for r in groups[s]]
            color=CYAN if j==current else (65,100,125)
            for a,z in zip(points,points[1:]):dashed(d,a,z,color,2,7,5)
        for n in near:
            if f<int(n['source_frame']):continue
            x,y=cp(float(n['assumed_ground_court_x']),float(n['assumed_ground_court_y']));d.ellipse((x-9,y-9,x+9,y+9),outline=ORANGE,width=3);txt(int(x+12),int(y-14),n['candidate_id'],19,ORANGE)
        # Predicted points can be very close; separate their labels with leader lines.
        for j,n in enumerate(far):
            if f<n['source_frame']:continue
            x,y=cp(n['assumed_ground_court_x'],n['assumed_ground_court_y']);diamond(d,x,y,YELLOW,10,3)
            lx,ly=(1335,198) if j==0 else (1660,198)
            dashed(d,(lx+35,ly+25),(x,y),YELLOW,1,4,4);txt(lx,ly,'推定'+str(j+1),19,YELLOW)
        for r in people[f]:
            x,y=cp(float(r['court_proxy_x']),float(r['court_proxy_y']));color=b.RGB[r['player_slot']]
            if 1330<x<1890 and 180<y<828:d.ellipse((x-6,y-6,x+6,y+6),fill=color);txt(int(x+8),int(y-13),b.NAMES[r['player_slot']],19,color)
        txt(1335,824,'○ 手前候補   ◇ 奥のモデル推定',19,ORANGE)
        txt(1335,860,'位置精度は未検証。最後の落点は未取得。',17)
        txt(24,840,f'元フレーム {f} ｜ 半分の速度',21)
        txt(24,878,'緑実線＝整備済み観測 ／ 黄破線・◇＝欠測の補完推定',21)
        d.rounded_rectangle((24,938,1895,1048),radius=8,fill=(19,34,48),outline=(65,85,105))
        txt(43,950,'コートの青破線＝打球者の代理点 → バウンド候補 → 次の打球者を結ぶ配球模式線',24,CYAN)
        txt(43,997,'実際の3D軌道・高さ・真下の連続位置ではありません。元の観測と推定は別に保存しています。',21)
        proc.stdin.write(np.asarray(image).tobytes());mapping.append(dict(display_frame=i,source_frame=f,display_time_sec=i/fps))
        if f in shots:image.save(OUT/f'estimated_routes_F{f}.jpg',quality=95)
    proc.stdin.close();err=proc.stderr.read();assert proc.wait()==0,err;b.write(OUT/'video_source_frame_map.csv',mapping)
    # Two common-scale court panels make close far predictions legible without spreading their coordinates.
    image=Image.new('RGB',(1600,1080),(12,22,32));d=ImageDraw.Draw(image)
    def txt(x,y,s,size=21,col=(225,235,245)):d.text((x,y),s,font=fonts[size],fill=col)
    txt(25,14,'044：手前の候補点と、奥の予測点・配球模式線',34)
    txt(26,66,'奥の2点は低信頼。コート内を仮定した曲線モデルが、ベースラインの境界に達した結果です。',21,YELLOW)
    for panel,shotids in enumerate([['H01','H02'],['H03','H04']]):
        left,top,w,h=260+panel*800,175,280,607
        def cp(u,v):return left+w*u,top+h*(1-v)
        txt(left-60,124,'1〜2打目' if panel==0 else '3〜4打目',28)
        d.rectangle((left,top,left+w,top+h),outline=(145,180,200),width=2)
        for v in [.231,.5,.769]:d.line([cp(0,v),cp(1,v)],fill=(205,220,235) if v==.5 else (80,120,145),width=3 if v==.5 else 1)
        for u in [.125,.875]:d.line([cp(u,0),cp(u,1)],fill=(80,120,145))
        for j,s in enumerate(shotids):
            route=groups[s];q=[cp(float(r['court_x']),float(r['court_y'])) for r in route]
            for a,z in zip(q,q[1:]):dashed(d,a,z,CYAN if j==0 else (165,120,245),3,9,7)
        n=near[panel];x,y=cp(float(n['assumed_ground_court_x']),float(n['assumed_ground_court_y']));d.ellipse((x-10,y-10,x+10,y+10),outline=ORANGE,width=3);txt(int(x+15),int(y-15),n['candidate_id'],21,ORANGE)
        n=far[panel];x,y=cp(n['assumed_ground_court_x'],n['assumed_ground_court_y']);diamond(d,x,y,YELLOW,12,3);txt(int(x+20),int(y+6),'奥推定'+str(panel+1),21,YELLOW)
        txt(left-80,828,'○ 手前：おおむね妥当とユーザー確認',19,ORANGE)
        txt(left-80,865,'◇ 奥：欠測区間からのモデル予測',19,YELLOW)
    txt(26,940,'破線は配球の模式線。実際の連続軌道・打球点・着地点の精度を保証するものではありません。',21)
    txt(26,988,'映像上の球軌跡は別に、観測＝緑実線／欠測補完＝黄破線で表示しています。',21)
    image.save(OUT/'estimated_delivery_routes_overview.jpg',quality=95)

def verify():
    for directory in [OUT,FIT]:
        p=b.load(directory/'protocol.json');assert all(b.sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    old={e.NEAR:b.load(e.NEAR/'publication_hashes.json'),e.CONF:b.load(e.CONF/'publication_hashes.json'),e.STROKE:b.load(e.STROKE/'publication_hashes.json'),b.OUT:b.load(b.OUT/'publication_hashes.json')}
    for directory,hh in old.items():assert all(b.sha(directory/n)==h for n,h in hh.items())
    raw={int(r['source_frame']):r for r in b.read(b.OUT/'ball_image_observations.csv')};path=b.read(FIT/'observed_and_estimated_image_path.csv');kinds=Counter()
    for r in path:
        original=raw[int(r['source_frame'])];assert r['original_observation_present']==original['observation_present']
        assert r['original_image_x_px']==original['image_x_px'] and r['original_image_y_px']==original['image_y_px']
        if original['observation_present']=='1':assert r['display_image_x_px']==original['image_x_px'] and r['display_image_y_px']==original['image_y_px'] and r['estimate_source']==''
        assert r['ground_court_x']==r['ground_court_y']=='';kinds[r['display_position_kind']]+=1
    assert kinds=={'observed_assisted':164,'estimated':31,'missing':18}
    models=b.load(FIT/'far_fit_models.json');H=np.array(b.load(b.CAL)['image_to_court_homography'])
    for m in models:
        u,v=b.project(H,m['image_x']*2,m['image_y']*2);assert abs(u-m['assumed_ground_court_x'])<1e-10 and abs(v-m['assumed_ground_court_y'])<1e-10
        assert -.000001<=u<=1.000001 and .499999<=v<=1.000001 and m['court_boundary_active']
        cy=m['y_coefficients'];assert min(cy[1:])>=-1e-7
    # Check retained failed code record against its original frozen hash.
    failed=OUT/'court_constrained_trial';pr=b.load(failed/'protocol.json');key=str(b.HERE/'constrain_044_far_bounce_fit.py')
    assert b.sha(failed/'executed_source_failed.py')==pr['protected_input_hashes'][key]
    cap=cv2.VideoCapture(str(OUT/'044_estimated_bounces_and_routes.mp4'));n=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape==(1080,1920,3);n+=1
    cap.release();assert n==213
    assert [int(r['source_frame']) for r in b.read(OUT/'video_source_frame_map.csv')]==b.load(b.OUT/'protocol.json')['display_source_frames']
    b.dump(OUT/'verification.json',dict(original164observations_exactly_preserved=True,original37missing_ball_labels_unchanged=True,
        estimated_missing_image_positions=31,remaining_missing_positions=18,remaining_annotated_missing=6,unannotated_display_times=12,
        far_bounce_predictions=2,boundary_active_low_confidence=2,unconstrained_failed_prediction_retained=True,
        physical_3D_or_airborne_ground_track_not_created=True,all_prior_publication_files_unchanged={str(d):len(v) for d,v in old.items()},
        video_decoded_frames=n,no_estimates_promoted_to_observations_or_GT=True))
    for p in [Path(__file__),b.HERE/'estimate_044_far_bounces.py',b.HERE/'constrain_044_far_bounce_fit.py']:shutil.copy2(p,OUT/('executed_'+p.name))
    b.dump(OUT/'publication_hashes.json',{str(n.relative_to(OUT)):b.sha(n) for n in sorted(OUT.rglob('*')) if n.is_file() and n.name!='publication_hashes.json'})
    print('Verified164unchanged observations,31separate image estimates,2low-confidence far bounces,213video frames.',flush=True)

if __name__=='__main__':{'render':render,'verify':verify}[sys.argv[1]]()
