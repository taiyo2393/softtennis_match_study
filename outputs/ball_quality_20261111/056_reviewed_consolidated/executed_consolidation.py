"""Publish current reviewed 056 overlay without altering any prior experiment."""
from pathlib import Path
import subprocess,shutil
from collections import Counter
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import build_044_spatiotemporal_pilot as b
from prepare_056_route_transfer import FRAMES,OFFSET,TIMES
from render_044_estimated_routes import dashed,diamond,GREEN,YELLOW,CYAN,ORANGE

OUT=b.HERE/'056_reviewed_consolidated'
REVIEW=b.HERE/'056_crop_reviewed'
PRIOR=b.HERE/'056_reviewed_exclusion_F154780'
EVENT=b.HERE/'056_route_transfer'
SEQ=b.HERE/'056_sequence_confirmed'
M2=b.HERE.parent/'ball_quality_20261110'

def main():
    OUT.mkdir(exist_ok=False)
    protected={}
    for folder in [REVIEW,PRIOR,SEQ]:
        for name,digest in b.load(folder/'publication_hashes.json').items():
            assert b.sha(folder/name)==digest;protected[str(folder/name)]=digest
    for p in [EVENT/'AI_event_proposals.json',EVENT/'bounce_fit_models.json',Path(__file__)]:protected[str(p)]=b.sha(p)
    b.dump(OUT/'protocol.json',dict(
        role='Consolidated reviewed demonstration; not raw automated tracking or independent accuracy evaluation.',
        protected_input_hashes=protected,source_clock='legacy_edited_match1',display_source_frames=TIMES,
        display_sampling='Uniform stride2; encoded atFPS/4, hence half-speed. All404reviewed observation times retained separately, including end odd frames not displayed here.',
        precedence='Current reviewed observation > old estimate > missing. No new interpolation. The confirmed ball at154780 replaces only the current display; original rejected coordinates remain in source history.',
        line_rule='Join only adjacent stride2 points in one unchanged flight phase, without crossing AI contact/bounce intervals or missing points.',
        court='Old player bbox-bottom proxies and six bounce candidates/estimates unchanged. Court dashed routes are schematic; no airborne ball projected by ground homography.',
    ))
    observations=b.read(REVIEW/'056_reviewed_ball_observations.csv')
    obs={int(r['source_frame']):r for r in observations}
    old={int(r['source_frame']):r for r in b.read(PRIOR/'observed_and_estimated_image_path.csv')}
    phase={int(r['source_frame']):r['phase'] for r in b.read(b.HERE/'056_post_hit_trial/event_segmented_model_path.csv')}
    paths=[]
    for f in TIMES:
        original=old[f];o=obs.get(f)
        row=dict(source_frame=f,kind='missing',display_image_x_px='',display_image_y_px='',
                 source='not_annotated_at_this_time' if o is None else 'annotated_unlocated_not_absence',
                 annotation_available=int(o is not None),phase=phase[f],exact_center_validated=0)
        if o and o['current_observation_present']=='1':
            row.update(kind='reviewed_observation',display_image_x_px=o['current_x_px'],display_image_y_px=o['current_y_px'],
                       source=o['current_coordinate_source'])
        elif original['kind']=='estimated':
            row.update(kind='estimated',display_image_x_px=original['display_image_x_px'],display_image_y_px=original['display_image_y_px'],
                       source=original['estimate_source'])
        paths.append(row)
    path={r['source_frame']:r for r in paths}
    b.write(OUT/'display_ball_path.csv',paths)
    shutil.copy2(REVIEW/'056_reviewed_ball_observations.csv',OUT/'reviewed_observations_all404times.csv')
    missing=[]
    for r in observations:
        if r['current_observation_present']=='1':continue
        f=int(r['source_frame']);display=path.get(f)
        missing.append(dict(source_frame=f,phase=phase[f],observed_ball_position='unlocated_not_absence',
                            display_has_old_estimate=int(display is not None and display['kind']=='estimated')))
    b.write(OUT/'remaining_observation_gaps.csv',missing)
    people=b.grouped(b.read(PRIOR/'player_court_proxies.csv'))
    strokes=b.read(SEQ/'stroke_order_confirmed.csv')
    bounces=b.read(PRIOR/'bounce_court_points.csv');routes=b.grouped([])
    route_rows=b.read(PRIOR/'schematic_court_routes.csv')
    routes={r['stroke_id']:[q for q in route_rows if q['stroke_id']==r['stroke_id']] for r in strokes}
    for name,source in [('player_court_proxies.csv',PRIOR/'player_court_proxies.csv'),
                        ('stroke_order_confirmed.csv',SEQ/'stroke_order_confirmed.csv'),
                        ('bounce_court_points.csv',PRIOR/'bounce_court_points.csv'),
                        ('schematic_court_routes.csv',PRIOR/'schematic_court_routes.csv')]:shutil.copy2(source,OUT/name)
    fonts={s:ImageFont.truetype(str(b.FONT),s) for s in [16,18,20,22,25,30,34]}
    fps=b.FPS/4
    proc=subprocess.Popen(['C:/ffmpeg/bin/ffmpeg.exe','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1920x1080','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'056_reviewed_players_ball_and_court.mp4')],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    mapping=[];connections=[];image_manifest=[]
    for i,f in enumerate(TIMES):
        source=FRAMES/f'{f-OFFSET:06d}.jpg'
        image_manifest.append(dict(source_frame=f,path=str(source),sha256=b.sha(source)))
        raw=Image.open(source).convert('RGB');image=Image.new('RGB',(1920,1080),(12,22,32));d=ImageDraw.Draw(image)
        def txt(x,y,msg,size=20,color=(225,235,245)):d.text((x,y),msg,font=fonts[size],fill=color)
        txt(24,12,'056：確認済みの修正をまとめた選手・球・配球表示',34)
        txt(24,60,'13打球の順序は確認済み ｜ 接触時刻・選手位置・バウンド点は暫定 ｜ 半分の速度',22)
        image.paste(raw.resize((1280,720),Image.Resampling.LANCZOS),(16,104));d=ImageDraw.Draw(image)
        for person in people[f]:
            color=b.RGB[person['player_slot']]
            x1,y1,x2,y2=[float(person[k])*2/3 for k in ['x1_px','y1_px','x2_px','y2_px']]
            d.rectangle((16+x1,104+y1,16+x2,104+y2),outline=color,width=2)
            txt(int(16+x1),int(78+y1),b.NAMES[person['player_slot']],20,color)
        for g in range(max(TIMES[0]+2,f-36),f+1,2):
            a,z=path[g-2],path[g]
            if a['kind']=='missing' or z['kind']=='missing':continue
            if not(phase[g].startswith('flight_') and phase[g-2]==phase[g-1]==phase[g]):continue
            points=[(16+float(r['display_image_x_px'])*2/3,104+float(r['display_image_y_px'])*2/3) for r in [a,z]]
            if a['kind']==z['kind']=='reviewed_observation':d.line(points,fill=GREEN,width=3)
            else:dashed(d,points[0],points[1],YELLOW,3,5,4)
            if g==f:connections.append(dict(first=g-2,last=g,phase=phase[g]))
        r=path[f]
        if r['kind']!='missing':
            x=16+float(r['display_image_x_px'])*2/3;y=104+float(r['display_image_y_px'])*2/3
            if r['kind']=='estimated':diamond(d,x,y,YELLOW,9,3)
            else:d.ellipse((x-8,y-8,x+8,y+8),outline=GREEN,width=3)
        status={'reviewed_observation':'球位置：人手整備・実球確認を反映','estimated':'球位置：補間・モデル推定（未確認）',
                'missing':'球位置：未取得' if r['annotation_available'] else '球位置：この時点は注釈なし'}[r['kind']]
        txt(32,780,status,20,YELLOW if r['kind']!='reviewed_observation' else GREEN)
        current=max((j for j,s in enumerate(strokes) if int(s['source_frame_first'])<=f),default=0)
        sid=strokes[current]['stroke_id'];left,top,w,h=1455,228,250,540
        def cp(u,v):return left+w*u,top+h*(1-v)
        txt(1335,109,'選手配置と配球模式線',30)
        txt(1335,157,'○ バウンド候補　◇ 予測点（未確認）',18,ORANGE)
        d.rectangle((left,top,left+w,top+h),outline=(145,180,200),width=2)
        for v in [.231,.5,.769]:d.line([cp(0,v),cp(1,v)],fill=(205,220,235) if v==.5 else (80,120,145),width=3 if v==.5 else 1)
        for u in [.125,.875]:d.line([cp(u,0),cp(u,1)],fill=(80,120,145))
        d.line([cp(.5,0),cp(.5,.231)],fill=(80,120,145));d.line([cp(.5,.769),cp(.5,1)],fill=(80,120,145))
        txt(left+95,186,'奥側');txt(left+95,776,'手前側')
        for j in range(max(0,current-1),current+1):
            points=[cp(float(v['court_x']),float(v['court_y'])) for v in routes[strokes[j]['stroke_id']] if v['court_x']!='']
            for a,z in zip(points,points[1:]):
                ok,aa,zz=cv2.clipLine((1330,180,575,625),tuple(round(v) for v in a),tuple(round(v) for v in z))
                if ok:dashed(d,aa,zz,CYAN if j==current else (65,100,125),2,7,5)
        for point in bounces:
            if f<int(point['source_frame']):continue
            x,y=cp(float(point['court_x']),float(point['court_y']));active=point['after_stroke']==sid
            if point['status'].startswith('AI_visible'):d.ellipse((x-8,y-8,x+8,y+8),outline=ORANGE,width=3 if active else 1)
            else:diamond(d,x,y,YELLOW,9,3 if active else 1)
            if active:txt(int(x+12),int(y-14),point['id'],18,ORANGE)
        for person in people[f]:
            x,y=cp(float(person['court_proxy_x']),float(person['court_proxy_y']));color=b.RGB[person['player_slot']]
            if 1320<x<1905 and 180<y<804:d.ellipse((x-6,y-6,x+6,y+6),fill=color);txt(int(x+8),int(y-13),b.NAMES[person['player_slot']],18,color)
        txt(1335,811,'人物：既存入力の範囲外' if f<154086 or f>154855 else '人物点は枠下端の代理位置',18,YELLOW)
        txt(1335,847,'破線は模式線。実際の3D軌道ではない。',16)
        txt(24,837,f'元フレーム {f} ｜ 表示中の打球 {current+1}/13：'+b.NAMES[strokes[current]['hitter_candidate']],22)
        footer='緑○＝人手整備・実球確認　黄◇・黄破線＝推定　打球・欠測をまたぐ線は切っています。'
        if abs(f-154324)<=6:footer='F154324：誤検出の青候補は除外。黄◇は従来の補間推定で、実球として未確認。'
        if abs(f-154780)<=6:footer='F154780：別物体の旧候補を除外し、別位置で実球と確認された点を表示。'
        txt(24,874,footer,20)
        start=max(0,min(len(strokes)-5,current-2))
        for k,s in enumerate(strokes[start:start+5]):
            j=start+k;x=24+k*376;color=b.RGB[s['hitter_candidate']]
            d.rounded_rectangle((x,921,x+358,1006),radius=7,fill=(29,49,66) if j==current else (18,31,43),outline=color if j==current else (50,70,85),width=2)
            txt(x+12,929,f'{j+1}：'+b.NAMES[s['hitter_candidate']]+'（順序確認済み）',22,color)
            txt(x+12,970,f'暫定 F{s["source_frame_first"]}〜{s["source_frame_last"]}',18)
        txt(24,1030,'球の高さ・速度・最終2バウンドの位置は測定していません。',20)
        proc.stdin.write(np.asarray(image).tobytes());mapping.append(dict(display_frame=i,source_frame=f,display_time_sec=i/fps))
        if f in [154270,154324,154780,154858]:image.save(OUT/f'consolidated_F{f}.jpg',quality=95)
    proc.stdin.close();err=proc.stderr.read();assert proc.wait()==0,err
    b.write(OUT/'video_source_frame_map.csv',mapping);b.write(OUT/'trace_connections.csv',connections)
    b.dump(OUT/'source_image_manifest.json',image_manifest)
    cap=cv2.VideoCapture(str(OUT/'056_reviewed_players_ball_and_court.mp4'));decoded=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape==(1080,1920,3)
        if decoded==350:cv2.imwrite(str(OUT.parent/'056_consolidated_decoded_check.jpg'),im)
        decoded+=1
    cap.release();assert decoded==len(TIMES)==401
    assert all(b.sha(Path(p))==digest for p,digest in protected.items())
    assert path[154270]['source']==path[154780]['source']=='crop_heatmap_candidate_human_confirmed_ball_enclosure'
    assert path[154324]['kind']=='estimated' and float(path[154324]['display_image_x_px'])!=910.4375
    assert len(missing)==50
    counts=dict(Counter(r['kind'] for r in paths))
    b.dump(OUT/'verification.json',dict(encoded_and_decoded_frames=decoded,display_counts=counts,
        retained_reviewed_observation_times=len(observations),observed_position_times=354,unlocated_annotation_times=len(missing),
        crop_positive_reviews_applied=[154270,154780],crop_negative_candidate_excluded=154324,
        all_protected_inputs_unchanged=True,person_bounce_and_schematic_tables_unchanged=True,
        new_interpolated_positions=0,new_3D_ball_positions=0,
        no_trace_connections_across_contact_or_bounce_intervals=True,independent_accuracy_established=False))
    registration=b.load(M2/'match2_external_validation/source_registration.json');video=Path(registration['source_video'])
    status=dict(source_video=str(video),source_file_present=video.is_file(),
        registered_bytes=registration['bytes'],current_bytes=video.stat().st_size if video.exists() else None,
        registration_path=str(M2/'match2_external_validation/source_registration.json'),registration_sha256=b.sha(M2/'match2_external_validation/source_registration.json'),
        known_previous_work='Camera calibration and person-tracking development including205..209s, fixed300s/900s and1200s/1800s/2400s windows.',
        fully_unseen_match=False,ball_method_transfer_result_not_found_in_examined_match2_publications=True,
        next_required_work='Fix rally serve/end boundaries from raw video before ball analysis; freeze baseline and any assisted repair procedure before viewing new ball outputs.',
        evaluation='Call it separate-match ball transfer, not a fully untouched-match final test. Report repair effort, missed visible balls and wrong-object candidates separately; do not convert person candidate availability to ID accuracy.',
        inspected_documents=[str(M2/name/'README.md') for name in ['match2_external_validation','match2_camera_setup','match2_camera_transfer_pilot','match2_unseen_crop_transfer']])
    b.dump(OUT/'match2_evaluation_readiness.json',status)
    (OUT/'README.md').write_text(
        '# 056：本人確認を反映した統合版\n\n'
        '13打球の打球者順、終盤15候補、時系列再探索4候補、局所画像A・Cの採用とBの棄却を統合した。'
        '緑は既存の人手整備と実球確認を反映する位置。正確な中心・全時点の追跡正解率を示すものではない。\n\n'
        '全404注釈時点はreviewed_observations_all404times.csvに保存し、実球位置あり354時点、未取得50時点。'
        '元376時点は2フレーム間隔、追加の終盤28時点は1フレーム間隔で、連続追跡率として計算しない。'
        '残る未取得はremaining_observation_gaps.csvに一覧化した。\n\n'
        '動画は従来と同じ401時点を2フレーム間隔・半速で表示。終盤の奇数フレームも全注釈表に保持するが、この動画には表示しない。'
        '確認済み位置を優先し、従来の補間点は黄のまま保存。新しい補間は行っていない。'
        'F154324の黄の推定点は、棄却した青の検出候補とは別の位置であり、実球確認済みにはしていない。\n\n'
        '選手位置・バウンド点・コート模式線は旧データをそのまま利用し、球の空中位置を平面へ投影していない。'
        '接触時刻、足元位置、バウンド、3D軌道、速度は追加の確認を受けていない。\n\n'
        'match2の保存と既存の人物追跡試走を確認した。既にカメラと人物追跡の開発に使っており、完全未見の試合とは扱わない。'
        '球の別試合移行試験には、先にプレー区間と比較条件を固定する。準備状況はmatch2_evaluation_readiness.json。\n',encoding='utf-8')
    shutil.copy2(__file__,OUT/'executed_consolidation.py')
    b.dump(OUT/'publication_hashes.json',{p.name:b.sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='publication_hashes.json'})
    print(b.load(OUT/'verification.json'));print('match2 source present:',status['source_file_present'],flush=True)

if __name__=='__main__':main()
