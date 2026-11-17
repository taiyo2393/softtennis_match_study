"""Expand the mini-court viewport while retaining every stored player coordinate."""
from pathlib import Path
import subprocess,shutil
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import build_044_spatiotemporal_pilot as b
from render_044_estimated_routes import dashed,diamond,YELLOW,CYAN,ORANGE

PARENT=b.HERE/'056_reviewed_consolidated'
OUT=b.HERE/'056_reviewed_wide_court'
XLIM=(-.25,1.25)
YLIM=(-.30,1.55)

def main():
    OUT.mkdir(exist_ok=False)
    protected=b.load(PARENT/'publication_hashes.json')
    assert all(b.sha(PARENT/name)==digest for name,digest in protected.items())
    people=b.grouped(b.read(PARENT/'player_court_proxies.csv'))
    strokes=b.read(PARENT/'stroke_order_confirmed.csv')
    bounces=b.read(PARENT/'bounce_court_points.csv')
    route_rows=b.read(PARENT/'schematic_court_routes.csv')
    routes={s['stroke_id']:[r for r in route_rows if r['stroke_id']==s['stroke_id']] for s in strokes}
    mapping=b.read(PARENT/'video_source_frame_map.csv')
    # Retain the previous court aspect ratio. Enlarge the view, not the coordinates.
    plot_top,plot_height,center_x=200.,630.,1620.
    height=plot_height/(YLIM[1]-YLIM[0]);width=height*250/540
    court_left=center_x-width/2
    court_top=plot_top+(YLIM[1]-1)*height
    def cp(u,v):return court_left+width*u,court_top+height*(1-v)
    view_left,view_top=cp(XLIM[0],YLIM[1]);view_right,view_bottom=cp(XLIM[1],YLIM[0])
    config=dict(x_normalized_limits=list(XLIM),y_normalized_limits=list(YLIM),
        court_bounds=[0,1,0,1],court_aspect_ratio_unchanged=250/540,
        old_minicourt_rectangle=[1455,228,250,540],new_court_rectangle=[court_left,court_top,width,height],
        new_view_rectangle=[view_left,view_top,view_right,view_bottom],
        coordinate_policy='No coordinate clamping, rescaling of stored data, extrapolation or added person positions. Only drawing transform changes.',
        parent_video_sha256=protected['056_reviewed_players_ball_and_court.mp4'])
    b.dump(OUT/'viewport_config.json',config)
    fonts={s:ImageFont.truetype(str(b.FONT),s) for s in [16,18,20,30]}
    cap=cv2.VideoCapture(str(PARENT/'056_reviewed_players_ball_and_court.mp4'))
    fps=cap.get(cv2.CAP_PROP_FPS)
    proc=subprocess.Popen(['C:/ffmpeg/bin/ffmpeg.exe','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
        '-s','1920x1080','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','17',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'056_reviewed_wide_court.mp4')],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    visible=[];count=0;n2_min=None
    for row in mapping:
        ok,frame=cap.read();assert ok
        f=int(row['source_frame']);rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        image=Image.fromarray(rgb);d=ImageDraw.Draw(image)
        # Touch only the right-hand panel, leaving the main image and stroke strip alone.
        d.rectangle((1320,100,1919,900),fill=(12,22,32))
        def txt(x,y,s,size=18,color=(225,235,245)):
            d.text((x,y),s,font=fonts[size],fill=color)
        txt(1335,109,'コート外まで含む選手配置',30)
        txt(1335,153,'白線＝コート　周囲の余白＝コート外',18)
        d.rectangle((view_left,view_top,view_right,view_bottom),fill=(16,31,43),outline=(45,70,85),width=1)
        x0,y1=cp(0,1);x1,y0=cp(1,0)
        d.rectangle((x0,y1,x1,y0),outline=(205,225,235),width=2)
        for v in [.231,.5,.769]:d.line([cp(0,v),cp(1,v)],fill=(205,220,235) if v==.5 else (80,120,145),width=2 if v==.5 else 1)
        for u in [.125,.875]:d.line([cp(u,0),cp(u,1)],fill=(80,120,145),width=1)
        d.line([cp(.5,0),cp(.5,.231)],fill=(80,120,145));d.line([cp(.5,.769),cp(.5,1)],fill=(80,120,145))
        txt(x1+14,y1-12,'奥ベースライン',16)
        txt(x1+14,y0-12,'手前ベースライン',16)
        current=max((j for j,r in enumerate(strokes) if int(r['source_frame_first'])<=f),default=0)
        sid=strokes[current]['stroke_id']
        for j in range(max(0,current-1),current+1):
            points=[cp(float(r['court_x']),float(r['court_y'])) for r in routes[strokes[j]['stroke_id']] if r['court_x']!='']
            for a,z in zip(points,points[1:]):
                ok,aa,zz=cv2.clipLine((round(view_left),round(view_top),round(view_right-view_left),round(view_bottom-view_top)),
                    tuple(round(v) for v in a),tuple(round(v) for v in z))
                if ok:dashed(d,aa,zz,CYAN if j==current else (65,100,125),2,6,4)
        for point in bounces:
            if f<int(point['source_frame']):continue
            x,y=cp(float(point['court_x']),float(point['court_y']));active=point['after_stroke']==sid
            if point['status'].startswith('AI_visible'):d.ellipse((x-5,y-5,x+5,y+5),outline=ORANGE,width=2 if active else 1)
            else:diamond(d,x,y,YELLOW,6,2 if active else 1)
            if active:txt(x+8,y-12,point['id'],16,ORANGE)
        for p in people[f]:
            u,v=float(p['court_proxy_x']),float(p['court_proxy_y']);x,y=cp(u,v)
            inside=view_left+7<x<view_right-7 and view_top+7<y<view_bottom-7
            assert inside,(f,p['player_slot'],u,v,x,y)
            color=b.RGB[p['player_slot']]
            d.ellipse((x-6,y-6,x+6,y+6),fill=color,outline=(12,22,32),width=1)
            txt(x+8,y-12,b.NAMES[p['player_slot']],18,color)
            visible.append(dict(source_frame=f,player_slot=p['player_slot'],court_proxy_x=p['court_proxy_x'],
                                court_proxy_y=p['court_proxy_y'],display_x_px=x,display_y_px=y,within_view=int(inside)))
            if p['player_slot']=='near_rear' and (n2_min is None or v<n2_min[0]):
                n2_min=(v,f);image.save(OUT/'N2_farthest_behind_baseline.jpg',quality=95)
        txt(1335,847,'人物点は枠下端の代理位置。座標は変更していません。',16)
        txt(1335,875,'配球の破線・バウンド候補は従来と同じ。',16)
        if not people[f]:txt(1335,815,'人物：既存入力の範囲外',16,YELLOW)
        array=np.asarray(image)
        assert np.array_equal(array[:,:1320],rgb[:,:1320])
        proc.stdin.write(array.tobytes());count+=1
        if f==154470:image.save(OUT/'wide_court_F154470.jpg',quality=95)
    ok,_=cap.read();assert not ok;cap.release()
    proc.stdin.close();err=proc.stderr.read();assert proc.wait()==0,err
    b.write(OUT/'all_displayed_player_points.csv',visible)
    shutil.copy2(PARENT/'video_source_frame_map.csv',OUT/'video_source_frame_map.csv')
    check=cv2.VideoCapture(str(OUT/'056_reviewed_wide_court.mp4'));decoded=0
    while True:
        ok,frame=check.read()
        if not ok:break
        assert frame.shape==(1080,1920,3)
        if int(mapping[decoded]['source_frame'])==n2_min[1]:cv2.imwrite(str(OUT.parent/'056_wide_court_decoded_check.jpg'),frame)
        decoded+=1
    check.release();assert decoded==count==401
    assert all(b.sha(PARENT/name)==digest for name,digest in protected.items())
    expected=sum(len(rows) for rows in people.values())
    assert len(visible)==expected
    b.dump(OUT/'verification.json',dict(
        parent_publication_files_unchanged=len(protected),encoded_frames_decoded=decoded,
        stored_player_and_ball_coordinates_unchanged=True,all_available_player_points_visible=len(visible),
        all_available_N2_points_visible=sum(r['player_slot']=='near_rear' for r in visible),
        N2_minimum_court_y=n2_min[0],N2_farthest_source_frame=n2_min[1],
        left_panel_unchanged_before_video_encoding=True,court_aspect_ratio_preserved=True,
        labels_or_observations_added=0,clamped_or_invented_positions=0))
    (OUT/'README.md').write_text(
        '# 056：右側コート図の表示範囲を拡張\n\n'
        '正規化コートの表示範囲をx=-0.25〜1.25、y=-0.30〜1.55へ広げた。'
        '白線がコート、周囲の余白がコート外。N2が手前のベースライン後方へ下がった場面を含め、'
        '入力に存在する全選手点が図内に入ることを確認した。\n\n'
        'コートの縦横比、保存座標、球位置、選手追跡、配球模式線、注釈を変更していない。'
        '元動画の左側と下部はエンコード前の画素一致を確認。動画の再圧縮による画素差は生じうる。'
        '人物入力がない終盤には新しい点を補っていない。\n',encoding='utf-8')
    shutil.copy2(__file__,OUT/'executed_render.py')
    b.dump(OUT/'publication_hashes.json',{p.name:b.sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='publication_hashes.json'})
    print(b.load(OUT/'verification.json'),flush=True)

if __name__=='__main__':main()
