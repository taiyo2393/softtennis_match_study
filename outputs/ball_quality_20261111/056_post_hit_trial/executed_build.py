"""Expose unchanged late model outputs; break traces at provisional events/gaps."""
from pathlib import Path
import subprocess, shutil
from collections import Counter
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import build_044_spatiotemporal_pilot as b
from prepare_056_route_transfer import FRAMES, OFFSET

OUT=b.HERE/'056_post_hit_trial'
BASE=b.PRIOR/'rally_repair_056'
REVIEW=b.HERE/'056_reviewed_exclusion_F154780'
SEQ=b.HERE/'056_sequence_confirmed'
FIRST,LAST=154810,154880
ORANGE=(255,190,65)

def phase(f, strokes, turns):
    for r in strokes:
        if int(r['source_frame_first'])<=f<=int(r['source_frame_last']):
            return 'contact_uncertain_'+r['stroke_id']
    for name,lo,hi in turns:
        if lo<=f<=hi:return 'bounce_uncertain_'+name
    earlier=[r for r in strokes if int(r['source_frame_last'])<f]
    sid=earlier[-1]['stroke_id'] if earlier else 'before_first'
    # The count partitions both sides of bounce uncertainty brackets.
    return 'flight_'+sid+'_'+str(sum(f>hi for _,lo,hi in turns))

def main():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/'protocol.json').exists(), 'Preserve completed trial.'
    protected={}
    for folder in [REVIEW, SEQ]:
        for name,digest in b.load(folder/'publication_hashes.json').items():
            assert b.sha(folder/name)==digest
            protected[str(folder/name)]=digest
    for p in [BASE/'baseline_all_frames.csv',BASE/'map_records.csv',
              b.HERE/'056_route_transfer/AI_event_proposals.json',Path(__file__)]:
        protected[str(p)]=b.sha(p)
    b.dump(OUT/'protocol.json',dict(
        date_jst='2026-11-11', purpose='Extend display beyond human ball annotations and split model traces at impacts/bounces.',
        protected_input_hashes=protected,
        model='Reuse original fixed threshold0.3 model outputs at every cached source frame. No new training, threshold or crop inference.',
        human_ball_annotation_last_source_frame=154838,
        video_source_range=[FIRST,LAST], playback_speed=0.25,
        event_timing='Existing AI intervals, not newly confirmed exact impact/bounce frames; hitter count/order human confirmed.',
        trace_rule='Only adjacent source frames with output, identical flight phase, at most10sourceframes trail. No line in contact/bounce intervals; no interpolation or extrapolation.',
        evaluation='Availability and trace-integrity checks only. No new accuracy or recovered-ball claims.',
    ))
    strokes=b.read(SEQ/'stroke_order_confirmed.csv')
    proposal=b.load(b.HERE/'056_route_transfer/AI_event_proposals.json')
    turns=[(r['id'],r['first'],r['last']) for r in proposal['visible_bounce_candidates']+proposal['predicted_bounce_gaps']]
    raw=b.read(BASE/'baseline_all_frames.csv'); rows=[]
    for r in raw:
        f=int(r['source_frame']); p=phase(f,strokes,turns)
        reject=(f==154780 and r['visible']=='1' and float(r['x_px'])==1102 and float(r['y_px'])==579)
        rows.append(dict(r,phase=p,review_status='model_candidate_unreviewed',
                         outside_existing_human_annotation_range=int(f<154088 or f>154838),
                         usable_candidate=int(r['visible']=='1' and not reject),
                         scoped_rejection_applied=int(reject),
                         connect_to_previous=0))
    connections=[]
    for a,z in zip(rows,rows[1:]):
        if (a['usable_candidate']==z['usable_candidate']==1 and
            a['phase']==z['phase'] and z['phase'].startswith('flight_') and
            int(z['source_frame'])-int(a['source_frame'])==1):
            z['connect_to_previous']=1
            connections.append(dict(first=int(a['source_frame']),last=int(z['source_frame']),phase=z['phase']))
    b.write(OUT/'event_segmented_model_path.csv',rows)
    b.write(OUT/'trace_connections.csv',connections)
    tail=[r for r in rows if int(r['source_frame'])>154838]
    b.write(OUT/'unreviewed_tail_candidates.csv',tail)
    b.dump(OUT/'source_image_manifest.json',[
        dict(source_frame=f,path=str(FRAMES/f'{f-OFFSET:06d}.jpg'),sha256=b.sha(FRAMES/f'{f-OFFSET:06d}.jpg'))
        for f in range(FIRST,LAST+1)])
    by={int(r['source_frame']):r for r in rows}
    fonts={s:ImageFont.truetype(str(b.FONT),s) for s in [20,23,30]}
    fps=b.FPS/4
    proc=subprocess.Popen(['C:/ffmpeg/bin/ffmpeg.exe','-y','-loglevel','error',
        '-f','rawvideo','-pix_fmt','rgb24','-s','1920x720','-r',str(fps),'-i','pipe:0',
        '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',
        str(OUT/'056_last_two_hits_model_candidates.mp4')],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    mapping=[]
    for f in range(FIRST,LAST+1):
        rgb=np.asarray(Image.open(FRAMES/f'{f-OFFSET:06d}.jpg').convert('RGB'))
        image=Image.new('RGB',(1920,720),(12,22,32));d=ImageDraw.Draw(image)
        def text(x,y,msg,size=23,color=(235,240,245)):
            d.text((x,y),msg,font=fonts[size],fill=color)
        text(20,10,'056：最後の2打球とその後 — 元モデルの未確認候補',30)
        text(20,56,f'F{f} ｜ 4分の1速 ｜ 左：全体　右：固定範囲の拡大（縦横比を維持）',20)
        image.paste(Image.fromarray(rgb).resize((960,540)),(0,100))
        image.paste(Image.fromarray(rgb[380:830,550:1350]).resize((960,540)),(960,100));d=ImageDraw.Draw(image)
        def pt(row,crop):
            x,y=float(row['x_px']),float(row['y_px'])
            return (960+(x-550)*1.2,100+(y-380)*1.2) if crop else (x/2,100+y/2)
        for g in range(max(FIRST+1,f-10),f+1):
            a,z=by[g-1],by[g]
            if z['connect_to_previous']!=1:continue
            for crop in [False,True]:
                p,q=pt(a,crop),pt(z,crop)
                if crop and not all(960<=v[0]<1920 and 100<=v[1]<640 for v in [p,q]):continue
                d.line([p,q],fill=ORANGE,width=2)
        r=by[f]
        if r['usable_candidate']:
            for crop in [False,True]:
                x,y=pt(r,crop)
                if crop and not(960<=x<1920 and 100<=y<640):continue
                d.ellipse((x-8,y-8,x+8,y+8),outline=ORANGE,width=3)
        text(20,651,'橙丸：未確認の球候補' if r['usable_candidate'] else '元モデルの球候補なし',23,ORANGE)
        text(630,651,'接触候補区間：線を切る' if r['phase'].startswith('contact_') else '打球・欠測をまたぐ線は描かない',23)
        text(20,685,'13打球の順序は確認済み。接触時刻・橙丸の正しさ・終盤のバウンド位置は未確認。',20)
        proc.stdin.write(np.asarray(image).tobytes())
        mapping.append(dict(display_frame=len(mapping),source_frame=f,display_time_sec=len(mapping)/fps))
        if f in [154850,154858,154870]:image.save(OUT/f'candidate_F{f}.jpg',quality=95)
    proc.stdin.close();err=proc.stderr.read();assert proc.wait()==0,err
    b.write(OUT/'video_source_frame_map.csv',mapping)
    cap=cv2.VideoCapture(str(OUT/'056_last_two_hits_model_candidates.mp4'));decoded=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape==(720,1920,3);decoded+=1
    cap.release();assert decoded==LAST-FIRST+1
    for name,digest in protected.items():assert b.sha(Path(name))==digest
    assert all(all(r[k]==s[k] for k in s) for r,s in zip(rows,raw))
    assert all(by[c['first']]['phase']==by[c['last']]['phase'] and c['phase'].startswith('flight_') for c in connections)
    b.dump(OUT/'verification.json',dict(
        source_model_rows_preserved=len(raw),raw_model_fields_exact=True,
        old_publications_unchanged=True,source_frames_decoded=decoded,
        tail_source_frames=len(tail),tail_model_candidate_frames=sum(r['usable_candidate'] for r in tail),
        new_human_labels=0,new_interpolated_positions=0,new_3D_or_ground_ball_positions=0,
        connections_across_contact_or_bounce_intervals=0,
        note='All counts measure output availability. Late final bounce positions remain unconfirmed.'))
    (OUT/'README.md').write_text(
        '# 056：打球区間で線を切り、終盤の元モデル候補を追加\n\n'
        '元モデルはF154880まで推論済み。人手整備の終了F154838以降を未確認候補として提示する。'
        '再学習・閾値変更・補間は行っていない。\n\n'
        'event_segmented_model_path.csv は803時点の元出力を全列保持し、既存の接触・バウンド暫定区間で'
        '線を分けた派生データ。接触区間内、欠測、異なる飛行区間をまたぐ接続は禁止する。'
        '13打球順は人の確認済みだが、時刻はAI暫定値のまま。\n\n'
        '動画の橙丸は全て元モデルの未確認候補。既存の人手修正位置を元モデルの出力と混ぜていない。'
        'F154838以降の出現数は精度・回復数ではない。終盤は確認後に採否を決める。\n',encoding='utf-8')
    shutil.copy2(__file__,OUT/'executed_build.py')
    b.dump(OUT/'publication_hashes.json',{p.name:b.sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='publication_hashes.json'})
    print(b.load(OUT/'verification.json'))

if __name__=='__main__':main()
