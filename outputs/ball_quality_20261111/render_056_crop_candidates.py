"""Review two gap proposals and a distinct new candidate on the rejected-object frame."""
from pathlib import Path
import subprocess,shutil
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import build_044_spatiotemporal_pilot as b
from prepare_056_route_transfer import FRAMES,OFFSET
from trial_056_crop_reacquisition import OUT,BASE

BLUE=(70,190,255);GREEN=(80,230,150);RED=(255,90,90)

def main():
    assert not (OUT/'056_crop_three_candidates.mp4').exists()
    rows=b.read(OUT/'candidate_rows.csv')
    targets=[r for r in rows if r['kind']=='crop_heatmap_candidate_unconfirmed' and
             r['role'] in ['previously_unresolved_gap','known_other_object_coordinate_diagnostic']]
    assert [int(r['source_frame']) for r in targets]==[154270,154324,154780]
    observations={int(r['source_frame']):r for r in b.read(BASE/'completed_review/assisted_states_sampled.csv')}
    fonts={s:ImageFont.truetype(str(b.FONT),s) for s in [20,23,30]}
    fps=b.FPS/4
    proc=subprocess.Popen(['C:/ffmpeg/bin/ffmpeg.exe','-y','-loglevel','error','-f','rawvideo',
        '-pix_fmt','rgb24','-s','1920x720','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-crf','18',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'056_crop_three_candidates.mp4')],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    mapping=[];sources={};sheet=Image.new('RGB',(1920,400),(12,22,32))
    for i,r in enumerate(targets):
        target=int(r['source_frame']);letter=chr(65+i)
        x0=max(0,min(1408,round(float(r['guide_x'])-256)))
        y0=max(0,min(792,round(float(r['guide_y'])-144)))
        for f in range(target-8,target+9):
            source=FRAMES/f'{f-OFFSET:06d}.jpg';sources[f]=dict(source_frame=f,path=str(source),sha256=b.sha(source))
            raw=Image.open(source).convert('RGB');frame=Image.new('RGB',(1920,720),(12,22,32));d=ImageDraw.Draw(frame)
            d.text((20,10),f'056：局所画像から出た候補 {letter} ／ 3（青◇は未確認）',font=fonts[30],fill='white')
            d.text((20,56),f'F{f} ｜ 対象 F{target} で停止 ｜ 左：全体　右：固定範囲の拡大',font=fonts[20],fill='white')
            frame.paste(raw.resize((960,540)),(0,100));crop=raw.crop((x0,y0,x0+512,y0+288))
            frame.paste(crop.resize((960,540)),(960,100));d=ImageDraw.Draw(frame)
            point=None;is_target=f==target
            if is_target:point=(float(r['candidate_x_px']),float(r['candidate_y_px']))
            else:
                old=observations.get(f)
                if f!=154780 and old and old['observation_present']=='1':point=(float(old['x_px']),float(old['y_px']))
            if point:
                x,y=point
                for xx,yy in [(x/2,100+y/2),(960+(x-x0)*1.875,100+(y-y0)*1.875)]:
                    if xx>=960 and not(960<=xx<1920 and 100<=yy<640):continue
                    if is_target:d.line([(xx,yy-11),(xx+11,yy),(xx,yy+11),(xx-11,yy),(xx,yy-11)],fill=BLUE,width=3)
                    else:d.ellipse((xx-7,yy-7,xx+7,yy+7),outline=GREEN,width=3)
            if is_target and target==154780:
                for xx,yy in [(1102/2,100+579/2),(960+(1102-x0)*1.875,100+(579-y0)*1.875)]:
                    d.line((xx-9,yy-9,xx+9,yy+9),fill=RED,width=3);d.line((xx-9,yy+9,xx+9,yy-9),fill=RED,width=3)
            d.text((20,651),'青◇：今回の新候補　緑○：既存の人手整備位置',font=fonts[23],fill='white')
            footer='Cの赤×は以前の誤検出。今回確認するのは別位置の青◇だけ。' if target==154780 else '空白は位置未取得。青◇が実球を囲めているかだけを確認します。'
            d.text((20,686),footer,font=fonts[20],fill=RED if target==154780 else (230,240,245))
            if is_target:
                frame.save(OUT/f'review_{letter}_F{f}.jpg',quality=95)
                tile=crop.resize((640,360));td=ImageDraw.Draw(tile)
                xx=(point[0]-x0)*1.25;yy=(point[1]-y0)*1.25
                td.ellipse((xx-10,yy-10,xx+10,yy+10),outline=BLUE,width=3)
                sheet.paste(tile,(i*640,40));ImageDraw.Draw(sheet).text((i*640+12,4),f'{letter} : F{f}',font=fonts[23],fill=BLUE)
            for repeat in range(16 if is_target else 1):
                proc.stdin.write(np.asarray(frame).tobytes())
                mapping.append(dict(display_frame=len(mapping),source_frame=f,review_target=target,letter=letter,
                                    target_hold_repeat=repeat if is_target else '',crop_x0=x0,crop_y0=y0))
    proc.stdin.close();err=proc.stderr.read();assert proc.wait()==0,err
    sheet.save(OUT/'three_crop_candidates.jpg',quality=95)
    b.write(OUT/'video_source_frame_map.csv',mapping)
    b.dump(OUT/'review_source_images.json',list(sources.values()))
    b.dump(OUT/'diagnostic_candidate_additional_review.json',dict(
        primary_missing_target_review_frames=[154270,154324],additional_diagnostic_review_frame=154780,
        reason='A spatially distinct new output on a frame containing a previously rejected object needs its own identity check. The prior rejection is not a full-frame negative label.',
        new_candidate_native_xy=[936.875,529.875],prior_rejected_native_xy=[1102,579],
        no_adoption_before_review=True))
    cap=cv2.VideoCapture(str(OUT/'056_crop_three_candidates.mp4'));count=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape==(720,1920,3)
        if count==10:cv2.imwrite(str(OUT.parent/'056_crop_decoded_check.jpg'),im)
        count+=1
    cap.release();assert count==len(mapping)==96
    protected=b.load(OUT/'protocol.json')['protected_input_hashes']
    assert all(b.sha(Path(p))==digest for p,digest in protected.items())
    controls=[r for r in rows if r['role']=='previously_ball_enclosing_candidate']
    distances=[float(r['comparison_distance_px']) for r in controls if r['comparison_distance_px']!='']
    b.dump(OUT/'verification.json',dict(encoded_frames_decoded=count,source_aspect_preserved=True,
        uniformly_enlarged_review_crop=[512,288,960,540],protected_inputs_unchanged=True,
        prior_confirmed_control_times=len(controls),controls_with_output=sum(r['kind']=='crop_heatmap_candidate_unconfirmed' for r in controls),
        missing_control_frames=[int(r['source_frame']) for r in controls if r['kind'].startswith('missing')],
        max_displacement_from_prior_candidate_px=max(distances),displacement_is_not_accuracy=True,
        new_gap_candidates=2,new_diagnostic_candidate=1,new_human_confirmations=0,review_pending=True))
    (OUT/'README.md').write_text(
        '# 056：局所画像入力による再取得比較\n\n'
        '1920×1080の既存キャッシュから960×540を切り出し、512×288へ入力した。'
        '全体画像入力に比べて球の画像上の大きさは2倍になるが、周囲の文脈も変わるため、解像度だけの効果とは断定しない。'
        '重みと成分閾値0.1、探索半径32pxは固定し、追加学習や条件探索は行っていない。\n\n'
        '欠測10時点中、F154270と154324の2時点に未確認候補が出た。'
        'F154780にも以前の誤検出とは別位置の候補が出ているため、診断対象として別途確認する。'
        '以前の赤丸が誤りという回答から、新しい位置の正しさは判断できない。\n\n'
        '確認済み19時点を入力すると18時点に候補が出たが、F154853は未取得となった。'
        '18点とも実球という新しい確認はまだなく、座標差は検出精度ではない。'
        '既存出力の全面置換は行わず、欠測時の補助手段として検討する。\n\n'
        '探索位置には人手整備済みの前後位置を使う。完全自動・独立試験ではない。'
        '原画像・注釈・旧出力を保持し、新候補の本人確認前には確定データへ追加しない。\n',encoding='utf-8')
    shutil.copy2(__file__,OUT/'executed_render.py')
    b.dump(OUT/'publication_hashes.json',{p.name:b.sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='publication_hashes.json'})
    print(b.load(OUT/'verification.json'))

if __name__=='__main__':main()
