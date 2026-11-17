"""Publish the six-gap review and the single new match2 candidate."""
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import build_044_spatiotemporal_pilot as common
from audit_match2_first_point_gaps import BASE, OUT


def main():
    result = common.read(OUT/'short_gap_comparison.csv')
    candidates = [row for row in result if row['candidate_x_px']]
    assert len(candidates)==1
    candidate = candidates[0]
    target = int(candidate['source_frame'])
    assert target==12389 and candidate['method']=='local_crop'
    x,y = float(candidate['candidate_x_px']),float(candidate['candidate_y_px'])
    notes = [
        ('12388–12389','球が奥選手の右側へ進む短い区間。F12389に球らしい像が残る。','比較対象。'),
        ('12420–12442','冒頭に黄色い像が残る。その後は手前選手の体と重なり、スイングに入る。','長さと打球動作のため今回の比較対象外。'),
        ('12444','手前選手が振り抜く場面。球とラケットを切り分けにくい。','両端の球同定が要確認。'),
        ('12446–12455','手前選手のラケット、ネット、奥選手が重なる。','両端の球同定が要確認。'),
        ('12461–12465','ネット付近で奥選手がラケットを出す。','飛行区間の両端が要確認。'),
        ('12470–12472','奥選手のラケット動作の直後。','飛行区間の両端が要確認。'),
    ]
    common.write(OUT/'visual_gap_notes.csv',[dict(source_frames=frames,AI_raw_image_observation=note,
        eligibility=decision,human_visibility_labels_added=0) for frames,note,decision in notes])
    font = ImageFont.truetype(str(common.FONT),24)
    small = ImageFont.truetype(str(common.FONT),20)
    video = OUT/'F12389_local_candidate_review.mp4'
    box=(970,380,1290,560)
    command=['C:/ffmpeg/bin/ffmpeg.exe','-hide_banner','-loglevel','error','-n',
        '-f','rawvideo','-pix_fmt','rgb24','-s','1920x660','-r','15000/1001','-i','pipe:0',
        '-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    mapping=[]
    with subprocess.Popen(command,stdin=subprocess.PIPE) as proc:
        for frame in range(12382,12398):
            raw=Image.open(BASE/'frames'/f'{frame}.jpg').convert('RGB')
            image=Image.new('RGB',(1920,660),'#101820')
            image.paste(raw.resize((960,540),Image.Resampling.LANCZOS),(0,78))
            image.paste(raw.crop(box).resize((960,540),Image.Resampling.LANCZOS),(960,78))
            draw=ImageDraw.Draw(image)
            draw.text((20,9),'match2 第1点　局所画像で出た球候補を確認',font=font,fill='white')
            draw.text((20,45),f'F{frame}　4分の1速　対象F12389で一時停止',font=small,fill='white')
            draw.text((20,628),'左は全体、右は拡大。青いひし形が今回の確認対象です。',font=small,fill='white')
            if frame==target:
                for cx,cy in [(x/2,78+y/2),(960+(x-box[0])*3,78+(y-box[1])*3)]:
                    draw.line([(cx,cy-13),(cx+13,cy),(cx,cy+13),(cx-13,cy),(cx,cy-13)],fill='#42bfff',width=3)
                image.save(OUT/'F12389_review.jpg',quality=95)
            for repeat in range(24 if frame==target else 1):
                proc.stdin.write(np.asarray(image).tobytes())
                mapping.append(dict(display_frame=len(mapping),source_frame=frame,hold_repeat=repeat))
        proc.stdin.close()
        assert proc.wait()==0
    common.write(OUT/'review_frame_map.csv',mapping)
    cap=cv2.VideoCapture(str(video))
    decoded=0
    while True:
        ok,frame=cap.read()
        if not ok:
            break
        assert frame.shape==(660,1920,3)
        decoded+=1
    cap.release()
    assert decoded==len(mapping)==39
    common.dump(OUT/'verification.json',dict(
        audited_unavailable_frames=44,gap_runs=6,short_gap_test_frames=2,
        full_frame_new_candidates=0,local_new_candidates=[target],new_confirmed_positions=0,
        baseline_batch_matches_frozen_output=True,review_video_decoded_frames=decoded))
    (OUT/'README.md').write_text(
        '# match2 第1点の未出力区間\n\n'
        '44フレームは6つの区間に分かれた。全時点の元画像を周辺と一緒に点検した。'
        '最長の23フレームでは手前選手の体との重なりと打球動作が見える。'
        '各時点の球の可視性は人手参照で確定していない。\n\n'
        '短い欠測F12388〜12389を比較した。両端の元モデル候補を原画像で点検して探索の基準にした。'
        '両端の本人確認は行っていない。056と同じ閾値0.1、探索半径32pxを使用した。'
        '全体画像は0点、局所画像はF12389の1点。F12388は未取得。'
        '新候補は動画で球同定を確認する。\n',encoding='utf-8')
    common.dump(OUT/'publication_hashes.json',{p.name:common.sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='publication_hashes.json'})
    print('Audited six gaps, rendered 39 frames for the F12389 candidate.',flush=True)


if __name__=='__main__':
    main()
