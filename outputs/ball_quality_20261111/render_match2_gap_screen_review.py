"""Render the preregistered spot checks from the gap-screen comparison."""
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import build_044_spatiotemporal_pilot as common
from screen_match2_first_three_gaps import FIRST, LATER, OUT


def main():
    pairs = common.read(OUT/'paired_candidates.csv')
    cases = [
        next(r for r in pairs if r['outcome']=='crop_only' and int(r['source_frame']) != 12389),
        next(r for r in pairs if r['outcome']=='full_only' and int(r['source_frame']) != 15505),
        next(r for r in pairs if r['distance_px'] and float(r['distance_px']) > 16),
    ]
    assert [int(r['source_frame']) for r in cases] == [12388,13840,13973]
    video = OUT/'three_gap_candidates_review.mp4'
    font = ImageFont.truetype(str(common.FONT), 24)
    small = ImageFont.truetype(str(common.FONT), 20)
    command = ['C:/ffmpeg/bin/ffmpeg.exe','-hide_banner','-loglevel','error','-n',
        '-f','rawvideo','-pix_fmt','rgb24','-s','1920x680','-r','15000/1001','-i','pipe:0',
        '-an','-c:v','libx264','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    mapping, manifest = [], []
    with subprocess.Popen(command, stdin=subprocess.PIPE) as proc:
        for letter, case in zip('ABC', cases, strict=True):
            point, target = int(case['point']), int(case['source_frame'])
            folder = FIRST if point == 1 else LATER/f'point_{point:02}'
            markers = [(float(case[f'{prefix}_x_px']), float(case[f'{prefix}_y_px']), color)
                       for prefix,color in [('full','#42cbff'),('crop','#ffca57')] if case[f'{prefix}_x_px']]
            center_x = sum(x for x,_,_ in markers)/len(markers)
            center_y = sum(y for _,y,_ in markers)/len(markers)
            left, top = max(0,min(1536,round(center_x-192))), max(0,min(864,round(center_y-108)))
            box = (left,top,left+384,top+216)
            manifest.append(dict(letter=letter, **case, display_crop=list(box)))
            for frame in range(target-9,target+10):
                with Image.open(folder/'frames'/f'{frame}.jpg') as image:
                    canvas = Image.new('RGB',(1920,680),'#101820')
                    canvas.paste(image.resize((960,540),Image.Resampling.LANCZOS),(0,86))
                    canvas.paste(image.crop(box).resize((960,540),Image.Resampling.LANCZOS),(960,86))
                draw = ImageDraw.Draw(canvas)
                draw.text((18,10),f'{letter}　match2 第{point}点　対象 F{target}',font=font,fill='white')
                draw.text((18,49),f'元フレーム F{frame}　4分の1速　対象フレームで停止',font=small,fill='white')
                draw.text((18,640),'青◇は全体画像、黄◇は局所画像の候補です。左は全体、右は拡大です。',font=small,fill='white')
                if frame == target:
                    for x,y,color in markers:
                        for cx,cy in [(x/2,86+y/2),(960+(x-left)*2.5,86+(y-top)*2.5)]:
                            outline = [(cx,cy-12),(cx+12,cy),(cx,cy+12),(cx-12,cy),(cx,cy-12)]
                            draw.line(outline,fill='#101820',width=6)
                            draw.line(outline,fill=color,width=3)
                    canvas.save(OUT/f'{letter}_F{target}.jpg',quality=95)
                for repeat in range(20 if frame == target else 1):
                    proc.stdin.write(np.asarray(canvas).tobytes())
                    mapping.append(dict(display_frame=len(mapping),letter=letter,source_frame=frame,hold_repeat=repeat))
        proc.stdin.close()
        assert proc.wait()==0
    common.write(OUT/'review_frame_map.csv',mapping)
    cap = cv2.VideoCapture(str(video))
    decoded = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            assert frame.shape == (680,1920,3)
            decoded += 1
    finally:
        cap.release()
    assert decoded == len(mapping) == 114
    common.dump(OUT/'review_cases.json',dict(cases=manifest,video=video.name,video_sha256=common.sha(video),decoded_frames=decoded))
    summary = common.read(OUT/'point_summary.csv')
    report = ['# match2 第1〜3点の欠測再探索', '',
        '基準モデルの未出力178フレームを46区間に分けた。直前2フレームに基準候補がある33区間の初回未出力フレームで、'
        '1フレーム先の位置を手掛かりに全体画像と局所画像を比較した。閾値0.1と半径32pxを使い、範囲内の検出成分が1つなら候補を返した。', '',
        '| ラリー | 対象 | 両方 | 全体だけ | 局所だけ | 候補なし |', '|---|---:|---:|---:|---:|---:|']
    for row in summary:
        report.append(f'| 第{row["point"]}点 | {row["eligible_first_frames"]} | {row["both"]} | {row["full_only"]} | {row["crop_only"]} | {row["neither"]} |')
    report.extend(['','33時点中24時点に候補が出た。全体画像は17時点、局所画像は20時点。両方式の候補が16pxを超えて離れた例は3時点。',
        'F15505の全体画像候補は既存の人手参照から2.97px。新しい出力の実球同定は追確認待ち。',
        'F12389は欠測2フレーム目のため今回の対象外。確認動画の3例は、方式ごとの最初の単独候補と、最初の16px超の不一致を選んだ。', ''])
    (OUT/'README.md').write_text('\n'.join(report),encoding='utf-8')
    common.dump(OUT/'publication_hashes.json',{p.name:common.sha(p) for p in sorted(OUT.iterdir()) if p.is_file()})
    print([r['source_frame'] for r in manifest],flush=True)


if __name__ == '__main__':
    main()
