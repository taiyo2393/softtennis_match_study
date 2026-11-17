"""Render the two model candidates that need a human identity check."""
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import build_044_spatiotemporal_pilot as common

EVALUATION = common.HERE / 'match2_first_three_points_evaluated'
OUT = common.HERE / 'match2_visibility_disagreement_review'
CASES = {
    12393: ('A', 'match2_ball_first_point', (950, 400, 1350, 625)),
    13896: ('B', 'match2_points_2_3/point_02', (1080, 0, 1592, 288)),
}


def render(row):
    target = int(row['source_frame'])
    letter, source, crop = CASES[target]
    x, y = float(row['candidate_x_px']), float(row['candidate_y_px'])
    left, top, right, bottom = crop
    scale = 960 / (right - left)
    assert (bottom - top) * scale == 540
    assert left < x < right and top < y < bottom
    frames = common.HERE / source / 'frames'
    name = f'{letter}_F{target}'
    video = OUT / f'{name}.mp4'
    heading = ImageFont.truetype(str(common.FONT), 26)
    detail = ImageFont.truetype(str(common.FONT), 20)
    command = ['C:/ffmpeg/bin/ffmpeg.exe', '-hide_banner', '-loglevel', 'error', '-n',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1920x680', '-r', '15000/1001',
               '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-crf', '17', '-pix_fmt', 'yuv420p',
               '-movflags', '+faststart', str(video)]
    mapping = []
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        for frame in range(target-14, target+15):
            with Image.open(frames / f'{frame}.jpg') as image:
                assert image.size == (1920, 1080)
                canvas = Image.new('RGB', (1920, 680), '#101820')
                canvas.paste(image.resize((960, 540), Image.Resampling.LANCZOS), (0, 86))
                canvas.paste(image.crop(crop).resize((960, 540), Image.Resampling.LANCZOS), (960, 86))
            draw = ImageDraw.Draw(canvas)
            draw.text((20, 9), f'{letter}　match2 第{row["point"]}点　対象 F{target}', font=heading, fill='white')
            draw.text((20, 48), f'元フレーム F{frame}　{frame/common.FPS:.3f}秒　4分の1速', font=detail, fill='#d3e0e9')
            footer = '停止中の青◇が確認する候補です。' if frame == target else '左は全体映像、右は同じ箇所の拡大です。'
            draw.text((20, 640), footer, font=detail, fill='white')
            if frame == target:
                for cx, cy in [(x/2, 86+y/2), (960+(x-left)*scale, 86+(y-top)*scale)]:
                    outline = [(cx, cy-14), (cx+14, cy), (cx, cy+14), (cx-14, cy), (cx, cy-14)]
                    draw.line(outline, fill='#101820', width=6)
                    draw.line(outline, fill='#48d3ff', width=3)
                canvas.save(OUT / f'{name}_still.jpg', quality=95)
            for repeat in range(26 if frame == target else 1):
                process.stdin.write(np.asarray(canvas).tobytes())
                mapping.append(dict(display_frame=len(mapping), source_frame=frame, hold_repeat=repeat))
        process.stdin.close()
        assert process.wait() == 0
    common.write(OUT / f'{name}_frame_map.csv', mapping)
    capture = cv2.VideoCapture(str(video))
    decoded = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            assert image.shape == (680, 1920, 3)
            decoded += 1
    finally:
        capture.release()
    assert decoded == len(mapping) == 54
    return dict(letter=letter, point=int(row['point']), source_frame=target,
                candidate_xy_px=[x, y], original_reference=row['truth_visibility'],
                video=video.name, video_sha256=common.sha(video), frames_decoded=decoded)


def main():
    comparison = EVALUATION / 'reference_comparison_all_24_times.csv'
    digest = common.sha(comparison)
    assert digest == common.load(EVALUATION / 'publication_hashes.json')[comparison.name]
    rows = [row for row in common.read(comparison) if row['result'] == 'output_when_not_observable']
    assert {int(row['source_frame']) for row in rows} == set(CASES)
    OUT.mkdir(exist_ok=False)
    cases = [render(row) for row in rows]
    common.dump(OUT / 'review_cases.json', dict(cases=cases, reference_comparison_sha256=digest,
        review_scope='Model-visible candidate identity check following the original raw-image reference review.'))
    print(cases, flush=True)


if __name__ == '__main__':
    main()
