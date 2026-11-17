"""Extract independent review material without displaying tracker candidates."""
from fractions import Fraction
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import build_044_spatiotemporal_pilot as common
import build_match2_player_ball_review as pilot


OUT = common.HERE / 'match2_raw_stroke_review'
BASELINE = common.HERE / 'match2_first_three_player_ball_v3'
FFMPEG = 'C:/ffmpeg/bin/ffmpeg.exe'
FPS = Fraction(60000, 1001)
WINDOWS = {1: (12312, 12577), 2: (13621, 14086), 3: (15201, 15522)}
DETAIL = (15475, 15522)


def encode(source, point, first, last, rate, detail=False):
    output = OUT / ('point3_contact_raw_detail.mp4' if detail else f'point{point}_raw.mp4')
    capture = cv2.VideoCapture(source)
    assert capture.isOpened() and capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    assert abs(capture.get(cv2.CAP_PROP_FPS) - float(FPS)) < .001
    width, height = 1440, 860
    result_fps = FPS * rate
    audio_tempo = 'atempo=0.5,atempo=0.5' if rate == Fraction(1, 4) else 'atempo=0.5'
    audio_length = Fraction(last - first + 1, 1) / FPS
    command = [
        FFMPEG, '-hide_banner', '-loglevel', 'error', '-y',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', str(result_fps),
        '-i', 'pipe:0', '-ss', f'{first / float(FPS):.9f}', '-t', f'{float(audio_length):.9f}',
        '-i', source, '-map', '0:v:0', '-map', '1:a:0', '-af', audio_tempo + ',apad',
        '-c:v', 'libx264', '-preset', 'fast', '-crf', '17',
        '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '128k', '-shortest',
        '-movflags', '+faststart', str(output),
    ]
    font = ImageFont.truetype(str(common.FONT), 25)
    small = ImageFont.truetype(str(common.FONT), 19)
    with subprocess.Popen(command, stdin=subprocess.PIPE) as proc:
        try:
            for f in range(first, last + 1):
                ok, frame = capture.read()
                assert ok and frame.shape[:2] == (1080, 1920)
                assert round(capture.get(cv2.CAP_PROP_POS_FRAMES)) == f + 1
                canvas = Image.new('RGB', (width, height), (13, 21, 31))
                if detail:
                    left = cv2.resize(frame, (896, 504), interpolation=cv2.INTER_AREA)
                    crop = cv2.resize(frame[535:850, 470:1030], (504, 284), interpolation=cv2.INTER_CUBIC)
                    canvas.paste(Image.fromarray(cv2.cvtColor(left, cv2.COLOR_BGR2RGB)), (12, 152))
                    canvas.paste(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)), (924, 262))
                else:
                    scaled = cv2.resize(frame, (width, 810), interpolation=cv2.INTER_AREA)
                    canvas.paste(Image.fromarray(cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)), (0, 50))
                draw = ImageDraw.Draw(canvas)
                title = f'match2 第{point}点　' + ('終盤の拡大・0.25倍速' if detail else '原映像・0.5倍速')
                draw.text((16, 7), title, font=font, fill='white')
                draw.text((1165, 10), f'元フレーム F{f}', font=small, fill='white')
                if detail:
                    draw.text((12, 118), '全体', font=small, fill='white')
                    draw.text((924, 228), '手前・赤黒ウェアの選手の周辺', font=small, fill='white')
                proc.stdin.write(np.asarray(canvas).tobytes())
        finally:
            proc.stdin.close()
            capture.release()
        assert proc.wait() == 0
    cap = cv2.VideoCapture(str(output))
    assert abs(cap.get(cv2.CAP_PROP_FPS) - float(result_fps)) < .001
    for _ in range(first, last + 1):
        ok, frame = cap.read()
        assert ok and frame.shape[:2] == (height, width)
    assert not cap.read()[0]
    cap.release()
    return dict(point=point, first_frame=first, last_frame=last, frames=last - first + 1,
                slowdown=float(rate), file=output.name, sha256=common.sha(output))


def main():
    OUT.mkdir(exist_ok=True)
    source = common.load(pilot.OUT / 'protocol.json')['source_video']
    frozen = {name: common.sha(BASELINE / name) for name in ('stroke_candidates.csv', 'schematic_routes.json')}
    clips = [encode(source, point, first, last, Fraction(1, 2))
             for point, (first, last) in WINDOWS.items()]
    clips.append(encode(source, 3, *DETAIL, Fraction(1, 4), detail=True))
    assert all(common.sha(BASELINE / name) == value for name, value in frozen.items())
    common.dump(OUT / 'review_manifest.json', dict(
        source_video=source, fps=str(FPS), clips=clips, frozen_candidate_artifacts=frozen,
        display='source video and retimed original audio; no candidate tracks or hit timing cues',
        question='Did the near red-and-black player make racket contact in the final exchange of point 3?',
        response_options=['contact', 'no contact', 'unclear'],
        reference_status='not collected; prior exposure to predictions means this is a follow-up review',
    ))
    print([(c['file'], c['frames']) for c in clips])


if __name__ == '__main__':
    main()
