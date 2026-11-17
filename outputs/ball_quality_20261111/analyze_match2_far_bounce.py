"""Screen missing far bounces and fit the one supported lob provisionally."""
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import build_044_spatiotemporal_pilot as common
import build_match2_player_ball_review as pilot
from estimate_044_far_bounces import constrained_y, matrices


OUT = common.HERE / 'match2_far_bounce_review'
CACHE = common.HERE / 'match2_points_2_3/point_02/frames'
ASCENDING_ROI = (914, 445, 947, 481)
VISUAL_STREAK = (13980, 966, 538)


def read_frame(f):
    frame = cv2.imread(str(CACHE / f'{f}.jpg'))
    assert frame is not None and frame.shape[:2] == (1080, 1920)
    return frame


def after_bounce_candidates():
    a, b, c, d = ASCENDING_ROI
    previous = None
    positions = {}
    for f in range(13996, 14018):
        crop = read_frame(f)[b:d, a:c]
        _, _, stats, centres = cv2.connectedComponentsWithStats(
            (crop.max(axis=2) > 141).astype(np.uint8))
        points = [(float(x + a), float(y + b)) for (x, y), (_, _, w, h, area) in
                  zip(centres[1:], stats[1:]) if 9 <= area <= 16 and 2 <= w <= 8 and 2 <= h <= 8]
        if previous:
            px, py = previous
            points = [(x, y) for x, y in points if 0 <= px - x <= 3 and -3 <= y - py <= 1.5]
        assert len(points) == 1, (f, points)
        previous = positions[f] = points[0]
    return positions


def profile(before, after, homography):
    samples = {**before, **after}
    xs, ys = np.array(list(samples.values())).T
    results = []
    for f in range(13974, 13996):
        ax, ay = matrices(list(samples), f)
        x = np.linalg.lstsq(ax, xs, rcond=None)[0]
        y_loss, y = constrained_y(ay, ys)
        rmse = float(np.sqrt((y_loss + np.sum((ax @ x - xs) ** 2)) / len(samples)))
        results.append(dict(frame=f, x_px=float(x[0]), y_px=float(y[0]), rms_image_px=rmse,
                            court_xy=common.project(homography, 2 * x[0], 2 * y[0])))
    return results


def render():
    path = OUT / 'match2_point2_far_bounce_candidate.mp4'
    cmd = ['C:/ffmpeg/bin/ffmpeg.exe', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', '1440x640', '-r', '15000/1001', '-i', 'pipe:0', '-an', '-c:v', 'libx264',
           '-preset', 'fast', '-crf', '17', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
    font = ImageFont.truetype(str(common.FONT), 25)
    small = ImageFont.truetype(str(common.FONT), 18)
    f_review, x_review, y_review = VISUAL_STREAK
    count = 0
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        try:
            for f in range(13963, 14023):
                frame = read_frame(f)
                canvas = Image.new('RGB', (1440, 640), (13, 23, 33))
                main = cv2.resize(frame, (960, 540), interpolation=cv2.INTER_AREA)
                x0, y0, x1, y1 = 875, 393, 1058, 576
                zoom = cv2.resize(frame[y0:y1, x0:x1], (440, 440), interpolation=cv2.INTER_CUBIC)
                canvas.paste(Image.fromarray(cv2.cvtColor(main, cv2.COLOR_BGR2RGB)), (0, 55))
                canvas.paste(Image.fromarray(cv2.cvtColor(zoom, cv2.COLOR_BGR2RGB)), (980, 100))
                draw = ImageDraw.Draw(canvas)
                draw.text((16, 9), 'match2 第2点・4打目　奥ベースライン付近　0.25倍速', font=font, fill='white')
                draw.text((1010, 13), f'元フレーム F{f}', font=small, fill='white')
                draw.text((980, 67), '奥の選手の足元を拡大', font=small, fill='white')
                repeats = 1
                if f == f_review:
                    x = 980 + 440 * (x_review - x0) / (x1 - x0)
                    y = 100 + 440 * (y_review - y0) / (y1 - y0)
                    draw.ellipse((x - 13, y - 13, x + 13, y + 13), outline=(255, 182, 83), width=2)
                    draw.text((980, 550), '橙丸　着地付近の球らしい像', font=small, fill=(255, 182, 83))
                    canvas.save(OUT / 'point2_far_bounce_candidate_still.png')
                    repeats = 10
                data = np.asarray(canvas).tobytes()
                for _ in range(repeats):
                    proc.stdin.write(data)
                count += repeats
        finally:
            proc.stdin.close()
        assert proc.wait() == 0
    cap = cv2.VideoCapture(str(path))
    for _ in range(count):
        ok, frame = cap.read()
        assert ok and frame.shape[:2] == (640, 1440)
    assert not cap.read()[0]
    cap.release()
    return dict(file=path.name, sha256=common.sha(path), rendered_frames=count, first_source_frame=13963,
                last_source_frame=14022, marked_frame_hold=dict(frame=f_review, repeats=10))


def main():
    OUT.mkdir(exist_ok=True)
    ball_hash = common.sha(pilot.BALL)
    ball = {int(row['source_frame']): row for row in common.read(pilot.BALL)}
    before = {f: tuple(float(ball[f][k]) for k in ('current_x_px', 'current_y_px'))
              for f in range(13950, 13974) if ball[f]['current_candidate_present'] == '1'}
    after = after_bounce_candidates()
    homography = common.load(pilot.PILOT / 'protocol.json')['image_to_court_homography_canonical']
    results = profile(before, after, homography)
    best = min(results, key=lambda r: r['rms_image_px'])
    plausible = [p for p in results if p['rms_image_px'] <= best['rms_image_px'] + 1]
    screening = [
        dict(point=1, stroke=2, status='undetermined', evidence='Ball partially hidden by near player; no clear ground reversal.'),
        dict(point=2, stroke=2, status='possible_airborne_return',
             evidence='Ball rises across the far half up to the F2 swing; no ground reversal identified.'),
        dict(point=2, stroke=4, status='far_bounce_candidate',
             evidence='Descending and rising branches separated by obstruction near F2; possible streak F13980.'),
        dict(point=3, stroke=2, status='possible_airborne_return',
             evidence='Visible ball travels above the far court until the reaching F1 swing; contact remains provisional.'),
        dict(point=3, stroke=4, status='likely_no_bounce_before_smash',
             evidence='High descending ball meets F1 overhead; pre-contact image height stays above far baseline.'),
    ]
    assert common.sha(pilot.BALL) == ball_hash
    result = dict(
        selected_far_directed_strokes=screening, conditional_model_best=best,
        sensitivity_within_one_image_px=dict(
            frame=[min(p['frame'] for p in plausible), max(p['frame'] for p in plausible)],
            court_y=[min(p['court_xy'][1] for p in plausible), max(p['court_xy'][1] for p in plausible)]),
        visual_streak=dict(zip(('frame', 'x_px', 'y_px'), VISUAL_STREAK)),
        incoming_candidate_positions=before, outgoing_candidate_positions=after, time_profile=results,
        review_video=render(), input_ball_path_sha256=ball_hash,
        status='One exploratory candidate. Ball height assumed zero only at the fit breakpoint. '
               'The fit does not constrain in/out. Sensitivity crosses the baseline. No candidate promoted.',
    )
    common.dump(OUT / 'result.json', result)
    print(best, result['sensitivity_within_one_image_px'], flush=True)


if __name__ == '__main__':
    main()
