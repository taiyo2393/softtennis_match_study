"""Freeze points 4–6 in chronology and apply the existing match2 ball baseline."""
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import build_044_spatiotemporal_pilot as common
from scan_match2_points_4_6 import OUT, SOURCE


POINTS = [
    dict(point=4, score_before=[2, 1], score_after=[3, 1], score_frames=[16654, 17368],
         serve_candidate=[16704, 16710], terminal_candidate=[17276, 17303], cache_span=[16570, 17303]),
    dict(point=5, score_before=[3, 1], score_after=[3, 2], score_frames=[18709, 19084],
         serve_candidate=[18712, 18718], terminal_candidate=[19041, 19059], cache_span=[18645, 19055]),
    dict(point=6, score_before=[3, 2], score_after=[3, 3], score_frames=[20757, 21430],
         serve_candidate=[20781, 20787], terminal_candidate=[21350, 21398], cache_span=[20705, 21398]),
]
FIRST_PROTOCOL = common.HERE / 'match2_ball_first_point/protocol.json'


def extract(capture, first, last):
    assert capture.set(cv2.CAP_PROP_POS_FRAMES, first)
    for f in range(first, last + 1):
        ok, frame = capture.read()
        assert ok and frame.shape == (1080, 1920, 3)
        assert round(capture.get(cv2.CAP_PROP_POS_FRAMES)) == f + 1
        yield f, frame


def prepare():
    OUT.mkdir(exist_ok=True)
    ref = OUT / 'raw_reference_frames'
    ref.mkdir(exist_ok=True)
    caps = cv2.VideoCapture(str(SOURCE))
    assert caps.isOpened() and abs(caps.get(cv2.CAP_PROP_FPS) - common.FPS) < .001
    points = []
    evidence = Image.new('RGB', (1160, 420), '#101820')
    draw = ImageDraw.Draw(evidence)
    font = ImageFont.truetype(str(common.FONT), 15)
    for i, control in enumerate(POINTS):
        p = control['point']
        lo, hi = control['cache_span']
        start, end = control['serve_candidate'][1] + 1, control['terminal_candidate'][0] - 1
        refs = np.rint(np.linspace(start, end, 6)).astype(int).tolist()
        med = np.rint(np.linspace(lo, hi, 120)).astype(int).tolist()
        point = dict(control, scoring_side=('far' if control['score_after'][0] > control['score_before'][0] else 'near'),
                     core_inclusive=[start, end], last_main_view_review_frame=hi,
                     landing_receiver_side=None, reference_frames=refs, median_frames=med)
        points.append(point)
        frames = OUT / f'point_{p:02}' / 'frames'
        frames.mkdir(parents=True, exist_ok=True)
        for f, frame in extract(caps, lo, hi):
            jpg = frames / f'{f}.jpg'
            assert cv2.imwrite(str(jpg), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if f in refs:
                saved = cv2.imread(str(jpg))
                assert cv2.imwrite(str(ref / f'point{p}_F{f}.png'), saved)
        for j, f in enumerate(point['score_frames']):
            _, frame = next(extract(caps, f, f))
            crop = Image.fromarray(cv2.cvtColor(frame[30:153, :580], cv2.COLOR_BGR2RGB))
            x, y = j * 580, i * 140
            evidence.paste(crop, (x, y))
            draw.text((x + 12, y + 121), f'第{p}点　{"前" if j == 0 else "後"}　F{f}', font=font, fill='white')
        print('prepared', p, start, end, flush=True)
    caps.release()
    evidence.save(OUT / 'score_changes_points_4_6.png')
    frozen = common.load(FIRST_PROTOCOL)
    common.dump(OUT / 'protocol.json', dict(
        source_video=str(SOURCE), source_fps='60000/1001',
        selection='Next three consecutive scoreboard increments after point 3. Replays excluded.',
        scoreboard_rows=['内本・内田組（奥）', '船水・上松組（手前）'],
        frozen_before_ball_outputs=True, points=points,
        detector=frozen['detector'], weight_sha256=frozen['weight_sha256'],
        raw_reference_sha256={p.name: common.sha(p) for p in sorted(ref.glob('*.png'))},
        scope='Rally boundary brackets are unverified visual candidates. Terminal context is retained only '
              'through the main-view frame cap. No shot identities or terminal type assumed from the score.',
    ))


def run():
    from run_match2_first_point_baseline import run_baseline

    protocol_file = OUT / 'protocol.json'
    protocol = common.load(protocol_file)
    protocol_sha = common.sha(protocol_file)
    model_root = common.SOURCE / 'third_party/RacketVision/source/BallTrack'
    weight = model_root / 'checkpoints/balltrack_best.pth'
    assert common.sha(weight) == protocol['weight_sha256']
    reports = []
    for point in protocol['points']:
        directory = OUT / f'point_{point["point"]:02}'
        first, last = point['core_inclusive']
        terminal = point['last_main_view_review_frame']
        total = run_baseline(weight, model_root, point['median_frames'], output_dir=directory,
                             core=range(first, terminal + 1))
        track = directory / 'baseline_unreviewed_candidates.csv'
        rows = common.read(track)
        core_count = sum(int(r['candidate_present']) for r in rows if int(r['source_frame']) <= last)
        report = dict(point=point['point'], core_frames=last-first+1, core_candidates=core_count,
                      terminal_context_frames=terminal-last, terminal_context_candidates=total-core_count,
                      candidate_file_sha256=common.sha(track))
        reports.append(report)
        print(report, flush=True)
    assert common.sha(protocol_file) == protocol_sha
    assert all(common.sha(OUT / 'raw_reference_frames' / n) == h
               for n, h in protocol['raw_reference_sha256'].items())
    common.dump(OUT / 'ball_baseline_verification.json', dict(
        per_point=reports, protocol_sha256=protocol_sha, independent_reference_points_annotated=0,
        interpretation='Candidate availability; terminal context counts separate.'))


if __name__ == '__main__':
    {'prepare': prepare, 'ball': run}[sys.argv[1]]()
