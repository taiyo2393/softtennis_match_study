"""Rerun existing TrackNet candidate extraction near reviewed false detections."""
from collections import Counter
import sys

import numpy as np
import torch

import build_044_spatiotemporal_pilot as common
from scan_match2_points_4_6 import OUT as TRACKING
from screen_match2_first_three_gaps import run_point
from sync_match2_points_4_6 import LAST_REVIEW_FRAME


OUT = common.HERE / 'match2_points_4_6_ball_recovery'
EXCLUSIONS = common.HERE / 'match2_points_4_6_ball_trail_audit/visual_exclusions.json'


def centre(row):
    return np.array([float(row['x_px']), float(row['y_px'])])


def targets():
    exclusions = common.load(EXCLUSIONS)
    bad = {f: group['point'] for group in exclusions['groups'] for f in group['frames']}
    report = common.load(TRACKING / 'ball_baseline_verification.json')
    windows, all_targets = [], []
    for record in report['per_point']:
        point, digest = record['point'], record['candidate_file_sha256']
        folder = TRACKING / f'point_{point:02}'
        path = folder / 'baseline_unreviewed_candidates.csv'
        assert common.sha(path) == digest
        baseline = {int(r['source_frame']): r for r in common.read(path)}
        valid = [f for f, r in baseline.items()
                 if f <= LAST_REVIEW_FRAME[point] and r['candidate_present'] == '1' and f not in bad]
        jobs = {}
        for f in sorted(g for g, p in bad.items() if p == point):
            prev = max((x for x in valid if x < f), default=None)
            following = min((x for x in valid if x > f), default=None)
            target = dict(point=point, source_frame=f, before=prev, after=following)
            if prev is not None and following is not None and following - prev <= 12:
                alpha = (f - prev) / (following - prev)
                guide = (1 - alpha) * centre(baseline[prev]) + alpha * centre(baseline[following])
                x = max(0, min(960, round(guide[0] - 480)))
                y = max(0, min(540, round(guide[1] - 270)))
                crop = (x, y, x + 960, y + 540)
                jobs[f] = (guide, crop)
                target.update(status='eligible', guide_xy=guide.tolist(), crop_xyxy=crop)
            else:
                target.update(status='no_short_two_sided_guide')
            all_targets.append(target)
        windows.append((point, folder, digest, baseline, [], jobs))
    assert len(all_targets) == len(bad)
    return windows, all_targets


def main():
    OUT.mkdir(exist_ok=True)
    windows, all_targets = targets()
    frozen = common.load(common.HERE / '056_temporal_reacquisition_trial/protocol.json')
    threshold, radius = frozen['candidate_threshold'], frozen['native_search_radius_px']
    assert (threshold, radius) == (.1, 32.)
    model = common.SOURCE / 'third_party/RacketVision/source/BallTrack'
    weight = model / 'checkpoints/balltrack_best.pth'
    assert common.sha(weight) == common.load(TRACKING / 'protocol.json')['weight_sha256']
    common.dump(OUT / 'protocol.json', dict(
        exclusions_sha256=common.sha(EXCLUSIONS), weight_sha256=common.sha(weight),
        targets=all_targets, candidate_threshold=threshold, radius_native_px=radius,
        guide='Linear image-space guide between nearest remaining original candidates in the reviewed view '
              'with endpoint span <=12 frames. '
              'Used only as a search guide. Verified visual exclusions remain fixed. No recursion or filling.',
        extraction='Use the existing full and 960x540 crop candidate extractor. A unique nearby component '
                   'is a recovery candidate, with identity subject to visual review.',
    ))
    sys.path.insert(0, str(model))
    from inference import BallInferencer

    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    detector = BallInferencer(str(model / 'configs/tracknetv3_base.py'), str(weight),
                              device='cuda', thre=.3, batchsize=4)
    assert detector.device.type == 'cuda'
    detector.model.eval().requires_grad_(False)
    proposals, counts = [], []
    by_frame = {t['source_frame']: t for t in all_targets}
    for window in windows:
        point, directory, digest, _, _, _ = window
        rows, reproduced = run_point(detector, window, threshold, radius)
        for row in rows:
            target = by_frame[row['source_frame']]
            row['previous_frames'] = f'{target["before"]},{target["after"]}'
            row['guide_anchor_frames'] = [target['before'], target['after']]
        proposals.extend(rows)
        counts.append(dict(point=point, reproduced_baseline_frames=reproduced))
        assert common.sha(directory / 'baseline_unreviewed_candidates.csv') == digest
        print(point, reproduced, len(rows), flush=True)
    outcomes = Counter()
    for full, crop in common.grouped(proposals).values():
        assert (full['method'], crop['method']) == ('full_image', 'local_crop')
        key = (full['candidate_x_px'] != '', crop['candidate_x_px'] != '')
        outcomes[{(True, True): 'both', (True, False): 'full_only',
                  (False, True): 'crop_only', (False, False): 'neither'}[key]] += 1
    common.dump(OUT / 'proposals.json', proposals)
    common.dump(OUT / 'result.json', dict(
        total_exclusions=len(all_targets), tested_frames=len(proposals) // 2,
        candidate_outcomes=dict(outcomes), baseline_reproductions=counts,
        visually_validated_recovery_frames=0,
    ))
    print(dict(outcomes), flush=True)


if __name__ == '__main__':
    main()
