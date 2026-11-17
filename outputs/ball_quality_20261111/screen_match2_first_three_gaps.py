"""Compare two gap-recovery inputs across all three match2 points."""
from collections import Counter
from itertools import groupby
import sys

import cv2
import numpy as np
import torch

import build_044_spatiotemporal_pilot as common
from trial_056_crop_reacquisition import components

FIRST = common.HERE / 'match2_ball_first_point'
LATER = common.HERE / 'match2_points_2_3'
OUT = common.HERE / 'match2_first_three_gap_screen'
FULL_BOX = (0, 0, 1920, 1080)


def prepare_points():
    checks = common.load(LATER / 'verification.json')['points']
    first_hash = common.load(FIRST / 'publication_hashes.json')['baseline_unreviewed_candidates.csv']
    definitions = [(1, FIRST, first_hash)] + [
        (row['point'], LATER / f'point_{row["point"]:02}', row['baseline_csv_sha256']) for row in checks]
    points = []
    for number, folder, digest in definitions:
        path = folder / 'baseline_unreviewed_candidates.csv'
        assert common.sha(path) == digest
        baseline = {int(row['source_frame']): row for row in common.read(path)}
        assert list(baseline) == list(range(min(baseline), max(baseline) + 1))
        gaps, jobs = [], {}
        for present, run in groupby(baseline, key=lambda f: baseline[f]['candidate_present']):
            frames = list(run)
            if present == '1':
                continue
            first, last = frames[0], frames[-1]
            eligible = all(f in baseline and baseline[f]['candidate_present'] == '1' for f in (first-2, first-1))
            gaps.append(dict(point=number, first_source_frame=first, last_source_frame=last,
                frames=len(frames), eligible_first_frame=int(eligible)))
            if eligible:
                positions = [np.array([float(baseline[f][key]) for key in ('x_px', 'y_px')]) for f in (first-2, first-1)]
                guide = positions[1] + (positions[1] - positions[0])
                left = max(0, min(960, round(guide[0] - 480)))
                top = max(0, min(540, round(guide[1] - 270)))
                jobs[first] = (guide, (left, top, left+960, top+540))
        points.append((number, folder, digest, baseline, gaps, jobs))
    return points


def run_point(detector, point, threshold, radius):
    number, folder, _, baseline, _, jobs = point
    background = np.load(folder / 'median.npz')['median']
    start, end = min(baseline), max(baseline)
    original_batches = sorted({start + 4*((frame-start)//4) for frame in jobs})

    def infer(batch, boxes):
        inputs = []
        for frame, (x0, y0, x1, y1) in zip(batch, boxes, strict=True):
            images = [background] + [cv2.imread(str(folder/'frames'/f'{f}.jpg')) for f in range(frame-3, frame+1)]
            assert all(image is not None and image.shape == (1080,1920,3) for image in images)
            samples = [cv2.resize(image[y0:y1, x0:x1], (512,288)) for image in images]
            inputs.append(np.moveaxis(np.asarray(samples, dtype=np.float32)/255., -1, 1).reshape(15,288,512))
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            output, _, _ = detector.model.forward(frames=torch.from_numpy(np.stack(inputs)).to(detector.device))
        return output[:,0].float().cpu().numpy()

    full, local = {}, {}
    reproduced = 0
    for first in original_batches:
        batch = list(range(first, min(first+4, end+1)))
        for frame, heatmap in zip(batch, infer(batch, [FULL_BOX]*len(batch)), strict=True):
            x,y,w,h,confidence = detector._predict_location(heatmap)
            row = baseline[frame]
            expected = (int(row['x_px']),int(row['y_px'])) if row['candidate_present'] == '1' else (0,0)
            assert (int((x+w/2)*1920/512),int((y+h/2)*1080/288)) == expected
            assert float(confidence) == float(row['confidence'])
            reproduced += 1
            if frame in jobs:
                full[frame] = heatmap
    targets = sorted(jobs)
    for offset in range(0, len(targets), 4):
        batch = targets[offset:offset+4]
        local.update(zip(batch, infer(batch, [jobs[frame][1] for frame in batch]), strict=True))
    rows = []
    for frame in targets:
        guide, crop = jobs[frame]
        for method, heatmap, box in [('full_image', full[frame], FULL_BOX), ('local_crop', local[frame], crop)]:
            left, top, right, bottom = box
            nearby = [candidate for candidate in components(heatmap,threshold,left,top,right-left,bottom-top)
                      if np.hypot(candidate['x_px']-guide[0],candidate['y_px']-guide[1]) <= radius]
            chosen = nearby[0] if len(nearby) == 1 else None
            rows.append(dict(point=number, source_frame=frame, method=method, eligible_components=len(nearby),
                candidate_x_px=chosen['x_px'] if chosen else '', candidate_y_px=chosen['y_px'] if chosen else '',
                peak=chosen['peak'] if chosen else '', guide_x_px=float(guide[0]), guide_y_px=float(guide[1]),
                previous_frames=f'{frame-2},{frame-1}', crop_xyxy=list(box)))
    return rows, reproduced


def main():
    points = prepare_points()
    fixed = common.load(common.HERE/'056_temporal_reacquisition_trial/protocol.json')
    threshold, radius = fixed['candidate_threshold'], fixed['native_search_radius_px']
    assert (threshold, radius) == (.1, 32.)
    model_root = common.SOURCE / 'third_party/RacketVision/source/BallTrack'
    weight = model_root / 'checkpoints/balltrack_best.pth'
    weight_hash = common.load(LATER/'protocol.json')['weight_sha256']
    assert common.sha(weight) == weight_hash == common.load(FIRST/'protocol.json')['weight_sha256']
    OUT.mkdir(exist_ok=False)
    common.dump(OUT/'protocol.json', dict(frozen_before_new_outputs=True,
        scope='Three already explored match2 points; exploratory candidate screening.',
        target_rule='First frame of each original no-output run, with two immediately preceding original candidates.',
        guide_rule='One-frame constant velocity; no recursive use of recovered candidates.',
        inputs='Existing per-point offline median plus t-3..t. Same weight, preprocessing, CUDA AMP.',
        threshold=threshold, radius_native_px=radius, crop_native_wh=[960,540],
        candidate_rule='Exactly one component within the search radius. Keep methods separate and expose conflicts.',
        selection_for_spot_check='First full-only and first crop-only output not previously reviewed; first pair more than 16px apart if present.',
        weight_sha256=weight_hash, points=[dict(point=n, baseline_sha256=h, eligible_frames=sorted(j)) for n,_,h,_,_,j in points]))
    common.write(OUT/'original_missing_runs.csv', [gap for _,_,_,_,gaps,_ in points for gap in gaps])
    sys.path.insert(0, str(model_root))
    from inference import BallInferencer

    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    detector = BallInferencer(str(model_root/'configs/tracknetv3_base.py'),str(weight),device='cuda',thre=.3,batchsize=4)
    assert detector.device.type == 'cuda'
    detector.model.eval().requires_grad_(False)
    all_rows, summaries, paired = [], [], []
    for point in points:
        number, folder, digest, baseline, gaps, jobs = point
        rows, reproduced = run_point(detector, point, threshold, radius)
        all_rows.extend(rows)
        by_frame = common.grouped(rows)
        categories = Counter()
        for frame, (full, local) in sorted(by_frame.items()):
            assert full['method'] == 'full_image' and local['method'] == 'local_crop'
            a, b = full['candidate_x_px'] != '', local['candidate_x_px'] != ''
            kind = 'both' if a and b else 'full_only' if a else 'crop_only' if b else 'neither'
            distance = float(np.hypot(full['candidate_x_px']-local['candidate_x_px'],full['candidate_y_px']-local['candidate_y_px'])) if a and b else None
            categories[kind] += 1
            paired.append(dict(point=number,source_frame=frame,outcome=kind,
                full_x_px=full['candidate_x_px'],full_y_px=full['candidate_y_px'],
                crop_x_px=local['candidate_x_px'],crop_y_px=local['candidate_y_px'],distance_px='' if distance is None else distance))
        assert common.sha(folder/'baseline_unreviewed_candidates.csv') == digest
        summary = dict(point=number, core_frames=len(baseline), missing_frames=sum(g['frames'] for g in gaps),
            missing_runs=len(gaps), eligible_first_frames=len(jobs), baseline_reproduced_frames=reproduced,
            **{key:categories[key] for key in ['both','full_only','crop_only','neither']})
        summaries.append(summary)
        print(summary, flush=True)
    common.write(OUT/'method_candidates.csv', all_rows)
    common.write(OUT/'paired_candidates.csv', paired)
    common.write(OUT/'point_summary.csv', summaries)
    common.dump(OUT/'result.json', dict(points=summaries, candidate_identity_status='pending except prior F12389/F15505 where output matches prior reviewed position',
        coordinates_or_references_modified=False))


if __name__ == '__main__':
    main()
