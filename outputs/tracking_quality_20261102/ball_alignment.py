"""Correct the verified MS-TrackNetV3 output-frame mismatch, preserving old runs.

Training Uball._gen_sequence ends its input at target t; released inference ends
at t-1. Thus old prediction t+1 has exactly the corrected input for target t.
Reuse those outputs without recomputing thousands of identical network inputs;
infer the last frame separately. Never interpolate or change thresholds here.
"""
import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

SOURCE = Path(r'C:\Users\ta239\Research\softtennis_match_study\study')
OUT = Path(__file__).resolve().parent


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def window(t, length=4, legacy=False):
    end = t if legacy else t + 1
    return [max(0, j) for j in range(end - length, end)]


def evaluate(labels, predictions, split):
    counts = Counter()
    errors = []
    for r in labels:
        if r['evaluation_split'] != split:
            continue
        state = r['truth_visibility']
        if state not in ('visible', 'not_visible'):
            counts['unevaluable'] += 1
            continue
        p = predictions[r['candidate_id']][int(r['local_frame'])]
        truth, pred = state == 'visible', int(p['visible']) == 1
        counts['frames'] += 1
        counts['truth_visible'] += truth
        counts['pred_visible'] += pred
        if truth and pred:
            e = math.hypot(float(p['x_px']) - float(r['truth_x_px']),
                           float(p['y_px']) - float(r['truth_y_px']))
            errors.append(e)
            counts['presence_tp'] += 1
            counts['localized_tp_16px'] += e <= 16
            counts['wrong_localization'] += e > 16
        elif truth:
            counts['presence_fn'] += 1
        elif pred:
            counts['presence_fp'] += 1
        else:
            counts['presence_tn'] += 1
    return dict(counts=dict(counts),
                localization_recall_16px=counts['localized_tp_16px']/max(1, counts['truth_visible']),
                localization_precision_16px=counts['localized_tp_16px']/max(1, counts['pred_visible']),
                mean_error_both_visible_px=float(np.mean(errors)) if errors else None,
                median_error_both_visible_px=float(np.median(errors)) if errors else None)


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--skip-last-inference', action='store_true',
                        help='Diagnostic only: mark final output absent rather than copy an incorrect frame.')
    args = parser.parse_args()
    for t in range(100):
        assert window(t) == window(t + 1, legacy=True)
    labels = read(SOURCE/'annotations/balltrack_groundtruth_match1_pilot.csv')
    cids = sorted({r['candidate_id'] for r in labels})
    inferencer = None
    if not args.skip_last_inference:
        sys.path.insert(0, str(SOURCE/'third_party/RacketVision/source/BallTrack'))
        import torch
        from inference import BallInferencer
        ballroot = SOURCE/'third_party/RacketVision/source/BallTrack'
        inferencer = BallInferencer(str(ballroot/'configs/tracknetv3_base.py'),
                                    str(ballroot/'checkpoints/balltrack_best.pth'),
                                    device='cuda', thre=.30, batchsize=1)
    legacy, fixed, reports = {}, {}, []
    for cid in cids:
        base = SOURCE/'data/balltrack/racketvision/match1'/cid
        oldpath = base/'runs/threshold030_median120/balltrack_predictions.csv'
        rows = read(oldpath)
        n = len(rows)
        assert [int(r['local_frame']) for r in rows] == list(range(n))
        result = []
        for t, r in enumerate(rows):
            p = rows[t+1] if t+1 < n else None
            new = dict(r)
            for key in ('x_px', 'y_px', 'visible', 'confidence'):
                new[key] = p[key] if p else 0
            new.update(position_source='model_observed',
                       temporal_alignment='input_ends_at_output_frame',
                       inference_origin=f'legacy_local_frame_{t+1}' if p else 'last_frame_not_inferred')
            result.append(new)
        if inferencer is not None:
            paths = [base/'frames'/f'{i:06d}.jpg' for i in window(n-1)]
            imgs = [cv2.imread(str(p)) for p in paths]
            if any(im is None for im in imgs):
                raise RuntimeError(f'Missing cached input: {cid}')
            h, w = imgs[-1].shape[:2]
            frames = np.array([cv2.resize(im, (inferencer.width, inferencer.height))
                               for im in imgs], dtype=np.float32)/255.
            median = np.load(base/'runs/threshold030_median120/median.npz')['median']
            median = cv2.resize(median, (inferencer.width, inferencer.height))/255.
            data = np.concatenate([np.moveaxis(median, -1, 0)[None],
                                   np.moveaxis(frames, -1, 1)], axis=0)
            tensor = torch.from_numpy(data.reshape(1, -1, inferencer.height,
                                                  inferencer.width)).float().to(inferencer.device)
            with torch.inference_mode():
                if inferencer.device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        pred, _, _ = inferencer.model.forward(frames=tensor)
                else:
                    pred, _, _ = inferencer.model.forward(frames=tensor)
            hm = pred[0, 0].float().cpu().numpy()
            x, y, bw, bh, conf = inferencer._predict_location(hm)
            cx, cy = int((x+bw/2)*w/inferencer.width), int((y+bh/2)*h/inferencer.height)
            result[-1].update(x_px=cx, y_px=cy, visible=int(cx != 0 or cy != 0),
                              confidence=round(conf, 4), inference_origin='fresh_corrected_last_frame')
        destination = OUT/'ball_aligned'/cid
        destination.mkdir(parents=True, exist_ok=True)
        with (destination/'balltrack_predictions.csv').open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(result[0]))
            writer.writeheader()
            writer.writerows(result)
        legacy[cid] = {int(r['local_frame']):r for r in rows}
        fixed[cid] = {int(r['local_frame']):r for r in result}
        reports.append(dict(candidate_id=cid, frames=n, old_csv=str(oldpath),
                            old_csv_sha256=hashlib.sha256(oldpath.read_bytes()).hexdigest(),
                            last_frame_inferred=inferencer is not None))
        print(f'{cid}: {n} aligned outputs', flush=True)
    report = dict(change='Structural timing correction; no threshold/model/label changes',
                  evidence=dict(training='dataset/uball.py:_gen_sequence ends at fid',
                                model='model/tracknet_v3.py:last_only selects last heatmap',
                                legacy='inference.py:_preprocess_batch uses range(i-seq_len,i)'),
                  labels_sha256=hashlib.sha256((SOURCE/'annotations/balltrack_groundtruth_match1_pilot.csv').read_bytes()).hexdigest(),
                  warning='Agreement with existing labels, whose independent annotation provenance is unverified. Existing test is not a new held-out external test.',
                  image_size='1920x1080', runs=reports,
                  metrics={s:dict(legacy=evaluate(labels, legacy, s), aligned=evaluate(labels, fixed, s))
                           for s in ('validation', 'test')})
    (OUT/'ball_alignment_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report['metrics'], indent=2))


if __name__ == '__main__':
    main()
