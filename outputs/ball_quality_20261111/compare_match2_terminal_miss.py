"""Test a past-only search at the last frame of the third point's analysis window."""
import sys

import cv2
import numpy as np
import torch

import build_044_spatiotemporal_pilot as common
from trial_056_crop_reacquisition import components

BASE = common.HERE / 'match2_points_2_3'
POINT = BASE / 'point_03'
EVALUATION = common.HERE / 'match2_first_three_points_evaluated'
OUT = common.HERE / 'match2_point_3_terminal_miss'
TARGET = 15505


def main():
    baseline_file = POINT / 'baseline_unreviewed_candidates.csv'
    base_info = common.load(BASE / 'verification.json')['points'][1]
    assert base_info['point'] == 3 and common.sha(baseline_file) == base_info['baseline_csv_sha256']
    baseline = {int(row['source_frame']): row for row in common.read(baseline_file)}
    before, previous = baseline[TARGET-2], baseline[TARGET-1]
    assert before['candidate_present'] == previous['candidate_present'] == '1'
    assert baseline[TARGET]['candidate_present'] == '0' and max(baseline) == TARGET
    recent = np.array([[float(row['x_px']), float(row['y_px'])] for row in [before, previous]])
    guide = recent[1] + (recent[1] - recent[0])
    settings = common.load(common.HERE / '056_temporal_reacquisition_trial/protocol.json')
    threshold, radius = settings['candidate_threshold'], settings['native_search_radius_px']
    assert (threshold, radius) == (.1, 32.)
    left = max(0, min(960, round(guide[0]-480)))
    top = max(0, min(540, round(guide[1]-270)))
    crop = (left, top, left+960, top+540)
    model_root = common.SOURCE / 'third_party/RacketVision/source/BallTrack'
    weight = model_root / 'checkpoints/balltrack_best.pth'
    weight_hash = common.load(BASE / 'protocol.json')['weight_sha256']
    assert common.sha(weight) == weight_hash
    OUT.mkdir(exist_ok=False)
    common.dump(OUT / 'protocol.json', dict(
        scope='Retrospective diagnosis of the sole visible reference miss among the first three points.',
        target=TARGET, inputs_to_guide=[TARGET-2, TARGET-1], guide_xy_px=guide.tolist(),
        guide_rule='One-frame constant velocity from the two preceding baseline candidates. Reference position is used only for scoring.',
        temporal_difference_from_056='Past-only guide, because this is the last frame of the analysis window.',
        threshold=threshold, search_radius_px=radius, crop_xyxy=list(crop),
        weight_sha256=weight_hash, baseline_sha256=common.sha(baseline_file),
        candidate_rule='Return exactly one heatmap component center inside the guide radius. No returned extrapolated point.',
    ))
    sys.path.insert(0, str(model_root))
    from inference import BallInferencer

    torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    detector = BallInferencer(str(model_root/'configs/tracknetv3_base.py'), str(weight), device='cuda', thre=.3, batchsize=4)
    assert detector.device.type == 'cuda'
    detector.model.eval().requires_grad_(False)
    background = np.load(POINT/'median.npz')['median']
    frames = {frame: cv2.imread(str(POINT/'frames'/f'{frame}.jpg')) for frame in range(TARGET-5, TARGET+1)}
    assert all(im.shape == (1080, 1920, 3) for im in frames.values())

    def infer(batch, box):
        x0, y0, x1, y1 = box
        inputs = []
        for frame in batch:
            images = [background] + [frames[i] for i in range(frame-3, frame+1)]
            samples = [cv2.resize(im[y0:y1, x0:x1], (512, 288)) for im in images]
            inputs.append(np.moveaxis(np.asarray(samples, dtype=np.float32)/255., -1, 1).reshape(15,288,512))
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            heatmaps, _, _ = detector.model.forward(frames=torch.from_numpy(np.stack(inputs)).to(detector.device))
        return heatmaps[:,0].float().cpu().numpy()

    full_box = (0, 0, 1920, 1080)
    batch = list(range(TARGET-2, TARGET+1))
    full = infer(batch, full_box)
    for frame, heatmap in zip(batch, full, strict=True):
        x, y, w, h, confidence = detector._predict_location(heatmap)
        row = baseline[frame]
        expected = (int(row['x_px']), int(row['y_px'])) if row['candidate_present'] == '1' else (0, 0)
        assert (int((x+w/2)*1920/512), int((y+h/2)*1080/288)) == expected
        assert float(confidence) == float(row['confidence'])
    local = infer([TARGET], crop)[0]
    predictions = []
    for method, hm, box in [('full_image', full[-1], full_box), ('local_crop', local, crop)]:
        x0, y0, x1, y1 = box
        nearby = [c for c in components(hm, threshold, x0, y0, x1-x0, y1-y0)
                  if np.linalg.norm(np.array([c['x_px'],c['y_px']])-guide) <= radius]
        choice = nearby[0] if len(nearby) == 1 else None
        predictions.append(dict(method=method, source_frame=TARGET, eligible_components=len(nearby),
            x_px=choice['x_px'] if choice else '', y_px=choice['y_px'] if choice else '',
            peak=choice['peak'] if choice else ''))
    np.savez_compressed(OUT/'heatmaps.npz', full=full[-1], local=local)
    common.write(OUT/'unscored_candidates.csv', predictions)

    comparison = EVALUATION/'reference_comparison_all_24_times.csv'
    assert common.sha(comparison) == common.load(EVALUATION/'publication_hashes.json')[comparison.name]
    reference, = [row for row in common.read(comparison) if int(row['source_frame']) == TARGET]
    assert reference['result'] == 'missing_visible_ball'
    truth = np.array([float(reference['truth_x_px']),float(reference['truth_y_px'])])
    for prediction in predictions:
        distance = float(np.linalg.norm(np.array([prediction['x_px'],prediction['y_px']])-truth)) if prediction['x_px'] != '' else None
        prediction.update(reference_x_px=truth[0], reference_y_px=truth[1],
                          distance_px='' if distance is None else distance,
                          result='no_candidate' if distance is None else 'within16px' if distance <= 16 else 'outside16px')
    common.write(OUT/'scored_comparison.csv', predictions)
    common.dump(OUT/'result.json', dict(target=TARGET, reference_xy_px=truth.tolist(), guide_xy_px=guide.tolist(),
        baseline_reproduced_in_final_batch=True, reference_used_for_scoring_only=True,
        predictions=predictions, scope='One known miss, after-the-fact method check.'))
    assert common.sha(baseline_file) == base_info['baseline_csv_sha256']
    print(predictions,flush=True)


if __name__ == '__main__':
    main()
