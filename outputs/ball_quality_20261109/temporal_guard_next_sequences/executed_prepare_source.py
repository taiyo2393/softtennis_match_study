"""Freeze3methods on26metadata-reserved times, then create unmarked annotation media."""
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT, SOURCE, read, load, write, sha, cv2
from run_manual_ball_finetune import tensors
from compare_temporal_candidate_paths import select
from compare_ball_candidate_decoders import components
from compare_training_tile_detail import TILES, transform, mapped, peak_choice
from compare_grouped_tile_gate import F, features
from compare_frozen_ball_backbones import predict
from evaluate_observed_ball_fallback import camera_mask
from ball_temporal_consistency_rule import decide
from calibrate_point_supervised_detector import compact, expand

BASE = ROOT.parent / 'ball_quality_20261109'
OUT = BASE / 'temporal_guard_next_sequences'
FIXED = BASE / 'frozen_tile_gate_confirmation'
CFG = dict(order='displacement', strength=.1)


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def main():
    assert not (OUT / 'protocol.json').exists(), 'Keep first predictions and reserved targets.'
    reservation = load(OUT / 'reservation.json')
    assert all(sha(Path(p)) == h for p, h in reservation['protected_input_hashes'].items())
    assert sha(OUT / 'reserved_targets.csv') == reservation['reserved_targets_sha256']
    assert sha(OUT / 'reserved_windows.json') == reservation['reserved_windows_sha256']
    windows = load(OUT / 'reserved_windows.json')
    targets = read(OUT / 'reserved_targets.csv')
    assert len(targets) == 26 and all(r['review_status'] == 'unreviewed' for r in targets)
    modelroot = SOURCE / 'third_party/RacketVision/source/BallTrack'
    weight = modelroot / 'checkpoints/balltrack_best.pth'
    helpers = [ROOT / n for n in ['ball_temporal_consistency_rule.py', 'run_manual_ball_finetune.py',
        'compare_temporal_candidate_paths.py', 'compare_ball_candidate_decoders.py', 'compare_training_tile_detail.py',
        'compare_grouped_tile_gate.py', 'compare_frozen_ball_backbones.py', 'evaluate_observed_ball_fallback.py', 'calibrate_point_supervised_detector.py']]
    protected = dict(reservation['protected_input_hashes'])
    for p in [OUT / 'reservation.json', OUT / 'reserved_targets.csv', OUT / 'reserved_windows.json', weight,
        ROOT.parent / 'ball_quality_20261104/annotation_fit.html',
        ROOT.parent / 'ball_quality_20261104/completed_review/manual_sequence_labels.csv']:
        protected[str(p)] = sha(p)
    protocol = dict(status='fixed_before_inference_and_annotation', date_jst='2026-11-09',
        reservation_sha256=sha(OUT / 'reservation.json'), targets=26,
        methods=reservation['methods_fixed'], models='Original TrackNet plus already-frozen full-fit logistic10head; no training or grid.',
        inputs='Median+BGRt-3..t,512x288,/255,CUDA AMP,cudnn.benchmark=False. Per-clip contiguous ascending batch4 over center+/-28frames; use same groups for native/degraded9tiles on target-containing groups.',
        guard='Immutable original anchors at center+/-20frames computed using full original selector support. Same fixed8frame two-anchor contradiction rule; no substitute for rejected position.',
        offline='Up to16future sourceframes for guard support plus full-clip median; not causal/real-time.',
        annotations='All26target images and49sourceframe context videos with target hold,unmarked by model; original1920x1080source. No reused center. Same labels/states and common exclusions for3methods.',
        selection_or_early_stop='No result-based sample replacement or model choice. All26predictions frozen before human annotation. No accuracy until all human states are saved.',
        protected_input_hashes=protected, helper_hashes={str(p): sha(p) for p in helpers}, script_sha256=sha(Path(__file__)),
        automatic_adoption=False)
    dump(OUT / 'protocol.json', protocol)
    shutil.copy2(Path(__file__), OUT / 'executed_prepare_source.py')
    head = dict(np.load(FIXED / 'frozen_training_head.npz'))
    sys.path.insert(0, str(modelroot))
    from inference import BallInferencer
    torch.backends.cudnn.benchmark = False
    engine = BallInferencer(str(modelroot / 'configs/tracknetv3_base.py'), str(weight), device='cuda', thre=.3, batchsize=4)
    engine.model.eval().requires_grad_(False)
    pools = {arm: {r['sample_id']: [] for r in targets} for arm in ['native', 'degraded']}
    contexts, camera_records, image_hashes = {}, {}, {}
    map_records, ix_pack, val_pack, offsets = [], [], [], [0]
    predictions, feature_rows, diagnostics, queries, cases, labels = [], [], [], [], [], []
    for w in windows:
        cid, wid = w['candidate_id'], w['window_id']
        center = w['center_local_frame']
        fd, mp = Path(w['frame_dir']), Path(w['median_path'])
        times = list(range(center - 28, center + 29))
        target_by = {int(r['local_frame']): r for r in targets if r['candidate_id'] == cid}
        small = {}
        for j in range(times[0] - 16, times[-1] + 13):
            p = fd / f'{j:06d}.jpg'
            image_hashes[str(p)] = sha(p)
            small[j] = cv2.resize(cv2.imread(str(p), cv2.IMREAD_REDUCED_COLOR_8), (160, 90), interpolation=cv2.INTER_AREA)[8:82].astype(np.int16)
        scores = np.zeros(int(w['frame_count']))
        for j in range(times[0] - 4, times[-1] + 13):
            scores[j] = float((np.abs(small[j] - small[j - 12]).mean(2) > 24).mean())
        mask = camera_mask(scores)
        camera_records[cid] = dict(valid_mask_first=times[0], valid_mask_last=times[-1],
            computed_score_first=times[0] - 4, computed_score_last=times[-1] + 12,
            score_pairs=[[j, float(scores[j])] for j in range(times[0] - 4, times[-1] + 13)],
            mask_pairs=[[j, int(mask[j])] for j in times], outside_support='unknown')
        median = np.load(mp)['median']
        medians = {'global': np.moveaxis(cv2.resize(median, (512, 288)).astype(np.float32) / 255., -1, 0)}
        for arm in pools:
            for tile, box in enumerate(TILES):
                medians[arm + str(tile)] = np.moveaxis(transform(median, box, arm).astype(np.float32) / 255., -1, 0)
        context = {}
        for start in range(0, len(times), 4):
            ts = times[start:start + 4]
            images = {}
            for j in sorted({j for t in ts for j in range(t - 3, t + 1)}):
                p = fd / f'{j:06d}.jpg'
                digest = sha(p)
                assert digest == image_hashes[str(p)]
                images[j] = cv2.imread(str(p))
                assert images[j].shape == (1080, 1920, 3)
            arms = [('global', -1, None)]
            if any(t in target_by for t in ts):
                arms += [(arm, tile, box) for arm in pools for tile, box in enumerate(TILES)]
            for arm, tile, box in arms:
                batch = [dict(frames=np.stack([cv2.resize(images[j], (512, 288)) if arm == 'global' else transform(images[j], box, arm)
                    for j in range(t - 3, t + 1)]), median_key='global' if arm == 'global' else arm + str(tile), row={}) for t in ts]
                with torch.inference_mode(), torch.amp.autocast('cuda'):
                    outputs, _, _ = engine.model.forward(frames=tensors(batch, medians, engine.device))
                for t, hm in zip(ts, outputs[:, 0].float().cpu().numpy(), strict=True):
                    if arm != 'global' and t not in target_by:
                        continue
                    cs = components(hm)
                    sid = target_by[t]['sample_id'] if t in target_by else f'{cid}_context_{t:06d}'
                    if arm == 'global':
                        context[t] = cs
                    else:
                        pools[arm][sid].extend(mapped(cs, tile))
                    ix, val = compact(hm)
                    assert components(expand(ix, val)) == cs
                    ix_pack.append(ix)
                    val_pack.append(val)
                    offsets.append(offsets[-1] + len(ix))
                    map_records.append(dict(candidate_id=cid, local_frame=t, sample_id=sid, arm=arm, tile_index=tile, batch_frames_json=json.dumps(ts)))
        contexts[cid] = context
        anchors = {}
        for t in range(center - 20, center + 21):
            c, _ = select(context, t, mask, CFG)
            if c:
                anchors[t] = c
        for t, target in sorted(target_by.items()):
            sid = target['sample_id']
            original, selection_status = select(context, t, mask, CFG)
            proposal = peak_choice(pools['native'][sid])
            score = 0.
            if proposal:
                x = features(proposal, pools['native'][sid], pools['degraded'][sid], TILES)
                score = float(predict(head, x[None])[0])
                feature_rows.append(dict(sample_id=sid, **dict(zip(F, x.tolist())), proposal_gate_score=score))
            supplement = original if original else proposal if proposal and score >= .5 else None
            d = decide(t, supplement, original is not None, anchors, mask)
            guarded = supplement if d['keep'] else None
            identity = {k: target[k] for k in ['case_id', 'sample_id', 'window_id', 'candidate_id', 'local_frame', 'source_frame']}
            for method, point in [('original_temporal', original), ('static_supplement', supplement), ('two_anchor_guard_all_outputs', guarded)]:
                predictions.append(dict(**identity, method=method, visible=int(point is not None), x_px=point['x_px'] if point else '',
                    y_px=point['y_px'] if point else '', confidence=point['bbox_mean'] if point else 0.,
                    original_status=selection_status, guard_reason=d['reason'] if method == 'two_anchor_guard_all_outputs' else 'not_applied'))
            diagnostics.append(dict(**identity, **d))
            input_paths = [fd / f'{j:06d}.jpg' for j in range(t - 3, t + 1)]
            queries.append(dict(**identity, input_frames_json=json.dumps([str(p) for p in input_paths]),
                input_frame_sha256_json=json.dumps([sha(p) for p in input_paths]), median_path=str(mp), median_sha256=sha(mp)))
            source = fd / f'{t:06d}.jpg'
            cases.append(dict(**identity, source_image=str(source), source_image_sha256=sha(source), center_reused=False,
                context_video=str(OUT / f'context_{target["case_id"]}.mp4')))
            labels.append(dict(**identity, phase_status='unknown', truth_visibility='unreviewed', truth_x_px='', truth_y_px='',
                reviewer_id='', annotated_at_utc='', notes='', center_source='pending_new_human_click', review_status='unreviewed'))
        print('Frozen method outputs:', wid, '13targets; no correctness scoring', flush=True)
    assert len(predictions) == 78 and len(map_records) == 582
    write(OUT / 'frozen_predictions.csv', predictions)
    write(OUT / 'guard_diagnostics.csv', diagnostics)
    write(OUT / 'query_model_inputs.csv', queries)
    write(OUT / 'manual_sequence_labels.csv', labels)
    shutil.copy2(OUT / 'manual_sequence_labels.csv', OUT / 'initial_labels_snapshot.csv')
    write(OUT / 'map_records.csv', map_records)
    if feature_rows:
        write(OUT / 'candidate_features.csv', feature_rows)
    dump(OUT / 'global_contexts.json', contexts)
    dump(OUT / 'candidate_pools.json', pools)
    dump(OUT / 'camera_features.json', camera_records)
    np.savez_compressed(OUT / 'component_rectangles.npz', indices=np.concatenate(ix_pack), values=np.concatenate(val_pack), offsets=np.array(offsets, dtype=np.int64))
    # Predictions are saved before annotation media; no model points in images/videos.
    for c in cases:
        w = next(w for w in windows if w['candidate_id'] == c['candidate_id'])
        target = int(c['local_frame'])
        output = Path(c['context_video'])
        cmd = [r'C:\ffmpeg\bin\ffmpeg.exe', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', '1280x800', '-r', str(w['fps']), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
        frames = 0
        with (OUT / f'{c["case_id"]}_render.log').open('wb') as log:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
            for j in range(target - 24, target + 25):
                p = Path(w['frame_dir']) / f'{j:06d}.jpg'
                image_hashes[str(p)] = sha(p)
                im = cv2.imread(str(p))
                assert im.shape == (1080, 1920, 3)
                frame = np.full((800, 1280, 3), 18, np.uint8)
                frame[80:] = cv2.resize(im, (1280, 720), interpolation=cv2.INTER_AREA)
                cv2.putText(frame, f'{c["case_id"]} | expand{c["candidate_id"][-3:]} local{j} | TARGET {target} | 0.5x', (12, 29), 0, .74, (240, 240, 240), 1, cv2.LINE_AA)
                if j == target:
                    cv2.rectangle(frame, (2, 81), (1277, 797), (0, 225, 255), 3)
                    cv2.putText(frame, 'TARGET FRAME - no model predictions shown', (12, 61), 0, .72, (0, 225, 255), 2, cv2.LINE_AA)
                for _ in range(32 if j == target else 2):
                    proc.stdin.write(frame.tobytes())
                    frames += 1
            proc.stdin.close()
            assert proc.wait() == 0 and frames == 128
        c.update(context_video_sha256=sha(output), decoded_video_frames=frames)
    dump(OUT / 'annotation_cases.json', cases)
    dump(OUT / 'source_image_hashes.json', image_hashes)
    template = (ROOT.parent / 'ball_quality_20261104/annotation_fit.html').read_text(encoding='utf-8')
    template = template.replace('4場面', '2場面').replace('全52件', '全26件')
    template = template.replace('<style>', '<style>[hidden]{display:none!important}')
    template = template.replace('準備中や終了後なら、その状態を選んでください。', '有効サーブの打球から得失点が成立するまでをプレー中とします。フォールト・レットの試行や準備・終了後は対象外です。境目が不明なら「プレー中か判断できない」を選んでください。')
    template = template.replace('let cases=[],', "let revision='',cases=[],")
    template = template.replace('case_id:c.case_id,truth_visibility:', 'case_id:c.case_id,revision:revision,truth_visibility:')
    template = template.replace('cases=data.cases;dirty=false;', 'cases=data.cases;revision=data.revision;dirty=false;')
    template = template.replace('cases=d.cases;options();', 'cases=d.cases;revision=d.revision;options();')
    (OUT / 'annotation.html').write_text(template, encoding='utf-8')
    assert all(sha(Path(p)) == h for p, h in protected.items())
    assert all(sha(Path(p)) == h for p, h in image_hashes.items())
    names = ['protocol.json', 'reservation.json', 'reserved_windows.json', 'reserved_targets.csv', 'frozen_predictions.csv',
        'guard_diagnostics.csv', 'query_model_inputs.csv', 'map_records.csv', 'global_contexts.json', 'candidate_pools.json',
        'camera_features.json', 'component_rectangles.npz', 'annotation_cases.json', 'source_image_hashes.json', 'annotation.html',
        'executed_prepare_source.py', 'initial_labels_snapshot.csv']
    if feature_rows:
        names.append('candidate_features.csv')
    manifest = dict(status='26pending_labels_78frozen_outputs_blind_media_ready', cases=26, predictions=78,
        cases_sha256=sha(OUT / 'annotation_cases.json'), predictions_sha256=sha(OUT / 'frozen_predictions.csv'),
        initial_labels_sha256=sha(OUT / 'initial_labels_snapshot.csv'), protected_input_hashes=protected,
        frozen_artifact_hashes={n: sha(OUT / n) for n in names}, helper_hashes=protocol['helper_hashes'],
        no_accuracy_computed=True, operational_changes=False)
    dump(OUT / 'manifest.json', manifest)
    print('Prepared26pending labels,78frozen outputs,and26unmarked videos. No accuracy computed.', flush=True)


if __name__ == '__main__':
    main()
