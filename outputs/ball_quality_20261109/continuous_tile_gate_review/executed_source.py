"""Fixed contiguous contexts for qualitative tracking review, never pseudo-GT."""
import argparse
import json
import math
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT, SOURCE, read, load, write, sha, cv2
from compare_grouped_tile_gate import F, features
from compare_frozen_ball_backbones import predict
from compare_training_tile_detail import TILES, transform, mapped, peak_choice
from compare_ball_candidate_decoders import components
from compare_temporal_candidate_paths import select
from evaluate_observed_ball_fallback import camera_mask
from run_manual_ball_finetune import tensors
from calibrate_point_supervised_detector import compact, expand

BASE = ROOT.parent / 'ball_quality_20261109'
FIXED = BASE / 'frozen_tile_gate_confirmation'
CHECK = ROOT.parent / 'ball_quality_20261104'
OUT = BASE / 'continuous_tile_gate_review'
CFG = dict(name='displacement_0.1', order='displacement', strength=.1)


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')


def verify_protected():
    protocol = load(OUT / 'protocol.json')
    assert all(sha(Path(p)) == h for p, h in protocol['protected_input_hashes'].items())
    assert all(sha(Path(p)) == h for p, h in protocol['helper_hashes'].items())
    assert sha(Path(__file__)) == protocol['script_sha256']
    return protocol


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'Preserve the first fixed continuous run.'
    assert load(FIXED / 'evaluation_report.json')['passes_prespecified_confirmation_guard']
    windows = load(CHECK / 'reserved_windows.json')
    plans = [dict(w, review_first=w['center_local_frame'] - 90, review_last=w['center_local_frame'] + 90,
        context_first=w['center_local_frame'] - 98, context_last=w['center_local_frame'] + 98) for w in windows]
    paths = [FIXED / n for n in ['protocol.json', 'frozen_training_head.npz', 'full_fit_decision.json',
        'evaluation_report.json', 'verification.json', 'frozen_confirmation_predictions.csv', 'candidate_pools.json']]
    paths += [CHECK / n for n in ['reserved_windows.json', 'frozen_predictions.csv', 'frozen_component_cache.json',
        'completed_review/manual_sequence_labels.csv', 'evaluation_report.json']]
    paths += [ROOT / n for n in ['reviewed_ball_overrides.json', 'reviewed_person_overrides.json', 'reviewed_tracking_bundles.json']]
    weight = SOURCE / 'third_party/RacketVision/source/BallTrack/checkpoints/balltrack_best.pth'
    paths += [weight] + [Path(p['median_path']) for p in plans]
    helpers = [ROOT / n for n in ['compare_grouped_tile_gate.py', 'compare_frozen_ball_backbones.py',
        'compare_training_tile_detail.py', 'compare_ball_candidate_decoders.py', 'compare_temporal_candidate_paths.py',
        'evaluate_observed_ball_fallback.py', 'run_manual_ball_finetune.py', 'calibrate_point_supervised_detector.py']]
    protocol = dict(status='fixed_before_contiguous_inference', date_jst='2026-11-09',
        selection='ALL4 previously reserved clips,center +/-90source frames inclusive,181frames each. No new output/error-based window selection.',
        target_frames=724, context_frames=788, plans=plans,
        algorithm='Exactly fixed original TrackNet and full-fit logistic10head,thresholds0.3/0.5,fixed9tiles,no training,new threshold,temporal gate or interpolation.',
        batching='Continuous ascending global times in batch4 including odd/even frames. Tile inference uses the SAME time batches,only when any target in batch lacks original temporal output; retain proposals only for missing targets. Existing original outputs always preserved.',
        numerical_check='Compare new globalcontext components and52anchor outputs with previous sparse-batch caches. Report any drift; never overwrite a new prediction with an old anchor or tune settings to force agreement.',
        camera='Same reduced-color8,160x90AREA,rows8:82,int16,lag12,meanBGRdiff>24fraction>.25,expand12before4after. Compute all lagged image support for entire context; outside support remains unknown.',
        phase='Only previous52target moments have human phase/ball annotation. Expanded contexts NOT assumed in-play; ask human review. No numeric in-play accuracy for unlabelled contexts.',
        diagnostics='Output presence counts,longest no-output run,consecutive-output jumps>64sourcepixels,and reappearances after>=6no-output frames are REVIEW FLAGS,not false detection or switch labels. No rejection based on flags.',
        playback='4full windows,1920x1080image unchanged +100pixel header. Each1x then0.5x; sourceorder monotonic within each pass,no interpolation,holds or trails. CYAN existing temporal output,GREEN added actual tile candidate; colors do not certify correctness.',
        interpretation='Known same-match development,qualitative continuous inspection. Sparse-anchor results cannot establish724frame or full-rally identity accuracy.',
        operational_adoption=False, detector_training=False, new_labels=False,
        protected_input_hashes={str(p): sha(p) for p in paths}, helper_hashes={str(p): sha(p) for p in helpers}, script_sha256=sha(Path(__file__)))
    dump(OUT / 'protocol.json', protocol)
    shutil.copy2(Path(__file__), OUT / 'executed_source.py')
    verify_protected()
    fixed_head = dict(np.load(FIXED / 'frozen_training_head.npz'))
    assert sha(FIXED / 'frozen_training_head.npz') == load(FIXED / 'full_fit_decision.json')['model_sha256']
    modelroot = weight.parent.parent
    sys.path.insert(0, str(modelroot))
    from inference import BallInferencer
    torch.backends.cudnn.benchmark = False
    engine = BallInferencer(str(modelroot / 'configs/tracknetv3_base.py'), str(weight), device='cuda', thre=.3, batchsize=4)
    engine.model.eval().requires_grad_(False)
    image_hashes, all_contexts, camera_records = {}, {}, {}
    rows, feature_rows, map_records = [], [], []
    pools = {arm: {} for arm in ['native', 'degraded']}
    ix_pack, val_pack, offsets = [], [], [0]
    old_context = load(CHECK / 'frozen_component_cache.json')
    global_diffs = []
    for p in plans:
        cid, wid = p['candidate_id'], p['window_id']
        fd = Path(p['frame_dir'])
        times = list(range(p['context_first'], p['context_last'] + 1))
        assert times[0] - 16 >= 0 and times[-1] + 12 < p['frame_count']
        camera_first, camera_last = times[0] - 4, times[-1] + 12
        small = {}
        for j in range(camera_first - 12, camera_last + 1):
            path = fd / f'{j:06d}.jpg'
            image_hashes[str(path)] = sha(path)
            im = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_8)
            small[j] = cv2.resize(im, (160, 90), interpolation=cv2.INTER_AREA)[8:82].astype(np.int16)
        camera_scores = np.zeros(p['frame_count'])
        for j in range(camera_first, camera_last + 1):
            camera_scores[j] = float((np.abs(small[j] - small[j - 12]).mean(2) > 24).mean())
        mask = camera_mask(camera_scores)
        camera_records[cid] = dict(computed_score_first=camera_first, computed_score_last=camera_last,
            valid_mask_first=times[0], valid_mask_last=times[-1],
            scores=[[j, float(camera_scores[j])] for j in range(camera_first, camera_last + 1)],
            mask=[[j, int(mask[j])] for j in times], outside_support='unknown')
        median = np.load(p['median_path'])['median']
        medians = {'global': np.moveaxis(cv2.resize(median, (512, 288)).astype(np.float32) / 255., -1, 0)}
        for arm in pools:
            for i, box in enumerate(TILES):
                medians[arm + str(i)] = np.moveaxis(transform(median, box, arm).astype(np.float32) / 255., -1, 0)

        def get_images(ts):
            images = {}
            for j in sorted({j for t in ts for j in range(t - 3, t + 1)}):
                path = fd / f'{j:06d}.jpg'
                digest = sha(path)
                if str(path) in image_hashes:
                    assert image_hashes[str(path)] == digest
                image_hashes[str(path)] = digest
                images[j] = cv2.imread(str(path))
                assert images[j].shape == (1080, 1920, 3)
            return images

        context = {}
        groups = [times[i:i + 4] for i in range(0, len(times), 4)]
        for ts in groups:
            images = get_images(ts)
            batch = [dict(frames=np.stack([cv2.resize(images[j], (512, 288)) for j in range(t - 3, t + 1)]), median_key='global', row={}) for t in ts]
            with torch.inference_mode(), torch.amp.autocast('cuda'):
                outputs, _, _ = engine.model.forward(frames=tensors(batch, medians, engine.device))
            for t, hm in zip(ts, outputs[:, 0].float().cpu().numpy(), strict=True):
                context[t] = components(hm)
                if str(t) in old_context[cid]:
                    prior = old_context[cid][str(t)]
                    global_diffs.append(dict(candidate_id=cid, local_frame=t, exact_dictionary_match=context[t] == prior,
                        new_component_count=len(context[t]), prior_component_count=len(prior)))
        all_contexts[cid] = context
        baseline = {}
        need = set()
        for t in range(p['review_first'], p['review_last'] + 1):
            c, status = select(context, t, mask, CFG)
            baseline[t] = c, status
            sid = f'{cid}_continuous_{t:06d}'
            if c is None:
                need.add(t)
                for arm in pools:
                    pools[arm][sid] = []
        print(wid, 'global frames', len(context), 'missing targets requiring tile proposals', len(need), flush=True)
        for number, ts in enumerate(groups):
            wanted = set(ts) & need
            if not wanted:
                continue
            images = get_images(ts)
            for arm in pools:
                for i, box in enumerate(TILES):
                    batch = [dict(frames=np.stack([transform(images[j], box, arm) for j in range(t - 3, t + 1)]), median_key=arm + str(i), row={}) for t in ts]
                    with torch.inference_mode(), torch.amp.autocast('cuda'):
                        outputs, _, _ = engine.model.forward(frames=tensors(batch, medians, engine.device))
                    for t, hm in zip(ts, outputs[:, 0].float().cpu().numpy(), strict=True):
                        if t not in wanted:
                            continue
                        sid = f'{cid}_continuous_{t:06d}'
                        cs = components(hm)
                        pools[arm][sid].extend(mapped(cs, i))
                        ix, vals = compact(hm)
                        assert components(expand(ix, vals)) == cs
                        ix_pack.append(ix)
                        val_pack.append(vals)
                        offsets.append(offsets[-1] + len(ix))
                        map_records.append(dict(sample_id=sid, candidate_id=cid, local_frame=t, arm=arm, tile_index=i, batch_frames_json=json.dumps(ts)))
            if number % 5 == 0:
                print(wid, 'tile batch', number + 1, '/', len(groups), flush=True)
        for t in range(p['review_first'], p['review_last'] + 1):
            sid = f'{cid}_continuous_{t:06d}'
            base, status = baseline[t]
            c = None if base else peak_choice(pools['native'][sid])
            score = 0.
            if c:
                x = features(c, pools['native'][sid], pools['degraded'][sid], TILES)
                score = float(predict(fixed_head, x[None])[0])
                feature_rows.append(dict(sample_id=sid, **dict(zip(F, x.tolist())), proposal_gate_score=score))
            added = base is None and c is not None and score >= .5
            actual = base if base is not None else c if added else None
            rows.append(dict(sample_id=sid, window_id=wid, candidate_id=cid, local_frame=t,
                source_frame=p['local_frame_to_source_offset'] + t,
                original_visible=int(base is not None), original_x_px=base['x_px'] if base else '', original_y_px=base['y_px'] if base else '',
                original_confidence=base['bbox_mean'] if base else 0., original_status=status,
                visible=int(actual is not None), x_px=actual['x_px'] if actual else '', y_px=actual['y_px'] if actual else '',
                confidence=actual['bbox_mean'] if actual else 0., output_source='original_temporal' if base else 'added_native_candidate' if added else 'no_output',
                proposal_gate_score=score, proposal_x_px=c['x_px'] if c else '', proposal_y_px=c['y_px'] if c else '',
                camera_flag=int(mask[t]), human_phase_status='not_newly_reviewed', human_ball_status='not_newly_reviewed'))
        print('Finished continuous inference:', wid, flush=True)
    assert len(rows) == 724 and sum(len(c) for c in all_contexts.values()) == 788
    write(OUT / 'continuous_predictions.csv', rows)
    write(OUT / 'candidate_features.csv', feature_rows)
    write(OUT / 'map_records.csv', map_records)
    write(OUT / 'sparse_context_numeric_comparison.csv', global_diffs)
    dump(OUT / 'global_contexts.json', all_contexts)
    dump(OUT / 'camera_features.json', camera_records)
    dump(OUT / 'input_image_hashes.json', image_hashes)
    dump(OUT / 'candidate_pools.json', pools)
    np.savez_compressed(OUT / 'component_rectangles.npz', indices=np.concatenate(ix_pack), values=np.concatenate(val_pack), offsets=np.array(offsets, dtype=np.int64))
    verify_protected()
    assert all(sha(Path(p)) == h for p, h in image_hashes.items())
    artifacts = ['continuous_predictions.csv', 'candidate_features.csv', 'map_records.csv', 'sparse_context_numeric_comparison.csv',
        'global_contexts.json', 'camera_features.json', 'input_image_hashes.json', 'candidate_pools.json', 'component_rectangles.npz', 'protocol.json']
    dump(OUT / 'inference_manifest.json', dict(status='continuous_outputs_frozen_before_human_review', frames=724,
        global_contexts=788, tile_maps=len(map_records), numerical_anchor_components_compared=len(global_diffs),
        numeric_component_mismatches=sum(not r['exact_dictionary_match'] for r in global_diffs),
        artifact_hashes={n: sha(OUT / n) for n in artifacts}, new_human_labels=0, adopted=False))
    print('All724 outputs frozen; unlabelled presence is not accuracy.', flush=True)


def longest_run(rows, key):
    longest = run = 0
    for r in rows:
        run = run + 1 if not int(r[key]) else 0
        longest = max(longest, run)
    return longest


def render():
    protocol = verify_protected()
    manifest = load(OUT / 'inference_manifest.json')
    assert all(sha(OUT / n) == h for n, h in manifest['artifact_hashes'].items())
    assert not (OUT / 'render_report.json').exists(), 'Preserve the reviewed videos.'
    rows = read(OUT / 'continuous_predictions.csv')
    summary, flags, videos = [], [], []
    for plan in protocol['plans']:
        wid = plan['window_id']
        group = [r for r in rows if r['window_id'] == wid]
        assert len(group) == 181
        fps = plan['fps']
        gap = 0
        last = None
        for r in group:
            if not int(r['visible']):
                gap += 1
                continue
            if last is not None:
                distance = math.hypot(float(r['x_px']) - float(last['x_px']), float(r['y_px']) - float(last['y_px']))
                if (gap == 0 and distance > 64) or gap >= 6:
                    flags.append(dict(window_id=wid, local_frame=r['local_frame'], source_frame=r['source_frame'],
                        previous_local_frame=last['local_frame'], no_output_frames_between=gap, distance_px=distance,
                        flag='large_adjacent_displacement' if gap == 0 else 'reappearance_after_no_output_run',
                        human_diagnosis='unreviewed_not_confirmed_switch'))
            last, gap = r, 0
        counts = Counter(r['output_source'] for r in group)
        summary.append(dict(window_id=wid, frames=len(group), source_duration_sec=len(group) / fps,
            original_output_frames=counts['original_temporal'], added_output_frames=counts['added_native_candidate'],
            original_no_output_frames=sum(not int(r['original_visible']) for r in group), no_output_frames=counts['no_output'],
            longest_original_no_output_run_frames=longest_run(group, 'original_visible'),
            longest_no_output_run_frames=longest_run(group, 'visible'),
            longest_no_output_run_duration_sec=longest_run(group, 'visible') / fps,
            center_accuracy='not_computed_unlabelled_continuous_context', phase_scope='not_newly_reviewed'))
        video = OUT / f'{wid}_continuous_review.mp4'
        cmd = [r'C:\ffmpeg\bin\ffmpeg.exe', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', '1920x1180', '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)]
        frames = 0
        with (OUT / f'{wid}_render.log').open('wb') as log:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
            for speed, repetitions in [('1x', 1), ('0.5x', 2)]:
                for r in group:
                    t = int(r['local_frame'])
                    path = Path(plan['frame_dir']) / f'{t:06d}.jpg'
                    assert sha(path) == load_hashes[str(path)]
                    frame = np.zeros((1180, 1920, 3), np.uint8)
                    frame[100:] = cv2.imread(str(path))
                    cv2.putText(frame, f'{wid} | expand{plan["candidate_id"][-3:]} | local {t} | source {r["source_frame"]} | {speed}', (14, 29), 0, .85, (240, 240, 240), 2, cv2.LINE_AA)
                    cv2.putText(frame, 'CYAN: original temporal output', (14, 60), 0, .75, (255, 205, 25), 2, cv2.LINE_AA)
                    cv2.putText(frame, 'GREEN: added candidate', (680, 60), 0, .75, (60, 240, 70), 2, cv2.LINE_AA)
                    state = 'NO OUTPUT (unknown)' if not int(r['visible']) else 'ADDED CANDIDATE' if r['output_source'] == 'added_native_candidate' else 'ORIGINAL OUTPUT'
                    cv2.putText(frame, f'{state} | Unreviewed context: color does not certify ball identity', (14, 90), 0, .70, (220, 220, 220), 1, cv2.LINE_AA)
                    if int(r['visible']):
                        color = (60, 240, 70) if r['output_source'] == 'added_native_candidate' else (255, 205, 25)
                        cv2.circle(frame, (int(float(r['x_px'])), int(float(r['y_px'])) + 100), 12, color, 2, cv2.LINE_AA)
                    for _ in range(repetitions):
                        proc.stdin.write(frame.tobytes())
                        frames += 1
            proc.stdin.close()
            assert proc.wait() == 0
        cap = cv2.VideoCapture(str(video))
        decoded = 0
        while True:
            ok, im = cap.read()
            if not ok:
                break
            assert im.shape == (1180, 1920, 3)
            decoded += 1
        cap.release()
        assert decoded == frames == 543
        videos.append(dict(window_id=wid, path=str(video), sha256=sha(video), frames=decoded,
            fps=fps, duration_sec=decoded / fps, full_source_aspect_preserved=True))
        print('Rendered', wid, '543display frames', flush=True)
    write(OUT / 'output_presence_summary.csv', summary)
    if flags:
        write(OUT / 'review_flags.csv', flags)
    # This comparison checks deployment/batch reproducibility, not a new test set.
    old_base = {r['sample_id']: r for r in read(CHECK / 'frozen_predictions.csv') if r['method'] == 'displacement_0.1'}
    old_added = {r['sample_id']: r for r in read(FIXED / 'frozen_confirmation_predictions.csv')}
    anchor_rows = []
    by_frame = {(r['candidate_id'], int(r['local_frame'])): r for r in rows}
    for sid, old in old_base.items():
        r = by_frame[old['candidate_id'], int(old['local_frame'])]
        old_prop = old_added[sid]
        def same_output(prior, keys):
            present = int(prior['visible']) == int(r[keys[0]])
            return present and (not int(prior['visible']) or all(float(prior[k]) == float(r[new]) for k, new in zip(['x_px', 'y_px', 'confidence'], keys[1:], strict=True)))
        anchor_rows.append(dict(sample_id=sid, window_id=r['window_id'], local_frame=r['local_frame'],
            baseline_exact=same_output(old, ['original_visible', 'original_x_px', 'original_y_px', 'original_confidence']),
            supplemented_exact=same_output(old_prop, ['visible', 'x_px', 'y_px', 'confidence'])))
    write(OUT / 'sparse_anchor_numeric_comparison.csv', anchor_rows)
    verify_protected()
    assert all(sha(OUT / n) == h for n, h in manifest['artifact_hashes'].items())
    dump(OUT / 'render_report.json', dict(status='four_continuous_windows_ready_for_human_review', videos=videos,
        annotated_source_frames=724, display_frames=2172, original_image_wh=[1920, 1080], video_wh=[1920, 1180],
        sparse_anchor_baseline_mismatches=sum(not r['baseline_exact'] for r in anchor_rows),
        sparse_anchor_supplemented_mismatches=sum(not r['supplemented_exact'] for r in anchor_rows),
        numerical_component_mismatches=manifest['numeric_component_mismatches'], review_flags=len(flags),
        center_accuracy_for_continuous_frames=None, human_phase_or_identity_answers_written=0,
        canonical_tracking_changed=False, automatic_adoption=False))
    print(json.dumps(load(OUT / 'render_report.json'), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['run', 'render'])
    args = parser.parse_args()
    if args.stage == 'run':
        run()
    else:
        load_hashes = load(OUT / 'input_image_hashes.json')
        render()
