"""Freeze a provisional rally window, render raw review, and run the ball baseline."""
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

import build_044_spatiotemporal_pilot as common
from prepare_match2_ball_first_point import OUT


SERVE_BRACKET = (12324, 12326)
END_BRACKET = (12570, 12578)
CORE = range(SERVE_BRACKET[1] + 1, END_BRACKET[0])
REVIEW_TIMES = range(12262, 12628, 2)


def make_review_video():
    fps = '15000/1001'
    command = ['C:/ffmpeg/bin/ffmpeg.exe', '-hide_banner', '-loglevel', 'error', '-n',
               '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '1280x800', '-r', fps, '-i', 'pipe:0',
               '-an', '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
               str(OUT / 'match2_first_point_raw_review.mp4')]
    font = ImageFont.truetype(str(common.FONT), 22)
    frame_map = []
    with subprocess.Popen(command, stdin=subprocess.PIPE) as process:
        for index, frame in enumerate(REVIEW_TIMES):
            canvas = Image.new('RGB', (1280, 800), '#101820')
            with Image.open(OUT / 'frames' / f'{frame}.jpg') as source:
                canvas.paste(source.resize((1280, 720), Image.Resampling.LANCZOS), (0, 80))
            draw = ImageDraw.Draw(canvas)
            draw.text((16, 9), 'match2 第1点　サーブと終端の確認用　0.5倍速', font=font, fill='white')
            if frame < SERVE_BRACKET[0]:
                event = 'サーブ前'
            elif frame <= SERVE_BRACKET[1]:
                event = 'サーブ接触候補'
            elif frame < END_BRACKET[0]:
                event = '追跡を比較する区間'
            elif frame <= END_BRACKET[1]:
                event = '終端候補'
            else:
                event = '得点表示の更新を確認する区間'
            draw.text((16, 45), f'元フレーム F{frame}　{frame/common.FPS:.3f}秒　{event}', font=font, fill='#cfdfed')
            if frame in [12326, 12574]:
                canvas.save(OUT / f'raw_review_F{frame}.jpg', quality=95)
            process.stdin.write(np.asarray(canvas).tobytes())
            frame_map.append(dict(display_frame=index, source_frame=frame))
        process.stdin.close()
        assert process.wait() == 0
    common.write(OUT / 'review_video_source_frames.csv', frame_map)
    capture = cv2.VideoCapture(str(OUT / 'match2_first_point_raw_review.mp4'))
    decoded = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            assert frame.shape == (800, 1280, 3)
            decoded += 1
    finally:
        capture.release()
    assert decoded == len(frame_map)
    return decoded


def run_baseline(weight, model_root, median_frames, *, output_dir=OUT, core=CORE):
    sys.path.insert(0, str(model_root))
    from inference import BallInferencer

    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    detector = BallInferencer(str(model_root / 'configs/tracknetv3_base.py'), str(weight),
                             device='cuda', thre=.3, batchsize=4)
    assert detector.device.type == 'cuda'
    detector.model.eval().requires_grad_(False)

    def read(frame):
        image = cv2.imread(str(output_dir / 'frames' / f'{frame}.jpg'))
        assert image is not None
        return image

    # Calculate the same native pixelwise median as the match1 baseline.
    samples = np.stack([read(frame) for frame in median_frames])
    median = np.empty((1080, 1920, 3), dtype=np.uint8)
    for top in range(0, 1080, 60):
        median[top:top+60] = np.median(samples[:, top:top+60], axis=0).astype(np.uint8)
    del samples
    np.savez_compressed(output_dir / 'median.npz', median=median)
    resized = {frame: cv2.resize(read(frame), (512, 288)) for frame in range(core.start - 3, core.stop)}
    med = np.moveaxis(cv2.resize(median, (512, 288)).astype(np.float32) / 255., -1, 0)
    predictions = []
    for start in range(core.start, core.stop, 4):
        frames = list(range(start, min(start + 4, core.stop)))
        inputs = []
        for frame in frames:
            context = np.stack([resized[f] for f in range(frame - 3, frame + 1)])
            context = np.moveaxis(context.astype(np.float32) / 255., -1, 1)
            inputs.append(np.concatenate([med[None], context], axis=0).reshape(15, 288, 512))
        inputs = torch.from_numpy(np.stack(inputs)).to(detector.device)
        with torch.inference_mode(), torch.amp.autocast('cuda'):
            output, _, _ = detector.model.forward(frames=inputs)
        for frame, heatmap in zip(frames, output[:, 0].float().cpu().numpy(), strict=True):
            x, y, w, h, confidence = detector._predict_location(heatmap)
            px, py = int((x+w/2)*1920/512), int((y+h/2)*1080/288)
            present = bool(px or py)
            predictions.append(dict(source_frame=frame, source_seconds=frame/common.FPS,
                                    candidate_present=int(present), x_px=px if present else '',
                                    y_px=py if present else '', confidence=float(confidence),
                                    input_first_source_frame=frame-3, input_last_source_frame=frame,
                                    review_status='unreviewed'))
    assert [r['source_frame'] for r in predictions] == list(core)
    common.write(output_dir / 'baseline_unreviewed_candidates.csv', predictions)
    return sum(r['candidate_present'] for r in predictions)


def main():
    assert not (OUT / 'protocol.json').exists()
    model_root = common.SOURCE / 'third_party/RacketVision/source/BallTrack'
    weight = model_root / 'checkpoints/balltrack_best.pth'
    prior_trial = common.HERE / '056_temporal_reacquisition_trial/protocol.json'
    prior_weight_hash = common.load(prior_trial)['protected_input_hashes'][str(weight)]
    assert common.sha(weight) == prior_weight_hash
    median_frames = np.rint(np.linspace(12288, 12576, 120)).astype(int).tolist()
    reference_frames = np.rint(np.linspace(CORE.start, CORE.stop-1, 12)).astype(int).tolist()
    common.dump(OUT / 'protocol.json', dict(
        frozen_before_ball_outputs=True,
        sample='First point of match2, previously examined for camera and person tracking.',
        source_selection='Chronological first point chosen from raw images and score display.',
        provisional_serve_contact=list(SERVE_BRACKET), provisional_termination=list(END_BRACKET),
        core_inclusive=[CORE.start, CORE.stop-1],
        event_evidence='Serve motion and separated ball in raw crop; descending ball and apparent bounce near baseline before score update. Event intervals are AI visual estimates; exact event and outcome type remain open.',
        reference_frames_selected_before_inference=reference_frames,
        reference_labeling='Raw frames first. Record visible ball center, not visible, or uncertain. Compare baseline only after references are saved.',
        detector='Same checkpoint, full view 512x288, BGR /255, background median and t-3..t, CUDA AMP, original largest component threshold0.3.',
        weight_sha256=prior_weight_hash, median_frame_scope=[12288,12576], median_frames=median_frames,
        planned_comparison='Raw baseline vs. separately reviewed assistance. Report ball identity errors, missing visible balls, and annotation workload. Use the frozen 0.1 / 32px / 12-frame method only for eligible bracketed gaps once reviewed anchors exist.',
    ))
    video_frames = make_review_video()
    reference = OUT / 'raw_reference_frames'
    reference.mkdir(exist_ok=False)
    for frame in reference_frames:
        with Image.open(OUT / 'frames' / f'{frame}.jpg') as image:
            image.save(reference / f'F{frame}.png')
    common.write(OUT / 'reference_annotation_template.csv', [dict(
        source_frame=frame, image=f'raw_reference_frames/F{frame}.png', visibility='', x_px='', y_px='')
        for frame in reference_frames])
    candidates = run_baseline(weight, model_root, median_frames)
    source = common.load(OUT / 'source_manifest.json')
    assert all(common.sha(OUT / 'frames' / name) == digest for name, digest in source['cached_image_sha256'].items())
    assert common.sha(weight) == prior_weight_hash
    common.dump(OUT / 'verification.json', dict(
        source_cache_frames=len(source['cached_image_sha256']), raw_review_video_frames_decoded=video_frames,
        baseline_target_frames=len(CORE), frames_with_candidate_output=candidates,
        baseline_target_frame_range=[CORE.start, CORE.stop-1], reference_frames=len(reference_frames),
        independent_ball_labels_available=0, frozen_match1_weight_verified=True,
        source_cache_unchanged=True, publication_date='2026-11-11'))
    (OUT / 'README.md').write_text(
        '# match2 第1点の球追跡試走\n\n'
        '元映像の所見から、サーブ接触候補をF12324〜12326、終端候補をF12570〜12578に置いた。'
        '点の確定時刻と終端の種類は要確認。境界の不確かさを考慮し、基準追跡はF12327〜12569の243フレームで実行した。\n\n'
        'match1で使用したTrackNetの重みと抽出閾値を使用した。背景画像はmatch2の主画面から作成した。'
        '先に固定した12時点の原画像と参照ラベル用の表を保存した。候補出力は未照合。\n\n'
        'この第1点は、既に別試合のカメラ対応と人物追跡の試走に使った区間。'
        '今回は同じ球検出器を別試合に移した小規模試走として扱う。\n', encoding='utf-8')
    common.dump(OUT / 'publication_hashes.json', {p.name: common.sha(p) for p in sorted(OUT.iterdir()) if p.is_file()})
    print(common.load(OUT / 'verification.json'), flush=True)


if __name__ == '__main__':
    main()
