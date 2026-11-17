"""Review the largest adjacent jump and largest reappearance jump, without tuning."""
import json
import subprocess
from pathlib import Path

from integrate_044_reviewed_tracks import ROOT, read, load, write, sha, cv2

OUT = ROOT.parent / 'ball_quality_20261109/continuous_tile_gate_review'


def main():
    assert load(OUT / 'verification.json')['status'] == 'passed'
    assert not (OUT / 'focus_review_manifest.json').exists()
    rows = read(OUT / 'continuous_predictions.csv')
    flags = read(OUT / 'review_flags.csv')
    adjacent = max((r for r in flags if r['flag'] == 'large_adjacent_displacement'), key=lambda r: float(r['distance_px']))
    reacquire = max((r for r in flags if r['flag'] == 'reappearance_after_no_output_run'), key=lambda r: float(r['distance_px']))
    assert (adjacent['window_id'], adjacent['local_frame']) == ('B', '275')
    assert (reacquire['window_id'], reacquire['local_frame']) == ('D', '515')
    selections = [('B', [275]), ('D', [515, 516])]
    plans = load(OUT / 'protocol.json')['plans']
    hashes = load(OUT / 'input_image_hashes.json')
    labels, videos = [], []
    protected = {str(OUT / n): sha(OUT / n) for n in ['protocol.json', 'continuous_predictions.csv', 'review_flags.csv',
        'inference_manifest.json', 'render_report.json', 'verification.json']}
    for wid, targets in selections:
        plan = next(p for p in plans if p['window_id'] == wid)
        group = [r for r in rows if r['window_id'] == wid]
        original_path = OUT / f'{wid}_continuous_review.mp4'
        protected[str(original_path)] = sha(original_path)
        output = OUT / f'{wid}_focus_review.mp4'
        fps = plan['fps']
        cmd = [r'C:\ffmpeg\bin\ffmpeg.exe', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
            '-s', '1920x1180', '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]
        cap = cv2.VideoCapture(str(original_path))
        count = 0
        with (OUT / f'{wid}_focus_render.log').open('wb') as log:
            proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=log)
            for r in group:
                ok, im = cap.read()
                assert ok
                t = int(r['local_frame'])
                # Replace only header speed information, keeping the entire source view and markers.
                cv2.rectangle(im, (0, 0), (1919, 99), (0, 0, 0), -1)
                cv2.putText(im, f'{wid} | local {t} | source {r["source_frame"]} | 0.5x + target hold', (14, 30), 0, .85, (240, 240, 240), 2, cv2.LINE_AA)
                cv2.putText(im, 'CYAN: original output | GREEN: added candidate', (14, 61), 0, .78, (230, 230, 230), 2, cv2.LINE_AA)
                cv2.putText(im, 'YELLOW BORDER = target frame to review, not an error label', (14, 90), 0, .70, (0, 225, 255), 1, cv2.LINE_AA)
                if t in targets:
                    cv2.rectangle(im, (1, 101), (1918, 1178), (0, 225, 255), 3)
                    labels.append(dict(window_id=wid, local_frame=t, source_frame=r['source_frame'],
                        output_source=r['output_source'], x_px=r['x_px'], y_px=r['y_px'],
                        identity_status='unreviewed', review_status='unreviewed', reviewer_id='', reviewed_at_utc='', notes=''))
                for _ in range(62 if t in targets else 2):
                    proc.stdin.write(im.tobytes())
                    count += 1
            cap.release()
            proc.stdin.close()
            assert proc.wait() == 0
        decoded = cv2.VideoCapture(str(output))
        frames = 0
        while True:
            ok, im = decoded.read()
            if not ok:
                break
            assert im.shape == (1180, 1920, 3)
            frames += 1
        decoded.release()
        assert frames == count == 362 + 60 * len(targets)
        videos.append(dict(window_id=wid, path=str(output), sha256=sha(output),
            source_frames=181, display_frames=count, duration_sec=count / fps, held_local_frames=targets,
            predictions_unchanged=True))
    write(OUT / 'focus_identity_labels.csv', labels)
    write(OUT / 'initial_focus_identity_labels.csv', labels)
    assert all(sha(Path(p)) == h for p, h in protected.items())
    manifest = dict(status='target_identity_review_pending', videos=videos, target_frames=3,
        selection='Largest adjacent-output displacement and largest post-gap reappearance displacement among the13predefined review flags. D516shares the selected515candidate and is displayed separately. Error diagnosis is pending,not assumed.',
        ai_visual_note='D515marker appears on spectator area after a14frame output gap; suspect erroneous reacquisition. B275is an existing original output with150pxout-and-back jump. These are AI inspection notes,not human center GT.',
        holds_are_not_new_observations=True, identity_accuracy=None, new_human_answers=0,
        canonical_tracking_changed=False, protected_input_hashes=protected,
        initial_labels_sha256=sha(OUT / 'initial_focus_identity_labels.csv'), script_sha256=sha(Path(__file__)))
    (OUT / 'focus_review_manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
