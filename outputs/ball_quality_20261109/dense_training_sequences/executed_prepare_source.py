"""Prepare prediction-blind, training-only sampled sequences; no inferred labels."""
import json
import shutil
import subprocess
from pathlib import Path

from integrate_044_reviewed_tracks import ROOT, SOURCE, read, load, write, sha, cv2, np

BASE = ROOT.parent / 'ball_quality_20261109'
OUT = BASE / 'dense_training_sequences'
PYTHON = SOURCE / 'third_party/racketvision-env/Scripts/python.exe'


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def replace_once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


def make_tools():
    server = (ROOT / 'serve_temporal_guard_packet.py').read_text(encoding='utf-8')
    server = replace_once(server, '"""Local prediction-blind26case annotation with durable journal and revision guard."""',
        '"""Training-only124case review; four existing human centers reused, no predictions."""\nimport sys\nsys.path.insert(0, ' + repr(str(ROOT)) + ')')
    server = replace_once(server, "OUT = ROOT.parent / 'ball_quality_20261109/temporal_guard_next_sequences'", 'OUT = Path(__file__).resolve().parent')
    server = replace_once(server, 'assert len(by_id) == 26', 'assert len(by_id) == 124')
    server = replace_once(server, "'source_frame', 'center_reused']", "'source_frame', 'center_reused', 'context_target_time_sec']")
    server = replace_once(server, "([EF]\\d{2})", "([ABCD]\\d{2})")
    server = replace_once(server, "answer = prepare_answer(payload, set(by_id))", "case = by_id.get(payload.get('case_id'))\n                    if case and case['center_reused']:\n                        payload = dict(payload, truth_visibility='visible', x=case['reused_x'], y=case['reused_y'])\n                    answer = prepare_answer(payload, set(by_id))")
    server = replace_once(server, "row.update(answer, review_status='human_frame_review', center_source='new_human_frame_review')", "row.update(answer, review_status='human_phase_only_existing_center' if case['center_reused'] else 'human_frame_review', center_source='existing_human_click' if case['center_reused'] else 'new_human_frame_review')")
    server = replace_once(server, 'annotation_cases=26,', 'annotation_cases=124,')
    (OUT / 'serve_review.py').write_text(server, encoding='utf-8')
    html = (BASE / 'temporal_guard_next_sequences/annotation.html').read_text(encoding='utf-8')
    html = replace_once(html, '<h1>短い連続区間の球中心を確認</h1>', '<h1>学習用：4場面の球中心を入力</h1>')
    html = replace_once(html, '2場面の各約0.4秒を、2フレームずつ確認します。', '4場面の各約1秒を、2フレーム間隔で確認します（全124時点、既存4点は入力済み）。')
    html = replace_once(html, '全26件は一度に終えなくても保存して再開できます。', '新規120件は一度に終えなくても保存して再開できます。今回は学習用の追加データで、精度評価用ではありません。')
    html = replace_once(html, '<summary>前後のスロー動画を見る（対象で一時停止表示）</summary>', '<summary>前後のスロー動画を見る</summary>')
    html = replace_once(html, '<p class="note">0.5倍速。黄色い枠で停止表示する瞬間が、下の静止画と同じです。</p>', '<p class="note" id="videoInfo"></p><button type="button" id="targetVideo">動画を対象時点へ戻す</button><p class="note">動画は前後の流れの確認用です。球中心は下の元静止画で指定してください。</p>')
    html = replace_once(html, "／ expand${c.candidate_id.slice(-3)}", "／ pilot${c.candidate_id.slice(-3)}")
    html = replace_once(html, "$('context').src=`video/${c.case_id}.mp4`;", "$('context').onloadedmetadata=()=>{if(serial!==id)return;$('context').playbackRate=.5;$('context').currentTime=c.context_target_time_sec};$('context').src=`video/${c.case_id}.mp4`;$('videoInfo').textContent=`0.5倍速。対象は動画の${c.context_target_time_sec.toFixed(3)}秒付近 ／ frame ${c.local_frame}。`;$('targetVideo').onclick=()=>{$('context').pause();$('context').currentTime=c.context_target_time_sec};")
    html = replace_once(html, "if(index<cases.length-1){index++;render()}", "if(cases.slice(index+1).some(c=>!saved(c))){index=index+1+cases.slice(index+1).findIndex(c=>!saved(c));render()}")
    (OUT / 'annotation.html').write_text(html, encoding='utf-8')
    ps = "$taskFolder = Split-Path -Parent $MyInvocation.MyCommand.Path\n$taskStatePath = Join-Path $taskFolder 'server_state.json'\n$taskRunning = $false\nif (Test-Path -LiteralPath $taskStatePath) {\n  $taskState = Get-Content -LiteralPath $taskStatePath -Raw | ConvertFrom-Json\n  try { $taskResponse = Invoke-WebRequest -Uri ($taskState.url + 'api/cases') -TimeoutSec 3; $taskRunning = ($taskResponse.StatusCode -eq 200) } catch {}\n}\nif (-not $taskRunning) {\n  Start-Process -FilePath " + "'" + str(PYTHON) + "'" + " -ArgumentList @('-X','utf8', ('\"' + (Join-Path $taskFolder 'serve_review.py') + '\"')) -WorkingDirectory $taskFolder -WindowStyle Hidden\n  for ($taskAttempt=0; $taskAttempt -lt 60; $taskAttempt++) {\n    Start-Sleep -Milliseconds 500\n    if (Test-Path -LiteralPath $taskStatePath) {\n      $taskState = Get-Content -LiteralPath $taskStatePath -Raw | ConvertFrom-Json\n      try { $taskResponse = Invoke-WebRequest -Uri ($taskState.url + 'api/cases') -TimeoutSec 2; if ($taskResponse.StatusCode -eq 200) { $taskRunning = $true; break } } catch {}\n    }\n  }\n}\nif (-not $taskRunning) { throw 'Review server did not become ready' }\nStart-Process $taskState.url\nWrite-Output $taskState.url\n"
    (OUT / 'start_review.ps1').write_text(ps, encoding='utf-8-sig')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (OUT / 'protocol.json').exists(), 'Never overwrite selection or human labels.'
    labels_path = ROOT / 'balltrack_expanded_centers_pilot/expanded_train_labels.csv'
    phase_path = BASE / 'training_phase_scope_review/phase_labels.csv'
    inventory_path = ROOT / 'play_start_audit/ball_pilot_start_inventory.csv'
    labels = read(labels_path)
    phases = {r['sample_id']: r for r in read(phase_path)}
    inventory = {r['ball_pilot_id']: r for r in read(inventory_path)}
    clips = sorted({r['candidate_id'] for r in labels}, key=lambda c: (int(inventory[c]['clip_first_source_frame']), c))
    assert len(clips) == 8 and all(r['finetune_role'] == 'train' for r in labels)
    # Selection reads only existing human phase/visibility and time; never predictions/errors.
    indices = [int((j + .5) * len(clips) / 4) for j in range(4)]
    selected = [clips[i] for i in indices]
    anchors = []
    for cid in selected:
        available = sorted([r for r in labels if r['candidate_id'] == cid and phases[r['sample_id']]['phase_status'] == 'play' and r['truth_visibility'] == 'visible'], key=lambda r: int(r['local_frame']))
        anchors.append(available[len(available) // 2])
    protected = {str(p): sha(p) for p in [labels_path, phase_path, inventory_path,
        BASE / 'crossfit_candidate_appearance/result.json', BASE / 'crossfit_candidate_appearance/verification.json',
        BASE / 'temporal_guard_next_sequences/completed_review/manual_sequence_labels.csv',
        ROOT.parent / 'ball_quality_20261104/completed_review/manual_sequence_labels.csv',
        ROOT / 'reviewed_ball_overrides.json', ROOT / 'reviewed_person_overrides.json', ROOT / 'reviewed_tracking_bundles.json',
        ROOT / 'serve_temporal_guard_packet.py', ROOT / 'serve_ball_phase_challenge.py',
        BASE / 'temporal_guard_next_sequences/annotation.html']}
    protocol = dict(status='fixed_before_new_image_inspection_or_labels', date_jst='2026-11-09',
        role='training_only_dense_sampled_sequences_never_confirmation',
        selection='Sort8existing training clips by source start; four chronological-strata midpoint indexes [1,3,5,7]. Within each selected clip, take upper median existing human-play/visible anchor sorted by frame. No prediction/error selection.',
        selected_clips=selected, anchor_local_frames=[int(a['local_frame']) for a in anchors],
        targets='Each anchor +/-30source frames, stride2:31targets,span60/59.94~1second;4windows124targets. Unsampled odd frames remain unlabelled.',
        reuse='Exact same-frame existing human ball centers and phase only; no propagation, interpolation or model prefill. Every other target stays unreviewed.',
        annotation='Click observable in-play target-ball center; otherwise mark unobservable/uncertain/nonplay. No invented location from trajectory. Human-not-visible is NOT physical ball absence and NOT an all-zero heatmap target.',
        scope='Valid serve contact through point determination for research main rally; excludes fault/let attempts. Anchor play label does not establish neighbours or precise boundaries.',
        display='Native1920x1080 still images, aspect preserving fit/2x/fullscreen; prediction hidden. One unmarked context video per window, source frame numbers in header; target seek time points inside encoded source frame.',
        downstream='Complete/audit labels before training; keep clips together for splits. This packet alone is not an independent accuracy test; existing confirmation clips never join training.',
        independent_trials=False, no_new_inference=True, no_training=True, automatic_adoption=False,
        clock='legacy_edited_match1_cached_frames_not_redownloaded_video',
        protected_input_hashes=protected, script_sha256=sha(Path(__file__)))
    dump(OUT / 'protocol.json', protocol)
    shutil.copy2(Path(__file__), OUT / 'executed_prepare_source.py')
    known = {(r['candidate_id'], int(r['local_frame'])): r for r in labels}
    cases, rows, queries, windows = [], [], [], []
    image_hashes = {}
    for wid, a in zip('ABCD', anchors, strict=True):
        cid, center = a['candidate_id'], int(a['local_frame'])
        folder = SOURCE / 'data/balltrack/racketvision/match1' / cid
        meta_path = folder / 'runs/threshold030_median120/run_metadata.json'
        meta = load(meta_path)
        fps = float(meta['fps'])
        lo, hi = center - 90, center + 90
        assert lo >= 3 and hi < int(meta['frame_count'])
        offset = int(inventory[cid]['clip_first_source_frame'])
        protected[str(meta_path)] = sha(meta_path)
        video = OUT / f'context_{wid}.mp4'
        command = [r'C:\ffmpeg\bin\ffmpeg.exe', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '1280x800', '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)]
        with (OUT / f'context_{wid}_render.log').open('wb') as log:
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
            for t in range(lo, hi + 1):
                path = folder / 'frames' / f'{t:06d}.jpg'
                image_hashes[str(path)] = sha(path)
                im = cv2.imread(str(path)); assert im.shape == (1080, 1920, 3)
                frame = np.full((800, 1280, 3), 18, np.uint8)
                frame[80:] = cv2.resize(im, (1280, 720), interpolation=cv2.INTER_AREA)
                cv2.putText(frame, f'TRAIN {wid} | pilot{cid[-3:]} | frame {t} | source {offset+t}', (12, 31), 0, .7, (240, 240, 240), 1, cv2.LINE_AA)
                cv2.putText(frame, 'Unmarked source context | use the still image to label the ball', (12, 63), 0, .61, (200, 215, 220), 1, cv2.LINE_AA)
                proc.stdin.write(frame.tobytes())
            proc.stdin.close(); assert proc.wait() == 0
        video_hash = sha(video)
        for number, t in enumerate(range(center - 30, center + 31, 2), 1):
            sid, case_id = f'{cid}_frame_{t:06d}', f'{wid}{number:02d}'
            paths = [folder / 'frames' / f'{j:06d}.jpg' for j in range(t - 3, t + 1)]
            for p in paths:
                image_hashes[str(p)] = sha(p)
            old = known.get((cid, t))
            # All possible same-frame records are preserved; only confirmed-visible-play is auto-complete.
            reused = bool(old and old['truth_visibility'] == 'visible' and phases[old['sample_id']]['phase_status'] == 'play')
            row = dict(case_id=case_id, sample_id=sid, window_id=wid, candidate_id=cid,
                local_frame=t, source_frame=offset + t, phase_status='play' if reused else 'unknown',
                truth_visibility='visible' if reused else 'unreviewed', truth_x_px=old['truth_x_px'] if reused else '', truth_y_px=old['truth_y_px'] if reused else '',
                reviewer_id='human_user' if reused else '', annotated_at_utc=old.get('annotated_at_utc', '') if reused else '',
                review_status='existing_human_center_and_phase' if reused else 'unreviewed',
                center_source='existing_human_click' if reused else 'pending_new_human_click',
                center_source_file=str(labels_path) if reused else '', center_source_file_sha256=sha(labels_path) if reused else '',
                phase_source_file=str(phase_path) if reused else '', phase_source_file_sha256=sha(phase_path) if reused else '',
                finetune_role='train', notes=old.get('notes', '') if reused else '')
            rows.append(row)
            cases.append(dict(case_id=case_id, sample_id=sid, window_id=wid, candidate_id=cid, local_frame=t, source_frame=offset + t,
                source_image=str(paths[-1]), source_image_sha256=sha(paths[-1]), context_video=str(video), context_video_sha256=video_hash,
                context_first_local_frame=lo, context_last_local_frame=hi, context_fps=fps, context_target_time_sec=(t - lo + .25) / fps,
                center_reused=reused, reused_x=row['truth_x_px'], reused_y=row['truth_y_px']))
            median = folder / 'runs/threshold030_median120/median.npz'
            protected[str(median)] = sha(median)
            queries.append(dict(sample_id=sid, case_id=case_id, candidate_id=cid, local_frame=t, source_frame=offset + t,
                finetune_role='train', input_frames_json=json.dumps([str(p) for p in paths]), input_frame_sha256_json=json.dumps([sha(p) for p in paths]),
                median_path=str(median), median_sha256=sha(median), input_ends_at_labelled_frame=1))
        windows.append(dict(window_id=wid, candidate_id=cid, anchor_local_frame=center, first_target=center-30, last_target=center+30, stride=2,
            target_count=31, span_sec=60/fps, fps=fps, first_source_frame=offset, context_video=str(video), context_frames=hi-lo+1))
    assert len(rows) == 124 and sum(c['center_reused'] for c in cases) == 4
    # Whole source clips already have training role; no held confirmation clip is imported.
    confirmation_ids = {r['candidate_id'] for r in read(BASE / 'temporal_guard_next_sequences/completed_review/manual_sequence_labels.csv') + read(ROOT.parent / 'ball_quality_20261104/completed_review/manual_sequence_labels.csv')}
    assert set(selected).isdisjoint(confirmation_ids)
    write(OUT / 'manual_sequence_labels.csv', rows)
    shutil.copy2(OUT / 'manual_sequence_labels.csv', OUT / 'initial_labels_snapshot.csv')
    write(OUT / 'query_model_inputs.csv', queries)
    dump(OUT / 'annotation_cases.json', cases)
    dump(OUT / 'windows.json', windows)
    dump(OUT / 'source_image_hashes.json', image_hashes)
    make_tools()
    assert all(sha(Path(p)) == h for p, h in protected.items())
    manifest = dict(status='training_only_annotation_ready', targets=124, existing_human_targets=4, pending_new_human_targets=120,
        windows=4, sampled_stride=2, context_videos=4, model_inference=False, new_training=False, predictions_prefilled=False,
        human_phase_not_propagated=True, protected_input_hashes=protected,
        frozen_artifact_hashes={n: sha(OUT / n) for n in ['protocol.json', 'executed_prepare_source.py', 'initial_labels_snapshot.csv', 'query_model_inputs.csv',
            'annotation_cases.json', 'windows.json', 'source_image_hashes.json', 'serve_review.py', 'annotation.html', 'start_review.ps1']})
    dump(OUT / 'manifest.json', manifest)
    print(json.dumps({k: v for k, v in manifest.items() if k not in ['protected_input_hashes', 'frozen_artifact_hashes']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
