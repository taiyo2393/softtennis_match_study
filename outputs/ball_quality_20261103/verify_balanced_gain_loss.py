"""Verify saved replay evidence; repair figure spacing without new inference."""
import json, math, textwrap
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from integrate_044_reviewed_tracks import ROOT, load, read, sha, cv2
from compare_ball_candidate_decoders import components
from diagnose_balanced_gain_loss import draw_gt, peak_record

OUT = ROOT / 'ball_balanced_gain_loss_diagnosis'
result = load(OUT / 'result.json')
protocol = load(OUT / 'protocol.json')
script = ROOT / 'diagnose_balanced_gain_loss.py'
assert sha(script) == protocol['script_sha256']
protected = dict(result['protected_input_hashes'])
protected.update(load(OUT / 'source_image_hashes.json'))
protected[str(OUT / 'result.json')] = sha(OUT / 'result.json')
protected[str(OUT / 'protocol.json')] = sha(OUT / 'protocol.json')
for name in ['video', 'figure']:
    p = result[name]
    assert sha(Path(p['path'])) == p['sha256']
    protected[p['path']] = p['sha256']
cases = load(OUT / 'cases.json')
assert len(cases) == 2
with np.load(OUT / 'replayed_heatmaps.npz') as saved:
    maps = {(str(c), str(m), int(t)): h.astype(np.float32)
            for c, m, t, h in zip(saved['case_ids'], saved['model_names'],
                                  saved['local_frames'], saved['heatmaps'], strict=True)}
assert len(maps) == 36
measure = []
rows = read(OUT / 'response_probes.csv')
assert len(rows) == len(maps)
for c in cases:
    for model, base in [('original', ROOT / 'ball_temporal_path_comparison'),
                        ('balanced', ROOT / 'ball_balanced_fixed_validation')]:
        cache = load(base / (c['role'] + '_component_cache.json'))[c['candidate_id']]
        for t in range(c['target_local_frame'] - 8, c['target_local_frame'] + 9, 2):
            hm = maps[c['case_id'], model, t]
            assert components(hm) == cache[str(t)]
            rec = peak_record(hm, c['manual_x'], c['manual_y'])
            row = next(r for r in rows if r['case_id'] == c['case_id'] and
                       r['model'] == model and int(r['local_frame']) == t)
            for key, value in rec.items():
                assert float(row[key]) == value, (key, row[key], value)
            measure.append(dict(case_id=c['case_id'], model=model, local_frame=t,
                                is_target=t == c['target_local_frame'], **rec))
        target = c['target_local_frame']
        output = c['old_output' if model == 'original' else 'new_output']
        cs = components(maps[c['case_id'], model, target])
        if int(output['visible']):
            assert len(cs) == 1
            assert (float(output['x_px']), float(output['y_px'])) == (cs[0]['x_px'], cs[0]['y_px'])
            assert math.hypot(float(output['x_px']) - c['manual_x'],
                              float(output['y_px']) - c['manual_y']) <= 16
        else:
            assert not cs

# Reuse exactly the original plotting logic with layout-only substitutions.
# Keep original PNG/result/protocol unchanged for provenance.
source = script.read_text(encoding='utf-8')
block = source[source.index('    font_path='):source.index('    all_frames=[]')]
block = textwrap.dedent(block).replace('(1200,1020)', '(1200,1100)')
block = block.replace('row*450', 'row*480').replace('(18,962)', '(18,1040)')
block = block.replace(',978)', ',1056)').replace(',958,', ',1036,').replace(',977)', ',1055)')
block = block.replace("OUT/'paired_evidence.png'", "OUT/'paired_evidence_v2.png'")
exec(compile(block, str(script) + ':layout_only', 'exec'))
assert Image.open(OUT / 'paired_evidence_v2.png').size == (1200, 1100)

cap = cv2.VideoCapture(result['video']['path'])
fps = cap.get(cv2.CAP_PROP_FPS)
decoded = 0
for_display = []
while True:
    ok, frame = cap.read()
    if not ok:
        break
    assert frame.shape == (800, 1280, 3)
    if decoded in [48, 176]:
        path = OUT / ('video_target_' + str(decoded) + '.png')
        assert cv2.imwrite(str(path), frame)
        for_display.append(str(path))
    decoded += 1
cap.release()
assert decoded == result['video']['display_frames'] == 256
assert abs(decoded / fps - result['video']['duration_sec']) < .001
assert all(sha(Path(p)) == h for p, h in protected.items())
verification = dict(status='passed', saved_maps_matching_candidates=36,
                    probe_records_reproduced=36, target_outputs_reproduced=4,
                    full_video_frames_decoded=decoded, video_fps=fps,
                    target_video_previews=for_display,
                    corrected_figure=str(OUT / 'paired_evidence_v2.png'),
                    corrected_figure_sha256=sha(OUT / 'paired_evidence_v2.png'),
                    figure_fix='Spacing only: second title and colorbar no longer overlap panels; original retained.',
                    protected_files_unchanged=len(protected), new_inference=False,
                    new_labels=False, operational_changes=False,
                    script_sha256=sha(Path(__file__)))
(OUT / 'verification.json').write_text(json.dumps(verification, indent=2), encoding='utf-8')
print(json.dumps(verification, indent=2))
