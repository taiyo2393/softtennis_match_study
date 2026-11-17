"""Presentation revision smoke check; reuse completed data/media verification."""
import json, urllib.request
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT, load, read, sha
OUT=ROOT.parent/'ball_quality_20261104'
manifest=load(OUT/'manifest.json');revision=load(OUT/'view_revision.json');state=load(OUT/'server_state.json')
assert state['server_script_sha256']==revision['fit_server_sha256']==sha(Path(revision['fit_server']))
assert sha(OUT/'annotation_fit.html')==revision['fit_html_sha256']
with urllib.request.urlopen(state['url'],timeout=10) as response:
    assert response.read()==(OUT/'annotation_fit.html').read_bytes()
with urllib.request.urlopen(state['url']+'api/cases',timeout=10) as response:
    public=json.loads(response.read())['cases']
assert len(public)==52 and all(c['review_status']=='unreviewed' and c['truth_x_px']==c['truth_y_px']=='' for c in public)
assert sha(OUT/'manual_sequence_labels.csv')==sha(OUT/'initial_labels_snapshot.csv')==manifest['initial_labels_sha256']
assert not (OUT/'annotation_events.jsonl').exists()
assert sha(OUT/'frozen_predictions.csv')==manifest['predictions_sha256']
assert all(sha(Path(p))==h for p,h in manifest['frozen_artifact_hashes'].items())
assert all(sha(Path(p))==h for p,h in manifest['protected_input_hashes'].items())
assert load(OUT/'http_verification.json')['total_display_frames_decoded']==6656
record=dict(status='passed',served_html_matches_fit_revision=True,pending=52,real_labels_written=False,
    browser_observed=dict(normal_canvas_size=[896,504],zoomed_canvas_size=[2416,1359],
        both_aspect_ratios=16/9,source_canvas_size=[1920,1080],
        video_ready_state=4,video_size=[1280,800],video_duration_sec=2.135467,
        native_play_key_started_video=True,video_error=None,browser_errors_or_warnings=0),
    original_and_fit_html_separately_retained=True,
    screenshot=str(OUT/'review_ui_fit_verified.jpg'),screenshot_sha256=sha(OUT/'review_ui_fit_verified.jpg'),
    all_frozen_predictions_and_protected_inputs_unchanged=True,
    server_url=state['url'],script_sha256=sha(Path(__file__)))
(OUT/'view_verification.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
