"""HTTP/media/integrity checks, without creating research annotations."""
import hashlib,json,urllib.request,urllib.error
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,load,sha,read

out=ROOT/'ball_inplay_challenge';state=load(out/'annotation_server_state.json');cases=load(out/'annotation_cases.json')
manifest=load(out/'annotation_manifest.json');before=sha(out/'manual_center_labels.csv')
events=out/'annotation_events.jsonl';events_before=sha(events) if events.exists() else None
def get(route,headers=None):
    with urllib.request.urlopen(urllib.request.Request(state['url']+route,headers=headers or {}),timeout=10) as r:return r.status,r.headers,r.read()
page=get('');assert page[0]==200 and '対象フレームはプレー中ですか'.encode() in page[2]
public=json.loads(get('api/cases')[2])['cases'];assert len(public)==4
assert all(not any(k in r for k in ['prediction_x_px','ranking_score','query_type','method']) for r in public)
for c in cases:
    r=get(f'image/{c["case_id"]}/{c["local_frame"]}.jpg');assert hashlib.sha256(r[2]).hexdigest()==c['source_image_sha256']
    r=get(f'video/{c["case_id"]}.mp4',{'Range':'bytes=0-1023'});assert r[0]==206 and len(r[2])==1024
for payload in [dict(case_id='A',truth_visibility='visible',x=0,y=0),dict(case_id='unknown',truth_visibility='visible',phase_status='play',x=0,y=0)]:
    request=urllib.request.Request(state['url']+'api/save',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'})
    try:urllib.request.urlopen(request,timeout=10)
    except urllib.error.HTTPError as error:assert error.code==400
    else:raise AssertionError('Missing phase or invalid ID accepted')
assert sha(out/'manual_center_labels.csv')==before and (sha(events) if events.exists() else None)==events_before
assert sha(out/'frozen_method_predictions.csv')==manifest['frozen_predictions_sha256']
assert all(sha(Path(p))==h for p,h in manifest['protected_input_hashes'].items())
report=dict(page_http=200,exact_image_hashes_checked=4,video_range_checks=4,predictions_hidden=True,
    missing_phase_rejected=True,invalid_case_rejected=True,labels_and_events_unchanged=True,frozen_method_predictions_unchanged=True,
    protected_inputs_unchanged=True,decoded_video_frames=sum(c['decoded_video_frames'] for c in cases),
    completed_rows=sum(r['truth_visibility']!='unreviewed' for r in read(out/'manual_center_labels.csv')))
(out/'http_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
