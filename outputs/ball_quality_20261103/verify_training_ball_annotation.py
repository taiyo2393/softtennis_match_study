"""Read-only HTTP checks plus deliberately invalid POST; never creates labels."""
import hashlib,json,urllib.request,urllib.error
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,sha

out=ROOT/'ball_training_case_expansion'
state=json.loads((out/'annotation_server_state.json').read_text(encoding='utf-8'))
cases=json.loads((out/'annotation_cases.json').read_text(encoding='utf-8'))
manifest=json.loads((out/'annotation_manifest.json').read_text(encoding='utf-8'))
labels=out/'manual_center_labels.csv'
before=sha(labels)
events=out/'annotation_events.jsonl'
events_before=sha(events) if events.exists() else None
def get(route,headers=None):
    with urllib.request.urlopen(urllib.request.Request(state['url']+route,headers=headers or {}),timeout=10) as r:
        return r.status,r.headers,r.read()
assert get('')[0]==200
public=json.loads(get('api/cases')[2])['cases']
assert len(public)==4 and all(not any('prediction' in k for k in c) for c in public)
for case in cases:
    cid=case['case_id'];frame=case['local_frame']
    image=get(f'image/{cid}/{frame}.jpg')
    assert hashlib.sha256(image[2]).hexdigest()==case['source_image_sha256']
    video=get(f'video/{cid}.mp4',{'Range':'bytes=0-1023'})
    assert video[0]==206 and len(video[2])==1024 and video[1]['Content-Range'].startswith('bytes 0-1023/')
bad=urllib.request.Request(state['url']+'api/save',data=json.dumps(dict(case_id='not-a-case',truth_visibility='visible',x=0,y=0)).encode(),headers={'Content-Type':'application/json'})
try:urllib.request.urlopen(bad,timeout=10)
except urllib.error.HTTPError as e:assert e.code==400
else:raise AssertionError('Invalid input accepted')
assert sha(labels)==before and (sha(events) if events.exists() else None)==events_before
assert all(sha(Path(p))==h for p,h in manifest['protected_input_hashes'].items())
report=dict(http_page=200,cases=len(public),prediction_coordinates_hidden=True,exact_image_hashes_checked=4,video_range_checks=4,invalid_post_status=400,labels_unchanged=True,annotation_events_unchanged=True,protected_inputs_unchanged=True)
(out/'http_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report))
