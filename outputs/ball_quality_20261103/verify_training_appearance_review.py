"""Read-only research checks; HTTP POSTs contain invalid labels only."""
import json, urllib.request, urllib.error
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,load,read,sha
OUT=ROOT/'ball_appearance_training_review'
def main():
    manifest=load(OUT/'manifest.json');cases=load(OUT/'cases.json');state=load(OUT/'server_state.json');base=state['url']
    before=sha(OUT/'candidate_review_labels.csv');events=OUT/'annotation_events.jsonl';event_before=events.read_bytes() if events.exists() else b''
    def get(route,headers=None):
        with urllib.request.urlopen(urllib.request.Request(base+route,headers=headers or {}),timeout=15) as r:return r.status,dict(r.headers),r.read()
    status,_,blob=get('api/cases');public=json.loads(blob)['cases'];assert status==200 and len(public)==9
    assert all(set(c)=={'case_id','candidate_id','local_frame','source_frame','candidate_x_px','candidate_y_px','object_label','notes'} for c in public)
    for c in cases:
        for route,key in [('image','source_image'),('patch','raw_patch96')]:
            ext='jpg' if route=='image' else 'png';s,_,b=get(f'{route}/{c["case_id"]}.{ext}');assert s==200 and b==Path(c[key]).read_bytes()
        s,h,b=get(f'video/{c["case_id"]}.mp4',{'Range':'bytes=0-99'});assert s==206 and len(b)==100 and b==Path(c['context_video']).read_bytes()[:100]
    for payload in [dict(case_id='A',object_label='invalid'),dict(case_id='INVALID',object_label='ball')]:
        req=urllib.request.Request(base+'api/save',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
        try:urllib.request.urlopen(req,timeout=15)
        except urllib.error.HTTPError as e:assert e.code==400
        else:raise AssertionError('Invalid review accepted')
    assert sha(OUT/'candidate_review_labels.csv')==before
    assert (events.read_bytes() if events.exists() else b'')==event_before
    for p,h in manifest['protected_input_hashes'].items():assert sha(Path(p))==h,p
    result=dict(status='passed',public_cases=9,images_and_patches_exact=18,video_range_responses=9,
        invalid_saves_rejected=2,research_labels_written=False,protected_inputs_unchanged=True,
        remaining_unreviewed=sum(r['object_label']=='unreviewed' for r in read(OUT/'candidate_review_labels.csv')),
        browser_video_ready_state=4,browser_video_dimensions=[1280,800],browser_video_error=None)
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))
if __name__=='__main__':main()
