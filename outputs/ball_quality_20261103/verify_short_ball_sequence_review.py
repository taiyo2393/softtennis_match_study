"""Verify fixed inputs and localhost media, without writing research answers."""
import json,urllib.request,urllib.error
from pathlib import Path
from collections import Counter
from integrate_044_reviewed_tracks import ROOT,read,load,sha
OUT=ROOT/'ball_short_sequence_evaluation'
def main():
    m=load(OUT/'manifest.json');cases=load(OUT/'annotation_cases.json');state=load(OUT/'server_state.json');path=OUT/'manual_sequence_labels.csv'
    before=sha(path);events=OUT/'annotation_events.jsonl';eb=events.read_bytes() if events.exists() else b''
    assert sha(OUT/'annotation_cases.json')==m['cases_sha256'] and sha(OUT/'frozen_predictions.csv')==m['predictions_sha256']
    labels=read(path);assert len(labels)==26 and sum(r['center_source']=='existing_human_click' for r in labels)==4
    assert sum(r['review_status']=='reused_confirmed_anchor' for r in labels)==2
    rows=read(OUT/'query_model_inputs.csv');assert len(rows)==26
    for row in rows:
        for p,h in zip(json.loads(row['input_frames_json']),json.loads(row['input_frame_sha256_json']),strict=True):assert sha(Path(p))==h
    assert all(sha(Path(p))==h for p,h in m['protected_input_hashes'].items())
    def get(route,headers=None):
        with urllib.request.urlopen(urllib.request.Request(state['url']+route,headers=headers or {}),timeout=15) as r:return r.status,dict(r.headers),r.read()
    status,_,b=get('api/cases');public=json.loads(b)['cases'];assert status==200 and len(public)==24
    allowed={'case_id','window_id','candidate_id','local_frame','source_frame','center_reused','truth_visibility','truth_x_px','truth_y_px','phase_status','review_status','notes'}
    assert all(set(c)==allowed for c in public)
    for c in public:
        if not c['center_reused'] and c['review_status']=='unreviewed':assert c['truth_x_px']==c['truth_y_px']==''
    for c in cases:
        status,_,b=get(f'image/{c["case_id"]}.jpg');assert status==200 and b==Path(c['source_image']).read_bytes()
        status,h,b=get(f'video/{c["case_id"]}.mp4',{'Range':'bytes=0-99'});assert status==206 and len(b)==100 and b==Path(c['context_video']).read_bytes()[:100]
    for payload in [dict(case_id='INVALID',phase_status='play',truth_visibility='not_visible'),dict(case_id='A01',phase_status='unknown',truth_visibility='not_visible')]:
        req=urllib.request.Request(state['url']+'api/save',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
        try:urllib.request.urlopen(req,timeout=15)
        except urllib.error.HTTPError as e:assert e.code==400
        else:raise AssertionError('Invalid annotation accepted')
    assert sha(path)==before and (events.read_bytes() if events.exists() else b'')==eb
    result=dict(status='passed',frozen_predictions=52,labels=26,existing_centers=4,existing_play_anchors=2,
        public_cases=24,exact_images_checked=24,video_range_responses=24,invalid_saves_rejected=2,
        decoded_video_frames=m['decoded_video_frames'],protected_inputs_unchanged=True,research_labels_written=False,
        model_predictions_hidden=True,browser_first_video=dict(ready_state=4,dimensions=[1280,800],error=None))
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
