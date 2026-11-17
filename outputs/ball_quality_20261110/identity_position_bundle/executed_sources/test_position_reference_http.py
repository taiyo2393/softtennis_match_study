"""Exercise save/reload against isolated synthetic data, never manual references."""
import json,threading,urllib.request,urllib.error,shutil
from pathlib import Path
from run_fixed_cohort_player_case_v3 import BASE,dump
from serve_position_reference_pilot import make_server

OUT=BASE/'identity_position_bundle/server_http_qa'

def request(base,path='',payload=None,expected=200):
    data=json.dumps(payload).encode() if payload is not None else None;req=urllib.request.Request(base+path,data=data,headers={'Content-Type':'application/json'} if data else {})
    try:
        with urllib.request.urlopen(req,timeout=3) as r:status=r.status;body=r.read();mime=r.headers.get('Content-Type','')
    except urllib.error.HTTPError as e:status=e.code;body=e.read();mime=e.headers.get('Content-Type','')
    assert status==expected,(path,status,expected)
    return json.loads(body) if 'application/json' in mime else body

def main():
    if OUT.exists():assert not (OUT/'verification.json').exists() and not (OUT/'human_reference_labels.json').exists(),'Only an untouched setup from a failed connection may be reused.'
    OUT.mkdir(exist_ok=True);pilot=BASE/'identity_position_bundle/reference_pilot';assert not (pilot/'human_reference_labels.json').exists(),'Human work already exists; do not use any real labels for QA.'
    dump(OUT/'public_manifest.json',dict(title='Synthetic test fixture',items=[dict(image_id='P001',image_file='P001.png',width=3840,height=2160)],slots=['near_left','near_right','far_left','far_right']));(OUT/'images').mkdir(exist_ok=True);shutil.copy2(pilot/'images/P001.png',OUT/'images/P001.png');(OUT/'index.html').write_text('<html>isolated synthetic test</html>',encoding='utf-8')
    server=make_server(OUT,'fixture-token','automated_qa_fixture');thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();base=f'http://127.0.0.1:{server.server_port}/fixture-token/'
    try:
        assert request(base,'api/session')['annotations']['labels']=={}
        assert request(base,'images/P001.png')==(OUT/'images/P001.png').read_bytes()
        for path in ['private_sampling_manifest.json','../private_sampling_manifest.json','images/../private_sampling_manifest.json']:request(base,path,expected=404)
        request(base.replace('/fixture-token/','/wrong-token/'),'api/session',expected=404)
        assert request(base,'api/label',dict(image_id='P001',slot='near_left',entry=dict(state='point',x=123.4,y=456.7)))['revision']==1
        assert request(base,'api/label',dict(image_id='P001',slot='far_left',entry=dict(state='unplaceable')))['revision']==2
        request(base,'api/label',dict(image_id='P001',slot='near_left',entry=dict(state='point',x=-1,y=456)),expected=400)
        request(base,'api/label',dict(image_id='../other',slot='near_left',entry=dict(state='absent')),expected=400)
        state=request(base,'api/session')['annotations'];assert state['revision']==2 and state['label_origin']=='automated_qa_fixture';assert state['labels']['P001']['near_left']['x']==123.4 and state['labels']['P001']['far_left']['state']=='unplaceable'
    finally:server.shutdown();server.server_close();thread.join(timeout=3)
    reopened=make_server(OUT,'reopened-token','automated_qa_fixture');thread=threading.Thread(target=reopened.serve_forever,daemon=True);thread.start();base=f'http://127.0.0.1:{reopened.server_port}/reopened-token/'
    try:
        state=request(base,'api/session')['annotations'];assert state['revision']==2 and state['labels']['P001']['far_left']['state']=='unplaceable'
        assert request(base,'api/label',dict(image_id='P001',slot='near_left',entry=dict(state='pending')))['revision']==3
    finally:reopened.shutdown();reopened.server_close();thread.join(timeout=3)
    assert not (pilot/'human_reference_labels.json').exists();dump(OUT/'verification.json',dict(status='passed',fixture_label_origin='automated_qa_fixture',checks=['empty_session','PNG_asset_bytes','private_route_rejection','wrong_token_rejection','point_save','unplaceable_save','invalid_coordinates','unknown_image','revision_preservation','server_restart_resume','undo_pending'],real_manual_reference_file_created=False));print('Passed11 HTTP save/reload fixtures; real manual references untouched.')

if __name__=='__main__':main()
