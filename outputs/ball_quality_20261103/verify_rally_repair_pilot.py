"""Verify real media read-only; test saves/timing only on synthetic temporary data."""
import argparse,json,subprocess,sys,tempfile,time,urllib.request,urllib.error
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,SOURCE,read,load,write,sha,cv2,np
from serve_rally_repair_pilot import self_test,verify_history

OUT=ROOT.parent/'ball_quality_20261109/rally_repair_pilot_044_v2'


def get(url,headers=None):
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers or {}),timeout=15) as r:return r.status,r.read()


def post(url,payload,expected=200):
    request=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
    try:
        with urllib.request.urlopen(request,timeout=15) as r:
            assert r.status==expected;return json.loads(r.read())
    except urllib.error.HTTPError as e:
        assert e.code==expected,(e.code,e.read());return json.loads(e.read())


def synthetic_http():
    with tempfile.TemporaryDirectory(prefix='rally_repair_SYNTHETIC_') as name:
        folder=Path(name);(folder/'serve_review.py').write_bytes((OUT/'serve_review.py').read_bytes())
        image=np.zeros((1080,1920,3),np.uint8);cv2.circle(image,(123,456),5,(255,255,255),-1);cv2.imwrite(str(folder/'synthetic.jpg'),image)
        cases,labels=[],[]
        schema=read(OUT/'initial_labels_snapshot.csv')[0]
        for i in range(201):
            cid=f'R{i+1:03d}';reused=i in [0,39,78,117,156,195]
            cases.append(dict(case_id=cid,local_frame=i,source_frame=10000+i,prediction_visible=int(i!=2),prediction_x_px=123.,prediction_y_px=456.,
                reused_manual_center=reused,source_image=str(folder/'synthetic.jpg'),video_target_time_sec=0.))
            r={k:'' for k in schema};r.update(case_id=cid,local_frame=str(i),source_frame=str(10000+i),review_status='reused_manual_center' if reused else 'unreviewed',
                decision='existing_manual_center' if reused else '',x_px='123' if reused else '',y_px='456' if reused else '',independent_new_center_gt='0')
            labels.append(r)
        write(folder/'review_labels.csv',labels);write(folder/'initial_labels_snapshot.csv',labels)
        (folder/'cases.json').write_text(json.dumps(cases),encoding='utf-8');(folder/'review.html').write_text('Synthetic fixture; no research answers.',encoding='utf-8')
        frozen={n:sha(folder/n) for n in ['initial_labels_snapshot.csv','cases.json','review.html','serve_review.py']}
        (folder/'manifest.json').write_text(json.dumps(dict(protected_input_hashes={},frozen_artifact_hashes=frozen)),encoding='utf-8')
        proc=subprocess.Popen([sys.executable,'-X','utf8',str(folder/'serve_review.py')],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        try:
            for _ in range(100):
                if (folder/'server_state.json').exists():break
                assert proc.poll() is None,proc.stderr.read().decode();time.sleep(.05)
            url=load(folder/'server_state.json')['url'];d=json.loads(get(url+'api/cases')[1]);revision=d['revision']
            client='synthetic_client'
            # A valid-looking answer cannot be stored before timing starts.
            post(url+'api/save',dict(case_id='R002',decision='accept_prediction',client_id=client,revision=revision),400)
            assert not (folder/'review_events.jsonl').exists()
            post(url+'api/timing',dict(client_id=client,action='start'))
            d=post(url+'api/save',dict(case_id='R002',decision='accept_prediction',client_id=client,revision=revision));revision2=d['revision']
            saved=next(c for c in d['cases'] if c['case_id']=='R002');assert saved['x_px']=='123.0' and saved['coordinate_source']=='prediction_explicitly_accepted_by_human'
            post(url+'api/save',dict(case_id='R004',decision='not_visible',client_id=client,revision=revision),409)
            d=post(url+'api/save',dict(case_id='R003',decision='correct_center',x=320,y=240,client_id=client,revision=revision2));revision3=d['revision']
            saved=next(c for c in d['cases'] if c['case_id']=='R003');assert saved['x_px']=='320' and saved['coordinate_source']=='human_click_with_prediction_visible'
            post(url+'api/save',dict(case_id='R001',decision='accept_prediction',client_id=client,revision=revision3),400)
            post(url+'api/timing',dict(client_id=client,action='pause'))
            post(url+'api/save',dict(case_id='R004',decision='not_visible',client_id=client,revision=revision3),400)
            verify_history(folder)
            rows=read(folder/'review_labels.csv');assert sum(r['review_status']=='human_assisted_review' for r in rows)==2
            assert all(r['independent_new_center_gt']=='0' for r in rows)
            assert len((folder/'review_events.jsonl').read_text(encoding='utf-8').splitlines())==2
        finally:
            proc.terminate();proc.wait(timeout=10)
    return dict(synthetic_http_valid_saves=2,synthetic_http_invalid_saves_rejected=4,synthetic_history_replay=True,synthetic_timing_start_pause=True)


def main(http):
    manifest=load(OUT/'manifest.json');before=sha(OUT/'review_labels.csv')
    assert all(sha(Path(p))==h for p,h in manifest['protected_input_hashes'].items())
    assert all(sha(OUT/n)==h for n,h in manifest['frozen_artifact_hashes'].items())
    assert read(OUT/'review_labels.csv')==read(OUT/'initial_labels_snapshot.csv')
    cases=load(OUT/'cases.json');assert len(cases)==201 and sum(c['reused_manual_center'] for c in cases)==6
    assert [c['local_frame'] for c in cases]==list(range(740,1141,2))
    assert all(c['source_frame']==114635+c['local_frame'] for c in cases)
    images=load(OUT/'source_image_hashes.json');assert len(images)==426 and all(sha(Path(p))==h for p,h in images.items())
    gt={int(r['local_frame']):r for r in read(ROOT/'balltrack_manual_finetune_pilot/audit_labels.csv') if r['candidate_id']=='match1_motion_pilot_016'}
    labels={r['case_id']:r for r in read(OUT/'review_labels.csv')}
    for c in cases:
        if c['reused_manual_center']:
            assert labels[c['case_id']]['x_px']==gt[c['local_frame']]['truth_x_px'] and labels[c['case_id']]['y_px']==gt[c['local_frame']]['truth_y_px']
    meta=load(SOURCE/'data/balltrack/racketvision/match1/match1_motion_pilot_016/runs/threshold030_median120/run_metadata.json')
    cap=cv2.VideoCapture(meta['clip']);targets={c['local_frame']:c for c in cases};checked=0
    for t in range(1141):
        ok,im=cap.read();assert ok
        if t in targets:
            encoded,jpg=cv2.imencode('.jpg',im);assert encoded and jpg.tobytes()==Path(targets[t]['source_image']).read_bytes();checked+=1
    cap.release();video=cv2.VideoCapture(str(OUT/'context.mp4'));decoded=0
    while True:
        ok,im=video.read()
        if not ok:break
        assert im.shape==(800,1280,3);decoded+=1
    video.release();assert decoded==426
    for c in cases:assert int(c['video_target_time_sec']*float(meta['fps']))==c['local_frame']-730
    self_test();verify_history(OUT)
    report=dict(status='passed',exact_source_clip_images=checked,context_decoded_frames=decoded,source_image_hashes=426,
        existing_manual_centers_preserved=6,pending_human_targets=195,source_clock_verified=True,synthetic_timer_validation=True,
        prediction_assisted_labels_not_independent_center_gt=True,real_labels_unchanged=True,real_timing_not_started=True,script_sha256=sha(Path(__file__)))
    if http:
        report.update(synthetic_http())
        url=load(OUT/'server_state.json')['url'];assert get(url)[1]==(OUT/'review.html').read_bytes()
        d=json.loads(get(url+'api/cases')[1]);assert len(d['cases'])==201 and not d['timing']['active'] and d['timing']['captured_ms']==0
        assert sum(c['review_status']=='unreviewed' for c in d['cases'])==195
        for c in cases:assert get(url+'image/'+c['case_id']+'.jpg')[1]==Path(c['source_image']).read_bytes()
        data=(OUT/'context.mp4').read_bytes()
        for header,expected in [('bytes=0-99',data[:100]),('bytes=-100',data[-100:])]:
            status,content=get(url+'video/context.mp4',{'Range':header});assert status==206 and content==expected
        for payload,code in [(dict(case_id='R002',decision='accept_prediction',revision='stale',client_id='qa_never_active'),409),
                (dict(case_id='not_real',decision='accept_prediction',revision=d['revision'],client_id='qa_never_active'),400),
                (dict(case_id='R002',decision='accept_prediction',revision=d['revision'],client_id='qa_never_active'),400)]:post(url+'api/save',payload,code)
        report.update(real_http_images_exact=201,real_http_video_ranges=2,real_invalid_saves_rejected=3)
    assert sha(OUT/'review_labels.csv')==before and not (OUT/'review_events.jsonl').exists() and not (OUT/'timing_events.jsonl').exists()
    assert all(sha(Path(p))==h for p,h in manifest['protected_input_hashes'].items())
    (OUT/('http_verification.json' if http else 'verification.json')).write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--http',action='store_true');main(parser.parse_args().http)
