"""Replay056original-model inference;synthetic writes only,real review stays empty."""
import argparse,importlib.util,json,subprocess,sys,tempfile,time
from pathlib import Path
import torch
from integrate_044_reviewed_tracks import ROOT,SOURCE,read,load,write,sha,cv2,np
from prepare_rally_repair_056 import stream_sha,clock_check,FIRST,LAST,TARGETS
from verify_rally_repair_pilot import get,post
from run_manual_ball_finetune import tensors
from calibrate_point_supervised_detector import expand

OUT=ROOT.parent/'ball_quality_20261109/rally_repair_056'


def synthetic_http():
    with tempfile.TemporaryDirectory(prefix='repair056_SYNTHETIC_') as name:
        folder=Path(name);(folder/'serve_review.py').write_bytes((OUT/'serve_review.py').read_bytes())
        image=np.zeros((1080,1920,3),np.uint8);cv2.circle(image,(123,456),5,(255,255,255),-1);cv2.imwrite(str(folder/'synthetic.jpg'),image)
        schema=read(OUT/'initial_labels_snapshot.csv')[0];cases=[];labels=[]
        for i in range(376):
            cid=f'R{i+1:03d}';reused=i==0
            cases.append(dict(case_id=cid,local_frame=i,source_frame=20000+i,prediction_visible=int(i!=2),prediction_x_px=123.,prediction_y_px=456.,reused_manual_center=reused,source_image=str(folder/'synthetic.jpg'),video_target_time_sec=0.))
            r={k:'' for k in schema};r.update(case_id=cid,local_frame=str(i),source_frame=str(20000+i),review_status='reused_manual_center' if reused else 'unreviewed',decision='existing_manual_center' if reused else '',x_px='123' if reused else '',y_px='456' if reused else '',independent_new_center_gt='0');labels.append(r)
        write(folder/'review_labels.csv',labels);write(folder/'initial_labels_snapshot.csv',labels)
        (folder/'cases.json').write_text(json.dumps(cases),encoding='utf-8');(folder/'review.html').write_text('Synthetic test only',encoding='utf-8')
        frozen={n:sha(folder/n) for n in ['initial_labels_snapshot.csv','cases.json','review.html','serve_review.py']}
        (folder/'manifest.json').write_text(json.dumps(dict(protected_input_hashes={},frozen_artifact_hashes=frozen)),encoding='utf-8')
        proc=subprocess.Popen([sys.executable,'-X','utf8',str(folder/'serve_review.py')],stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
        try:
            for _ in range(100):
                if (folder/'server_state.json').exists():break
                assert proc.poll() is None,proc.stderr.read().decode();time.sleep(.05)
            url=load(folder/'server_state.json')['url'];d=json.loads(get(url+'api/cases')[1]);rev=d['revision'];client='synthetic056_client'
            post(url+'api/save',dict(case_id='R002',decision='accept_prediction',client_id=client,revision=rev),400)
            post(url+'api/timing',dict(client_id=client,action='start'))
            d=post(url+'api/save',dict(case_id='R002',decision='accept_prediction',client_id=client,revision=rev));rev2=d['revision']
            post(url+'api/save',dict(case_id='R004',decision='not_visible',client_id=client,revision=rev),409)
            d=post(url+'api/save',dict(case_id='R003',decision='correct_center',x=320,y=240,client_id=client,revision=rev2));rev3=d['revision']
            post(url+'api/save',dict(case_id='R001',decision='accept_prediction',client_id=client,revision=rev3),400)
            post(url+'api/timing',dict(client_id=client,action='pause'))
            post(url+'api/save',dict(case_id='R004',decision='not_visible',client_id=client,revision=rev3),400)
            spec=importlib.util.spec_from_file_location('fixture056',folder/'serve_review.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.verify_history(folder)
            rows=read(folder/'review_labels.csv');a,b=rows[1:3]
            assert a['coordinate_source']=='prediction_explicitly_accepted_by_human' and a['x_px']=='123.0'
            assert b['coordinate_source']=='human_click_with_prediction_visible' and b['x_px']=='320'
            assert sum(r['review_status']=='human_assisted_review' for r in rows)==2 and all(r['independent_new_center_gt']=='0' for r in rows)
        finally:proc.terminate();proc.wait(timeout=10)
    return dict(synthetic_http_valid_saves=2,synthetic_invalid_saves_rejected=4,synthetic_history_exact=True,synthetic_timing_start_pause=True)


def main(http):
    protocol=load(OUT/'protocol.json');manifest=load(OUT/'manifest.json');before=sha(OUT/'review_labels.csv')
    assert sha(OUT/'executed_prepare_source.py')==protocol['script_sha256']
    assert all(sha(Path(p))==h for p,h in manifest['protected_input_hashes'].items())
    assert all(sha(OUT/n)==h for n,h in manifest['frozen_artifact_hashes'].items())
    weight=SOURCE/'third_party/RacketVision/source/BallTrack/checkpoints/balltrack_best.pth'
    assert sha(weight)=='00d707b9db7a49561c411e4765956e79bcd7c7e20c7a0a535073440b3e972342'
    source=Path(protocol['source_video']);assert stream_sha(source)==protocol['source_video_sha256']
    assert clock_check(source)==load(OUT/'source_clock_check.json')
    cases=load(OUT/'cases.json');labels=read(OUT/'review_labels.csv')
    assert labels==read(OUT/'initial_labels_snapshot.csv') and len(cases)==len(labels)==376
    assert [c['source_frame'] for c in cases]==TARGETS and all(not c['reused_manual_center'] for c in cases)
    assert all(r['review_status']=='unreviewed' and r['x_px']==r['y_px']=='' for r in labels)
    hashes=load(OUT/'source_image_hashes.json');assert len(hashes)==803 and all(sha(Path(p))==h for p,h in hashes.items())
    cap=cv2.VideoCapture(str(source));cap.set(1,FIRST);fd=OUT/'frames'
    for local in range(803):
        ok,im=cap.read();assert ok and im.shape==(2160,3840,3)
        im=cv2.resize(im,(1920,1080),interpolation=cv2.INTER_AREA);ok,jpg=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok and jpg.tobytes()==(fd/f'{local:06d}.jpg').read_bytes()
    cap.release()
    samples=load(OUT/'median_samples.json');ids=np.unique(np.linspace(0,802,120,dtype=int));assert samples['local_frame_indexes']==ids.tolist()
    median=np.load(OUT/'median.npz')['median'];assert median.shape==(1080,1920,3) and sha(OUT/'median.npz')==samples['median_sha256']
    for y in range(0,1080,120):
        expected=np.median(np.stack([cv2.imread(str(fd/f'{i:06d}.jpg'))[y:y+120] for i in ids]),axis=0).astype(np.uint8)
        assert np.array_equal(expected,median[y:y+120])
    sys.path.insert(0,str(weight.parent.parent));from inference import BallInferencer
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    engine=BallInferencer(str(weight.parent.parent/'configs/tracknetv3_base.py'),str(weight),device='cuda',thre=.3,batchsize=4)
    engine.model.eval().requires_grad_(False);medians={'m':np.moveaxis(cv2.resize(median,(512,288)).astype(np.float32)/255.,-1,0)}
    packed=np.load(OUT/'component_rectangles.npz');records=read(OUT/'map_records.csv');preds=read(OUT/'baseline_all_frames.csv');assert len(preds)==803 and len(records)==800
    for lo in range(3,803,4):
        times=list(range(lo,min(lo+4,803)));batch=[]
        for t in times:batch.append(dict(frames=np.stack([cv2.resize(cv2.imread(str(fd/f'{j:06d}.jpg')),(512,288)) for j in range(t-3,t+1)]),median_key='m',row={}))
        with torch.inference_mode(),torch.amp.autocast('cuda'):output,_,_=engine.model.forward(frames=tensors(batch,medians,engine.device))
        for t,hm in zip(times,output[:,0].float().cpu().numpy(),strict=True):
            i=t-3;m=records[i];a,b=map(int,packed['offsets'][i:i+2]);idx=packed['indices'][a:b];val=packed['values'][a:b]
            assert int(m['local_frame'])==t and int(m['source_frame'])==FIRST+t and json.loads(m['batch_frames_json'])==times
            assert np.array_equal(hm.ravel()[idx],val.astype(np.float32))
            # Exact original decoder agrees on full replay map and compact stored map.
            location=engine._predict_location(hm);assert location==engine._predict_location(expand(idx,val))
            x,y,w,h,conf=location;px,py=int((x+w/2)*1920/512),int((y+h/2)*1080/288);p=preds[t]
            assert (int(p['source_frame']),int(p['x_px']),int(p['y_px']),int(p['visible']))==(FIRST+t,px,py,int(px!=0 or py!=0)) and float(p['confidence'])==float(conf)
            assert int(p['input_first_source_frame'])==FIRST+t-3 and int(p['input_last_source_frame'])==FIRST+t
    for c in cases:
        p=preds[c['local_frame']];assert c['source_frame']==FIRST+c['local_frame']
        assert (c['prediction_visible'],c['prediction_x_px'],c['prediction_y_px'])==(int(p['visible']),float(p['x_px']),float(p['y_px']))
        assert int(c['video_target_time_sec']*(60000/1001))==c['local_frame']
    video=cv2.VideoCapture(str(OUT/'context.mp4'));decoded=0
    while True:
        ok,im=video.read()
        if not ok:break
        assert im.shape==(800,1280,3);decoded+=1
    video.release();assert decoded==803
    spec=importlib.util.spec_from_file_location('review056',OUT/'serve_review.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.self_test();module.verify_history(OUT)
    report=dict(status='passed',source_clock_reference_sheets_byte_exact=True,source_video_hash_verified=True,raw_source_cached_images_exact=803,
        native_median_pixelwise_verified=True,original_weights_verified=True,replayed_original_model_heatmaps=800,reconstructed_original_decoder_outputs=800,
        target_times_verified=376,context_decoded_frames=803,real_labels_unmodified=True,real_timing_not_started=True,new_training=False,script_sha256=sha(Path(__file__)))
    if http:
        report.update(synthetic_http());url=load(OUT/'server_state.json')['url'];assert get(url)[1]==(OUT/'review.html').read_bytes()
        d=json.loads(get(url+'api/cases')[1]);assert len(d['cases'])==376 and d['timing']['captured_ms']==0 and not d['timing']['active']
        for c in cases:assert get(url+'image/'+c['case_id']+'.jpg')[1]==Path(c['source_image']).read_bytes()
        data=(OUT/'context.mp4').read_bytes()
        for header,expected in [('bytes=0-99',data[:100]),('bytes=-100',data[-100:])]:
            status,content=get(url+'video/context.mp4',{'Range':header});assert status==206 and content==expected
        for payload,status in [(dict(case_id='R001',decision='accept_prediction',revision='stale',client_id='qa_never_active'),409),
            (dict(case_id='not_real',decision='accept_prediction',revision=d['revision'],client_id='qa_never_active'),400),
            (dict(case_id='R001',decision='correct_center',x=1,y=1,revision=d['revision'],client_id='qa_never_active'),400)]:post(url+'api/save',payload,status)
        report.update(real_http_images_exact=376,real_http_video_ranges=2,real_invalid_saves_rejected=3)
    assert sha(OUT/'review_labels.csv')==before and not (OUT/'review_events.jsonl').exists() and not (OUT/'timing_events.jsonl').exists()
    assert all(sha(Path(p))==h for p,h in manifest['protected_input_hashes'].items())
    (OUT/('http_verification.json' if http else 'verification.json')).write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--http',action='store_true');main(parser.parse_args().http)
