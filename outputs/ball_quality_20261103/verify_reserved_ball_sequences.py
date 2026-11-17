"""Verify clocks, clip decoding, frozen outputs, media and optionally local HTTP."""
import argparse, json, shutil, urllib.request, urllib.error
from pathlib import Path
import numpy as np
from integrate_044_reviewed_tracks import ROOT, load, read, sha, cv2
from compare_temporal_candidate_paths import select, OFFSETS
from evaluate_observed_ball_fallback import camera_mask

OUT=ROOT.parent/'ball_quality_20261104'

def main(http=False):
    m=load(OUT/'manifest.json');windows=load(OUT/'reserved_windows.json');cases=load(OUT/'annotation_cases.json')
    labels_path=OUT/'manual_sequence_labels.csv';before=sha(labels_path)
    assert before==m['initial_labels_sha256']
    labels=read(labels_path);assert len(labels)==52 and all(g['review_status']=='unreviewed' for g in labels)
    assert all(g['truth_x_px']==g['truth_y_px']=='' and not g['reviewer_id'] for g in labels)
    snapshot=OUT/'initial_labels_snapshot.csv'
    if snapshot.exists():assert sha(snapshot)==before
    else:shutil.copy2(labels_path,snapshot)
    assert all(sha(Path(p))==h for p,h in m['frozen_artifact_hashes'].items())
    assert all(sha(Path(p))==h for p,h in m['protected_input_hashes'].items())
    image_hashes=load(OUT/'source_image_hashes.json')
    assert all(sha(Path(p))==h for p,h in image_hashes.items())
    previous=load(OUT/'inventory_manifest.json');assert all(sha(Path(p))==h for p,h in previous['protected_input_hashes'].items())
    cache=load(OUT/'frozen_component_cache.json');camera=load(OUT/'camera_features.json');pred=read(OUT/'frozen_predictions.csv')
    queries=read(OUT/'query_model_inputs.csv');assert len(queries)==52 and len(pred)==104
    old_inputs=set()
    for role in ['train','validation','audit']:
        for g in read(ROOT/'balltrack_manual_finetune_pilot'/(role+'_labels.csv')):old_inputs.update(json.loads(g['input_frame_sha256_json']))
    for q in queries:
        paths=json.loads(q['input_frames_json']);hashes=json.loads(q['input_frame_sha256_json'])
        assert [int(Path(p).stem) for p in paths]==list(range(int(q['local_frame'])-3,int(q['local_frame'])+1))
        assert all(sha(Path(p))==h and h not in old_inputs for p,h in zip(paths,hashes,strict=True))
    matched=0;full_camera_checks=0;overview=[]
    for w in windows:
        cid=w['candidate_id'];fd=Path(w['frame_dir']);times=sorted(map(int,cache[cid]))
        # Full-clip reference computation checks that partial mask support was sufficient.
        reduced=[];scores=[]
        for t in range(w['frame_count']):
            im=cv2.imread(str(fd/f'{t:06d}.jpg'),cv2.IMREAD_REDUCED_COLOR_8)
            im=cv2.resize(im,(160,90),interpolation=cv2.INTER_AREA)[8:82].astype(np.int16)
            scores.append(float((np.abs(im-reduced[t%12]).mean(2)>24).mean()) if t>=12 else 0.)
            if t<12:reduced.append(im)
            else:reduced[t%12]=im
        full_camera_checks+=len(scores);mask=camera_mask(scores)
        for t,v in camera[cid]['score_pairs']:assert scores[t]==v
        assert all(int(mask[t])==v for t,v in camera[cid]['mask_at_context'])
        for p in [p for p in pred if p['candidate_id']==cid]:
            cfg=None if p['method']=='bbox_area' else dict(name='displacement_0.1',order='displacement',strength=.1)
            context={int(k):v for k,v in cache[cid].items()};t=int(p['local_frame'])
            assert all(t+d in context for d in OFFSETS)
            c,status=select(context,t,mask,cfg)
            assert status==p['selection_status'] and int(c is not None)==int(p['visible'])
            if c:
                assert float(p['x_px'])==c['x_px'] and float(p['y_px'])==c['y_px']
                assert float(p['confidence'])==c['bbox_mean'] and int(p['selected_component_index'])==c['component_index']
        # Sequentially decode original clip and reproduce the exact cachedJPEGs.
        targets={c['local_frame'] for c in cases if c['candidate_id']==cid}
        meta=load(Path(w['metadata_path']));cap=cv2.VideoCapture(meta['clip'])
        assert cap.isOpened()
        for t in range(max(targets)+1):
            ok,im=cap.read();assert ok
            if t in targets:
                ok,encoded=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok
                assert encoded.tobytes()==(fd/f'{t:06d}.jpg').read_bytes(),(cid,t,'cache/clip mismatch')
                matched+=1
                if t==w['center_local_frame']:
                    tile=np.full((396,640,3),20,np.uint8);tile[36:]=cv2.resize(im,(640,360),interpolation=cv2.INTER_AREA)
                    cv2.putText(tile,f'{w["window_id"]}: expand{cid[-3:]} / local{t} / unlabelled',(9,25),0,.65,(240,240,240),1,cv2.LINE_AA)
                    overview.append(tile)
        cap.release();print('Clock and full-camera verified:',cid,flush=True)
    assert matched==52
    cv2.imwrite(str(OUT/'reserved_source_overview.jpg'),np.vstack([np.hstack(overview[:2]),np.hstack(overview[2:])]))
    decoded=0
    for c in cases:
        assert sha(Path(c['source_image']))==c['source_image_sha256']
        assert sha(Path(c['context_video']))==c['context_video_sha256']
        cap=cv2.VideoCapture(c['context_video']);n=0
        while True:
            ok,frame=cap.read()
            if not ok:break
            assert frame.shape==(800,1280,3);n+=1
        cap.release();assert n==c['decoded_video_frames']==128;decoded+=n
    status=dict(status='passed',pending_human_labels=52,model_predictions_reproduced=104,
        target_jpegs_exactly_matching_sequential_clip_decode=matched,full_camera_reference_frames=full_camera_checks,
        videos_fully_decoded=52,total_display_frames_decoded=decoded,
        frozen_artifacts_verified=len(m['frozen_artifact_hashes']),input_images_verified=len(image_hashes),
        old20_input_hash_duplicates=0,real_labels_written=False,operational_changes=False,accuracy=None,
        protected_inputs_unchanged=True,script_sha256=sha(Path(__file__)))
    if http:
        state=load(OUT/'server_state.json');events=OUT/'annotation_events.jsonl';eb=events.read_bytes() if events.exists() else b''
        def get(route,headers=None):
            with urllib.request.urlopen(urllib.request.Request(state['url']+route,headers=headers or {}),timeout=15) as r:return r.status,dict(r.headers),r.read()
        code,_,body=get('api/cases');public=json.loads(body)['cases'];assert code==200 and len(public)==52
        allowed={'case_id','window_id','candidate_id','local_frame','source_frame','center_reused','truth_visibility','truth_x_px','truth_y_px','phase_status','review_status','notes'}
        assert all(set(c)==allowed and c['truth_x_px']==c['truth_y_px']=='' for c in public)
        for c in cases:
            code,_,body=get('image/'+c['case_id']+'.jpg');assert code==200 and body==Path(c['source_image']).read_bytes()
            code,_,body=get('video/'+c['case_id']+'.mp4',{'Range':'bytes=0-99'});assert code==206 and body==Path(c['context_video']).read_bytes()[:100]
        for payload in [dict(case_id='INVALID',phase_status='play',truth_visibility='not_visible'),
                        dict(case_id='A01',phase_status='unknown',truth_visibility='not_visible'),
                        dict(case_id='D13',phase_status='play',truth_visibility='visible',x=1920,y=100)]:
            req=urllib.request.Request(state['url']+'api/save',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
            try:urllib.request.urlopen(req,timeout=15)
            except urllib.error.HTTPError as e:assert e.code==400
            else:raise AssertionError('Invalid save accepted')
        assert (events.read_bytes() if events.exists() else b'')==eb
        status.update(http_images_exact=52,http_video_range_responses=52,invalid_saves_rejected=3,predictions_hidden=True)
    assert sha(labels_path)==before
    (OUT/('http_verification.json' if http else 'verification.json')).write_text(json.dumps(status,indent=2),encoding='utf-8')
    print(json.dumps(status,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--http',action='store_true');main(p.parse_args().http)
