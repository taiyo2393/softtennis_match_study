"""Verify phase-only assets/API and frozen comparisons without research answers."""
import argparse,json,subprocess,urllib.request,urllib.error
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,load,read,sha
from run_manual_ball_finetune import metrics
from evaluate_training_phase_scope import classify
from serve_training_phase_scope import verify_history

OUT=ROOT.parent/'ball_quality_20261109/training_phase_scope_review'

def main(http=False):
    m=load(OUT/'manifest.json');before=sha(OUT/'phase_labels.csv');initial=read(OUT/'initial_phase_labels.csv');current=read(OUT/'phase_labels.csv')
    assert before==m['initial_phase_sha256'] and current==initial
    assert sum(r['review_status']=='unreviewed' for r in current)==120 and sum(int(r['locked']) for r in current)==3
    assert all(r['phase_status']=='nonplay' and r['phase_detail']=='ended' for r in current if int(r['locked']))
    verify_history(OUT)
    for table in [m['protected_input_hashes'],m['immutable_artifact_hashes']]:assert all(sha(Path(p))==h for p,h in table.items())
    expected={'current_temporal':(66,4),'native_missing_only':(77,14),'crossscale_agreement16':(72,5),'logistic10_oof':(77,7)}
    for method in m['frozen_methods']:
        assert sha(Path(method['path']))==method['sha256'];counts=metrics(read(Path(method['path'])))['counts']
        assert (counts['localized_tp_16px'],counts['wrong_localization']+counts['presence_fp'])==expected[method['method']]
    g=dict(truth_visibility='visible',truth_x_px='100',truth_y_px='100');p=dict(visible='1',x_px='116',y_px='100')
    assert classify(g,p)==('correct16px',16)
    assert classify(g,dict(p,x_px='117'))==('wrong_position',17)
    assert classify(g,dict(p,visible='0'))==('missing_visible_ball',None)
    assert classify(dict(g,truth_visibility='not_visible'),p)==('output_when_not_observable',None)
    clips=load(OUT/'clips.json');codecs=[]
    for c in clips:
        probe=subprocess.run([r'C:\ffmpeg\bin\ffprobe.exe','-v','error','-select_streams','v:0','-show_entries','stream=codec_name,width,height,nb_frames,avg_frame_rate','-of','json',c['video_path']],check=True,capture_output=True,text=True)
        stream=json.loads(probe.stdout)['streams'][0]
        assert stream['codec_name']=='h264' and (stream['width'],stream['height'])==(1920,1080)
        assert int(stream['nb_frames'])==c['frame_count']
        codecs.append(dict(clip_id=c['clip_id'],codec=stream['codec_name'],frames=c['frame_count']))
    verification=dict(status='passed',all123_jpegs_previously_verified_exact=True,clips_h264_1920x1080_verified=8,
        prior_answers_reused=3,new_phase_answers_written=0,pending=120,all4_mixed_results_reproduced=True,
        conditional_metric_synthetic_cases_passed=4,no_phase_conditional_accuracy_computed=True,
        labels_and_frozen_methods_unchanged=True,codecs=codecs,
        html_sha256=sha(OUT/'review.html'),server_sha256=sha(ROOT/'serve_training_phase_scope.py'),script_sha256=sha(Path(__file__)))
    if http:
        state=load(OUT/'server_state.json')
        def get(route,headers=None):
            with urllib.request.urlopen(urllib.request.Request(state['url']+route,headers=headers or {}),timeout=15) as r:return r.status,dict(r.headers),r.read()
        code,_,html=get('');assert code==200 and html==(OUT/'review.html').read_bytes()
        code,_,body=get('api/data');public=json.loads(body);assert code==200 and len(public['clips'])==8 and len(public['samples'])==123
        allowed={'sample_id','clip_id','candidate_id','local_frame','source_frame','clip_time_sec','phase_status','phase_detail','review_status','notes','locked'}
        assert all(set(s)==allowed for s in public['samples'])
        assert all(not any(k in s for k in ['x_px','y_px','truth_visibility','outcome','proposal_gate_score']) for s in public['samples'])
        assets=load(OUT/'image_assets.json')
        for sid,asset in assets.items():
            code,_,body=get('image/'+sid+'.jpg');assert code==200 and body==Path(asset['path']).read_bytes()
        for c in clips:
            path=Path(c['video_path']);code,_,body=get('video/'+c['clip_id']+'.mp4',{'Range':'bytes=0-99'});assert code==206 and body==path.read_bytes()[:100]
            code,_,body=get('video/'+c['clip_id']+'.mp4',{'Range':'bytes=-100'});assert code==206 and body==path.read_bytes()[-100:]
        group=[s for s in public['samples'] if s['clip_id']=='001']
        current_payload=dict(clip_id='001',revision=public['revision'],annotations=[dict(sample_id=s['sample_id'],phase_status=s['phase_status']) for s in group],notes='',helper_range=None)
        invalid=[dict(current_payload,clip_id='INVALID'),dict(current_payload,revision='stale'),dict(current_payload,truth_x_px=10)]
        for payload in invalid:
            request=urllib.request.Request(state['url']+'api/save',data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
            try:urllib.request.urlopen(request,timeout=15)
            except urllib.error.HTTPError as e:assert e.code in [400,409]
            else:raise AssertionError('Invalid save accepted')
        assert sha(OUT/'phase_labels.csv')==before and not (OUT/'annotation_events.jsonl').exists()
        verification.update(http_images_exact=123,http_prefix_and_suffix_video_ranges=16,invalid_requests_rejected=3,public_predictions_hidden=True)
    (OUT/('http_verification.json' if http else 'verification.json')).write_text(json.dumps(verification,indent=2),encoding='utf-8')
    print(json.dumps(verification,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--http',action='store_true');main(p.parse_args().http)
