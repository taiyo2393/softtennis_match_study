"""Four reviewed challenge centers with unlabelled before/after visual context."""
import json,sys,subprocess
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,load,read,write,sha,cv2,np
from compare_temporal_candidate_paths import get_contexts,select
from compare_ball_candidate_decoders import choose

OUT=ROOT/'ball_temporal_path_comparison';CHALLENGE=ROOT/'ball_inplay_challenge'
def main():
    assert not (OUT/'render_protocol.json').exists()
    decision=load(OUT/'decision.json');config=load(OUT/'training_selection.json')['chosen_configuration']
    assert all(decision[role]['nonregression'] for role in ['validation','challenge','audit'])
    model_path=load(OUT/'protocol.json')['original_model'];modelroot=Path(model_path).parents[1];sys.path.insert(0,str(modelroot))
    protected=dict(load(OUT/'verified_inputs.json')['protected_input_hashes'])
    labels={r['sample_id']:r for r in read(CHALLENGE/'manual_center_labels.csv')};templates=read(CHALLENGE/'query_model_inputs.csv')
    cases={c['sample_id']:c for c in load(CHALLENGE/'annotation_cases.json')};expected={r['sample_id']:r for r in read(OUT/'challenge_selected.csv')}
    protocol=dict(status='fixed_before_context_render',chosen_config=config,chosen_config_sha256=sha(OUT/'training_selection.json'),
        cases=['A','B','C','D'],context_half_window_source_frames=24,source_image_wh=[1920,1080],render_wh=[1920,1180],
        header_height=100,output_fps=29.97002997002997,speed='0.5x; target held for30extra output frames',
        colors='Orange:original area, pink:single-frame peak, blue:fixed temporal selector, green+:manual center at TARGET only',
        truth='Only4target frames have manual centers. Context predictions are not newly verified. Display holds are not extra observations.',
        automatic_adoption=False)
    (OUT/'render_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    queries=[];timeline=[]
    for r in templates:
        case=cases[r['sample_id']];target=int(r['local_frame']);frame_dir=Path(case['source_image']).parent
        for t in range(target-24,target+25):
            paths=[frame_dir/f'{j:06d}.jpg' for j in range(t-3,t+1)]
            queries.append(dict(r,sample_id=f'{case["candidate_id"]}_unlabelled_context_{t:06d}',local_frame=str(t),
                source_frame=str(case['clip_first_source_frame']+t),input_frames_json=json.dumps([str(p) for p in paths]),
                input_frame_sha256_json=json.dumps([sha(p) for p in paths]),truth_visibility='unreviewed',truth_x_px='',truth_y_px=''))
            timeline.append(dict(case_id=case['case_id'],sample_id=r['sample_id'],candidate_id=case['candidate_id'],target=target,local_frame=t,
                source_frame=case['clip_first_source_frame']+t,frame_dir=frame_dir))
    from inference import BallInferencer
    infer=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),model_path,device='cuda',thre=.3,batchsize=4)
    contexts,masks=get_contexts(infer,queries,'render_context',protected)
    video=OUT/'temporal_candidate_comparison_h264.mp4';fps=protocol['output_fps']
    cmd=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1920x1180','-r',str(fps),'-i','pipe:0',
        '-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    colors=dict(AREA=(0,165,255),PEAK=(215,80,255),TEMPORAL=(255,205,25));n=0;observations=[]
    with (OUT/'render.log').open('wb') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        for r in timeline:
            t=r['local_frame'];target=r['target'];cid=r['candidate_id'];cs=contexts[cid][t];temporal,status=select(contexts[cid],t,masks[cid],config)
            picks=dict(AREA=choose(cs,'bbox_area'),PEAK=choose(cs,'component_peak'),TEMPORAL=temporal)
            im=cv2.imread(str(r['frame_dir']/f'{t:06d}.jpg'));assert im.shape[:2]==(1080,1920)
            frame=np.zeros((1180,1920,3),np.uint8);frame[100:]=im
            cv2.putText(frame,f'{r["case_id"]} | pilot{cid[-3:]} local{t} | target{target} | 0.5x | TEMPORAL RULE FIXED ON TRAINING',(12,29),0,.79,(245,245,245),2,cv2.LINE_AA)
            for text,pos,color in [('ORANGE: original area',12,colors['AREA']),('PINK: single-frame peak',480,colors['PEAK']),('BLUE: temporal',1010,colors['TEMPORAL']),('GREEN+: manual center',1450,(90,255,70))]:
                cv2.putText(frame,text,(pos,61),0,.69,color,2,cv2.LINE_AA)
            cv2.putText(frame,'TARGET: human center available' if t==target else 'Unlabelled temporal context - no accuracy labels here',(12,89),0,.66,(230,230,230),1,cv2.LINE_AA)
            for method,c in picks.items():
                if c:
                    p=(c['x_px'],c['y_px']+100);radius={'AREA':22,'PEAK':16,'TEMPORAL':10}[method]
                    cv2.circle(frame,p,radius,colors[method],2,cv2.LINE_AA)
                    offset={'AREA':36,'PEAK':-31,'TEMPORAL':-12}[method]
                    cv2.putText(frame,method,(min(1770,p[0]+26),p[1]+offset),0,.60,colors[method],2,cv2.LINE_AA)
                    observations.append(dict(case_id=r['case_id'],candidate_id=cid,local_frame=t,source_frame=r['source_frame'],method=method,x_px=c['x_px'],y_px=c['y_px'],
                        human_label_available=int(t==target),temporal_status=status if method=='TEMPORAL' else ''))
            if t==target:
                truth=labels[r['sample_id']];p=(int(truth['truth_x_px']),int(truth['truth_y_px'])+100)
                cv2.drawMarker(frame,p,(90,255,70),cv2.MARKER_CROSS,25,2,cv2.LINE_AA);cv2.rectangle(frame,(1,100),(1918,1178),(50,230,255),3)
                frozen=expected[r['sample_id']]
                assert temporal and (temporal['x_px'],temporal['y_px'])==(int(frozen['x_px']),int(frozen['y_px']))
                if r['case_id']=='A':assert cv2.imwrite(str(OUT/'temporal_comparison_preview.jpg'),frame)
            for _ in range(31 if t==target else 1):proc.stdin.write(frame.tobytes());n+=1
        proc.stdin.close();assert proc.wait()==0
    write(OUT/'render_context_predictions.csv',observations)
    cap=cv2.VideoCapture(str(video));decoded=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape[:2]==(1180,1920);decoded+=1
    cap.release();assert decoded==n==316
    assert all(sha(Path(p))==h for p,h in protected.items())
    report=dict(video=str(video.resolve()),sha256=sha(video),source_display_frames=196,decoded_display_frames=decoded,duration_seconds=decoded/fps,
        manual_centers=4,target_outputs_match_frozen_evaluation=True,native_aspect_and_image_dimensions_preserved=True,
        context_accuracy_unconfirmed=True,protected_inputs_unchanged=True)
    (OUT/'render_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))

if __name__=='__main__':main()
