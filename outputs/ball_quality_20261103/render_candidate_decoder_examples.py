"""Context video of predeclared area/peak alternatives on three training cases.

Human center is drawn ONLY on its exact annotated frame, repeated for readability.
No unlabelled context frame contributes to accuracy or gets an inferred truth.
"""
import json,subprocess
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,load,read,write,sha,cv2,np
from run_manual_ball_finetune import prepare
from run_hard_case_ball_finetune import infer_maps
from compare_ball_candidate_decoders import components,choose

OUT=ROOT/'ball_candidate_decoder_comparison'

def main():
    import sys
    comparison=load(OUT/'protocol.json');decision=load(OUT/'decision.json')
    assert decision['selected_method']=='bbox_area' and not (OUT/'context_render_protocol.json').exists()
    labels=read(ROOT/'balltrack_expanded_centers_pilot/expanded_train_labels.csv')[-3:]
    cases=load(ROOT/'ball_training_case_expansion/annotation_cases.json');names={c['sample_id']:c['case_id'] for c in cases}
    render_protocol=dict(methods=['bbox_area','component_peak'],chosen_for_visualisation='Peak was a predeclared alternative, not selected for operational use',
        labels='Three training examples only, A/C/D; B excluded as post-play by user',context_half_window_frames=24,
        truth='Exact human frame only; no interpolated truth; unlabelled context never counted as accuracy',
        original_image_wh=[1920,1080],render_wh=[1920,1180],header_height=100,output_fps=29.97002997002997,
        speed='0.5x from59.94fps, extra30 output frames hold target',audio=False,
        original_case_hashes={r['sample_id']:sha(Path(json.loads(r['input_frames_json'])[-1])) for r in labels})
    (OUT/'context_render_protocol.json').write_text(json.dumps(render_protocol,indent=2),encoding='utf-8')
    samples=[];case_rows=[]
    for r in labels:
        target=int(r['local_frame']);frame_dir=Path(json.loads(r['input_frames_json'])[-1]).parent
        for t in range(target-24,target+25):
            frames=[frame_dir/f'{j:06d}.jpg' for j in range(t-3,t+1)]
            samples.append(dict(r,sample_id=f'{r["candidate_id"]}_context_{t:06d}',local_frame=str(t),
                source_frame=str(int(r['source_frame'])+t-target),
                input_frames_json=json.dumps([str(p) for p in frames]),input_frame_sha256_json=json.dumps([sha(p) for p in frames])))
            case_rows.append(dict(case_id=names[r['sample_id']],target=target,target_truth_x=int(r['truth_x_px']),target_truth_y=int(r['truth_y_px']),
                frame_dir=frame_dir,local_frame=t,source_frame=int(r['source_frame'])+t-target,candidate_id=r['candidate_id']))
    medians={};prepared=prepare(samples,medians)
    modelroot=Path(comparison['source_checkpoint']).parents[1];sys.path.insert(0,str(modelroot))
    from inference import BallInferencer
    infer=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),comparison['source_checkpoint'],device='cuda',thre=.3,batchsize=4)
    maps=infer_maps(infer,prepared,medians);fps=render_protocol['output_fps'];video=OUT/'area_vs_peak_training_context_h264.mp4'
    cmd=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1920x1180','-r',str(fps),'-i','pipe:0',
        '-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    rendered=0;observations=[];expected={}
    orange=(0,165,255);blue=(255,205,30);green=(90,255,70)
    with (OUT/'context_render.log').open('wb') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        for context,hm in zip(case_rows,maps,strict=True):
            t=context['local_frame'];target=context['target'];is_target=t==target;cs=components(hm)
            area=choose(cs,'bbox_area');peak=choose(cs,'component_peak')
            image=cv2.imread(str(context['frame_dir']/f'{t:06d}.jpg'));assert image.shape[:2]==(1080,1920)
            im=np.zeros((1180,1920,3),np.uint8);im[100:]=image
            cv2.putText(im,f'{context["case_id"]} | pilot{context["candidate_id"][-3:]} local {t} | target {target} | 0.5x | DIAGNOSTIC ONLY',(14,29),0,.84,(245,245,245),2,cv2.LINE_AA)
            cv2.putText(im,'ORANGE: original largest box',(14,61),0,.72,orange,2,cv2.LINE_AA)
            cv2.putText(im,'BLUE: highest peak',(540,61),0,.72,blue,2,cv2.LINE_AA)
            cv2.putText(im,'GREEN +: manual center on TARGET ONLY',(1000,61),0,.72,green,2,cv2.LINE_AA)
            cv2.putText(im,'TARGET - human label available' if is_target else 'Temporal context: no human labels; not an accuracy test',(14,89),0,.65,(230,230,230),1,cv2.LINE_AA)
            for method,c,color,radius in [('AREA',area,orange,21),('PEAK',peak,blue,13)]:
                if c:
                    p=(c['x_px'],c['y_px']+100);cv2.circle(im,p,radius,color,2,cv2.LINE_AA)
                    cv2.putText(im,method,(min(1820,p[0]+24),p[1]+(28 if method=='AREA' else -22)),0,.65,color,2,cv2.LINE_AA)
                    observations.append(dict(case_id=context['case_id'],candidate_id=context['candidate_id'],local_frame=t,source_frame=context['source_frame'],method=method,
                        x_px=c['x_px'],y_px=c['y_px'],peak=c['component_peak'],bbox_mean=c['bbox_mean'],component_count=len(cs),human_label_available=int(is_target)))
            if is_target:
                p=(context['target_truth_x'],context['target_truth_y']+100)
                cv2.drawMarker(im,p,green,cv2.MARKER_CROSS,23,2,cv2.LINE_AA);cv2.rectangle(im,(1,100),(1918,1178),(50,230,255),3)
                for method,c in [('AREA',area),('PEAK',peak)]:
                    expected[(context['case_id'],method)]=(c['x_px'],c['y_px'])
                if context['case_id']=='C':assert cv2.imwrite(str(OUT/'decoder_comparison_preview.jpg'),im)
            for _ in range(31 if is_target else 1):proc.stdin.write(im.tobytes());rendered+=1
        proc.stdin.close();assert proc.wait()==0
    # Actual video endpoints/target coordinates match the independent sparse prediction export.
    for method,name in [('AREA','bbox_area'),('PEAK','component_peak')]:
        for r in read(OUT/f'three_training_examples_{name}.csv'):
            label=names[r['sample_id']];assert expected[label,method]==(int(r['x_px']),int(r['y_px']))
    write(OUT/'training_context_predictions.csv',observations)
    cap=cv2.VideoCapture(str(video));decoded=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(1180,1920);decoded+=1
    cap.release();assert decoded==rendered==237
    assert all(sha(Path(p))==h for p,h in comparison['protected_input_hashes'].items())
    report=dict(output=str(video.resolve()),source_context_frames=147,decoded_display_frames=decoded,duration_seconds=decoded/fps,
        source_aspect_ratio_preserved=True,target_prediction_coordinates_match_sparse_export=True,protected_inputs_unchanged=True,
        repeated_frames_are_display_holds_not_new_observations=True,manual_target_count=3,context_accuracy_unknown=True,sha256=sha(video))
    (OUT/'context_render_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))

if __name__=='__main__':main()
