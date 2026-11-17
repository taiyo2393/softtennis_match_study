"""Isolated whole-point-context proposal, preserving all prior human edits.

044 valid second-service bracket through provisional point-end bracket. The
primary reviewed registry/v4 stays unchanged. Candidate counts are not accuracy.
"""
import json,math,sys,subprocess
from pathlib import Path
from collections import Counter
import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha
import compare_temporal_candidate_paths as temporal
from compare_ball_candidate_decoders import choose
from run_manual_ball_finetune import metrics

OUT=ROOT/'rally_044_temporal_proposal'
CID='match1_motion_pilot_016'
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def same_location(a,b):
    return int(a['visible'])==int(b['visible']) and float(a['x_px'])==float(b['x_px']) and float(a['y_px'])==float(b['y_px'])
def jumps(rows,start,end):
    found=[];by={int(r['local_frame']):r for r in rows}
    for t in range(start+2,end+1):
        a,b=by[t-2],by[t]
        if int(a['visible']) and int(b['visible']):
            distance=math.hypot(float(a['x_px'])-float(b['x_px']),float(a['y_px'])-float(b['y_px']))
            if distance>100:found.append(dict(local_frame=t,previous_local_frame=t-2,source_frame=b['source_frame'],distance_px=distance))
    return found

def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'protocol.json').exists(),'Preserve proposal snapshots.'
    registry=load(ROOT/'reviewed_ball_overrides.json');reference=registry['clips'][CID]
    baseline_path=Path(reference['reviewed_output']);raw_path=Path(reference['original_baseline'])
    assert sha(baseline_path)==reference['reviewed_output_sha256']
    baseline=read(baseline_path);raw=read(raw_path);assert len(baseline)==len(raw)==1169
    scope_path=ROOT/'play_start_audit/044_scope_report.json';scope=load(scope_path)
    offset=int(raw[0]['source_frame']);main_start,main_end=scope['candidate_main_play_local_frames']
    start=scope['valid_serve_contact_source_frame_interval'][0]-offset
    end=scope['point_end_source_frame_interval'][1]-offset
    assert (start,end,main_start,main_end)==(730,1155,739,1140)
    positive=set(reference['confirmed_ball_candidate_local_frames'])
    negative=set(reference['confirmed_false_raw_local_frames'])|{int(r['local_frame']) for r in reference['confirmed_false_tiled_candidates']}
    pinned=set(reference['reviewed_changed_local_frames'])|positive|negative|{int(r['local_frame']) for r in baseline if r['human_review_id']}
    assert len(positive)==18 and all(start<=t<=end for t in pinned)
    for t in range(len(raw)):
        assert int(raw[t]['local_frame'])==int(baseline[t]['local_frame'])==t
        assert int(raw[t]['source_frame'])==int(baseline[t]['source_frame'])==offset+t
        if not same_location(raw[t],baseline[t]):assert t in pinned
    selected_path=ROOT/'ball_temporal_path_comparison/training_selection.json';config=load(selected_path)['chosen_configuration']
    trial=load(ROOT/'ball_temporal_path_comparison/decision.json')
    assert config==dict(name='displacement_0.1',order='displacement',strength=.1)
    assert all(trial[k]['nonregression'] for k in ['validation','challenge','audit'])
    base=SOURCE/'data/balltrack/racketvision/match1'/CID;meta_path=base/'runs/threshold030_median120/run_metadata.json';meta=load(meta_path)
    modelroot=SOURCE/'third_party/RacketVision/source/BallTrack';weight=modelroot/'checkpoints/balltrack_best.pth'
    protected=dict(load(ROOT/'ball_temporal_path_comparison/verified_inputs.json')['protected_input_hashes'])
    paths=[baseline_path,raw_path,scope_path,selected_path,ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',
        ROOT/'reviewed_tracking_bundles.json',meta_path,weight,Path(__file__).parent/'compare_temporal_candidate_paths.py']
    person=load(ROOT/'reviewed_person_overrides.json')['rallies']['match1_rally_window_044'];paths.append(Path(person['reviewed_output']))
    paths.extend(Path(p) for p in reference['review_records'])
    for path in paths:protected[str(path)]=sha(path)
    assert all(sha(Path(p))==h for p,h in protected.items())
    protocol=dict(status='fixed_before_whole_scope_inference',rally='match1_rally_window_044',ball_clip=CID,
        local_frames=[start,end],candidate_main_play_local_frames=[main_start,main_end],clock='legacy edited match1, integer source-frame join',
        boundaries='Display starts at first possible valid2ndservice contact and ends at last possible point-end; both brackets remain provisional',
        baseline_current_reviewed=str(baseline_path),baseline_sha256=sha(baseline_path),raw_original=str(raw_path),raw_sha256=sha(raw_path),
        config_fixed_on_training=config,config_selection_sha256=sha(selected_path),model_and_threshold_unchanged=True,
        protected_frames=sorted(pinned),confirmed_real_candidate_frames=sorted(positive),candidate_rejection_context_frames=sorted(negative),
        merge='Freeze all existing reviewed fields at every touched review frame, including existing missing output. Else change only actual component selection, no interpolation.',
        rejection_semantics='Keeping a reviewed missing row is conservative freezing, not an assertion that no physical ball exists. Do not propagate positive/negative labels.',
        model_context='Same offline +/-8source frames, stride2. Compute only bounded context around730..1155, not first-fault/preparation video.',
        jump_diagnostic='Two-source-frame displacements>100px for review prioritisation only; not an error label or adoption criterion',
        video='Current reviewed vs merged proposal, native1920x1080 plus100px header,0.5x audio; no new person-ID or score inference',
        existing_person_input='Unchanged; ball-only comparison video avoids extending person boxes outside their available scope',
        automatic_operational_adoption=False,independent_full_track_accuracy=False,protected_input_hashes=protected,script_sha256=sha(Path(__file__)))
    dump(OUT/'protocol.json',protocol)
    mp=base/'runs/threshold030_median120/median.npz';queries=[]
    for t in range(start,end+1):
        frames=[base/'frames'/f'{j:06d}.jpg' for j in range(t-3,t+1)]
        queries.append(dict(sample_id=f'{CID}_scope_{t:06d}',candidate_id=CID,local_frame=t,source_frame=offset+t,
            input_frames_json=json.dumps([str(p) for p in frames]),input_frame_sha256_json=json.dumps([sha(p) for p in frames]),
            median_path=str(mp),median_sha256=sha(mp),truth_visibility='unreviewed',truth_x_px='',truth_y_px=''))
    sys.path.insert(0,str(modelroot));from inference import BallInferencer
    torch.backends.cudnn.benchmark=False
    infer=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(weight),device='cuda',thre=.3,batchsize=4)
    temporal.OUT=OUT;contexts,masks=temporal.get_contexts(infer,queries,'full_scope',protected)
    cs=contexts[CID];mask=masks[CID]
    proposal=[];diagnostics=[];changes=[];raw_reproduction=[]
    for t,old in enumerate(baseline):
        extra=dict(proposal_policy='outside_scope_unchanged',temporal_selection_status='',temporal_candidate_x_px='',temporal_candidate_y_px='',
            temporal_selected_component_index='',temporal_context_last_possible_source_frame='',output_newly_human_reviewed=0,proposal_interpolated=0)
        row=dict(old,**extra)
        if start<=t<=end:
            area=choose(cs[t],'bbox_area');chosen,status=temporal.select(cs,t,mask,config)
            reproduced=dict(visible=int(area is not None),x_px=area['x_px'] if area else 0,y_px=area['y_px'] if area else 0)
            assert reproduced['visible']==int(raw[t]['visible'])
            difference=math.hypot(float(raw[t]['x_px'])-reproduced['x_px'],float(raw[t]['y_px'])-reproduced['y_px'])
            assert difference<=3,difference
            raw_reproduction.append(difference)
            row.update(temporal_selection_status=status,temporal_candidate_x_px=chosen['x_px'] if chosen else 0,
                temporal_candidate_y_px=chosen['y_px'] if chosen else 0,temporal_selected_component_index=chosen['component_index'] if chosen else -1,
                temporal_context_last_possible_source_frame=offset+min(t+8,meta['frame_count']-1))
            if t in pinned:
                row['proposal_policy']='existing_review_frame_preserved'
            else:
                row['proposal_policy']='unreviewed_temporal_candidate_selection'
                px,py=(chosen['x_px'],chosen['y_px']) if chosen else (0,0)
                assert int(old['visible'])==int(px!=0 or py!=0)
                if (float(old['x_px']),float(old['y_px']))!=(px,py):
                    row.update(x_px=px,y_px=py,visible=int(px!=0 or py!=0),confidence=chosen['bbox_mean'] if chosen else 0,
                        position_source='model_observed_temporal_candidate_selected',coordinate_extractor='threshold030_component_bbox_center_no_interpolation',
                        ball_candidate_review='unreviewed_temporal_selection',human_review_id='',precise_center_reviewed_here=0,interpolated_here=0)
            diagnostics.append(dict(local_frame=t,source_frame=offset+t,scope='candidate_main_play' if main_start<=t<=main_end else 'boundary_uncertainty',
                raw_x_px=raw[t]['x_px'],raw_y_px=raw[t]['y_px'],raw_visible=raw[t]['visible'],
                temporal_x_px=chosen['x_px'] if chosen else 0,temporal_y_px=chosen['y_px'] if chosen else 0,
                temporal_status=status,review_frame_frozen=int(t in pinned),proposal_changed=int(not same_location(old,row))))
            if not same_location(old,row):changes.append(dict(local_frame=t,source_frame=offset+t,before_x=old['x_px'],before_y=old['y_px'],after_x=row['x_px'],after_y=row['y_px'],
                distance_px=math.hypot(float(old['x_px'])-float(row['x_px']),float(old['y_px'])-float(row['y_px'])),review_status='unreviewed_proposal',selection_status=status))
        if t in pinned or not start<=t<=end:assert all(str(row[k])==str(old[k]) for k in old)
        assert int(row['visible'])==int(old['visible'])
        proposal.append(row)
    write(OUT/'balltrack_predictions_proposed.csv',proposal);write(OUT/'full_scope_selection_diagnostics.csv',diagnostics)
    if changes:write(OUT/'changed_unreviewed_observations.csv',changes)
    for t in positive:assert all(str(proposal[t][k])==str(baseline[t][k]) for k in baseline[t]) and int(proposal[t]['visible'])
    old_jumps=jumps(baseline,main_start,main_end);new_jumps=jumps(proposal,main_start,main_end)
    if old_jumps:write(OUT/'current_large_displacements.csv',old_jumps)
    if new_jumps:write(OUT/'proposed_large_displacements.csv',new_jumps)
    # Already existing sparse manual centers: safety check only, not new labels.
    audit_rows=[r for r in read(ROOT/'balltrack_manual_finetune_pilot/audit_labels.csv') if r['candidate_id']==CID and start<=int(r['local_frame'])<=end]
    sparse={}
    for name,table in [('current_reviewed',baseline),('proposal',proposal)]:
        evaluated=[dict(table[int(r['local_frame'])],truth_visibility=r['truth_visibility'],truth_x_px=r['truth_x_px'],truth_y_px=r['truth_y_px']) for r in audit_rows]
        sparse[name]=metrics(evaluated) if evaluated else None
    dump(OUT/'existing_manual_center_check.json',dict(annotated_frames=[int(r['local_frame']) for r in audit_rows],metrics=sparse,
        interpretation='Previously seen sparse centers only; no full-track accuracy or re-labelling'))
    # One native-aspect view with both outputs. A green ring pins prior user review.
    import cv2
    fps=meta['fps'];silent=OUT/'comparison_silent.mp4';final=OUT/'comparison_h264.mp4'
    cmd=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1920x1180','-r',str(fps),'-i','pipe:0',
        '-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(silent)]
    with (OUT/'render.log').open('wb') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        for t in range(start,end+1):
            old=baseline[t];new=proposal[t];image=cv2.imread(str(base/'frames'/f'{t:06d}.jpg'));assert image.shape[:2]==(1080,1920)
            frame=np.full((1180,1920,3),18,np.uint8);frame[100:]=image
            cv2.putText(frame,f'044 | local {t} | source {offset+t} | valid-service to point-end context | 0.5x',(12,28),0,.83,(240,240,240),2,cv2.LINE_AA)
            cv2.putText(frame,'ORANGE: current reviewed output',(12,60),0,.72,(0,165,255),2,cv2.LINE_AA)
            cv2.putText(frame,'BLUE: temporal proposal',(690,60),0,.72,(255,200,25),2,cv2.LINE_AA)
            cv2.putText(frame,'GREEN: prior user-reviewed ball',(1220,60),0,.72,(80,255,100),2,cv2.LINE_AA)
            state='PROVISIONAL CONTACT/END BRACKET' if t<main_start or t>main_end else 'Candidate main-play scope; unmarked times are not human validated'
            cv2.putText(frame,state+' | Missing output stays blank',(12,89),0,.64,(220,220,220),1,cv2.LINE_AA)
            if t in positive:
                p=(round(float(old['x_px'])),100+round(float(old['y_px'])));cv2.circle(frame,p,16,(80,255,100),2,cv2.LINE_AA)
            else:
                for name,r,color,radius in [('CURRENT',old,(0,165,255),20),('PROPOSED',new,(255,200,25),12)]:
                    if int(r['visible']):
                        p=(round(float(r['x_px'])),100+round(float(r['y_px'])));cv2.circle(frame,p,radius,color,2,cv2.LINE_AA)
                        if not same_location(old,new):cv2.putText(frame,name,(min(1760,p[0]+24),p[1]+(-18 if name=='PROPOSED' else 28)),0,.60,color,2,cv2.LINE_AA)
            if changes and t==changes[0]['local_frame']:assert cv2.imwrite(str(OUT/'comparison_preview.jpg'),frame)
            for _ in range(2):proc.stdin.write(frame.tobytes())
        proc.stdin.close();assert proc.wait()==0
    probe=subprocess.run([r'C:\ffmpeg\bin\ffprobe.exe','-v','error','-show_entries','stream=codec_type','-of','json',meta['clip']],capture_output=True,text=True,check=True)
    has_audio=any(r['codec_type']=='audio' for r in json.loads(probe.stdout)['streams'])
    if has_audio:
        graph=f'[1:a]atrim=start={start/fps:.9f}:end={(end+1)/fps:.9f},asetpts=PTS-STARTPTS,atempo=0.5[a]'
        subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-i',str(silent),'-i',meta['clip'],'-filter_complex',graph,
            '-map','0:v:0','-map','[a]','-c:v','copy','-c:a','aac','-movflags','+faststart',str(final)],check=True)
    else:
        subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-i',str(silent),'-c','copy','-movflags','+faststart',str(final)],check=True)
    cap=cv2.VideoCapture(str(final));decoded=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape[:2]==(1180,1920);decoded+=1
    cap.release();assert decoded==2*(end-start+1)==852
    assert all(sha(Path(p))==h for p,h in protected.items())
    report=dict(status='whole_scope_comparison_ready_not_promoted',local_frames=[start,end],source_frames=end-start+1,
        candidate_main_play_frames=main_end-main_start+1,changed_unreviewed_frames=len(changes),
        changed_candidate_main_play_frames=sum(main_start<=c['local_frame']<=main_end for c in changes),
        reviewed_frames_frozen=len(pinned),confirmed_ball_candidates_preserved=len(positive),
        current_output_frames=sum(int(r['visible']) for r in baseline[start:end+1]),proposal_output_frames=sum(int(r['visible']) for r in proposal[start:end+1]),
        current_main_play_output_frames=sum(int(r['visible']) for r in baseline[main_start:main_end+1]),
        proposed_main_play_output_frames=sum(int(r['visible']) for r in proposal[main_start:main_end+1]),
        large_two_frame_displacements_over100px=dict(current=len(old_jumps),proposal=len(new_jumps),meaning='Review cue only; neither errors nor accuracy'),
        maximum_raw_baseline_reproduction_difference_px=max(raw_reproduction),decoded_video_frames=decoded,duration_seconds=decoded/fps,
        audio=has_audio,video=str(final.resolve()),video_sha256=sha(final),proposal_csv_sha256=sha(OUT/'balltrack_predictions_proposed.csv'),
        all_previous_review_fields_preserved=True,missing_and_presence_flags_preserved=True,interpolated_positions=0,
        operational_registries_unchanged=True,person_input_unchanged=True,protected_inputs_unchanged=True,full_track_accuracy_established=False,
        next='Watch current/proposed full scope. New blue positions remain unreviewed; do not treat output availability or jump counts as tracking accuracy.')
    dump(OUT/'verification.json',report);dump(OUT/'verified_inputs.json',dict(protected_input_hashes=protected,all_unchanged=True));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
