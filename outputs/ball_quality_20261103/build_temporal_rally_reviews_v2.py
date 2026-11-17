"""Whole-rally-context comparison; identical component preserves stored pixels.

No training/threshold/rule changes. Small re-inference bbox jitter is not a
candidate-selection change. All existing reviewed-frame fields take precedence.
"""
import json,sys,subprocess,math
from pathlib import Path
import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2
import compare_temporal_candidate_paths as temporal
from compare_ball_candidate_decoders import choose
from evaluate_observed_ball_fallback import camera_mask
from run_manual_ball_finetune import metrics
from build_044_temporal_comparison import same_location,jumps

ROOT_OUT=ROOT/'temporal_rally_reviews_v2'
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def harmonize(rows):
    keys=list(dict.fromkeys(k for r in rows for k in r));return [{k:r.get(k,'') for k in keys} for r in rows]

def render(out,base,meta,current,proposed,start,end,main_range,positives,preview_frame,display_id):
    fps=meta['fps'];silent=out/'comparison_silent.mp4';final=out/'comparison_h264.mp4'
    cmd=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1920x1180','-r',str(fps),'-i','pipe:0',
        '-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(silent)]
    with (out/'render.log').open('wb') as log:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        for t in range(start,end+1):
            a,b=current[t],proposed[t];im=cv2.imread(str(base/'frames'/f'{t:06d}.jpg'));assert im.shape[:2]==(1080,1920)
            frame=np.full((1180,1920,3),18,np.uint8);frame[100:]=im
            cv2.putText(frame,f'{display_id} | local {t} | source {a["source_frame"]} | whole point context | 0.5x',(12,28),0,.83,(240,240,240),2,cv2.LINE_AA)
            cv2.putText(frame,'ORANGE: current output',(12,60),0,.72,(0,165,255),2,cv2.LINE_AA)
            cv2.putText(frame,'BLUE: proposed temporal selection',(540,60),0,.72,(255,200,25),2,cv2.LINE_AA)
            cv2.putText(frame,'GREEN: prior reviewed ball (unchanged)',(1240,60),0,.66,(80,255,100),2,cv2.LINE_AA)
            if main_range is None:scope='Stored valid serve precedes clip; exact point end is not confirmed'
            elif not main_range[0]<=t<=main_range[1]:scope='PROVISIONAL SERVICE/END BRACKET'
            else:scope='Candidate main-play scope; unchanged points are not new accuracy labels'
            cv2.putText(frame,scope+' | Missing output stays blank',(12,89),0,.63,(225,225,225),1,cv2.LINE_AA)
            if t in positives:
                cv2.circle(frame,(round(float(a['x_px'])),100+round(float(a['y_px']))),16,(80,255,100),2,cv2.LINE_AA)
            else:
                for name,r,color,radius in [('CURRENT',a,(0,165,255),20),('PROPOSED',b,(255,200,25),12)]:
                    if int(r['visible']):
                        p=(round(float(r['x_px'])),100+round(float(r['y_px'])));cv2.circle(frame,p,radius,color,2,cv2.LINE_AA)
                        if not same_location(a,b):cv2.putText(frame,name,(min(1750,p[0]+24),p[1]+(-18 if name=='PROPOSED' else 28)),0,.60,color,2,cv2.LINE_AA)
            if t==preview_frame:assert cv2.imwrite(str(out/'comparison_preview.jpg'),frame)
            proc.stdin.write(frame.tobytes());proc.stdin.write(frame.tobytes())
        proc.stdin.close();assert proc.wait()==0
    probe=subprocess.run([r'C:\ffmpeg\bin\ffprobe.exe','-v','error','-show_entries','stream=codec_type','-of','json',meta['clip']],capture_output=True,text=True,check=True)
    audio=any(r['codec_type']=='audio' for r in json.loads(probe.stdout)['streams'])
    if audio:
        graph=f'[1:a]atrim=start={start/fps:.9f}:end={(end+1)/fps:.9f},asetpts=PTS-STARTPTS,atempo=0.5[a]'
        subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-i',str(silent),'-i',meta['clip'],'-filter_complex',graph,
            '-map','0:v:0','-map','[a]','-c:v','copy','-c:a','aac','-movflags','+faststart',str(final)],check=True)
    else:subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-i',str(silent),'-c','copy','-movflags','+faststart',str(final)],check=True)
    cap=cv2.VideoCapture(str(final));n=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape[:2]==(1180,1920);n+=1
    cap.release();assert n==2*(end-start+1)
    return dict(video=str(final.resolve()),sha256=sha(final),decoded_video_frames=n,duration_seconds=n/fps,audio=audio)

def process(suffix,infer,config,shared_protected):
    cid=f'match1_motion_pilot_{suffix:03d}';out=ROOT_OUT/cid;out.mkdir(exist_ok=True)
    assert not (out/'protocol.json').exists()
    registry=load(ROOT/'reviewed_ball_overrides.json');reference=registry['clips'].get(cid)
    raw_path=ROOT.parent/'tracking_quality_20261102/ball_aligned'/cid/'balltrack_predictions.csv'
    current_path=Path(reference['reviewed_output']) if reference else raw_path
    current=read(current_path);raw=read(raw_path);offset=int(raw[0]['source_frame'])
    base=SOURCE/'data/balltrack/racketvision/match1'/cid;meta_path=base/'runs/threshold030_median120/run_metadata.json';meta=load(meta_path)
    positives=set(reference['confirmed_ball_candidate_local_frames']) if reference else set();pinned=set()
    if reference:
        assert sha(current_path)==reference['reviewed_output_sha256']
        pinned=set(reference['reviewed_changed_local_frames'])|positives|set(reference['confirmed_false_raw_local_frames'])
        pinned|={int(r['local_frame']) for r in reference.get('confirmed_false_tiled_candidates',[])}
        pinned|={int(r['local_frame']) for r in current if r['human_review_id']}
    if suffix==16:
        start,end=730,1155;main_range=[739,1140];display_id='044';cache_dir=ROOT/'rally_044_temporal_proposal'
    else:
        assert suffix==15;start,end=0,meta['frame_count']-1;main_range=None;display_id='040';cache_dir=None
        inventory=next(r for r in read(ROOT/'play_start_audit/ball_pilot_start_inventory.csv') if r['ball_pilot_id']==cid)
        assert inventory['legacy_serve_review_status']=='impact_confirmed' and float(inventory['legacy_serve_timestamp_sec'])<float(inventory['clip_start_sec'])
    protected=dict(shared_protected)
    for path in [raw_path,current_path,meta_path,ROOT/'play_start_audit/ball_pilot_start_inventory.csv']:
        protected[str(path)]=sha(path)
    if reference:
        for path in reference['review_records']:protected[path]=sha(Path(path))
    if cache_dir:
        for name in ['full_scope_component_cache.json','full_scope_context_image_hashes.json','verified_inputs.json']:
            path=cache_dir/name;protected[str(path)]=sha(path)
    protocol=dict(status='fixed_before_export',ball_pilot_id=cid,rally_candidate_id='match1_rally_window_'+display_id,
        displayed_local_frames=[start,end],candidate_main_play_range=main_range,clock='legacy edited match1 integer sourceframes',
        config=config,baseline=str(current_path),baseline_sha256=sha(current_path),raw=str(raw_path),
        protected_review_frames=sorted(pinned),positive_candidate_frames=sorted(positives),
        merge='Pin all previously reviewed fields at touched frames. If selected component==fresh area component, preserve stored output exactly; only different component can change position.',
        jitter_handling='A1-2pixel bbox difference upon re-inference is not a method effect. Preserve stored positions on unchanged selections.',
        prior044_export='Supersedes preliminary044 export whose8 tiny differences were re-inference jitter; no model/parameter changes.',
        exact_boundaries_confirmed=False,score_labels_joined=False,person_tracks_changed=False,automatic_adoption=False,
        protected_input_hashes=protected,script_sha256=sha(Path(__file__)))
    dump(out/'protocol.json',protocol)
    temporal.OUT=out
    if cache_dir:
        # Reuse exactly the already computed9-frame candidate contexts. No rerun/tuning.
        old_verified=load(cache_dir/'verified_inputs.json')['protected_input_hashes']
        assert all(sha(Path(p))==h for p,h in old_verified.items())
        image_hashes=load(cache_dir/'full_scope_context_image_hashes.json');assert all(sha(Path(p))==h for p,h in image_hashes.items())
        contexts={c:{int(t):cs for t,cs in table.items()} for c,table in load(cache_dir/'full_scope_component_cache.json').items()}
        camera_path=ROOT.parent/'ball_quality_20261102/temporal_guard/camera_features'/f'{cid}.json';masks={cid:camera_mask(load(camera_path)['changed_fraction'])}
        protected[str(camera_path)]=sha(camera_path)
    else:
        mp=base/'runs/threshold030_median120/median.npz';mp_hash=sha(mp);queries=[]
        for t in range(max(3,start),end+1):
            paths=[base/'frames'/f'{j:06d}.jpg' for j in range(t-3,t+1)]
            queries.append(dict(sample_id=f'{cid}_scope_{t:06d}',candidate_id=cid,local_frame=t,source_frame=offset+t,
                input_frames_json=json.dumps([str(p) for p in paths]),input_frame_sha256_json=json.dumps([sha(p) for p in paths]),
                median_path=str(mp),median_sha256=mp_hash,truth_visibility='unreviewed',truth_x_px='',truth_y_px=''))
        contexts,masks=temporal.get_contexts(infer,queries,'full_scope',protected)
    table=contexts[cid];mask=masks[cid];proposal=[];changes=[];diagnostics=[];numeric_jitter=[];selection_changes=[]
    for t,a in enumerate(current):
        b=dict(a,proposal_policy='outside_inferred_scope_unchanged',temporal_selection_status='',temporal_candidate_x_px='',temporal_candidate_y_px='',
            temporal_component_index='',newly_human_reviewed=0,proposal_interpolated=0)
        if start<=t<=end and t in table:
            area=choose(table[t],'bbox_area');chosen,status=temporal.select(table,t,mask,config)
            assert bool(area)==bool(int(raw[t]['visible']))
            delta=math.hypot(area['x_px']-float(raw[t]['x_px']),area['y_px']-float(raw[t]['y_px'])) if area else 0
            assert delta<=3,(t,delta)
            different=area is not None and chosen is not None and area['component_index']!=chosen['component_index']
            if different:selection_changes.append(t)
            elif delta>0:numeric_jitter.append(dict(local_frame=t,source_frame=offset+t,reinference_difference_px=delta))
            b.update(temporal_selection_status=status,temporal_candidate_x_px=chosen['x_px'] if chosen else 0,
                temporal_candidate_y_px=chosen['y_px'] if chosen else 0,temporal_component_index=chosen['component_index'] if chosen else -1)
            if t in pinned:b['proposal_policy']='existing_review_frame_preserved'
            elif not different:b['proposal_policy']='same_component_existing_output_preserved'
            else:
                assert chosen is not None
                b.update(proposal_policy='unreviewed_new_temporal_component',x_px=chosen['x_px'],y_px=chosen['y_px'],confidence=chosen['bbox_mean'],
                    position_source='model_observed_temporal_candidate_selected',coordinate_extractor='original_component_bbox_center_no_interpolation',
                    ball_candidate_review='unreviewed_temporal_selection',human_review_id='',precise_center_reviewed_here=0,interpolated_here=0)
            diagnostics.append(dict(local_frame=t,source_frame=offset+t,different_component=int(different),review_frame_frozen=int(t in pinned),
                new_coordinate_output=int(not same_location(a,b)),same_component_numeric_difference_px=delta if not different else '',selection_status=status))
        if t in pinned or b['proposal_policy'] in ['same_component_existing_output_preserved','outside_inferred_scope_unchanged']:
            assert all(str(b[k])==str(a[k]) for k in a)
        assert int(a['visible'])==int(b['visible'])
        if not same_location(a,b):changes.append(dict(local_frame=t,source_frame=offset+t,before_x=a['x_px'],before_y=a['y_px'],after_x=b['x_px'],after_y=b['y_px'],
            distance_px=math.hypot(float(a['x_px'])-float(b['x_px']),float(a['y_px'])-float(b['y_px'])),status='unreviewed_genuine_component_change'))
        proposal.append(b)
    proposal=harmonize(proposal);write(out/'balltrack_predictions_proposed.csv',proposal);write(out/'selection_diagnostics.csv',diagnostics)
    if changes:write(out/'changed_observations.csv',changes)
    if numeric_jitter:write(out/'reinference_jitter_not_applied.csv',numeric_jitter)
    for t in pinned:assert all(str(proposal[t][k])==str(current[t][k]) for k in current[t])
    sparse={}
    for role in ['validation','audit']:
        labels=[r for r in read(ROOT/'balltrack_manual_finetune_pilot'/f'{role}_labels.csv') if r['candidate_id']==cid and start<=int(r['local_frame'])<=end]
        if labels:
            sparse[role]=dict(frames=[int(r['local_frame']) for r in labels])
            for name,rows in [('current',current),('proposed',proposal)]:
                sparse[role][name]=metrics([dict(rows[int(r['local_frame'])],truth_visibility=r['truth_visibility'],truth_x_px=r['truth_x_px'],truth_y_px=r['truth_y_px']) for r in labels])
    challenge_labels=[r for r in read(ROOT/'ball_inplay_challenge/manual_center_labels.csv') if r['candidate_id']==cid and r['phase_status']=='play']
    if challenge_labels:
        sparse['four_case_challenge_subset']=dict(frames=[int(r['local_frame']) for r in challenge_labels])
        for name,rows in [('current',current),('proposed',proposal)]:
            sparse['four_case_challenge_subset'][name]=metrics([dict(rows[int(r['local_frame'])],truth_visibility=r['truth_visibility'],truth_x_px=r['truth_x_px'],truth_y_px=r['truth_y_px']) for r in challenge_labels])
    dump(out/'existing_sparse_center_checks.json',sparse)
    # Keep sparse label output separate; never call changed frames all corrected.
    old_jump=jumps(current,start,end);new_jump=jumps(proposal,start,end)
    if old_jump:write(out/'current_large_displacements.csv',old_jump)
    if new_jump:write(out/'proposed_large_displacements.csv',new_jump)
    preview=changes[0]['local_frame'] if changes else min(positives) if positives else start
    video=render(out,base,meta,current,proposal,start,end,main_range,positives,preview,display_id)
    assert all(sha(Path(p))==h for p,h in protected.items())
    report=dict(status='whole_scope_candidate_review_ready_not_promoted',ball_pilot_id=cid,rally_candidate_id='match1_rally_window_'+display_id,
        displayed_local_frames=[start,end],source_frames=end-start+1,raw_different_component_frames=selection_changes,
        genuine_output_changes=len(changes),changed_local_frames=[r['local_frame'] for r in changes],reviewed_frames_preserved=len(pinned),
        confirmed_real_candidates_preserved=len(positives),numerical_jitter_frames_not_applied=len(numeric_jitter),
        current_output_frames=sum(int(r['visible']) for r in current[start:end+1]),proposed_output_frames=sum(int(r['visible']) for r in proposal[start:end+1]),
        large_displacement_counts=dict(current=len(old_jump),proposed=len(new_jump),meaning='diagnostic cues, not accuracy/errors'),
        existing_sparse_center_checks=sparse,all_previous_review_fields_preserved=True,unchanged_component_coordinates_exactly_preserved=True,
        presence_and_missing_flags_preserved=True,interpolated_positions=0,person_tracks_changed=False,operational_registry_changed=False,
        protected_inputs_unchanged=True,full_track_accuracy_established=False,proposal_csv_sha256=sha(out/'balltrack_predictions_proposed.csv'),**video)
    dump(out/'verification.json',report);dump(out/'verified_inputs.json',dict(protected_input_hashes=protected,all_unchanged=True))
    print(json.dumps(report,indent=2),flush=True);return report

def main():
    ROOT_OUT.mkdir(exist_ok=True);assert not (ROOT_OUT/'summary.json').exists()
    chosen=ROOT/'ball_temporal_path_comparison/training_selection.json';config=load(chosen)['chosen_configuration']
    assert config==dict(name='displacement_0.1',order='displacement',strength=.1)
    modelroot=SOURCE/'third_party/RacketVision/source/BallTrack';sys.path.insert(0,str(modelroot));from inference import BallInferencer
    protected=dict(load(ROOT/'ball_temporal_path_comparison/verified_inputs.json')['protected_input_hashes'])
    for path in [chosen,ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json']:
        protected[str(path)]=sha(path)
    person=load(ROOT/'reviewed_person_overrides.json')['rallies']['match1_rally_window_044'];protected[person['reviewed_output']]=sha(Path(person['reviewed_output']))
    torch.backends.cudnn.benchmark=False
    infer=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),str(modelroot/'checkpoints/balltrack_best.pth'),device='cuda',thre=.3,batchsize=4)
    reports=[process(suffix,infer,config,protected) for suffix in [16,15]]
    dump(ROOT_OUT/'summary.json',dict(status='two_whole_context_comparisons_ready',reports=reports,model_parameters_unchanged=True,automatic_promotion=False))
    dump(ROOT/'rally_044_temporal_proposal/superseded_notice.json',dict(status='preliminary_export_not_for_use',reason='Eight1-2px differences were same-component re-inference jitter, not selector improvements',
        replacement=str((ROOT_OUT/'match1_motion_pilot_016').resolve()),old_inputs_and_outputs_retained=True))

if __name__=='__main__':main()
