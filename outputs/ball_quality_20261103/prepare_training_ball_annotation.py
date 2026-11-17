"""Four training-only frames for new BLINDED manual center annotation."""
import json,subprocess
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2,np

OUT=ROOT/'ball_training_case_expansion'
CHOSEN=[('A','003',291),('B','006',884),('C','007',851),('D','010',262)]

def main():
    assert not (OUT/'annotation_manifest.json').exists()
    mining=load(OUT/'mining_report.json');assert all(sha(Path(p))==h for p,h in mining['protected_input_hashes'].items())
    available={(r['candidate_id'],int(r['local_frame'])):r for r in read(OUT/'all_mined_candidates.csv')}
    inventory={r['ball_pilot_id']:r for r in read(ROOT/'play_start_audit/ball_pilot_start_inventory.csv')}
    cases=[];rows=[];fps=59.94005994005994
    for display,suffix,t in CHOSEN:
        cid=f'match1_motion_pilot_{suffix}';candidate=available[cid,t];base=SOURCE/'data/balltrack/racketvision/match1'/cid
        target=base/'frames'/f'{t:06d}.jpg';meta=load(base/'runs/threshold030_median120/run_metadata.json')
        assert t-30>=0 and t+30<meta['frame_count'] and suffix not in ['002','008','011','012','013','014','015','016','017','018','019','020']
        assert int(candidate['source_frame'])==int(inventory[cid]['clip_first_source_frame'])+t
        video=OUT/f'context_{display}_{suffix}_{t}_h264.mp4'
        cmd=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1280x800','-r',str(fps),'-i','pipe:0',
            '-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
        n=0
        with (OUT/f'context_{display}_render.log').open('wb') as log:
            proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
            for j in range(t-30,t+31):
                im=cv2.imread(str(base/'frames'/f'{j:06d}.jpg'));assert im is not None and im.shape[:2]==(1080,1920)
                frame=np.full((800,1280,3),22,np.uint8);frame[80:]=cv2.resize(im,(1280,720),interpolation=cv2.INTER_AREA)
                cv2.putText(frame,f'{display} | pilot{suffix} local {j} | TARGET {t} | 0.5x | No model markers',(12,28),0,.72,(245,245,245),1,cv2.LINE_AA)
                cv2.putText(frame,'TARGET FRAME - annotate the separate original image' if j==t else 'Temporal context only; do not copy this frame position',(12,61),0,.62,(80,230,255) if j==t else (225,225,225),1,cv2.LINE_AA)
                if j==t:cv2.rectangle(frame,(1,80),(1278,798),(80,230,255),3)
                for _ in range(2+(30 if j==t else 0)):proc.stdin.write(frame.tobytes());n+=1
            proc.stdin.close();assert proc.wait()==0
        cap=cv2.VideoCapture(str(video));decoded=0
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape[:2]==(800,1280);decoded+=1
        cap.release();assert decoded==n==152
        sample=f'{cid}_frame_{t:06d}'
        cases.append(dict(case_id=display,display_label=display,sample_id=sample,candidate_id=cid,local_frame=t,
            source_frame=int(candidate['source_frame']),source_timestamp_sec=candidate['source_timestamp_sec'],
            source_image=str(target),source_image_sha256=sha(target),context_video=str(video.resolve()),context_video_sha256=sha(video),
            context_local_frames=[t-30,t+30],decoded_video_frames=decoded,source_width=1920,source_height=1080,
            phase_status='AI overview suggests court activity; human phase unknown',role='training_expansion_only'))
        rows.append(dict(sample_id=sample,evaluation_split='training_expansion',candidate_id=cid,play_window_id=inventory[cid]['play_window_id'],sample_order=len(rows)+1,
            local_frame=t,source_frame=candidate['source_frame'],source_timestamp_sec=candidate['source_timestamp_sec'],
            sampling_method='training_only_temporal_jump_active_sampling_not_false_label',prediction_visible=1,prediction_x_px=candidate['model_x_px'],prediction_y_px=candidate['model_y_px'],prediction_confidence=candidate['model_confidence'],
            truth_visibility='unreviewed',truth_x_px='',truth_y_px='',reviewer_id='',label_confidence='',phase_status='unknown',notes='',annotated_at_utc=''))
    assert len(rows)==4 and len({r['candidate_id'] for r in rows})==4
    write(OUT/'manual_center_labels.csv',rows)
    (OUT/'annotation_cases.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    manifest=dict(status='four_training_frames_ready_for_blinded_manual_annotation',cases=len(cases),
        training_only=True,source_clips=[3,6,7,10],existing300_labels_unchanged=True,initial_reviewed_rows=0,
        original_image_native_wh=[1920,1080],model_points_not_rendered_in_annotation_ui=True,
        no_auto_negative_or_center_labels=True,video_decode_verified=True,
        context_warning='Video is context, image is exact target. If ball is invisible choose not_visible; if ambiguous/multiple target unclear choose uncertain. Do not infer a position.',
        selection='Top temporal-deviation candidate per clip, prefer clips not containing the2 old hard cases; excludes old labelled frames and transitions. Not representative evaluation sampling.',
        phase_warning='No new human-confirmed rally boundary; optional phase field remains unknown until reviewed.',
        cases_sha256=sha(OUT/'annotation_cases.json'),initial_labels_sha256=sha(OUT/'manual_center_labels.csv'),
        protected_input_hashes=mining['protected_input_hashes'])
    assert all(sha(Path(p))==h for p,h in manifest['protected_input_hashes'].items())
    (OUT/'annotation_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in manifest.items() if k!='protected_input_hashes'},indent=2))

if __name__=='__main__':main()
