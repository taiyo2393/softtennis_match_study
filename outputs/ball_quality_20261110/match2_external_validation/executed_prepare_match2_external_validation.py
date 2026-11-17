"""Immutable source registration and prospective clip selection, before tracking."""
import sys,json,subprocess,shutil
from pathlib import Path
from fractions import Fraction
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,SOURCE,read,write,sha,dump

OUT=BASE/'match2_external_validation'
VIDEO=SOURCE/'data/raw/match2.mp4'
FPS=Fraction(60000,1001)
FFPROBE=Path(r'C:\ffmpeg\bin\ffprobe.exe')


def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def prepare():
    assert VIDEO.exists() and not OUT.exists()
    metadata=json.loads(subprocess.check_output([str(FFPROBE),'-v','error','-show_format','-show_streams','-of','json',str(VIDEO)],text=True,encoding='utf-8'))
    v=next(s for s in metadata['streams'] if s['codec_type']=='video')
    assert (v['width'],v['height'])==(1920,1080) and Fraction(v['avg_frame_rate'])==FPS
    assert int(v['nb_frames'])==182506
    protected={str(n):sha(n) for n in [Path(__file__),BASE/'native_body_auxiliary_bundle/publication_hashes.json',
        BASE/'partial_body_identity_trial/publication_hashes.json',ROOT/'partial_body_identity.py',ROOT/'run_partial_body_identity_trial.py',
        ROOT/'court_initialization_guard.py',ROOT/'pair_initialization_guard.py',ROOT/'near_prefix_initialization.py',
        ROOT/'camera_valid_near_prefix.py',ROOT/'run_fixed_cohort_player_case_v3.py',
        SOURCE/'models/yolo11m.pt',SOURCE/'models/yolo11m-pose.pt']}
    protected.update({str(n):sha(n) for n in (BASE/'current22_player_inputs').iterdir() if n.is_file()})
    cases=[]
    for i,second in enumerate([60,300,900],1):
        first=round(second*FPS);last=round((second+15)*FPS)-1
        times=sorted(set(int(round(x)) for x in np.linspace(first,last,31)))
        references=[round((second+offset)*FPS) for offset in [1,7.5,14]]
        assert len(times)==31 and first<=min(references)<=max(references)<=last
        cases.append(dict(clip_id=f'match2_fixed_clip_{i:03}',nominal_start_sec=second,nominal_duration_sec=15,
            first_source_frame=first,last_source_frame=last,source_frames=last-first+1,
            availability_sample_source_frames=times,raw_reference_source_frames=references,
            selection='Fixed elapsed time, before raw clip previews or tracker outputs. Keep camera cuts and failures; no replacement.',
            rally_boundaries_confirmed=False,in_play_confirmed=False,camera_eligibility_pending=True))
    OUT.mkdir();dump(OUT/'source_ffprobe.json',metadata)
    dump(OUT/'source_registration.json',dict(source_video=str(VIDEO),source_video_sha256=sha(VIDEO),bytes=VIDEO.stat().st_size,
        source_url='https://www.youtube.com/watch?v=iRRLb6MLtXc',match='2024全日本選手権男子決勝',
        players=['内本隆文','内田理久','船水颯人','上松俊貴'],
        development_match='2025全日本選手権男子決勝',development_players=['橋場柊一郎','菊山太陽','上岡俊介','丸山海斗'],
        separate_match_and_player_sets_disjoint=True,source_file_unmodified=True,
        dimensions=[v['width'],v['height']],fps=v['avg_frame_rate'],frames=int(v['nb_frames']),
        duration_video_sec=float(v['duration']),codec=v['codec_name'],
        nominal_fps_not_effective_motion_rate=True))
    dump(OUT/'preflight_protocol.json',dict(cases=cases,source_registration_sha256=sha(OUT/'source_registration.json'),
        timestamp='2026-11-10 Asia/Tokyo; do not infer event clock from file modification time',
        before_any_match2_detector_pose_or_association_outputs=True,
        selected_time_clips_not_three_independent_matches_or_rallies=True,
        source_dimensions=[1920,1080],development_dimensions=[3840,2160],
        source_frame_or_clock_shift_allowed=False,source_fps_conversion_allowed=False,
        source_video_or_old_match1_inputs_modified=False,
        cadence_windows_sec=[30,300,900,1800,2700],cadence_window_source_frames=120,
        cadence_measure='Adjacent-frame pixel changes and parity statistics on grayscale thumbnails, core-court ROI; include lag2 and activity measures. No fixed threshold converts motion evidence to true fps.',
        scene_preview_seconds=[0,15,30,45,60,120,300,900,1800,2700],
        camera_and_resolution_adapter_status='Not ready: existing fixed3840/2160 image geometry and feature thresholds must be audited. Do not directly feed1080 rows to legacy4K association. A simple upscale adds no image detail. New camera calibration must be recorded before predictions; development camera calibration cannot be assumed valid.',
        held_out_status='New match has no tracking outputs yet. Calibration/setup may inspect raw images; any algorithm tuning after seeing new tracking errors converts the corresponding evaluation material to development data. Record all interventions.',
        planned_comparison='Freeze applicable legacy native tracking plus candidate auxiliary body association before output. Keep native and body-only states separate, missing coordinates unchanged. Do not choose a different method per clip.',
        reference_plan='Nine raw frames fixed at clip+1,7.5,14 seconds. Future manual labels must allow unplaceable and unknown identity. Do not use AI poses or native boxes as truth; no new labels created in preflight.',
        evaluation='Native box recall/precision and identity switches need independent labels. Position error only when a human reference point is placeable; report missing outputs and clip-level results. Three clips are one-match pilot evidence.',
        outcome_prediction_or_new_winner_labels=False,
        protected_input_hashes=protected))
    print('Registered match2; froze3x15s clips,93 availability times and9 raw reference frames before predictions.',flush=True)


def cadence():
    p=load(OUT/'preflight_protocol.json');registration=load(OUT/'source_registration.json')
    assert sha(VIDEO)==registration['source_video_sha256']
    cap=cv2.VideoCapture(str(VIDEO));assert cap.isOpened()
    records=[];reports=[]
    for sec in p['cadence_windows_sec']:
        first=round(sec*FPS);cap.set(1,first);images=[]
        for i in range(p['cadence_window_source_frames']):
            ok,im=cap.read();assert ok and round(cap.get(1))==first+i+1
            # Fixed image dimensions/crop, independent of person/ball outputs.
            gray=cv2.cvtColor(cv2.resize(im,(480,270),interpolation=cv2.INTER_AREA),cv2.COLOR_BGR2GRAY)
            images.append(gray[81:257,48:432])
        stack=np.asarray(images,dtype=np.float32);diff=np.mean(np.abs(stack[1:]-stack[:-1]),axis=(1,2))
        lag2=np.mean(np.abs(stack[2:]-stack[:-2]),axis=(1,2))
        exact=np.all(stack[1:]==stack[:-1],axis=(1,2))
        for i,value in enumerate(diff):records.append(dict(window_start_sec=sec,from_source_frame=first+i,to_source_frame=first+i+1,
            from_frame_parity=(first+i)%2,mean_absolute_gray_change=float(value),exact_thumbnail_duplicate=int(exact[i]),
            lag2_change_to_next_next=float(lag2[i]) if i<len(lag2) else ''))
        parity={str(k):float(np.median([float(v) for i,v in enumerate(diff) if (first+i)%2==k])) for k in [0,1]}
        ratio=min(parity.values())/max(max(parity.values()),1e-12)
        report=dict(window_start_sec=sec,first_source_frame=first,decoded_frames=len(stack),
            adjacent_pair_count=len(diff),exact_thumbnail_duplicate_pairs=int(exact.sum()),
            adjacent_MAE_median=float(np.median(diff)),adjacent_MAE_q10=float(np.quantile(diff,.1)),adjacent_MAE_q90=float(np.quantile(diff,.9)),
            lag2_MAE_median=float(np.median(lag2)),from_frame_parity_medians=parity,smaller_larger_parity_ratio=ratio,
            interpretation_requires_scene_activity_and_is_not_true_fps_measurement=True)
        reports.append(report);print(report,flush=True)
    cap.release();write(OUT/'cadence_pair_changes.csv',records);dump(OUT/'cadence_result.json',dict(windows=reports,
        sample_windows=5,decoded_frames=600,independent_true_motion_GT=False,frame_interpolation_or_fps_conversion_applied=False))


def previews():
    p=load(OUT/'preflight_protocol.json');cap=cv2.VideoCapture(str(VIDEO));manifest=[];tiles=[]
    for sec in p['scene_preview_seconds']:
        f=round(sec*FPS);cap.set(1,f);ok,im=cap.read();assert ok and round(cap.get(1))==f+1
        tile=cv2.resize(im,(640,360),interpolation=cv2.INTER_AREA)
        panel=np.zeros((390,640,3),np.uint8);panel[30:]=tile
        cv2.putText(panel,f'match2 raw | {sec}s | source F{f}',(8,21),cv2.FONT_HERSHEY_SIMPLEX,.62,(240,240,240),1,cv2.LINE_AA)
        name=f'raw_{sec:04}s_F{f}.jpg';assert cv2.imwrite(str(OUT/name),panel)
        manifest.append(dict(requested_sec=sec,source_frame=f,filename=name,tracker_overlay=False));tiles.append(panel)
    sheet=np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,len(tiles),2)])
    assert cv2.imwrite(str(OUT/'raw_scene_overview.jpg'),sheet)
    references=[]
    dest=OUT/'raw_reference_pilot';dest.mkdir()
    for e in p['cases']:
        for index,f in enumerate(e['raw_reference_source_frames'],1):
            cap.set(1,f);ok,im=cap.read();assert ok and round(cap.get(1))==f+1
            name=f'{e["clip_id"]}_ref{index}_F{f}.png';assert cv2.imwrite(str(dest/name),im)
            references.append(dict(clip_id=e['clip_id'],reference_index=index,source_frame=f,filename=name,
                image_sha256=sha(dest/name),dimensions=[1920,1080],human_labels_present=False,model_overlays_present=False))
    cap.release();dump(OUT/'raw_scene_manifest.json',dict(items=manifest));dump(dest/'manifest.json',dict(items=references,
        sampling_frozen_before_model_outputs=True,manual_identity_or_position_truth_created=False))
    print('Saved10 raw scene previews and9 original-resolution reference PNGs without model overlays.',flush=True)


if __name__=='__main__':{'prepare':prepare,'cadence':cadence,'previews':previews}[sys.argv[1]]()
