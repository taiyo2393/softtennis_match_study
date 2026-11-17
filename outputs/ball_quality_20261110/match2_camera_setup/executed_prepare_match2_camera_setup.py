"""Raw-video phase/camera inspection before any new-match person predictions."""
import sys,json
from pathlib import Path
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,write,sha,dump
from prepare_match2_external_validation import VIDEO,FPS,load

OUT=BASE/'match2_camera_setup'
PARENT=BASE/'match2_external_validation'


def prepare():
    assert not OUT.exists();s=load(PARENT/'source_registration.json')
    assert sha(VIDEO)==s['source_video_sha256']
    references=load(PARENT/'raw_reference_pilot/manifest.json')['items']
    anchor=next(r for r in references if r['source_frame']==54006)
    files=[Path(__file__),PARENT/'publication_hashes.json',PARENT/'preflight_protocol.json',PARENT/'raw_reference_pilot'/anchor['filename']]
    files.extend(n for n in (BASE/'current22_player_inputs').iterdir() if n.is_file())
    OUT.mkdir();dump(OUT/'protocol.json',dict(source_video=str(VIDEO),source_video_sha256=s['source_video_sha256'],
        raw_timeline_seconds=list(range(90,241,5)),
        purpose='Locate actual play/main-camera sequence on raw images; no tracking/pose/ball outputs used.',
        calibration_anchor_source_frame=54006,calibration_anchor_image=str(PARENT/'raw_reference_pilot'/anchor['filename']),
        calibration_anchor_selection='First already-frozen main-view reference in900s preflight clip. Raw image inspected before proposed geometry; not selected using tracking success.',
        calibration_kind='New-camera normalized plane proposal from visible court paint, not surveyed coordinates or human tracking GT.',
        preview_or_label_frames_are_not_independent_matches=True,
        no_replacement_of_prior_fixed_time_windows=True,
        source_clock_or_old_tracking_data_changed=False,
        new_match_tracking_predictions_created=False,
        protected_input_hashes={str(n):sha(n) for n in files}))


def timeline():
    p=load(OUT/'protocol.json');cap=cv2.VideoCapture(p['source_video']);tiles=[];manifest=[]
    for sec in p['raw_timeline_seconds']:
        f=round(sec*FPS);cap.set(1,f);ok,im=cap.read();assert ok and round(cap.get(1))==f+1
        panel=np.zeros((390,640,3),np.uint8);panel[30:]=cv2.resize(im,(640,360),interpolation=cv2.INTER_AREA)
        cv2.putText(panel,f'match2 raw | {sec}s | F{f}',(8,21),cv2.FONT_HERSHEY_SIMPLEX,.64,(240,240,240),1,cv2.LINE_AA)
        name=f'timeline_{sec:04}s_F{f}.jpg';assert cv2.imwrite(str(OUT/name),panel)
        manifest.append(dict(requested_sec=sec,source_frame=f,file=name,model_predictions=False));tiles.append(panel)
    cap.release()
    for first in range(0,len(tiles),12):
        chunk=tiles[first:first+12]
        while len(chunk)%3:chunk.append(np.zeros_like(tiles[0]))
        sheet=np.vstack([np.hstack(chunk[i:i+3]) for i in range(0,len(chunk),3)])
        assert cv2.imwrite(str(OUT/f'timeline_sheet_{first//12+1}.jpg'),sheet)
    dump(OUT/'raw_timeline_manifest.json',dict(items=manifest))
    im=cv2.imread(p['calibration_anchor_image']);assert im.shape[:2]==(1080,1920)
    # Display crop only; no artificial coordinates or original-image edits.
    crop=im[490:620,550:1400]
    assert cv2.imwrite(str(OUT/'calibration_far_lines_detail.jpg'),cv2.resize(crop,(1700,260),interpolation=cv2.INTER_CUBIC))
    print('Prepared31 raw phase thumbnails and main-view court detail.',flush=True)


if __name__=='__main__':{'prepare':prepare,'timeline':timeline}[sys.argv[1]]()
