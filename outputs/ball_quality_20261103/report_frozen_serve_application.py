"""Post-generation comparison and a failure review video; no inference edits."""
import csv,json,subprocess
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
from apply_frozen_serve_phases import OUT,FROZEN,preflight
from extract_serve_start_features import ROOT,SOURCE,sha

def main():
    preflight()
    with (ROOT/'play_start_audit/ball_pilot_start_inventory.csv').open(encoding='utf-8-sig',newline='') as f:
        inventory={r['ball_pilot_id']:r for r in csv.DictReader(f)}
    results=[]
    for pid in ['003','008']:
        cid=f'match1_motion_pilot_{pid}';dest=OUT/cid
        p=json.loads((dest/'phases.json').read_text());s=json.loads((dest/'summary.json').read_text())
        protocol=json.loads((dest/'protocol.json').read_text());features=json.loads((dest/'features.json').read_text())
        assert p['frozen_protocol_file_sha256']==sha(FROZEN)
        assert p['runner_sha256']==sha(ROOT/'apply_frozen_serve_phases.py')
        assert p['feature_sha256']==sha(dest/'features.json')
        assert features['protocol_sha256']==sha(dest/'protocol.json')
        assert p['labels_read'] is False and p['serve_outcome_inferred'] is False
        assert all(q['outcome']=='unknown' and q['point_id'] is None for q in p['candidates'])
        entry=inventory[cid]
        if entry['legacy_serve_timestamp_sec']:
            reference=(float(entry['legacy_serve_timestamp_sec'])-float(entry['clip_start_sec']))*protocol['fps']
            reference_status='legacy_impact_timestamp_not_independent_fine_contact_interval'
        else:reference=None;reference_status='no_existing_fine_contact_reference_in_inventory'
        results.append(dict(summary=s,mapped_rally_id=entry['rally_window_candidate_id'],
            reasons=dict(Counter(q['decision'] for q in p['candidates'])),
            legacy_contact_local_frame=reference,reference_status=reference_status,
            phases_sha256=sha(dest/'phases.json'),boundary_accuracy_measured=False,
            conclusion='No retained automatic start; do not interpret this as no rally.'))
    prior=json.loads((ROOT/'play_start_audit/inventory_summary.json').read_text())
    for name,expected in prior['source_hashes'].items():assert sha(Path(name))==expected
    (OUT/'comparison.json').write_text(json.dumps(dict(results=results,frozen_conditions_unchanged=True,
        source_review_tables_unchanged=True,independent_accuracy_established=False,
        decision='Do not gate tracking with these automatic starts. Preserve reviewed/provisional boundaries and do not remove no-output clips from evaluation.'),indent=2),encoding='utf-8')
    (OUT/'verification.json').write_text(json.dumps(dict(frozen_code_parameters_and_calibration_match=True,
        extractor_four_development_frames_equivalent=True,feature_and_runner_hashes_match=True,
        outcomes_remain_unknown=True,source_review_tables_unchanged=True,
        warning='Software/provenance checks, not independent start-detection accuracy.'),indent=2),encoding='utf-8')
    # Show the failure, retaining the serve-action proposal rather than deleting it.
    cid='match1_motion_pilot_003';dest=OUT/cid
    p=json.loads((dest/'phases.json').read_text());motion=json.loads((dest/'motion_context.json').read_text())['frames']
    meta=json.loads((SOURCE/'data/balltrack/racketvision/match1'/cid/'runs/threshold030_median120/run_metadata.json').read_text())
    fps=meta['fps'];start,end=0,240;silent=dest/'failure_review_silent.mp4';final=dest/'failure_review_h264.mp4'
    cmd=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1280x820','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',str(silent)]
    with (dest/'failure_render.log').open('wb') as log:
        process=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
        for t in range(start,end+1):
            im=cv2.imread(str(SOURCE/'data/balltrack/racketvision/match1'/cid/'frames'/f'{t:06d}.jpg'));assert im is not None
            canvas=np.full((820,1280,3),22,np.uint8);canvas[100:]=cv2.resize(im,(1280,720),interpolation=cv2.INTER_AREA)
            text=lambda s,y,c=(235,235,235):cv2.putText(canvas,s,(12,y),cv2.FONT_HERSHEY_SIMPLEX,.61,c,1,cv2.LINE_AA)
            text(f'003 | fixed v3 conditions | local frame {t} | 0.5x | NO RETAINED START',25)
            m=motion[t//2]
            text(f'Four-person quiet evidence: {m["four_player_quiet"]} | no output does not mean no rally',53)
            active=any(q['contact_candidate_first_local_frame']<=t<=q['contact_candidate_last_local_frame'] for q in p['candidates'])
            detail='Yellow = raw action/audio interval155-186; DEFERRED: waiting context missing' if active else 'Waiting-context rule cannot establish the start; contact/outcome remain unconfirmed.'
            color=(0,200,255) if active else (110,110,110)
            text(detail,81,color if active else (215,215,215));cv2.rectangle(canvas,(0,98),(1279,819),color,3)
            if t==170:assert cv2.imwrite(str(dest/'failure_review_preview.png'),canvas)
            for _ in range(2):process.stdin.write(canvas.tobytes())
        process.stdin.close();assert process.wait()==0
    graph=f'[1:a]atrim=start={start/fps:.9f}:end={(end+1)/fps:.9f},asetpts=PTS-STARTPTS,atempo=0.5[a]'
    subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-i',str(silent),'-i',meta['clip'],'-filter_complex',graph,'-map','0:v:0','-map','[a]','-c:v','copy','-c:a','aac','-movflags','+faststart',str(final)],check=True)
    cap=cv2.VideoCapture(str(final));n=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape[:2]==(820,1280);n+=1
    cap.release();assert n==2*(end-start+1)
    probe=json.loads(subprocess.check_output([r'C:\ffmpeg\bin\ffprobe.exe','-v','error','-show_streams','-of','json',str(final)]))
    assert any(s['codec_type']=='audio' for s in probe['streams'])
    (dest/'render_report.json').write_text(json.dumps(dict(decoded_frames=n,audio=True,source_aspect_preserved=True,
        phase_output_sha256=sha(dest/'phases.json'),video=str(final.resolve()),source_frames=[start,end],speed=.5),indent=2),encoding='utf-8')
    print(json.dumps(dict(results=[r['summary'] for r in results],source_tables_unchanged=True,video=str(final)),indent=2))

if __name__=='__main__':main()
