"""Render retained start and suppressed rally actions, full view with audio."""
import json,subprocess
import cv2
import numpy as np
from extract_serve_start_features import ROOT,SOURCE,sha

def main():
    folder=ROOT/'serve_phase_pilot_v3/match1_motion_pilot_002'
    data=json.loads((folder/'phases.json').read_text());fps=json.loads((folder/'protocol.json').read_text())['fps']
    start,end=650,1000;silent=folder/'phase_review_silent.mp4';final=folder/'phase_review_h264.mp4'
    cmd=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1280x820','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',str(silent)]
    with (folder/'phase_review_render.log').open('wb') as err:
        proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=err)
        for t in range(start,end+1):
            im=cv2.imread(str(SOURCE/'data/balltrack/racketvision/match1/match1_motion_pilot_002/frames'/f'{t:06d}.jpg'));assert im is not None
            canvas=np.full((820,1280,3),22,np.uint8);canvas[100:]=cv2.resize(im,(1280,720),interpolation=cv2.INTER_AREA)
            active=next((q for q in data['candidates'] if q['contact_candidate_first_local_frame']<=t<=q['contact_candidate_last_local_frame']),None)
            accepted=[q for q in data['accepted_candidates'] if q['raised_first_local_frame']<=t]
            last=accepted[-1] if accepted else None
            phase='WAITING CANDIDATE' if last is None or last['raised_first_local_frame']<650 else 'SERVICE MOTION CANDIDATE' if t<=last['contact_candidate_last_local_frame'] else 'ACTIVITY / END UNKNOWN'
            color=(80,220,100) if active and active['accepted_candidate'] else (150,150,150) if active else (70,70,70)
            text=lambda s,y,c=(235,235,235):cv2.putText(canvas,s,(12,y),cv2.FONT_HERSHEY_SIMPLEX,.63,c,1,cv2.LINE_AA)
            text(f'002 | local frame {t} | phase: {phase} | 0.5x',25)
            text('GREEN = retained start candidate | GREY = action suppressed during activity',53)
            detail='Contact is a candidate interval; validity and point end are not inferred.'
            if active:
                a=active['contact_candidate_first_local_frame'];b=active['contact_candidate_last_local_frame']
                detail=f'Proposal {a}-{b}: '+('RETAINED (not verified contact)' if active['accepted_candidate'] else 'SUPPRESSED (activity lock; not a new start)')
            text(detail,81,color if active else (210,210,210));cv2.rectangle(canvas,(0,98),(1279,819),color,3)
            if t==760:assert cv2.imwrite(str(folder/'phase_review_preview.png'),canvas)
            for _ in range(2):proc.stdin.write(canvas.tobytes())
        proc.stdin.close();assert proc.wait()==0
    meta=json.loads((SOURCE/'data/balltrack/racketvision/match1/match1_motion_pilot_002/runs/threshold030_median120/run_metadata.json').read_text())
    graph=f'[1:a]atrim=start={start/fps:.9f}:end={(end+1)/fps:.9f},asetpts=PTS-STARTPTS,atempo=0.5[a]'
    subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-i',str(silent),'-i',meta['clip'],'-filter_complex',graph,'-map','0:v:0','-map','[a]','-c:v','copy','-c:a','aac','-movflags','+faststart',str(final)],check=True)
    cap=cv2.VideoCapture(str(final));count=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(820,1280);count+=1
    cap.release();assert count==2*(end-start+1)
    probe=json.loads(subprocess.check_output([r'C:\ffmpeg\bin\ffprobe.exe','-v','error','-show_streams','-of','json',str(final)]))
    assert any(s['codec_type']=='audio' for s in probe['streams'])
    (folder/'render_report.json').write_text(json.dumps(dict(video=str(final.resolve()),decoded_frames=count,
        source_local_frames=[start,end],speed=.5,audio=True,source_content_aspect_ratio_preserved=True,
        phase_output_sha256=sha(folder/'phases.json'),accuracy_claim=False),indent=2),encoding='utf-8')
    print(final)

if __name__=='__main__':main()
