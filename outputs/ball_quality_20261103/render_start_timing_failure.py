"""Show the fixed-parameter timing failure, not only the development success."""
import json
import subprocess
import cv2
import numpy as np
from extract_serve_start_features import SOURCE,ROOT


def main():
    dest=ROOT/'serve_start_pilot_tiled_v2/match1_motion_pilot_002'
    proposals=json.loads((dest/'proposals.json').read_text())['proposals']
    p=proposals[2];assert p['contact_candidate_first_local_frame']==709
    start,end=680,790;fps=59.94005994005994
    video=dest/'timing_failure_silent.mp4';final=dest/'timing_failure_h264.mp4'
    command=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1280x800',
        '-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    with (dest/'timing_failure_render.log').open('wb') as err:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=err)
        for t in range(start,end+1):
            im=cv2.imread(str(SOURCE/'data/balltrack/racketvision/match1/match1_motion_pilot_002/frames'/f'{t:06d}.jpg'));assert im is not None
            canvas=np.full((800,1280,3),22,np.uint8);canvas[80:800]=cv2.resize(im,(1280,720),interpolation=cv2.INTER_AREA)
            auto=709<=t<=740;reference=747<=t<=758
            text=lambda s,y,color=(235,235,235):cv2.putText(canvas,s,(12,y),cv2.FONT_HERSHEY_SIMPLEX,.64,color,1,cv2.LINE_AA)
            text(f'002 local frame {t} | fixed-parameter application | 0.5x',27)
            text('Orange=automatic candidate709-740 | cyan=prior AI contact bracket747-758 (not exact ground truth)',57)
            color=(0,180,255) if auto else (255,220,0) if reference else (110,110,110)
            cv2.rectangle(canvas,(0,78),(1279,799),color,3)
            for _ in range(2):process.stdin.write(canvas.tobytes())
        process.stdin.close();assert process.wait()==0
    metadata=json.loads((SOURCE/'data/balltrack/racketvision/match1/match1_motion_pilot_002/runs/threshold030_median120/run_metadata.json').read_text())
    graph=f'[1:a]atrim=start={start/fps:.9f}:end={(end+1)/fps:.9f},asetpts=PTS-STARTPTS,atempo=0.5[a]'
    subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-i',str(video),'-i',metadata['clip'],
        '-filter_complex',graph,'-map','0:v:0','-map','[a]','-c:v','copy','-c:a','aac','-movflags','+faststart',str(final)],check=True)
    cap=cv2.VideoCapture(str(final));decoded=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(800,1280);decoded+=1
    cap.release();assert decoded==2*(end-start+1)
    (dest/'timing_failure_render_report.json').write_text(json.dumps(dict(video=str(final.resolve()),decoded_frames=decoded,
        automatic_interval=[709,740],existing_AI_reference_interval=[747,758],reference_is_not_independent_exact_groundtruth=True),indent=2),encoding='utf-8')
    print(final)


if __name__=='__main__':main()
