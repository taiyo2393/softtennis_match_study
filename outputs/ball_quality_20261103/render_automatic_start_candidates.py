"""Show automatically generated candidates, keeping reviewed outcomes separate."""
import json
import subprocess

import cv2
import numpy as np

from extract_serve_start_features import SOURCE,ROOT

OUT=ROOT/'serve_start_pilot_tiled_v2'


def main():
    cid='match1_motion_pilot_016';dest=OUT/cid
    proposals=json.loads((dest/'proposals.json').read_text())['proposals']
    protocol=json.loads((dest/'protocol.json').read_text());fps=protocol['fps']
    metadata=json.loads((SOURCE/'data/balltrack/racketvision/match1'/cid/'runs/threshold030_median120/run_metadata.json').read_text())
    assert len(proposals)==2,'Review output structure before rendering a two-proposal summary'
    windows=[]
    for p in proposals:
        center=round(p['audio_peak_local_sec']*fps)
        windows.append((max(0,center-65),min(protocol['frames']-1,center+65)))
    video=dest/'automatic_start_candidates_silent.mp4';final=dest/'automatic_start_candidates_h264.mp4'
    command=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1472x640',
        '-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    encoded=0
    with (dest/'render.log').open('wb') as error:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=error)
        for i,(p,(start,end)) in enumerate(zip(proposals,windows)):
            boxes=np.array(p['evidence_person_boxes']);box=np.median(boxes,axis=0)
            left=max(0,min(1920-256,round((box[0]+box[2])/2)-128));top=max(0,min(1080-256,round((box[1]+box[3])/2)-128))
            for t in range(start,end+1):
                im=cv2.imread(str(SOURCE/'data/balltrack/racketvision/match1'/cid/'frames'/f'{t:06d}.jpg'));assert im is not None
                canvas=np.full((640,1472,3),22,np.uint8);canvas[74:614,:960]=cv2.resize(im,(960,540),interpolation=cv2.INTER_AREA)
                canvas[74:586,960:1472]=cv2.resize(im[top:top+256,left:left+256],(512,512),interpolation=cv2.INTER_NEAREST)
                active=p['contact_candidate_first_local_frame']<=t<=p['contact_candidate_last_local_frame']
                color=(0,190,255) if active else (175,175,175)
                cv2.putText(canvas,f'044 | automatic candidate {i+1}/2 | local frame {t} | 0.5x replay',(12,27),cv2.FONT_HERSHEY_SIMPLEX,.65,(235,235,235),1,cv2.LINE_AA)
                cv2.putText(canvas,'Yellow border = proposed contact interval; fault/valid outcome is NOT classified.',(12,55),cv2.FONT_HERSHEY_SIMPLEX,.58,(235,235,235),1,cv2.LINE_AA)
                cv2.rectangle(canvas,(0,72),(1471,616),color,2)
                cv2.putText(canvas,'Right: 2x nearest-neighbor crop',(970,610),cv2.FONT_HERSHEY_SIMPLEX,.55,(235,235,235),1,cv2.LINE_AA)
                for _ in range(2):process.stdin.write(canvas.tobytes());encoded+=1
        process.stdin.close();assert process.wait()==0
    filters=[]
    for i,(start,end) in enumerate(windows):
        filters.append(f'[a{i}]atrim=start={start/fps:.9f}:end={(end+1)/fps:.9f},asetpts=PTS-STARTPTS,atempo=0.5[c{i}]')
    graph='[1:a]asplit=2[a0][a1];'+';'.join(filters)+';[c0][c1]concat=n=2:v=0:a=1[outa]'
    subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-i',str(video),'-i',metadata['clip'],
        '-filter_complex',graph,'-map','0:v:0','-map','[outa]','-c:v','copy','-c:a','aac','-movflags','+faststart',str(final)],check=True)
    cap=cv2.VideoCapture(str(final));decoded=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(640,1472)
        if decoded==2*65:cv2.imwrite(str(dest/'automatic_candidate_preview.png'),im)
        decoded+=1
    cap.release();assert decoded==encoded
    report=dict(video=str(final.resolve()),decoded_frames=decoded,speed=.5,audio_time_stretched=.5,
        context_windows_local_frames=windows,overlays='Automatic proposal intervals only, no reference contact markers or predicted fault/valid labels.',
        visualization='Full view at half size plus selected person256px crop enlarged2x without aspect distortion.')
    (dest/'render_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
