"""Review044 identities with a fixed far-side crop and preserved aspect."""
import json,subprocess
from collections import defaultdict
import cv2
import numpy as np
from track_044_far_pair import OUT,INIT,END,FIRST,image_at,a
from extract_serve_start_features import ROOT,SOURCE,sha

COLORS=dict(near_front=(80,255,80),near_rear=(255,190,40),far_rear=(255,80,240),far_front=(0,230,255))
NAMES=dict(zip(a.SLOTS,('N1','N2','F1','F2')))

def main():
    path=OUT/'observed_player_paths.csv';by=defaultdict(list)
    for r in a.read(path):by[int(r['source_frame'])].append(r)
    fps=59.94005994005994;silent=OUT/'person_review_silent.mp4';final=OUT/'person_review_h264.mp4'
    command=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1440x1320','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','18','-pix_fmt','yuv420p',str(silent)]
    with (OUT/'render.log').open('wb') as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        for f in range(INIT,END+1):
            im=image_at(f);canvas=np.full((1320,1440,3),22,np.uint8)
            canvas[80:890]=cv2.resize(im,(1440,810),interpolation=cv2.INTER_AREA)
            canvas[910:1310]=cv2.resize(im[430:730,600:1680],(1440,400),interpolation=cv2.INTER_LINEAR)
            text=lambda s,x,y,size=.64,c=(235,235,235):cv2.putText(canvas,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,c,1,cv2.LINE_AA)
            text(f'044 | local frame {f-FIRST} | F1 purple = initial far server; F2 yellow = partner | 0.5x',12,27)
            text('Observed boxes only; gaps remain blank. Identity correspondence requires review.',12,56,.61)
            for r in by[f]:
                slot=r['player_slot'];color=COLORS[slot]
                x1,y1,x2,y2=[float(r[k])*.5 for k in ['x1','y1','x2','y2']]
                p1=(round(.75*x1),80+round(.75*y1));p2=(round(.75*x2),80+round(.75*y2))
                cv2.rectangle(canvas,p1,p2,color,2)
                text(NAMES[slot],min(1390,max(0,p1[0])),max(98,p1[1]-6),.67,color)
                if slot.startswith('far_'):
                    # All boxes in this run fit the fixed far crop. Clip only drawing.
                    l=round((x1-600)*4/3);rr=round((x2-600)*4/3)
                    top=910+round((y1-430)*4/3);bottom=910+round((y2-430)*4/3)
                    cv2.rectangle(canvas,(max(0,l),max(910,top)),(min(1439,rr),min(1309,bottom)),color,2)
                    text(NAMES[slot],max(0,min(1390,l)),max(930,top-5),.8,color)
            slots={r['player_slot'] for r in by[f]}
            state=' | '.join(f'{NAMES[s]}: '+('observed' if s in slots else 'missing') for s in ['far_rear','far_front'])
            text('Fixed far crop (same aspect ratio; 4/3 enlargement) | '+state,12,907,.55)
            if f-FIRST in [725,850,1000,1103]:cv2.imwrite(str(OUT/f'person_preview_local{f-FIRST}.jpg'),canvas)
            for _ in range(2):process.stdin.write(canvas.tobytes())
        process.stdin.close();assert process.wait()==0
    meta=json.loads((SOURCE/'data/balltrack/racketvision/match1/match1_motion_pilot_016/runs/threshold030_median120/run_metadata.json').read_text())
    start=INIT-FIRST;end=END-FIRST
    graph=f'[1:a]atrim=start={start/fps:.9f}:end={(end+1)/fps:.9f},asetpts=PTS-STARTPTS,atempo=0.5[a]'
    subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-i',str(silent),'-i',meta['clip'],'-filter_complex',graph,'-map','0:v:0','-map','[a]','-c:v','copy','-c:a','aac','-movflags','+faststart',str(final)],check=True)
    cap=cv2.VideoCapture(str(final));n=0
    while True:
        ok,frame=cap.read()
        if not ok:break
        assert frame.shape[:2]==(1320,1440);n+=1
    cap.release();assert n==2*(END-INIT+1)
    probe=json.loads(subprocess.check_output([r'C:\ffmpeg\bin\ffprobe.exe','-v','error','-show_streams','-of','json',str(final)]))
    assert any(s['codec_type']=='audio' for s in probe['streams'])
    (OUT/'render_report.json').write_text(json.dumps(dict(video=str(final.resolve()),decoded_frames=n,audio=True,
        source_local_frames=[start,end],speed=.5,source_aspect_preserved=True,
        fixed_crop_source_pixels=[600,430,1680,730],crop_enlargement=4/3,paths_sha256=sha(path),identity_accuracy_established=False),indent=2),encoding='utf-8')
    print(final,'decoded',n,flush=True)

if __name__=='__main__':main()
