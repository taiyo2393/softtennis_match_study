"""Scene-aware temporal preview after the existing glove guard."""
import csv
import json
import subprocess

import cv2
import numpy as np

from temporal_consensus_guard import ROOT,OUT,SOURCE,read,sha,camera_features,camera_mask,filter_rows


def main():
    protocol=json.loads((OUT/'preview_protocol.json').read_text());reports=[]
    for cid in ('match1_motion_pilot_013','match1_motion_pilot_018'):
        original=ROOT/'glove_guard/continuous_predictions'/cid/'balltrack_predictions.csv'
        rows=read(original);scores=camera_features(cid)
        output,diagnostics=filter_rows(rows,scores,protocol['radius'],protocol['camera'])
        dest=OUT/'scene_aware_preview'/cid;dest.mkdir(parents=True,exist_ok=True)
        assert not (dest/'balltrack_predictions.csv').exists()
        with (dest/'balltrack_predictions.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(output[0]));writer.writeheader();writer.writerows(output)
        reports.append(dict(candidate_id=cid,frames=len(rows),additional_withheld=len(diagnostics),diagnostics=diagnostics,
            input_sha256=sha(original),output_sha256=sha(dest/'balltrack_predictions.csv'),camera_masked_frames=np.flatnonzero(camera_mask(scores,True)).tolist()))
        if not cid.endswith('018'):continue
        video=dest/'temporal_review_slow_h264.mp4'
        fps=59.94005994005994
        command=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1600x640',
                 '-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
        with (dest/'ffmpeg.log').open('wb') as err:
            process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=err)
            for t,p in enumerate(output):
                im=cv2.imread(str(SOURCE/'data/balltrack/racketvision/match1'/cid/'frames'/f'{t:06d}.jpg'));assert im is not None
                canvas=np.full((640,1600,3),22,np.uint8)
                canvas[74:614,:960]=cv2.resize(im,(960,540),interpolation=cv2.INTER_AREA)
                left,top=600,420
                canvas[74:434,960:1600]=im[top:top+360,left:left+640]
                txt=lambda s,x,y,size=.65,color=(235,235,235):cv2.putText(canvas,s,(x,y),cv2.FONT_HERSHEY_SIMPLEX,size,color,1,cv2.LINE_AA)
                txt(f'018  frame {t} | temporal + scene guard | 0.25x replay',12,27)
                txt('Green=retained | orange X=withheld raw candidate | right=native-scale fixed crop',12,55)
                scene=bool(camera_mask(scores,True)[t])
                if scene:reason='CAMERA TRANSITION: trajectory suspended'
                elif int(p['temporal_guard_flag']):reason='TEMPORAL: unsupported excursion'
                elif int(p['glove_guard_flag']):reason='GLOVE: raised wrist candidate'
                elif int(p['visible']):reason='DETECTION RETAINED'
                else:reason='NO BASELINE DETECTION'
                txt(reason,974,466,.57)
                txt('No interpolated ball positions.',974,500,.57)
                txt('Raw detections retained in CSV.',974,534,.57)
                txt('Experimental development preview.',974,568,.57)
                if int(p['raw_visible']):
                    x=round(float(p['raw_x_px']));y=round(float(p['raw_y_px']))
                    withheld=int(p['visible'])==0;color=(0,175,255) if withheld else (80,255,100)
                    points=[(x//2,74+y//2)]
                    if left<=x<left+640 and top<=y<top+360:points.append((960+x-left,74+y-top))
                    for point in points:
                        if withheld:cv2.drawMarker(canvas,point,color,cv2.MARKER_TILTED_CROSS,16,2)
                        else:cv2.circle(canvas,point,7,color,1)
                if scene:cv2.rectangle(canvas,(0,72),(1599,616),(0,175,255),2)
                for _ in range(4):process.stdin.write(canvas.tobytes())
            process.stdin.close();assert process.wait()==0
        cap=cv2.VideoCapture(str(video));decoded=0
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape[:2]==(640,1600)
            if decoded in (208*4,229*4,230*4):cv2.imwrite(str(dest/f'frame_{decoded//4}_preview.png'),im)
            decoded+=1
        cap.release();assert decoded==len(rows)*4
        reports[-1].update(video=str(video.resolve()),decoded_video_frames=decoded,speed=.25)
    (OUT/'preview_report.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print(json.dumps([{k:v for k,v in r.items() if k!='diagnostics'} for r in reports],indent=2))


if __name__=='__main__':main()
