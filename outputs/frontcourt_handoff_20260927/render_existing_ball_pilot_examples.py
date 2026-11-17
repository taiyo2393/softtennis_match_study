import csv,cv2,subprocess,json
from pathlib import Path
p=Path(__file__).resolve().parent;root=Path(r'C:\Users\ta239\Research\softtennis_match_study\study');fps=60000/1001;source=root/'data/raw/match1.mp4';cap=cv2.VideoCapture(str(source))
for tag,cid,start in [('001','match1_motion_pilot_001',0),('007','match1_motion_pilot_002',700)]:
 folder=root/'data/balltrack/racketvision/match1'/cid/'runs/threshold030_median120'
 with (folder/'balltrack_predictions.csv').open(encoding='utf-8-sig') as f:rs=list(csv.DictReader(f))
 rs=[r for r in rs if start<=int(r['local_frame'])<start+240];assert len(rs)==240
 first=int(rs[0]['source_frame']);cap.set(1,first);temp=p/f'ball_pilot_{tag}_review_intermediate.mp4';writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1920,1080));assert writer.isOpened()
 for j,r in enumerate(rs):
  assert int(r['source_frame'])==first+j
  ok,im=cap.read();assert ok;im=cv2.resize(im,(1920,1080))
  if int(r['visible']):
   x,y=round(float(r['x_px'])),round(float(r['y_px']));cv2.circle(im,(x,y),12,(255,255,0),2);cv2.putText(im,f'{float(r["confidence"]):.2f}',(x+14,y),0,.55,(255,255,0),1)
  cv2.rectangle(im,(0,1020),(1920,1080),(0,0,0),-1);cv2.putText(im,f'{tag} source F{first+j} | cyan circle = model candidate, not verified ball',(15,1055),0,.65,(255,255,255),1);writer.write(im)
 writer.release()
 out=p/f'ball_pilot_{tag}_candidate_review_h264.mp4'
 subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-i',str(temp),'-ss',str(first/fps),'-t',str(240/fps),'-i',str(source),'-map','0:v:0','-map','1:a:0','-c:v','libx264','-preset','fast','-crf','20','-pix_fmt','yuv420p','-c:a','aac','-shortest','-movflags','+faststart',str(out)],check=True)
 print(tag,'source frame',first,'frames',len(rs),'with model candidates',sum(int(r['visible']) for r in rs))
cap.release()
