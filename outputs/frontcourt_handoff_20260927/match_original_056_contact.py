import subprocess,cv2,numpy as np,json
from pathlib import Path
p=Path(__file__).resolve().parent;root=Path(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study\data\raw');fps=60000/1001
cap=cv2.VideoCapture(str(root/'match1.mp4'));cap.set(1,154081);ok,target=cap.read();assert ok;cap.release();target=cv2.resize(target,(960,540));roi=(slice(180,440),slice(180,400));tg=cv2.cvtColor(target,cv2.COLOR_BGR2GRAY)[roi].astype(np.float32)
cmd=[r'C:\ffmpeg\bin\ffmpeg.exe','-v','error','-ss','2747','-i',str(root/'match1_original.mp4'),'-frames:v','180','-vf','scale=960:540','-pix_fmt','bgr24','-f','rawvideo','pipe:1'];res=subprocess.run(cmd,capture_output=True,check=True);a=np.frombuffer(res.stdout,np.uint8).reshape(-1,540,960,3)
errors=[]
for im in a:
 gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)[roi].astype(np.float32);errors.append(float(np.mean(np.abs(gray-tg))))
best=int(np.argmin(errors));new_first=int(np.ceil(2747*fps-1e-6));newframe=new_first+best;offset=newframe-154081
print('best',best,'original frame',newframe,'offset',offset,'error',errors[best]);out=dict(old_frame=154081,new_frame=newframe,new_first_decoded_frame_assuming_CFR=new_first,frame_offset_candidate=offset,scope='local 056 alignment; whole-video mapping not certified',mean_roi_absolute_difference=errors[best]);(p/'056_original_alignment.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
cv2.putText(target,'Edited F154081',(20,520),0,1,(0,255,255),2);im=a[best].copy();cv2.putText(im,f'Original F{newframe}',(20,520),0,1,(0,255,255),2);cv2.imwrite(str(p/'056_original_edited_matched_frame.jpg'),np.hstack([target,im]))
lo=max(0,best-4);hi=min(len(a),best+7);tiles=[]
for k in range(lo,hi):
 im=a[k].copy();crop=im[175:435,180:400].copy();cv2.putText(crop,f'F{new_first+k}',(3,25),0,.5,(0,255,255),1);tiles.append(crop)
while len(tiles)%4:tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(str(p/'056_original_contact_sequence.jpg'),np.vstack([np.hstack(tiles[k:k+4]) for k in range(0,len(tiles),4)]))
