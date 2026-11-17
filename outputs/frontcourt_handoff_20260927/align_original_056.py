import subprocess,cv2,numpy as np
from pathlib import Path
p=Path(__file__).resolve().parent;src=r'C:\Users\ta239\Research\softtennis_match_study\study\data\raw\match1_original.mp4';tiles=[]
for t in [2745,2750,2755,2760,2765,2770]:
 r=subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-v','error','-ss',str(t),'-i',src,'-frames:v','1','-vf','scale=960:540','-f','rawvideo','-pix_fmt','bgr24','pipe:1'],capture_output=True,check=True);im=np.frombuffer(r.stdout,np.uint8).reshape(540,960,3).copy();cv2.putText(im,f'Original t={t}',(20,520),0,1,(0,255,255),2);tiles.append(im)
cv2.imwrite(str(p/'original_056_alignment_overview.jpg'),np.vstack([np.hstack(tiles[i:i+2]) for i in range(0,6,2)]))
