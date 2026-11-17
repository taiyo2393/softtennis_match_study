from pathlib import Path
import cv2,numpy as np
p=Path('outputs/frontcourt_handoff_20260927'); cells=[]
for i in range(63):
 im=cv2.imread(str(p/f'scoreboard_scan_{i//3:02d}.jpg'))
 tile=im[(i%3)*540+36:(i%3)*540+288,:900]
 c=np.zeros((150,160,3),np.uint8); crop=tile[55:201,655:825]
 c[25:145,10:150]=cv2.resize(crop,(140,120))
 cv2.putText(c,f'{i+1:03}',(8,20),0,.6,(255,255,255),1); cells.append(c)
cv2.imwrite(str(p/'game_score_start_overview.jpg'),np.vstack([np.hstack(cells[j:j+7]) for j in range(0,63,7)]))
