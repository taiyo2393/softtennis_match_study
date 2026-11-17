"""One video-based identity question; no coordinates or labels overwritten."""
import sys,subprocess,shutil
import cv2,numpy as np
from PIL import Image,ImageDraw,ImageFont
import build_044_spatiotemporal_pilot as b
from prepare_056_route_transfer import OUT,FRAMES,OFFSET

def main():
    ball={int(r['source_frame']):r for r in b.read(OUT/'ball_image_observations.csv')};target=154780
    fps=b.FPS/4;font={s:ImageFont.truetype(str(b.FONT),s) for s in [21,26,30]}
    proc=subprocess.Popen(['C:/ffmpeg/bin/ffmpeg.exe','-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x720','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(OUT/'F154780_ball_identity_review.mp4')],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    mapping=[]
    for f in range(154770,154787):
        im=cv2.imread(str(FRAMES/f'{f-OFFSET:06d}.jpg'));assert im is not None
        image=Image.new('RGB',(1920,720),(12,22,32));d=ImageDraw.Draw(image)
        d.text((22,13),'056：赤丸の候補は実球か（F154780で停止）',font=font[30],fill=(235,240,245))
        d.text((22,58),f'元フレーム {f} ｜ 4分の1速＋停止 ｜ 左：全体、右：固定範囲の拡大',font=font[21],fill=(235,240,245))
        rgb=cv2.cvtColor(im,cv2.COLOR_BGR2RGB);image.paste(Image.fromarray(rgb).resize((960,540),Image.Resampling.LANCZOS),(0,105))
        image.paste(Image.fromarray(rgb[400:760,650:1290]).resize((960,540),Image.Resampling.LANCZOS),(960,105));d=ImageDraw.Draw(image)
        r=ball.get(f)
        if r and r['observation_present']=='1':
            x,y=float(r['image_x_px']),float(r['image_y_px']);color=(255,85,85) if f==target else (85,230,150)
            d.ellipse((x/2-7,105+y/2-7,x/2+7,105+y/2+7),outline=color,width=3)
            xx=960+(x-650)*1.5;yy=105+(y-400)*1.5;d.ellipse((xx-12,yy-12,xx+12,yy+12),outline=color,width=3)
        d.text((22,670),'赤丸の1点だけを確認。前後の緑丸も正解と決め付けていません。',font=font[26],fill=(255,200,120))
        repeats=31 if f==target else 1
        if f==target:image.save(OUT/'F154780_ball_identity_review.jpg',quality=95)
        for _ in range(repeats):
            proc.stdin.write(np.asarray(image).tobytes());mapping.append(dict(display_frame=len(mapping),source_frame=f,pause=int(f==target)))
    proc.stdin.close();err=proc.stderr.read();assert proc.wait()==0,err
    b.write(OUT/'one_point_review_source_map.csv',mapping)
    cap=cv2.VideoCapture(str(OUT/'F154780_ball_identity_review.mp4'));count=round(cap.get(7));assert count==len(mapping)==47
    cap.set(1,20);ok,frame=cap.read();assert ok and frame.shape==(720,1920,3);cap.release()
    b.dump(OUT/'one_point_review_manifest.json',dict(target_frame=target,first_frame=154770,last_frame=154786,encoded_frames=count,
        crop_native_xyxy=[650,400,1290,760],uniform_enlargement=1.5,source_aspect_preserved=True,
        question_scope='Only whether original red candidate at154780 encloses actual rally ball. Not exact center/other frames/event count GT.',
        no_label_or_coordinate_changes=True))
    shutil.copy2(__file__,OUT/'executed_one_point_review.py')
    b.dump(OUT/'publication_hashes.json',{str(n.relative_to(OUT)):b.sha(n) for n in sorted(OUT.rglob('*')) if n.is_file() and n.name!='publication_hashes.json'})

if __name__=='__main__':main()
