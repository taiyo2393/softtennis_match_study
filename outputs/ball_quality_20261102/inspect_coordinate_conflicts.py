"""Inspection media and cache/clip consistency; never rewrite labels."""
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from build_joint_failure_review import COMPARE,OUT,SOURCE,getcases,frame,read


def main():
    rows=read(COMPARE/'test_frame_comparison.csv');reports=[]
    for case in getcases():
        if case['case_id'] not in ('08_013_f435','23_018_f216'):continue
        cid=case['candidate_id'];t=int(case['local_frame'])
        r=next(r for r in rows if r['candidate_id']==cid and int(r['local_frame'])==t)
        points=[(float(r[f'{name}_x_px']),float(r[f'{name}_y_px'])) for name in ('baseline','totnet')]
        points.append((float(r['truth_x_px']),float(r['truth_y_px'])))
        x=round(sum(p[0] for p in points)/3)-192;y=round(sum(p[1] for p in points)/3)-108
        font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',17)
        canvas=Image.new('RGB',(1152,550),'#171b24');d=ImageDraw.Draw(canvas)
        d.text((8,3),f"{case['case_id']} | native 384x216 crops | top raw / bottom marked",font=font,fill='white')
        d.text((8,27),'green: existing label | cyan: baseline | pink: TOTNet; markers only on target frame',font=font,fill='white')
        for col,delta in enumerate((-2,0,2)):
            im=frame(case,t+delta);patch=Image.fromarray(cv2.cvtColor(im[y:y+216,x:x+384],cv2.COLOR_BGR2RGB))
            canvas.paste(patch,(col*384,75));canvas.paste(patch,(col*384,315))
            d.text((col*384+7,52),f'frame {t+delta}',font=font,fill='white')
            if delta==0:
                for (px,py),color in zip(points,('#00ddff','#ff66cc','#50ff70')):
                    xx=col*384+px-x;yy=315+py-y
                    d.ellipse((xx-8,yy-8,xx+8,yy+8),outline=color,width=2)
        canvas.save(OUT/f"{case['case_id']}_coordinate_conflict.png")
        base=SOURCE/'data/balltrack/racketvision/match1'/cid
        meta=json.loads((base/'runs/threshold030_median120/run_metadata.json').read_text())
        cap=cv2.VideoCapture(meta['clip']);assert cap.isOpened()
        # Decode from clip start to avoid trusting random access seek accuracy.
        decoded=None
        for i in range(t+1):
            ok,decoded=cap.read();assert ok,(cid,i)
        cap.release();cached=frame(case,t)
        mad=float(np.abs(decoded.astype(np.float32)-cached.astype(np.float32)).mean())
        # Recompress the decoded source like a JPEG cache; exact identity is only reported if verified.
        matches=[]
        for quality in (90,95,100):
            ok,encoded=cv2.imencode('.jpg',decoded,[cv2.IMWRITE_JPEG_QUALITY,quality]);assert ok
            rec=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
            if np.array_equal(rec,cached):matches.append(quality)
        reports.append(dict(case_id=case['case_id'],clip=meta['clip'],local_frame=t,
                            sequential_decode_cache_mean_abs_difference=mad,
                            jpeg_reencoding_exact_match_qualities=matches,
                            existing_label=points[2],baseline=points[0],totnet=points[1]))
    (OUT/'coordinate_conflict_report.json').write_text(json.dumps(reports,indent=2),encoding='utf-8')
    print(json.dumps(reports,indent=2))


if __name__=='__main__':main()
