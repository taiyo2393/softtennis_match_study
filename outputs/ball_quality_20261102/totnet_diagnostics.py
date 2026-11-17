"""Fixed first validation clip: all 15 labeled samples, native-scale crops."""
import json
from pathlib import Path

import cv2
from PIL import Image, ImageDraw, ImageFont

from compare_totnet import OUT, LABELS, SOURCE, read, predictions, baseline


def main():
    cid='match1_motion_pilot_001'
    labels=[r for r in read(LABELS) if r['candidate_id']==cid]
    assert len(labels)==15
    choice=json.loads((OUT/'selection.json').read_text())
    net=predictions(read(OUT/'validation_raw.csv'),choice['selected_threshold'])[cid]
    base=baseline(labels)[cid]
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',14)
    titlefont=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',19)
    canvas=Image.new('RGB',(5*280,3*230+78),'#171b24');draw=ImageDraw.Draw(canvas)
    draw.text((12,7),'TOTNet diagnostic: first validation clip 001, all 15 annotated samples',font=titlefont,fill='white')
    draw.text((12,36),'Native-scale 256 x 160 crops | cyan: baseline | pink: TOTNet | green: existing label (provenance unverified)',font=font,fill='white')
    for i,r in enumerate(labels):
        t=int(r['local_frame']);b=base[t];p=net[t]
        path=SOURCE/'data/balltrack/racketvision/match1'/cid/'frames'/f'{t:06d}.jpg'
        im=cv2.imread(str(path));assert im is not None
        h,w=im.shape[:2]
        if r['truth_visibility']=='visible':cx,cy=float(r['truth_x_px']),float(r['truth_y_px'])
        elif int(b['visible']):cx,cy=float(b['x_px']),float(b['y_px'])
        elif int(p['visible']):cx,cy=float(p['x_px']),float(p['y_px'])
        else:cx,cy=w/2,h/2
        x=max(0,min(w-256,round(cx)-128));y=max(0,min(h-160,round(cy)-80))
        ox=(i%5)*280+12;oy=(i//5)*230+80
        crop=Image.fromarray(cv2.cvtColor(im[y:y+160,x:x+256],cv2.COLOR_BGR2RGB))
        cd=ImageDraw.Draw(crop)
        for a,color in ((b,'#00ddff'),(p,'#ff66cc')):
            if int(a['visible']):
                xx=float(a['x_px'])-x;yy=float(a['y_px'])-y
                cd.ellipse((xx-7,yy-7,xx+7,yy+7),outline=color,width=2)
        if r['truth_visibility']=='visible':
            xx=float(r['truth_x_px'])-x;yy=float(r['truth_y_px'])-y
            cd.line((xx-4,yy,xx+4,yy),fill='#50ff70',width=1)
            cd.line((xx,yy-4,xx,yy+4),fill='#50ff70',width=1)
        canvas.paste(crop,(ox,oy))
        draw.text((ox,oy+164),f"frame {t} | label: {r['truth_visibility']}",font=font,fill='white')
        draw.text((ox,oy+184),f"base: {'point' if int(b['visible']) else 'missing'} | TOT: {'point' if int(p['visible']) else 'missing'}",font=font,fill='#cccccc')
    canvas.save(OUT/'validation_001_all_samples.png')
    print(OUT/'validation_001_all_samples.png')


if __name__=='__main__':main()
