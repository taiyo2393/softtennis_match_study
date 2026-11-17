from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import build_044_spatiotemporal_pilot as b
from prepare_056_route_transfer import FRAMES, OFFSET

OUT=b.HERE/'056_post_hit_trial'
OUT.mkdir(exist_ok=True)
BASE=b.PRIOR/'rally_repair_056'
by={int(r['source_frame']):r for r in b.read(BASE/'baseline_all_frames.csv')}
font=ImageFont.truetype(str(b.FONT),20)
for page,times in enumerate([range(154820,154849,2),range(154850,154879,2)],1):
    sheet=Image.new('RGB',(1600,1120),(15,24,34))
    for i,f in enumerate(times):
        im=Image.open(FRAMES/f'{f-OFFSET:06d}.jpg').convert('RGB')
        # Fixed 800x500 crop, shown at 0.5 scale with uniform aspect ratio.
        tile=im.crop((550,380,1350,880)).resize((400,250))
        d=ImageDraw.Draw(tile);r=by[f]
        if r['visible']=='1':
            x=(float(r['x_px'])-550)/2;y=(float(r['y_px'])-380)/2
            d.ellipse((x-6,y-6,x+6,y+6),outline=(255,190,65),width=2)
        x=i%4*400;y=i//4*280
        sheet.paste(tile,(x,y+30));ImageDraw.Draw(sheet).text((x+6,y+3),f'F{f}  model candidate',font=font,fill='white')
    sheet.save(OUT/f'baseline_grid_{page:02d}.jpg',quality=94)
print(OUT)
