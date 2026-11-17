"""Inspect all changed points, not only sparsely annotated frames."""
import csv
import json
from pathlib import Path

import cv2
from PIL import Image,ImageDraw,ImageFont

from temporal_consensus_guard import OUT,BASE,SOURCE,LABEL,read,sha

DEST=OUT/'suppression_audit_20261103'


def main():
    DEST.mkdir(exist_ok=True)
    validation=json.loads((OUT/'validation_diagnostics.json').read_text())['diagnostics']
    test=json.loads((OUT/'test_results.json').read_text())['diagnostics']
    annotations={(r['candidate_id'],int(r['local_frame'])):r for r in read(LABEL)}
    cases=[]
    for split,diags in (('validation',validation),('test',test)):
        for d in diags:
            cid=d['candidate_id'];t=d['local_frame'];p=read(BASE/cid/'balltrack_predictions.csv')[t]
            cases.append(dict(case_id=f"{len(cases)+1:02d}_{cid[-3:]}_f{t}",evaluation_split=split,
                candidate_id=cid,local_frame=t,source_timestamp_sec=p['source_timestamp_sec'],
                withheld_x_px=float(p['x_px']),withheld_y_px=float(p['y_px']),
                existing_annotation=annotations.get((cid,t)),diagnostic=d,
                review_scope='AI visual diagnostic; independent review pending'))
    (DEST/'cases.json').write_text(json.dumps(cases,indent=2,ensure_ascii=False),encoding='utf-8')
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',16)
    for page in range((len(cases)+4)//5):
        canvas=Image.new('RGB',(960,1220),'#171b24');draw=ImageDraw.Draw(canvas)
        draw.text((6,5),'Every temporal-only withheld output | raw native 320x180 crops | t-4, t, t+4',font=font,fill='white')
        draw.text((6,27),'Fixed crop around withheld coordinate; no position marks. Orange frame is target time.',font=font,fill='white')
        for row,c in enumerate(cases[page*5:page*5+5]):
            cid=c['candidate_id'];t=c['local_frame'];top=60+row*230
            x=max(0,min(1920-320,round(c['withheld_x_px'])-160));y=max(0,min(1080-180,round(c['withheld_y_px'])-90))
            draw.text((5,top),f"{c['case_id']} | error {c['diagnostic']['error_px']:.0f}px | annotation {'yes' if c['existing_annotation'] else 'none'}",font=font,fill='white')
            for col,delta in enumerate((-4,0,4)):
                f=t+delta
                im=cv2.imread(str(SOURCE/'data/balltrack/racketvision/match1'/cid/'frames'/f'{f:06d}.jpg'));assert im is not None
                canvas.paste(Image.fromarray(cv2.cvtColor(im[y:y+180,x:x+320],cv2.COLOR_BGR2RGB)),(col*320,top+25))
                draw.text((col*320+4,top+27),str(f),font=font,fill='#00ff88',stroke_width=1,stroke_fill='black')
                if delta==0:draw.rectangle((320,top+25,639,top+204),outline='#ffaa00',width=2)
        canvas.save(DEST/f'temporal_withheld_page_{page+1:02d}.jpg',quality=95)
    summary=dict(withheld_frames=len(cases),validation=len(validation),test=len(test),
                 overlap_with_existing_annotation=sum(c['existing_annotation'] is not None for c in cases),
                 source_labels_sha256=sha(LABEL),warning='Counts describe changed outputs, not confirmed errors removed.')
    (DEST/'inventory.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
