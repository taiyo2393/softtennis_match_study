"""Verify preservation and render every withheld candidate for inspection."""
import json
import math
from pathlib import Path

import cv2
from PIL import Image,ImageDraw,ImageFont

from evaluate_glove_guard import OUT,SOURCE,BASE,LABEL,read,sha


def main():
    all_flags=[]
    for cid in ('match1_motion_pilot_013','match1_motion_pilot_018'):
        dest=OUT/'continuous_predictions'/cid
        report=json.loads((dest/'report.json').read_text())
        assert sha(BASE/cid/'balltrack_predictions.csv')==report['original_csv_sha256']
        original=read(BASE/cid/'balltrack_predictions.csv');new=read(dest/'balltrack_predictions.csv')
        assert len(original)==len(new)
        for r,p in zip(original,new):
            assert r['local_frame']==p['local_frame']
            assert all(p['raw_'+k]==r[k] for k in ('x_px','y_px','visible','confidence'))
            if int(p['glove_guard_flag']):
                assert int(p['visible'])==0 and float(p['x_px'])==float(p['y_px'])==0
                all_flags.append(dict(candidate_id=cid,local_frame=int(p['local_frame']),x_px=float(p['raw_x_px']),y_px=float(p['raw_y_px'])))
            else:assert all(r[k]==p[k] for k in ('x_px','y_px','visible','confidence'))
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',17)
    canvas=Image.new('RGB',(4*320,math.ceil(len(all_flags)/4)*210+50),'#171b24');d=ImageDraw.Draw(canvas)
    d.text((8,8),'All 13 withheld candidates in two continuous clips | native 320x180 raw crops',font=font,fill='white')
    for i,r in enumerate(all_flags):
        im=cv2.imread(str(SOURCE/'data/balltrack/racketvision/match1'/r['candidate_id']/'frames'/f"{r['local_frame']:06d}.jpg"));assert im is not None
        x=max(0,min(1920-320,round(r['x_px'])-160));y=max(0,min(1080-180,round(r['y_px'])-90))
        ox=i%4*320;oy=i//4*210+50
        canvas.paste(Image.fromarray(cv2.cvtColor(im[y:y+180,x:x+320],cv2.COLOR_BGR2RGB)),(ox,oy))
        d.text((ox+5,oy+182),f"{r['candidate_id'][-3:]} frame {r['local_frame']}",font=font,fill='white')
    canvas.save(OUT/'all_continuous_withheld_candidates.jpg',quality=95)
    protocol=json.loads((OUT/'guard_protocol.json').read_text());assert sha(LABEL)==protocol['label_sha256']
    result=dict(continuous_frames_verified=690,withheld_candidates=len(all_flags),all_raw_coordinates_preserved=True,
        all_nonwithheld_rows_unchanged=True,baseline_source_files_unchanged=True,original_annotation_unchanged=True,
        inspected_targets=all_flags,scope='Integrity checks and diagnostic media only, not independent localization accuracy.')
    (OUT/'output_integrity.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='inspected_targets'},indent=2))
    print([(r['candidate_id'][-3:],r['local_frame']) for r in all_flags])


if __name__=='__main__':main()
