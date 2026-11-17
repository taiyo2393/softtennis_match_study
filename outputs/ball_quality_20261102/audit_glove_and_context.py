"""Validation error inspection and note-based context inventory, never new truth labels."""
import csv
import json
import math
from collections import Counter
from pathlib import Path

import cv2
from PIL import Image,ImageDraw,ImageFont

SOURCE=Path(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study')
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'glove_guard';OUT.mkdir(exist_ok=True)
BASE=ROOT.parent/'tracking_quality_20261102/ball_aligned'


def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    labels=read(SOURCE/'annotations/balltrack_groundtruth_match1_pilot.csv')
    baseline={cid:read(BASE/cid/'balltrack_predictions.csv') for cid in sorted({r['candidate_id'] for r in labels})}
    errors=[];inventory=[]
    for r in labels:
        p=baseline[r['candidate_id']][int(r['local_frame'])];note=r['notes']
        if 'サービス前' in note or 'サーブ前' in note:context='existing_note_pre_serve'
        elif 'ラリー終了後' in note:context='existing_note_after_rally'
        else:context='unresolved'
        err=math.hypot(float(p['x_px'])-float(r['truth_x_px']),float(p['y_px'])-float(r['truth_y_px'])) if r['truth_visibility']=='visible' and int(p['visible']) else None
        good=r['truth_visibility']=='visible' and err is not None and err<=16
        inventory.append(dict(sample_id=r['sample_id'],evaluation_split=r['evaluation_split'],candidate_id=r['candidate_id'],
            local_frame=r['local_frame'],source_timestamp_sec=r['source_timestamp_sec'],existing_note_context_seed=context,
            seed_provenance='existing_note_only_not_new_context_ground_truth',existing_notes=note,
            truth_visibility=r['truth_visibility'],baseline_visible=p['visible'],baseline_correct_within16px=good))
        if r['evaluation_split']=='validation' and int(p['visible']) and not good:
            errors.append(dict(r,baseline_x_px=p['x_px'],baseline_y_px=p['y_px'],baseline_error_px=err))
    for name,rows in [('context_inventory.csv',inventory),('validation_nonlocalized_outputs.csv',errors)]:
        with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    font=ImageFont.truetype('C:/Windows/Fonts/segoeui.ttf',17)
    for page in range(math.ceil(len(errors)/3)):
        canvas=Image.new('RGB',(1152,775),'#171b24');draw=ImageDraw.Draw(canvas)
        draw.text((8,8),'Validation nonlocalized baseline outputs | raw 384x216 crops at t-6, t, t+6',font=font,fill='white')
        for j,r in enumerate(errors[page*3:page*3+3]):
            t=int(r['local_frame']);x=max(0,min(1920-384,round(float(r['baseline_x_px']))-192));y=max(0,min(1080-216,round(float(r['baseline_y_px']))-108))
            top=44+j*243
            draw.text((8,top),f"{r['candidate_id'][-3:]} frame {t} | label {r['truth_visibility']} | baseline ({r['baseline_x_px']},{r['baseline_y_px']})",font=font,fill='white')
            for k,delta in enumerate((-6,0,6)):
                f=max(0,t+delta);p=SOURCE/'data/balltrack/racketvision/match1'/r['candidate_id']/'frames'/f'{f:06d}.jpg'
                im=cv2.imread(str(p));assert im is not None,str(p)
                patch=Image.fromarray(cv2.cvtColor(im[y:y+216,x:x+384],cv2.COLOR_BGR2RGB))
                canvas.paste(patch,(k*384,top+24))
                draw.text((k*384+5,top+26),str(f),font=font,fill='#00ff88',stroke_width=1,stroke_fill='black')
                if not delta:draw.rectangle((k*384,top+24,(k+1)*384-1,top+239),outline='#ffaa00',width=2)
        canvas.save(OUT/f'validation_errors_{page+1:02d}.jpg',quality=95)
    summary=dict(total_samples=len(inventory),validation_nonlocalized=len(errors),
        context_seeds=dict(Counter(r['existing_note_context_seed'] for r in inventory)),
        context_seed_counts_by_split={s:dict(Counter(r['existing_note_context_seed'] for r in inventory if r['evaluation_split']==s)) for s in ('validation','test')},
        warning='Context seeds are based on existing notes, independent of tracker correctness; unresolved is not in-play. Do not use seeds as confirmed rally boundaries.')
    (OUT/'inventory_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    print([(r['candidate_id'][-3:],r['local_frame'],r['baseline_x_px'],r['baseline_y_px'],r['notes']) for r in errors])


if __name__=='__main__':main()
