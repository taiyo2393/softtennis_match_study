"""Review2b: read-only review2 snapshot, rally-local envelope and visual audit."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import build_court_view as b

BACKUP=b.OUT/'review2b_before'
ENVELOPE='observed_x_depth_envelope'
RULE=dict(radius_source_frames=15,percentile=75,minimum_observations=3,
          eligibility='raw height <= 1.35 * .73 * (raw y2 - 415); occluded boxes included',
          direction='max(raw y2, percentile); original boxes and height gate unchanged',
          scope='same point, fixed far player and rally; before speed gate',
          status_threshold_m=.5,interval_threshold_m=1.,serve_priority=True,
          shift_definition='raw projected court_y minus envelope projected court_y, metres; potential input shift also recorded on occluded/overridden/rejected rows',
          applied_definition='unoccluded, height-eligible, accepted observation, no serve/lift override, envelope_shift_m > 1e-9')


def snapshot():
    path=b.OUT/'review2b_before.json'
    if path.exists():return json.loads(path.read_text(encoding='utf-8'))
    s=json.loads((b.OUT/'cleaning_summary.json').read_text(encoding='utf-8'))
    assert s.get('review2') and not s.get('review2b'), 'Need an original review2 baseline'
    BACKUP.mkdir(exist_ok=True)
    names=['cleaning_summary.json','review2_far_depth_counts.csv','review2_report.md','README.md',
           'hitter_auto.csv','human_confirmed_hitters_p07_11.csv','identity_summary.json','wrist_identity.csv']
    names += [f'{prefix}_p{p:02d}.csv' for p in range(7,12)
              for prefix in ('cleaned_player_positions','identity_positions','identity_map')]
    for name in names:shutil.copyfile(b.OUT/name,BACKUP/name)
    value=dict(version='review2b_20261119',sha256={n:b.sha(BACKUP/n) for n in names})
    b.dump(path,value)
    return value


def envelope_rows(source):
    project,H=b.abc.geometry(b.load_module);result={}
    for rally in sorted({int(r['rally_no']) for r in source}):
        raw=sorted((r for r in source if int(r['rally_no'])==rally),key=lambda r:r['source_frame'])
        eligible=[r for r in raw if r['y2']-r['y1']<=1.35*.73*(r['y2']-415)]
        for r in raw:
            f=int(r['source_frame']);values=[z['y2'] for z in eligible if abs(z['source_frame']-f)<=15]
            ok=r['y2']-r['y1']<=1.35*.73*(r['y2']-415)
            depth=max(r['y2'],float(np.percentile(values,75))) if ok and len(values)>=3 else r['y2']
            x=(r['x1']+r['x2'])/2
            shift=(project(H,x,r['y2'])[1]-project(H,x,depth)[1])*23.77
            result[f]=dict(envelope_y2_px=depth,envelope_shift_m=float(shift),
                           envelope_window_observations=len(values),envelope_height_eligible=bool(ok))
    return result


def freeze_horizontal(p,rows):
    old=pd.read_csv(BACKUP/f'cleaned_player_positions_p{p:02d}.csv').set_index(['source_frame','player_slot'])
    for r in rows:
        if not r['player_slot'].startswith('far_'):continue
        prior=old.loc[r['source_frame'],r['player_slot']]
        if r['status'] in b.VALID and pd.notna(prior.court_x):r['court_x']=float(prior.court_x)
    return rows


def applied_mask(table):
    direct=table.filter_status.isin(['observed',ENVELOPE])
    if 'speed_reset_base_status' in table:
        direct |= table.filter_status.eq('observed_after_speed_reset') & table.speed_reset_base_status.isin(['observed',ENVELOPE])
    return (table.player_slot.str.startswith('far_') & direct
            & pd.to_numeric(table.envelope_shift_m,errors='coerce').gt(1e-9))


def summarize(p,table,segments):
    stats=[];intervals=[]
    selected=pd.Series(False,index=table.index)
    for seg in segments:selected |= table.source_frame.between(seg['start_frame'],seg['end_frame'])
    applied=applied_mask(table)&selected
    table['envelope_applied']=applied_mask(table)
    for slot in b.SLOTS[2:]:
        g=table[applied&table.player_slot.eq(slot)]
        shift=pd.to_numeric(g.envelope_shift_m)
        stats.append(dict(point=p,player=b.NAMES[slot],player_slot=slot,scope='displayed_all_source_frames',
                          applied_frames=len(g),status_envelope_frames=int(g.filter_status.eq(ENVELOPE).sum()),
                          median_shift_m=float(shift.median()) if len(g) else 0.,
                          p95_shift_m=float(shift.quantile(.95)) if len(g) else 0.,
                          max_shift_m=float(shift.max()) if len(g) else 0.))
        for rally,group in g[g.envelope_shift_m.astype(float).ge(1.)].groupby('rally_no'):
            runs=[]
            for r in group.sort_values('source_frame').to_dict('records'):
                if not runs or r['source_frame']!=runs[-1][-1]['source_frame']+1:runs.append([])
                runs[-1].append(r)
            for run in runs:
                a,z=int(run[0]['source_frame']),int(run[-1]['source_frame'])
                intervals.append(dict(point=p,player=b.NAMES[slot],player_slot=slot,rally_no=int(rally),
                                      start_frame=a,end_frame=z,duration_source_frames=z-a+1,
                                      midpoint_frame=(a+z)//2,max_shift_m=max(float(r['envelope_shift_m']) for r in run)))
    return stats,intervals


def check_images(intervals):
    """Original left-camera pixels, fixed crops per person across start/mid/end."""
    font=ImageFont.truetype('C:/Windows/Fonts/meiryo.ttc',20)
    for e in intervals:
        p=e['point'];mid=e['midpoint_frame'];frames=[e['start_frame'],mid,e['end_frame']]
        table=pd.read_csv(b.OUT/f'cleaned_player_positions_p{p:02d}.csv').set_index(['source_frame','player_slot'])
        pixels={f:list(b.source_frames(f,f))[0][1] for f in sorted(set(frames))}
        image=Image.new('RGB',(1920,1080),(12,22,32));draw=ImageDraw.Draw(image)
        draw.text((20,8),f"review2b 第{p}点 {e['player']} F{e['start_frame']}–{e['end_frame']} ({e['duration_source_frames']}F) 中央 F{mid}",font=font,fill='white')
        draw.text((20,40),'元映像の同一画素範囲を拡大：橙＝元y2、シアン＝包絡y2。各行は同じ人物。',font=font,fill='white')
        for row,slot in enumerate(b.SLOTS[2:]):
            center=table.loc[mid,slot]
            cx=(center.x1+center.x2)/2;cy=(center.y1+max(center.y2,center.envelope_y2_px))/2
            left=max(0,min(1720,int(cx-100)));top=max(0,min(952,int(cy-64)))
            for col,f in enumerate(frames):
                q=table.loc[f,slot];tile=Image.fromarray(pixels[f]).crop((left,top,left+200,top+128)).resize((640,410))
                d=ImageDraw.Draw(tile)
                if pd.notna(q.y2):
                    coords=((q.x1-left)*3.2,(q.y1-top)*410/128,(q.x2-left)*3.2,(q.y2-top)*410/128)
                    d.rectangle(coords,outline=(255,160,60),width=2)
                    y=(q.envelope_y2_px-top)*410/128
                    d.line((coords[0],y,coords[2],y),fill=(0,255,255),width=3)
                x=col*640;y=110+row*460
                image.paste(tile,(x,y))
                label=f'{b.NAMES[slot]} F{f} y2={q.y2:.1f} env={q.envelope_y2_px:.1f} shift={q.envelope_shift_m:.2f}m'
                draw.text((x+5,y-30),label,font=font,fill='white')
        draw.text((20,1035),'静止画だけで真の後退を否定できない。開始・中央・終了の足元と枠切れを比較する。',font=font,fill='white')
        name=f'check_review2b_p{p:02d}_F{mid:05d}.jpg'
        image.save(b.OUT/name,quality=95)
        e['check_image']=name;e['visual_assessment']='pending visual inspection'


def attach_visual_review(summary):
    path=b.OUT/'review2b_visual_review.json'
    if path.exists():
        review=json.loads(path.read_text(encoding='utf-8'))
        for e in summary['review2b_checks']:
            e['visual_assessment']=review['assessments'].get(e['check_image'],'未確認')
        summary['review2b_visual_review_method']=review['method']


def refresh_counts(summary):
    for e in summary['review2_depth_counts']:
        p=e['point'];slot=e['player_slot']
        old=pd.read_csv(BACKUP/f'cleaned_player_positions_p{p:02d}.csv')
        now=pd.read_csv(b.OUT/f'cleaned_player_positions_p{p:02d}.csv')
        selected=set()
        for seg in summary['points'][str(p)]['rendering']['segments']:
            selected.update(range(seg['start_frame'],seg['end_frame']+1))
        old=old[old.player_slot.eq(slot)&old.source_frame.isin(selected)].set_index('source_frame')
        now=now[now.player_slot.eq(slot)&now.source_frame.isin(selected)].set_index('source_frame')
        ov=old.status.isin(b.VALID)&old.court_y.notna();nv=now.status.isin(b.VALID)&now.court_y.notna()
        common=ov&nv
        e.update(review2_valid_frames=int(ov.sum()),review2b_valid_frames=int(nv.sum()),
                 review2b_new_valid_frames=int((~ov&nv).sum()),review2b_lost_valid_frames=int((ov&~nv).sum()),
                 common_review2_above_25_3m=int((common&old.court_y.mul(23.77).gt(25.3)).sum()),
                 common_review2b_above_25_3m=int((common&now.court_y.mul(23.77).gt(25.3)).sum()))
