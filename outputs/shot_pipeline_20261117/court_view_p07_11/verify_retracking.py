"""Produce one before/after frame and audit the affected interval and artifacts."""
import json
from pathlib import Path
import subprocess

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import build_court_view as view

OUT=view.OUT
PIPE=view.PIPE


def interval_check():
    first,last,frame=25353,25965,25652
    old=pd.read_csv(PIPE/'tracking/point_08/player_court_proxies.csv')
    new=pd.read_csv(PIPE/'tracking_v2/point_08/player_court_proxies.csv')
    result=dict(first_frame=first,last_frame=last,expected_frames=last-first+1,
                check_frame=frame,coordinate_limits_for_inside=[0,1],versions={})
    for version,table in [('original',old),('v2',new)]:
        rows=table[table.source_frame.between(first,last)&(table.player_slot=='near_front')]
        result['versions'][version]=dict(present_frames=len(rows),missing_frames=last-first+1-len(rows),
                                        inside_court_u_frames=int(rows.court_proxy_x.between(0,1).sum()),
                                        outside_court_u_frames=int((~rows.court_proxy_x.between(0,1)).sum()),
                                        min_u=float(rows.court_proxy_x.min()),max_u=float(rows.court_proxy_x.max()),
                                        track_ids=sorted(int(i) for i in rows.track_id.unique()))
        result['versions'][version].update(
            inside_court_uv_frames=int((rows.court_proxy_x.between(0,1)&rows.court_proxy_y.between(0,1)).sum()),
            min_v=float(rows.court_proxy_y.min()),max_v=float(rows.court_proxy_y.max()))
        missing=sorted(set(range(first,last+1))-set(rows.source_frame))
        runs=[]
        for f in missing:
            if runs and f==runs[-1][1]+1:runs[-1][1]=f
            else:runs.append([f,f])
        result['versions'][version]['missing_intervals']=runs
    rgb=None
    for _,rgb in view.source_frames(frame,frame):pass
    canvas=Image.new('RGB',(1920,630),(12,22,32))
    font=ImageFont.truetype('C:/Windows/Fonts/meiryo.ttc',22)
    small=ImageFont.truetype('C:/Windows/Fonts/meiryo.ttc',17)
    d=ImageDraw.Draw(canvas)
    for i,(title,table) in enumerate([('元の追跡：N1が右外の線審を追跡',old),('再追跡：審判候補を除外した4スロット',new)]):
        xoff=i*960
        canvas.paste(Image.fromarray(rgb).resize((960,540)),(xoff,48))
        d.text((xoff+12,8),title,font=font,fill=(235,242,249))
        for row in table[table.source_frame==frame].itertuples():
            color=view.RGB[row.player_slot]
            box=(xoff+row.x1/2,48+row.y1/2,xoff+row.x2/2,48+row.y2/2)
            d.rectangle(box,outline=color,width=3)
            d.text((box[0],max(48,box[1]-24)),f'{view.NAMES[row.player_slot]} ID{row.track_id}',font=small,fill=color,
                   stroke_width=1,stroke_fill=(0,0,0))
    d.text((12,596),f'match2 第8点 F{frame} ｜ N1/N2はスロット名。単一コマの目視確認であり全区間の精度検証ではない。',font=small,fill=(230,235,245))
    canvas.save(OUT/f'retrack_p08_F{frame}.jpg',quality=92)
    result['check_image']=f'retrack_p08_F{frame}.jpg'
    checks=[25353,25443,25652,25820,25825,25826,25863,25965]
    sheet=Image.new('RGB',(1920,4*590),(12,22,32))
    sd=ImageDraw.Draw(sheet)
    for i,f in enumerate(checks):
        rgb=list(view.source_frames(f,f))[0][1]
        x,y=(i%2)*960,(i//2)*590
        sheet.paste(Image.fromarray(rgb).resize((960,540)),(x,y+40))
        rows=new[(new.source_frame==f)&(new.player_slot=='near_front')]
        text=f'F{f}  N1欠測' if rows.empty else f'F{f}  N1 ID{int(rows.iloc[0].track_id)}'
        sd.text((x+8,y+5),text,font=font,fill=(255,100,90) if rows.empty else (65,220,120))
        for r in rows.itertuples():
            sd.rectangle((x+r.x1/2,y+40+r.y1/2,x+r.x2/2,y+40+r.y2/2),outline=(65,220,120),width=3)
    sheet.save(OUT/'retrack_p08_interval_contact_sheet.jpg',quality=92)
    result['interval_check_image']='retrack_p08_interval_contact_sheet.jpg'
    view.dump(OUT/'retrack_interval_check.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


def final_audit():
    report=json.loads((OUT/'cleaning_summary.json').read_text(encoding='utf-8'))
    hitters=pd.read_csv(OUT/'hitter_auto.csv')
    baseline=pd.read_csv(OUT/'hitter_auto_before_retrack.csv')
    reconstructed=pd.read_csv(OUT/'hitter_auto_tracking_v1.csv')
    assert len(hitters)==32
    assert baseline[['point','frame','hitter','status']].equals(reconstructed[['point','frame','hitter','status']])
    protected=json.loads((PIPE/'tracking_v2/input_sha256_before.json').read_text(encoding='utf-8'))
    assert all(view.sha(Path(p))==h for p,h in protected.items()), 'Protected originals changed'
    sheet=Image.new('RGB',(1536,2320),(12,22,32))
    draw=ImageDraw.Draw(sheet)
    font=ImageFont.truetype('C:/Windows/Fonts/meiryo.ttc',18)
    for row,point in enumerate(range(7,12)):
        for col,check in enumerate(report['points'][str(point)]['rendering']['checks']):
            x,y=768*col,464*row
            draw.text((x+8,y+3),f'第{point}点 ラベルF{check["label_frame"]} / 表示F{check["sampled_frame"]}',font=font,fill=(235,242,249))
            with Image.open(OUT/check['file']) as im:
                sheet.paste(im.resize((768,432)),(x,y+30))
    sheet.save(OUT/'verification_contact_sheet.jpg',quality=92)
    exclusion=json.loads((PIPE/'tracking_v2/verification.json').read_text(encoding='utf-8'))
    # official_candidates.csv also contains the exhaustive review-image index.
    review=pd.read_csv(PIPE/'tracking_v2/official_candidates.csv')
    assert len(review)==len(exclusion['excluded_candidates'])
    large=[]
    for directory in [PIPE/'tracking_v2',OUT]:
        for path in directory.rglob('*'):
            if not path.is_file() or path.stat().st_size<=256*1024:continue
            relative=path.relative_to(view.ROOT).as_posix()
            ignored=subprocess.run(['git','check-ignore','-q','--',relative],cwd=view.ROOT).returncode==0
            assert ignored,relative
            large.append(dict(file=relative,bytes=path.stat().st_size,ignored=ignored))
    pd.DataFrame(large).to_csv(OUT/'large_file_audit.csv',index=False,encoding='utf-8-sig')
    track_scope=view.load_module('track_scope',PIPE/'track_points.py')
    scopes=track_scope.load_scopes(view.ROOT/'outputs/match2_point_inventory_20261117/point_inventory.csv',list(range(7,12)))
    for point in range(7,12):
        first,_,last=scopes[point]
        availability=pd.read_csv(PIPE/f'tracking_v2/point_{point:02d}/player_frame_availability.csv')
        assert availability.source_frame.tolist()==list(range(first,last+1))
        assert report['points'][str(point)]['rendering']['full_decode_verified']
    # Independent array-based check of the lift counts, maxima and flags.
    tables={p:pd.read_csv(PIPE/f'tracking_v2/point_{p:02d}/player_court_proxies.csv') for p in range(7,12)}
    import numpy as np
    for shot in hitters.itertuples():
        if shot.lift_status!='evaluated':
            assert shot.jump_or_step_flag==0
            continue
        table=tables[shot.point]
        group=table[table.player_slot==shot.hitter_slot]
        delta=np.abs(group.source_frame.to_numpy()-shot.frame)
        box_height=(group.y2-group.y1).to_numpy()
        bottoms=group.y2.to_numpy()
        reference_mask=(delta>=20)&(delta<=40)
        height=np.median(box_height[reference_mask])
        lift=np.median(bottoms[reference_mask])-bottoms[delta<=20]
        count=int(np.count_nonzero(lift>=height*.06))
        assert shot.lift_frames==count and shot.jump_or_step_flag==int(count>=5)
        assert abs(shot.max_lift_px-max(0,float(lift.max())))<1e-9
    result=dict(protected_inputs_unchanged=len(protected),hitter_count=len(hitters),
                hitter_status_counts=hitters.status.value_counts().to_dict(),
                hitter_changes=report['hitter_changes'],jump_or_step_count=int(hitters.jump_or_step_flag.sum()),
                jump_or_step_list=hitters.loc[hitters.jump_or_step_flag==1,['point','frame','hitter','lift_frames','max_lift_px']].to_dict('records'),
                large_files_ignored=len(large),all_five_videos_full_decode_verified=True,
                baseline_reconstruction_matches_previous_decisions=True,lift_values_independently_checked=True)
    view.dump(OUT/'retracking_verification.json',result)
    inventory=[]
    for path in sorted(OUT.rglob('*')):
        if not path.is_file():continue
        relative=path.relative_to(view.ROOT).as_posix()
        ignored=subprocess.run(['git','check-ignore','-q','--',relative],cwd=view.ROOT).returncode==0
        inventory.append(dict(file=path.relative_to(OUT).as_posix(),bytes=path.stat().st_size,ignored=ignored))
    pd.DataFrame(inventory).to_csv(OUT/'file_inventory.csv',index=False,encoding='utf-8-sig')
    report['large_files']=[r for r in inventory if r['bytes']>256*1024]
    view.dump(OUT/'cleaning_summary.json',report)
    view.write_readme(report,report['hitter_decisions'])
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--final',action='store_true')
    args=parser.parse_args()
    interval_check()
    if args.final:final_audit()
