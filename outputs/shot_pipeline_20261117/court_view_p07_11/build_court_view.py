"""Read-only inputs; all writes stay next to this script. Run with python -B.

056 cleaning and rendering adapted to per-source-frame observations. Pose helper
functions are imported without calling their main functions or original writers.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
from statistics import median
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import retracking_support as retrack
import abc_support as abc

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
PIPE = OUT.parent
ROOT = PIPE.parent.parent
REF = ROOT / 'outputs/ball_quality_20261111'
FFMPEG = 'C:/ffmpeg/bin/ffmpeg.exe'
FFPROBE = 'C:/ffmpeg/bin/ffprobe.exe'
VIDEO = Path('C:/Users/ta239/Documents/Codex/2026-09-07/ai-2d-2d-1-2-0/softtennis_frontcourt_study/data/raw/match2.mp4')
FPS = 59.94  # Preserve 056 cleaning thresholds; decoding uses exact 60000/1001.
DECODE_FPS = 60000 / 1001
SLOTS = ['near_front', 'near_rear', 'far_rear', 'far_front']
NAMES = dict(zip(SLOTS, ['N1', 'N2', 'F1', 'F2']))
RGB = dict(zip(SLOTS, [(65,220,120), (70,170,255), (220,90,230), (250,210,65)]))
OBS = ('observed', 'observed_after_height_reset', 'observed_after_speed_reset', 'observed_head_anchored', 'observed_x_depth_interpolated', 'observed_x_depth_serve_anchor', 'observed_x_depth_lift_held', 'observed_x_depth_envelope')
STATUSES = (*OBS, 'rejected_box_height', 'rejected_speed', 'interpolated', 'missing', 'offscreen_hold')
VALID = (*OBS, 'interpolated', 'offscreen_hold')
REVIEW2_CHECKS = {7:[22848,23482,23803],8:[25246,25260,25863]}
XLIM, YLIM = (-.25, 1.25), (-.3, 1.55)


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=1 if path.name=='cleaning_summary.json' else 2, allow_nan=False)+'\n', encoding='utf-8',newline='\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def source_frames(first, last, step=1, width=1920, height=1080):
    """Seek locally, then select by ORIGINAL PTS, verified with showinfo.

    -copyts retains source timestamps; the 1/4-frame margins avoid microsecond
    rounding at interval boundaries. Output frames are selected by rounded source
    frame number rather than a seek-relative counter. No AV1 OpenCV decoding.
    """
    frames = list(range(first, last+1, step))
    logdir = OUT/'decode_logs'
    logdir.mkdir(exist_ok=True)
    logpath = logdir/f'F{first}_{last}_step{step}.log'
    select = (f'between(t,{(first-.25)/DECODE_FPS:.9f},{(last+.25)/DECODE_FPS:.9f})'
              f'*not(mod(round(t*60000/1001)-{first},{step}))')
    cmd = [FFMPEG, '-hide_banner', '-loglevel', 'info', '-copyts',
           '-ss', f'{max(0,first-3)/DECODE_FPS:.9f}', '-i', str(VIDEO),
           '-an', '-vf', f"select='{select}',showinfo,scale={width}:{height}",
           '-frames:v', str(len(frames)), '-fps_mode', 'passthrough',
           '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1']
    with logpath.open('wb') as err:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=err)
        try:
            for f in frames:
                need = width*height*3
                buf = bytearray()
                while len(buf) < need:
                    block = proc.stdout.read(need-len(buf))
                    if not block:
                        raise RuntimeError(f'Decode ended at F{f}; see {logpath}')
                    buf.extend(block)
                yield f, np.frombuffer(buf, np.uint8).reshape(height,width,3)
            proc.stdout.close()
            assert proc.wait() == 0, str(logpath)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    times = [float(v) for v in re.findall(r'\bn:\s*\d+.*?pts_time:([\d.]+)', logpath.read_text())]
    actual = [round(t*DECODE_FPS) for t in times]
    assert actual == frames, (logpath, actual[:3], frames[:3], len(actual),len(frames))


def distance(a, z):
    return math.hypot((a[0]-z[0])*10.97, (a[1]-z[1])*23.77)


def clean_player(slot, source, frames, *, far_revision=True, speed_reset=False):
    """Final far geometry is separate from the frozen legacy B geometry."""
    if far_revision and slot.startswith('far_'):
        from far_depth_revision import clean_far
        return clean_far(slot, source, frames)
    source = sorted(source, key=lambda r: r['source_frame'])
    if not source:
        return [dict(source_frame=f,player_slot=slot,status='missing',filter_status='missing',court_x='',court_y='') for f in frames],dict(global_median_box_height_px=None,height_resets=[])
    heights = [r['y2']-r['y1'] for r in source]
    assert heights and all(math.isfinite(h) and h>0 for h in heights)
    global_height = median(heights)
    # B retains the previous pre-link geometry. Only final A uses fixed people.
    head_history = deque()
    project,H = abc.geometry(load_module)
    history = deque()  # (source_frame, height), NOT a fixed number of samples.
    accepted, pending, resets, diagnostics = [], [], [], {}
    from speed_reset import SpeedReset, STATUS as SPEED_RESET
    speed_streak = SpeedReset(speed_reset)
    previous_raw = None
    warm = True
    warm_first = None

    def jump_test(frame, x, y):
        if not accepted:
            return '', False
        prev = accepted[-1]
        jump = distance((x,y), (prev['court_x'],prev['court_y']))
        return jump, jump > 8*(frame-prev['source_frame'])/FPS+1

    for raw, height in zip(source, heights):
        f = int(raw['source_frame'])
        assert f not in diagnostics
        x,y = raw['court_proxy_x'], raw['court_proxy_y']
        occluded = raw.get('lower_occluded',False)
        while head_history and head_history[0][0] < f-3*DECODE_FPS:
            head_history.popleft()
        head_height = median(h for _,h in head_history) if head_history else global_height
        head_x,head_y = project(H,(raw['x1']+raw['x2'])/2,raw['y1']+head_height)
        head_used = occluded
        if head_used:
            x,y = head_x,head_y
        assert math.isfinite(x) and math.isfinite(y)
        while history and history[0][0] < f-30:
            history.popleft()
        reference = global_height if warm or not history else median(h for _,h in history)
        jump, speed_bad = jump_test(f,x,y)
        raw_jump = distance((x,y),(previous_raw['effective_court_x'], previous_raw['effective_court_y'])) if previous_raw else ''
        raw_limit = 8*(f-previous_raw['source_frame'])/FPS+1 if previous_raw else ''
        status = ('rejected_speed' if speed_bad else 'observed_head_anchored') if occluded else ('rejected_box_height' if not .75*reference <= height <= 1.35*reference else ('rejected_speed' if speed_bad else 'observed'))
        reset_start = speed_streak.update(f,status,raw.get('rally_no'))
        reset_base = 'observed_head_anchored' if occluded else 'observed'
        if reset_start is not None:
            status = SPEED_RESET
        r = dict(raw, source_frame=f, status=status, filter_status=status,
                 speed_reset_frame=f if reset_start is not None else '',
                 speed_reject_start_frame=reset_start if reset_start is not None else '',
                 speed_reset_base_status=reset_base if reset_start is not None else '',
                 court_x=x if status in OBS else '', court_y=y if status in OBS else '',
                 box_height_px=height, reference_height_px=reference, jump_m=jump,
                 raw_previous_frame=previous_raw['source_frame'] if previous_raw else '',
                 raw_jump_m=raw_jump, raw_speed_limit_m=raw_limit,
                 height_reset_start_frame='', height_reset_trigger_frame='',
                 interpolation_start_frame='', interpolation_end_frame='',
                 raw_court_x='', raw_court_y='', smoothing_shift_m='')
        r.update(lower_occluded=bool(occluded),head_reference_height_px=head_height,
                 head_foot_y_px=raw['y1']+head_height,head_court_x=head_x,head_court_y=head_y,
                 effective_court_x=x,effective_court_y=y)
        diagnostics[f] = r
        previous_raw = r
        if status in OBS:
            accepted.append(r)
            if not occluded:
                history.append((f,height))
                head_history.append((f,height))
            if not occluded:
                if warm_first is None:
                    warm_first = f
                if f-warm_first >= 8:  # Original five observations at 2F cadence span 8F.
                    warm = False
        if occluded:
            # Occluded points neither count toward height rejection nor bridge a reset.
            pending.clear()
            continue
        if status != 'rejected_box_height':
            pending.clear()
            continue
        pending.append(r)
        if f-pending[0]['source_frame'] <= 30:
            continue
        history.clear()
        reference = median(c['box_height_px'] for c in pending)
        restored, speed_kept = [], []
        start = pending[0]['source_frame']
        for c in pending:
            jump, speed_bad = jump_test(c['source_frame'],c['court_proxy_x'],c['court_proxy_y'])
            raw_bad = c['raw_jump_m'] != '' and c['raw_jump_m'] > c['raw_speed_limit_m']
            status = 'rejected_speed' if speed_bad or raw_bad else 'observed_after_height_reset'
            reset_start = speed_streak.update(c['source_frame'],status,c.get('rally_no'))
            if reset_start is not None:
                status = SPEED_RESET
                c.update(speed_reset_frame=c['source_frame'],speed_reject_start_frame=reset_start,
                         speed_reset_base_status='observed_after_height_reset')
            c.update(status=status, filter_status=status, reference_height_px=reference,
                     jump_m=jump, height_reset_start_frame=start, height_reset_trigger_frame=f,
                     court_x=c['court_proxy_x'] if status in OBS else '',
                     court_y=c['court_proxy_y'] if status in OBS else '')
            if status in OBS:
                accepted.append(c)
                history.append((c['source_frame'],c['box_height_px']))
                head_history.append((c['source_frame'],c['box_height_px']))
                restored.append(c['source_frame'])
            else:
                speed_kept.append(c['source_frame'])
        if not history:
            history.extend((c['source_frame'],c['box_height_px']) for c in pending)
        resets.append(dict(start_frame=start, trigger_frame=f, reset_reference_height_px=reference,
                           restored_frames=restored, rejected_speed_frames=speed_kept))
        warm = False
        pending.clear()

    runs = []
    for r in accepted:
        r.update(raw_court_x=r['court_x'],raw_court_y=r['court_y'])
        if not runs or r['source_frame']-runs[-1][-1]['source_frame']>30:
            runs.append([])
        runs[-1].append(r)
    for run in runs:
        for r in run:
            window = [p for p in run if abs(p['source_frame']-r['source_frame'])<=8]
            # Original minimum 3 samples at 2F cadence => >=4F evidence span.
            if window[-1]['source_frame']-window[0]['source_frame'] >= 4:
                r['court_x'] = median(p['raw_court_x'] for p in window)
                r['court_y'] = median(p['raw_court_y'] for p in window)
            r['smoothing_shift_m'] = distance((r['court_x'],r['court_y']),(r['raw_court_x'],r['raw_court_y']))
    interpolated = {}
    for a,z in zip(accepted,accepted[1:]):
        start,end = a['source_frame'],z['source_frame']
        if end-start>30:
            continue
        for f in range(start+1,end):
            t = (f-start)/(end-start)
            interpolated[f] = (a['court_x']+t*(z['court_x']-a['court_x']),
                               a['court_y']+t*(z['court_y']-a['court_y']),start,end)
    template = {k:'' for k in next(iter(diagnostics.values()))}
    result=[]
    for f in frames:
        r = dict(diagnostics[f]) if f in diagnostics else dict(template, source_frame=f,player_slot=slot,status='missing',filter_status='missing')
        if f in interpolated:
            assert r['status'] not in OBS
            x,y,a,z = interpolated[f]
            r.update(status='interpolated',court_x=x,court_y=y,interpolation_start_frame=a,interpolation_end_frame=z)
        assert (r['court_x']!='') == (r['status'] in VALID)
        result.append(r)
    return result, dict(global_median_box_height_px=global_height,height_resets=resets)


def supplement_pose(tracks):
    target = [27464,28292,28375,28526,28597]
    out = OUT/'pose_supplement'
    out.mkdir(exist_ok=True)
    cached = out/'wrist_timeseries_additional.csv'
    if cached.exists():
        return pd.read_csv(cached), json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    manifest = dict(status='running', target_frames=target, intervals=[], kp_conf=.3, smooth_window=3)
    try:
        # extract_pose sets a legacy config variable on import. Importing it does
        # not create files; redirect it BEFORE the first ultralytics import.
        ext = load_module('extract_pose_readonly', PIPE/'pose/extract_pose.py')
        os.environ['YOLO_CONFIG_DIR'] = str(out/'ultralytics_config')
        ext.CACHE = out
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(4)
        cv2.setNumThreads(1)
        device = '0' if torch.cuda.is_available() else 'cpu'
        model = YOLO(str(ext.DEFAULT_MODEL))
        manifest.update(device=device, model=str(ext.DEFAULT_MODEL), model_sha256=sha(ext.DEFAULT_MODEL))
        rows=[]
        lookup = {(int(r.source_frame),r.player_slot):r for r in tracks[9].itertuples()}
        args = SimpleNamespace(pad_x=.6,pad_y=.25,crop_size=384)
        for center in target:
            first,last = center-15, center+10
            pending_images,pending_rows=[],[]
            def flush():
                if not pending_images:
                    return
                results = model.predict(source=pending_images, device=device,imgsz=384,conf=.25,iou=.7,max_det=10,rect=False,batch=32,verbose=False,save=False,stream=False,project=str(out/'predict_runs'),name='pose',exist_ok=True)
                assert len(results)==len(pending_rows)
                for result,row in zip(results,pending_rows):
                    ext.assign_detection(result,row)
                pending_images.clear()
                pending_rows.clear()
            for f,rgb in source_frames(first,last):
                raw=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
                for slot in SLOTS:
                    tr=lookup.get((f,slot))
                    row=ext.base_row(9,f'add_{center}',f,slot,tr)
                    rows.append(row)
                    if tr is not None:
                        crop=ext.make_crop(raw,row,args.pad_x,args.pad_y,args.crop_size)
                        if crop is not None:
                            pending_images.append(crop)
                            pending_rows.append(row)
                            if len(pending_images)>=32:
                                flush()
            flush()
            manifest['intervals'].append(dict(center_frame=center,start_frame=first,end_frame_inclusive=last,source_frames=26))
            print(f'Pose supplement F{center}: 26 frames complete',flush=True)
        pose=pd.DataFrame(rows,columns=ext.FIELDS)
        pose.to_csv(out/'pose_additional.csv',index=False,encoding='utf-8-sig',lineterminator='\n')
        combine=load_module('combine_audio_pose_readonly',PIPE/'pose/combine_audio_pose.py')
        wrist,diagnostics=combine.wrist_timeseries(pose,.3,3)
        wrist.to_csv(cached,index=False,encoding='utf-8-sig',lineterminator='\n')
        manifest.update(status='complete',pose_rows=len(pose),valid_wrist_speed=int(wrist.wrist_speed.notna().sum()),
                        fallback_torso_scope='each 26-frame supplement interval/player, using the existing function')
        dump(out/'manifest.json',manifest)
        return wrist,manifest
    except Exception as exc:
        manifest.update(status='failed',error=str(exc))
        dump(out/'manifest.json',manifest)
        print(f'Pose supplement unavailable: {exc}',flush=True)
        return pd.DataFrame(),manifest


def decide_hitters(labels,wrist):
    result=[]
    assert not wrist.duplicated(['point','source_frame','player_slot']).any()
    for row in labels.to_dict('records'):
        p,f,side=int(row['point']),int(row['frame']),row['side']
        slots=SLOTS[:2] if side=='N' else SLOTS[2:]
        window=wrist[(wrist.point==p)&wrist.source_frame.between(f-8,f+4)]
        maxima=[]
        counts=[]
        for s in slots:
            values=window.loc[window.player_slot==s,'wrist_speed']
            values=values[np.isfinite(values)&(values>=0)]
            maxima.append(float(values.max()) if len(values) else None)
            counts.append(len(values))
        status,hitter,reason='undecided','','missing_wrist'
        if all(v is not None for v in maxima):
            hi=int(maxima[1]>maxima[0]);lo=1-hi
            if maxima[hi]>maxima[lo] and maxima[hi]>=1.3*maxima[lo]:
                status,hitter,reason='auto_wrist',slots[hi],'ratio_ge_1.3'
            else:
                reason='ratio_lt_1.3_or_tie'
        result.append(dict(point=p,frame=f,side=side,rally_no=int(row['rally_no']),
                           is_serve=bool(row['is_serve']),outcome=row['outcome'],
                           player_1=slots[0],player_1_max=maxima[0],player_1_valid_frames=counts[0],
                           player_2=slots[1],player_2_max=maxima[1],player_2_valid_frames=counts[1],
                           hitter_slot=hitter,hitter=NAMES[hitter] if hitter else side+'?',
                           status=status,reason=reason,known_tracking_issue=p==8 and side=='N'))
    return result


def count_status(rows,field):
    return {s:{v:sum(r['player_slot']==s and r[field]==v for r in rows) for v in STATUSES} for s in SLOTS}


def side_intervals(rows,p,kind):
    intervals=[]
    for slot in SLOTS:
        run=[]
        def finish():
            if run and (run[-1]['source_frame']-run[0]['source_frame'])/FPS>=1:
                intervals.append(dict(point=p,player_slot=slot,player=NAMES[slot],coordinate_source=kind,
                                      start_frame=run[0]['source_frame'],end_frame=run[-1]['source_frame'],
                                      duration_sec=(run[-1]['source_frame']-run[0]['source_frame'])/FPS,
                                      frame_count=len(run),min_court_x=min(r['x'] for r in run),max_court_x=max(r['x'] for r in run)))
        for r in (r for r in rows if r['player_slot']==slot):
            x=r['court_x'] if kind=='cleaned' else r['court_proxy_x']
            outside=x!='' and math.isfinite(x) and (x<-.12 or x>1.12)
            if not outside or (run and r['source_frame']!=run[-1]['source_frame']+1):
                finish();run=[]
            if outside:
                run.append(dict(r,x=x))
        finish()
    return intervals


def dashed(d,a,z,color,width=2,dash=6,gap=4):
    length=math.dist(a,z)
    if not length:
        return
    for start in np.arange(0,length,dash+gap):
        end=min(start+dash,length)
        d.line([(a[0]+(z[0]-a[0])*v/length,a[1]+(z[1]-a[1])*v/length) for v in (start,end)],fill=color,width=width)


def render_point(p,segments,rows,shots,using_v2=False,balls=None):
    fonts={s:ImageFont.truetype('C:/Windows/Fonts/meiryo.ttc',s) for s in (16,18,20,22,26,30)}
    lookup={(r['source_frame'],r['player_slot']):r for r in rows}
    ball_lookup={} if balls is None else {int(r.source_frame):r for r in balls[balls.point==p].itertuples() if r.candidate_present}
    # Same 056 wide-court mapping and margins.
    height=630/(YLIM[1]-YLIM[0]);width=height*250/540
    left=1620-width/2;top=200+(YLIM[1]-1)*height
    def cp(u,v):return left+width*u,top+height*(1-v)
    vl,vt=cp(XLIM[0],YLIM[1]);vr,vb=cp(XLIM[1],YLIM[0])
    video=OUT/f'point_{p:02d}.mp4'
    encoder=subprocess.Popen([FFMPEG,'-y','-v','error','-f','rawvideo','-pix_fmt','rgb24','-s','1920x1080','-r','15','-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    rendered=[];outside=[];render_rows=[];checks=[]
    # Exactly sampled impact frames (every 2F). Odd-parity impacts: nearest
    # sampled frame, ties earlier; exact label remains explicitly displayed.
    for segment in segments:
        times=list(range(segment['start_frame'],segment['end_frame']+1,2))
        segment['rendered_frames']=len(times)
        segment_shots=[s for s in shots if s['rally_no']==segment['rally_no']]
        selected=[segment_shots[0],segment_shots[-1]]
        if len(segments)>1:
            selected=[segment_shots[-1]]
        for s in selected:
            sampled=min(times,key=lambda f:(abs(f-s['frame']),f))
            checks.append(dict(label_frame=s['frame'],sampled_frame=sampled,rally_no=s['rally_no'],output_frame=len(rendered)+times.index(sampled)))
        requested=sorted(set({7:[23422,23548],8:[25470],10:[30633],11:[32999]}.get(p,[])+REVIEW2_CHECKS.get(p,[])))
        # Exact odd-parity requested frames are inserted in the source map, not relabelled.
        exact_checks=[f for f in requested if segment['start_frame']<=f<=segment['end_frame']]
        if exact_checks:
            times=sorted(set(times+exact_checks))
            for check in checks:
                if check['rally_no']==segment['rally_no']:
                    check['output_frame']=len(rendered)+times.index(check['sampled_frame'])
            segment['rendered_frames']=len(times)
        decoded=source_frames(times[0],times[-1],1,1280,720)
        selected_times=set(times)
        for f,raw in decoded:
            if f not in selected_times:continue
            image=Image.new('RGB',(1920,1080),(12,22,32));d=ImageDraw.Draw(image)
            def txt(x,y,text,size=20,color=(225,235,245)):
                assert d.textbbox((x,y),text,font=fonts[size])[2]<1920, text
                d.text((x,y),text,font=fonts[size],fill=color)
            txt(24,10,f'match2 第{p}点：選手配置と打球',30)
            txt(24,54,'打球の時刻・側は本人の確認 ｜ 打者は人が確認済み ｜ 約半分の速度',22)
            image.paste(Image.fromarray(raw),(16,104));d=ImageDraw.Draw(image)
            for slot in SLOTS:
                r=lookup[f,slot];render_rows.append(r)
                if r['filter_status']=='offscreen_hold' or r.get('x1','')=='':continue
                color=RGB[slot]
                x1,y1,x2,y2=[float(r[k])*2/3 for k in ('x1','y1','x2','y2')]
                corners=[(16+x1,104+y1),(16+x2,104+y1),(16+x2,104+y2),(16+x1,104+y2)]
                rejected=r['filter_status'] not in OBS and not (slot.startswith('far_') and r.get('lower_occluded',False))
                if rejected:
                    for a,z in zip(corners,corners[1:]+corners[:1]):dashed(d,a,z,color,width=1)
                else:d.rectangle((corners[0],corners[2]),outline=color,width=2)
                txt(max(17,int(16+x1)),max(105,int(78+y1)),NAMES[slot]+(' 下隠れ' if slot.startswith('far_') and r.get('lower_occluded',False) else '')+(' 除外' if rejected else ''),20,color)
            ball=ball_lookup.get(f)
            if ball is not None:
                bx,by=16+ball.x_px*2/3,104+ball.y_px*2/3
                d.ellipse((bx-4,by-4,bx+4,by+4),outline=(255,125,35),width=2)
                d.line((bx-6,by,bx+6,by),fill=(255,125,35),width=1)
                d.line((bx,by-6,bx,by+6),fill=(255,125,35),width=1)
            txt(32,780,'球：自動検出（未確認）　橙の小印＝候補（誤検出を含む）',20,(255,155,75))
            txt(1335,109,'コート外まで含む選手配置',30)
            txt(1335,153,'白線＝コート　周囲の余白＝コート外',18)
            d.rectangle((vl,vt,vr,vb),fill=(16,31,43),outline=(45,70,85))
            x0,y1=cp(0,1);x1,y0=cp(1,0)
            d.rectangle((x0,y1,x1,y0),outline=(205,225,235),width=2)
            for v in (.231,.5,.769):d.line([cp(0,v),cp(1,v)],fill=(205,220,235) if v==.5 else (80,120,145),width=2 if v==.5 else 1)
            for u in (.125,.875):d.line([cp(u,0),cp(u,1)],fill=(80,120,145))
            for a,z in [((.5,0),(.5,.231)),((.5,.769),(.5,1))]:d.line([cp(*a),cp(*z)],fill=(80,120,145))
            txt(x1+14,y1-12,'奥ベースライン',16);txt(x1+14,y0-12,'手前ベースライン',16)
            current=max((i for i,s in enumerate(segment_shots) if s['frame']<=f),default=0)
            for j in range(max(0,current-1),current+1):
                if j+1>=len(segment_shots) or f<segment_shots[j]['frame']:continue
                a,z=segment_shots[j:j+2]
                if not(a['hitter_slot'] and z['hitter_slot']):continue
                aa=lookup[a['frame'],a['hitter_slot']];zz=lookup[z['frame'],z['hitter_slot']]
                if aa['status'] not in VALID or zz['status'] not in VALID:continue
                axy,zxy=cp(aa['court_x'],aa['court_y']),cp(zz['court_x'],zz['court_y'])
                ok,a_clip,z_clip=cv2.clipLine((round(vl),round(vt),round(vr-vl),round(vb-vt)),tuple(round(v) for v in axy),tuple(round(v) for v in zxy))
                if ok:dashed(d,a_clip,z_clip,(65,200,235) if j==current else (65,100,125))
            for slot in SLOTS:
                r=lookup[f,slot]
                if r['status'] not in VALID:continue
                u,v=r['court_x'],r['court_y'];x,y=cp(u,v)
                if not(vl+7<x<vr-7 and vt+7<y<vb-7):
                    outside.append(dict(source_frame=f,player_slot=slot,status=r['status'],court_x=u,court_y=v))
                    continue
                color=RGB[slot]
                head=r['status'].startswith('observed_x_depth_')
                held=r['status']=='offscreen_hold'
                if held:
                    for angle in range(0,360,60):d.arc((x-7,y-7,x+7,y+7),angle,angle+35,fill='white',width=2)
                else:
                    d.ellipse((x-6,y-6,x+6,y+6),fill=color if r['status'] in OBS else None,outline='white' if head else ((12,22,32) if r['status'] in OBS else color),width=2 if head or r['status'] not in OBS else 1)
                txt(min(x+8,1730) if held else x+8,y-12,NAMES[slot]+(' 画面外・推定' if held else ''),16 if held else 18,color)
            txt(1335,818,'破線の白抜き円＝画面外・推定（最終位置保持）',16)
            txt(1335,840,'白縁＝横は観測・前後は補正／補間／仮定',16)
            txt(1335,862,'破線＝打球者間の模式線（球軌道ではない）',16)
            txt(1335,884,'塗り＝採用　白抜き＝全座標補間。精度未検証。',16)
            txt(24,837,f'元フレーム F{f} ｜ ラリー {segment["rally_no"]} ｜ 打球 {shots.index(segment_shots[current])+1}/{len(shots)}',22)
            note='N1/N2/F1/F2はラリー内で固定した人物名（自動対応）。前衛・後衛の役割ではありません。'
            if p==8:
                note=('第8点：審判候補を除外して再追跡。打者は確認済み、位置精度は未検証。' if using_v2 else
                      '第8点：手前の誤追跡に注意（線審を含む疑い）。枠・配置・打球者は自動修正していません。')
            txt(24,874,note,20,(250,210,65) if p==8 else (225,235,245))
            # The 056 five-cell sliding strip, limited to the active rally.
            start=max(0,min(len(segment_shots)-5,current-2))
            for k,s in enumerate(segment_shots[start:start+5]):
                j=start+k;x=24+k*376;color=RGB[s['hitter_slot']] if s['hitter_slot'] else (190,200,215)
                d.rounded_rectangle((x,921,x+358,1006),radius=7,fill=(29,49,66) if j==current else (18,31,43),outline=color if j==current else (50,70,85),width=2)
                status={'auto_wrist':'手首自動','auto_ball':'球自動','auto_ball_wrist':'球＋手首'}.get(s['status'],'未決定')
                txt(x+12,927,f'{shots.index(s)+1}：{s["hitter"]}（確認済）',22,color)
                suffix=(' '+s['outcome']) if s['outcome'] in ('winner','out','net') else ''
                txt(x+12,969,f'F{s["frame"]}'+suffix,18)
                if s.get('auto_hitter',s['hitter'])!=s['hitter']:
                    txt(x+220,971,f'自動: {s["auto_hitter"]}',16,(250,210,65))
                elif s['jump_or_step_flag']:
                    txt(x+240,971,'前後位置△',16,(250,210,65))
            txt(24,1030,'打者は確認済み（第7点6は名前の解釈を要再確認）。人物対応・位置精度・球候補は未検証です。',20)
            encoder.stdin.write(np.asarray(image).tobytes())
            if f in exact_checks:
                image.save(OUT/f'check_abc_p{p:02d}_F{f}.jpg',quality=95)
                if f in REVIEW2_CHECKS.get(p,[]):image.save(OUT/f'check_review2_p{p:02d}_F{f:05d}.jpg',quality=95)
            for check in checks:
                if check['sampled_frame']==f and check['rally_no']==segment['rally_no']:
                    name=f'check_p{p:02d}_F{check["label_frame"]}_shown_F{f}.jpg'
                    image.save(OUT/name,quality=92);check['file']=name
            rendered.append(dict(output_frame=len(rendered),point=p,rally_no=segment['rally_no'],source_frame=f))
            if len(rendered)%100==0:print(f'Point {p}: rendered {len(rendered)}',flush=True)
    encoder.stdin.close();err=encoder.stderr.read();assert encoder.wait()==0,err
    pd.DataFrame(rendered).to_csv(OUT/f'video_source_frame_map_p{p:02d}.csv',index=False,encoding='utf-8-sig',lineterminator='\n')
    # Decode every frame through EOF; ffprobe also checks encoded frame count.
    verify=subprocess.run([FFMPEG,'-v','error','-xerror','-i',str(video),'-map','0:v:0','-f','null','-'],capture_output=True)
    assert verify.returncode==0,verify.stderr.decode(errors='replace')
    probe=json.loads(subprocess.check_output([FFPROBE,'-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames,r_frame_rate,width,height,duration','-of','json',str(video)]))['streams'][0]
    assert int(probe['nb_read_frames'])==len(rendered)
    assert probe['r_frame_rate']=='15/1' and (probe['width'],probe['height'])==(1920,1080)
    return dict(video=video.name,frames=len(rendered),duration_sec=float(probe['duration']),segments=segments,
                checks=checks,full_decode_verified=True,ffprobe=probe,
                displayed_status_counts=count_status(render_rows,'status'),
                displayed_filter_status_counts=count_status(render_rows,'filter_status'),
                outside_plot_omitted_count=len(outside),outside_plot_omitted=outside)


RULES=dict(source_fps_for_cleaning=FPS,source_fps_exact='60000/1001',court_width_m=10.97,court_length_m=23.77,
           height_ratio_low=.75,height_ratio_high=1.35,height_history_source_frames=30,
           height_reference='Median of previously accepted heights at f-30 <= prior_frame < f. During startup use global player/point median until accepted evidence spans >=8 source frames (original five 2F samples). If rolling history is empty, use global median.',
           startup_accepted_evidence_span_source_frames=8,
           height_reset_source_frames_strictly_greater_than=30,
           height_reset='Consecutive raw height rejections, missing raw samples do not break run (056 rule). Retrospectively reconsider run, replace height history, retain speed rejection from previous accepted and adjacent raw points, including entry/return.',
           speed_max_m_s=8,jump_allowance_m=1,
           speed_rule='distance_m > 8*(source_frame-previous_accepted_frame)/59.94+1; height rejection has priority',
           smoothing_radius_source_frames=8,smoothing_max_accepted_gap_source_frames=30,
           smoothing_min_evidence_span_source_frames=4,
           smoothing='Separate, frozen-input x/y medians in inclusive +/-8F window in runs with accepted gaps <=30F. Window must span >=4F (original three 2F samples); no recursive updates, one-sided end windows allowed.',
           interpolation_max_endpoint_gap_source_frames=30,extrapolation=False,
           interpolation='Linear between smoothed accepted endpoints, every source frame. status is final; filter_status preserves rejection before interpolation.',
           wrist_window_source_frames=[-8,4],wrist_ratio_min=1.3,
           wrist='Both same-side players need >=1 finite nonnegative wrist_speed. Unique larger peak >=1.3*smaller; both zero/ties undecided. No guessed replacement.',
           output_fps=15,output_step_source_frames=2,
           rally_window_source_frames=[-60,90],court_xlim=XLIM,court_ylim=YLIM,
           plot_skip_margin_px=7,side_outside_x_thresholds=[-.12,1.12],side_outside_min_duration_sec=1,
           side_outside='Consecutive source frames, no bridging missing values; duration=(end-start)/59.94 >=1. Report cleaned positions and raw proxies separately over full exported point span.')


def write_readme(summary,hitters):
    abc.write_readme(summary,hitters)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review2',action='store_true',help='Rebuild human-reviewed hitters, reentry, offscreen holds and far depth')
    parser.add_argument('--far-depth-only',action='store_true',help='Reuse frozen B/C and near outputs; rebuild only final far A and videos')
    parser.add_argument('--skip-pose',action='store_true')
    parser.add_argument('--tracking-dir',type=Path,default=PIPE/'tracking_v2')
    parser.add_argument('--prepare-only',action='store_true',help='Run A/B and export diagnostics before ball inference/rendering')
    args=parser.parse_args()
    if args.review2:
        from review2 import main as review_main
        review_main()
        return
    if args.far_depth_only:
        from far_depth_revision import rebuild
        rebuild()
        return
    tracking_dir=args.tracking_dir.resolve()
    using_v2=tracking_dir==(PIPE/'tracking_v2').resolve()
    if using_v2 and args.skip_pose:
        parser.error('--skip-pose requires --tracking-dir pointing to the original tracking')
    assert OUT==ROOT/'outputs/shot_pipeline_20261117/court_view_p07_11'
    inputs=[PIPE/'labels_human/human_labels_20261118.csv', PIPE/'labels_human/README.md',
            PIPE/'pose/wrist_timeseries.csv',PIPE/'pose/extract_pose.py',PIPE/'pose/combine_audio_pose.py',
            REF/'clean_056_player_positions.py',REF/'render_056_cleaned_wide_court.py',REF/'056_cleaned_wide_court/README.md']
    inputs+=list((PIPE/'pose').rglob('*'))
    inputs=[p for p in inputs if p.is_file()]
    inputs+=[tracking_dir/f'point_{p:02d}/player_court_proxies.csv' for p in range(7,12)]
    inputs+=list((PIPE/'tracking').rglob('*'))
    inputs+=list((PIPE/'tracking_v2').rglob('*'))
    inputs=[p for p in inputs if p.is_file()]
    protected={str(p):sha(p) for p in inputs}
    before_path=OUT/'hitter_auto_before_abc.csv'
    if not before_path.exists():
        before_path.write_bytes((OUT/'hitter_auto.csv').read_bytes())
    previous_hitters=pd.read_csv(before_path)
    old_positions=OUT/'p07_F1_before_abc.csv'
    if not old_positions.exists():
        old=pd.read_csv(OUT/'cleaned_player_positions_p07.csv')
        old[(old.player_slot=='far_rear')&old.source_frame.between(23368,23870)].to_csv(old_positions,index=False,encoding='utf-8-sig',lineterminator='\n')
    dump(OUT/'input_sha256_before.json',protected)
    labels=pd.read_csv(inputs[0],keep_default_na=False)
    labels=labels[(labels.condition=='without_candidates')&(labels.status=='completed')].sort_values(['point','frame'])
    assert len(labels)==32 and labels.groupby('point').size().to_dict()=={7:8,8:8,9:8,10:4,11:4}
    tracks={p:pd.read_csv(tracking_dir/f'point_{p:02d}/player_court_proxies.csv') for p in range(7,12)}
    for p,d in tracks.items():
        assert not d.duplicated(['source_frame','player_slot']).any()
    original_tracks=tracks.copy()
    segments_by_point={}
    maps={};identity={};project,H=abc.geometry(load_module)
    for p in range(7,12):
        segments_by_point[p]=[dict(rally_no=int(rally),start_frame=int(g.frame.min())-60,end_frame=int(g.frame.max())+90)
                              for rally,g in labels[labels.point==p].groupby('rally_no',sort=True)]
        marked=abc.mark_occlusion(original_tracks[p])
        tracks[p],maps[p],identity[p]=abc.relink(marked,segments_by_point[p],
            lambda slot, source, frames: clean_player(slot,source,frames,far_revision=False),project,H)
        tracks[p].to_csv(OUT/f'identity_positions_p{p:02d}.csv',index=False,encoding='utf-8-sig',lineterminator='\n')
        maps[p].to_csv(OUT/f'identity_map_p{p:02d}.csv',index=False,encoding='utf-8-sig',lineterminator='\n')
    dump(OUT/'identity_summary.json',identity)
    if args.prepare_only:
        print(json.dumps({p:dict(switches=v['corrected_switches'],unlinked=v['unlinked_observation_count']) for p,v in identity.items()},ensure_ascii=False,indent=2),flush=True)
        return
    balls_path=OUT/'ball_candidates.csv'
    ball_python=ROOT/'study/third_party/racketvision-env/Scripts/python.exe'
    subprocess.run([str(ball_python),'-B','-X','utf8',str(OUT/'run_ball_candidates.py')],check=True)
    balls=pd.read_csv(balls_path)
    # ID2 is the line official verified in the point-8 source images. A v2
    # directory name alone does not establish that the known failure was fixed.
    point8_repaired=using_v2 and not ((tracks[8].track_id==2)&tracks[8].player_slot.str.startswith('near')).any()
    wrist=pd.read_csv(PIPE/'pose/wrist_timeseries.csv')
    if not args.skip_pose:
        extra,pose_manifest=supplement_pose(original_tracks)
        if len(extra):
            old=set(zip(wrist.point,wrist.source_frame,wrist.player_slot))
            extra=extra[[key not in old for key in zip(extra.point,extra.source_frame,extra.player_slot)]]
            wrist=pd.concat([wrist,extra],ignore_index=True)
    else:pose_manifest=dict(status='skipped')
    # Freeze a reproducible original-tracking reference before replacing wrists.
    baseline_hitters=decide_hitters(labels,wrist)
    baseline=pd.DataFrame(baseline_hitters)
    baseline.to_csv(OUT/'hitter_auto_tracking_v1.csv',index=False,encoding='utf-8-sig',lineterminator='\n')
    pose_retrack=dict(status='not_using_v2',windows=[])
    if using_v2:
        assert not args.skip_pose, 'v2 must recompute affected wrists; remove --skip-pose'
        extra,pose_retrack=retrack.recompute_pose(original_tracks,labels,source_frames,load_module,sha,dump)
        wrist=retrack.replace_wrist(wrist,extra)
    wrist=abc.remap_wrist(wrist,maps)
    wrist.to_csv(OUT/'wrist_identity.csv',index=False,encoding='utf-8-sig',lineterminator='\n')
    hitters=decide_hitters(labels,wrist)
    hitters=abc.combine_ball(hitters,tracks,balls)
    recomputed={(w['point'],w['center_frame']) for w in pose_retrack.get('windows',[])}
    for s in hitters:
        s['legacy_tracking_issue']=s['known_tracking_issue']
        s['known_tracking_issue']=bool(s['known_tracking_issue'] and not point8_repaired)
        s['tracking_version']='v2' if using_v2 else str(tracking_dir)
        s['pose_recomputed']=(s['point'],s['frame']) in recomputed
    hitters=retrack.add_lift(hitters,tracks)
    pd.DataFrame(hitters).to_csv(OUT/'hitter_auto.csv',index=False,encoding='utf-8-sig',lineterminator='\n')
    summary=dict(rules=RULES,points={},pose_supplement=pose_manifest,
                 tracking_directory=str(tracking_dir.relative_to(ROOT)),pose_retrack=pose_retrack,
                 point8_line_official_removed=bool(point8_repaired),
                 hitter_changes=retrack.changes(hitters,baseline),
                 jump_or_step_count=sum(s['jump_or_step_flag'] for s in hitters),
                 hitter_status_counts=dict(Counter(s['status'] for s in hitters)),hitter_decisions=hitters,
                 side_outside_intervals=[],raw_side_outside_intervals=[],independent_position_accuracy_validated=False,
                 independent_hitter_accuracy_validated=False)
    compact_identity={p:{k:v for k,v in d.items() if k!='unlinked_observations'} for p,d in identity.items()}
    summary.update(identity=compact_identity,ball_manifest=json.loads((OUT/'ball_manifest.json').read_text(encoding='utf-8')),
                   ball_candidate_shots=sum(s['ball_candidate_frames']>0 for s in hitters),
                   changes_from_previous=retrack.changes(hitters,previous_hitters),
                   requested_checks=[f'check_abc_p{p:02d}_F{f}.jpg' for p,f in [(7,23422),(10,30633),(11,32999)]])
    summary['rules']['legacy_height_rules_apply_to']='near_* and frozen pre-link B geometry only'
    from far_depth_revision import RULE as far_rule
    summary['rules'].update(far_depth=far_rule,
                            identity=dict(max_gap_sec=1,max_speed_m_s=8,allowance_m=1.5,fixed_per_rally=True),
                            ball=dict(window=[-3,3],box_pad_x_fraction=.3,box_pad_top_fraction=.3,ratio_strictly_less_than=.5))
    excluded=json.loads((PIPE/'tracking_v2/verification.json').read_text(encoding='utf-8'))['excluded_candidates'] if using_v2 else []
    summary['official_candidates']=[{k:r[k] for k in ('point','track_id','exclusion_mode','exclusion_ranges')} for r in excluded]
    summary['official_candidates_detail']='../tracking_v2/verification.json'
    summary['tracking_comparison']=pd.read_csv(PIPE/'tracking_v2/comparison.csv').to_dict('records') if using_v2 else []
    summary['rules']['lift']=dict(impact_window_inclusive=[-20,20],baseline_abs_frame_inclusive=[20,40],
                                threshold_fraction_of_baseline_median_height=.06,flag_min_frames=5,
                                correct_position=False)
    for p in range(7,12):
        from far_depth_revision import mark_occlusion as mark_final_occlusion
        tracks[p]=mark_final_occlusion(tracks[p])
        segments=segments_by_point[p]
        first=min(int(tracks[p].source_frame.min()),segments[0]['start_frame'])
        last=max(int(tracks[p].source_frame.max()),segments[-1]['end_frame'])
        all_frames=list(range(first,last+1))
        cleaned=[];diagnostics={}
        for slot in SLOTS:
            source=tracks[p][tracks[p].player_slot==slot].to_dict('records')
            rows,details=clean_player(slot,source,all_frames)
            cleaned.extend(rows);diagnostics[slot]=details
        cleaned.sort(key=lambda r:(r['source_frame'],SLOTS.index(r['player_slot'])))
        pd.DataFrame(cleaned).to_csv(OUT/f'cleaned_player_positions_p{p:02d}.csv',index=False,encoding='utf-8-sig',lineterminator='\n')
        if p==7:
            old=pd.read_csv(old_positions)
            new=[r['court_y']*23.77 for r in cleaned if r['player_slot']=='far_rear' and 23368<=r['source_frame']<=23870 and r['status'] in VALID]
            raw=original_tracks[p]
            raw=raw[(raw.player_slot=='far_rear')&raw.source_frame.between(23368,23870)]
            old_y=old.court_y.dropna()*23.77
            summary['p07_F1_y_comparison_m']=dict(first=23368,last=23870,
                    before_cleaned_range=[float(old_y.min()),float(old_y.max())],
                    before_raw_range=[float(raw.court_proxy_y.min()*23.77),float(raw.court_proxy_y.max()*23.77)],
                    after_cleaned_range=[min(new),max(new)],after_valid_frames=len(new))
        intervals=side_intervals(cleaned,p,'cleaned')
        raw_intervals=side_intervals(tracks[p].sort_values(['source_frame','player_slot']).to_dict('records'),p,'raw_proxy')
        summary['side_outside_intervals']+=intervals
        summary['raw_side_outside_intervals']+=raw_intervals
        info=dict(first_source_frame=first,last_source_frame=last,exported_frames=len(all_frames),
                  exported_rows=len(cleaned),input_rows=len(tracks[p]),cleaning_diagnostics=diagnostics,
                  status_counts_by_player=count_status(cleaned,'status'),filter_status_counts_by_player=count_status(cleaned,'filter_status'))
        info['rendering']=render_point(p,segments,cleaned,[s for s in hitters if s['point']==p],point8_repaired,balls)
        summary['points'][str(p)]=info
        dump(OUT/'cleaning_summary.json',summary)
        print(f'Point {p} complete: {info["rendering"]["frames"]} frames, decoded through EOF',flush=True)
    assert all(sha(Path(p))==h for p,h in protected.items()), 'Read-only input changed'
    summary['input_hashes_verified_unchanged']=len(protected)
    summary['remaining_issues']=dict(independent_position_accuracy_validated=False,
        far_depth='Occluded depth is interpolation, not observation; motion during occlusion is invisible; no independent ground truth.',
        p08_F25470='B unchanged: track279 cannot link to track248; missing observation, interpolated court point')
    summary['large_files']=[]
    write_readme(summary,hitters)
    inventory=abc.inventory()
    summary['large_files']=[r for r in inventory if r['bytes']>256*1024]
    pd.DataFrame(inventory).to_csv(OUT/'file_inventory.csv',index=False,encoding='utf-8-sig',lineterminator='\n')
    dump(OUT/'cleaning_summary.json',summary)
    write_readme(summary,hitters)
    print(json.dumps(dict(hitter_status_counts=summary['hitter_status_counts'],
                          side_outside_intervals=summary['side_outside_intervals'],
                          large_files=summary['large_files']),ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
