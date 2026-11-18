"""Final far-player A only. Saved B/C and near-player tables are immutable inputs."""
from bisect import bisect_left
from collections import Counter
import copy
import json
from statistics import median
import subprocess
import sys

import numpy as np
import pandas as pd

import build_court_view as b
import a_height_revision as previous
from speed_reset import SpeedReset, STATUS as SPEED_RESET

DEPTH = 'observed_x_depth_interpolated'
ENVELOPE = 'observed_x_depth_envelope'
ACCEPTED = ('observed', DEPTH, 'observed_x_depth_serve_anchor', 'observed_x_depth_lift_held', ENVELOPE, SPEED_RESET)
STATUSES = ('observed', DEPTH, 'rejected_box_height', 'rejected_speed', 'interpolated', 'missing')
CHECKS = [(7,23422),(10,30633),(11,32999)]
RULE = dict(horizon_y_px=415., k=.73, height_reject_multiplier=1.35,
            parameter_source='User-specified constants; no fitting performed',
            occlusion='same-frame near/far horizontal overlap >= 0.10 * far width AND near y1 < far y2; no height ratio',
            depth='Nearest accepted unoccluded y2 before/after within the same fixed person and rally; linear in source frame; unlimited gap; one side held; neither missing',
            x='current box centre x; project (x, depth_y_px) with unchanged tracking homography',
            anchor_selection='Iterate speed filtering and accepted unoccluded depth anchors to a fixed point; anchors refer to final filter_status=observed, before smoothing',
            height_reset=False, head_anchor=False, speed_max_m_s=8, jump_allowance_m=1,
            smoothing_radius_source_frames=8, interpolation_max_endpoint_gap_source_frames=30,
            limitation='Occluded depth is interpolation, not observation; forward/backward motion during occlusion is invisible')


def mark_occlusion(table):
    """Only A changes its overlap criterion. B source files are never rewritten."""
    rows=table.to_dict('records')
    near={}
    for r in rows:
        if r['player_slot'].startswith('near_'):
            near.setdefault(r['source_frame'],[]).append(r)
    for r in rows:
        if not r['player_slot'].startswith('far_'):
            continue
        ids=[]
        for n in near.get(r['source_frame'],[]):
            overlap=max(0.,min(n['x2'],r['x2'])-max(n['x1'],r['x1']))
            if overlap >= .1*(r['x2']-r['x1']) and n['y1'] < r['y2']:
                ids.append(str(int(n['track_id'])))
        r.update(lower_occluded=bool(ids),occluder_track_ids=';'.join(ids))
    return pd.DataFrame(rows)


def clean_far(slot, source, frames, *, depth_overrides=None, x_reference=None, envelopes=None, speed_reset=False):
    project,H=b.abc.geometry(b.load_module)
    depth_overrides=depth_overrides or {}
    envelopes=envelopes or {}
    diagnostics={}; iterations={}
    for rally in sorted({int(r['rally_no']) for r in source}):
        raw=sorted((r for r in source if int(r['rally_no'])==rally),key=lambda r:r['source_frame'])
        candidates={int(r['source_frame']):r for r in raw if not r['lower_occluded'] and
                    r['y2']-r['y1'] <= 1.35*.73*(r['y2']-415)}
        candidates={f:dict(r,anchor_depth_y_px=envelopes.get(f,{}).get('envelope_y2_px',r['y2']))
                    for f,r in candidates.items()}
        for r in raw:
            override=depth_overrides.get(int(r['source_frame']))
            if override and override['status']=='observed_x_depth_serve_anchor':
                candidates[int(r['source_frame'])]=dict(r,anchor_depth_y_px=override['depth_y_px'])
        anchors=candidates.copy(); seen=set()
        for iteration in range(len(raw)+2):
            key=tuple(anchors)
            assert key not in seen, 'Depth-anchor/speed filtering did not converge'
            seen.add(key)
            af=sorted(anchors); accepted=[]; current=[]
            speed_streak=SpeedReset(speed_reset)
            for original in raw:
                r=dict(original); f=int(r['source_frame']); occ=bool(r['lower_occluded'])
                # Legacy head fields in B are not final A observations.
                for name in ('head_reference_height_px','head_foot_y_px','head_court_x','head_court_y',
                             'height_reference_window_observations','height_reference_scope'):
                    r[name]=''
                height=r['y2']-r['y1']; reference=.73*(r['y2']-415)
                envelope=envelopes.get(f,{})
                depth=envelope.get('envelope_y2_px',r['y2']); a=z=''
                if occ:
                    i=bisect_left(af,f)
                    a=af[i-1] if i else ''
                    z=af[i] if i<len(af) else ''
                    if a!='' and z!='':
                        t=(f-a)/(z-a); depth=(1-t)*anchors[a].get('anchor_depth_y_px',anchors[a]['y2'])+t*anchors[z].get('anchor_depth_y_px',anchors[z]['y2'])
                    elif a!='' or z!='':
                        anchor=anchors[a if a!='' else z]
                        depth=anchor.get('anchor_depth_y_px',anchor['y2'])
                    else: depth=None
                override=depth_overrides.get(f)
                if override:
                    depth=override['depth_y_px']
                x=y=jump=''; speed_bad=False
                if depth is not None:
                    x,y=project(H,(r['x1']+r['x2'])/2,depth)
                    if x_reference is not None and f in x_reference:
                        x=x_reference[f]
                    if accepted:
                        prev=accepted[-1]
                        jump=b.distance((x,y),(prev['effective_court_x'],prev['effective_court_y']))
                        speed_bad=jump > 8*(f-prev['source_frame'])/b.FPS+1
                status=(override['status'] if override else 'missing' if depth is None else
                        'rejected_box_height' if not occ and height>1.35*reference else
                        'rejected_speed' if speed_bad else DEPTH if occ else
                        ENVELOPE if envelope.get('envelope_shift_m',0)>=.5 else 'observed')
                reset_start=speed_streak.update(f,status,rally)
                reset_base=DEPTH if occ else ENVELOPE if envelope.get('envelope_shift_m',0)>=.5 else 'observed'
                if reset_start is not None:status=SPEED_RESET
                r.update(envelope)
                r.update(source_frame=f,status=status,filter_status=status,
                         speed_reset_frame=f if reset_start is not None else '',
                         speed_reject_start_frame=reset_start if reset_start is not None else '',
                         speed_reset_base_status=reset_base if reset_start is not None else '',
                         court_x=x if status in ACCEPTED else '',court_y=y if status in ACCEPTED else '',
                         effective_court_x=x,effective_court_y=y,depth_y_px=depth if depth is not None else '',
                         depth_anchor_before_frame=a,depth_anchor_after_frame=z,
                         box_height_px=height,reference_height_px=reference,jump_m=jump,
                         head_anchor_used=False,height_reset_start_frame='',height_reset_trigger_frame='',
                         interpolation_start_frame='',interpolation_end_frame='',
                         raw_court_x='',raw_court_y='',smoothing_shift_m='',
                         raw_previous_frame=raw[len(current)-1]['source_frame'] if current else '',
                         raw_jump_m='',raw_speed_limit_m='')
                if current and depth is not None and current[-1]['effective_court_x']!='':
                    r['raw_jump_m']=b.distance((x,y),(current[-1]['effective_court_x'],current[-1]['effective_court_y']))
                    r['raw_speed_limit_m']=8*(f-current[-1]['source_frame'])/b.FPS+1
                current.append(r)
                if status in ACCEPTED: accepted.append(r)
            updated={r['source_frame']:dict(r,anchor_depth_y_px=r['depth_y_px']) for r in current
                     if r['filter_status'] in ('observed',ENVELOPE,'observed_x_depth_serve_anchor') or
                     (r['filter_status']==SPEED_RESET and not r['lower_occluded'])}
            if tuple(updated)==tuple(anchors): break
            anchors=updated
        else: raise AssertionError('Depth anchors failed to converge')
        iterations[str(rally)]=iteration+1
        runs=[]
        for r in accepted:
            r.update(raw_court_x=r['court_x'],raw_court_y=r['court_y'])
            if not runs or r['source_frame']-runs[-1][-1]['source_frame']>30: runs.append([])
            runs[-1].append(r)
        for run in runs:
            for r in run:
                window=[p for p in run if abs(p['source_frame']-r['source_frame'])<=8]
                if window[-1]['source_frame']-window[0]['source_frame']>=4:
                    r['court_x']=median(p['raw_court_x'] for p in window)
                    r['court_y']=median(p['raw_court_y'] for p in window)
                r['smoothing_shift_m']=b.distance((r['court_x'],r['court_y']),(r['raw_court_x'],r['raw_court_y']))
        interpolated={}
        for a,z in zip(accepted,accepted[1:]):
            start,end=a['source_frame'],z['source_frame']
            if end-start>30: continue
            for f in range(start+1,end):
                t=(f-start)/(end-start)
                interpolated[f]=(a['court_x']+t*(z['court_x']-a['court_x']),a['court_y']+t*(z['court_y']-a['court_y']),start,end)
        for r in current: diagnostics[r['source_frame']]=r
        template={k:'' for k in current[0]}
        for f,(x,y,a,z) in interpolated.items():
            r=diagnostics.setdefault(f,dict(template,source_frame=f,player_slot=slot,filter_status='missing',rally_no=rally))
            assert r['filter_status'] not in ACCEPTED
            r.update(status='interpolated',court_x=x,court_y=y,interpolation_start_frame=a,interpolation_end_frame=z)
    result=[]
    for f in frames:
        r=diagnostics.get(f,dict(source_frame=f,player_slot=slot,status='missing',filter_status='missing',court_x='',court_y=''))
        result.append(r)
    return result,dict(global_median_box_height_px=None,height_resets=[],height_model=RULE,anchor_iterations_by_rally=iterations)


def snapshot():
    path=b.OUT/'far_depth_revision_before.json'
    if path.exists(): return json.loads(path.read_text(encoding='utf-8'))
    old=json.loads((b.OUT/'a_height_revision_before.json').read_text(encoding='utf-8'))
    summary=json.loads((b.OUT/'cleaning_summary.json').read_text(encoding='utf-8'))
    frozen={name:b.sha(b.ROOT/name) for name in old['frozen_sha256']}
    protected={name:b.sha(b.ROOT/name) for name in old['protected_sha256']}
    near={}; metrics=[]
    for p in range(7,12):
        table=pd.read_csv(b.OUT/f'cleaned_player_positions_p{p:02d}.csv')
        near[str(p)]=dict(columns=list(table),sha256=previous.near_digest(table,list(table)))
        metrics+=previous.metrics(p,table,summary['points'][str(p)]['rendering']['segments'])
    ball={str(p.relative_to(b.ROOT)):[p.stat().st_size,p.stat().st_mtime_ns]
          for d in (b.ROOT/'outputs').glob('ball_quality_*') for p in d.rglob('*') if p.is_file()}
    value=dict(frozen_sha256=frozen,protected_sha256=protected,ball_quality_metadata=ball,near=near,before_metrics=metrics,
               previous_report_sha256=b.sha(b.OUT/'a_height_revision_report.json'))
    b.dump(path,value)
    return value


def rebuild():
    before=snapshot()
    summary=json.loads((b.OUT/'cleaning_summary.json').read_text(encoding='utf-8'))
    summary['rules'].pop('head_anchor',None)
    summary['rules']['far_depth']=RULE
    summary['requested_checks']=[f'check_abc_p{p:02d}_F{f}.jpg' for p,f in CHECKS]
    summary['remaining_issues']=dict(independent_position_accuracy_validated=False,
        far_depth=RULE['limitation'],p08_F25470='Frozen B: track279 unlinked; ordinary short-gap interpolation remains')
    summary.pop('a_height_revision_report',None)
    summary['far_depth_revision_report']='far_depth_revision_report.json'
    balls=pd.read_csv(b.OUT/'ball_candidates.csv')
    for p in range(7,12):
        old=pd.read_csv(b.OUT/f'cleaned_player_positions_p{p:02d}.csv').fillna('')
        frames=sorted(old.source_frame.unique().tolist())
        fixed=mark_occlusion(pd.read_csv(b.OUT/f'identity_positions_p{p:02d}.csv').fillna(''))
        cleaned=old[old.player_slot.str.startswith('near_')].to_dict('records')
        info=summary['points'][str(p)]
        for slot in b.SLOTS[2:]:
            rows,details=clean_far(slot,fixed[fixed.player_slot==slot].to_dict('records'),frames)
            cleaned+=rows; info['cleaning_diagnostics'][slot]=details
        cleaned.sort(key=lambda r:(r['source_frame'],b.SLOTS.index(r['player_slot'])))
        pd.DataFrame(cleaned).to_csv(b.OUT/f'cleaned_player_positions_p{p:02d}.csv',index=False,encoding='utf-8-sig',lineterminator='\n')
        info.update(status_counts_by_player=b.count_status(cleaned,'status'),filter_status_counts_by_player=b.count_status(cleaned,'filter_status'))
        info['rendering']=b.render_point(p,info['rendering']['segments'],cleaned,[s for s in summary['hitter_decisions'] if s['point']==p],summary['point8_line_official_removed'],balls)
        b.dump(b.OUT/'cleaning_summary.json',summary)
        print(f'Point {p}: rebuilt, full decode verified',flush=True)
    finalize(summary,before)


def finalize(summary=None,before=None):
    if summary is None: summary=json.loads((b.OUT/'cleaning_summary.json').read_text(encoding='utf-8'))
    summary['side_outside_intervals']=[]
    for p in range(7,12):
        table=pd.read_csv(b.OUT/f'cleaned_player_positions_p{p:02d}.csv').fillna('')
        summary['side_outside_intervals']+=b.side_intervals(table.to_dict('records'),p,'cleaned')
    table=pd.read_csv(b.OUT/'cleaned_player_positions_p07.csv')
    values=table.loc[(table.player_slot=='far_rear')&table.source_frame.between(23368,23870)&table.status.isin(b.VALID),'court_y']*23.77
    summary['p07_F1_y_comparison_m'].update(after_cleaned_range=[float(values.min()),float(values.max())],after_valid_frames=len(values))
    b.dump(b.OUT/'cleaning_summary.json',summary)
    verify_and_report(before)


def verify_and_report(before=None):
    current=json.loads((b.OUT/'cleaning_summary.json').read_text(encoding='utf-8'))
    if current.get('review2'):
        from verify_review2 import main as verify_review
        verify_review()
        return
    if before is None: before=json.loads((b.OUT/'far_depth_revision_before.json').read_text(encoding='utf-8'))
    for section in ('frozen_sha256','protected_sha256'):
        for name,digest in before[section].items(): assert b.sha(b.ROOT/name)==digest,name
    ball={str(p.relative_to(b.ROOT)):[p.stat().st_size,p.stat().st_mtime_ns]
          for d in (b.ROOT/'outputs').glob('ball_quality_*') for p in d.rglob('*') if p.is_file()}
    assert ball==before['ball_quality_metadata']
    assert b.sha(b.OUT/'a_height_revision_report.json')==before['previous_report_sha256']
    summary=json.loads((b.OUT/'cleaning_summary.json').read_text(encoding='utf-8'))
    after=[]; bounds=[]; video_audit={}; checks=[]
    project,H=b.abc.geometry(b.load_module)
    for p in range(7,12):
        table=pd.read_csv(b.OUT/f'cleaned_player_positions_p{p:02d}.csv')
        reference=before['near'][str(p)]
        assert previous.near_digest(table,reference['columns'])==reference['sha256'],f'near p{p}'
        assert not table[table.player_slot.str.startswith('far_')].filter_status.isin(['observed_head_anchored','observed_after_height_reset']).any()
        segments=summary['points'][str(p)]['rendering']['segments']
        after+=previous.metrics(p,table,segments)
        selected=pd.Series(False,index=table.index)
        for s in segments: selected|=table.source_frame.between(s['start_frame'],s['end_frame'])
        fixed=mark_occlusion(pd.read_csv(b.OUT/f'identity_positions_p{p:02d}.csv'))
        for slot in b.SLOTS[2:]:
            g=table[table.player_slot==slot].set_index('source_frame')
            raw=fixed[fixed.player_slot==slot].set_index('source_frame')
            accepted=[]; last_rally=None
            for f,r in raw.sort_index().iterrows():
                z=g.loc[f]; occ=bool(r.lower_occluded)
                assert bool(z.lower_occluded)==occ
                if r.rally_no!=last_rally: accepted=[];last_rally=r.rally_no
                anchors=g[(g.rally_no==r.rally_no)&g.filter_status.eq('observed')]
                if occ:
                    left=anchors[anchors.index<f]; right=anchors[anchors.index>f]
                    if len(left) and len(right):
                        a,c=left.iloc[-1],right.iloc[0]; t=(f-a.name)/(c.name-a.name)
                        depth=(1-t)*a.y2+t*c.y2
                    elif len(left) or len(right): depth=(left.iloc[-1] if len(left) else right.iloc[0]).y2
                    else: depth=None
                else: depth=r.y2
                if depth is None: assert z.filter_status=='missing';continue
                assert np.isclose(z.depth_y_px,depth)
                xy=project(H,(r.x1+r.x2)/2,depth)
                assert np.allclose(xy,[z.effective_court_x,z.effective_court_y],rtol=1e-12,atol=1e-12)
                speed=bool(accepted) and b.distance(xy,accepted[-1][1])>8*(f-accepted[-1][0])/b.FPS+1
                expected='rejected_box_height' if not occ and r.y2-r.y1>1.35*.73*(r.y2-415) else 'rejected_speed' if speed else DEPTH if occ else 'observed'
                assert z.filter_status==expected,(p,slot,f,z.filter_status,expected)
                if expected in ACCEPTED: accepted.append((f,xy))
            for rally,all_rows in g.groupby('rally_no'):
                accepted_rows=all_rows[all_rows.filter_status.isin(ACCEPTED)].sort_index()
                runs=[]
                for f,r in accepted_rows.iterrows():
                    assert np.allclose([r.raw_court_x,r.raw_court_y],[r.effective_court_x,r.effective_court_y])
                    if not runs or f-runs[-1][-1][0]>30: runs.append([])
                    runs[-1].append((f,r))
                for run in runs:
                    for f,r in run:
                        window=[(n,z) for n,z in run if abs(n-f)<=8]
                        xy=([median(z.raw_court_x for _,z in window),median(z.raw_court_y for _,z in window)]
                            if window[-1][0]-window[0][0]>=4 else [r.raw_court_x,r.raw_court_y])
                        assert np.allclose([r.court_x,r.court_y],xy,rtol=1e-12,atol=1e-12)
            for f,r in g[g.status.eq('interpolated')].iterrows():
                a,z=int(r.interpolation_start_frame),int(r.interpolation_end_frame)
                assert a<f<z and z-a<=30
                assert g.loc[a,'filter_status'] in ACCEPTED and g.loc[z,'filter_status'] in ACCEPTED
                t=(f-a)/(z-a)
                xy=(1-t)*g.loc[a,['court_x','court_y']].astype(float)+t*g.loc[z,['court_x','court_y']].astype(float)
                assert np.allclose([r.court_x,r.court_y],xy,rtol=1e-12,atol=1e-12)
            for scope,mask in [('displayed_source_intervals',selected),('full_export',pd.Series(True,index=table.index))]:
                good=table[mask & table.player_slot.eq(slot)&table.status.isin(b.VALID)]
                bounds.append(dict(point=p,player_slot=slot,name=b.NAMES[slot],scope=scope,
                                   valid_frames=len(good),below_11_9_m=int((good.court_y*23.77<11.9).sum()),above_28_m=int((good.court_y*23.77>28).sum())))
        for cp,cf in CHECKS:
            if cp==p:
                checks+=table[(table.source_frame==cf)&table.player_slot.str.startswith('far_')].fillna('').to_dict('records')
        frame_map=pd.read_csv(b.OUT/f'video_source_frame_map_p{p:02d}.csv')
        expected=[]
        for segment in segments:
            times=list(range(segment['start_frame'],segment['end_frame']+1,2))
            requested={7:[23422,23548],8:[25470],10:[30633],11:[32999]}.get(p,[])
            expected+=sorted(set(times+[f for f in requested if segment['start_frame']<=f<=segment['end_frame']]))
        assert expected==frame_map.source_frame.tolist()
        video=b.OUT/f'point_{p:02d}.mp4'
        subprocess.run([b.FFMPEG,'-v','error','-xerror','-i',str(video),'-map','0:v:0','-f','null','-'],check=True,capture_output=True)
        probe=json.loads(subprocess.check_output([b.FFPROBE,'-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames','-of','json',str(video)]))
        assert int(probe['streams'][0]['nb_read_frames'])==len(frame_map)
        video_audit[str(p)]=dict(frames=len(frame_map),full_video_decode_verified=True,source_frame_map_verified=True)
    report=dict(scope='Displayed rally intervals, ALL original source frames including unsampled ones; no spatial clipping. Quantiles use valid final statuses; maximum Euclidean court step uses adjacent valid original frames within each rally.',
                before_metrics=before['before_metrics'],after_metrics=after,out_of_depth_range=bounds,requested_checks=checks,
                preservation=dict(B_C_sha256_unchanged=len(before['frozen_sha256']),protected_sha256_unchanged=len(before['protected_sha256']),ball_quality_metadata_unchanged=len(ball),near_all_previous_columns_unchanged=True),
                videos=video_audit,far_rule=RULE)
    b.dump(b.OUT/'far_depth_revision_report.json',report)
    b.dump(b.OUT/'verification.json',dict(preservation=report['preservation'],points=video_audit,far_geometry_and_speed_verified=True,far_smoothing_and_short_interpolation_verified=True))
    summary.update(displayed_position_metrics=after,preservation_audit=report['preservation'],verification='verification.json')
    for p in range(7,12):
        summary['points'][str(p)]['far_height_model']=dict(horizon_y_px=415,k=.73,fitted=False)
    pd.DataFrame([{k:v for k,v in r.items() if k not in ('status_counts','filter_status_counts','max_step_ties')} for r in after]).to_csv(b.OUT/'displayed_position_metrics.csv',index=False,encoding='utf-8-sig')
    for field,name in [('status','far_status_counts.csv'),('filter_status','far_filter_status_counts.csv')]:
        counts=[dict(point=r['point'],player_slot=r['player_slot'],scope='displayed_source_intervals',**{s:r[field+'_counts'].get(s,0) for s in STATUSES}) for r in after if r['player_slot'].startswith('far_')]
        counts += [dict(point=p,player_slot=slot,scope='full_export',**{s:summary['points'][str(p)][field+'_counts_by_player'][slot].get(s,0) for s in STATUSES}) for p in range(7,12) for slot in b.SLOTS[2:]]
        pd.DataFrame(counts).to_csv(b.OUT/name,index=False,encoding='utf-8-sig')
    pd.DataFrame(bounds).to_csv(b.OUT/'far_depth_out_of_range_counts.csv',index=False,encoding='utf-8-sig')
    b.dump(b.OUT/'cleaning_summary.json',summary)
    write_report(report,summary)
    b.write_readme(summary,summary['hitter_decisions'])
    inventory=b.abc.inventory()
    pd.DataFrame(inventory).to_csv(b.OUT/'file_inventory.csv',index=False,encoding='utf-8-sig')
    summary['large_files']=[r for r in inventory if r['bytes']>256*1024]
    b.dump(b.OUT/'cleaning_summary.json',summary)
    print(json.dumps(dict(preservation=report['preservation'],videos=video_audit),ensure_ascii=False,indent=2),flush=True)


def write_report(report,summary):
    lines=['# A：奥側の前後位置補間への修正','',
           '5本のMP4を再生成し、全編復号・コマ数・元F対応を確認。B/Cの16ファイルとtracking/・tracking_v2/・pose/の215ファイルはSHA-256一致。ball_quality_*は全ファイルの一覧・サイズ・更新時刻一致。手前側は全従来列が一致。git commitなし。','',
           '高さ式は全点で地平線415px、k=0.73を指定値のまま使用し、再推定していない。下隠れは手前枠との横重なりが奥枠幅の10%以上かつ手前y1<奥y2のみ。隠れた奥側は同一人物・同一ラリーの前後の下隠れでない採用点の画像y2を時間補間（長さ無制限、片側保持、両側なしは欠測）。横は当該Fの枠中心。補間後にtrackingと同じ射影、従来の速度規則・±8元F中央値平滑化・30元F以内の通常補間を適用。補間のアンカーは速度採用結果が一致するまで更新する。','',
           '**限界：下が隠れている間の前後位置は観測ではなく補間であり、その間の前後への動きは見えない。位置精度の改善は独立した正解で未検証。**','',
           '## 集計範囲','',
           '前回a_height_revision_report.mdと同じ表示区間内の全元F（動画で間引いたFも含む）。空間クリップなし。最終有効statusのcourt_y×23.77mから百分位を算出。同じラリーの隣接する有効元Fのユークリッド移動の最大。欠測・ラリー境界はまたがない。前回欄は今回修正直前のCSVから再集計。N1=near_front、N2=near_rear、F1=far_rear、F2=far_front。','',
           '## 奥側の最終status件数','',
           '表示区間内。全出力範囲もfar_status_counts.csvに保存。棄却後に通常補間されたFはinterpolatedとなるため、補間前の棄却はfar_filter_status_counts.csvを参照。奥側のhead_anchoredとafter_height_resetは全点0。','',
           '|点|選手|observed|observed_x_depth_interpolated|rejected_box_height|rejected_speed|interpolated|missing|',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for r in report['after_metrics']:
        if r['player_slot'].startswith('far_'):
            lines.append('|'+ '|'.join([str(r['point']),r['name'],*(str(r['status_counts'].get(s,0)) for s in STATUSES)])+'|')
    lines+=['','## 4人の位置・最大移動（前回 → 今回）','',
            '|点|選手|有効F/対象F 前→今|y5% m 前→今|y50% m 前→今|y95% m 前→今|最大移動 m/元F 前→今|元Fの遷移 前→今|',
            '|---|---|---|---|---|---|---|---|']
    for old,new in zip(report['before_metrics'],report['after_metrics']):
        assert (old['point'],old['player_slot'])==(new['point'],new['player_slot'])
        values=[str(new['point']),new['name'],f'{old["valid_source_frames"]}/{old["total_source_frames"]} → {new["valid_source_frames"]}/{new["total_source_frames"]}']
        for k in ('court_y_p05_m','court_y_p50_m','court_y_p95_m','max_step_m_per_source_frame'):
            digits=6 if k.startswith('max') else 3
            values.append(f'{old[k]:.{digits}f} → {new[k]:.{digits}f}')
        values.append(f'F{old["max_step_from_frame"]}→F{old["max_step_to_frame"]} / F{new["max_step_from_frame"]}→F{new["max_step_to_frame"]}')
        lines.append('|'+ '|'.join(values)+'|')
    lines+=['','## 奥側の範囲外の元F数','',
            '表示区間内の最終有効位置で集計。y<11.9mとy>28mは厳密不等号。全出力範囲もfar_depth_out_of_range_counts.csvに併記。欠測は分母・件数に含めない。','',
            '|点|選手|有効F|y<11.9m|y>28m|','|---|---|---:|---:|---:|']
    for r in report['out_of_depth_range']:
        if r['scope']=='displayed_source_intervals':
            lines.append(f'|{r["point"]}|{r["name"]}|{r["valid_frames"]}|{r["below_11_9_m"]}|{r["above_28_m"]}|')
    lines+=['',
            'ネットより手前のy<11.9mは表示区間内の奥側全員で0Fとなった。一方、y>28mは第7点F1で344F、第8点F1で1F、第9点F2で11F、第11点F2で56F残る。第11点F2のy95%と最大移動も増加している。指定規則への変更を位置精度の一様な改善とはみなせない。範囲外を消すクリップや追加棄却は行っていない。',
            '第7点F23422のF1は下隠れで前後補間となるが29.640m。第10点F30633のF1は下隠れ条件に該当せず、下端観測で12.306m。第11点F32999のF2も下隠れに該当せず、下端観測で26.441mとなる。']
    lines+=['','## 動画','']
    for p in range(7,12):
        v=summary['points'][str(p)]['rendering']
        lines.append(f'- [第{p}点](point_{p:02d}.mp4)：{v["frames"]}コマ、{v["duration_sec"]:.3f}秒、全編復号済み。')
    lines+=['','## 指定確認画像','']
    for p,f in CHECKS:
        lines.append(f'![第{p}点 F{f}](check_abc_p{p:02d}_F{f}.jpg)')
        for r in report['requested_checks']:
            if r['source_frame']==f:
                y=r['court_y']*23.77 if r['court_y']!='' else None
                lines.append(f'- {b.NAMES[r["player_slot"]]}：track {int(r["track_id"])}、下隠れ={r["lower_occluded"]}、{r["status"]}、court_y={y:.3f}m。' if y is not None else f'- {b.NAMES[r["player_slot"]]}：{r["status"]}。')
    (b.OUT/'far_depth_revision_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')


if __name__=='__main__':
    if '--verify' in sys.argv: verify_and_report()
    elif '--finalize' in sys.argv: finalize()
    else: rebuild()
