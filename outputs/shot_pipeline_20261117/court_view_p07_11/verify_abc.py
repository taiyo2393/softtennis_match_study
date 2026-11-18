"""Independent audit of saved geometry, identities, evidence and full video decoding."""
from collections import Counter
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

import build_court_view as b


def audit_decision(h):
    slots=[h.player_1,h.player_2]
    evidence=json.loads(h.ball_evidence_json)
    minima=[]
    for i in range(2):
        values=[(r['distances_px'][i],r['frame'],r['box_widths_px'][i]) for r in evidence if r['distances_px'][i] is not None]
        minima.append(min(values) if values else None)
        saved=getattr(h,f'ball_player_{i+1}_distance_px')
        assert (minima[-1] is None and pd.isna(saved)) or np.isclose(minima[-1][0],saved)
    ball=''
    if all(v is not None for v in minima):
        distances=[v[0] for v in minima]
        near=int(distances[1]<distances[0]);far=1-near
        if distances[near]<distances[far] and ((distances[near]==0 and distances[far]>=minima[far][2]) or distances[far]>0 and distances[near]/distances[far]<.5):
            ball=slots[near]
    wrist=''
    speeds=[h.player_1_max,h.player_2_max]
    if np.isfinite(speeds).all() and max(speeds)>min(speeds) and max(speeds)>=1.3*min(speeds):
        wrist=slots[int(speeds[1]>speeds[0])]
    assert (h.ball_hitter_slot if pd.notna(h.ball_hitter_slot) else '')==ball
    assert (h.wrist_hitter_slot if pd.notna(h.wrist_hitter_slot) else '')==wrist
    expected='undecided';winner=''
    if ball and wrist and ball==wrist:expected,winner='auto_ball_wrist',ball
    elif ball and not wrist:expected,winner='auto_ball',ball
    elif wrist and not evidence:expected,winner='auto_wrist',wrist
    assert h.status==expected
    assert (h.hitter_slot if pd.notna(h.hitter_slot) else '')==winner


def main():
    out=b.OUT
    summary=json.loads((out/'cleaning_summary.json').read_text(encoding='utf-8'))
    if 'far_depth' in summary['rules']:
        from far_depth_revision import verify_and_report
        verify_and_report()
        return
    protected=json.loads((out/'input_sha256_before.json').read_text(encoding='utf-8'))
    assert all(b.sha(Path(p))==digest for p,digest in protected.items())
    summary['input_hashes_verified_unchanged']=len(protected)
    summary['official_candidates']=[{k:r[k] for k in ('point','track_id','exclusion_mode','exclusion_ranges')} for r in summary['official_candidates']]
    summary['official_candidates_detail']='../tracking_v2/verification.json'
    b.dump(out/'cleaning_summary.json',summary)
    snapshot=json.loads((out/'ball_quality_readonly_snapshot.json').read_text(encoding='utf-8'))
    now={str(p.relative_to(b.ROOT)):[p.stat().st_size,p.stat().st_mtime_ns] for d in (b.ROOT/'outputs').glob('ball_quality_*') for p in d.rglob('*') if p.is_file()}
    assert snapshot==now,'ball_quality_* inventory/metadata changed'
    labels=pd.read_csv(b.PIPE/'labels_human/human_labels_20261118.csv')
    labels=labels[(labels.condition=='without_candidates')&(labels.status=='completed')]
    hits=pd.read_csv(out/'hitter_auto.csv')
    assert len(hits)==32 and set(zip(hits.point,hits.frame))==set(zip(labels.point,labels.frame))
    assert dict(Counter(hits.status))==summary['hitter_status_counts']
    for h in hits.itertuples():audit_decision(h)
    balls=pd.read_csv(out/'ball_candidates.csv')
    assert not balls.duplicated(['point','source_frame']).any()
    assert balls.review_status.eq('unreviewed').all()
    assert summary['ball_candidate_shots']==int(hits.ball_candidate_frames.gt(0).sum())
    wrist=pd.read_csv(out/'wrist_identity.csv')
    for h in hits.itertuples():
        for i in (1,2):
            slot=getattr(h,f'player_{i}')
            values=wrist.loc[(wrist.point==h.point)&(wrist.player_slot==slot)&wrist.source_frame.between(h.frame-8,h.frame+4),'wrist_speed']
            values=values[np.isfinite(values)&(values>=0)]
            saved=getattr(h,f'player_{i}_max')
            assert (len(values)==0 and pd.isna(saved)) or np.isclose(values.max(),saved,rtol=1e-12)
    project,H=b.abc.geometry(b.load_module)
    audit=dict(input_hashes_verified_unchanged=len(protected),ball_quality_files_metadata_unchanged=len(snapshot),hitter_decisions_verified=32,points={})
    for point in range(7,12):
        raw=pd.read_csv(b.PIPE/f'tracking_v2/point_{point:02d}/player_court_proxies.csv')
        xy=np.array([project(H,r.bottom_center_x,r.bottom_center_y) for r in raw.itertuples()])
        assert np.allclose(xy,raw[['court_proxy_x','court_proxy_y']],rtol=1e-12,atol=1e-12)
        positions=pd.read_csv(out/f'cleaned_player_positions_p{point:02d}.csv')
        fixed=pd.read_csv(out/f'identity_positions_p{point:02d}.csv')
        boxes=fixed.set_index(['source_frame','player_slot'])
        for shot in hits[hits.point==point].itertuples():
            for candidate in json.loads(shot.ball_evidence_json):
                f=candidate['frame']
                ball=balls[(balls.point==point)&(balls.source_frame==f)].iloc[0]
                assert np.allclose([candidate['x_px'],candidate['y_px'],candidate['confidence']],[ball.x_px,ball.y_px,ball.confidence])
                for i,slot in enumerate((shot.player_1,shot.player_2)):
                    key=(f,slot)
                    if key not in boxes.index:
                        assert candidate['distances_px'][i] is None
                        continue
                    box=boxes.loc[key]
                    w,h=box.x2-box.x1,box.y2-box.y1
                    xgap=max(0,box.x1-.3*w-ball.x_px,ball.x_px-box.x2-.3*w)
                    ygap=max(0,box.y1-.3*h-ball.y_px,ball.y_px-box.y2)
                    assert np.isclose(candidate['distances_px'][i],np.hypot(xgap,ygap))
        for (_,slot),g in fixed.groupby(['rally_no','player_slot']):
            previous=None
            for r in g.sort_values('source_frame').itertuples():
                if r.identity_status=='linked' and previous is not None:
                    dt=(r.source_frame-previous.source_frame)/b.DECODE_FPS
                    d=np.hypot((r.link_x-previous.link_x)*10.97,(r.link_y-previous.link_y)*23.77)
                    assert 0<dt<=1 and d<=8*dt+1.5+1e-10
                previous=r
        mapping=pd.read_csv(out/f'identity_map_p{point:02d}.csv')
        accepted_map=mapping[mapping.player_slot.notna()]
        assert accepted_map.groupby(['rally_no','track_id']).player_slot.nunique().eq(1).all()
        assert not accepted_map.duplicated(['source_frame','player_slot']).any()
        assert len(fixed)==len(accepted_map)
        assert not positions.duplicated(['source_frame','player_slot']).any()
        assert positions.groupby('source_frame').size().eq(4).all()
        info=summary['points'][str(point)]
        for slot in b.SLOTS:
            group=positions[positions.player_slot==slot].set_index('source_frame')
            assert dict(Counter(group.status))=={k:v for k,v in info['status_counts_by_player'][slot].items() if v}
            good=group.status.isin(b.VALID)
            assert np.isfinite(group.loc[good,['court_x','court_y']]).all().all()
            assert group.loc[~good,['court_x','court_y']].isna().all().all()
            for r in group[group.status=='interpolated'].itertuples():
                a,z=int(r.interpolation_start_frame),int(r.interpolation_end_frame)
                assert a<r.Index<z and z-a<=30
                assert group.loc[a,'filter_status'] in b.OBS and group.loc[z,'filter_status'] in b.OBS
                alpha=(r.Index-a)/(z-a)
                expected=group.loc[a,['court_x','court_y']].astype(float).to_numpy()*(1-alpha)+group.loc[z,['court_x','court_y']].astype(float).to_numpy()*alpha
                assert np.allclose([r.court_x,r.court_y],expected,rtol=1e-12,atol=1e-12)
            source=fixed[fixed.player_slot==slot]
            if slot.startswith('far_'):
                assert not group.filter_status.eq('observed_after_height_reset').any()
                assert group.height_reset_trigger_frame.isna().all()
                accepted=[]
                for r in group[group.filter_status!='missing'].itertuples():
                    rally=source[source.rally_no==r.rally_no]
                    window=rally[rally.source_frame.between(r.Index-180,r.Index+180)]
                    selected=window if len(window)>=30 else rally
                    reference=float(np.percentile((selected.y2-selected.y1).to_numpy(),80,method='linear'))
                    assert np.isclose(r.reference_height_px,reference,rtol=1e-12)
                    assert np.isclose(r.head_reference_height_px,reference,rtol=1e-12)
                    assert r.height_reference_window_observations==len(window)
                    assert r.height_reference_scope==('window' if len(window)>=30 else 'rally')
                    head=r.box_height_px<.88*reference
                    assert bool(r.head_anchor_used)==head
                    xy=project(H,(r.x1+r.x2)/2,r.y1+reference)
                    assert np.allclose(xy,[r.head_court_x,r.head_court_y],rtol=1e-12)
                    effective=xy if head else (r.court_proxy_x,r.court_proxy_y)
                    assert np.allclose(effective,[r.effective_court_x,r.effective_court_y],rtol=1e-12)
                    speed_bad=bool(accepted) and b.distance(effective,accepted[-1][1])>8*(r.Index-accepted[-1][0])/b.FPS+1
                    expected='rejected_box_height' if r.box_height_px>1.35*reference else 'rejected_speed' if speed_bad else 'observed_head_anchored' if head else 'observed'
                    assert r.filter_status==expected,(point,slot,r.Index,r.filter_status,expected)
                    if expected in b.OBS:accepted.append((r.Index,effective))
        frame_map=pd.read_csv(out/f'video_source_frame_map_p{point:02d}.csv')
        expected=[]
        for rally,g in labels[labels.point==point].groupby('rally_no',sort=True):
            first,last=int(g.frame.min())-60,int(g.frame.max())+90
            times=list(range(first,last+1,2));checks={7:[23422,23548],8:[25470],11:[32999]}.get(point,[])
            times=sorted(set(times+[f for f in checks if first<=f<=last]))
            expected.extend(times)
            assert balls[(balls.point==point)&(balls.rally_no==rally)].source_frame.tolist()==list(range(first,last+1))
        assert frame_map.source_frame.tolist()==expected
        video=out/f'point_{point:02d}.mp4'
        subprocess.run([b.FFMPEG,'-v','error','-xerror','-i',str(video),'-map','0:v:0','-f','null','-'],check=True,capture_output=True)
        probe=json.loads(subprocess.check_output([b.FFPROBE,'-v','error','-count_frames','-select_streams','v:0','-show_entries','stream=nb_read_frames','-of','json',str(video)]))
        assert int(probe['streams'][0]['nb_read_frames'])==len(expected)==info['rendering']['frames']
        audit['points'][str(point)]=dict(source_frames_verified=len(expected),position_rows_verified=len(positions),identity_observations_verified=len(accepted_map),head_geometry_verified=True,full_video_decode_verified=True)
    for name in summary['requested_checks']:assert (out/name).exists()
    b.dump(out/'verification.json',audit)
    inventory=b.abc.inventory()
    summary['verification']='verification.json'
    summary['large_files']=[r for r in inventory if r['bytes']>256*1024]
    b.dump(out/'cleaning_summary.json',summary)
    b.write_readme(summary,summary['hitter_decisions'])
    pd.DataFrame(inventory).to_csv(out/'file_inventory.csv',index=False,encoding='utf-8-sig',lineterminator='\n')
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
