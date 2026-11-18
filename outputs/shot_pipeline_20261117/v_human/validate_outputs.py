"""Independent data/denominator checks and final pre-existing-file audit."""
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from common import HERE, PIPE, ROOT, labels, sha, dump, ignore_large


def main():
    human=labels()
    shots=pd.read_csv(HERE/'shot_table/shot_table.csv',keep_default_na=False)
    assert len(shots)==32 and shots.contact_frame.tolist()==human.frame.tolist()
    assert shots.hitter_side.tolist()==human.side.tolist()
    assert shots.shot_outcome.tolist()==human.outcome.tolist()
    assert shots.is_serve.astype(str).str.lower().tolist()==human.is_serve.astype(str).str.lower().tolist()
    assert shots.hitter_slot.eq('').all() and shots.confidence.eq('').all()
    conversion_rows=0
    slot_names={'N1':'near_front','N2':'near_rear','F1':'far_rear','F2':'far_front'}
    for point,g in shots.groupby('point'):
        proxies=pd.read_csv(PIPE/f'tracking/point_{point:02d}/player_court_proxies.csv')
        for r in g.itertuples():
            for slot,name in slot_names.items():
                a=proxies[(proxies.player_slot==name)&((proxies.source_frame-r.contact_frame).abs()<=3)].copy()
                a=a[np.isfinite(a.court_proxy_x)&np.isfinite(a.court_proxy_y)]
                if a.empty:
                    assert getattr(r,slot+'_x_m')=='' and getattr(r,slot+'_y_m')==''
                else:
                    a['distance']=(a.source_frame-r.contact_frame).abs()
                    p=a.sort_values(['distance','source_frame']).iloc[0]
                    np.testing.assert_allclose([float(getattr(r,slot+'_x_m')),float(getattr(r,slot+'_y_m'))],
                        [p.court_proxy_x*10.97,p.court_proxy_y*23.77],rtol=1e-9,atol=1e-8)
                    assert int(getattr(r,slot+'_frame_offset'))==p.source_frame-r.contact_frame
                    conversion_rows+=1
    for _,g in shots.groupby(['point','rally_no']):
        gaps=np.diff(g.contact_frame.to_numpy(int))
        np.testing.assert_array_equal(pd.to_numeric(g.previous_shot_interval_frames.iloc[1:]),gaps)
        assert g.previous_shot_interval_frames.iloc[0]=='' and g.next_shot_interval_frames.iloc[-1]==''
    audio=HERE/'audio_eval'
    candidates=pd.read_csv(audio/'candidates.csv')
    evaluation=pd.read_csv(audio/'evaluation.csv',dtype={'point':str})
    ints=pd.read_csv(audio/'intervals.csv')
    null=pd.read_csv(audio/'chance_distribution.csv',dtype={'point':str})
    assert candidates.candidate_id.is_unique
    for row in evaluation.itertuples():
        selected=ints[ints.scope==row.scope]
        if row.point!='all':
            selected=selected[selected.point==int(row.point)]
        kept=candidates if row.variant=='v1' else candidates[candidates.kept_v2]
        count,matched,labels_count=0,0,0
        for it in selected.itertuples():
            c=kept[(kept.point==it.point)&(kept.candidate_frame>=it.start_frame)&(kept.candidate_frame<it.end_frame_exclusive)]
            h=human[(human.point==it.point)&(human.frame>=it.start_frame)&(human.frame<it.end_frame_exclusive)]
            count+=len(c);labels_count+=len(h)
            # Independent maximum-cardinality ordered matching on a line.
            f=sorted(h.frame);cf=sorted(c.candidate_frame);i=j=0
            while i<len(f) and j<len(cf):
                if cf[j]<f[i]-row.tolerance_frames:j+=1
                elif cf[j]>f[i]+row.tolerance_frames:i+=1
                else:matched+=1;i+=1;j+=1
        assert (count,matched,labels_count)==(row.n_candidates,row.n_matched,row.n_labels)
        z=null[(null.scope==row.scope)&(null.variant==row.variant)&(null.point==row.point)&
               (null.tolerance_frames==row.tolerance_frames)]
        assert len(z)==1000 and z.replicate.nunique()==1000
        np.testing.assert_allclose(row.empirical_p_one_sided,(1+(z.n_matched>=matched).sum())/1001,atol=1e-14)
        np.testing.assert_allclose(row.chance_mean,z.recall.mean(),atol=1e-14)
    motion=pd.read_csv(HERE/'net_motion/motions.csv')
    assert len(motion)==576 and len(motion[['id']].drop_duplicates())==32
    assert (motion.end_frame<=motion.contact_frame+np.rint(motion.W_sec*motion.fps)).all()
    next_mask=motion.next_contact_frame.notna()
    assert (motion.loc[next_mask,'end_frame']<=motion.loc[next_mask,'next_contact_frame']).all()
    summary=pd.read_csv(HERE/'net_motion/summary.csv')
    assert motion.groupby(['coordinate_method','W_sec','D_m']).size().eq(32).all()
    for r in summary.itertuples():
        g=motion[(motion.coordinate_method==r.coordinate_method)&(motion.W_sec==r.W_sec)&(motion.D_m==r.D_m)]
        if r.scope.startswith('exclude_p8_near'):
            g=g[~g.exclude_p8_near]
        if r.scope=='exclude_p8_near_and_swapped':
            g=g[~g.swapped_either]
        if r.serves=='exclude':
            g=g[~g.is_serve]
        valid=g[g.classification!='unclassifiable']
        assert len(g)==r.total_shots and len(valid)==r.classifiable
        assert len(g)-len(valid)==r.unclassifiable
        if r.metric=='presence':
            assert r.denominator==len(valid)
            n=(valid.motion_present==(1 if r.category=='motion' else 0)).sum()
            assert n==r.n
        elif r.metric=='complexity':
            m=valid[valid.motion_present==1]
            assert r.denominator==len(m) and (m.classification==r.category).sum()==r.n
    # Independently verify nonserve role selection and every resolvable serve.
    roles=pd.read_csv(HERE/'net_motion/role_selection.csv',keep_default_na=False)
    for point,g in roles.groupby('point'):
        track=pd.read_csv(PIPE/f'tracking/point_{point:02d}/player_court_proxies.csv')
        for r in g.itertuples():
            if r.front_slot=='':continue
            frame=int(float(r.next_contact_frame)) if r.is_serve else r.contact_frame
            p=track[(track.source_frame==frame)&track.player_slot.isin(
                ['near_front','near_rear'] if r.front_side=='N' else ['far_front','far_rear'])].copy()
            p['distance']=(p.court_proxy_y*23.77-11.885).abs()
            if r.is_serve:
                receiver=p.sort_values(['distance','player_slot'],ascending=[False,True]).iloc[0].player_slot
                assert r.receiver_slot==receiver and r.front_slot!=receiver
            else:
                assert r.front_slot==p.sort_values(['distance','player_slot']).iloc[0].player_slot
    # Verify cache fallback follows frame ranges rather than old rally names.
    traj=pd.read_csv(HERE/'net_motion/trajectories_main.csv')
    caches=pd.read_csv(HERE/'net_motion/pose_cache_ranges.csv')
    for r in traj[traj.coordinate_method=='ankle'].itertuples():
        point=int(r.id[1:3]); c=caches[caches.point==point]
        inside=((c.start_frame<=r.source_frame)&(r.source_frame<=c.end_frame_inclusive)).any()
        if isinstance(r.foot_source,str) and r.foot_source.startswith('bbox_cache_'):
            assert not inside
    manifest=json.loads((HERE/'preservation_manifest.json').read_text(encoding='utf-8'))
    for name,digest in manifest['old_files_sha256'].items():
        assert sha(ROOT/name)==digest,name
    old_names={p.relative_to(ROOT).as_posix() for p in PIPE.rglob('*') if p.is_file() and not p.is_relative_to(HERE)}
    expected={n for n in manifest['old_files_sha256'] if n.startswith(PIPE.relative_to(ROOT).as_posix()+'/')}
    assert old_names==expected
    ignored=ignore_large()
    for path in HERE.rglob('*'):
        if path.is_file() and path.stat().st_size>256*1024:
            assert path.relative_to(HERE).as_posix() in ignored
    dump(HERE/'validation.json',dict(passed=True,shot_rows=32,coordinate_values_checked=conversion_rows,
        audio_evaluation_rows=len(evaluation),monte_carlo_rows=len(null),motion_rows=len(motion),
        summary_rows=len(summary),old_files_unchanged=len(manifest['old_files_sha256']),
        large_files_ignored=ignored,independent_checks=['coordinate conversion and nearest frame',
        'ordered maximum one-to-one audio matching','empirical p values from all replicates',
        'motion denominators and end caps','receiver and net role selection','pose fallback frame ranges']))
    print('PASS: 32 human labels, independent matching and p values, 576 motion rows, all old files unchanged.')


if __name__=='__main__':
    main()
