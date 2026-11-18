"""Audit saved outputs without rerendering or changing existing inputs."""
from collections import Counter
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

import build_court_view as b


def main():
    out = b.OUT
    summary = json.loads((out/'cleaning_summary.json').read_text(encoding='utf-8'))
    protected = json.loads((out/'input_sha256_before.json').read_text(encoding='utf-8'))
    assert all(b.sha(Path(p)) == digest for p,digest in protected.items())
    labels = pd.read_csv(b.PIPE/'labels_human/human_labels_20261118.csv')
    labels = labels[(labels.condition=='without_candidates') & (labels.status=='completed')]
    hits = pd.read_csv(out/'hitter_auto.csv')
    assert len(hits)==32
    assert set(zip(hits.point,hits.frame))==set(zip(labels.point,labels.frame))
    wrist = pd.read_csv(b.PIPE/'pose/wrist_timeseries.csv')
    extra = pd.read_csv(out/'pose_supplement/wrist_timeseries_additional.csv')
    assert not set(zip(wrist.point,wrist.source_frame,wrist.player_slot)) & set(zip(extra.point,extra.source_frame,extra.player_slot))
    wrist = pd.concat([wrist,extra],ignore_index=True)
    retracked=pd.read_csv(out/'pose_retrack_supplement/wrist_timeseries_v2.csv')
    wrist=b.retrack.replace_wrist(wrist,retracked)
    # Recompute maxima directly from the saved feature tables, independently
    # of the classifier helper. Include one-sided missingness in the audit.
    for h in hits.itertuples():
        maxima=[]
        for index in (1,2):
            slot=getattr(h,f'player_{index}')
            values=wrist.loc[(wrist.point==h.point) & (wrist.player_slot==slot) & wrist.source_frame.between(h.frame-8,h.frame+4),'wrist_speed']
            values=values[np.isfinite(values)&(values>=0)]
            peak=values.max() if len(values) else np.nan
            stored=getattr(h,f'player_{index}_max')
            assert (np.isnan(peak) and np.isnan(stored)) or np.isclose(peak,stored,rtol=1e-12)
            maxima.append(peak)
        valid=np.isfinite(maxima).all()
        expected=valid and max(maxima)>min(maxima) and max(maxima)>=1.3*min(maxima)
        assert (h.status=='auto_wrist')==expected
        if expected:
            assert h.hitter_slot==getattr(h,f'player_{1+int(maxima[1]>maxima[0])}')
        else:
            assert pd.isna(h.hitter_slot) and h.hitter==h.side+'?'
    audit=dict(input_hashes_verified_unchanged=len(protected),hitter_decisions_verified=32,
               visual_checks='Saved impact snapshots in verification_contact_sheet.jpg; interval coverage audited separately in retrack_interval_check.json.',points={})
    for point in range(7,12):
        positions=pd.read_csv(out/f'cleaned_player_positions_p{point:02d}.csv')
        info=summary['points'][str(point)]
        assert not positions.duplicated(['source_frame','player_slot']).any()
        assert positions.groupby('source_frame').size().eq(4).all()
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
        mapping=pd.read_csv(out/f'video_source_frame_map_p{point:02d}.csv')
        expected=[]
        for _,g in labels[labels.point==point].groupby('rally_no',sort=True):
            expected.extend(range(int(g.frame.min())-60,int(g.frame.max())+91,2))
        assert mapping.source_frame.tolist()==expected
        assert len(mapping)==info['rendering']['frames']
        audit['points'][str(point)]=dict(source_frame_map_verified=True,position_rows_verified=len(positions),
                                       interpolation_endpoints_verified=True,encoded_frames=info['rendering']['frames'],
                                       full_video_decode_verified=info['rendering']['full_decode_verified'])
    audit['large_csv_files']=[dict(file=str(p.relative_to(out)).replace('\\','/'),bytes=p.stat().st_size)
                              for p in sorted(out.rglob('*.csv')) if p.stat().st_size>256*1024]
    b.dump(out/'verification.json',audit)
    # Refresh the inventory after the visual audit, which adds a contact sheet.
    def inventory():
        rows=[]
        for path in sorted(out.rglob('*')):
            if not path.is_file() or path.name=='file_inventory.csv':continue
            relative=str(path.relative_to(b.ROOT)).replace('\\','/')
            ignored=subprocess.run(['git','check-ignore','-q','--',relative],cwd=b.ROOT).returncode==0
            r=dict(file=str(path.relative_to(out)).replace('\\','/'),bytes=path.stat().st_size,ignored=ignored)
            if path.suffix in ('.mp4','.jpg','.png') or r['bytes']>256*1024:
                assert ignored, relative
            rows.append(r)
        return rows
    summary['verification']='verification.json'
    summary['large_files']=[r for r in inventory() if r['bytes']>256*1024]
    b.dump(out/'cleaning_summary.json',summary)
    b.write_readme(summary,summary['hitter_decisions'])
    pd.DataFrame(inventory()).to_csv(out/'file_inventory.csv',index=False,encoding='utf-8-sig')
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
