"""Check v2 exports against original inputs without rerunning the classifier."""
from pathlib import Path
from fractions import Fraction
import hashlib
import json

import cv2
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BASE = OUT.parent.parent
ROOT = BASE.parent.parent


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    manifest = json.loads((OUT/'input_manifest.json').read_text(encoding='utf-8'))
    assert all(sha(ROOT/p)==h for p,h in manifest['input_sha256'].items())
    assert all(sha(OUT.parent/p)==h for p,h in manifest['v1_preserved_sha256'].items())
    m = pd.read_csv(OUT/'motions.csv',keep_default_na=False)
    assert len(m)==86*3*3*2
    assert not m.duplicated(['id','coordinate_method','W_sec','D_m']).any()
    durations = []
    for _,r in m.iterrows():
        protocol = json.loads((BASE/f'tracking/point_{r.point:02d}/protocol.json').read_text())
        fps = float(Fraction(protocol['source_fps']))
        expected_end = int(r.contact_frame)+round(r.W_sec*fps)
        if str(r.next_contact_frame):
            expected_end = min(expected_end,int(float(r.next_contact_frame)))
        assert r.end_frame==expected_end
        durations.append((r.end_frame-r.contact_frame)/fps)
        if r.classification=='compound':
            segments = json.loads(r.segments_json)
            assert len(segments)>=2
            assert all(s['end_frame']-s['start_frame']>=4 and s['length_m']>=r.D_m-1e-12 for s in segments)
    # Role rules: obtain next hitters directly from original label CSVs.
    slot_names = {'N1':'near_front','N2':'near_rear','F1':'far_rear','F2':'far_front'}
    serves = 0
    receiver_fallbacks = 0
    for point in range(7,12):
        labels = pd.read_csv(BASE/f'labels/point_{point:02d}.csv',keep_default_na=False)
        for rally,group in labels.groupby('rally'):
            group = group.sort_values('shot_index').reset_index(drop=True)
            for i,r in group.iterrows():
                if not str(r.is_serve).lower()=='true' or rally=='A': continue
                selected = m[(m.point==point)&(m.rally==rally)&(m.shot_index==r.shot_index)]
                assert len(selected)==18
                nxt = group.iloc[i+1]
                expected_receiver = slot_names.get(nxt.hitter_slot)
                basis = 'next_label_hitter_slot'
                if expected_receiver is None:
                    tracks = pd.read_csv(BASE/f'tracking/point_{point:02d}/player_court_proxies.csv')
                    side = ['near_front','near_rear'] if r.hitter_side=='F' else ['far_rear','far_front']
                    pair = tracks[(tracks.source_frame==nxt.contact_frame)&tracks.player_slot.isin(side)].copy()
                    assert len(pair)==2
                    baseline = 0 if r.hitter_side=='F' else 1
                    pair['baseline_distance'] = (pair.court_proxy_y-baseline).abs()
                    expected_receiver = pair.sort_values(['baseline_distance','player_slot']).iloc[0].player_slot
                    basis = 'nearest_baseline_at_next_contact'
                    receiver_fallbacks += 1
                assert (selected.receiver_slot==expected_receiver).all()
                assert (selected.front_slot!=expected_receiver).all()
                assert (selected.receiver_basis==basis).all()
                serves += 1
    # Check every exported raw ankle against original keypoints, including missing
    # and box fallbacks. This catches coordinate scale and frame/slot mismatches.
    trajectories = pd.read_csv(OUT/'trajectories_main.csv')
    checked = 0
    for point in range(7,12):
        raw_pose = pd.read_csv(BASE/f'pose/pose_cache/point_{point:02d}_rally_B.csv')
        pose = raw_pose.set_index(['source_frame','player_slot'])
        ids = set(m[(m.point==point)&(m.coordinate_method=='ankle')].id)
        g = trajectories[(trajectories.coordinate_method=='ankle')&trajectories.id.isin(ids)]
        for _,r in g.iterrows():
            key = (r.source_frame,r.front_slot)
            if key not in pose.index:
                assert r.foot_source in ['bbox_cache_outside','tracking_missing'] or pd.isna(r.foot_source)
                continue
            p = pose.loc[key]
            valid = []
            for k in [15,16]:
                xy = p[[f'kp_{k}_x',f'kp_{k}_y']].to_numpy(float)
                if p[f'kp_{k}_conf']>=0.3 and np.isfinite(xy).all(): valid.append(xy)
            expected = np.mean(valid,axis=0) if valid else np.array([np.nan,np.nan])
            actual = r[['raw_image_x','raw_image_y']].to_numpy(float)
            # CSV float serialization is 10 significant digits.
            np.testing.assert_allclose(actual,expected,rtol=1e-9,atol=1e-7,equal_nan=True)
            checked += 1
    reviews = pd.read_csv(OUT/'review_template.csv',keep_default_na=False)
    reference = pd.read_csv(OUT.parent/'review_claude_v1.csv')
    assert len(reviews)==30
    assert reviews.iloc[:20][['review_id','id']].equals(reference[['review_id','id']])
    assert reviews.id.is_unique
    image_shapes = {}
    for _,r in reviews.iterrows():
        img = cv2.imread(str(OUT/r.sheet))
        assert img is not None and img.shape[:2]==(390,320+400*len(r.sampled_frames.split(';')))
        image_shapes[r.review_id] = list(img.shape[:2])
    ignored = set((OUT/'.gitignore').read_text(encoding='utf-8').splitlines())
    for p in OUT.rglob('*'):
        if p.is_file() and p.stat().st_size>256*1024:
            assert '/'+p.relative_to(OUT).as_posix() in ignored,p
    assert np.isclose(sum([31.6,2.7,27.1,9.4,12.1,6.1,1.8,11.1]),101.9)
    results = dict(passed=True,v1_and_input_hashes_unchanged=True,motion_rows=len(m),
                   interval_rules_checked=len(m),serve_roles_checked=serves,
                   serve_receiver_baseline_fallbacks_checked=receiver_fallbacks,
                   raw_ankle_positions_checked=checked,review_images_checked=len(reviews),
                   review_shapes=image_shapes,large_artifacts_ignored=True,
                   note='Export consistency checks; not independent tracking or motion ground truth')
    (OUT/'validation_report.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in results.items() if k!='review_shapes'},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
