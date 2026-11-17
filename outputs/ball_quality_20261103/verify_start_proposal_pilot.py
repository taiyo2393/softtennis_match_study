"""Check missing-pose handling, gap grouping and data provenance."""
import json
import numpy as np
from extract_serve_start_features import ROOT,SOURCE,sha
from propose_serve_starts import describe,propose


def main():
    kp=np.zeros((17,3));kp[5]=[10,40,.9];kp[6]=[30,40,.9]
    p=describe(dict(box=[0,0,100,200],keypoints=kp.tolist(),confidence=.9),0)
    assert p is not None and p['raised'] is False  # Missing wrists at(0,0) cannot be "raised".
    entries=[dict(local_frame=t,foot=[50.,200.],height=200.,raised=(10<=t<=20 or 30<=t<=40),box=[0,0,100,200]) for t in range(0,60,2)]
    times=np.arange(0,1.5,.01);rise=np.zeros(len(times));rise[65]=10
    candidates,episodes=propose({0:entries},dict(time_sec=times,rise_db=rise),59.94005994)
    assert len(candidates)==1 and len(episodes)==1
    assert candidates[0]['raised_first_local_frame']==10 and candidates[0]['raised_last_local_frame']==40
    assert candidates[0]['serve_outcome']=='unknown'
    out=ROOT/'serve_start_pilot_tiled_v2';frozen=json.loads((out/'frozen_detector.json').read_text())
    assert sha(ROOT/'propose_serve_starts.py')==frozen['proposal_code_sha256']
    for pilot in ('016','002'):
        dest=out/f'match1_motion_pilot_{pilot}';r=json.loads((dest/'proposals.json').read_text())
        assert r['labels_read'] is False and r['code_sha256']==frozen['proposal_code_sha256']
        assert json.loads((dest/'proposal_protocol.json').read_text())==frozen['proposal_parameters']
        assert all(p['serve_outcome']=='unknown' for p in r['proposals'])
    prior=json.loads((ROOT/'play_start_audit/inventory_summary.json').read_text())
    for path,expected in prior['source_hashes'].items():
        from pathlib import Path
        assert sha(Path(path))==expected
    report=dict(missing_wrists_not_raised=True,short_pose_occlusion_merged=True,outcome_not_inferred=True,
        same_frozen_parameters_and_code_on_both_clips=True,source_review_tables_unchanged=True,
        labels_not_read_by_feature_or_proposal_stage=True,
        limitation='These are consistency and provenance checks, not start detection performance.')
    (out/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
