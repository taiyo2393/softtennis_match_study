"""Check compact-map exactness using representable AMP values and reject FP32 loss."""
import json
from pathlib import Path
import numpy as np
from integrate_044_reviewed_tracks import ROOT,sha
from calibrate_point_supervised_detector import compact,expand,components_at,THRESHOLDS
from compare_ball_candidate_decoders import components

def main():
    rng=np.random.default_rng(20261103);checks=0
    for _ in range(3):
        hm=rng.uniform(0,.25,(288,512)).astype(np.float32)
        hm[30:60,40:100]=rng.uniform(.25,1,(30,60));hm[150:165,300:320]=.75;hm[155:160,305:309]=.1
        # Real inference outputs are FP16 under AMP, including values inside holes.
        hm=hm.astype(np.float16).astype(np.float32)
        idx,v=compact(hm);recon=expand(idx,v);assert components(hm)==components_at(hm,.3)
        for t in THRESHOLDS:assert components_at(hm,t)==components_at(recon,t);checks+=1
    hm=np.full((288,512),np.float16(.25),np.float32);idx,v=compact(hm);assert len(idx)==0
    assert all(not components_at(expand(idx,v),t) for t in THRESHOLDS)
    bad=np.zeros((288,512),np.float32);bad[50:60,50:60]=np.float16(.8);bad[55,55]=.1
    try:compact(bad)
    except AssertionError:pass
    else:raise AssertionError('Non-AMP precision loss was not rejected')
    out=ROOT/'ball_point_detector_calibration';out.mkdir(exist_ok=True)
    report=dict(status='passed',exact_threshold_map_checks=checks,empty_maps_checked=True,non_amp_loss_rejected=True,
        fixture_note='Embedded first self-test used unrounded FP32 0.1 inside a retained hole, so the intended loss guard rejected it. This fixture rounds all values to actual AMP precision; production inference was unchanged.',
        test_script_sha256=sha(Path(__file__)))
    (out/'compact_map_test_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
