"""Failure-sensitive software checks, not ball accuracy evaluation."""
import json
from collections import defaultdict
from pathlib import Path
from propose_044_ball_branches import trace,OUT,P
from integrate_044_reviewed_tracks import ROOT,load,read
from extract_serve_start_features import sha

def row(x,y=100):return dict(x_px=str(x),y_px=str(y),visible='1',peak='.5',rank='1')

def main():
    # Ambiguous two-ball candidates are left unresolved, not chosen by rank.
    raw=[row(100),row(100),row(100)]
    by={0:[row(100)],1:[row(90),row(110)],2:[row(100)]}
    selected,contexts,connected=trace(by,raw,0,2)
    assert 1 not in selected and contexts[1]['decision']=='ambiguous' and connected
    # Missing heatmaps remain missing; a gap beyond the fixed limit cannot connect.
    raw=[row(100) for _ in range(15)]
    selected,_,connected=trace({0:[row(100)],14:[row(100)]},raw,0,14)
    assert selected.keys()=={0} and connected is False
    # Curved motion is accepted without creating intermediate coordinates.
    raw=[row(100,100),row(120,130),row(140,100)]
    selected,_,connected=trace({i:[r] for i,r in enumerate(raw)},raw,0,2)
    assert connected and all(selected[i] is raw[i] for i in range(3))
    manifest=load(ROOT/'reviewed_ball_overrides.json')['clips']['match1_motion_pilot_016']
    report=load(OUT/'report.json');rows=read(OUT/'branch_proposals.csv')
    assert sha(Path(manifest['reviewed_output']))==manifest['reviewed_output_sha256']==report['reviewed_input_sha256']
    assert set(map(int,report['changes']))=={891,892,893,960,968,970}
    assert all(r['branch_proposal_action']=='unchanged_reviewed_input' for r in rows if int(r['local_frame']) in [773,784,1103])
    for window in report['windows']:
        assert window['forward_connected'] and window['backward_connected']
    # Raw968 fails distance constraints in both directions, not a low-score test.
    window=report['windows'][1]
    for side in ['forward_context','backward_context']:
        c=window[side]['968'];assert c['raw_distance_px']>c['distance_gate_px'] and c['decision']!='ambiguous'
    (OUT/'failure_sensitive_checks.json').write_text(json.dumps(dict(ambiguous_candidates_not_forced=True,
        long_gaps_do_not_connect=True,no_invented_intermediate_coordinates=True,
        previous_user_confirmed_points_preserved=True,raw968_exclusion_is_two_sided_distance_supported=True,
        independent_accuracy_evaluation=False),indent=2),encoding='utf-8')
    print('Passed: ambiguity, long gap, observed curved path, reviewed input integrity and two-sided968 check.')

if __name__=='__main__':main()
