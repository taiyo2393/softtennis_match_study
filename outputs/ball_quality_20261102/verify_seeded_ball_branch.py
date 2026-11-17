"""Association regression checks and source/output integrity."""
import json
import numpy as np

from track_seeded_ball_branch import associate
from extract_multiball_candidates import DEST,ROOT,SOURCE,read,sha


def point(t,x,y,rank=1):return dict(local_frame=t,x_px=x,y_px=y,peak=.5,rank=rank)


def main():
    # A second, denser object's detections must not steal the seeded branch.
    points=[point(t,100+t,200) for t in (0,1,3,4)]+[point(t,900,600) for t in range(5)]
    assigned,_=associate(points,3,(103,200),0,4)
    assert sorted(assigned)==[0,1,3,4]
    assert all(float(c['x_px'])<200 for c in assigned.values())
    assert 2 not in assigned  # No synthetic fill of the missed observation.
    # Equidistant alternatives are kept as ambiguous, not broken by rank.
    assigned,events=associate([point(0,100,100),point(1,95,100),point(1,105,100,2)],0,(100,100),0,1)
    assert 1 not in assigned and any(e['event']=='ambiguous_candidates_not_assigned' for e in events)
    # Reappearance after a long gap is not silently linked to the same object.
    assigned,_=associate([point(0,100,100),point(20,101,100)],0,(100,100),0,20)
    assert sorted(assigned)==[0]
    # Every emitted position must be one of the retained observed components.
    candidates=read(DEST/'candidates.csv');output=read(DEST/'seeded_branch.csv')
    for r in output:
        if int(r['visible']):
            matches=[c for c in candidates if c['local_frame']==r['local_frame'] and c['rank']==r['component_rank']]
            assert len(matches)==1 and all(r[k]==matches[0][k] for k in ('x_px','y_px','peak'))
        else:assert float(r['x_px'])==float(r['y_px'])==0
        assert int(r['interpolated'])==0
    extraction=json.loads((DEST/'extraction_report.json').read_text());protocol=json.loads((DEST/'protocol.json').read_text())
    baseline=ROOT.parent/'tracking_quality_20261102/ball_aligned/match1_motion_pilot_002/balltrack_predictions.csv'
    assert sha(baseline)==protocol['baseline_sha256']
    assert len(extraction['baseline_reproduction_checks'])==81
    assert max(c['baseline_delta_px'] for c in extraction['baseline_reproduction_checks'])<=2
    frozen=json.loads((ROOT/'temporal_guard/protocol.json').read_text())
    assert sha(SOURCE/'annotations/balltrack_groundtruth_match1_pilot.csv')==frozen['label_sha256']
    report=dict(seed_association_survives_competing_object=True,ambiguous_alternatives_not_forced=True,
        gaps_not_interpolated=True,long_gap_not_relinked=True,all_positions_are_observed_components=True,
        original_model_reproduced_on_81_frames=True,original_predictions_and_labels_unchanged=True,
        selected_components=sum(int(r['visible']) for r in output),
        warning='These are consistency checks, not ball identity or localization accuracy measurements.')
    (DEST/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
