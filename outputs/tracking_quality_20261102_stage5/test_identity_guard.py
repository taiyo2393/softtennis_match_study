import sys
from pathlib import Path
import numpy as np
from identity_guard import IdentityGuardTracker,identity_separation,old

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'tracking_quality_20261102_stage4'))
from test_temporal_tracker import scene


def ambiguous_scene(frame,changed=False):
    rows,features = scene(frame)
    # Two far players have identical appearance, but distinct spatial tracks.
    features[3] = features[2].copy()
    if changed:
        rows[2] = dict(rows[2],track_id=105)
    return rows,features


def main():
    tracker = IdentityGuardTracker()
    for f in range(8):
        tracker.step(f,*ambiguous_scene(f))
    # Stable continuation remains possible despite same uniforms.
    output = tracker.step(8,*ambiguous_scene(8))
    assert any(i == 2 for i,_,_,_,_ in output)
    oldframe = tracker.states['far_rear']['frame']
    # Crucial regression: global assignment sees one missing player, but the
    # OTHER, currently assigned player must still participate in identity check.
    for f in range(9,19):
        output = tracker.step(f,*ambiguous_scene(f,changed=True))
        assert not any(i == 2 for i,_,_,_,_ in output)
    assert tracker.states['far_rear']['frame'] == oldframe
    assert tracker.guard_events
    assert all(e['identity_status'] == 'unresolved_not_assigned' for e in tracker.guard_events)
    # Freeze the appearance anchors even if the working gallery changes.
    original = tracker.identity_anchors['far_rear'][0].copy()
    tracker.states['far_rear']['gallery'].append(np.ones_like(original)/original.shape[-1])
    assert np.array_equal(tracker.identity_anchors['far_rear'][0],original)
    # Distinct clothes provide an identity difference, not just position rank.
    clean = IdentityGuardTracker()
    for f in range(8):
        clean.step(f,*scene(f))
    rows,features = scene(8)
    rows[2]['track_id'] = 105
    output = clean.step(8,rows,features)
    assert any(i == 2 for i,_,_,_,_ in output)
    print('Passed: partner comparison includes tracked partner; ambiguous handover withheld; state rollback; immutable anchors; stable IDs and distinguishable handovers retained.')


if __name__ == '__main__':
    main()
