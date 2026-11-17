"""Behavior tests for delayed initialization, long gaps and distractors."""
import copy
import numpy as np
from temporal_tracker import TemporalTracker, recovery_cost, a


def scene(frame,missing=False,new_id=105,wrong_color=False):
    rows = []
    features = []
    for tid,x,y,h,w,color in [(1,1800,1500,300,100,0),(2,2800,1800,500,200,1),
                               (5,1500,1230,160,60,2),(4,2300,1280,180,70,3)]:
        if tid == 5 and missing:
            continue
        if tid == 5 and frame > 100:
            tid,x = new_id,x+100
            color = 4 if wrong_color else color
        rows.append(dict(source_frame=frame,track_id=tid,x1=x-w/2,x2=x+w/2,y1=y-h,y2=y,
                         bottom_center_x=x,bottom_center_y=y,confidence=.95,in_initial_roi=1))
        features.append(np.stack([np.eye(5,dtype=np.float32)[color]]*2))
    return rows,features


def test_delayed_initialization():
    t = TemporalTracker()
    for f in range(10):
        assert t.step(f,*scene(f,missing=True)) == []
    for f in range(10,17):
        assert t.step(f,*scene(f)) == []
    assert len(t.step(17,*scene(17))) == 4
    assert t.initialized_frame == 17
    assert t.events[0]['window_start'] == 10


def test_long_reacquisition():
    t = TemporalTracker()
    for f in range(8):
        t.step(f,*scene(f))
    for f in range(8,120):
        assert all(i != 2 for i,_,_,_,_ in t.step(f,*scene(f,missing=True)))
    # A changed detector ID inside a spatially coherent sequence must not
    # erase confirmation, and the gap itself must never be backfilled.
    for f in range(120,127):
        assert all(i != 2 for i,_,_,_,_ in t.step(f,*scene(f,new_id=105 if f<124 else 106)))
    output = t.step(127,*scene(127,new_id=106))
    recovered = [x for x in output if x[0] == 2]
    assert len(recovered) == 1 and recovered[0][-1] == 'gallery_reacquisition_candidate'
    assert t.events[-1]['confirmation_frames'] == 8
    assert t.events[-1]['confirmation_raw_ids'] == [105,106]
    assert t.events[-1]['last_observed_frame'] == 7


def test_distractor_rejected():
    t = TemporalTracker()
    for f in range(8):
        t.step(f,*scene(f))
    for f in range(8,120):
        t.step(f,*scene(f,missing=True))
    for f in range(120,140):
        assert all(i != 2 for i,_,_,_,_ in t.step(f,*scene(f,wrong_color=True)))
    assert not any(e['event'] == 'reacquired_candidate' for e in t.events)
    state = t.states['far_rear']
    row,feature = scene(140)[0][2],scene(140)[1][2]
    # Wrong court side and head-only fragments remain rejected.
    assert recovery_cost(state,dict(row,bottom_center_y=1800,y2=1800,y1=1640),feature,140) > 1e5
    assert recovery_cost(state,dict(row,y1=float(row['y2'])-40),feature,140) > 1e5


if __name__ == '__main__':
    test_delayed_initialization()
    test_long_reacquisition()
    test_distractor_rejected()
    print('Passed: delayed initialization; 8-frame coherent confirmation after >90-frame gap; changing raw IDs; no gap backfill; wrong appearance/side/fragment rejection.')
