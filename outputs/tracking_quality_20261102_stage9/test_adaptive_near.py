import sys
from pathlib import Path
from adaptive_near import AdaptiveNearTracker,reference,old

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'tracking_quality_20261102_stage4'))
from test_temporal_tracker import scene


def pose(frame,height):
    rows,features=scene(frame)
    rows[0]=dict(rows[0],y1=float(rows[0]['y2'])-height)
    return rows,features


def main():
    t=AdaptiveNearTracker()
    for f in range(8):t.step(f,*pose(f,600))
    assert t.states
    for f in range(8,78):
        h=max(240,600-6*(f-7))
        output=t.step(f,*pose(f,h))
        assert any(i==0 for i,_,_,_,_ in output),(f,h)
    # Tall serve initialization must not permanently reject normal body height.
    assert t.states['near_front']['frame']==77
    assert t.states['near_front']['reference_height']<600
    before=list(t.heights['near_front'])
    output=t.step(78,*pose(78,70))
    assert not any(i==0 for i,_,_,_,_ in output)
    assert list(t.heights['near_front'])==before,'Rejected fragment contaminated reference'
    assert reference([70]*30)>=345.6
    print('Passed: tall-serve to ordinary-pose continuity; rolling reference; head fragment rejected; rejected observations do not adapt reference; absolute floor retained.')


if __name__=='__main__':main()
