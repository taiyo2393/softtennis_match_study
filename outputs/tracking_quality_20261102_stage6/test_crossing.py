"""Same-uniform crossing regression with detector-ID changes and occlusion."""
import numpy as np
import global_pair_tracking as g


def main():
    cache=[]
    for f in range(31):
        items=[]
        for i,x in enumerate((1500+20*f,2100-20*f)):
            if f==15 and i==1:continue
            tid=(5 if x<1800 else 2) if f else (5,2)[i]
            r=dict(source_frame=f,track_id=tid,x1=x-15,x2=x+15,y1=1040,y2=1200,
                   bottom_center_x=x,bottom_center_y=1200,confidence=.9)
            items.append(dict(row=r,point=np.array([x,1200]),height=160,appearance=[.2,.2]))
        cache.append(items)
    path,_=g.solve(cache,dict(seed_ids=dict(far_rear=5,far_front=2)))
    assert len(path)==31
    final=[cache[-1][j]['point'][0] for j in path[-1]]
    assert final[0]>final[1],final
    for choices,items in zip(path,cache):
        present=[j for j in choices if j>=0]
        assert len(set(present))==len(present)
        assert all(j<len(items) for j in present)
    assert -1 in path[15], 'Cannot emit two different players from one occluded box'
    print('Passed crossing: physical paths preserved despite source ID changes; one occluded person remains missing; no box is assigned twice.')


if __name__=='__main__':main()
