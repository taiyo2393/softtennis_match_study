import numpy as np
import tracklet_pair_tracking as g


def row(frame,tid,x):
    return dict(source_frame=frame,track_id=tid,x1=x-5,x2=x+5,y1=1040,y2=1200,
                bottom_center_x=x,bottom_center_y=1200,confidence=.9)


def main():
    raw=[row(0,5,1500),row(1,5,1501),row(2,5,1502),row(3,5,2600),row(10,5,2601)]
    index,_=g.tracklet_index(raw)
    assert index[(0,5)]==index[(2,5)]
    assert index[(2,5)]!=index[(3,5)]
    assert index[(3,5)]!=index[(10,5)]
    cache=[]
    for f in range(20):
        items=[]
        for i,x in enumerate((1500,1540)):
            tid=(5,2)[i]
            appearance=[0.,0.] if f==0 else ([.65,0.] if i==0 else [0.,.65])
            items.append(dict(row=row(f,tid,x),point=np.array([float(x),1200.]),height=160,
                              appearance=appearance,tracklet_key=f'{tid}:0',stable_tracklet=True))
        cache.append(items)
    path,_=g.solve(cache,dict(seed_ids=dict(far_rear=5,far_front=2)))
    ownership={}
    for items,choices in zip(cache,path):
        for person,j in enumerate(choices):
            if j<0:continue
            key=items[j]['tracklet_key']
            assert ownership.setdefault(key,person)==person
    assert ownership['5:0']==0 and ownership['2:0']==1
    print('Passed: smooth tracklet retained; position jumps and long detector gaps split; future appearance preference cannot exchange a continuous tracklet between identities.')


if __name__=='__main__':main()
