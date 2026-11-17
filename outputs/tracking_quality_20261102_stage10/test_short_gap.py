import gap_pair_tracking as g


def row(f,x):return dict(source_frame=f,track_id=1,x1=x-30,x2=x+30,y1=1040,y2=1200,bottom_center_x=x,bottom_center_y=1200)


def main():
    rows=[row(f,1500+2*f) for f in [0,1,2,3,4,14,15,16,17,18]]
    result=g.short_gap_continuity(rows,5)
    assert result and result['missing_frames']==9
    index,_=g.tracklet_index(rows)
    assert index[(4,1)]==index[(14,1)]
    reversed_motion=[row(f,1500+2*f) for f in [0,1,2,3,4]]+[row(f,1700-20*(f-14)) for f in [14,15,16,17,18]]
    assert g.short_gap_continuity(reversed_motion,5) is None
    longgap=[row(f,1500+2*f) for f in [0,1,2,3,4,50,51,52,53,54]]
    assert g.short_gap_continuity(longgap,5) is None
    jumped=[row(f,1500) for f in range(5)]+[row(f,2800) for f in range(14,19)]
    index,_=g.tracklet_index(jumped)
    assert index[(4,1)]!=index[(14,1)]
    print('Passed: two-sided smooth short gap retained; inconsistent motion, jumps and long gaps split.')


if __name__=='__main__':main()
