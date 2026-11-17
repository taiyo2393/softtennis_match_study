from temporal_recovery import recover


def main():
    rows=[dict(local_frame=i,source_frame=100+i,x_px=x,y_px=100,visible=v,confidence=.6,threshold=.30)
          for i,x,v in [(0,100,1),(1,0,0),(2,0,0),(3,130,1)]]
    config=dict(max_gap=4,min_peak=.15,radius_px=16.)
    cs=[dict(local_frame=i,x_px=100+10*i,y_px=100,peak=.2) for i in (1,2)]
    got=recover(rows,cs,config)
    assert [int(r['visible']) for r in got]==[1,1,1,1]
    assert got[1]['x_px']==110 and got[2]['x_px']==120
    assert got[1]['position_source']=='weak_heatmap_temporally_supported'
    for i in (0,3):
        assert all(got[i][k]==rows[i][k] for k in rows[i])
    assert [int(r['visible']) for r in recover(rows,[],config)]==[1,0,0,1]
    assert [int(r['visible']) for r in recover(rows,cs[:1],config)]==[1,0,0,1]
    far=[dict(c,x_px=1000) for c in cs]
    assert [int(r['visible']) for r in recover(rows,far,config)]==[1,0,0,1]
    weak=[dict(c,peak=.12) for c in cs]
    assert [int(r['visible']) for r in recover(rows,weak,config)]==[1,0,0,1]
    assert not recover(rows[1:],cs,config)[0]['visible']
    print('Passed: baseline detections preserved; recovered coordinates equal model components; no synthetic positions without heatmaps; full-gap support, proximity, confidence and two anchors required.')


if __name__=='__main__':main()
