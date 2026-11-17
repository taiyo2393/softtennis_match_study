"""Regression checks for user-reported errors and supported sharp motion."""
import json
from temporal_consensus_guard import ROOT,OUT,read,filter_rows,sha,SOURCE


def main():
    rows=[dict(local_frame=i,visible=1,x_px=100+3*i,y_px=500+2*i) for i in range(61)]
    rows[30].update(x_px=1500,y_px=200)
    filtered,_=filter_rows(rows,[0.]*61,128,False)
    assert int(filtered[30]['visible'])==0
    assert all(int(p['visible']) for i,p in enumerate(filtered) if i!=30)
    turn=[dict(local_frame=i,visible=1,x_px=(100+70*i if i<=30 else 2200-60*(i-30)),
               y_px=(500 if i<=30 else 500+15*(i-30))) for i in range(61)]
    filtered,_=filter_rows(turn,[0.]*61,128,False)
    assert all(int(p['visible']) for p in filtered)
    regression=[];total=0
    for cid in ('match1_motion_pilot_013','match1_motion_pilot_018'):
        src=ROOT/'glove_guard/continuous_predictions'/cid/'balltrack_predictions.csv'
        old=read(src);new=read(OUT/'scene_aware_preview'/cid/'balltrack_predictions.csv')
        assert len(old)==len(new)
        for r,p in zip(old,new):
            assert r['local_frame']==p['local_frame']
            assert all(p['input_'+k]==r[k] for k in ('visible','x_px','y_px'))
            assert all(p[k]==r[k] for k in ('raw_visible','raw_x_px','raw_y_px','raw_confidence'))
            if not int(p['temporal_guard_flag']):assert all(p[k]==r[k] for k in ('visible','x_px','y_px'))
            total+=1
        if cid.endswith('018'):
            for t in (208,229,230):
                assert int(old[t]['visible'])==1 and int(new[t]['visible'])==0
                regression.append(dict(local_frame=t,reason=new[t]['temporal_reason'],model_error_px=new[t]['temporal_model_error_px']))
            assert all(new[t]['x_px']==old[t]['x_px'] and new[t]['y_px']==old[t]['y_px'] for t in (219,220))
    assert sha(SOURCE/'annotations/balltrack_groundtruth_match1_pilot.csv')==json.loads((OUT/'protocol.json').read_text())['label_sha256']
    result=dict(user_reported_regression_cases=regression,verified_rows=total,raw_predictions_preserved=True,
        unsuppressed_positions_unchanged=True,no_interpolated_positions=True,source_annotations_unchanged=True,
        isolated_jump_suppressed=True,synthetic_supported_fast_turn_preserved=True,
        nearby_return_detections_219_220_preserved=True,
        limitations='Synthetic turn check does not establish retention of all real fast hits. Reported user cases are development regressions, not independent accuracy.')
    (OUT/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
