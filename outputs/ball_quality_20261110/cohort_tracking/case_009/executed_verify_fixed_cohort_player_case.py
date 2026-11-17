"""Verify a fixed new cohort case without requiring complete observations."""
import argparse,csv,hashlib,json,math
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from verify_player_prefix_expansion import calculate_frame,calculate_summary,close

ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'ball_quality_20261110'
SLOTS=['near_front','near_rear','far_rear','far_front'];NAMES=['N1','N2','F1','F2']
COLORS=[(80,255,80),(255,190,40),(255,80,240),(0,230,255)]
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);tag=ap.parse_args().tag
    out=BASE/'cohort_tracking'/f'case_{tag}';result=load(out/'case_result.json');p=load(out/'protocol.json')
    assert not (out/'verification.json').exists()
    assert result['status']=='new_player_candidate_path_and_prefix_constructed'
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    assert all(sha(out/n)==h for n,h in result['artifact_hashes'].items())
    reference=load(BASE/'cohort_tracking/case_001/protocol.json')
    assert p['near_parameters']==reference['near_parameters'] and p['far_parameters']==reference['far_parameters']
    assert p['detector']==reference['detector'] and p['far_crop']==reference['far_crop']
    times=p['requested_input_frames'];first=p['context_first_source_frame'];last=p['context_last_source_frame'];cid=p['candidate_id']
    assert times==list(range(p['contact_candidate_last_source_frame']+30,p['contact_candidate_last_source_frame']+91,2))
    rawmap=defaultdict(list)
    for r in read(out/f'crop_runs/augmented_detections_{tag}.csv'):rawmap[int(r['source_frame']),r['track_id']].append(r)
    selected=read(out/'four_player_path.csv');people=defaultdict(dict)
    for r in selected:
        f=int(r['source_frame']);assert first<=f<=last and r['player_slot'] not in people[f]
        assert r['position_source']=='detected_box'
        assert any(all(r[k]==candidate[k] for k in ['x1','y1','x2','y2','bottom_center_x','bottom_center_y','confidence']) for candidate in rawmap[f,r['track_id']])
        people[f][r['player_slot']]=r
        assert all(math.isfinite(float(r[k])) for k in ['x1','y1','x2','y2','bottom_center_x','bottom_center_y'])
        assert abs(float(r['timestamp_sec'])-f/(60000/1001))<1e-7
    for ps in people.values():assert len({r['track_id'] for r in ps.values()})==len(ps)
    expected=[calculate_frame(cid,people,f,f-times[0]) for f in times];actual=read(out/'prefix_frame_features.csv')
    assert len(actual)==len(expected)==31;checked=0
    for a,b in zip(actual,expected):
        assert a['rally_candidate_id']==b['rally_candidate_id']
        for k,v in b.items():
            if k!='rally_candidate_id':close(a[k],v);checked+=1
    snapshot,summary=calculate_summary(expected)
    for name,calculated in [('snapshot_player_features.csv',snapshot),('prefix_player_summary_features.csv',summary)]:
        row=read(out/name)[0];assert set(row)==set(calculated)|{'rally_candidate_id'}
        for k,v in calculated.items():close(row[k],v);checked+=1
    assert result['input_times_with_four_observations']==sum(len(people[f])==4 for f in times)
    assert result['full_context_four_observation_frames']==sum(len(people[f])==4 for f in range(first,last+1))
    assert p['algorithm_parameters_changed'] is False and p['interpolation'] is False
    views=sorted({max(first,times[0]-60),p['contact_candidate_first_source_frame'],p['contact_candidate_last_source_frame'],
        times[0],times[0]+30,times[-1],min(last,times[-1]+15),min(last,times[-1]+30),last})
    cap=cv2.VideoCapture(p['source_video']);cells=[];far_cells=[]
    for f in views:
        cap.set(1,f);ok,im=cap.read();assert ok and int(round(cap.get(1)))==f+1
        for i,s in enumerate(SLOTS):
            r=people[f].get(s)
            if r is None:continue
            x1,y1,x2,y2=[round(float(r[k])) for k in ['x1','y1','x2','y2']]
            cv2.rectangle(im,(x1,y1),(x2,y2),COLORS[i],5)
            cv2.putText(im,f'{NAMES[i]} raw{r["track_id"]}',(x1,max(35,y1-12)),0,1.3,COLORS[i],3,cv2.LINE_AA)
        header=np.zeros((32,640,3),dtype=np.uint8)
        cv2.putText(header,f'{tag} F{f} | {len(people[f])}/4 candidates',(10,24),0,.65,(255,255,255),1,cv2.LINE_AA)
        cells.append(np.vstack([header,cv2.resize(im,(640,360),interpolation=cv2.INTER_AREA)]))
        far_image=cv2.resize(im[800:1460,850:3050],(1000,300),interpolation=cv2.INTER_AREA)
        head=np.zeros((32,1000,3),dtype=np.uint8)
        cv2.putText(head,f'{tag} far-pair context F{f}; candidates, not independent IDs',(10,24),0,.65,(255,255,255),1,cv2.LINE_AA)
        far_cells.append(np.vstack([head,far_image]))
    cap.release()
    while len(cells)%3:cells.append(np.zeros_like(cells[0]))
    assert cv2.imwrite(str(out/'candidate_sequence_grid.jpg'),np.vstack([np.hstack(cells[i:i+3]) for i in range(0,len(cells),3)]))
    assert cv2.imwrite(str(out/'far_pair_detail_grid.jpg'),np.vstack(far_cells))
    videos=[]
    for path in [out/f'tracking_quality_20261102_stage4/players_{tag}_temporal/tracking_review_h264.mp4',out/f'tracking_quality_20261102_stage7/players_{tag}_global/global_review_h264.mp4']:
        cap=cv2.VideoCapture(str(path));count=0
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape[:2]==(1080,1920);count+=1
        cap.release();assert count==last-first+1;videos.append(dict(path=str(path),frames=count,sha256=sha(path)))
    phase=load(out/'phase_evidence/manifest.json');assert phase['input_grid_source_frames']==times
    assert all(sha(out/'phase_evidence'/n)==h for n,h in phase['images'].items())
    # Recreate source JPEGs, including the missing endpoint; never substitute a better frame.
    cap=cv2.VideoCapture(p['source_video']);source_checks=0
    for f in [times[0],times[len(times)//2],times[-1],phase['context_grid_source_frames'][-1]]:
        cap.set(1,f);ok,im=cap.read();assert ok and int(round(cap.get(1)))==f+1
        ok,buf=cv2.imencode('.jpg',cv2.resize(im,(1920,1080),interpolation=cv2.INTER_AREA),[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok
        assert hashlib.sha256(buf.tobytes()).hexdigest()==phase['images'][f'F{f}.jpg'];source_checks+=1
    cap.release()
    missing=[dict(source_frame=f,missing_slots=[s for s in SLOTS if s not in people[f]]) for f in times if len(people[f])!=4]
    report=dict(status='passed',candidate_id=cid,selected_boxes_matched_to_augmented_detection_source=len(selected),
        recalculated_feature_values=checked,requested_times=31,four_observation_times=result['input_times_with_four_observations'],
        missing_requested_times=missing,source_JPEG_recreations=source_checks,video_frames_decoded=sum(v['frames'] for v in videos),videos=videos,
        numerical_parameters_equal_retained001=True,QA_grid_source_frames=views,
        interpolation_found=False,duplicate_slot_or_raw_detection_assignments_found=False,
        protected_sources_unchanged=True,independent_identity_or_position_accuracy_established=False)
    (out/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
