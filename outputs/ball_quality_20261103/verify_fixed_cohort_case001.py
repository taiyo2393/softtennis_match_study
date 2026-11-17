"""Verify source-box provenance, feature arithmetic and render fixed QA images."""
import csv,hashlib,json,math
from collections import defaultdict
from pathlib import Path
import cv2
import numpy as np
from verify_player_prefix_expansion import calculate_frame,calculate_summary,close

ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'ball_quality_20261110'
OUT=BASE/'cohort_tracking/case_001'
SLOTS=['near_front','near_rear','far_rear','far_front']
COLORS=[(80,255,80),(255,190,40),(255,80,240),(0,230,255)]

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    result=load(OUT/'case_result.json');protocol=load(OUT/'protocol.json')
    assert result['status']=='new_player_candidate_path_and_prefix_constructed'
    assert all(sha(Path(p))==h for p,h in protocol['protected_input_hashes'].items())
    assert all(sha(OUT/n)==h for n,h in result['artifact_hashes'].items())
    raw=read(OUT/'crop_runs/augmented_detections_001.csv');rawmap=defaultdict(list)
    for r in raw:rawmap[int(r['source_frame']),r['track_id']].append(r)
    selected=read(OUT/'four_player_path.csv');people=defaultdict(dict)
    for r in selected:
        f=int(r['source_frame']);slot=r['player_slot'];assert slot not in people[f]
        assert r['position_source']=='detected_box' and 'interpolat' not in r['position_source'].lower()
        assert any(all(r[k]==candidate[k] for k in ['x1','y1','x2','y2','bottom_center_x','bottom_center_y','confidence']) for candidate in rawmap[f,r['track_id']])
        people[f][slot]=r
        assert all(math.isfinite(float(r[k])) for k in ['x1','y1','x2','y2','bottom_center_x','bottom_center_y'])
        assert abs(float(r['timestamp_sec'])-f/(60000/1001))<1e-7
    for f,ps in people.items():assert len({r['track_id'] for r in ps.values()})==len(ps)
    actual=read(OUT/'prefix_frame_features.csv');assert len(actual)==31
    expected=[calculate_frame('match1_rally_window_001',people,117+d,d) for d in range(0,61,2)]
    checked=0
    for a,b in zip(actual,expected):
        assert a['rally_candidate_id']==b['rally_candidate_id']
        for key,val in b.items():
            if key!='rally_candidate_id':close(a[key],val);checked+=1
    snapshot,summary=calculate_summary(expected)
    for name,calculated in [('snapshot_player_features.csv',snapshot),('prefix_player_summary_features.csv',summary)]:
        row=read(OUT/name)[0];assert set(row)==set(calculated)|{'rally_candidate_id'}
        for k,v in calculated.items():close(row[k],v);checked+=1
    assert result['input_times_with_four_observations']==sum(len(people[f])==4 for f in range(117,178,2))==31
    assert result['full_context_four_observation_frames']==sum(len(people[f])==4 for f in range(600))==542
    # Same numerical algorithms/settings as retained source implementations; no selection of best version.
    assert protocol['algorithm_parameters_changed'] is False and protocol['interpolation'] is False
    cap=cv2.VideoCapture(protocol['source_video']);images={};views=[7,87,117,147,177,237,357,477,599]
    cells=[];far_cells=[]
    for f in views:
        cap.set(1,f);ok,im=cap.read();assert ok and int(round(cap.get(1)))==f+1
        for i,slot in enumerate(SLOTS):
            r=people[f].get(slot)
            if r is None:continue
            x1,y1,x2,y2=[round(float(r[k])) for k in ['x1','y1','x2','y2']]
            cv2.rectangle(im,(x1,y1),(x2,y2),COLORS[i],5)
            cv2.putText(im,f'{["N1","N2","F1","F2"][i]} raw{r["track_id"]}',(x1,max(35,y1-12)),0,1.3,COLORS[i],3,cv2.LINE_AA)
        preview=cv2.resize(im,(640,360),interpolation=cv2.INTER_AREA);header=np.zeros((32,640,3),dtype=np.uint8)
        cv2.putText(header,f'001 F{f} | {len(people[f])}/4 candidates',(10,24),0,.65,(255,255,255),1,cv2.LINE_AA)
        cells.append(np.vstack([header,preview]))
        # Preserve aspect ratio for distant pair detail (2200x660 =>1000x300).
        far_image=cv2.resize(im[800:1460,850:3050],(1000,300),interpolation=cv2.INTER_AREA)
        far_header=np.zeros((32,1000,3),dtype=np.uint8)
        cv2.putText(far_header,f'001 far-pair context F{f}; candidates, not independent IDs',(10,24),0,.65,(255,255,255),1,cv2.LINE_AA)
        far_cells.append(np.vstack([far_header,far_image]))
    cap.release()
    assert cv2.imwrite(str(OUT/'candidate_sequence_grid.jpg'),np.vstack([np.hstack(cells[i:i+3]) for i in range(0,9,3)]))
    assert cv2.imwrite(str(OUT/'far_pair_detail_grid.jpg'),np.vstack(far_cells))
    videos=[]
    for p in [OUT/'tracking_quality_20261102_stage4/players_001_temporal/tracking_review_h264.mp4',OUT/'tracking_quality_20261102_stage7/players_001_global/global_review_h264.mp4']:
        cap=cv2.VideoCapture(str(p));count=0
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape[:2]==(1080,1920);count+=1
        cap.release();assert count==600;videos.append(dict(path=str(p),frames=count,sha256=sha(p)))
    report=dict(status='passed',selected_boxes_matched_to_augmented_detection_source=len(selected),
        recalculated_feature_values=checked,requested_times=31,four_observation_times=31,
        video_frames_decoded=sum(v['frames'] for v in videos),videos=videos,
        interpolation_found=False,duplicate_slot_or_raw_detection_assignments_found=False,
        protected_sources_unchanged=True,independent_identity_or_position_accuracy_established=False)
    (OUT/'verification.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
