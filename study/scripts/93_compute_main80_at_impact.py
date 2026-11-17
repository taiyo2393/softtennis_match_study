"""Add t=0 agreement without post-impact frames; leave original analyses intact."""
from pathlib import Path
import csv,json,hashlib,importlib.util,statistics
from collections import defaultdict
import numpy as np

BASE=Path(__file__).resolve().parents[1]
OUT=BASE/'analysis/rear_hitter_main80/impact_timepoint'
def read(p):
 with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(name,rows):
 with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
spec=importlib.util.spec_from_file_location('court_features',BASE/'scripts/86_build_court_plane_preimpact_features.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
meta_path=BASE/'analysis/match1_court_plane_preimpact_motion_features.metadata.json'
meta=json.loads(meta_path.read_text(encoding='utf8'))
old_path=BASE/'analysis/rear_hitter_main80/main80_event_timepoints.csv'
old=read(old_path);events={r['event_id']:r for r in old}
assert len(old)==320 and len(events)==80
cal_path=Path(meta['calibration']);cal=json.loads(cal_path.read_text(encoding='utf8'))
H=np.asarray(cal['image_to_court_homography'],dtype=np.float32)
inputs=[old_path,meta_path,cal_path,*map(Path,meta['poses'])]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
poses=defaultdict(dict)
for p in meta['poses']:
 for r in read(p):
  if r['event_id'] not in events:continue
  xy=mod.transform_ankle_pair(r,H)
  if xy is not None:
   frame=int(r['source_frame']);event=poses[r['event_id']]
   if frame in event:assert event[frame]['court_x']==xy[0]
   event[frame]={'court_x':xy[0]}
def direction(dx,end):
 if dx is None:return 'missing'
 dx=float(f'{dx:.6f}') # match legacy CSV precision before thresholding
 if abs(dx)<.015:return 'stay'
 positive='front_left' if end=='far' else 'front_right'
 return positive if dx>=.015 else ('front_right' if positive=='front_left' else 'front_left')
reproduced=0
for r in old:
 point=int(r['impact_source_frame'])+round(float(r['timepoint_sec'])*mod.FPS)
 a,_=mod.median_coordinate(poses[r['event_id']],point-6,1,'court_x')
 b,_=mod.median_coordinate(poses[r['event_id']],point,1,'court_x')
 assert a is not None and b is not None
 assert f'{b-a:.6f}'==r['lateral_displacement_court_width'],r['event_id']
 assert direction(b-a,r['target_front_camera_end'])==r['movement_direction_target_front_view']
 reproduced+=1
def frameset(center,mode,impact):
 if mode=='centered_clipped_at_impact':return [f for f in range(center-1,center+2) if f<=impact]
 if mode=='trailing_three':return list(range(center-2,center+1))
 if mode=='raw_endpoint':return [center]
 raise ValueError(mode)
results=[]
for mode in ['centered_clipped_at_impact','trailing_three','raw_endpoint']:
 for eid,r in sorted(events.items()):
  impact=int(r['impact_source_frame']);sample=[];used=[]
  for center in [impact-6,impact]:
   fs=[f for f in frameset(center,mode,impact) if f in poses[eid]]
   assert all(f<=impact for f in fs)
   vals=[poses[eid][f]['court_x'] for f in fs]
   sample.append(statistics.median(vals) if vals else None);used.append(fs)
  dx=sample[1]-sample[0] if all(v is not None for v in sample) else None
  movement=direction(dx,r['target_front_camera_end']);actual=r['actual_direction_target_front_view']
  directional=movement in ['front_left','front_right']
  results.append(dict(mode=mode,event_id=eid,target_front_player_id=r['target_front_player_id'],hitter_id=r['hitter_id'],target_front_camera_end=r['target_front_camera_end'],impact_source_frame=impact,timepoint_sec=0,motion_start_source_frame=impact-6,motion_end_source_frame=impact,start_sample_frames=';'.join(map(str,used[0])),end_sample_frames=';'.join(map(str,used[1])),start_samples=len(used[0]),end_samples=len(used[1]),lateral_displacement_court_width='' if dx is None else f'{dx:.6f}',actual_direction_target_front_view=actual,movement_direction_target_front_view=movement,movement_is_directional=int(directional),direction_matches_actual=int(movement==actual) if directional else '',missing_pose=int(dx is None)))
summary=[]
for mode in dict.fromkeys(r['mode'] for r in results):
 rows=[r for r in results if r['mode']==mode]
 for front,hitter in [(None,None),('P1','H2'),('P2','H1')]:
  rs=rows if front is None else [r for r in rows if r['target_front_player_id']==front and r['hitter_id']==hitter]
  n=sum(r['movement_is_directional'] for r in rs);m=sum(r['direction_matches_actual']==1 for r in rs)
  summary.append(dict(mode=mode,group='全体' if front is None else f'前衛{front} 対 後衛{hitter}',timepoint_sec=0,total_events=len(rs),directional_events=n,matches=m,disagreements=n-m,below_threshold=sum(r['movement_direction_target_front_view']=='stay' for r in rs),missing_pose=sum(r['missing_pose'] for r in rs),agreement_percent=100*m/n if n else None,classification_percent=100*n/len(rs)))
OUT.mkdir(parents=True,exist_ok=True)
write('event_timepoints.csv',results);write('summary.csv',summary)
primary=[r for r in results if r['mode']=='centered_clipped_at_impact']
audit={'reproduced_existing_rows':reproduced,'events':len(events),'fps':mod.FPS,'window_frames':6,'window_seconds':6/mod.FPS,'threshold_court_width':.015,'primary_mode':'centered_clipped_at_impact','smoothing_note':'Legacy centered radius-1 median retained, but samples after impact removed. Endpoint uses impact-1 and impact when available. Sensitivities use trailing three frames or raw endpoints.','start_sample_counts':{str(n):sum(r['start_samples']==n for r in primary) for n in range(4)},'end_sample_counts':{str(n):sum(r['end_samples']==n for r in primary) for n in range(4)},'source_sha256':hashes,'slides_modified':False}
for p in inputs:assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[str(p)]
(OUT/'metadata.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({'audit':{k:v for k,v in audit.items() if k!='source_sha256'},'summary':summary},ensure_ascii=False,indent=2))
