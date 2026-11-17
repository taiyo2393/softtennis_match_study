import csv,json,hashlib,math,statistics
from pathlib import Path
import cv2,numpy as np
p=Path(__file__).resolve().parent;source=Path(r'C:\Users\ta239\Downloads\manual_position_reference (3).json');blob=source.read_bytes();doc=json.loads(blob);annotations=doc['annotations'];assert doc['schema']=='manual_image_footpoint_reference_v1'
expected={('001',117),('007',15593),('056',154116)};assert len(annotations)==12;assert len({(a['candidate_id'],a['image_slot']) for a in annotations})==12
(p/'manual_position_reference_received_20261102.json').write_bytes(blob)
def read(n):
 with (p/n).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
auto={}
for r in read('additional_four_player_snapshot_proposals.csv'):
 if r['candidate_id']=='001':auto[('001',r['image_slot'])]=(int(r['sample_frame']),float(r['bottom_x_px']),float(r['bottom_y_px']))
for r in read('six_service_selected_position_candidates.csv'):
 if r['candidate_id']=='007':auto[('007',r['image_slot'])]=(int(r['source_frame']),float(r['bottom_x_px']),float(r['bottom_y_px']))
slots={'near_front':'near_right','near_rear':'near_left','far_front':'far_left','far_rear':'far_right'}
for r in read('serve_phase_position_snapshot_candidates.csv'):
 if r['candidate_id'].endswith('_056'):auto[('056',slots[r['player_slot']])]=(int(r['sample_source_frame']),float(r['bottom_center_x_px']),float(r['bottom_center_y_px']))
root=Path(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study');H=np.array(json.loads((root/'config/match1_court_plane_calibration.json').read_text(encoding='utf-8'))['image_to_court_homography'],np.float32)
rows=[]
for a in annotations:
 n=a['candidate_id'];slot=a['image_slot'];assert(n,a['source_frame']) in expected;frame,x,y=auto[(n,slot)];assert frame==a['source_frame'];r=dict(candidate_id=n,source_frame=frame,image_slot=slot,manual_status=a['status'],auto_x_px=x,auto_y_px=y,manual_x_px='',manual_y_px='',dx_auto_minus_manual_px='',dy_auto_minus_manual_px='',image_distance_px='',court_x_delta_candidate='',court_y_delta_candidate='',scope='manual_image_reference_disagreement_not_physical_ground_truth')
 if a['status']=='annotated':
  mx,my=float(a['x_px']),float(a['y_px']);assert 0<=mx<=3840 and 0<=my<=2160;dx,dy=x-mx,y-my;q=cv2.perspectiveTransform(np.float32([[[x,y],[mx,my]]]),H)[0];r.update(manual_x_px=mx,manual_y_px=my,dx_auto_minus_manual_px=dx,dy_auto_minus_manual_px=dy,image_distance_px=math.hypot(dx,dy),court_x_delta_candidate=float(q[0,0]-q[1,0]),court_y_delta_candidate=float(q[0,1]-q[1,1]))
 rows.append(r)
with (p/'manual_auto_position_comparison.csv').open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
summary=[]
for n in ['all','001','007','056']:
 subset=[r for r in rows if n=='all' or r['candidate_id']==n];valid=[r for r in subset if r['image_distance_px']!=''];d=[r['image_distance_px'] for r in valid]
 summary.append(dict(candidate_id=n,requested=len(subset),evaluated=len(valid),unjudgeable=len(subset)-len(valid),median_image_distance_px=statistics.median(d) if d else None,mean_image_distance_px=statistics.mean(d) if d else None,max_image_distance_px=max(d) if d else None,mean_dx_px=statistics.mean([r['dx_auto_minus_manual_px'] for r in valid]) if d else None,mean_dy_px=statistics.mean([r['dy_auto_minus_manual_px'] for r in valid]) if d else None))
(p/'manual_auto_position_summary.json').write_text(json.dumps(dict(source_file=str(source),sha256=hashlib.sha256(blob).hexdigest(),results=summary,limitations=['3 convenience/systematic pilot frames; not tracking accuracy','Human previously exposed to candidate imagery; not strict blinded independent validation','Foot midpoint and box bottom are different definitions','Homography transfer error not evaluated; no physical accuracy claim']),indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2));print('Largest disagreements:',json.dumps(sorted([r for r in rows if r['image_distance_px']!=''],key=lambda r:r['image_distance_px'],reverse=True)[:3],ensure_ascii=False))
