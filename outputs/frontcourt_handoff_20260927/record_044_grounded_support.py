import csv
from pathlib import Path
p=Path(__file__).resolve().parent
for name in ['independent_footpoint_annotation_template.csv','human_review_log.csv','044_second_snapshot_candidates.csv','footpoint_measurement_risks.csv']:
 with (p/name).open(encoding='utf-8-sig',newline='') as f:
  rd=csv.DictReader(f);fields=list(rd.fieldnames);rows=list(rd)
 if name=='independent_footpoint_annotation_template.csv':
  r=next(r for r in rows if r['candidate_id']=='044' and r['source_frame']=='115403' and r['player_slot']=='near_rear')
  r.update(support_status='one_foot_grounded',ground_plane_position_status='support_contact_visible_coordinate_annotation_pending',reviewer='human_user_2026-11-02',review_status='support_state_human_confirmed_coordinates_unreviewed',evidence='044_support_state_quarter_speed_h264.mp4; user response 片足が接地')
 elif name=='human_review_log.csv':
  if not any(r['review_id']=='20261102_044_F115403_one_foot_grounded' for r in rows):rows.append(dict(review_id='20261102_044_F115403_one_foot_grounded',date='2026-11-02',candidate_id='match1_rally_window_044',field='near_rear_support_state_F115403',value='one_foot_grounded',reviewer='human_user',evidence='044_support_state_quarter_speed_h264.mp4; user response 片足が接地',prior_value='ai_visual_provisional',scope='single-frame support state only; foot coordinate, anatomical side and calibration not confirmed'))
 elif name=='044_second_snapshot_candidates.csv':
  for key in ['support_state_review_status','ground_plane_observation_status']:
   if key not in fields:
    fields.append(key)
    for r in rows:r[key]='unreviewed'
  r=next(r for r in rows if r['player_slot']=='near_rear');r['support_state_review_status']='human_confirmed_one_foot_grounded';r['ground_plane_observation_status']='support_contact_coordinate_pending'
 else:
  for r in rows:
   if r['candidate_id']=='044' and r['slot']=='near_rear':r['finding']='Human confirmed one foot grounded; supporting-foot coordinate not independently annotated';r['status']='human_confirmed_support_state_only'
 with (p/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print('044 support state recorded; coordinate labels remain blank')
