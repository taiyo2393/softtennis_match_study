import csv
from pathlib import Path
p=Path(__file__).resolve().parent
for name in ['independent_footpoint_annotation_template.csv','human_review_log.csv','serve_phase_position_snapshot_candidates.csv','footpoint_measurement_risks.csv']:
 with (p/name).open(encoding='utf-8-sig',newline='') as f:
  rd=csv.DictReader(f);fields=list(rd.fieldnames);rows=list(rd)
 if name=='independent_footpoint_annotation_template.csv':
  r=next(r for r in rows if r['candidate_id']=='056' and r['source_frame']=='154116' and r['player_slot']=='near_front')
  r.update(support_status='both_feet_airborne',ground_plane_position_status='unavailable_from_visible_foot_contact_airborne',reviewer='human_user_2026-11-02',review_status='support_state_human_confirmed_coordinates_unreviewed',evidence='056_support_state_quarter_speed_h264.mp4; user response 浮いてる')
 elif name=='human_review_log.csv':
  if not any(r['review_id']=='20261102_056_F154116_airborne' for r in rows):rows.append(dict(review_id='20261102_056_F154116_airborne',date='2026-11-02',candidate_id='match1_rally_window_056',field='near_front_support_state_F154116',value='both_feet_airborne',reviewer='human_user',evidence='056_support_state_quarter_speed_h264.mp4; explicit question about F154116; user response 浮いてる',prior_value='ai_visual_provisional',scope='support state at one frame only; coordinates and ground-plane projection unconfirmed'))
 elif name=='serve_phase_position_snapshot_candidates.csv':
  for key in ['support_state_review_status','ground_plane_observation_status']:
   if key not in fields:
    fields.append(key)
    for r in rows:r[key]='unreviewed'
  for r in rows:
   if r['candidate_id']=='match1_rally_window_056' and r['sample_source_frame']=='154116' and r['player_slot']=='near_front':
    r['support_state_review_status']='human_confirmed_both_feet_airborne';r['ground_plane_observation_status']='no_direct_ground_contact_airborne';r['position_review_status']='bbox_candidate_only_not_ground_truth_airborne'
 else:
  for r in rows:
   if r['candidate_id']=='056' and r['frame']=='154116' and r['slot']=='near_front':r['finding']='Both feet airborne confirmed by human; box bottom is not observed ground-contact position';r['status']='human_confirmed_support_state_only'
 with (p/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print('Support state recorded in four review files; no coordinates filled or snapshot time changed')
