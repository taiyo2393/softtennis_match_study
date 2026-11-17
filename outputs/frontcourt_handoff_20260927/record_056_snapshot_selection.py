import csv
from pathlib import Path
p=Path(__file__).resolve().parent
for name in ['snapshot_selection_reviews.csv','human_review_log.csv','serve_phase_position_snapshot_candidates.csv']:
 with (p/name).open(encoding='utf-8-sig',newline='') as f:
  rd=csv.DictReader(f);fields=list(rd.fieldnames);rows=list(rd)
 if name=='snapshot_selection_reviews.csv':
  if not any(r['candidate_id']=='match1_rally_window_056' and r['source_frame']=='154116' for r in rows):rows.append(dict(candidate_id='match1_rally_window_056',source_frame='154116',selected_track_ids='1;2;4;42',review_status='human_confirmed',reviewer='human_user_2026-10-03',scope='four distinct match players at this frame; personal identities, temporal continuity and ground points not confirmed'))
 elif name=='human_review_log.csv':
  if not any(r['review_id']=='20261003_056_F154116_four_player_selection' for r in rows):rows.append(dict(review_id='20261003_056_F154116_four_player_selection',date='2026-10-03',candidate_id='match1_rally_window_056',field='four_player_selection_F154116',value='confirmed',reviewer='human_user',evidence='deferred_seed_F154116.jpg; user response 出来てる',prior_value='ai_proposed',scope='single-frame player selection only; not temporal identity or ground-point accuracy'))
 else:
  if 'player_selection_review_status' not in fields:
   fields.append('player_selection_review_status')
   for r in rows:r['player_selection_review_status']='unreviewed'
  for r in rows:
   if r['candidate_id']=='match1_rally_window_056':r['player_selection_review_status']='human_confirmed_at_snapshot'
 with (p/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print('056 F154116 single-frame selection recorded')
