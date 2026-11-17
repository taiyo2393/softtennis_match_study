import csv
from pathlib import Path
p=Path(__file__).resolve().parent
for name in ['rally_label_review.csv','human_review_log.csv']:
 with (p/name).open(encoding='utf-8-sig',newline='') as f:
  rd=csv.DictReader(f); fields=list(rd.fieldnames);rows=list(rd)
 if name=='rally_label_review.csv':
  for key in ['point_end_event_type','point_end_event_review_status']:
   if key not in fields:
    fields.append(key)
    for r in rows:r[key]=''
  r=next(x for x in rows if x['candidate_id']=='match1_rally_window_044')
  r['point_end_event_type']='out';r['point_end_event_review_status']='human_confirmed_event_type_frame_pending'
  note='User confirmed point-ending event out in 044_point_end_event_review_h264.mp4; exact out landing frame, responsible hitter, line and landing coordinates not confirmed.'
  if note not in r['review_note']:r['review_note']+=' | '+note
 elif not any(r['review_id']=='20261003_044_point_end_out' for r in rows):
  rows.append(dict(review_id='20261003_044_point_end_out',date='2026-10-03',candidate_id='match1_rally_window_044',field='point_end_event_type',value='out',reviewer='human_user',evidence='044_point_end_event_review_h264.mp4; user response アウト',prior_value='unreviewed',scope='event type only; exact frame, hitter and landing position not confirmed'))
 with (p/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print('044 point-ending event out recorded; exact frame remains pending')
