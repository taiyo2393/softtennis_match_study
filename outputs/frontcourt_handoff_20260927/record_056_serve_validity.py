import csv
from pathlib import Path
p=Path(__file__).resolve().parent
for name in ['rally_label_review.csv','human_review_log.csv','serve_phase_position_snapshot_candidates.csv']:
 with (p/name).open(encoding='utf-8-sig',newline='') as f:
  reader=csv.DictReader(f);fields=reader.fieldnames;rows=list(reader)
 if name=='rally_label_review.csv':
  r=next(x for x in rows if x['candidate_id']=='match1_rally_window_056');r['serve_review_status']='opening_serve_valid_human_confirmed_contact_frame_pending'
  note='User explicitly confirmed opening serve valid in 056_opening_serve_h264.mp4. Contact interval F154081-F154086 remains AI provisional; first/second attempt number not established.'
  if note not in r['review_note']:r['review_note']+=' | '+note
 elif name=='human_review_log.csv':
  if not any(r['review_id']=='20261003_056_opening_serve_valid' for r in rows):rows.append(dict(review_id='20261003_056_opening_serve_valid',date='2026-10-03',candidate_id='match1_rally_window_056',field='opening_serve_outcome',value='valid',reviewer='human_user',evidence='056_opening_serve_h264.mp4; user response 有効',prior_value='unreviewed',scope='serve validity only; contact frame and serve attempt number not confirmed'))
 else:
  for r in rows:
   if r['candidate_id']=='match1_rally_window_056':r['sampling_validity_status']='serve_valid_human_confirmed_contact_interval_provisional'
 with (p/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print('056 valid serve recorded without confirming contact timing or identities')
