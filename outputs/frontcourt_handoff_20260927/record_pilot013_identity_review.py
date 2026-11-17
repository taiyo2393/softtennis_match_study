import csv
from pathlib import Path
p=Path(__file__).resolve().parent
for name in ['identity_review.csv','human_review_log.csv']:
 with (p/name).open(encoding='utf-8-sig',newline='') as f:
  rd=csv.DictReader(f);fields=rd.fieldnames;rows=list(rd)
 if name=='identity_review.csv':
  r=next(x for x in rows if x['pilot']=='pilot_013');r['review_status']='human_confirmed_junction_identity';r['reviewer']='human_user 2026-10-03; prior Codex visual review 2026-09-27'
  evidence='pilot_013_identity_review_half_speed_h264.mp4; user response 同じ'
  if evidence not in r['evidence']:r['evidence']+=';'+evidence
  note='Human confirmed ID10 and ID22 same player at junction; does not validate ground points, one interpolated frame, all new-ID frames or full-rally continuity.'
  if note not in r['notes']:r['notes']+=' | '+note
 elif not any(x['review_id']=='20261003_pilot013_junction_identity' for x in rows):
  rows.append(dict(review_id='20261003_pilot013_junction_identity',date='2026-10-03',candidate_id='match1_rally_window_056',field='pilot013_ID10_to_ID22_identity',value='same_player',reviewer='human_user',evidence='pilot_013_identity_review_half_speed_h264.mp4; user response 同じ',prior_value='ai_visual_supported_human_pending',scope='junction identity only; interpolation and position errors not validated'))
 with (p/name).open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
print('Both pilot junction identities now human confirmed')
