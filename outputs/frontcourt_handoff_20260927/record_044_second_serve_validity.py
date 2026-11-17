"""Apply explicit user confirmation of second-serve validity only."""
import csv
from pathlib import Path
OUT=Path(__file__).resolve().parent

def read(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as f:
        r=csv.DictReader(f)
        return list(r.fieldnames),list(r)
def write(name,fields,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
fields,rows=read('044_serve_attempt_review.csv')
row=next(r for r in rows if r['attempt_number']=='2')
row['outcome_review_status']='human_confirmed'
write('044_serve_attempt_review.csv',fields,rows)
fields,rows=read('rally_label_review.csv')
row=next(r for r in rows if r['candidate_id']=='match1_rally_window_044')
row['serve_review_status']='second_serve_valid_human_confirmed_contact_frame_pending'
note='Human user 2026-10-03 confirmed effective second serve in 044_second_serve_review_h264.mp4; contact frame remains an AI-proposed interval.'
if note not in row['review_note']:
    row['review_note']+=' | '+note
write('rally_label_review.csv',fields,rows)
fields,rows=read('human_review_log.csv')
rid='20261003_044_second_serve_valid'
if not any(r['review_id']==rid for r in rows):
    rows.append(dict(review_id=rid,date='2026-10-03',candidate_id='match1_rally_window_044',
                     field='second_serve_outcome',value='valid',reviewer='human_user',
                     evidence='044_second_serve_review_h264.mp4; user response 有効',
                     prior_value='ai_visual_supported',scope='service validity only; exact contact frame not confirmed'))
write('human_review_log.csv',fields,rows)
print('044 second serve valid: human confirmed')
