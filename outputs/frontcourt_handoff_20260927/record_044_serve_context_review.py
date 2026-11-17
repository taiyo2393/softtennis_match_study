"""Record user-confirmed clip context; keep effective serve frames pending."""
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

fields,rows=read('rally_label_review.csv')
for key in ('first_serve_outcome','clip_context_review_status'):
    if key not in fields:
        fields.append(key)
        for r in rows:
            r[key]=''
r=next(r for r in rows if r['candidate_id']=='match1_rally_window_044')
r['first_serve_outcome']='fault_human_confirmed'
r['clip_context_review_status']='human_confirmed_first_serve_start_through_point_end'
r['serve_review_status']='effective_serve_contact_pending_after_first_fault'
note='Human user 2026-10-03: clip covers serve contact through rally end; first serve faulted. Exact first/second contact and end frames not supplied. Current F114790 sampling anchor requires rechecking against effective second serve.'
if note not in r['review_note']:
    r['review_note']+=' | '+note
write('rally_label_review.csv',fields,rows)

fields,rows=read('human_review_log.csv')
for field,value in (
    ('first_serve_outcome','fault'),
    ('clip_context','serve_contact_through_point_end'),
    ('clip_start','first_serve_start'),
):
    rid='20261003_044_'+field
    if not any(r['review_id']==rid for r in rows):
        rows.append(dict(review_id=rid,date='2026-10-03',
                         candidate_id='match1_rally_window_044',field=field,value=value,
                         reviewer='human_user',evidence='user explicit description of current 044 clip',
                         prior_value='unreviewed',
                         scope='qualitative clip/event context; exact source frames remain pending'))
write('human_review_log.csv',fields,rows)
print('044: first serve fault and clip context recorded; effective serve anchor pending')
