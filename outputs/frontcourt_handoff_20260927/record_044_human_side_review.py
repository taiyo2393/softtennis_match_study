"""Record the user's 2026-10-03 side label without confirming team mapping."""
import csv
from pathlib import Path

OUT=Path(__file__).resolve().parent

def read(name):
    with (OUT/name).open(encoding='utf-8-sig',newline='') as file:
        reader=csv.DictReader(file)
        return list(reader.fieldnames),list(reader)

def write(name,fields,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

fields,rows=read('rally_label_review.csv')
if 'winner_camera_side_review_status' not in fields:
    fields.append('winner_camera_side_review_status')
    for row in rows:
        row['winner_camera_side_review_status']='unreviewed'
row=next(r for r in rows if r['candidate_id']=='match1_rally_window_044')
row['winner_camera_side']='near'
row['winner_camera_side_review_status']='human_confirmed'
row['team_side_mapping_status']='conflict_with_prior_ai_side_mapping_review_needed'
row['reviewer']='Human user side review 2026-10-03; Codex provisional score review 2026-09-27'
note='Human user 2026-10-03 explicitly labeled winner camera side near after 044_full_point_context_h264.mp4; prior AI far label superseded; score-winning team remains provisional; team-side mapping conflict unresolved.'
if note not in row['review_note']:
    row['review_note'] += ' | '+note
write('rally_label_review.csv',fields,rows)

fields,rows=read('full_point_context_queue.csv')
row=next(r for r in rows if r['candidate_id']=='match1_rally_window_044')
row['winner_camera_side_confirmed']='near'
row['clip_file']='044_full_point_context_h264.mp4'
row['review_status']='winner_side_human_confirmed_other_fields_pending'
write('full_point_context_queue.csv',fields,rows)

log='human_review_log.csv'
if (OUT/log).exists():
    fields,rows=read(log)
else:
    fields=['review_id','date','candidate_id','field','value','reviewer','evidence','prior_value','scope']
    rows=[]
if not any(r['review_id']=='20261003_044_winner_side' for r in rows):
    rows.append(dict(review_id='20261003_044_winner_side',date='2026-10-03',
                     candidate_id='match1_rally_window_044',field='winner_camera_side',
                     value='near',reviewer='human_user',
                     evidence='044_full_point_context_h264.mp4; user response 手前',
                     prior_value='far_ai_provisional',
                     scope='camera side only; team identity and event boundaries not confirmed'))
write(log,fields,rows)
print('044 winner camera side: near, human confirmed; team mapping conflict pending')
