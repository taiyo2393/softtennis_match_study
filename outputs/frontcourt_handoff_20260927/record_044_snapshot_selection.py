"""Record human confirmation of four distinct player boxes at one frame."""
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

name='snapshot_selection_reviews.csv'
if (OUT/name).exists():
    fields,rows=read(name)
else:
    fields=['candidate_id','source_frame','selected_track_ids','review_status','reviewer','scope']
    rows=[]
if not any(r['candidate_id']=='match1_rally_window_044' and r['source_frame']=='115403' for r in rows):
    rows.append(dict(candidate_id='match1_rally_window_044',source_frame='115403',
                     selected_track_ids='1;2;20;422',review_status='human_confirmed',
                     reviewer='human_user_2026-10-03',
                     scope='four distinct match players at this frame; personal identities, side-role assignment, temporal continuity and ground points not confirmed'))
write(name,fields,rows)

fields,rows=read('human_review_log.csv')
rid='20261003_044_F115403_four_player_selection'
if not any(r['review_id']==rid for r in rows):
    rows.append(dict(review_id=rid,date='2026-10-03',candidate_id='match1_rally_window_044',
                     field='four_player_selection_F115403',value='confirmed',reviewer='human_user',
                     evidence='044_second_snapshot_F115403.jpg; user response ok to four distinct player selection question',
                     prior_value='ai_proposed',scope='single-frame player selection only'))
write('human_review_log.csv',fields,rows)
print('044 F115403 four-player selection: human confirmed')
