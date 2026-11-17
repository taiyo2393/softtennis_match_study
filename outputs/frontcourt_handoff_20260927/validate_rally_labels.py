"""Fail if a candidate is marked ready without reviewable rally and result labels."""
import csv
from pathlib import Path

path=Path(__file__).resolve().parent/'rally_label_review.csv'
with path.open(encoding='utf-8-sig',newline='') as f:
    rows=list(csv.DictReader(f))
assert len(rows)==63
assert len({r['candidate_id'] for r in rows})==len(rows)
required=('rally_id','start_source_frame','end_source_frame','winner_team_id',
          'winner_camera_side','winner_evidence_source_frame','reviewer')
ready=[]
for row in rows:
    assert row['analysis_eligible'] in ('true','false')
    if row['analysis_eligible']=='true':
        assert all(row[k] for k in required),row['candidate_id']
        assert int(row['start_source_frame'])<int(row['end_source_frame'])
        assert row['winner_review_status']=='human_confirmed',row['candidate_id']
        assert row['boundary_review_status']=='human_confirmed',row['candidate_id']
        assert row['team_side_mapping_status']=='human_confirmed',row['candidate_id']
        ready.append(row['candidate_id'])
print(f'Candidates: {len(rows)}; analysis eligible: {len(ready)}; score-labeled: {sum(bool(r["winner_team_id"]) for r in rows)}')

by_id={r['candidate_id']:r for r in rows}
with (path.parent/'event_review_ranges.csv').open(encoding='utf-8-sig',newline='') as f:
    events=list(csv.DictReader(f))
assert len(events)==4
assert len({(e['candidate_id'],e['event']) for e in events})==len(events)
for event in events:
    row=by_id[event['candidate_id']]
    first=int(event['first_possible_source_frame'])
    last=int(event['last_possible_source_frame'])
    assert first<=last
    assert event['status']=='visual_interval_review_needed'
    assert row['analysis_eligible']=='false'
    assert not row['start_source_frame'] and not row['end_source_frame']
    candidate_start=round(float(row['candidate_start_sec'])*60000/1001)
    candidate_end=round(float(row['candidate_end_sec'])*60000/1001)
    if event['event']=='serve_contact' and first<candidate_start:
        print(f"Boundary conflict: {event['candidate_id']} provisional serve interval begins {candidate_start-first} frames before candidate start")
    if event['event']=='point_end':
        assert first<=candidate_end, event['candidate_id']
print(f'Provisional event intervals: {len(events)}; none are confirmed frame labels')
