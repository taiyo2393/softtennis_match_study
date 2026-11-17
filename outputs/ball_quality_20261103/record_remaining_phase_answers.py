"""Record only the explicit human chat answers for three target moments."""
import json,datetime,shutil
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,read,load,write,sha

OUT=ROOT.parent/'ball_quality_20261109/remaining_error_phase_review'
assert not (OUT/'completed_review.json').exists(),'Preserve completed review; amendments need a new event.'
m=load(OUT/'manifest.json');path=OUT/'phase_review_labels.csv'
assert sha(path)==m['initial_review_labels_sha256']
assert all(sha(Path(p))==h for p,h in m['protected_input_hashes'].items())
initial=OUT/'initial_phase_review_labels.csv';shutil.copy2(path,initial)
rows=read(path);assert len(rows)==3 and all(r['phase_status']=='unreviewed' for r in rows)
now=datetime.datetime.now(datetime.timezone.utc).isoformat();events=[]
answers={('match1_motion_pilot_001',464):dict(question_item=0,answer='終了後'),
         ('match1_motion_pilot_006',279):dict(question_item=1,answer='2か所とも終了後'),
         ('match1_motion_pilot_006',341):dict(question_item=1,answer='2か所とも終了後')}
for row in rows:
    response=answers[(row['candidate_id'],int(row['local_frame']))];before=dict(row)
    row.update(phase_status='ended',reviewer_id='human_user',reviewed_at_utc=now,
        notes='User explicitly confirmed target moment(s) after rally: '+response['answer'])
    events.append(dict(candidate_id=row['candidate_id'],local_frame=int(row['local_frame']),before=before,after=dict(row),
        source='explicit_user_chat_reply',request_call_id='call_793d5907eb2e4a13a7d8894ade729bb4',**response))
write(path,rows)
(OUT/'phase_review_events.jsonl').write_text(''.join(json.dumps(e,ensure_ascii=False)+'\n' for e in events),encoding='utf-8')
replay={(r['candidate_id'],r['local_frame']):r for r in read(initial)}
for e in events:
    key=e['candidate_id'],str(e['local_frame']);assert replay[key]==e['before'];replay[key]=e['after']
assert all(replay[r['candidate_id'],r['local_frame']]==r for r in read(path))
assert sha(initial)==m['initial_review_labels_sha256']
assert all(sha(Path(p))==h for p,h in m['protected_input_hashes'].items())
result=dict(status='three_target_phases_human_confirmed_ended',date_jst='2026-11-09',confirmed_targets=3,
    original_answers=[dict(review_id='A',answer='終了後'),dict(review_id='B',answer='2か所とも終了後')],
    target_phases=[dict(candidate_id=r['candidate_id'],local_frame=int(r['local_frame']),source_frame=int(r['source_frame']),phase_status=r['phase_status']) for r in rows],
    labels_sha256=sha(path),events_sha256=sha(OUT/'phase_review_events.jsonl'),initial_snapshot_sha256=sha(initial),
    scope='Only highlighted target moments; neither precise rally end nor neighbouring-frame phases established.',
    conclusion='All three remaining added errors of the grouped gate occurred after the rally according to the user.',
    no_error_only_exclusion_or_new_inplay_metric=True,old_comparison_retained=dict(original=[66,4],tiled=[77,14],grouped_gate=[77,7]),
    adoption_unchanged=True,model_or_canonical_tracking_changed=False,ball_centers_changed=False,
    next='Establish phase scope across successes and failures before any in-play effectiveness claim or further tuning.',
    protected_inputs_unchanged=True,script_sha256=sha(Path(__file__)))
(OUT/'completed_review.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps(result,indent=2,ensure_ascii=False))
