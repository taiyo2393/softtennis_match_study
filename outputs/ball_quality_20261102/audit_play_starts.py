"""Join existing start evidence to ball clips without relabeling boundaries."""
import csv
import hashlib
import json
from pathlib import Path

SOURCE=Path(r'C:\Users\ta239\Research\softtennis_match_study\study')
WORKSPACE=Path(__file__).resolve().parents[2]
HANDOFF=WORKSPACE/'outputs/frontcourt_handoff_20260927'
OUT=WORKSPACE/'outputs/ball_quality_20261103/play_start_audit'
BASE=WORKSPACE/'outputs/tracking_quality_20261102/ball_aligned'


def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    queue_path=SOURCE/'data/segments/match1_motion_prediction_pilot_queue.csv'
    play_path=SOURCE/'data/segments/match1_reviewed_play_windows.csv'
    serve_path=SOURCE/'data/segments/match1_serve_impact_review.csv'
    brackets_path=HANDOFF/'six_service_contact_brackets.csv'
    attempts_path=HANDOFF/'044_serve_attempt_review.csv'
    refinement_path=HANDOFF/'effective_serve_contact_refinement_candidates.csv'
    label_path=HANDOFF/'rally_label_review.csv'
    queue=read(queue_path);plays={r['play_window_id']:r for r in read(play_path)}
    serves={r['window_id']:r for r in read(serve_path)}
    brackets={r['candidate_id']:r for r in read(brackets_path)}
    refinements={r['candidate_id']:r for r in read(refinement_path)}
    labels={r['candidate_id']:r for r in read(label_path)}
    valid_attempts={r['candidate_id']:r for r in read(attempts_path) if r['attempt_outcome']=='valid'}
    rows=[]
    for q in queue:
        cid=q['candidate_id'];p=plays[q['play_window_id']];wid=p['source_window_id']
        raw=read(BASE/cid/'balltrack_predictions.csv');first=int(raw[0]['source_frame']);last=int(raw[-1]['source_frame'])
        meta=json.loads((SOURCE/'data/balltrack/racketvision/match1'/cid/'runs/threshold030_median120/run_metadata.json').read_text())
        lo=hi=None;provenance='';validity='not_confirmed';contact_status='no_fine_bracket_in_current_handoff'
        if wid in valid_attempts:
            r=valid_attempts[wid];lo=int(r['contact_first_possible_source_frame']);hi=int(r['contact_last_possible_source_frame'])
            provenance='044_serve_attempt_review.csv valid attempt';validity=r['outcome_review_status'];contact_status=r['contact_review_status']
        elif wid in brackets:
            r=brackets[wid];lo=int(r['contact_candidate_first_frame']);hi=int(r['contact_candidate_last_frame'])
            provenance='six_service_contact_brackets.csv';validity=r['effective_serve_status'];contact_status=r['review_status']
        elif wid in refinements:
            r=refinements[wid];lo=int(r['contact_first_possible_source_frame']);hi=int(r['contact_last_possible_source_frame'])
            provenance='effective_serve_contact_refinement_candidates.csv';contact_status=r['review_status']
            validity=labels.get(wid,{}).get('clip_context_review_status','') or 'not_confirmed'
        legacy=serves.get(wid,{})
        timestamp=legacy.get('human_valid_serve_impact_timestamp_sec','')
        relation='unresolved'
        if lo is not None:
            relation='contact_before_clip' if hi<first else 'contact_after_clip' if lo>last else 'clip_contains_contact_candidate'
        elif timestamp:
            relation='legacy_timestamp_before_clip' if float(timestamp)<float(q['source_start_timestamp_sec']) else 'legacy_timestamp_after_clip' if float(timestamp)>float(q['source_end_timestamp_sec']) else 'clip_contains_legacy_timestamp_only'
        rows.append(dict(ball_pilot_id=cid,play_window_id=q['play_window_id'],rally_window_candidate_id=wid,
            clock='legacy_edited_match1_not_redownloaded_match1_original',clip_start_sec=q['source_start_timestamp_sec'],
            clip_end_sec=q['source_end_timestamp_sec'],clip_first_source_frame=first,clip_last_source_frame=last,
            legacy_start_method=p['start_establishment_method'],legacy_serve_timestamp_sec=timestamp,
            legacy_serve_review_status=legacy.get('impact_review_status',''),
            effective_contact_candidate_first_source_frame=lo if lo is not None else '',
            effective_contact_candidate_last_source_frame=hi if hi is not None else '',
            first_contact_candidate_local_frame=max(0,lo-first) if lo is not None and first<=hi<=last else '',
            last_contact_candidate_local_frame=hi-first if lo is not None and first<=hi<=last else '',
            contact_candidate_start_sec=f'{lo/meta["fps"]:.6f}' if lo is not None else '',
            contact_candidate_end_sec=f'{hi/meta["fps"]:.6f}' if hi is not None else '',
            contact_evidence=provenance,contact_review_status=contact_status,serve_validity_status=validity,
            clip_contact_relation=relation,tracking_gate_status='provisional_not_human_confirmed_start',
            end_status='legacy_camera_departure_not_exact_rally_end'))
    with (OUT/'ball_pilot_start_inventory.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    from collections import Counter
    summary=dict(ball_pilot_clips=len(rows),legacy_start_methods=dict(Counter(r['legacy_start_method'] for r in rows)),
        fine_start_brackets_available=sum(r['effective_contact_candidate_first_source_frame']!='' for r in rows),
        clip_contact_relation=dict(Counter(r['clip_contact_relation'] for r in rows)),
        source_hashes={str(p):sha(p) for p in (queue_path,play_path,serve_path,brackets_path,attempts_path,refinement_path,label_path)},
        warning='Retains prior reviews; no automatic boundary/serve-validity confirmation. These IDs are different namespaces, linked by tables.')
    (OUT/'inventory_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='source_hashes'},indent=2))
    print(json.dumps(next(r for r in rows if r['ball_pilot_id']=='match1_motion_pilot_002'),indent=2))


if __name__=='__main__':main()
