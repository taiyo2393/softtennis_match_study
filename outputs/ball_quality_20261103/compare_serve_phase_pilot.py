"""Compare saved phase proposals to old AI intervals after inference only."""
import json
from segment_serve_phases import OUT
from extract_serve_start_features import ROOT,sha

def main():
    prior=ROOT/'serve_start_pilot_tiled_v2/comparison.json'
    refs=json.loads(prior.read_text());results=[]
    for old in refs['results']:
        cid=old['pilot'];p=OUT/f'match1_motion_pilot_{cid}/phases.json';d=json.loads(p.read_text())
        overlaps=lambda rows,r:[[q['contact_candidate_first_local_frame'],q['contact_candidate_last_local_frame']] for q in rows if q['contact_candidate_first_local_frame']<=r['last'] and q['contact_candidate_last_local_frame']>=r['first']]
        comparisons=[]
        for c in old['comparisons']:
            r=c['reference']
            comparisons.append(dict(reference=r,repair_only_overlaps=overlaps(d['repaired_only_candidates'],r),
                phase_retained_overlaps=overlaps(d['accepted_candidates'],r),
                phase_deferred_overlaps=overlaps([q for q in d['candidates'] if q['review_action']=='defer_insufficient_start_evidence'],r)))
        results.append(dict(pilot=cid,comparisons=comparisons,phase_output_sha256=sha(p),role='development_and_tuning_not_independent_test',
            suppressed_during_activity=sum(q['review_action']=='suppress_during_activity' for q in d['candidates']),
            deferred=sum(q['review_action']=='defer_insufficient_start_evidence' for q in d['candidates'])))
    report=dict(results=results,reference_file_sha256=sha(prior),
        independent_accuracy_established=False,validity_classification=False,
        limitation='References are prior AI contact intervals. Human-reviewed outcomes on044 are not automatic inference. No complete negative labels or exact contact GT.')
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
