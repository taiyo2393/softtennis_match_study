"""Compare proposals to existing intervals only after label-blind generation."""
import csv
import json
from pathlib import Path

from extract_serve_start_features import SOURCE,ROOT,sha

OUT=ROOT/'serve_start_pilot_tiled_v2'
HANDOFF=ROOT.parent/'frontcourt_handoff_20260927'


def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def main():
    frozen=json.loads((OUT/'frozen_detector.json').read_text())
    assert sha(ROOT/'propose_serve_starts.py')==frozen['proposal_code_sha256']
    inventory=read(ROOT/'play_start_audit/ball_pilot_start_inventory.csv')
    attempts=read(HANDOFF/'044_serve_attempt_review.csv');result=[]
    for pilot in ('016','002'):
        path=OUT/f'match1_motion_pilot_{pilot}'
        generated=json.loads((path/'proposals.json').read_text())
        assert generated['labels_read'] is False and generated['code_sha256']==frozen['proposal_code_sha256']
        assert json.loads((path/'proposal_protocol.json').read_text())==frozen['proposal_parameters']
        entry=next(r for r in inventory if r['ball_pilot_id']==f'match1_motion_pilot_{pilot}')
        first=int(entry['clip_first_source_frame']);references=[]
        if pilot=='016':
            for a in attempts:
                references.append(dict(attempt_number=a['attempt_number'],outcome=a['attempt_outcome'],
                    first=int(a['contact_first_possible_source_frame'])-first,
                    last=int(a['contact_last_possible_source_frame'])-first,
                    outcome_status=a['outcome_review_status'],timing_status=a['contact_review_status']))
        else:references.append(dict(attempt_number='unknown',outcome='unknown',
            first=int(entry['first_contact_candidate_local_frame']),last=int(entry['last_contact_candidate_local_frame']),
            outcome_status=entry['serve_validity_status'],timing_status=entry['contact_review_status']))
        comparisons=[]
        for ref in references:
            hits=[]
            for i,p in enumerate(generated['proposals']):
                if p['contact_candidate_first_local_frame']<=ref['last'] and p['contact_candidate_last_local_frame']>=ref['first']:
                    hits.append(dict(proposal_index=i,audio_peak_frame=p['audio_peak_local_sec']*59.94005994,
                        proposed_interval=[p['contact_candidate_first_local_frame'],p['contact_candidate_last_local_frame']]))
            comparisons.append(dict(reference=ref,overlapping_proposals=hits))
        covered={h['proposal_index'] for c in comparisons for h in c['overlapping_proposals']}
        result.append(dict(pilot=pilot,proposals=len(generated['proposals']),comparisons=comparisons,
            extra_proposals_without_reference_match=[dict(index=i,interval=[p['contact_candidate_first_local_frame'],p['contact_candidate_last_local_frame']],
                audio_peak_frame=p['audio_peak_local_sec']*59.94005994,status='unreviewed_extra_not_automatically_false') for i,p in enumerate(generated['proposals']) if i not in covered],
            role='development_and_tuning' if pilot=='016' else 'fixed_parameter_same_match_application_not_external_test'))
    report=dict(results=result,automatic_validity_classification=False,exact_contact_accuracy_established=False,
        comparison_basis='Existing AI contact intervals with separately human-reviewed validity for044; intervals are not independent exact ground truth.',
        warning='Overlap on two previously viewed clips is feasibility evidence only. It is not recall/precision on a complete independently labeled match.')
    (OUT/'comparison.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
