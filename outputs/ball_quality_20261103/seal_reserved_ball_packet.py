"""Record overnight handoff after verification, preserving all pending labels."""
import json
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,sha
OUT=ROOT.parent/'ball_quality_20261104'
assert not (OUT/'overnight_handoff.json').exists()
m=load(OUT/'manifest.json');old=load(ROOT/'balltrack_manual_finetune_pilot/data_manifest.json')
for role,h in old['partition_hashes'].items():
    assert sha(ROOT/'balltrack_manual_finetune_pilot'/(role+'_labels.csv'))==h
assert sha(SOURCE/'annotations/balltrack_groundtruth_match1_pilot.csv')==old['source_sha256']
assert all(sha(Path(p))==h for p,h in m['protected_input_hashes'].items())
assert all(sha(Path(p))==h for p,h in m['frozen_artifact_hashes'].items())
assert sha(OUT/'manual_sequence_labels.csv')==sha(OUT/'initial_labels_snapshot.csv')==m['initial_labels_sha256']
assert not (OUT/'annotation_events.jsonl').exists() and not (OUT/'evaluation_report.json').exists()
assert all(g['review_status']=='unreviewed' for g in read(OUT/'manual_sequence_labels.csv'))
decision=ROOT/'ball_balanced_fixed_validation/operational_decision.json'
assert load(decision)['status']=='not_adopted'
for name in ['verification.json','http_verification.json','view_verification.json']:
    assert load(OUT/name)['status']=='passed'
paths=[OUT/n for n in ['README.md','inventory_manifest.json','sequence_protocol.json','manifest.json','verification.json',
    'http_verification.json','view_verification.json','view_revision.json','initial_labels_snapshot.csv','start_review.ps1']]
paths += [ROOT/n for n in ['inventory_next_ball_sequences.py','prepare_reserved_ball_sequences.py','make_reserved_review_tools.py',
    'serve_reserved_ball_sequence_review.py','serve_reserved_ball_sequence_review_fit.py','evaluate_reserved_ball_sequences.py',
    'verify_reserved_ball_sequences.py','fit_reserved_review_view.py','verify_reserved_review_view.py','seal_reserved_ball_packet.py']]
handoff=dict(status='human_review_pending_no_background_training',date_jst='2026-11-04',
    completed=['70clip temporal exposure inventory','4windows reserved before viewing predictions',
               '104frozen predictions from2already fixed methods','52blind review images and context videos',
               'decoder/clock/selection/media/HTTP/browser checks','post-annotation evaluation implementation'],
    pending=dict(human_target_reviews=52,new_accuracy=None),
    blocked_reason='Independent phase and source-image ball-center truth must come from human review; do not substitute model predictions.',
    scope='Additional same-match development sequences, prior nonpilot exposure not fully known; not external or independent match validation.',
    prior_model_decision=str(decision),prior_model_decision_sha256=sha(decision),
    retained='Original weights, existing reviewed tracking references and labels',
    training_runs=0,new_human_labels=0,operational_changes=False,
    current_review_url=load(OUT/'server_state.json')['url'],
    restart_command=str(OUT/'start_review.ps1'),
    final_artifact_hashes={str(p):sha(p) for p in paths})
(OUT/'overnight_handoff.json').write_text(json.dumps(handoff,indent=2,ensure_ascii=False),encoding='utf-8')
print(json.dumps({k:v for k,v in handoff.items() if k!='final_artifact_hashes'},indent=2,ensure_ascii=False))
