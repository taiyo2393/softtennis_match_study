"""Freeze completed manual references before evaluating the fixed position policies."""
from pathlib import Path
import json,hashlib,shutil,math
from collections import Counter
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,sha,dump
import evaluate_position_reference_pilot as evaluator

BUNDLE=BASE/'identity_position_bundle'
PILOT=BUNDLE/'reference_pilot'
SLOTS=['near_left','near_right','far_left','far_right']

def main():
    raw=(PILOT/'human_reference_labels.json').read_bytes();labels=json.loads(raw);assert labels['label_origin']=='manual_reference_ui';manifest=json.loads((PILOT/'private_sampling_manifest.json').read_text(encoding='utf-8'));expected={r['image_id'] for r in manifest['items']};assert set(labels['labels'])==expected
    counts=Counter()
    for image,entries in labels['labels'].items():
        assert set(entries)==set(SLOTS)
        for e in entries.values():
            assert e['state'] in ['point','unplaceable','absent','uncertain'];counts[e['state']]+=1
            if e['state']=='point':assert math.isfinite(e['x']) and math.isfinite(e['y']) and 0<=e['x']<3840 and 0<=e['y']<2160
            else:assert e['x'] is None and e['y'] is None
    assert sum(counts.values())==52
    stage=BASE/f'position_reference_evaluation_20261110_r{labels["revision"]}';assert not stage.exists();stage.mkdir();snapshot=stage/'human_reference_labels.json';snapshot.write_bytes(raw);assert sha(snapshot)==hashlib.sha256(raw).hexdigest();shutil.copy2(PILOT/'private_sampling_manifest.json',stage/'private_sampling_manifest.json');shutil.copy2(PILOT/'private_sampling_protocol.json',stage/'private_sampling_protocol.json')
    protected={str(n):sha(n) for n in [BUNDLE/'publication_hashes.json',BUNDLE/'position_policy_observations.csv',PILOT/'private_sampling_manifest.json',PILOT/'private_sampling_protocol.json',PILOT/'index.html',ROOT/'evaluate_position_reference_pilot.py',Path(__file__),snapshot]}
    dump(stage/'protocol.json',dict(reference_snapshot_revision=labels['revision'],reference_snapshot_sha256=sha(snapshot),reference_state_counts=dict(counts),images=13,manual_reference_fields=52,reference_definition='Approximate shoe-midpoint x and lower shoe-bottom y in the original image. No physical ground-contact or surveyed-coordinate claim.',reference_source='User completed the UI without displaying model point/box outputs. Reviewer had seen prior development material, so full reviewer blinding is not claimed.',position_policies_fixed_before_new_reference_coordinates=True,evaluation_algorithm_sha256=sha(ROOT/'evaluate_position_reference_pilot.py'),uniform10_and_known_warning4_reported_separately=True,one_shared_image_not_an_independent_extra_observation=True,within_side_unordered_correspondence_not_ID_accuracy=True,missing_predictions_and_unplaceable_references_not_discarded=True,no_error_pass_threshold_or_after_label_tuning=True,no_outcome_model_fit=True,protected_input_hashes=protected))
    evaluator.PILOT=stage;evaluator.OUT=BUNDLE;evaluator.main()
    assert all(sha(Path(n))==h for n,h in protected.items());dump(stage/'completion_snapshot.json',dict(status='manual_references_frozen_and_evaluated',stage=str(stage),label_revision=labels['revision'],state_counts=dict(counts),live_reference_file_changed=False,canonical_inputs_changed=False));print(stage);print(dict(counts))

if __name__=='__main__':main()
