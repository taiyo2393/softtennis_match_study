"""Retain failed ungated trial and freeze a camera-conditioned follow-up."""
import sys
import shutil
from pathlib import Path
from run_fixed_cohort_player_case_v3 import ROOT, BASE, SOURCE, read, sha, dump
import run_near_prefix_initialization_trial as prior
from camera_valid_near_prefix import CameraValidNearPrefix

OUT = BASE/'camera_valid_near_prefix_trial'
CAMERA = SOURCE/'data/segments/analysis_ready_main_camera_segments.csv'


def prepare():
    assert not OUT.exists()
    p=prior.load(prior.OUT/'protocol.json')
    failed=prior.load(prior.OUT/'result.json')
    assert not failed['all_handoffs_compatible']
    p['protected_input_hashes']=dict(p['protected_input_hashes'])
    for n in [Path(__file__),ROOT/'camera_valid_near_prefix.py',prior.OUT/'protocol.json',prior.OUT/'result.json',CAMERA]:
        p['protected_input_hashes'][str(n)]=sha(n)
    rows=[r for r in read(CAMERA) if r['video_id']=='match1' and r['usable_for_analysis'].lower()=='true']
    p.update(camera_eligibility_source=str(CAMERA),
        camera_intervals=[[int(r['start_source_frame']),int(r['end_source_frame'])] for r in rows],
        camera_input_human_checked_counts={state:sum(r['human_checked']==state for r in rows) for state in sorted(set(r['human_checked'] for r in rows))},
        additional_condition='Supplementary near seeding and continuation only inside the already-existing main-camera intervals; clear window/state otherwise. Same homography cannot be used on close-ups or dissolves.',
        camera_condition_warning='Inherited geometry/background model segments, human_checked=false. Offline boundaries may use future image information. Not an independently validated camera classifier or end-to-end causal tracker.',
        condition_chosen_after_seeing_ungated009_failure=True,
        ignored_or_shifted_requested_input_frames=False,
        camera_condition_applied_to_existing_baseline=False,
        ungated_failure_retained=str(prior.OUT))
    OUT.mkdir();dump(OUT/'protocol.json',p)
    print('Frozen inherited main-camera eligibility for all ten contexts.',flush=True)


def run():
    p=prior.load(OUT/'protocol.json')
    prior.OUT=OUT
    prior.NearPrefix=lambda H,adaptive_size=True:CameraValidNearPrefix(H,p['camera_intervals'],adaptive_size)
    prior.run()
    for name in ['camera_valid_near_prefix.py','run_camera_valid_near_prefix_trial.py']:
        shutil.copy2(ROOT/name,OUT/('executed_'+name))


if __name__=='__main__':{'prepare':prepare,'run':run}[sys.argv[1]]()
