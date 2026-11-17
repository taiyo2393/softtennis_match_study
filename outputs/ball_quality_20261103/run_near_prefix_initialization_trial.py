"""Freeze and run ten matched cached prefix-only initialization comparisons."""
import itertools
import json
import shutil
from pathlib import Path
import numpy as np
from run_fixed_cohort_player_case_v3 import ROOT, BASE, read, write, sha, dump
from diagnose_initial_pair_appearance import cache
from run_position_proxy_support import native_table
from near_prefix_initialization import NearPrefix, PARAMETERS, old, adaptive
from court_initialization_guard import eligible

OUT = BASE / 'near_prefix_initialization_trial'


def load(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def prepare():
    assert not OUT.exists(), 'Retain previous attempts.'
    parent = load(BASE / 'position_proxy_support_trial/protocol.json')
    diagnostic = load(BASE / 'pair_initialization_development/diagnostic_protocol.json')
    cases = parent['cases']
    assert [e['tag'] for e in cases] == ['002','005','006','009','012','015','016','017','018','019']
    protected = {}
    files = [Path(__file__), ROOT/'near_prefix_initialization.py', ROOT/'court_initialization_guard.py',
             ROOT/'confirmed_continuity_guard.py', Path(old.__file__), Path(old.previous.__file__),
             Path(old.a.__file__), Path(adaptive.__file__), Path(diagnostic['calibration']),
             BASE/'position_reference_evaluation_20261110_r55/human_reference_labels.json',
             BASE/'identity_position_bundle/publication_hashes.json',
             BASE/'position_reference_evaluation_20261110_r55/publication_hashes.json']
    for e in cases:
        files.extend([Path(e['raw_csv']), Path(e['cache'])/'cached_detections.csv',
                      Path(e['cache'])/'appearance_cache.npz',
                      Path(e['temporal_folder'])/'selected_four.csv',
                      Path(e['temporal_folder'])/'events.json',
                      BASE/f'torso_temporal_association_trial/case_{e["tag"]}/torso_team_soft_cost/far_pair_path.csv'])
    files.extend(p for p in (BASE/'current22_player_inputs').iterdir() if p.is_file())
    for p in files:
        protected[str(p)] = sha(p)
    OUT.mkdir()
    dump(OUT/'protocol.json', dict(cases=cases, parameters=PARAMETERS,
        calibration=diagnostic['calibration'], source_video=parent['source_video'],
        source_video_sha256=parent['source_video_sha256'],
        policy_fixed_before_new_outputs=True,
        trigger='Known006 all-four initialization delay; development after position-reference review.',
        selection='All ten retained contexts, same native detections and fixed31 times. No substitutions.',
        method='Initialize only the near pair after8 past consecutive native raw-ID observations. Same geometry/court and near-clothing pair rank. Continue only same raw IDs with existing motion/appearance/size gates; gap<=2, no gallery update or recovery. Stop supplementary output at the unchanged all-four initialization event.',
        far_pair_independently_initialized=False,
        handoff='Diagnostic only: compare prefix slot raw IDs to common-seed IDs. Do not relabel prior frames using later identities. A disagreement blocks adoption.',
        existing_post_initialization_rows_modified=False,
        comparison='Existing frozen near paths plus torso-team-soft-cost far paths. Exact unchanged boxes after common initialization by construction, not a new full-tracker replay.',
        label_use='FrozenP011 manually placed near points are a same-match development diagnostic only. No thresholds fitted, and no independent accuracy improvement claimed.',
        initial_same_raw_ID_is_not_proof_of_person_identity=True,
        all_source_frames_are_original_edited_video_clock=True,
        canonical_pipeline_replaced=False, outcome_labels_changed=False,
        protected_input_hashes=protected))
    print('Frozen ten contexts and existing31 input times.', flush=True)


def run():
    p = load(OUT/'protocol.json')
    assert all(sha(Path(n)) == h for n,h in p['protected_input_hashes'].items())
    H = np.array(load(Path(p['calibration']))['image_to_court_homography'])
    reports = []
    availability = []
    for e in p['cases']:
        tag=e['tag']; dest=OUT/f'case_{tag}'; dest.mkdir()
        rows, array, offsets = cache(e)
        baseline = native_table(e)
        event = load(Path(e['temporal_folder'])/'events.json')[0]
        start = event['frame']
        tr = NearPrefix(H, e.get('adaptive_size', True))
        early=[]; diagnostic=[]
        for i, f in enumerate(range(e['first'], start)):
            current=rows[offsets[i]:offsets[i+1]]
            features=[None if np.isnan(x).all() else x for x in array[offsets[i]:offsets[i+1]]]
            near=[]; far=[]
            for r in current:
                if not eligible(r,H):continue
                x,y=old.a.foot(r)/[3840,2160]; h=old.a.height(r)/2160
                if y>=.61 and h>=.08:near.append(r)
                elif .55<=y<.61 and .30<=x<=.63 and h>=.05:far.append(r)
            diagnostic.append(dict(source_frame=f,near_candidates=len(near),far_candidates=len(far),
                near_raw_ids=';'.join(r['track_id'] for r in near),far_raw_ids=';'.join(r['track_id'] for r in far)))
            for slot,r,margin,value in tr.step(f,current,features):
                assert int(r['source_frame'])==f
                early.append(dict(r,player_slot=slot,position_source='detected_box',
                    identity_status='near_prefix_development_candidate_not_GT',
                    identity_source='near_pair_prefix_same_raw_ID',
                    assignment_cost=round(value,5),alternative_cost_margin=round(margin,5)))
        if early:write(dest/'additional_near_prefix.csv',early)
        if diagnostic:write(dest/'waiting_period_diagnostic.csv',diagnostic)
        dump(dest/'events.json',tr.events)
        combined={f:dict(group) for f,group in baseline.items()}
        for r in early:
            f=int(r['source_frame']); slot=r['player_slot']
            assert slot not in combined.get(f,{})
            combined.setdefault(f,{})[slot]=r
        flat=[row for f,group in sorted(combined.items()) for slot,row in sorted(group.items())]
        write(dest/'candidate_native_observations.csv',flat)
        # All prior outputs stay byte-equivalent field-by-field in memory.
        assert all(combined[f][s]==r for f,g in baseline.items() for s,r in g.items())
        ids=tr.events[0]['seed_ids'] if tr.events else {}
        near_seeds={s:t for s,t in event['seed_ids'].items() if s.startswith('near_')}
        compatible=not ids or ids==near_seeds
        for f in e['requested_input_frames']:
            counts=[]
            for table in [baseline,combined]:
                g=table.get(f,{})
                counts.extend([sum(s.startswith('near_') for s in g),sum(s.startswith('far_') for s in g)])
            availability.append(dict(tag=tag,source_frame=f,baseline_near=counts[0],baseline_far=counts[1],candidate_near=counts[2],candidate_far=counts[3]))
        relevant=[r for r in availability if r['tag']==tag]
        report=dict(tag=tag,common_initialized_frame=start,
            near_prefix_initialized_frame=tr.events[0]['frame'] if tr.events else None,
            near_prefix_seed_ids=ids,common_near_seed_ids=near_seeds,
            slot_handoff_matches_raw_IDs=compatible,
            additional_native_boxes=len(early),
            additional_requested_time_native_boxes=sum(r['candidate_near']-r['baseline_near'] for r in relevant),
            baseline_near_complete_times=sum(r['baseline_near']==2 for r in relevant),
            candidate_near_complete_times=sum(r['candidate_near']==2 for r in relevant),
            baseline_four_complete_times=sum(r['baseline_near']+r['baseline_far']==4 for r in relevant),
            candidate_four_complete_times=sum(r['candidate_near']+r['candidate_far']==4 for r in relevant),
            existing_observations_unchanged=True,identity_accuracy_established=False)
        dump(dest/'result.json',report); reports.append(report)
        print(json.dumps(report),flush=True)
    write(OUT/'fixed_time_availability_not_accuracy.csv',availability)
    labels=load(BASE/'position_reference_evaluation_20261110_r55/human_reference_labels.json')
    points=[np.array([v['x'],v['y']]) for k,v in labels['labels']['P011'].items() if k.startswith('near_') and v['state']=='point']
    observations=[r for r in read(OUT/'case_006/additional_near_prefix.csv') if int(r['source_frame'])==10546]
    assert len(points)==len(observations)==2
    predicted=[old.a.foot(r) for r in observations]
    order=min(itertools.permutations(range(2)),key=lambda order:sum(np.linalg.norm(predicted[i]-points[j]) for i,j in enumerate(order)))
    errors=[dict(player_slot=observations[i]['player_slot'],raw_id=observations[i]['track_id'],
        native_point=predicted[i].tolist(),manual_point=points[j].tolist(),
        error_px=float(np.linalg.norm(predicted[i]-points[j])),
        dx=float(predicted[i][0]-points[j][0]),dy=float(predicted[i][1]-points[j][1])) for i,j in enumerate(order)]
    dump(OUT/'P011_development_reference_diagnostic.json',dict(image_id='P011',source_frame=10546,
        baseline_near_predictions=0,candidate_near_predictions=2,errors=errors,
        mean_error_px=float(np.mean([x['error_px'] for x in errors])),
        unordered_within_side_position_comparison_not_ID_accuracy=True,
        reference_known_before_development=True,independent_test=False))
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    dump(OUT/'result.json',dict(cases=reports,retained_contexts=10,
        all_existing_native_observations_unchanged=True,
        all_handoffs_compatible=all(r['slot_handoff_matches_raw_IDs'] for r in reports),
        protected_inputs_unchanged=True,canonical_pipeline_replaced=False,
        independent_tracking_accuracy_established=False))
    for name in ['near_prefix_initialization.py','run_near_prefix_initialization_trial.py']:
        shutil.copy2(ROOT/name,OUT/('executed_'+name))


if __name__=='__main__':
    import sys
    {'prepare':prepare,'run':run}[sys.argv[1]]()
