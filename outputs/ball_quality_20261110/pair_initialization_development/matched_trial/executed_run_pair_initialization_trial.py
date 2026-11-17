"""Matched cached full-context temporal replays; no far-path or input adoption."""
import json,shutil
from pathlib import Path
from collections import Counter
import numpy as np
from diagnose_initial_pair_appearance import OUT,load,cache
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,write,sha,dump
from court_initialization_guard import court_initialization_class,PARAMETERS as COURT_PARAMETERS
from pair_initialization_guard import pair_initialization_class,PARAMETERS
from confirmed_continuity_guard import adaptive,old

def main():
    diag=load(OUT/'diagnostic_protocol.json');cases=diag['cases'];assert load(OUT/'diagnostic_verification.json')['status']=='passed';assert all(sha(Path(n))==h for n,h in diag['protected_input_hashes'].items())
    out=OUT/'matched_trial';out.mkdir();h=np.array(load(Path(diag['calibration']))['image_to_court_homography'])
    protected=dict(diag['protected_input_hashes'])
    for n in [OUT/'diagnostic_protocol.json',OUT/'initial_candidate_pairs.csv',Path(__file__),ROOT/'pair_initialization_guard.py',ROOT/'court_initialization_guard.py',Path(adaptive.__file__),Path(old.__file__),Path(old.previous.__file__)]:protected[str(n)]=sha(n)
    for n in (BASE/'current21_player_inputs').iterdir():
        if n.is_file():protected[str(n)]=sha(n)
    modes=['court_seed_only','pair_rank','pair_rank_abstain']
    dump(out/'protocol.json',dict(cases=cases,modes=modes,parameters=PARAMETERS,court_parameters=COURT_PARAMETERS,source_video=diag['source_video'],source_video_sha256=diag['source_video_sha256'],calibration=diag['calibration'],policy_fixed_before_replay=True,selection='All18 diagnosed cases, no exclusions or substitution. Same-match development after known018 failure.',margin_rationale='0.06 reused as engineering choice from existing negative-appearance margin. Not calibrated for pair ranking, no parameter sweep. Compare ungated ranking and conservative abstention to expose cost.',uniform_assumption='Near teammates share clothing appearance; may fail with mismatched uniforms, two officials or strong front/back appearance differences.',new_absolute_similarity_threshold=False,tracking_and_recovery_parameters_changed=False,new_person_ID_bindings=False,phase_or_outcomes_used_to_select=False,policy_uses_future_frames=False,canonical_pipeline_replaced=False,analysis_inputs_changed=False,independent_identity_accuracy_established=False,protected_input_hashes=protected))
    reports=[];cols=['source_frame','player_slot','track_id','x1','y1','x2','y2','identity_source','assignment_cost','alternative_cost_margin'];signature=lambda rs:[tuple(str(r[k]) for k in cols) for r in rs];boxkey=lambda r:tuple(str(r[k]) for k in cols[:7])
    for entry in cases:
        tag=entry['tag'];folder=out/f'case_{tag}';folder.mkdir();raw,array,offset=cache(entry);base=adaptive.AdaptiveNearTracker if entry['adaptive_size'] else old.TemporalTracker
        trackers=dict(court_seed_only=court_initialization_class(base,h)(),pair_rank=pair_initialization_class(base,h)(),pair_rank_abstain=pair_initialization_class(base,h,abstain=True)());selected={m:[] for m in modes}
        for i,f in enumerate(range(entry['first'],entry['last']+1)):
            rows=raw[offset[i]:offset[i+1]];features=[None if np.isnan(x).all() else x for x in array[offset[i]:offset[i+1]]]
            for mode,tr in trackers.items():
                for slot,j,margin,value,origin in tr.step(f,rows,features):selected[mode].append(dict(rows[j],player_slot=old.a.SLOTS[slot],position_source='detected_box',identity_status='pair_initialization_development_candidate_not_GT',identity_source=origin,assignment_cost=round(value,5),alternative_cost_margin=round(margin,5)))
        assert signature(selected['court_seed_only'])==signature(read(Path(entry['reference_mode_folder'])/'selected_four.csv')),tag
        original={boxkey(row) for row in selected['court_seed_only']};result={}
        for mode,tr in trackers.items():
            dest=folder/mode;dest.mkdir();rows=selected[mode];assert len({(row['source_frame'],row['player_slot']) for row in rows})==len(rows)
            if rows:write(dest/'selected_four.csv',rows)
            dump(dest/'events.json',tr.events)
            if hasattr(tr,'pair_seed_log'):dump(dest/'pair_seed_log.json',tr.pair_seed_log)
            actual={boxkey(row) for row in rows};init=tr.events[0] if tr.events else {};start=init.get('frame');result[mode]=dict(initialized_frame=start,seed_ids=init.get('seed_ids'),initialization_delta_from_court_filter=None if start is None else start-trackers['court_seed_only'].initialized_frame,temporal_rows=len(rows),near_rows=sum(row['player_slot'].startswith('near_') for row in rows),same_temporal_output_as_court_filter=signature(rows)==signature(selected['court_seed_only']),removed_box_rows=len(original-actual),added_box_rows=len(actual-original),new_person_ID_bindings=False,abstained_frames=sum(x['abstained'] for x in getattr(tr,'pair_seed_log',[])),ambiguous_candidate_frames=len(getattr(tr,'pair_seed_log',[])))
        report=dict(tag=tag,role=entry['role'],court_filter_temporal_replay_exact=True,modes=result,independent_identity_accuracy_established=False);dump(folder/'result.json',report);reports.append(report);print(json.dumps(report,ensure_ascii=False),flush=True)
    assert all(sha(Path(n))==h for n,h in protected.items())
    dump(out/'result.json',dict(status='completed18_matched_development_replays',cases=reports,court_filter_exact_replays=len(reports),protected_sources_unchanged=True,analysis_inputs_changed=False,canonical_pipeline_replaced=False,independent_identity_accuracy_established=False))
    for name in ['pair_initialization_guard.py','run_pair_initialization_trial.py','diagnose_initial_pair_appearance.py']:shutil.copy2(ROOT/name,out/('executed_'+name))

if __name__=='__main__':main()
