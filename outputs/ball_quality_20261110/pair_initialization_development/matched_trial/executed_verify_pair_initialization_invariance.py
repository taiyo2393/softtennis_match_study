"""Check initial person selection under row permutation and arbitrary ID relabeling."""
from pathlib import Path
import numpy as np
from diagnose_initial_pair_appearance import OUT,load,cache
from run_fixed_cohort_player_case_v3 import ROOT,sha,dump
from pair_initialization_guard import pair_initialization_class
from confirmed_continuity_guard import adaptive,old

def main():
    p=load(OUT/'matched_trial/protocol.json');h=np.array(load(Path(p['calibration']))['image_to_court_homography']);reports=[]
    for e in p['cases']:
        raw,array,offset=cache(e);base=adaptive.AdaptiveNearTracker if e['adaptive_size'] else old.TemporalTracker
        for abstain in [False,True]:
            outcomes=[]
            for change in ['original','reverse_rows','rename_IDs']:
                tr=pair_initialization_class(base,h,abstain)();initial_selector=old.selector.seed_by_court_geometry
                for i,f in enumerate(range(e['first'],e['last']+1)):
                    rows=[dict(r) for r in raw[offset[i]:offset[i+1]]];features=[None if np.isnan(x).all() else x for x in array[offset[i]:offset[i+1]]]
                    if change=='reverse_rows':rows.reverse();features.reverse()
                    if change=='rename_IDs':
                        for row in rows:row['track_id']=str(int(row['track_id'])+20000000)
                    tr.initialize(f,rows,features);assert old.selector.seed_by_court_geometry is initial_selector
                    if tr.initialized_frame is not None:break
                event=tr.events[0] if tr.events else None
                outcomes.append(None if event is None else dict(frame=event['frame'],seeds={k:v-(20000000 if change=='rename_IDs' else 0) for k,v in event['seed_ids'].items()}))
            assert outcomes[0]==outcomes[1]==outcomes[2],(e['tag'],abstain,outcomes)
            reports.append(dict(tag=e['tag'],abstain=abstain,initial_selection_invariant=True,initialization=outcomes[0]))
    assert all(sha(Path(n))==value for n,value in p['protected_input_hashes'].items())
    dest=OUT/'matched_trial/invariance_verification.json';assert not dest.exists();dump(dest,dict(status='passed',comparisons=len(reports),variants_per_comparison=3,row_permutation_and_ID_relabeling_preserve_initial_selection=True,global_selector_restored_after_each_call=True,cases=reports,script_sha256=sha(Path(__file__)),independent_identity_accuracy_established=False));print('Passed',len(reports),'initial-selection invariance comparisons.')

if __name__=='__main__':main()
