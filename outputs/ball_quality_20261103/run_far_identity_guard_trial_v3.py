"""Third development comparison: gate only short-lived candidate tracklets."""
from pathlib import Path
import json,shutil
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,write,sha,dump
from run_far_identity_guard_trial import build_cache,export,OUT,load
import far_identity_guard_v2 as guard

def main():
    previous=OUT/'matched_trial';p=load(previous/'protocol.json');assert load(previous/'result.json')['baseline_exact_replays']==10;assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items());out=OUT/'matched_trial_v3';out.mkdir()
    protected=dict(p['protected_input_hashes'])
    for n in [previous/'protocol.json',previous/'result.json',previous/'changed_observation_review/manifest.json',Path(__file__),ROOT/'far_identity_guard_v2.py',OUT/'matched_trial_v2/protocol.json',OUT/'matched_trial_v2/result.json']+[n for n in previous.glob('case_*/*/far_pair_path.csv')]:protected[str(n)]=sha(n)
    dump(out/'protocol.json',dict(cases=p['cases'],modes=['baseline','unstable_tracklet_identity_guard'],guard_parameters=dict(guard.PARAMETERS,gate_only_unstable_tracklets=True,stable_tracklet_min_observations=8),far_parameters=p['far_parameters'],policy_fixed_before_v3_outputs=True,development_reason='Both earlier variants rejected visually supported established player tracklets. Restrict new veto to candidates already marked unstable by retained eight-observation rule; keep established-tracklet costs unchanged. Existing offline future-defined stability is retained, not an online claim. No numeric sweep; this is a third same-match development choice after inspecting earlier variants.',parent_protocol=str(previous/'protocol.json'),parent_protocol_sha256=sha(previous/'protocol.json'),same_detections_near_paths_and_solver=True,baseline_outputs_reused_from_exact_first_trial=True,canonical_pipeline_replaced=False,independent_GT_added=False,protected_input_hashes=protected))
    fields=['source_frame','player_slot','track_id','x1','y1','x2','y2','bottom_center_x','bottom_center_y'];key=lambda r:tuple(r[k] for k in fields);reports=[]
    for e in p['cases']:
        tag=e['tag'];dest=out/f'case_{tag}';dest.mkdir();event,near,cache,scores=build_cache(e)
        # Reuse validated image features; no new detector run or changed anchors.
        from diagnose_initial_pair_appearance import cache as read_cache
        from collections import defaultdict
        import numpy as np
        raw,array,offset=read_cache(e);features={(int(r['source_frame']),r['track_id']):(None if np.isnan(f).all() else f) for r,f in zip(raw,array)};byframe=defaultdict(list)
        for r in raw:byframe[int(r['source_frame'])].append(r)
        anchors={slot:[features[f,str(tid)] for f in range(event['window_start'],event['frame']+1)] for slot,tid in event['seed_ids'].items()}
        chosen=[r for r in byframe[event['frame']] if int(r['track_id']) in event['seed_ids'].values()];negative=[features[event['frame'],r['track_id']] for r in byframe[event['frame']] if int(r['track_id']) not in event['seed_ids'].values() and all(guard.original.t.overlap_fraction(r,s)<.1 for s in chosen)];decisions=[]
        for items in cache:
            for item in items:
                row=item['row'];allowed,d=guard.gate(features[int(row['source_frame']),row['track_id']],anchors,negative);allowed=[True,True] if item['stable_tracklet'] else allowed;item['identity_allowed']=allowed
                for i,slot in enumerate(['far_rear','far_front']):decisions.append(dict(source_frame=row['source_frame'],track_id=row['track_id'],player_slot=slot,allowed=int(allowed[i]),own_distance=d['own'][i],near_distance=d['near'],own_upper=d['own_upper'][i],corroborating_negative_count=d['corroborating_negative_counts'][i]))
        write(dest/'candidate_gate_diagnostic.csv',decisions);path,search=guard.solve(cache,event)
        for f,choices in enumerate(path):
            for i,j in enumerate(choices):assert j<0 or cache[f][j]['identity_allowed'][i]
        rows=export(cache,path);write(dest/'far_pair_path.csv',rows);baseline=read(previous/f'case_{tag}/baseline/far_pair_path.csv');bk={key(r) for r in baseline};nk={key(r) for r in rows}
        result=dict(tag=tag,baseline_far_observations=len(baseline),far_observations=len(rows),removed_baseline_boxes=len(bk-nk),added_boxes=len(nk-bk),path_cost=search['best_path_cost'],same_output_as_baseline=bk==nk,candidate_ID_bindings=False);dump(dest/'result.json',result);reports.append(result);print(json.dumps(result),flush=True)
    assert all(sha(Path(n))==h for n,h in protected.items());dump(out/'result.json',dict(status='completed10_matched_same_match_development_cases',cases=reports,baseline_exact_replays_reused=10,protected_sources_unchanged=True,canonical_pipeline_replaced=False,independent_identity_accuracy_established=False))

if __name__=='__main__':main()
