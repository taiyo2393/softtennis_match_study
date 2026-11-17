"""Compare fixed spatial/appearance veto on the same ten development cases."""
from pathlib import Path
from collections import defaultdict
import numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,write,sha,dump
from run_far_identity_guard_trial import build_cache,export,load
from diagnose_initial_pair_appearance import cache as read_cache
import far_spatial_identity_guard as guard

OUT=BASE/'far_spatial_guard_development'
MODES=['spatial_background','spatial_background_and_near']

def main():
    old=BASE/'far_identity_guard_development/matched_trial';previous=load(old/'protocol.json');assert load(old/'result.json')['baseline_exact_replays']==10;assert all(sha(Path(n))==h for n,h in previous['protected_input_hashes'].items());OUT.mkdir();protected=dict(previous['protected_input_hashes'])
    for n in [Path(__file__),ROOT/'far_spatial_identity_guard.py',ROOT/'run_far_identity_guard_trial.py',ROOT/'diagnose_initial_pair_appearance.py',old/'protocol.json',old/'result.json',BASE/'far_identity_guard_development/final_verification.json']+[n for n in old.glob('case_*/baseline/far_pair_path.csv')]:protected[str(n)]=sha(n)
    dump(OUT/'protocol.json',dict(cases=previous['cases'],modes=MODES,parameters=guard.PARAMETERS,far_parameters=previous['far_parameters'],policy_fixed_before_new_outputs=True,development_after_019_and_control_review=True,selection='Same ten retained development cases; no case substitution or parameter sweep.',rationale='Upper-color mismatch must overlap an initial nonplayer rectangle; near-color mismatch must overlap an observed or past-only short-gap exclusion rectangle. No broad positive-color cutoff and no stability-based blanket exemption.',threshold_rationale='Reuse0.04 appearance margin,0.55 overlap,8-frame recovery interval and0.10 height velocity cap. These are engineering choices, not calibrated probabilities.',fixed_camera_negative_rectangles_not_new_GT=True,no_predicted_coordinates_exported=True,baseline_outputs_reused_from_exact_replays=True,near_paths_unchanged=True,canonical_pipeline_replaced=False,independent_accuracy_established=False,protected_input_hashes=protected))
    fields=['source_frame','player_slot','track_id','x1','y1','x2','y2','bottom_center_x','bottom_center_y'];key=lambda r:tuple(r[k] for k in fields);reports=[]
    for e in previous['cases']:
        tag=e['tag'];dest=OUT/f'case_{tag}';dest.mkdir();event,near,cache,scores=build_cache(e);raw,array,offset=read_cache(e);features={(int(r['source_frame']),r['track_id']):(None if np.isnan(f).all() else f) for r,f in zip(raw,array)};byframe=defaultdict(list)
        for r in raw:byframe[int(r['source_frame'])].append(r)
        anchors={slot:[features[f,str(tid)] for f in range(event['window_start'],event['frame']+1)] for slot,tid in event['seed_ids'].items()};chosen=[r for r in byframe[event['frame']] if int(r['track_id']) in event['seed_ids'].values()]
        negatives=[dict(row=r,feature=features[event['frame'],r['track_id']]) for r in byframe[event['frame']] if int(r['track_id']) not in event['seed_ids'].values() and all(guard.original.t.overlap_fraction(r,s)<.1 for s in chosen)]
        history={};decisions=[];regions_record=[]
        for index,items in enumerate(cache):
            f=event['frame']+index;regions=guard.near_regions(f,near[f],history)
            for region in regions:
                if region['age']:regions_record.append(dict(source_frame=f,near_slot=region['slot'],source_observation_frame=int(region['row']['source_frame']),age=region['age'],kind=region['kind'],**{k:region['row'][k] for k in ['x1','y1','x2','y2']}))
            for item in items:
                row=item['row'];feature=features[f,row['track_id']];bg,bd=guard.gate(row,feature,anchors,negatives,regions,include_near=False);both,dd=guard.gate(row,feature,anchors,negatives,regions)
                item['background_allowed']=bg;item['spatial_allowed']=both
                for i,slot in enumerate(['far_rear','far_front']):decisions.append(dict(source_frame=f,track_id=row['track_id'],player_slot=slot,background_allowed=int(bg[i]),spatial_allowed=int(both[i]),own_distance=dd[i]['own_distance'],own_upper=dd[i]['own_upper'],background_witnesses=str(dd[i]['background_witnesses']),near_witnesses=str(dd[i]['near_witnesses'])))
        write(dest/'candidate_gate_diagnostic.csv',decisions)
        if regions_record:write(dest/'predicted_exclusion_regions_not_output_positions.csv',regions_record)
        baseline=read(old/f'case_{tag}/baseline/far_pair_path.csv');bk={key(r) for r in baseline};mode_reports={}
        for mode in MODES:
            current=[[dict(item,identity_allowed=item['background_allowed' if mode=='spatial_background' else 'spatial_allowed']) for item in items] for items in cache];path,search=guard.solve(current,event)
            for f,choices in enumerate(path):
                for i,j in enumerate(choices):assert j<0 or current[f][j]['identity_allowed'][i]
            rows=export(current,path);folder=dest/mode;folder.mkdir();write(folder/'far_pair_path.csv',rows);nk={key(r) for r in rows};mode_reports[mode]=dict(far_observations=len(rows),removed_baseline_boxes=len(bk-nk),added_boxes=len(nk-bk),same_output_as_baseline=nk==bk,path_cost=search['best_path_cost']);print(dict(tag=tag,mode=mode,**mode_reports[mode]),flush=True)
        report=dict(tag=tag,modes=mode_reports,predicted_exclusion_region_rows=len(regions_record));dump(dest/'result.json',report);reports.append(report)
    assert all(sha(Path(n))==h for n,h in protected.items());dump(OUT/'result.json',dict(status='completed10_matched_same_match_development_cases',cases=reports,baseline_exact_replays_reused=10,protected_sources_unchanged=True,canonical_pipeline_replaced=False,independent_identity_accuracy_established=False))

if __name__=='__main__':main()
