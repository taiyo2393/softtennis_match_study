"""Matched ten-case development trial using identical detections and near paths."""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,write,sha,dump
from diagnose_initial_pair_appearance import cache as read_cache
from diagnose_far_identity_failures import OUT,load
import far_identity_guard as guard
far=guard.original
SLOTS=['far_rear','far_front']
MODES=['baseline','slot_near_guard','slot_near_upper_guard']

def case_plans():
    entries={e['tag']:e for e in load(BASE/'pair_initialization_development/diagnostic_protocol.json')['cases']};plans=[]
    for tag in ['002','005','006','009','012','015','016','017','018','019']:
        if tag=='019':
            c=BASE/f'cohort_tracking/case_{tag}';s=load(c/f'tracking_quality_20261102_stage4/players_{tag}_temporal/summary.json');e=dict(tag=tag,raw_csv=s['input_csv'],first=s['first_frame'],last=s['last_frame'],cache=str(BASE/f'pair_initialization_transfer_{tag}'));temporal=c/f'tracking_quality_20261102_stage4/players_{tag}_temporal';reference=c/f'tracking_quality_20261102_stage7/players_{tag}_global/far_pair_path.csv'
        else:
            e=dict(entries[tag]);temporal=BASE/f'pair_initialization_development/matched_trial/case_{tag}/pair_rank'
            if tag in ['002','005']:reference=ROOT.parent/f'tracking_quality_20261102_stage10/experiment/tracking_quality_20261102_stage7/players_{tag}_global/far_pair_path.csv'
            elif tag in ['009','015']:reference=BASE/f'court_initialization_trial/case_{tag}/full_court_seed_only/tracking_quality_20261102_stage7/players_{tag}_global/far_pair_path.csv'
            elif tag in ['012','018']:reference=BASE/f'cohort_tracking/case_{tag}/near_pair_development/tracking_quality_20261102_stage7/players_{tag}_global/far_pair_path.csv'
            elif tag=='017':reference=BASE/f'court_initialization_transfer_{tag}/court_seed_only/tracking_quality_20261102_stage7/players_{tag}_global/far_pair_path.csv'
            else:reference=BASE/f'cohort_tracking/case_{tag}/tracking_quality_20261102_stage7/players_{tag}_global/far_pair_path.csv'
        e.update(temporal_folder=str(temporal),reference_far_path=str(reference),role='known019_AI_failure_development' if tag=='019' else 'retained_same_match_regression_control_not_independent_GT');plans.append(e)
    return plans

def build_cache(e):
    raw,array,offset=read_cache(e);byframe=defaultdict(list);features={}
    for r,feature in zip(raw,array):byframe[int(r['source_frame'])].append(r);features[int(r['source_frame']),r['track_id']]=None if np.isnan(feature).all() else feature
    temp=Path(e['temporal_folder']);event=load(temp/'events.json')[0];near=defaultdict(list)
    for r in read(temp/'selected_four.csv'):
        if r['player_slot'].startswith('near_'):near[int(r['source_frame'])].append(r)
    anchors={slot:[features[f,str(tid)] for f in range(event['window_start'],event['frame']+1)] for slot,tid in event['seed_ids'].items()}
    chosen=[r for r in byframe[event['frame']] if int(r['track_id']) in event['seed_ids'].values()];negative=[features[event['frame'],r['track_id']] for r in byframe[event['frame']] if int(r['track_id']) not in event['seed_ids'].values() and all(far.t.overlap_fraction(r,s)<.1 for s in chosen)]
    segments,counts=far.tracklet_index(read(Path(e['raw_csv'])));cache=[];scores=[]
    for f in range(event['frame'],e['last']+1):
        items=[]
        for r in byframe[f]:
            x,y=far.a.foot(r)/[3840,2160]
            if not (.25<x<.75 and .52<y<.665):continue
            if any(far.t.overlap_fraction(r,n)>.55 for n in near[f]):continue
            feature=features[f,r['track_id']];ds=[min(far.a.app_distance(a,feature) for a in anchors[s]) for s in SLOTS]
            if min(ds)>far.P['appearance_max']:continue
            if negative and min(far.a.app_distance(a,feature) for a in negative)+.04<min(ds):continue
            key=segments[f,int(r['track_id'])];items.append(dict(row=r,appearance=ds,point=far.a.foot(r),height=far.a.height(r),tracklet_key=key,stable_tracklet=counts[key]>=far.P['stable_tracklet_min_observations']))
        if f==event['frame']:
            for slot in SLOTS:
                tid=event['seed_ids'][slot]
                if not any(int(x['row']['track_id'])==tid for x in items):
                    r=next(r for r in byframe[f] if int(r['track_id'])==tid);key=segments[f,tid];items.append(dict(row=r,appearance=[0.,0.],point=far.a.foot(r),height=far.a.height(r),tracklet_key=key,stable_tracklet=counts[key]>=far.P['stable_tracklet_min_observations']))
        for item in items:
            feature=features[f,item['row']['track_id']];allowed,d=guard.gate(feature,anchors,negative);upper,du=guard.gate(feature,anchors,negative,upper=True);item['guard_allowed']=allowed;item['upper_allowed']=upper
            for i,slot in enumerate(SLOTS):scores.append(dict(source_frame=f,track_id=item['row']['track_id'],player_slot=slot,slot_near_allowed=int(allowed[i]),slot_near_upper_allowed=int(upper[i]),own_distance=d['own'][i],near_distance=d['near'],own_upper=d['own_upper'][i],negative_upper=d['negative_upper'],slot_near_reason=d['reasons'][i],slot_near_upper_reason=du['reasons'][i]))
        cache.append(items)
    return event,near,cache,scores

def export(cache,path):
    return [dict(cache[f][j]['row'],player_slot=SLOTS[i],position_source='detected_box',identity_status='offline_path_candidate_requires_review',tracklet_key=cache[f][j]['tracklet_key'],stable_tracklet=int(cache[f][j]['stable_tracklet'])) for f,choices in enumerate(path) for i,j in enumerate(choices) if j>=0]

def main():
    out=OUT/'matched_trial';out.mkdir();plans=case_plans();protected={str(n):sha(n) for n in [Path(__file__),ROOT/'far_identity_guard.py',Path(far.__file__),Path(far.a.__file__),OUT/'diagnostic_protocol.json',OUT/'baseline019_far_appearance_diagnostic.csv',BASE/'pair_initialization_transfer_019/AI_failure_probes.csv']}
    for e in plans:
        for n in [Path(e['raw_csv']),Path(e['cache'])/'cached_detections.csv',Path(e['cache'])/'appearance_cache.npz',Path(e['temporal_folder'])/'selected_four.csv',Path(e['temporal_folder'])/'events.json',Path(e['reference_far_path'])]:protected[str(n)]=sha(n)
    for n in (BASE/'current22_player_inputs').iterdir():
        if n.is_file():protected[str(n)]=sha(n)
    dump(out/'protocol.json',dict(cases=plans,modes=MODES,policy_fixed_before_new_outputs=True,selection='Known019 failure plus002/005/006/009/012/015/016/017/018 retained controls, frozen before new guard outputs. Includes previous initial-person failures with corrected near paths. Not representative or independent accuracy sampling.',guard_parameters=guard.PARAMETERS,far_parameters=far.P,new_numeric_threshold_search=False,negative_margin_rationale='Retain old0.04 appearance-negative margin and0.70 appearance limit; apply per identity, add initial near-player negatives and optional upper-body comparison. No values fitted on probe correctness.',same_detections_and_near_paths=True,source_video_sha256=load(BASE/'cohort_tracking/case_019/protocol.json')['source_video_sha256'],near_paths_not_retracked=True,interpolation=False,canonical_pipeline_replaced=False,independent_GT_added=False,protected_input_hashes=protected))
    reports=[];fields=['source_frame','player_slot','track_id','x1','y1','x2','y2','bottom_center_x','bottom_center_y'];signature=lambda rows:[tuple(r[k] for k in fields) for r in rows]
    for e in plans:
        tag=e['tag'];dest=out/f'case_{tag}';dest.mkdir();event,near,cache,scores=build_cache(e);write(dest/'candidate_gate_diagnostic.csv',scores);dump(dest/'initialization.json',event);mode_reports={};base_keys=None
        for mode in MODES:
            current=cache if mode=='baseline' else [[dict(item,identity_allowed=item['guard_allowed' if mode=='slot_near_guard' else 'upper_allowed']) for item in items] for items in cache]
            path,search=far.solve(current,event) if mode=='baseline' else guard.solve(current,event);rows=export(current,path);folder=dest/mode;folder.mkdir();write(folder/'far_pair_path.csv',rows);keys={tuple(r[k] for k in fields) for r in rows}
            if mode=='baseline':
                assert signature(rows)==signature(read(Path(e['reference_far_path']))),tag;base_keys=keys
                if tag=='019':
                    replay,rs=guard.solve(cache,event);assert path==replay and search['best_path_cost']==rs['best_path_cost'];dump(dest/'copied_solver_baseline_verification.json',dict(status='passed',path_and_cost_exact_without_veto=True))
            else:
                for f,choices in enumerate(path):
                    for i,j in enumerate(choices):assert j<0 or current[f][j]['identity_allowed'][i]
            mode_reports[mode]=dict(far_observations=len(rows),removed_baseline_boxes=len(base_keys-keys),added_boxes=len(keys-base_keys),path_cost=search['best_path_cost'],same_output_as_baseline=keys==base_keys,candidate_ID_bindings=False)
            print(json.dumps(dict(tag=tag,mode=mode,**mode_reports[mode])),flush=True)
        report=dict(tag=tag,baseline_replay_exact=True,modes=mode_reports);dump(dest/'result.json',report);reports.append(report)
    assert all(sha(Path(n))==h for n,h in protected.items());dump(out/'result.json',dict(status='completed10_matched_same_match_development_cases',cases=reports,baseline_exact_replays=len(reports),protected_sources_unchanged=True,canonical_pipeline_replaced=False,independent_identity_accuracy_established=False))

if __name__=='__main__':main()
