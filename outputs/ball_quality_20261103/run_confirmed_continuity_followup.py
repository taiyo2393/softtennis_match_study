"""Use frozen cached observations for exploratory follow-up after guardB/C regressions."""
import csv,hashlib,json
from pathlib import Path
from collections import Counter
import numpy as np
from confirmed_continuity_guard import confirmed_class,adaptive,old
ROOT=Path(__file__).resolve().parent;BASE=ROOT.parent/'ball_quality_20261110';SOURCE=BASE/'automatic_reacquisition_trial_v2';OUT=BASE/'automatic_reacquisition_followup'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def write(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def main():
    OUT.mkdir();source=load(SOURCE/'protocol.json');assert load(SOURCE/'run_result.json')['baseline_exact_cases']==5
    protected={str(p):sha(p) for p in [SOURCE/'protocol.json',SOURCE/'run_result.json',ROOT/'confirmed_continuity_guard.py',Path(__file__)]}
    for plan in source['cases']:
        folder=SOURCE/f'case_{plan["tag"]}'
        for name in ['cache_manifest.json','appearance_cache.npz','cached_detections.csv','result.json','baseline/selected_four.csv']:protected[str(folder/name)]=sha(folder/name)
    dump(OUT/'protocol.json',dict(date_jst='2026-11-10',scope='Exploratory same-match follow-up after observing original guardB/C regressions; not prespecified independent evaluation.',reason='003 valid ordinary-continuity switch disappeared44 frames under stricter recovery-only route. Follow-up confirms ordinary plausible candidates for8 frames instead of imposing stricter recovery appearance gate.',mode='confirmed_normal_continuity_plus_seed_opponent_negatives',constants='Ordinary missing cost1.05, recovery_source_frames8, recovery_negative_margin0.06 retained. No numeric search.',manual_ID_binding=False,detector_reused=True,interpolation=False,future_frames_used_in_near_association=False,cache='Same exact appearance features and observations as five-case baseline replays.',control_review='AI images support same player for002 new78,003 new202,025 N2new84.025 N1new107 has only one accepted edge-of-image observation before absence;8-frame confirmation cannot retain it. Report that sacrifice rather than count it as a retained switch.',success_scope='Probe known011 error and three sustained controls. Do not claim all four changes retained, universal improvement or replacement of canonical pipeline.',protected_input_hashes=protected,source_cases=source['cases']))
    results=[]
    for plan in source['cases']:
        tag=plan['tag'];folder=SOURCE/f'case_{tag}';cache=load(folder/'cache_manifest.json');assert sha(folder/'appearance_cache.npz')==cache['cache_sha256'] and sha(folder/'cached_detections.csv')==cache['detections_sha256']
        raw=read(folder/'cached_detections.csv');z=np.load(folder/'appearance_cache.npz');array=z['features'];offset=z['frame_offsets'];base=adaptive.AdaptiveNearTracker if plan['adaptive_size'] else old.TemporalTracker;tracker=confirmed_class(base)();selected=[]
        for index,frame in enumerate(range(plan['first'],plan['last']+1)):
            rows=raw[offset[index]:offset[index+1]];features=[None if np.isnan(f).all() else f for f in array[offset[index]:offset[index+1]]]
            for i,j,margin,value,origin in tracker.step(frame,rows,features):selected.append(dict(rows[j],player_slot=old.a.SLOTS[i],position_source='detected_box',identity_status='automatic_development_candidate_not_GT',identity_source=origin,assignment_cost=round(value,5),alternative_cost_margin=round(margin,5)))
            if index%900==0:print(tag,'followup',index,flush=True)
        dest=OUT/f'case_{tag}';dest.mkdir();write(dest/'selected_four.csv',selected);dump(dest/'events.json',tracker.events)
        baseline=read(folder/'baseline/selected_four.csv');key=lambda r:tuple(str(r[k]) for k in ['source_frame','player_slot','track_id','x1','y1','x2','y2'])
        oldset={key(r) for r in baseline if r['player_slot'].startswith('near_')};newset={key(r) for r in selected if r['player_slot'].startswith('near_')};summary=dict(tag=tag,near_counts=dict(Counter(r['player_slot'] for r in selected if r['player_slot'].startswith('near_'))),removed_near_observations=len(oldset-newset),added_near_observations=len(newset-oldset),initialized_frame=tracker.initialized_frame,identity_binding_by_AI=False)
        dump(dest/'result.json',summary);results.append(summary);print(json.dumps(summary),flush=True)
    assert all(sha(Path(n))==h for n,h in protected.items());dump(OUT/'run_result.json',dict(status='completed_exploratory_followup',cases=results,protected_sources_unchanged=True,canonical_pipeline_changed=False,independent_accuracy_established=False))
if __name__=='__main__':main()
