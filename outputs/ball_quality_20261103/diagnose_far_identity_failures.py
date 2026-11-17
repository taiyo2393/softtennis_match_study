"""Measure existing candidate appearance gates on retained019 failure probes."""
import json,sys
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,write,sha,dump
sys.path.insert(0,str(ROOT.parent/'tracking_quality_20261102_stage10'))
import gap_pair_tracking as far

OUT=BASE/'far_identity_guard_development'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def distance(a,b,part=None):
    if a is None or b is None:return 1.
    return far.a.app_distance(a,b) if part is None else float(cv2.compareHist(a[part],b[part],cv2.HISTCMP_BHATTACHARYYA))

def main():
    OUT.mkdir();case=BASE/'cohort_tracking/case_019';cache=BASE/'pair_initialization_transfer_019';event=load(case/'tracking_quality_20261102_stage4/players_019_temporal/events.json')[0];raw=read(cache/'cached_detections.csv');z=np.load(cache/'appearance_cache.npz');features=z['features'];byframe=defaultdict(list);feature={}
    for r,f in zip(raw,features):
        ff=None if np.isnan(f).all() else f;byframe[int(r['source_frame'])].append(r);feature[int(r['source_frame']),r['track_id']]=ff
    anchors={slot:[feature[f,str(tid)] for f in range(event['window_start'],event['frame']+1)] for slot,tid in event['seed_ids'].items()}
    chosen=[r for r in byframe[event['frame']] if int(r['track_id']) in event['seed_ids'].values()]
    negatives=[feature[event['frame'],r['track_id']] for r in byframe[event['frame']] if int(r['track_id']) not in event['seed_ids'].values() and all(far.t.overlap_fraction(r,s)<.1 for s in chosen)]
    probes=read(cache/'AI_failure_probes.csv');probe_keys={(r['source_frame'],r['player_slot'],r['track_id']) for r in probes};files=[Path(__file__),Path(far.__file__),Path(far.a.__file__),cache/'AI_failure_probes.csv',cache/'cached_detections.csv',cache/'appearance_cache.npz',case/'four_player_path.csv',case/'tracking_quality_20261102_stage4/players_019_temporal/events.json'];protected={str(n):sha(n) for n in files}
    dump(OUT/'diagnostic_protocol.json',dict(candidate='019',purpose='Diagnosis before new guard outputs; no independent GT. Score all existing far observations, flag earlier AI-diagnosed probes separately.',current_far_parameters=far.P,existing_negative_margin=.04,anchor_window=[event['window_start'],event['frame']],scope='Same-match development; upper/lower clothing histograms are cues, not learned person ReID.',protected_input_hashes=protected))
    out=[]
    for row in read(case/'four_player_path.csv'):
        if not row['player_slot'].startswith('far_'):continue
        f=int(row['source_frame']);feat=feature[f,row['track_id']];own=anchors[row['player_slot']];near=anchors['near_front']+anchors['near_rear'];ds={slot:min(distance(a,feat) for a in anchors[slot]) for slot in ['far_rear','far_front']}
        r=dict(source_frame=f,player_slot=row['player_slot'],track_id=row['track_id'],AI_probe=int((row['source_frame'],row['player_slot'],row['track_id']) in probe_keys),own_far_distance=ds[row['player_slot']],other_far_distance=ds['far_front' if row['player_slot']=='far_rear' else 'far_rear'],initial_negative_distance=min((distance(a,feat) for a in negatives),default=1),near_anchor_distance=min(distance(a,feat) for a in near))
        for part,name in [(0,'upper'),(1,'lower')]:
            r[f'own_{name}']=min(distance(a,feat,part) for a in own);r[f'negative_{name}']=min((distance(a,feat,part) for a in negatives),default=1);r[f'near_{name}']=min(distance(a,feat,part) for a in near)
        out.append(r)
    write(OUT/'baseline019_far_appearance_diagnostic.csv',out);assert all(sha(Path(n))==h for n,h in protected.items());dump(OUT/'diagnostic_verification.json',dict(status='passed',far_rows=len(out),AI_probe_rows=sum(r['AI_probe'] for r in out),protected_sources_unchanged=True,identity_accuracy_established=False))
    for r in out:
        if r['AI_probe']:print(json.dumps(r),flush=True)

if __name__=='__main__':main()
