"""Reproduce candidate prefilters at the four missing prefix times; no path changes."""
import sys,json,hashlib
from pathlib import Path
from collections import defaultdict
import cv2

ROOT=Path(__file__).resolve().parent
CASE=ROOT.parent/'ball_quality_20261110/cohort_tracking/case_010'
sys.path.insert(0,str(ROOT.parent/'tracking_quality_20261102_stage10'))
import gap_pair_tracking as far
a=far.a;t=far.t
def main():
    event=json.loads((CASE/'tracking_quality_20261102_stage4/players_010_temporal/events.json').read_text(encoding='utf-8'))[0]
    raw=a.read(CASE/'crop_runs/augmented_detections_010.csv');byframe=defaultdict(list);near=defaultdict(list)
    for r in raw:
        if int(r['track_id'])>=0 and float(r['confidence'])>=.25:byframe[int(r['source_frame'])].append(r)
    for r in a.read(CASE/'four_player_path.csv'):
        if r['player_slot'].startswith('near_'):near[int(r['source_frame'])].append(r)
    protocol=json.loads((CASE/'protocol.json').read_text(encoding='utf-8'));cap=cv2.VideoCapture(protocol['source_video']);anchors={};negatives=[]
    for f in range(event['window_start'],event['frame']+1):
        cap.set(1,f);ok,im=cap.read();assert ok and round(cap.get(1))==f+1;im=cv2.resize(im,(1920,1080))
        for s in ['far_rear','far_front']:
            r=next(r for r in byframe[f] if int(r['track_id'])==event['seed_ids'][s]);anchors.setdefault(s,[]).append(a.appearance(im,r))
        if f==event['frame']:
            chosen=[r for r in byframe[f] if int(r['track_id']) in event['seed_ids'].values()]
            negatives=[a.appearance(im,r) for r in byframe[f] if int(r['track_id']) not in event['seed_ids'].values() and all(t.overlap_fraction(r,s)<.1 for s in chosen)]
    findings=[]
    for f in [23768,23770,23772,23774,23776,23778]:
        r=next(r for r in byframe[f] if r['track_id']=='41');cap.set(1,f);ok,im=cap.read();assert ok and round(cap.get(1))==f+1;im=cv2.resize(im,(1920,1080));feat=a.appearance(im,r)
        ds=[min(a.app_distance(x,feat) for x in anchors[s]) for s in ['far_rear','far_front']];negative=min([a.app_distance(x,feat) for x in negatives],default=None)
        x,y=a.foot(r)/[3840,2160];overlap=max([t.overlap_fraction(r,n) for n in near[f]],default=0)
        gates=dict(position_roi=bool(.25<x<.75 and .52<y<.665),near_overlap=bool(overlap<=.55),appearance=bool(min(ds)<=far.P['appearance_max']),negative_appearance=negative is None or bool(negative+.04>=min(ds)))
        findings.append(dict(source_frame=f,raw_id=41,native_detection_exists=True,appearance_distances=[float(x) for x in ds],nearest_negative_distance=None if negative is None else float(negative),max_near_overlap=float(overlap),prefilter_passes=gates,all_prefilters_pass=all(gates.values()),final_path_selected=f in [23768,23778]))
    cap.release()
    dest=CASE/'input_gap_prefilter_diagnosis.json';assert not dest.exists()
    dest.write_text(json.dumps(dict(findings=findings,interpretation='Candidates passing filters may still be omitted by the approximate joint beam search. This diagnosis does not establish a counterfactual optimal path, identity GT or parameter correction.',paths_changed=False,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    print(json.dumps(findings))
if __name__=='__main__':main()
