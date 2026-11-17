"""Save initialization overlays and distinguish box changes from cost metadata."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,sha,dump
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--kind',choices=['court','pair'],default='pair');args=ap.parse_args();tag=args.tag;assert len(tag)==3 and tag.isdigit()
    c=BASE/'cohort_tracking'/f'case_{tag}';t=BASE/f'{args.kind}_initialization_transfer_{tag}';out=t/'initialization_evidence';out.mkdir();p=load(c/'protocol.json');tp=load(t/'protocol.json');plan=load(Path(tp['frozen_plan']));result=load(t/'result.json');params=plan.get('court_parameters',plan.get('frozen_parameters'))
    h=np.array(load(Path(plan['calibration']))['image_to_court_homography']);ih=np.linalg.inv(h);raw=read(c/f'crop_runs/augmented_detections_{tag}.csv');byframe=defaultdict(list)
    for r in raw:byframe[int(r['source_frame'])].append(r)
    unique={};cap=cv2.VideoCapture(p['source_video']);manifest=[]
    for mode,r in result['modes'].items():
        if r['initialized_frame'] is None:continue
        signature=(r['initialized_frame'],tuple(r['seed_ids'].items()));unique.setdefault(signature,[]).append(mode)
    for index,((f,seeds),modes) in enumerate(unique.items(),1):
        cap.set(1,f);ok,im=cap.read();assert ok and round(cap.get(1))==f+1;seeds=dict(seeds)
        for coords,color in [([(0,0),(1,0),(1,1),(0,1)],(255,255,255)), ([(params['court_x_min'],params['court_y_min']),(params['court_x_max'],params['court_y_min']),(params['court_x_max'],params['court_y_max']),(params['court_x_min'],params['court_y_max'])],(0,170,255))]:
            poly=cv2.perspectiveTransform(np.array([coords],np.float64),ih)[0];cv2.polylines(im,[np.round(poly).astype(np.int32)],True,color,5)
        for slot,tid in seeds.items():
            r=next(r for r in byframe[f] if r['track_id']==str(tid));b=[round(float(r[k])) for k in ['x1','y1','x2','y2']];cv2.rectangle(im,tuple(b[:2]),tuple(b[2:]),(80,255,80),5);cv2.putText(im,f'{slot} raw{tid}',(b[0],max(40,b[1]-12)),0,1.2,(80,255,80),3)
        path=out/f'initialization_{index:02d}_F{f}.jpg';assert cv2.imwrite(str(path),cv2.resize(im,(1920,1080),interpolation=cv2.INTER_AREA));manifest.append(dict(source_frame=f,seed_ids=seeds,modes=modes,image_file=path.name,sha256=sha(path)))
    cap.release();dump(out/'manifest.json',dict(source_video_sha256=p['source_video_sha256'],calibration=plan['calibration'],calibration_sha256=sha(Path(plan['calibration'])),images=manifest,script_sha256=sha(Path(__file__)),independent_calibration_transfer_accuracy_established=False))
    cols=['source_frame','player_slot','track_id','x1','y1','x2','y2'];metadata=['identity_source','assignment_cost','alternative_cost_margin'];key=lambda r:tuple(r[k] for k in cols);base=read(t/'baseline/selected_four.csv');bm={key(r):r for r in base};report={}
    for mode in [mode for mode in result['modes'] if mode!='baseline']:
        path=t/mode/'selected_four.csv';rows=read(path) if path.exists() else [];rm={key(r):r for r in rows};common=bm.keys()&rm.keys();report[mode]=dict(temporal_box_ID_rows_equal=[key(r) for r in rows]==[key(r) for r in base],removed_temporal_box_ID_rows=len(bm.keys()-rm.keys()),added_temporal_box_ID_rows=len(rm.keys()-bm.keys()),metadata_differing_fields_on_matching_boxes={k:sum(bm[i][k]!=rm[i][k] for i in common) for k in metadata})
    dump(t/'temporal_metadata_diagnostic.json',report);print(json.dumps(dict(initialization_images=len(manifest),temporal_diagnostic=report),ensure_ascii=False))
if __name__=='__main__':main()
