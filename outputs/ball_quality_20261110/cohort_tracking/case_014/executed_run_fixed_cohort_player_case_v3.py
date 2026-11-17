"""Retained numerical player pipeline with explicit inherited contact-bracket inputs.

Version3 accepts a separately frozen AI contact proposal for previously unanchored cases; numerical tracking algorithms are retained.
"""
import argparse,csv,hashlib,json,math,os,sys
from collections import Counter,defaultdict
from pathlib import Path
import cv2
import numpy as np

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'ball_quality_20261110'
SOURCE=Path(r'C:\Users\ta239\Documents\Codex\2026-09-07\ai-2d-2d-1-2-0\softtennis_frontcourt_study')
FPS=60000/1001
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def write(p,rows):
    assert rows
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);ap.add_argument('--anchor-file');args=ap.parse_args();tag=args.tag
    assert tag.isdigit() and len(tag)==3
    queuepath=BASE/'cohort_and_baselines/fixed30_processing_queue.csv'
    queue=read(queuepath);entry=next(r for r in queue if r['candidate_id'].endswith('_'+tag))
    assert entry['existing_fixed_window_player_features']=='0','Retain existing reviewed feature sources.'
    handoff=ROOT.parent/'frontcourt_handoff_20260927'
    labels={r['candidate_id']:r for r in read(handoff/'rally_label_review.csv')}
    label=labels[entry['candidate_id']]
    bracket_path=handoff/'six_service_contact_brackets.csv'
    brackets={r['candidate_id']:r for r in read(bracket_path)}
    anchor_path=Path(args.anchor_file).resolve() if args.anchor_file else None
    new_anchor=None
    if entry['candidate_id'] in brackets:
        b=brackets[entry['candidate_id']]
        contact_first=int(b['contact_candidate_first_frame']);contact_last=int(b['contact_candidate_last_frame'])
        contact_status=b['review_status']+';'+b['effective_serve_status'];contact_evidence=b['evidence']
    elif label['reviewed_serve_impact_sec']:
        contact_first=contact_last=round(float(label['reviewed_serve_impact_sec'])*FPS)
        contact_status='legacy_record_precision_unknown_not_new_GT';contact_evidence='rally_label_review.csv:reviewed_serve_impact_sec'
    elif anchor_path is not None:
        new_anchor=json.loads(anchor_path.read_text(encoding='utf-8'))
        assert new_anchor['candidate_id']==entry['candidate_id'] and new_anchor['source_clock']=='legacy_edited_match1'
        assert new_anchor['fixed_before_new_person_tracking'] is True
        assert all(sha(Path(n))==h for n,h in new_anchor['evidence_hashes'].items())
        contact_first=int(new_anchor['contact_candidate_first_frame']);contact_last=int(new_anchor['contact_candidate_last_frame'])
        assert contact_first<=contact_last
        contact_status=new_anchor['review_status']+';'+new_anchor['effective_serve_status'];contact_evidence=new_anchor['evidence']
    else:
        raise RuntimeError('No inherited contact or frozen AI proposal; preserve pending case.')
    t0=contact_last+30;times=list(range(t0,t0+61,2))
    dest=BASE/'cohort_tracking'/('case_'+tag);dest.mkdir(parents=True,exist_ok=True)
    assert not (dest/'protocol.json').exists(),'Keep attempted runs, including failures.'
    first=round(float(entry['candidate_start_sec'])*FPS);last=math.ceil(float(entry['candidate_end_sec'])*FPS)-1
    assert first<=times[0]<=times[-1]<=last, 'Fixed input exceeds inherited candidate context; do not silently truncate or shift.'
    os.environ['YOLO_CONFIG_DIR']=str(dest/'ultralytics_config')
    for name in ['tracking_quality_20261102_stage9','tracking_quality_20261102_stage4','tracking_quality_20261102_stage10']:
        sys.path.insert(0,str(ROOT.parent/name))
    import adaptive_near as adaptive
    import run_temporal_experiment as temporal
    import gap_pair_tracking as far
    from ultralytics import YOLO
    import ultralytics
    import torch
    a=adaptive.old.a;previous=adaptive.old.previous
    assert torch.cuda.is_available(),'Retained crop implementation requires device0; stop rather than silently change implementation.'
    from importlib import util
    trackerconfig=Path(ultralytics.__file__).parent/'cfg/trackers/bytetrack.yaml'
    video=SOURCE/'data/raw/match1.mp4';weight=SOURCE/'models/yolo11m.pt'
    code=[Path(m.__file__) for m in [adaptive,adaptive.old,temporal,far,previous,a]]
    code += [SOURCE/'scripts/96_select_four_players_pilot.py',Path(__file__),trackerconfig,weight,queuepath,
        BASE/'cohort_and_baselines/protocol.json',bracket_path,handoff/'rally_label_review.csv',
        ROOT/'run_fixed_cohort_player_case.py',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json',ROOT/'reviewed_ball_overrides.json']
    code += [ROOT/'run_fixed_cohort_player_case_v2.py']
    if anchor_path is not None:code.append(anchor_path)
    protected={str(p):sha(p) for p in code};sourcehash=sha(video)
    if new_anchor is not None:assert sourcehash==new_anchor['source_video_sha256']
    assert sourcehash=='547e5cfd2ed6b0693a83201b1241ba6eb93ab7be8f3f2b135dee2cc48f94258d'
    dump(dest/'protocol.json',dict(date_jst='2026-11-10',candidate_id=entry['candidate_id'],
        selection='Chronological first pending cohort case; retained even if initialization or tracking fails.',
        context_first_source_frame=first,context_last_source_frame=last,
        contact_candidate_first_source_frame=contact_first,contact_candidate_last_source_frame=contact_last,
        contact_status=contact_status,contact_evidence=contact_evidence,requested_input_frames=times,
        new_AI_contact_proposal_used=new_anchor is not None,contact_proposal_file=str(anchor_path) if anchor_path else None,
        phase_warning='Fixed prefix is not certified in-play until chronological image audit; no future-outcome model is fitted.',
        context_warning='Inherited candidate context, not independently confirmed full-rally event bounds; may include pre/post-play frames. Only fixed early window will be feature input.',
        source_video=str(video),source_video_sha256=sourcehash,detector='Existing YOLO11m/ByteTrack:person,conf0.25,imgsz1280,device0',
        far_crop='Retained stage2 crop[.18,.43,.82,.68],imgsz1280; same duplicate/border rules',
        near='Retained stage9 adaptive temporal initialization/reacquisition',far='Retained stage10 offline pair search with short-gap tracklet continuity',
        interpolation=False,algorithm_parameters_changed=False,
        near_parameters=dict(adaptive.old.PARAMETERS,near_size_adaptation=adaptive.PARAMETERS),far_parameters=far.P,
        evaluation='Algorithm transfer/development on already known match; output availability is not identity accuracy. No new independent human GT.',
        protected_input_hashes=protected))
    rawfolder=dest/'raw_input'/('full_rally_'+tag);rawfolder.mkdir(parents=True)
    rawpath=rawfolder/'bytetrack.csv'
    cap=cv2.VideoCapture(str(video));assert cap.isOpened();cap.set(1,first)
    model=YOLO(str(weight));rows=[];roi=np.float32([[.02,.90],[.98,.90],[.68,.51],[.32,.51]])
    for f in range(first,last+1):
        ok,im=cap.read();assert ok and int(round(cap.get(1)))==f+1
        prediction=model.track(im,persist=True,classes=[0],conf=.25,imgsz=1280,tracker=str(trackerconfig),device=0,verbose=False)[0]
        for box in prediction.boxes:
            x1,y1,x2,y2=box.xyxy[0].cpu().tolist();x,y=(x1+x2)/2,y2
            rows.append(dict(source_frame=f,timestamp_sec=f/FPS,track_id=int(box.id.item()) if box.id is not None else -1,
                x1=x1,y1=y1,x2=x2,y2=y2,bottom_center_x=x,bottom_center_y=y,confidence=float(box.conf.item()),
                in_initial_roi=int(cv2.pointPolygonTest(roi,(x/3840,y/2160),False)>=0)))
        if (f-first)%120==0:print(tag,'full-frame detection',f-first,'/',last-first+1,flush=True)
    cap.release();write(rawpath,rows);del model
    dump(rawfolder/'four_player_selection_summary.json',[dict(seed_frame=first,analysis_ready_end_frame=last,
        warning='Legacy helper field names; context bounds only, no manually selected seed.')])
    a.BASE=dest/'raw_input';previous.OUT=dest/'crop_runs';previous.OUT.mkdir()
    augmented=previous.augment(tag)
    temporal.OUT=dest/'tracking_quality_20261102_stage4';temporal.OUT.mkdir()
    temporal.TemporalTracker=adaptive.AdaptiveNearTracker
    temporal.PARAMETERS=dict(adaptive.old.PARAMETERS,near_size_adaptation=adaptive.PARAMETERS)
    interval=dict(tag=tag,first_frame=first,last_frame=last,frames=last-first+1)
    summary=temporal.process(interval,augmented)
    if summary['initialized_frame'] is None:
        dump(dest/'case_result.json',dict(status='no_four_player_initialization',preserved_in_cohort=True,
            frames=interval['frames'],algorithm_parameters_changed=False,identity_accuracy_established=False))
        print('No initialization; retain case and missing data.',flush=True);return
    far.OUT=dest/'tracking_quality_20261102_stage7';far.OUT.mkdir()
    oldargs=sys.argv;sys.argv=[str(far.__file__),'--tag',tag]
    try:far.main()
    finally:sys.argv=oldargs
    nearrows=[r for r in read(temporal.OUT/f'players_{tag}_temporal/selected_four.csv') if r['player_slot'].startswith('near_')]
    farrows=read(far.OUT/f'players_{tag}_global/far_pair_path.csv')
    combined=[]
    for branch,rs in [('adaptive_near',nearrows),('offline_short_gap_far',farrows)]:
        combined.extend(dict(r,processing_branch=branch) for r in rs)
    combined.sort(key=lambda r:(int(r['source_frame']),a.SLOTS.index(r['player_slot'])))
    byframe=defaultdict(list)
    for r in combined:byframe[int(r['source_frame'])].append(r)
    for f,rs in byframe.items():
        assert len({r['player_slot'] for r in rs})==len(rs) and len({r['track_id'] for r in rs})==len(rs),(tag,f)
    write(dest/'four_player_path.csv',combined)
    # Fixed input range was resolved and recorded before detection; never use initialization as service time.
    from build_rally_analysis_prototype import pair_features,summarize_prefix
    feature_rows=[]
    for f in times:
        people={r['player_slot']:r for r in byframe[f]}
        feature_rows.append(dict(rally_candidate_id=entry['candidate_id'],relative_source_frame=f-t0,**pair_features(people)))
    write(dest/'prefix_frame_features.csv',feature_rows)
    snapshot,summ,_=summarize_prefix(tag,[dict(r,ball_observation_present=0,ball_x_norm='',ball_y_norm='') for r in feature_rows])
    write(dest/'snapshot_player_features.csv',[snapshot]);write(dest/'prefix_player_summary_features.csv',[summ])
    assert all(sha(Path(p))==h for p,h in protected.items())
    report=dict(status='new_player_candidate_path_and_prefix_constructed',candidate_id=entry['candidate_id'],
        context_frames=last-first+1,initialized_frame=summary['initialized_frame'],requested_input_frames=times,
        contact_candidate_first_source_frame=contact_first,contact_candidate_last_source_frame=contact_last,
        contact_status=contact_status,contact_evidence=contact_evidence,phase_audit_status='pending',
        requested_times=len(times),input_times_with_four_observations=sum(len(byframe[f])==4 for f in times),
        full_context_four_observation_frames=sum(len(byframe[f])==4 for f in range(first,last+1)),
        selected_slot_counts=dict(Counter(r['player_slot'] for r in combined)),
        identity_accuracy_established=False,independent_position_accuracy_established=False,
        new_human_annotation_requested=False,algorithm_parameters_changed=False,old_sources_unchanged=True,
        artifact_hashes={p.name:sha(p) for p in dest.glob('*.csv')})
    dump(dest/'case_result.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
