"""Preserve failed automatic009; bind its actual server's observed raw track by AI seed."""
import csv,hashlib,json,shutil
from collections import defaultdict,Counter
from pathlib import Path
import cv2
import numpy as np
from build_rally_analysis_prototype import pair_features,summarize_prefix
from verify_player_prefix_expansion import calculate_frame,calculate_summary,close

ROOT=Path(__file__).resolve().parent;CASE=ROOT.parent/'ball_quality_20261110/cohort_tracking/case_009'
OUT=CASE/'AI_seed_binding';SLOTS=['near_front','near_rear','far_rear','far_front'];NAMES=['N1','N2','F1','F2'];COLORS=[(80,255,80),(255,190,40),(255,80,240),(0,230,255)]
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def write(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)

def main():
    OUT.mkdir();p=load(CASE/'protocol.json');original=load(CASE/'case_result.json');cid=p['candidate_id']
    seed_frame=p['contact_candidate_last_source_frame']-30;assert seed_frame==21855
    first=seed_frame-7;last=p['context_last_source_frame'];raw_id='148'
    raw_path=CASE/'crop_runs/augmented_detections_009.csv';raw=read(raw_path);baseline=read(CASE/'four_player_path.csv')
    source_rows=[r for r in raw if r['track_id']==raw_id and first<=int(r['source_frame'])<=last]
    assert len(source_rows)==last-first+1==300 and {int(r['source_frame']) for r in source_rows}==set(range(first,last+1))
    assert all(r['candidate_origin']=='original_full_frame' for r in source_rows)
    assert all(r['track_id']!=raw_id for r in baseline if first<=int(r['source_frame'])<=last)
    input_files=[CASE/'protocol.json',CASE/'case_result.json',CASE/'four_player_path.csv',raw_path,CASE/'verification.json',Path(__file__)]
    protected={str(n):sha(n) for n in input_files}
    cap=cv2.VideoCapture(p['source_video']);cap.set(1,seed_frame);ok,im=cap.read();assert ok
    assert cv2.imwrite(str(OUT/'seed_F21855_native.jpg'),cv2.resize(im,(1920,1080),interpolation=cv2.INTER_AREA),[cv2.IMWRITE_JPEG_QUALITY,95])
    seed=next(r for r in source_rows if int(r['source_frame'])==seed_frame)
    dump(OUT/'seed_design.json',dict(candidate_id=cid,selection='Repair known N2 non-player binding only; seed image is existing QA time(contact nominal minus30), not a new best-visibility feature time.',
        seed_source_frame=seed_frame,target_slot='near_rear',native_detector_raw_id=raw_id,source_native_frame_sha256=sha(OUT/'seed_F21855_native.jpg'),
        seed_box={k:seed[k] for k in ['x1','y1','x2','y2']},export_scope=[first,last],
        source='Codex AI image correspondence: actual near-right server, not the left-edge non-player.',
        human_identity_GT=False,independent_accuracy_evaluation=False,interpolation=False,
        valid_only_for_this_existing_raw_track_and_scope=True,fixed_input_frames_unchanged=True))
    q=dict(p);q.update(context_first_source_frame=first,original_context_first_source_frame=p['context_first_source_frame'],
        original_context_last_source_frame=last,derived_variant='AI_seed_bound_native_observations',
        source_of_assignment='AI selected raw148 atF21855; only observed boxes of that track in declared scope are bound to N2.',
        full_automatic_baseline_valid=False,fully_automatic_result=False,AI_identity_binding_added=True,
        detector_or_association_numeric_parameters_changed=False,
        protected_input_hashes=protected,original_automatic_protocol=str(CASE/'protocol.json'),
        caveat='Case-specific assisted repair. Other slots frozen from baseline; no claim of a new validated automatic tracker. Fixed prefix unchanged.')
    dump(OUT/'protocol.json',q)
    kept=[r for r in baseline if first<=int(r['source_frame'])<=last and r['player_slot']!='near_rear']
    repaired=[dict(r,player_slot='near_rear',position_source='detected_box',identity_status='AI_seed_bound_candidate_not_independent_GT',
        identity_source='AI_seed_F21855_raw148_continuity',identity_review_status='AI_image_review_only_not_human_GT',
        processing_branch='AI_seed_bound_native_observation',assistance_source='seed_design.json') for r in source_rows]
    combined=sorted(kept+repaired,key=lambda r:(int(r['source_frame']),SLOTS.index(r['player_slot'])))
    write(OUT/'four_player_path.csv',combined);people=defaultdict(dict)
    for r in combined:
        f=int(r['source_frame']);assert r['player_slot'] not in people[f];people[f][r['player_slot']]=r
    for ps in people.values():assert len({r['track_id'] for r in ps.values()})==len(ps)
    times=p['requested_input_frames'];fs=[dict(rally_candidate_id=cid,relative_source_frame=f-times[0],**pair_features(people[f])) for f in times]
    write(OUT/'prefix_frame_features.csv',fs)
    snap,summary,_=summarize_prefix('009',[dict(r,ball_observation_present=0,ball_x_norm='',ball_y_norm='') for r in fs])
    write(OUT/'snapshot_player_features.csv',[snap]);write(OUT/'prefix_player_summary_features.csv',[summary])
    # Independent arithmetic, and exact preservation of unaffected observations.
    expected=[calculate_frame(cid,people,f,f-times[0]) for f in times];checked=0
    for a,b in zip(fs,expected):
        for k,v in b.items():
            if k!='rally_candidate_id':close(a[k],v);checked+=1
    es,em=calculate_summary(expected)
    for actual,calc in [(snap,es),(summary,em)]:
        for k,v in calc.items():close(actual[k],v);checked+=1
    assert [r for r in combined if r['player_slot']!='near_rear']==sorted(kept,key=lambda r:(int(r['source_frame']),SLOTS.index(r['player_slot'])))
    rawmap={(int(r['source_frame']),r['track_id']):r for r in raw}
    for r in combined:
        native=rawmap[int(r['source_frame']),r['track_id']]
        assert all(r[k]==native[k] for k in ['x1','y1','x2','y2','bottom_center_x','bottom_center_y','confidence'])
    views=[seed_frame,21885,21915,21945,21975,22130];cells=[]
    preview=OUT/'assisted_intermediate.mp4';writer=cv2.VideoWriter(str(preview),cv2.VideoWriter_fourcc(*'mp4v'),60000/1001,(1920,1080));assert writer.isOpened()
    cap.set(1,first)
    for f in range(first,last+1):
        ok,im=cap.read();assert ok and int(round(cap.get(1)))==f+1
        for i,s in enumerate(SLOTS):
            row=people[f].get(s)
            if row is None:continue
            box=[round(float(row[k])) for k in ['x1','y1','x2','y2']];cv2.rectangle(im,tuple(box[:2]),tuple(box[2:]),COLORS[i],5)
            cv2.putText(im,f'{NAMES[i]} raw{row["track_id"]}',(box[0],max(35,box[1]-12)),0,1.3,COLORS[i],3,cv2.LINE_AA)
        image=cv2.resize(im,(1920,1080),interpolation=cv2.INTER_AREA)
        cv2.putText(image,f'009 F{f} | AI-seeded N2 correction; not automatic-ID accuracy',(15,1055),0,.72,(255,255,255),2,cv2.LINE_AA);writer.write(image)
        if f in views:
            h=np.zeros((32,640,3),np.uint8);cv2.putText(h,f'009 AI-seeded F{f} | N2 raw148',(8,24),0,.65,(255,255,255),1,cv2.LINE_AA)
            cells.append(np.vstack([h,cv2.resize(image,(640,360),interpolation=cv2.INTER_AREA)]))
    writer.release();cap.release();assert len(cells)==6
    assert cv2.imwrite(str(OUT/'candidate_sequence_grid.jpg'),np.vstack([np.hstack(cells[:3]),np.hstack(cells[3:])]))
    import subprocess
    video=OUT/'AI_seeded_review_h264.mp4'
    subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-v','error','-i',str(preview),'-an','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
    c=cv2.VideoCapture(str(video));n=0
    while True:
        ok,im=c.read()
        if not ok:break
        n+=1
    c.release();assert n==300
    assert all(sha(Path(n))==h for n,h in protected.items())
    cov=load(CASE/'pre_outcome_covariates.json');cov['evidence_images']=[x for x in cov['evidence_images'] if x['source_frame']>=first]
    assert cov['evidence_images'];dump(OUT/'pre_outcome_covariates.json',cov)
    dump(OUT/'case_result.json',dict(status='AI_seed_bound_path_and_prefix_constructed',candidate_id=cid,requested_input_frames=times,
        requested_times=31,input_times_with_four_observations=sum(len(people[f])==4 for f in times),
        repair_scope=[first,last],original_automatic_N2_rejected_as_non_player=True,
        AI_seed_binding_added=True,fully_automatic_result=False,independent_identity_or_position_accuracy_established=False,
        source_other_slots_unchanged=True,interpolation=False,
        artifact_hashes={n.name:sha(n) for n in OUT.glob('*.csv')}))
    dump(OUT/'verification.json',dict(status='passed',matched_native_boxes=len(combined),recalculated_values=checked,
        unaffected_observations_exactly_retained=len(kept),replacement_native_observations=len(repaired),
        duplicate_raw_or_slot_assignments=False,fixed_input_frames_unchanged=True,original_baseline_and_sources_unchanged=True,
        assisted_video_frames_decoded=n,AI_seed_binding_added=True,independent_tracking_accuracy_established=False))
    print(json.dumps(load(OUT/'case_result.json'),ensure_ascii=False))

if __name__=='__main__':main()
