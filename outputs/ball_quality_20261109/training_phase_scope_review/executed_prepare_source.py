"""Freeze all123 training times and methods for prediction-blind phase review."""
import json,shutil
from pathlib import Path
from collections import Counter
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2

OUT=ROOT.parent/'ball_quality_20261109/training_phase_scope_review'
OLD=ROOT.parent/'ball_quality_20261109'

def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')

def main():
    OUT.mkdir(parents=True,exist_ok=True);assert not (OUT/'protocol.json').exists(),'Preserve the review packet.'
    train_path=ROOT/'balltrack_expanded_centers_pilot/expanded_train_labels.csv';train=read(train_path)
    prior_path=OLD/'remaining_error_phase_review/phase_review_labels.csv';prior=read(prior_path)
    assert len(train)==123 and all(r['phase_status']=='ended' and r['reviewer_id']=='human_user' for r in prior)
    known={(r['candidate_id'],r['local_frame']):r for r in prior};assert len(known)==3
    methods={'current_temporal':ROOT/'ball_temporal_path_comparison/train_displacement_0.1.csv',
        'native_missing_only':OLD/'training_tile_detail_comparison/native_missing_only_verified_predictions.csv',
        'crossscale_agreement16':OLD/'training_tile_gate_grouped/crossscale_agreement16_predictions.csv',
        'logistic10_oof':OLD/'training_tile_gate_grouped/logistic10_oof_predictions.csv'}
    protected={str(p):sha(p) for p in [train_path,prior_path,OLD/'remaining_error_phase_review/completed_review.json',
        ROOT/'reviewed_ball_overrides.json',ROOT/'reviewed_person_overrides.json',ROOT/'reviewed_tracking_bundles.json',
        SOURCE/'annotations/balltrack_groundtruth_match1_pilot.csv',
        SOURCE/'third_party/RacketVision/source/BallTrack/checkpoints/balltrack_best.pth',
        ROOT.parent/'ball_quality_20261104/completed_review/manual_sequence_labels.csv',ROOT.parent/'ball_quality_20261104/evaluation_report.json',*methods.values()]}
    protocol=dict(status='fixed_before_any_new_phase_answers',date_jst='2026-11-09',
        sample='ALL123 predefined training times across8clips,independent of method success/failure; no point resampling.',
        annotation='Human play/nonplay/uncertain at target moments only; no ball clicks,precise contact/end boundaries,player IDs,or score-side questions.',
        known='Reuse3explicit ended answers as locked nonplay with detail ended;120new target phases pending.',
        interface='One video per clip,optional time interval helper applies play inside/nonplay outside to unconfirmed target rows; individual rows can be uncertain. All labels require explicit clip save. Interval helpers are NOT exact rally boundary GT.',
        blind='No method output,ball-center annotation,error category,visibility label,or predicted phase in public UI/API.',
        comparison='Freeze4existing methods and original ball GT now. After ALL phases reviewed,apply identical play filter to every method. Report nonplay/uncertain exclusions and per-clip denominators,visible center error16px,missing and output-on-unobservable counts. Retain original mixed-phase reports.',
        interpretation='Phase-conditional same-match development results,not automatic start/end accuracy or end-to-end video system performance. No retraining before this comparison.',
        adoption=False,new_detector_training=False,new_candidate_gate_training=False,protected_input_hashes=protected,script_sha256=sha(Path(__file__)))
    dump(OUT/'protocol.json',protocol);shutil.copy2(Path(__file__),OUT/'executed_prepare_source.py')
    shutil.copy2(train_path,OUT/'frozen_ball_groundtruth.csv')
    frozen=[]
    for name,path in methods.items():
        target=OUT/('frozen_'+name+'.csv');shutil.copy2(path,target)
        rs=read(target);assert len(rs)==123 and {r['sample_id'] for r in rs}=={r['sample_id'] for r in train}
        frozen.append(dict(method=name,path=str(target),sha256=sha(target)))
    clips=[];phases=[];source_images={};source_checks=[]
    for cid in sorted({r['candidate_id'] for r in train}):
        group=sorted([r for r in train if r['candidate_id']==cid],key=lambda r:int(r['local_frame']))
        base=SOURCE/'data/balltrack/racketvision/match1'/cid;mp=base/'runs/threshold030_median120/run_metadata.json';meta=load(mp)
        protected[str(mp)]=sha(mp);video=Path(meta['clip']);protected[str(video)]=sha(video)
        offset=int(group[0]['source_frame'])-int(group[0]['local_frame'])
        assert all(int(g['source_frame'])==offset+int(g['local_frame']) for g in group)
        target_by={int(g['local_frame']):g for g in group};cap=cv2.VideoCapture(str(video));assert cap.isOpened()
        for t in range(max(target_by)+1):
            ok,im=cap.read();assert ok
            if t in target_by:
                assert im.shape==(1080,1920,3)
                path=base/'frames'/f'{t:06d}.jpg';ok,encoded=cv2.imencode('.jpg',im,[cv2.IMWRITE_JPEG_QUALITY,95]);assert ok
                assert encoded.tobytes()==path.read_bytes(),(cid,t,'clock mismatch')
                source_images[str(path)]=sha(path);source_checks.append(dict(candidate_id=cid,local_frame=t,jpeg_matches_clip=True))
        cap.release()
        clips.append(dict(clip_id=cid[-3:],candidate_id=cid,fps=meta['fps'],frame_count=meta['frame_count'],
            duration_sec=meta['frame_count']/meta['fps'],source_frame_offset=offset,video_path=str(video),video_sha256=sha(video),
            sample_ids=[g['sample_id'] for g in group]))
        for g in group:
            key=cid,g['local_frame'];reused=known.get(key);image=base/'frames'/f'{int(g["local_frame"]):06d}.jpg'
            phases.append(dict(sample_id=g['sample_id'],clip_id=cid[-3:],candidate_id=cid,local_frame=g['local_frame'],source_frame=g['source_frame'],
                clip_time_sec=f'{int(g["local_frame"])/meta["fps"]:.9f}',phase_status='nonplay' if reused else 'unreviewed',
                phase_detail='ended' if reused else '',review_status='prior_human_confirmed' if reused else 'unreviewed',
                reviewer_id=reused['reviewer_id'] if reused else '',reviewed_at_utc=reused['reviewed_at_utc'] if reused else '',
                notes='以前の回答：終了後' if reused else '',evidence_origin=str(prior_path) if reused else '',
                locked=int(bool(reused))))
    assert len(phases)==123 and sum(r['locked'] for r in phases)==3
    write(OUT/'phase_labels.csv',phases);shutil.copy2(OUT/'phase_labels.csv',OUT/'initial_phase_labels.csv')
    write(OUT/'source_clip_checks.csv',source_checks);dump(OUT/'clips.json',clips);dump(OUT/'frozen_methods.json',frozen);dump(OUT/'source_image_hashes.json',source_images)
    assets={g['sample_id']:dict(path=str(SOURCE/'data/balltrack/racketvision/match1'/g['candidate_id']/'frames'/f'{int(g["local_frame"]):06d}.jpg'),sha256=source_images[str(SOURCE/'data/balltrack/racketvision/match1'/g['candidate_id']/'frames'/f'{int(g["local_frame"]):06d}.jpg')]) for g in train}
    dump(OUT/'image_assets.json',assets)
    assert all(sha(Path(p))==h for p,h in protected.items())
    manifest=dict(status='all_training_phase_review_ready',clips=8,total_targets=123,prior_confirmed=3,pending=120,
        initial_phase_sha256=sha(OUT/'initial_phase_labels.csv'),frozen_methods=frozen,protected_input_hashes=protected,
        immutable_artifact_hashes={str(OUT/n):sha(OUT/n) for n in ['protocol.json','executed_prepare_source.py','frozen_ball_groundtruth.csv',
            'initial_phase_labels.csv','clips.json','image_assets.json','source_image_hashes.json','source_clip_checks.csv','frozen_methods.json']},
        all123_jpegs_exactly_matching_source_clip=True,new_ball_annotations=0,automatic_phase_annotations=0)
    dump(OUT/'manifest.json',manifest);print(json.dumps({k:v for k,v in manifest.items() if k not in ['protected_input_hashes','immutable_artifact_hashes','frozen_methods']},indent=2))

if __name__=='__main__':main()
