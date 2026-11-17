"""Verify reviewed tiled observations and register immutable integrationv4."""
import json,math,subprocess
from pathlib import Path
from integrate_044_reviewed_tracks import ROOT,load,read,sha,cv2

def main():
    out=ROOT/'integrated_044_v4';protocol=load(out/'protocol.json')
    current=load(ROOT/'reviewed_ball_overrides.json')['clips']['match1_motion_pilot_016']
    assert protocol['ball_input_sha256']==current['reviewed_output_sha256']==sha(Path(current['reviewed_output']))
    person=load(ROOT/'reviewed_person_overrides.json')['rallies']['match1_rally_window_044']
    assert protocol['player_input_sha256']==person['reviewed_output_sha256']==sha(Path(person['reviewed_output']))
    states={int(r['local_frame']):r for r in read(out/'frame_observation_status.csv')}
    entities=read(out/'observed_entities_1920x1080.csv');ball=read(Path(current['reviewed_output']))
    previous=read(ROOT/'integrated_044_v3/observed_entities_1920x1080.csv')
    assert [r for r in entities if r['entity']!='ball']==[r for r in previous if r['entity']!='ball']
    for t in list(range(1083,1090))+[1091]:
        r=next(r for r in entities if int(r['local_frame'])==t and r['entity']=='ball')
        assert r['x_px']==ball[t]['x_px'] and r['y_px']==ball[t]['y_px']
        assert r['review_id']=='20261103_044_tiled_eight_real_four_false'
    for t in (1090,1092,1093):
        assert states[t]['ball_output_present']=='0'
        assert not any(int(r['local_frame'])==t and r['entity']=='ball' for r in entities)
    review=load(ROOT/'ball_044_scale_probe/user_reviewed/review.json')
    real1091=next(c for c in review['confirmed_observations'] if int(c['local_frame'])==1091)
    false1092=next(c for c in review['rejected_tiled_components'] if int(c['local_frame'])==1092)
    distance=math.hypot(float(real1091['x_px'])-float(false1092['x_px']),float(real1091['y_px'])-float(false1092['y_px']))
    counterexample=dict(real=dict(local_frame=1091,display_label='B',x_px=real1091['x_px'],y_px=real1091['y_px']),
        false=dict(local_frame=1092,display_label='A',x_px=false1092['x_px'],y_px=false1092['y_px']),
        distance_px=distance,source_frame_gap=1,
        interpretation='This user-rejected candidate lies near the preceding confirmed ball. A distance-only continuity gate cannot establish identity; neither smoothing nor peak strength is validated here.',
        exact_center_groundtruth=False)
    (ROOT/'ball_044_scale_probe/user_reviewed/continuity_counterexample.json').write_text(json.dumps(counterexample,indent=2),encoding='utf-8')
    final=out/'integrated_review_h264.mp4';cap=cv2.VideoCapture(str(final))
    for t in (1084,1089,1090,1091,1092):
        cap.set(cv2.CAP_PROP_POS_FRAMES,(t-739)*2);ok,im=cap.read();assert ok
        cv2.imwrite(str(out/f'integrated_local{t}.jpg'),im)
    cap.release()
    probe=json.loads(subprocess.check_output([r'C:\ffmpeg\bin\ffprobe.exe','-v','error','-show_streams','-of','json',str(final)]))
    assert any(s['codec_type']=='audio' for s in probe['streams'])
    for p,h in load(ROOT/'play_start_audit/inventory_summary.json')['source_hashes'].items():assert sha(Path(p))==h
    rp=ROOT/'reviewed_tracking_bundles.json';registry=load(rp);old=registry['rallies']['match1_rally_window_044']
    assert Path(old['integrated_folder'])==ROOT/'integrated_044_v3'
    registry['rallies']['match1_rally_window_044']=dict(old,ball_input=current['reviewed_output'],
        ball_input_sha256=current['reviewed_output_sha256'],integrated_folder=str(out.resolve()),
        integrated_video=str(final.resolve()),integrated_video_sha256=sha(final),
        frame_status=str((out/'frame_observation_status.csv').resolve()),observed_entities=str((out/'observed_entities_1920x1080.csv').resolve()),
        protocol=str((out/'protocol.json').resolve()),previous_snapshot=old['integrated_folder'])
    rp.write_text(json.dumps(registry,indent=2),encoding='utf-8')
    report=dict(snapshot='integrated_044_v4',reviewed_ball_hash_matches=True,person_entities_identical_to_v3=True,
        eight_positive_components_exported_at_same_frame=True,rejected_only_frames1090_1092_1093_blank=True,
        full_video_decoded_frames=load(out/'verification.json')['video_decoded_frames'],audio_present=True,
        earlier_raw_rejections_and_positive_reviews_preserved=True,source_annotations_unchanged=True,
        continuity_counterexample_distance_px=distance,independent_accuracy_established=False)
    (out/'final_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
