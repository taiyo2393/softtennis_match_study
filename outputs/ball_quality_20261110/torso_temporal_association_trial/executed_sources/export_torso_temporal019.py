"""Export a separate partial repair; never adopt it as a fully solved019 path."""
import json,subprocess
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,write,sha,dump
from build_rally_analysis_prototype import pair_features,summarize_prefix
from verify_player_prefix_expansion import calculate_frame,calculate_summary,close

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    c=BASE/'cohort_tracking/case_019';stage=BASE/'torso_temporal_association_trial';trial=stage;out=stage/'export019_torso_temporal';out.mkdir();plan=load(trial/'protocol.json');assert all(sha(Path(n))==h for n,h in plan['protected_input_hashes'].items());p=load(c/'protocol.json');near=[r for r in read(c/'tracking_quality_20261102_stage4/players_019_temporal/selected_four.csv') if r['player_slot'].startswith('near_')];far=read(trial/'case_019/torso_team_soft_cost/far_pair_path.csv');slots=['near_front','near_rear','far_rear','far_front'];rows=sorted(near+far,key=lambda r:(int(r['source_frame']),slots.index(r['player_slot'])));protected={str(n):sha(n) for n in [trial/'protocol.json',trial/'result.json',trial/'case_019/torso_team_soft_cost/far_pair_path.csv',c/'protocol.json',c/'four_player_path.csv',c/'tracking_quality_20261102_stage4/players_019_temporal/selected_four.csv',c/'crop_runs/augmented_detections_019.csv',Path(__file__)]}
    dump(out/'protocol.json',dict(p,experiment='Pooled team torso soft cost,019 partial development repair',known_or_ambiguous_mixed_boxes_remaining=True,processing_intervention='torso_temporal_soft_cost_development',AI_identity_binding_added=False,canonical_pipeline_changed=False,protected_input_hashes=protected))
    write(out/'four_player_path.csv',rows);people=defaultdict(dict);native={(r['source_frame'],r['track_id']):r for r in read(c/'crop_runs/augmented_detections_019.csv')}
    for r in rows:
        f=int(r['source_frame']);assert r['player_slot'] not in people[f];people[f][r['player_slot']]=r;raw=native[r['source_frame'],r['track_id']];assert all(raw[k]==r[k] for k in ['x1','y1','x2','y2','bottom_center_x','bottom_center_y','confidence'])
    for group in people.values():assert len({r['track_id'] for r in group.values()})==len(group)
    times=p['requested_input_frames'];features=[dict(rally_candidate_id=p['candidate_id'],relative_source_frame=f-times[0],**pair_features(people[f])) for f in times];snapshot,summary,_=summarize_prefix('019',[dict(r,ball_observation_present=0,ball_x_norm='',ball_y_norm='') for r in features]);checked=0;wanted=[calculate_frame(p['candidate_id'],people,f,f-times[0]) for f in times]
    for x,y in zip(features,wanted):
        for k,value in y.items():
            if k!='rally_candidate_id':close(x[k],value);checked+=1
    es,em=calculate_summary(wanted)
    for x,y in [(snapshot,es),(summary,em)]:
        for k,value in y.items():close(x[k],value);checked+=1
    for name,data in [('prefix_frame_features.csv',features),('snapshot_player_features.csv',[snapshot]),('prefix_player_summary_features.csv',[summary])]:write(out/name,data);assert read(out/name)==read(c/name)
    cap=cv2.VideoCapture(p['source_video']);cap.set(1,p['context_first_source_frame']);fps=60000/1001;temp=out/'review_mp4v.mp4';writer=cv2.VideoWriter(str(temp),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1920,1080));assert writer.isOpened();colors=[(80,255,80),(255,190,40),(255,80,240),(0,230,255)];names=['N1','N2','F1','F2']
    for f in range(p['context_first_source_frame'],p['context_last_source_frame']+1):
        ok,im=cap.read();assert ok and round(cap.get(1))==f+1;im=cv2.resize(im,(1920,1080))
        for slot,r in people[f].items():
            i=slots.index(slot);bb=[round(float(r[k])*.5) for k in ['x1','y1','x2','y2']];cv2.rectangle(im,tuple(bb[:2]),tuple(bb[2:]),colors[i],2);cv2.putText(im,f'{names[i]} / raw{r["track_id"]}',(bb[0],max(24,bb[1]-8)),0,.65,colors[i],2)
        cv2.rectangle(im,(0,1010),(1920,1080),(0,0,0),-1);cv2.putText(im,f'019 F{f} | experimental torso soft cost; mixed boxes remain',(15,1038),0,.62,(255,255,255),1);cv2.putText(im,'Rejected boxes become missing. No interpolation. Not independent identity accuracy.',(15,1070),0,.6,(255,255,255),1);writer.write(im)
    cap.release();writer.release();video=out/'review_h264.mp4';subprocess.run([r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-i',str(temp),'-an','-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],check=True)
    cap=cv2.VideoCapture(str(video));decoded=0
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(1080,1920);decoded+=1
    cap.release();assert decoded==930;assert all(sha(Path(n))==h for n,h in protected.items());report=dict(status='passed',verification_scope='Source-box/feature/video verification, not solved-person tracking',matched_native_observations=len(rows),recalculated_feature_values=checked,video_frames_decoded=decoded,requested_frames_unchanged=True,prefix_features_equal_original=True,four_candidate_requested_times=sum(len(people[f])==4 for f in times),requested_times=31,remaining_known_mixed_box_frames=[46511,46634],accepted_into_comparison_inputs=False,canonical_pipeline_changed=False,independent_identity_accuracy_established=False)
    dump(out/'verification.json',report);dump(out/'case_result.json',dict(status='partial_far_guard_development_export',artifact_hashes={n.name:sha(n) for n in out.glob('*.csv')}));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
