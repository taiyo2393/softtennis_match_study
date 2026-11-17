"""Prepare four BLINDED validation-only phase/center queries with locked scores.

Initial sustained-run sampling found zero eligible queries. Revise query design,
not any model/scorer, using unlabelled inventory only; preserve the first protocol.
"""
import json,sys,subprocess
from pathlib import Path
import numpy as np
import torch
from integrate_044_reviewed_tracks import ROOT,SOURCE,load,read,write,sha,cv2
from run_manual_ball_finetune import prepare
from run_hard_case_ball_finetune import infer_maps
from compare_ball_candidate_decoders import components,choose,METHODS

OUT=ROOT/'ball_inplay_challenge';DATA=ROOT/'balltrack_manual_finetune_pilot'
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False),encoding='utf-8')
def one(text,before,after):
    assert text.count(before)==1,(before,text.count(before))
    return text.replace(before,after)

def main():
    assert not (OUT/'annotation_manifest.json').exists()
    initial=load(OUT/'mining_protocol.json');inventory=read(OUT/'mined_frame_inventory.csv')
    assert not (OUT/'mining_report.json').exists()
    initial_outcome=dict(status='no_eligible_sustained_runs_no_labels_created',sampled_frames=len(inventory),
        spatial_competition_frames=sum(int(r['spatial_competition']) for r in inventory),
        disagreement_frames=sum(int(r['method_disagreement']) for r in inventory),
        initial_protocol_preserved=True,original_rule='At least3 consecutive stride2 sampled competing frames, excluding old labels +/-8')
    dump(OUT/'initial_sampling_outcome.json',initial_outcome)
    protocol=dict(status='revised_query_sampling_before_new_labels',initial_protocol_sha256=sha(OUT/'mining_protocol.json'),
        revision_reason='Competition is brief; original sustained-run criterion gives zero queries. No new labels have been collected.',
        role='validation_challenge_only_never_training',clips=initial['clips'],search_scope=initial['query_scope'],
        candidate_frame_pool='Reuse unchanged scanned frame inventory, no extra model or threshold search',
        selection='One per clip: prefer area-vs-peak disagreement, largest center distance, ties earliest. Else spatial competition with largest separation.',
        existing_label_exclusion='At least4 sourceframes from old centers AND no exact4-input image hash overlaps with any old labelled input',
        caveat='Same-clip neighbouring frames remain temporally correlated; not independent test samples',
        camera_exclusion='Cached transition mask retained unchanged',
        main_play='Human target-phase review, not exact event-frame annotation; service/end guards do not auto-confirm phase',
        comparison='All4 candidate rules fixed and frozen before labels; no fitting or score calibration; non-play excluded from primary point counts',
        labels_needed='Play phase plus ball center/visibility only; nonplay or phase-uncertain can skip ball center',
        sampling='Purposefully enriched challenge with disagreement-preferred examples, not population accuracy or random evaluation',
        source_inventory_sha256=sha(OUT/'mined_frame_inventory.csv'),automatic_adoption=False)
    dump(OUT/'sampling_revision_protocol.json',protocol)
    old=read(DATA/'train_labels.csv')+read(DATA/'validation_labels.csv')+read(DATA/'audit_labels.csv')
    known_hashes={h for r in old for h in json.loads(r['input_frame_sha256_json'])}
    entries={r['ball_pilot_id']:r for r in read(ROOT/'play_start_audit/ball_pilot_start_inventory.csv')}
    templates={r['candidate_id']:r for r in read(DATA/'validation_labels.csv')}
    selected=[];inputs=[]
    for suffix in initial['clips']:
        cid=f'match1_motion_pilot_{suffix:03d}';old_times=[int(r['local_frame']) for r in old if r['candidate_id']==cid]
        pool=[r for r in inventory if r['candidate_id']==cid and int(r['spatial_competition']) and not int(r['camera_transition'])
            and min(abs(int(r['local_frame'])-t) for t in old_times)>=4]
        pool.sort(key=lambda r:(-int(r['method_disagreement']),-float(r['area_peak_distance_px']) if int(r['method_disagreement']) else -float(r['max_candidate_separation_px']),int(r['local_frame'])))
        chosen=None;base=SOURCE/'data/balltrack/racketvision/match1'/cid
        for r in pool:
            t=int(r['local_frame']);paths=[base/'frames'/f'{j:06d}.jpg' for j in range(t-3,t+1)];hashes=[sha(p) for p in paths]
            if set(hashes)&known_hashes:continue
            chosen=r;break
        assert chosen is not None,cid
        cid_entry=entries[cid];sample=f'{cid}_frame_{t:06d}';case_id=chr(65+len(selected));meta=load(base/'runs/threshold030_median120/run_metadata.json')
        selected.append(dict(case_id=case_id,sample_id=sample,candidate_id=cid,local_frame=t,
            source_frame=int(chosen['source_frame']),source_image=str(paths[-1]),source_image_sha256=hashes[-1],
            source_width=1920,source_height=1080,play_window_id=cid_entry['play_window_id'],
            rally_window_candidate_id=cid_entry['rally_window_candidate_id'],clip=meta['clip'],frame_count=meta['frame_count'],
            clip_first_source_frame=int(cid_entry['clip_first_source_frame']),source_fps=meta['fps'],
            query_type='disagreement' if int(chosen['method_disagreement']) else 'competing_candidates_agree_control',
            old_label_minimum_distance=min(abs(t-j) for j in old_times),phase_status='human_pending',role='validation_challenge_never_training'))
        template=templates[cid]
        inputs.append(dict(sample_id=sample,candidate_id=cid,local_frame=t,source_frame=chosen['source_frame'],finetune_role='validation',
            truth_visibility='unreviewed',truth_x_px='',truth_y_px='',input_frames_json=json.dumps([str(p) for p in paths]),
            input_frame_sha256_json=json.dumps(hashes),median_path=template['median_path'],median_sha256=template['median_sha256']))
    assert len(selected)==4 and {r['candidate_id'] for r in selected}<={r['candidate_id'] for r in read(DATA/'validation_labels.csv')}
    write(OUT/'selected_query_frames.csv',selected);write(OUT/'query_model_inputs.csv',inputs)
    modelroot=Path(initial['model']).parents[1];sys.path.insert(0,str(modelroot))
    from inference import BallInferencer
    infer=BallInferencer(str(modelroot/'configs/tracknetv3_base.py'),initial['model'],device='cuda',thre=.3,batchsize=4)
    medians={};samples=prepare(inputs,medians);maps=infer_maps(infer,samples,medians);locked=[]
    for case,hm in zip(selected,maps,strict=True):
        cs=components(hm);assert len(cs)>=2
        for method in METHODS:
            c=choose(cs,method);locked.append(dict(case_id=case['case_id'],sample_id=case['sample_id'],candidate_id=case['candidate_id'],
                local_frame=case['local_frame'],source_frame=case['source_frame'],method=method,visible=1,x_px=c['x_px'],y_px=c['y_px'],ranking_score=c[method]))
    write(OUT/'frozen_method_predictions.csv',locked)
    labels=[]
    for case in selected:
        cid=case['candidate_id'];t=case['local_frame'];case_id=case['case_id'];base=Path(case['source_image']).parent
        video=OUT/f'phase_context_{case_id}_{cid[-3:]}_{t}_h264.mp4';fps=case['source_fps']/2
        cmd=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-y','-f','rawvideo','-pix_fmt','bgr24','-s','1280x800','-r',str(fps),'-i','pipe:0',
            '-an','-c:v','libx264','-preset','fast','-crf','17','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
        count=0
        with (OUT/f'phase_context_{case_id}_render.log').open('wb') as log:
            proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=log)
            for j in range(t%2,case['frame_count'],2):
                im=cv2.imread(str(base/f'{j:06d}.jpg'));assert im.shape[:2]==(1080,1920)
                frame=np.full((800,1280,3),20,np.uint8);frame[80:]=cv2.resize(im,(1280,720),interpolation=cv2.INTER_AREA)
                cv2.putText(frame,f'{case_id} | pilot{cid[-3:]} local {j} | TARGET {t} | 1x | No model markers',(12,29),0,.70,(245,245,245),1,cv2.LINE_AA)
                cv2.putText(frame,'TARGET FRAME - phase and ball center refer to THIS instant' if j==t else 'Full clip context; clip edges are NOT confirmed rally boundaries',(12,62),0,.61,(80,235,255) if j==t else (220,220,220),1,cv2.LINE_AA)
                if j==t:cv2.rectangle(frame,(1,80),(1278,798),(80,235,255),3)
                for _ in range(31 if j==t else 1):proc.stdin.write(frame.tobytes());count+=1
            proc.stdin.close();assert proc.wait()==0
        cap=cv2.VideoCapture(str(video));decoded=0
        while True:
            ok,im=cap.read()
            if not ok:break
            assert im.shape[:2]==(800,1280);decoded+=1
        cap.release();assert decoded==count
        case.update(context_video=str(video.resolve()),context_video_sha256=sha(video),decoded_video_frames=decoded,video_duration_sec=decoded/fps)
        labels.append(dict(sample_id=case['sample_id'],case_id=case_id,evaluation_split='validation_challenge_never_training',candidate_id=cid,
            local_frame=t,source_frame=case['source_frame'],truth_visibility='unreviewed',truth_x_px='',truth_y_px='',reviewer_id='',
            label_confidence='',phase_status='unknown',phase_review_status='',ball_annotation_scope='',notes='',annotated_at_utc=''))
    write(OUT/'manual_center_labels.csv',labels);dump(OUT/'annotation_cases.json',selected)
    # Reuse tested center entry and bounded local HTTP handling in an isolated copy.
    server=(ROOT/'serve_training_ball_annotation.py').read_text(encoding='utf-8')
    server=one(server,"OUT=ROOT/'ball_training_case_expansion'","OUT=ROOT/'ball_inplay_challenge'")
    server=one(server,'def validate(payload,ids):','def validate_center(payload,ids):')
    wrapper='''def validate(payload,ids):
    phase=payload.get('phase_status','unknown')
    if phase not in ('play','preparation_or_ended','uncertain'):raise ValueError('対象フレームがプレー中かを選んでください。')
    skipped=phase!='play' and payload.get('truth_visibility')=='unreviewed'
    center=dict(payload,phase_status='unknown' if phase=='uncertain' else phase)
    if skipped:center['truth_visibility']='uncertain'
    result=validate_center(center,ids)
    result.update(phase_status=phase,phase_review_status='human_uncertain' if phase=='uncertain' else 'human_confirmed',
        ball_annotation_scope='skipped_nonplay_or_phase_uncertain' if skipped else 'human_ball_review')
    return result

'''
    server=one(server,'def atomic_write(path,rows):',wrapper+'def atomic_write(path,rows):')
    server=one(server,"(ROOT/'ball_training_annotation.html')","(OUT/'phase_challenge_annotation.html')")
    server=one(server,"phase_status='unknown')","phase_status='play')")
    server=one(server,"dict(case_id='other')]:","dict(case_id='other'),dict(phase_status='unknown')]:")
    server_path=ROOT/'serve_ball_phase_challenge.py';server_path.write_text(server,encoding='utf-8')
    html=(ROOT/'ball_training_annotation.html').read_text(encoding='utf-8')
    html=html.replace('学習用の球中心・4フレーム','評価用：プレー区分と球中心').replace('学習用の球中心：4フレームだけ','評価用：4場面のプレー区分と球中心')
    html=one(html,'球が見えたら、<strong>下の対象画像で球の中心をクリック</strong>してください。見えない・判断できない場合は、その状態を選んで保存します。',
        '動画で黄色の枠が出る時点について、<strong>プレー中かを選んでください</strong>。プレー中なら、対象画像の球中心をクリックします。準備中・終了後や場面不明なら中心入力は省略できます。')
    phase='<label>場面（任意） <select id="phase"><option value="unknown">不明</option><option value="play">プレー中</option><option value="preparation_or_ended">準備中・プレー後</option></select></label>'
    html=one(html,phase,'')
    details='<details open><summary>前後の動きを確認する（対象前後0.5秒・スロー）</summary><p class="small">黄色の枠と「TARGET FRAME」で止まる箇所が対象画像です。クリックするのは上の画像です。</p><video id="context" controls preload="metadata"></video></details>'
    html=one(html,details,'')
    replacement='<details open><summary>1. 動画で場面を確認（通常速度・対象で停止）</summary><p class="small">黄色の枠とTARGET FRAMEで止まる時点が確認対象です。サーブ開始やラリー終了の正確なフレーム指定は不要です。</p><video id="context" controls preload="metadata"></video></details><label><strong>2. 対象フレームはプレー中ですか（必須）</strong> <select id="phase"><option value="unknown">選択してください</option><option value="play">プレー中</option><option value="preparation_or_ended">準備中・プレー後</option><option value="uncertain">判断できない</option></select></label><p>3. プレー中なら、下の対象画像で球中心をクリック。見えない場合は球の状態を選択してください。</p>'
    html=one(html,'<h2 id="title"></h2><p id="frameInfo" class="sub small"></p>','<h2 id="title"></h2><p id="frameInfo" class="sub small"></p>'+replacement)
    html=html.replace('位置・可視性の教師として使える回答','主プレーの評価に使える回答')
    html=one(html,"['visible','not_visible'].includes(c.truth_visibility)","c.phase_status==='play'&&['visible','not_visible'].includes(c.truth_visibility)")
    html=one(html,"if(state==='unreviewed'){","if(phase.value==='unknown'){message('対象フレームがプレー中かを選んでください。','error');return;}if(state==='unreviewed'&&phase.value==='play'){")
    html=one(html,'新しい4件専用のファイルに保存します。以前の300件や追跡結果は変更しません。','評価専用の4件です。学習には使いません。以前の300件、追加学習用の4件、追跡結果は変更しません。')
    html_path=OUT/'phase_challenge_annotation.html';html_path.write_text(html,encoding='utf-8')
    protected=dict(initial['protected_input_hashes'])
    for path in [OUT/'frozen_method_predictions.csv',OUT/'sampling_revision_protocol.json',ROOT/'ball_training_case_expansion/manual_center_labels.csv']:
        protected[str(path)]=sha(path)
    assert all(sha(Path(p))==h for p,h in protected.items())
    dump(OUT/'annotation_manifest.json',dict(status='four_blinded_validation_challenge_queries_ready',cases=4,cases_sha256=sha(OUT/'annotation_cases.json'),
        initial_labels_sha256=sha(OUT/'manual_center_labels.csv'),frozen_predictions_sha256=sha(OUT/'frozen_method_predictions.csv'),
        html_sha256=sha(html_path),server_sha256=sha(server_path),training_use_forbidden=True,phase_human_pending=True,
        protected_input_hashes=protected,original_labels_and_reviewed_outputs_unchanged=True))
    print(json.dumps([{k:c[k] for k in ['case_id','candidate_id','local_frame','query_type','old_label_minimum_distance','decoded_video_frames']} for c in selected],indent=2))

if __name__=='__main__':main()
