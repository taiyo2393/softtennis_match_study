"""Bounded body-only review artifacts and auditable development decision."""
import sys,json,subprocess,shutil
from collections import defaultdict,Counter
from pathlib import Path
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,FPS,read,write,sha,dump
from run_near_prefix_initialization_trial import load
from run_position_proxy_support_head_independent import all_pose_geometry
from run_body_part_localization_probe import overlap

OUT=BASE/'partial_body_identity_trial'
PRIOR=BASE/'camera_valid_near_prefix_trial'
POLICY='exclude_near_rectangles'


def draw(im,group,native,f,tag):
    canvas=cv2.resize(im,(1280,720),interpolation=cv2.INTER_AREA)
    for slot,r in native.items():
        bb=[round(float(r[k])/3) for k in ['x1','y1','x2','y2']]
        color=(255,200,60) if slot.startswith('near_') else (180,180,180)
        cv2.rectangle(canvas,tuple(bb[:2]),tuple(bb[2:]),color,1)
    for slot,person in group.items():
        color=(230,80,250) if slot=='far_rear' else (50,240,250)
        kp=np.asarray(person['keypoints']);polygon=np.rint(kp[[5,6,12,11],:2]/3).astype(np.int32)
        cv2.polylines(canvas,[polygon],True,color,2)
        center=np.rint(kp[[5,6],:2].mean(axis=0)/3).astype(int)
        name='F1?' if slot=='far_rear' else 'F2?'
        cv2.putText(canvas,name,tuple(center+[4,-12]),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2,cv2.LINE_AA)
    output=np.zeros((820,1280,3),np.uint8);output[100:]=canvas
    lines=[f'{tag} source F{f} | OFFLINE body-ID hypotheses | playback 0.25x (source stride2)',
        f'Existing far native boxes: {sum(s.startswith("far_") for s in native)}/2 | Body hypotheses: {len(group)}/2 | NEW foot coordinates: 0',
        'Magenta F1? / yellow F2?: model torso only. Cyan/gray rectangles: existing native observations.']
    for i,line in enumerate(lines):cv2.putText(output,line,(12,25+30*i),cv2.FONT_HERSHEY_SIMPLEX,.61,(245,245,245),1,cv2.LINE_AA)
    return output


def render():
    p=load(OUT/'protocol.json');e=next(e for e in p['cases'] if e['tag']=='006')
    poses=all_pose_geometry('006');rows=read(OUT/'case_006'/f'{POLICY}_fixed_body_hypotheses.csv')
    selected=defaultdict(dict)
    for r in rows:
        if r['body_ID_candidate']=='1':
            f=int(r['source_frame']);selected[f][r['player_slot']]=next(v for v in poses[f,'far_context'] if v['pose_index']==int(r['pose_index']))
    native=defaultdict(dict)
    for r in read(PRIOR/'case_006/candidate_native_observations.csv'):native[int(r['source_frame'])][r['player_slot']]=r
    times=e['requested_input_frames'];stills=[10536,10546,10558,10560,10574,10576]
    dump(OUT/'media_protocol.json',dict(tag='006',requested_frames=times,illustrative_stills=stills,
        chosen_after_result=True,not_accuracy_sampling=True,model_torso_display_not_pixel_visibility_truth=True,
        original_image_dimensions=[3840,2160],display_image_dimensions=[1280,720],header_pixels=100,
        playback_speed=.25,source_frame_stride=2,source_clock_unchanged=True))
    path=OUT/'006_body_ID_without_new_positions.mp4';assert not path.exists()
    command=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','1280x820',
        '-r',str(FPS/8),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
    cap=cv2.VideoCapture(p['source_video']);cap.set(1,min(times));wanted=set(times);mapping=[]
    with (OUT/'encoding.log').open('wb') as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        for f in range(min(times),max(times)+1):
            if f not in wanted:assert cap.grab();continue
            ok,im=cap.read();assert ok and round(cap.get(1))==f+1
            panel=draw(im,selected[f],native[f],f,'006');process.stdin.write(panel.tobytes())
            if f in stills:assert cv2.imwrite(str(OUT/f'006_F{f}_body_ID.jpg'),panel)
            mapping.append(dict(output_frame=len(mapping),source_frame=f))
        cap.release();process.stdin.close();assert process.wait()==0
    write(OUT/'video_frame_map.csv',mapping)
    cap=cv2.VideoCapture(str(path));count=0;fps=cap.get(5)
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(820,1280);count+=1
    cap.release();assert count==31 and abs(fps-FPS/8)<.001
    # Only nonmatching existing native rectangle, for AI review, not a GT error.
    tag='017';f=42134;ps=all_pose_geometry(tag);r=next(r for r in read(OUT/f'case_{tag}/{POLICY}_fixed_body_hypotheses.csv') if int(r['source_frame'])==f and r['player_slot']=='far_front')
    person=next(v for v in ps[f,'far_context'] if v['pose_index']==int(r['pose_index']))
    native={r['player_slot']:r for r in read(PRIOR/f'case_{tag}/candidate_native_observations.csv') if int(r['source_frame'])==f}
    cap=cv2.VideoCapture(p['source_video']);cap.set(1,f);ok,im=cap.read();assert ok and round(cap.get(1))==f+1;cap.release()
    assert cv2.imwrite(str(OUT/'017_F42134_body_vs_native.jpg'),draw(im,{'far_front':person},native,f,tag))
    dump(OUT/'media_verification.json',dict(status='passed',frames_decoded=count,fps=fps,
        source_stride2_map_complete=True,original_aspect_preserved=True,interpolated_identity_or_position_frames=0))


def detail_sheet():
    p=load(OUT/'protocol.json');poses=all_pose_geometry('006')
    rows=[r for r in read(OUT/f'case_006/{POLICY}_fixed_body_hypotheses.csv') if r['player_slot']=='far_rear' and r['body_ID_candidate']=='1' and r['baseline_native_present']=='0']
    assert len(rows)==9
    canvas=np.zeros((1170,1350,3),np.uint8);cap=cv2.VideoCapture(p['source_video'])
    for i,r in enumerate(rows):
        f=int(r['source_frame']);person=next(v for v in poses[f,'far_context'] if v['pose_index']==int(r['pose_index']))
        cap.set(1,f);ok,im=cap.read();assert ok and round(cap.get(1))==f+1
        im=im[1050:1386,1950:2370].copy()
        polygon=np.rint(np.asarray(person['keypoints'])[[5,6,12,11],:2]-[1950,1050]).astype(np.int32)
        cv2.polylines(im,[polygon],True,(230,80,250),2)
        cell=np.zeros((390,450,3),np.uint8);cell[30:]=cv2.resize(im,(450,360),interpolation=cv2.INTER_CUBIC)
        cv2.putText(cell,f'F{f}: F1? body hypothesis, NO foot xy',(8,21),cv2.FONT_HERSHEY_SIMPLEX,.50,(240,240,240),1,cv2.LINE_AA)
        y,x=divmod(i,3);canvas[y*390:(y+1)*390,x*450:(x+1)*450]=cell
    cap.release();assert cv2.imwrite(str(OUT/'006_all9_added_F1_body_hypotheses.jpg'),canvas)


def finalize():
    p=load(OUT/'protocol.json');rows=read(OUT/'all_fixed_body_hypotheses.csv');result=load(OUT/'result.json')
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    assert len(rows)==1240 and all(r['analysis_x']==r['analysis_y']=='' for r in rows)
    assert len({(r['tag'],r['policy'],r['source_frame'],r['player_slot']) for r in rows})==1240
    summaries=[]
    for e in p['cases']:
        tag=e['tag'];native=defaultdict(dict)
        for r in read(PRIOR/f'case_{tag}/candidate_native_observations.csv'):native[int(r['source_frame'])][r['player_slot']]=r
        for mode in ['unmasked_torso','exclude_near_rectangles']:
            group=[r for r in rows if r['tag']==tag and r['policy']==mode]
            ids=defaultdict(list)
            for r in group:
                if r['body_ID_candidate']=='1':ids[int(r['source_frame'])].append(r['pose_index'])
            assert all(len(v)==len(set(v)) for v in ids.values())
            comparisons=Counter()
            for r in group:
                if r['body_ID_candidate']!='1':continue
                f=int(r['source_frame']);bb=json.loads(r['body_box_json']);same=native[f].get(r['player_slot'])
                others=[slot for slot,v in native[f].items() if slot!=r['player_slot'] and overlap(bb,[float(v[k]) for k in ['x1','y1','x2','y2']],union=True)>=.5]
                if others:comparisons['overlaps_other_native_slot']+=1
                elif same and float(r['baseline_same_slot_box_iou'])>=.5:comparisons['same_slot_native_box']+=1
                elif same:comparisons['different_box_extent_or_unresolved_identity']+=1
                else:comparisons['baseline_missing']+=1
            summary=dict(tag=tag,policy=mode,baseline_native_far_boxes=sum(s.startswith('far_') for f in e['requested_input_frames'] for s in native[f]),
                body_hypotheses=sum(r['body_ID_candidate']=='1' for r in group),
                supported_far_pairs=sum(len(ids[f])==2 for f in e['requested_input_frames']),
                baseline_missing_and_body_unsupported=sum(r['body_ID_candidate']=='0' and r['baseline_native_present']=='0' for r in group),
                correspondence_categories_json=json.dumps(dict(comparisons)))
            summaries.append(summary)
    write(OUT/'model_correspondence_not_accuracy.csv',summaries)
    six=[r for r in rows if r['policy']==POLICY and r['tag']=='006']
    assert sum(r['body_ID_candidate']=='1' and r['baseline_native_present']=='0' for r in six)==29
    assert next(r for r in summaries if r['tag']=='006' and r['policy']==POLICY)['supported_far_pairs']==20
    tests=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'test_partial_body_identity.py')],capture_output=True,text=True)
    assert tests.returncode==0
    dump(OUT/'invariant_tests.json',dict(status='passed',synthetic_tests=5,output=tests.stdout+tests.stderr,
        no_foot_positions_in_all1240_rows=True,unique_pose_per_far_identity_per_frame=True,
        synthetic_tests_not_accuracy_validation=True))
    dump(OUT/'processing_decision.json',dict(
        conclusion='Body-only offline identity hypotheses can supplement006:29 hypotheses at missing native slot-times,20 for the visible far player and9 around emergence of the overlapped player. Deep overlap remains unsupported.',
        native_replacement='Not adopted: standalone body tracker loses existing native support in other contexts, especially009/019. Future common anchors are required for006. Existing native association is not identity GT.',
        masking_comparison='No improvement to006 availability from removing near rectangles; overall model support534/620 unmasked vs528/620 masked. No evidence of a net benefit from this mask in the retained sample; preserve both modes, no tuning.',
        extra_positions='None. Every additional body hypothesis has blank foot/analysis coordinates. Existing native position outputs are untouched.',
        unresolved_reference='017/F42134 far_front has model-body/native IoU0.453. A body/native box-extent difference is not automatically an identity switch.',
        next='Review whether the9 newly supported006 orange-player frames correspond to the intended person, and keep unresolved11 earlier slot-times missing. Then freeze this auxiliary candidate stream and prioritize held-out-match/person validation over more same-match thresholds.',
        independent_identity_accuracy_claim=False,current22_replaced=False))
    (OUT/'README.md').write_text('''# 身体のID候補と足元位置を分けるオフライン比較

既存の主画面条件付き手前追跡を固定し、全10区間・指定31時点・奥2スロットに、身体だけの別経路を作った。各方式620行、2方式計1240行は欠測も含む。既存の人物検出・姿勢推定キャッシュを使い、新しい姿勢推論・重み学習・得点ラベル・閾値探索は行っていない。

## 006で得られた候補

従来の奥側の実検出経路が未出力だった40スロット時点のうち29件に身体ID候補が出た。内訳は、見えている奥の選手F2候補がF10536〜10574の20時点、重なりから出る奥の選手F1候補がF10558〜10574の9時点。先の11時点F10536〜10556のF1候補は欠測のまま。

指定31時点で奥2人の身体候補がそろうのは20時点。これはモデル候補の出現数で、本人同定精度ではない。従来の4人の実検出枠・位置がそろう時点数は11/31のまま。追加した身体候補から足元・実空間座標は1件も作っていない。

## 方法と将来情報

各区間の従来の4人初期化8ソースフレームにある身体像を、既存F1/F2への対応の手がかりにする。肩・腰の姿勢推定信頼度0.5以上の胴体領域からHSV色特徴を取り、外観・移動・大きさ・競合候補とのマージンで2人を別々の姿勢候補へ対応させる。モデルが出す肩・腰も可視画素の正解ではない。

006はF10575の共通初期化を手がかりに前へたどるため、早いフレームには後の画像を使う。リアルタイム追跡ではない。初期化は指定入力の末尾F10596より前だが、ラリーの有効区間や得点予測の適格性を新たに確認したわけではない。ほかの区間は共通初期化からキャッシュ済み姿勢フレームを順に進む。

身体経路は独立した補助候補である。座標欄analysis_x/analysis_yは全1240行で空欄。動画の胴体ポリゴンや内部の移動計算を、観測された足元位置に読み替えてはいけない。

## 比較と不採用の判断

胴体領域をそのまま使う方式と、手前選手の既存枠に入る画素を除く方式を同じ条件で比較した。006の身体候補数は両方式で同じ51/62、追加29件も同じ。全10区間では534/620と528/620。画素除外による候補の正解率向上は示せず、出力数も増えなかった。パラメータを合わせ直さず両結果を残す。

他の区間には既存経路より身体候補の欠測が増える例がある。特に009・019では減少が大きい。この身体経路で既存追跡を置き換えることは採用しない。既存経路と同じ人物枠に重なることも、二つのモデルの整合を示すだけで本人同定の独立正解ではない。

017/F42134のF2候補は従来枠とのIoUが0.453である。これだけで入れ替わりと判定せず、枠範囲の差か人物対応の差かを別途画像で点検する。全方式・全区間の欠測や不一致をmodel_correspondence_not_accuracy.csvに残した。

## 成果物と現状

- `006_body_ID_without_new_positions.mp4`：指定31時点を0.25倍速。紫F1?・黄F2?はモデルの胴体候補。足元点は追加していない。
- `all_fixed_body_hypotheses.csv`：全1240行の支持・欠測・既存モデルとの対応。
- `case_*/feature_metadata.csv`：胴体画素数、画素除外、特徴が取れなかった場合も保存。
- `protocol.json`：特徴抽出前に固定した方法・対象・将来情報の使用。
- `processing_decision.json`：置き換え不採用と評価の留保。

current22、従来の追跡経路・足元座標、得点側・区間ラベル、以前の位置参照評価は変更していない。同一試合で既知の失敗を見て作った開発比較であり、独立した本人同定精度は未評価。今後は追加候補の映像対応を確かめ、別試合での評価を優先する。
''',encoding='utf-8')
    assert load(OUT/'media_verification.json')['status']=='passed'
    dump(OUT/'final_verification.json',dict(status='passed',contexts=10,fixed_times=310,
        two_policy_far_slot_rows=1240,body_candidates_unmasked=534,body_candidates_masked=528,
        new006_body_candidates=29,remaining006_unsupported_preinit_slot_times=11,
        no_new_analysis_or_foot_coordinates=True,unique_pose_assignment=True,
        protected_inputs_and_current22_unchanged=True,offline_future_anchor_explicit=True,
        tests_passed=5,media_decode_passed=True,independent_accuracy_claim=False))
    for n in ['partial_body_identity.py','run_partial_body_identity_trial.py','test_partial_body_identity.py','finalize_partial_body_identity_trial.py']:
        shutil.copy2(ROOT/n,OUT/('executed_'+n))
    hashes={str(n.relative_to(OUT)):sha(n) for n in sorted(OUT.rglob('*')) if n.is_file() and n.name!='publication_hashes.json'}
    dump(OUT/'publication_hashes.json',hashes);assert all(sha(OUT/n)==h for n,h in hashes.items())
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    print('Published body-only comparison; native tracker and foot coordinates unchanged.',flush=True)


if __name__=='__main__':{'render':render,'detail':detail_sheet,'finalize':finalize}[sys.argv[1]]()
