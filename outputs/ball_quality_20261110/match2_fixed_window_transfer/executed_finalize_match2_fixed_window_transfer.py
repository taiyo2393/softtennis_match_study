"""Render frozen-window camera ablation and verify native/scoped outputs."""
import sys,json,subprocess,shutil
from pathlib import Path
from collections import defaultdict,Counter
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,write,sha,dump
from run_match2_fixed_window_transfer import OUT,MODES,VIDEO,FPS,load

STYLES={'near_front':('N1',(60,240,60)),'near_rear':('N2',(250,220,50)),
        'far_rear':('F1',(230,70,230)),'far_front':('F2',(50,230,250))}


def table(path):
    result=defaultdict(list)
    if path.exists():
        for r in read(path):result[int(r['source_frame'])].append(r)
    return result


def overlay(image,rows,width=1280):
    height=round(image.shape[0]*width/image.shape[1]);scale=width/image.shape[1]
    image=cv2.resize(image,(width,height),interpolation=cv2.INTER_AREA)
    for r in rows:
        name,color=STYLES[r['player_slot']];bb=[round(float(r[k])*scale) for k in ['x1','y1','x2','y2']]
        label='S'+r['identity_namespace'].split(':S')[-1].lstrip('0')+' '+name if ':S' in r['identity_namespace'] else name
        cv2.rectangle(image,tuple(bb[:2]),tuple(bb[2:]),color,2)
        cv2.putText(image,label,(bb[0],max(18,bb[1]-7)),cv2.FONT_HERSHEY_SIMPLEX,.62,color,2,cv2.LINE_AA)
        cv2.circle(image,(round(float(r['bottom_center_x'])*scale),round(float(r['bottom_center_y'])*scale)),4,color,-1)
    return image


def render():
    p=load(OUT/'protocol.json');media=[]
    for e in p['cases'][1:]:
        tag=e['clip_id'];dest=OUT/tag;a,b=[table(dest/f'{m}_native_observations.csv') for m in MODES]
        frame_rows=read(dest/'frame_availability_not_accuracy.csv');frames={int(r['source_frame']):r for r in frame_rows if r['mode']==MODES[1]}
        segments=load(dest/'reset_segments.json')
        initial=[v['frame'] for s in segments for v in s['events'] if v['event']=='initialized']
        supported=[f for f,r in frames.items() if r['line_geometry_supported']=='1']
        incomplete=[f for f in supported if initial and f>=min(initial) and len(b[f])<4]
        diagnostic=set(e['raw_reference_source_frames']+initial+incomplete)
        if supported:diagnostic.update([min(supported),supported[len(supported)//2],max(supported)])
        outside=[f for f,r in frames.items() if r['line_geometry_supported']=='0' and a[f]]
        if outside:diagnostic.update([min(outside),max(outside)])
        comparisons=[(18566,18702,18821) if tag.endswith('002') else (54582,54844)][0]
        diagnostic.update(comparisons)
        first,last=e['first_source_frame'],e['last_source_frame'];path=OUT/f'{tag}_camera_reset_candidates.mp4';assert not path.exists()
        dump(dest/'review_media_protocol.json',dict(first_source_frame=first,last_source_frame=last,
            source_fps='60000/1001',display_fps=float(FPS)/2,display_source_stride=2,playback_speed=1,
            image_aspect_preserved=True,all_source_frames_still_analyzed=True,diagnostic_stills=sorted(diagnostic),
            selection='Fixed raw reference frames plus initialization/line transitions and every post-init incomplete frame; selected diagnostics after results, not accuracy sampling.',
            IDs='Segment-local candidates only. Cross-camera or cross-clip identity is not bound.'))
        command=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','1280x820',
            '-r',str(float(FPS)/2),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(path)]
        cap=cv2.VideoCapture(str(VIDEO));cap.set(1,first);mapping=[]
        with (dest/'encoding.log').open('wb') as log:
            process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
            for f in range(first,last+1):
                use=(f-first)%2==0
                if not use and f not in diagnostic:assert cap.grab();continue
                ok,im=cap.read();assert ok and round(cap.get(1))==f+1
                canvas=np.zeros((820,1280,3),np.uint8);canvas[100:]=overlay(im,b[f])
                supported_now=frames[f]['line_geometry_supported']=='1'
                lines=[f'{tag} | source F{f} | {f/float(FPS):.3f}s | normal speed (source stride2)',
                    f'Line support: {supported_now} | no-reset: {len(a[f])}/4 | camera-reset: {len(b[f])}/4',
                    'IDs are local to each segment; cross-camera persons NOT linked. Native box-bottom proxies, no interpolation.']
                for i,line in enumerate(lines):cv2.putText(canvas,line,(12,25+30*i),cv2.FONT_HERSHEY_SIMPLEX,.58,(245,245,245),1,cv2.LINE_AA)
                if use:process.stdin.write(canvas.tobytes());mapping.append(dict(output_frame=len(mapping),source_frame=f))
                if f in diagnostic:assert cv2.imwrite(str(dest/f'F{f}_reset_review.jpg'),canvas)
                if f in comparisons:
                    cmp=np.zeros((640,1920,3),np.uint8)
                    cmp[100:,:960]=overlay(im,a[f],960);cmp[100:,960:]=overlay(im,b[f],960)
                    titles=[f'F{f} | LEFT: frozen tracker, no camera reset | RIGHT: reset on unsupported view',
                        f'No reset {len(a[f])}/4 ; reset {len(b[f])}/4 | Candidate boxes, not independent person truth']
                    for i,line in enumerate(titles):cv2.putText(cmp,line,(15,32+34*i),cv2.FONT_HERSHEY_SIMPLEX,.8,(245,245,245),1,cv2.LINE_AA)
                    assert cv2.imwrite(str(dest/f'F{f}_two_arm_comparison.jpg'),cmp)
            cap.release();process.stdin.close();assert process.wait()==0
        write(dest/'video_frame_map.csv',mapping)
        check=cv2.VideoCapture(str(path));fps=check.get(5);decoded=0
        while True:
            ok,im=check.read()
            if not ok:break
            assert im.shape[:2]==(820,1280);decoded+=1
        check.release();assert decoded==len(mapping)==450 and abs(fps-float(FPS)/2)<.001
        media.append(dict(clip_id=tag,decoded_video_frames=decoded,all899_source_frames_analyzed=True,
            video_uses450_real_source_frames=True,fps=fps,frame_map_complete=True))
    dump(OUT/'media_verification.json',dict(status='passed',cases=media))


def finalize():
    p=load(OUT/'protocol.json');result=load(OUT/'result.json');reports=[];events=[];equivalence=[]
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items()) and sha(VIDEO)==p['source_video_sha256']
    for e in p['cases'][1:]:
        tag=e['clip_id'];dest=OUT/tag;raw=read(dest/'native_full_frame_detections.csv');fields=['source_frame','track_id','x1','y1','x2','y2','bottom_center_x','bottom_center_y']
        native={tuple(r[k] for k in fields) for r in raw};segments=load(dest/'reset_segments.json');line={int(r['source_frame']):r['line_geometry_supported']=='1' for r in read(dest/'line_geometry_support.csv')}
        observations={mode:read(dest/f'{mode}_native_observations.csv') for mode in MODES}
        if segments and segments[0]['first_frame']==e['first_source_frame']:
            end=segments[0]['last_frame']
            comparison_fields=fields+['player_slot','identity_source','association_cost','alternative_margin']
            left=[tuple(r[k] for k in comparison_fields) for r in observations[MODES[0]] if int(r['source_frame'])<=end]
            right=[tuple(r[k] for k in comparison_fields) for r in observations[MODES[1]] if int(r['source_frame'])<=end]
            assert left==right
            equivalence.append(dict(clip_id=tag,first_source_frame=e['first_source_frame'],last_source_frame=end,
                continuous_supported_frames=end-e['first_source_frame']+1,matching_native_assignments=len(left),
                same_IDs_boxes_sources_costs_and_margins_before_cut=True))
        for mode,rows in observations.items():
            assert len({(r['source_frame'],r['player_slot']) for r in rows})==len(rows)
            assert all(tuple(r[k] for k in fields) in native for r in rows)
            by=defaultdict(list)
            for r in rows:by[int(r['source_frame'])].append(r)
            assert all(len({r['track_id'] for r in group})==len(group) for group in by.values())
            if mode==MODES[1]:
                assert all(line[int(r['source_frame'])] for r in rows)
                for s in segments:
                    selected=[r for r in rows if r['identity_namespace']==s['segment_id']]
                    init=[v for v in s['events'] if v['event']=='initialized']
                    if selected:
                        assert init and all(int(r['source_frame'])>=init[0]['frame'] for r in selected)
                        assert init[0]['frame']-init[0]['window_start']+1==8
                        assert all(s['first_frame']<=int(r['source_frame'])<=s['last_frame'] for r in selected)
            # Raw-ID handovers are diagnostics; not counted as identity switches.
            groups=defaultdict(list)
            for r in rows:groups[r['identity_namespace'],r['player_slot']].append(r)
            for (scope,slot),group in groups.items():
                group.sort(key=lambda r:int(r['source_frame']))
                for a,b in zip(group,group[1:]):
                    if a['track_id']!=b['track_id']:events.append(dict(clip_id=tag,mode=mode,identity_namespace=scope,player_slot=slot,
                        last_source_frame=a['source_frame'],source_frame=b['source_frame'],old_raw_id=a['track_id'],new_raw_id=b['track_id'],independent_ID_switch_confirmed=0))
        reset=observations[MODES[1]];by=defaultdict(set)
        for r in reset:by[int(r['source_frame'])].add(r['player_slot'])
        post_init=[]
        for s in segments:
            initialized=[v['frame'] for v in s['events'] if v['event']=='initialized']
            if initialized:
                for f in range(initialized[0],s['last_frame']+1):
                    if len(by[f])<4:post_init.append(dict(source_frame=f,missing_slots=sorted(set(['near_front','near_rear','far_rear','far_front'])-by[f])))
        report=dict(clip_id=tag,reset_segments=[dict(segment_id=s['segment_id'],first_frame=s['first_frame'],last_frame=s['last_frame']) for s in segments],
            reset_post_initialization_missing=post_init,all_exported_rows_match_native_detections=True,no_reset_arm_unsupported_rows_preserved=True,
            reset_has_no_unsupported_frame_emissions=True,independent_ID_or_position_verified=False)
        reports.append(report);dump(dest/'verification.json',report)
    if events:write(OUT/'raw_ID_handover_diagnostic_not_ID_switch_GT.csv',events)
    fixed=read(OUT/'all93_fixed_times_both_modes.csv');assert len(fixed)==186
    for mode in MODES:
        a=[r for r in fixed if r['mode']==mode];assert len(a)==93
        assert sum(r['status']=='preplay_not_processed' for r in a)==31
    dump(OUT/'verification.json',dict(status='passed',processed_clips=2,tracked_source_frames=1798,
        preserved_preplay_case=True,fixed_times_per_mode=93,preplay_unknown_times_per_mode=31,
        all_selected_rows_match_original_frame_detection=True,no_unsupported_camera_emissions_in_reset_arm=True,
        no_pre_confirmation_backfill=True,segment_local_ID_namespaces=True,raw_ID_handover_records=len(events),cases=reports,
        unchanged_first_supported_segment_controls=equivalence,
        protected_previous_work_unchanged=True,independent_accuracy_established=False))
    dump(OUT/'processing_decision.json',dict(status='frozen_additional_window_camera_ablation_completed',
        conclusion='Frozen association alone can initialize using an incompatible view and retain those seeds, or continue emitting after a cut. A separate camera-reset control arm prevents such state carryover by construction and improves main-view candidate availability in002.',
        not_proven='This does not establish person-ID accuracy, camera classification accuracy or cross-camera ReID. Line-geometry failures may suppress valid views; the method is not automatically adopted into canonical data.',
        next='Review the4 remaining post-init missing frames in003 and within-segment raw-ID handovers; extend the frozen native/crop/far-path comparison only after recording geometry assumptions. Keep camera cuts and preparation footage distinct from in-play player errors.',
        core_association_thresholds_changed=False,new_homography_fitted_to_player_outputs=False,
        main_camera_to_closeup_identity_reconnected=False,current22_or_outcome_labels_changed=False))
    (OUT/'README.md').write_text('''# match2の固定15秒窓への展開：画角切替の扱い

先に固定した60〜75秒・300〜315秒・900〜915秒の3窓を維持した。60秒窓は原映像の時系列点検で試合前の準備と判断し、人物・ボール推論は実行せず、31指定時点は未処理として残した。窓を都合のよい場面へ差し替えていない。これはAIの局面判断であり、人手のプレー区間正解ではない。

残り2窓の全1798ソースフレームを同じYOLO11m/ByteTrack全画面検出へ通し、同じ外観特徴とコート対応・追跡定数を2方式へ与えた。追加の遠方切出し・奥ペア全経路探索・身体補助候補は未使用。

## 比較

| 窓 | 白線整合がある枚数 | 切替無視：その範囲で4候補 | 切替時リセット：その範囲で4候補 |
|---|---:|---:|---:|
| 300〜315秒 | 303/899 | 1 | 296 |
| 900〜915秒 | 636/899 | 625 | 625 |

白線整合は、以前に作った同じ4本の白線支持条件を各元画像へ適用した結果。カメラ・人物・可視性の独立正解ではない。主画面のズームや線の遮蔽による誤拒否もありうる。新しいフレームに合わせた閾値調整やHの更新は行っていない。

切替を無視した方式は002でF18566に初期化したが、白線整合が始まるのはF18578。別画角／移行中の状態を主画面へ持ち越している。リセット方式はF18578から確認を始め、F18585から4候補を出した。初期化待ち7枚を除く296枚は4候補がそろう。

003の白線整合範囲はF53946〜54581。リセット方式は最初の7枚を待ち、625枚で4候補。F54328・54330・54538・54544には1人の欠測が残る。補間していない。切替無視方式は白線不整合になった後も293行の候補を出し続けた。不整合範囲の全候補を「誤った人物」と人手認定したわけではなく、校正の前提を満たさない画角への出力として区別する。

## 画角が変わった後のID

比較用リセット方式は、白線支持を失ったら出力とスロット状態・初期化待ちを捨てる。再び支持を得たら通常の8連続フレーム確認から始める。人物検出の同じストリームは両方式で共有しており、人物検出を好都合に変更していない。

IDはクリップと主画面区間を含む名前で保存する。新しい区間のF1/N1が、以前と同じ実選手だとは束縛していない。これは画角をまたぐ本人再同定の解決ではない。同じ文字を再利用して同一人物が追えたように見せないための区別である。画角切替へのリセットは今回追加して比較した制御で、追跡器内部の数値定数は前回のまま。

## 評価と成果物

候補数を正解率へ読み替えない。別試合は増えておらず、この2窓もmatch2の一部。同一人物の独立ラベル・位置正解・画角正解はまだない。raw IDの変更は別CSVに残し、モデルのraw ID変更を本人IDの入替えと同一視していない。

- match2_fixed_clip_002_camera_reset_candidates.mp4、003の同名動画：通常速度、元フレーム2枚ごとの実画像を表示。全899枚ずつは分析・保存済み。
- 各窓の比較画像：左が切替無視、右がリセット。接写・切替時の挙動も保存。
- all93_fixed_times_both_modes.csv：両方式で元の93指定時点を維持。試合前の31時点は空欄・未処理。
- 各窓のnative_full_frame_detections.csvとappearance_features.npz：同じ元候補・外観特徴を再実験できる形で保存。
- reset_segments.json：初期化待ち・区間・ローカルIDの範囲を明示。

旧動画、元のmatch2、既存のラリー・得点側ラベル、current22、前回の試走は変更していない。今回の比較を既存の正本へ自動採用したものではない。
''',encoding='utf-8')
    assert load(OUT/'media_verification.json')['status']=='passed'
    for name in ['run_match2_fixed_window_transfer.py','finalize_match2_fixed_window_transfer.py']:shutil.copy2(ROOT/name,OUT/('executed_'+name))
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items()) and sha(VIDEO)==p['source_video_sha256']
    hashes={str(n.relative_to(OUT)):sha(n) for n in sorted(OUT.rglob('*')) if n.is_file() and n.name!='publication_hashes.json'}
    dump(OUT/'publication_hashes.json',hashes);assert all(sha(OUT/n)==h for n,h in hashes.items())
    print('Published fixed-window ablation;1798 analyzed frames, all native coordinates and old inputs preserved.',flush=True)


if __name__=='__main__':{'render':render,'finalize':finalize}[sys.argv[1]]()
