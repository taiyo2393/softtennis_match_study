"""Review first transfer outputs and preserve camera setup/development limits."""
import sys,json,subprocess,shutil
from pathlib import Path
from collections import defaultdict,Counter
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,SOURCE,read,write,sha,dump
from run_match2_camera_transfer_pilot import OUT,SETUP,VIDEO,FPS,load


def render():
    p=load(OUT/'protocol.json');rs=read(OUT/'court_mapped_camera_gates/selected_native_observations.csv')
    by=defaultdict(list)
    for r in rs:by[int(r['source_frame'])].append(r)
    first,last=p['first_source_frame'],p['last_source_frame']
    wanted=[12295,12348,12408,12468,12506,12507,12526]
    dump(OUT/'review_media_protocol.json',dict(first_source_frame=first,last_source_frame=last,
        source_dimensions=[1920,1080],display_dimensions=[1280,720],header_height=100,
        source_fps='60000/1001',playback_speed=.5,review_stills=wanted,
        selection='Diagnostic stills after output include initialization and both missing F2 frames, not random accuracy sampling.',
        colors='N1green,N2cyan,F1purple,F2yellow; detections/box-bottom proxies are candidates, not independent truth.',
        original_image_aspect_preserved=True,interpolated_or_backfilled_boxes=False))
    video=OUT/'match2_first_point_four_player_candidates.mp4';assert not video.exists()
    command=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24','-s','1280x820',
        '-r',str(float(FPS)/2),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','19','-pix_fmt','yuv420p','-movflags','+faststart',str(video)]
    styles={'near_front':('N1',(60,240,60)),'near_rear':('N2',(250,220,50)),
        'far_rear':('F1',(230,70,230)),'far_front':('F2',(50,230,250))}
    cap=cv2.VideoCapture(str(VIDEO));cap.set(1,first);mapping=[]
    with (OUT/'encoding.log').open('wb') as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        for f in range(first,last+1):
            ok,im=cap.read();assert ok and round(cap.get(1))==f+1
            im=cv2.resize(im,(1280,720),interpolation=cv2.INTER_AREA)
            for r in by[f]:
                name,color=styles[r['player_slot']];bb=[round(float(r[k])*2/3) for k in ['x1','y1','x2','y2']]
                cv2.rectangle(im,tuple(bb[:2]),tuple(bb[2:]),color,2)
                cv2.putText(im,name,(bb[0],max(18,bb[1]-7)),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2,cv2.LINE_AA)
                cv2.circle(im,(round(float(r['bottom_center_x'])*2/3),round(float(r['bottom_center_y'])*2/3)),4,color,-1)
            canvas=np.zeros((820,1280,3),np.uint8);canvas[100:]=im
            lines=[f'match2 | source F{f} | {f/float(FPS):.3f}s | playback 0.5x',
                f'Camera-mapped gates: {len(by[f])}/4 candidates | Same detections with legacy image gates: 0/4',
                'First-point excerpt. Box-bottom dots are image proxies; independent identity/position accuracy unmeasured.']
            for i,line in enumerate(lines):cv2.putText(canvas,line,(12,25+30*i),cv2.FONT_HERSHEY_SIMPLEX,.59,(245,245,245),1,cv2.LINE_AA)
            process.stdin.write(canvas.tobytes())
            if f in wanted:assert cv2.imwrite(str(OUT/f'F{f}_tracking_review.jpg'),canvas)
            mapping.append(dict(output_frame=len(mapping),source_frame=f))
        cap.release();process.stdin.close();assert process.wait()==0
    write(OUT/'video_frame_map.csv',mapping)
    cap=cv2.VideoCapture(str(video));count=0;fps=cap.get(5)
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(820,1280);count+=1
    cap.release();assert count==last-first+1 and abs(fps-float(FPS)/2)<.001
    dump(OUT/'media_verification.json',dict(status='passed',decoded_frames=count,fps=fps,source_frame_map_complete=True,original_aspect_preserved=True))


def finalize():
    p=load(OUT/'protocol.json');result=load(OUT/'result.json');control=load(OUT/'old_context_regression.json')
    assert control['status']=='passed' and len(control['cases'])==10
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items()) and sha(VIDEO)==p['source_video_sha256']
    native=read(OUT/'native_full_frame_detections.csv');selected=read(OUT/'court_mapped_camera_gates/selected_native_observations.csv')
    fields=['source_frame','track_id','x1','y1','x2','y2','bottom_center_x','bottom_center_y']
    source={tuple(r[k] for k in fields) for r in native}
    assert all(tuple(r[k] for k in fields) in source for r in selected)
    assert len({(r['source_frame'],r['player_slot']) for r in selected})==len(selected)
    mode=result['modes']['court_mapped_camera_gates'];assert mode['initialized_frame']==12295
    assert all(int(r['source_frame'])>=12295 for r in selected)
    by=defaultdict(set)
    for r in selected:by[int(r['source_frame'])].add(r['player_slot'])
    missing=[dict(source_frame=f,missing_slots=sorted(set(['near_front','near_rear','far_rear','far_front'])-by[f])) for f in range(12295,p['last_source_frame']+1) if len(by[f])<4]
    assert missing==[dict(source_frame=12506,missing_slots=['far_front']),dict(source_frame=12507,missing_slots=['far_front'])]
    dump(OUT/'missing_output_diagnostic.json',dict(after_initialization=missing,initialization_wait_frames=list(range(12288,12295)),
        raw_detection_at_missing_frames=[r for r in native if int(r['source_frame']) in [12506,12507]],
        new_ID_or_position_imputation=False,independent_reason_label=False))
    tests=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'test_court_paint_center_fit.py')],capture_output=True,text=True)
    assert tests.returncode==0
    dump(SETUP/'paint_center_tests.json',dict(status='passed',synthetic_cases=3,output=tests.stdout+tests.stderr,not_accuracy_validation=True))
    dump(SETUP/'geometry_trial/failed_attempt_summary.json',dict(status='rejected_preserved',
        failure='All9 frames rejected, including3 main-view images: near-baseline raw paint-pixel spread was about3.04px, incorrectly treated as centerline alignment error against3px gate.',
        correction='Separate paint thickness from centerline fit residual;20-bin median paint centers. Constants unchanged; no person tracking outputs used. Failure files retained.'))
    (SETUP/'README.md').write_text('''# match2の主画面・試合開始・コート対応

90〜240秒を5秒刻み、最初の得点周辺200〜211秒を1秒刻みで原映像を点検した。前半には練習・準備と接写があり、205〜209秒付近に最初の得点前の主画面が見える。210秒のスコア更新・ディゾルブも残した。これはAIの時系列所見で、正式なサーブ接触・ラリー終端・得点側の人手ラベルではない。

## 新しいコートの対応

先に固定した900秒区間の主画面F54006を基準に、ダブルス外周の白線へ新たな正規化コート面を当てた。x0/1は画面左/右、y0/1は手前/奥のベースライン。最初の白線画素の散らばりを測る試行は、太いベースラインの幅をずれと混同して失敗した。失敗をgeometry_trialに保存し、白線の20分割ごとの中心を使うgeometry_center_trialへ修正。距離の閾値を広げて通したものではない。

既存9原画像のうち主画面候補3枚で4境界線の支持を得た。接写4枚と別画角2枚は支持不足のまま。これは同じ画像を見て準備した幾何の整合点検で、独立したカメラ分類正解率ではない。別の主画面でも推定交点は基準から最大約7.2ネイティブ画素動く。測量誤差や接地点精度とは解釈しない。

新カメラの足元候補を一時的に旧コート面と対応する画面領域へ写し、初期化・前後の領域判定に使う処理を作った。仮想的な位置を観測座標として出力せず、移動・外観・重なりの計算と保存位置は実際の画像座標のまま。1080/4Kの座標変換とは別の対応である。今後のカメラ切替や大きなズームに同じ校正を使えると保証したものではない。

## 最初の人物追跡試走

../match2_camera_transfer_pilot/ に、205〜209秒の239フレームで比較を保存した。人物予測を見る前に、同一の全画面検出を旧画角条件とコート対応条件へ渡す2方式を固定。旧動画10区間の7488フレームでは、旧校正を与えた場合に従来と同じ割当て・数値コスト（許容1e-9）になることを確認した。

新動画の結果は別フォルダのREADMEに記載。まだ完全な最良パイプラインや独立した本人同定精度の評価ではない。旧動画・ラリーラベル・current22は変更していない。
''',encoding='utf-8')
    (OUT/'README.md').write_text('''# match2：最初の主画面でのカメラ対応試走

2024年男子決勝の205〜209秒（元フレームF12288〜12526、239枚）で、同じYOLO11m/ByteTrack全画面検出と外観特徴を2方式へ渡した。区間は原映像の画角とスコア推移から選び、人物の予測結果を見る前に固定した。ラリー全体や正式な接触・終端を認定した区間ではない。

| 方式 | 初期化 | 4スロットの候補がある枚数 |
|---|---|---:|
| 解像度だけを合わせ、旧画面内の前後領域条件を維持 | なし | 0/239 |
| 新旧コート面を対応させ、領域判定だけを移す | F12295 | 230/239 |

最初の7枚は従来どおり8連続フレーム確認の待ち。確認前へ遡って出力していない。初期化後のF12506・12507には奥F2の欠測がある。補間やIDの手指定で埋めていない。保存した926件すべてが、そのフレームの実検出枠と完全に一致することを確認した。

## 方法と限定

新旧の校正で同じ正規化コート面を対応させ、初期化時の前後候補領域と継続時の前後領域判定だけを移した。画像上の移動、外観、サイズ、重なり、8フレーム確認の条件は維持。仮想座標は領域判定の内部だけで使い、保存する枠・下端中心は元の1920×1080座標へ戻す。下端中心は画像上の代理点であり、物理的接地点とは認定していない。

コート校正には後の主画面F54006の原画像を用いた。人物予測・得点側の正解は使っていないが、オフラインのカメラ設定を与えた比較である。自動オンライン校正とは呼ばない。

旧校正を自分自身へ写す条件で、旧動画10区間の全7488フレームを従来・新クラスで再実行し、割当て・初期化・数値コスト（許容1e-9）の一致を確認した。これは互換性の検証であり、旧動画の本人同定正解率ではない。

この試走は全画面検出＋時間方向の追跡だけ。遠方切り出しの追加検出、奥ペアのオフライン全経路探索、身体だけの補助候補はまだ加えていない。完全な最良方式を別試合で評価した結果とは扱わない。

## 評価上の留保

230/239は4候補の出現数で、正解率ではない。人手による独立したID・位置の正解は付いていない。最初の1得点の短い部分であり、長い遮蔽、再取得、コートチェンジ、別会場への汎化も未評価。今後に新動画の失敗を見て調整した場合は、調整後の成績を独立評価と区別する。

元のmatch1、match2、ラリー・得点側ラベル、current22は変更していない。

- match2_first_point_four_player_candidates.mp4：0.5倍速、縦横比を保存。N1緑/N2水色/F1紫/F2黄。
- native_full_frame_detections.csv：239枚すべての人物検出候補。
- court_mapped_camera_gates/selected_native_observations.csv：新画角対応の926候補。
- old_context_regression.json：全10区間の互換性確認。
- missing_output_diagnostic.json：2枚のF2欠測と初期化待ち7枚を明示。
''',encoding='utf-8')
    assert load(OUT/'media_verification.json')['status']=='passed'
    dump(OUT/'processing_decision.json',dict(status='first_new_match_camera_transfer_feasibility_complete',
        result='With provided new camera plane, temporal4-slot candidates initialize; original image-region gates do not. This is output availability, not independent identity/position accuracy.',
        no_post_match2_tracking_threshold_tuning=True,remaining_missing_F2_frames=[12506,12507],
        next='Inspect the2 missing F2 frames and extend the same frozen comparison to the other preselected contexts/full pipeline without hiding camera cuts or initialization failures. Independent manual reference is still needed for accuracy.',
        canonical_pipeline_changed=False))
    dump(OUT/'final_verification.json',dict(status='passed',source_frames=239,original_source_dimensions=[1920,1080],
        new_mode_init_frame=12295,native_candidate_observations=926,all_selected_rows_match_native_detection=True,
        four_candidate_frames=230,unfilled_initial_wait_frames=7,unfilled_F2_frames=[12506,12507],
        old_control_contexts=10,old_control_source_frames=sum(r['all_source_frames'] for r in control['cases']),
        assignments_and_costs_match_legacy_under_original_camera=True,media_decode_passed=True,
        protected_previous_work_and_videos_unchanged=True,independent_accuracy_established=False))
    for dest,names in [(SETUP,['prepare_match2_camera_setup.py','fit_match2_court_geometry.py','fit_match2_court_geometry_centers.py','test_court_paint_center_fit.py']),
        (OUT,['calibrated_camera_temporal_tracker.py','run_match2_camera_transfer_pilot.py','finalize_match2_camera_transfer.py'])]:
        for name in names:shutil.copy2(ROOT/name,dest/('executed_'+name))
        hashes={str(n.relative_to(dest)):sha(n) for n in sorted(dest.rglob('*')) if n.is_file() and n.name!='publication_hashes.json'}
        dump(dest/'publication_hashes.json',hashes);assert all(sha(dest/n)==h for n,h in hashes.items())
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items()) and sha(VIDEO)==p['source_video_sha256']
    print('Published camera setup and first transfer pilot; missing outputs retained.',flush=True)


if __name__=='__main__':{'render':render,'finalize':finalize}[sys.argv[1]]()
