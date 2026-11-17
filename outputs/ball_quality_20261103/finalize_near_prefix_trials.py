"""Publish bounded diagnostics, preserved failure, video and integrity checks."""
import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from run_fixed_cohort_player_case_v3 import ROOT, BASE, FPS, read, write, sha, dump
from run_near_prefix_initialization_trial import load
from run_position_proxy_support import native_table

OUT=BASE/'camera_valid_near_prefix_trial'
FAILED=BASE/'near_prefix_initialization_trial'


def render():
    p=load(OUT/'protocol.json'); e=next(e for e in p['cases'] if e['tag']=='006')
    before=native_table(e);after=defaultdict(dict)
    for r in read(OUT/'case_006/candidate_native_observations.csv'):
        after[int(r['source_frame'])][r['player_slot']]=r
    first,last=10430,10610
    dump(OUT/'review_media_protocol.json',dict(tag='006',first_source_frame=first,last_source_frame=last,
        selection='Illustrative video around camera-eligible near initialization, fixed input and common initialization; chosen after results, not evaluation sampling.',
        native_source_dimensions=[3840,2160],display_image_dimensions=[1280,720],header_height=80,
        aspect_preserved=True,playback_speed=.5,video_fps=FPS/2,
        green='New near-prefix native boxes, not GT',cyan='Unchanged baseline native boxes',
        overlays='No interpolated boxes, pose points or future-coordinate backfill.'))
    dest=OUT/'006_early_pair_comparison.mp4'; assert not dest.exists()
    command=[r'C:\ffmpeg\bin\ffmpeg.exe','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','bgr24',
        '-s','1280x800','-r',str(FPS/2),'-i','pipe:0','-an','-c:v','libx264','-preset','fast','-crf','20',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(dest)]
    log=(OUT/'video_encoding.log').open('wb')
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
    cap=cv2.VideoCapture(p['source_video']);cap.set(1,first)
    mapping=[]; views=[]
    labels={'near_front':'N1','near_rear':'N2','far_rear':'F1','far_front':'F2'}
    stills={10437,10536,10546,10574,10575}
    for f in range(first,last+1):
        ok,frame=cap.read();assert ok and round(cap.get(1))==f+1
        frame=cv2.resize(frame,(1280,720),interpolation=cv2.INTER_AREA)
        for slot,r in after.get(f,{}).items():
            color=(50,230,50) if r.get('identity_source')=='near_pair_prefix_same_raw_ID' else (255,230,50)
            xy=[round(float(r[k])/3) for k in ['x1','y1','x2','y2']]
            cv2.rectangle(frame,tuple(xy[:2]),tuple(xy[2:]),color,2)
            cv2.putText(frame,f"{labels[slot]} raw{r['track_id']}",(xy[0],max(16,xy[1]-8)),cv2.FONT_HERSHEY_SIMPLEX,.55,color,2)
            cv2.circle(frame,(round(float(r['bottom_center_x'])/3),round(float(r['bottom_center_y'])/3)),4,color,-1)
        canvas=np.zeros((800,1280,3),np.uint8);canvas[80:]=frame
        text=[f'006 | source F{f} | original clock | playback 0.5x',
              f'Baseline native boxes: {len(before.get(f,{}))}/4 | Candidate: {len(after.get(f,{}))}/4 | green: additional near pair; cyan: unchanged']
        for i,line in enumerate(text):cv2.putText(canvas,line,(15,28+32*i),cv2.FONT_HERSHEY_SIMPLEX,.65,(245,245,245),1,cv2.LINE_AA)
        process.stdin.write(canvas.tobytes())
        if f in stills:
            name=f'006_F{f}_comparison.jpg';assert cv2.imwrite(str(OUT/name),canvas);views.append(canvas)
        mapping.append(dict(output_frame=f-first,source_frame=f))
    cap.release();process.stdin.close();assert process.wait()==0;log.close()
    write(OUT/'video_source_frame_map.csv',mapping)
    montage=np.vstack([cv2.resize(v,(960,600)) for v in views[1:3]])
    assert cv2.imwrite(str(OUT/'006_fixed_times_review.jpg'),montage)
    check=cv2.VideoCapture(str(dest));count=0
    while True:
        ok,frame=check.read()
        if not ok:break
        assert frame.shape[:2]==(800,1280);count+=1
    actual_fps=check.get(cv2.CAP_PROP_FPS);check.release()
    assert count==last-first+1 and abs(actual_fps-FPS/2)<.001
    dump(OUT/'review_media_verification.json',dict(status='passed',decoded_output_frames=count,
        actual_fps=actual_fps,all_original_images_resized_with_unchanged_aspect=True,
        audio_included=False,source_to_output_map_complete=True))


def finalize():
    reports=load(OUT/'result.json');p=load(OUT/'protocol.json')
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    rows=read(OUT/'fixed_time_availability_not_accuracy.csv')
    assert len(rows)==310
    changed=[r for r in rows if r['candidate_near']!=r['baseline_near'] or r['candidate_far']!=r['baseline_far']]
    assert len(changed)==20 and {r['tag'] for r in changed}=={'006'}
    assert all(r['candidate_far']==r['baseline_far'] for r in rows)
    initializations=[r for r in reports['cases'] if r['near_prefix_initialized_frame'] is not None]
    assert {r['tag'] for r in initializations}=={'005','006'}
    for tag in ['005','006']:
        r=next(r for r in initializations if r['tag']==tag)
        last=[x for x in read(OUT/f'case_{tag}/additional_near_prefix.csv') if int(x['source_frame'])==r['common_initialized_frame']-1]
        assert {x['player_slot']:int(x['track_id']) for x in last}==r['common_near_seed_ids']
    assert not (OUT/'case_009/additional_near_prefix.csv').exists()
    tests=[]
    for n in ['test_near_prefix_initialization.py','test_camera_valid_near_prefix.py']:
        result=subprocess.run([sys.executable,'-X','utf8',str(ROOT/n)],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        tests.append(dict(script=n,exit_code=result.returncode,output=result.stdout+result.stderr))
        (OUT/('executed_'+n)).write_text((ROOT/n).read_text(),encoding='utf-8')
    dump(OUT/'invariant_test_results.json',dict(tests=tests,synthetic_tests_not_real_world_accuracy=True))
    dump(OUT/'processing_decision.json',dict(
        ungated_variant='Rejected:009 close-up/dissolve seeded two large native boxes as court players. Failed outputs retained.',
        camera_conditioned_variant='Retain as development comparator only; not merged into current22 or the published position-reference evaluation.',
        benefit='006 fixed31 times now have two native near boxes at all31 instead of11. Adds40 native boxes at20 requested times, no extra far boxes.',
        remaining='006 far-pair initialization delay remains, four-complete times11/31. Native raw-ID continuity does not establish identity accuracy. Existing main-camera intervals are automated and not human-validated.',
        position_reference='P011 two new near native proxies vs known manual shoe reference:2.49px and15.04px in native4K. This label was already known, so a development diagnostic only.',
        next='Investigate far-side candidate availability/initial-seed geometry during006 fixed window; keep actual detector boxes and missing states. Do not widen gates or infer the hidden player just to make four outputs.',
        current22_replaced=False,source_clock_or_input_times_shifted=False,source_labels_updated=False,
        independent_test_completed=False))
    README='''# 手前ペアの先行初期化：主画面区間を条件とする開発比較

006で手前2人の出力まで奥側の初期化を待っていた問題を切り分けた。指定入力F10536〜10596の31時点は固定し、手前の実検出枠だけを先に出す。過去8連続ソースフレームで同じ2つのraw IDを確認し、以後は同じID・既存の移動／外観／大きさ条件・最大2フレーム間隔だけで継続する。新ID再取得、座標補間、確認窓への遡及出力は行わない。

## 結果

| 項目 | 従来 | 主画面条件付き候補 |
|---|---:|---:|
| 006の手前ペア開始フレーム | 10575 | 10437 |
| 006の指定31時点で手前2枠がある時点数 | 11 | 31 |
| 006の指定31時点で4枠がある時点数 | 11 | 11 |
| 他9区間の指定入力の変更 | — | 0 |

手前開始は138ソースフレーム、約2.30秒早い。指定時点20件に手前の実検出枠40個を追加した。ほかに005の指定入力より前の11フレームに22枠を追加。10区間すべてで従来の出力行と奥側の経路は保存されている。これは凍結済みの経路へ初期化前の行を加えた比較であり、全追跡器を再実行して一致したという意味ではない。先行初期化が起きた005と006では直前フレームから従来初期化までのスロット別raw IDも一致した。残る8区間は先行出力なし。

## 採用しなかった試行と画角条件

最初の画角条件なしの試行は `../near_prefix_initialization_trial/` に失敗ごと保存した。009のF21046付近には接写とコート全景のディゾルブがあり、画面全体を占める2人物枠を手前ペアとして初期化した。固定コートのホモグラフィを別画角へ適用しても、有効な人物選択にはならない。

条件付き案は元から存在する `analysis_ready_main_camera_segments.csv` の区間を入力として利用する。区間外では確認窓と先行追跡状態を捨てる。009の接写からの先行出力は0になった。この区間情報は幾何と背景の自動判定で、全行human_checked=false。将来画像を含むオフライン区間抽出を前提とし、新たに主画面識別を独立評価したわけでも、完全オンライン化したわけでもない。009の失敗を見て条件を追加した開発である。

## 位置参照と評価の限界

凍結済みP011（006／F10546）では、元の予測0点に対して手前2点が出た。人手の靴位置参照との差はネイティブ4K上で2.49px、15.04px（平均8.77px）。同一側内の順序なし対応であり、本人同定精度や実空間誤差ではない。この参照を見た後に方法を作ったため、独立評価として成功数へ算入しない。以前の参照評価一式も上書きしない。

31/31は候補の出現率である。006の奥側と4人全体の不足は未解決。8連続ソースフレームも、動画の重複フレームを含む短い確認窓であり、8件の独立証拠ではない。合成データ7検査は確認窓の因果性・ID変更の禁止・カメラ区間外の状態破棄等の実装を確認するだけで、映像上の追跡正解率を保証しない。

## 判断と成果物

主画面条件付き案は別の開発比較として保存し、current22、得点側ラベル、開始終了ラベル、指定入力時点、公開済み位置参照評価は変更しない。次は006の奥側について、検出自体が不足するのか初期選択の条件が狭いのかを切り分ける。

- `006_early_pair_comparison.mp4`：0.5倍速、緑は追加手前枠、シアンは従来枠。元映像の縦横比を保存。
- `fixed_time_availability_not_accuracy.csv`：全10区間×31時点を欠測も含めて保存。
- `P011_development_reference_diagnostic.json`：既知参照との差と評価上の留保。
- `protocol.json`、`final_verification.json`、`publication_hashes.json`：条件・保護対象・検証と出力の固定。
'''
    (OUT/'README.md').write_text(README,encoding='utf-8')
    (FAILED/'README.md').write_text('''# 画角条件なしの手前先行初期化：不採用

006では指定入力の手前2枠を補えたが、009の接写・ディゾルブ中F21046で画面全体を占める人物を手前ペアとして初期化した。固定コートの領域判定が別画角に適用される失敗であり、raw IDが8ソースフレーム続くことだけでは本人同定も画角も保証できない。このままの一般採用を棄却する。

失敗画像、追加候補7行、条件、実行コード、全10区間の結果を削除せず保存した。current22は変更していない。主画面条件を追加した後続比較は `../camera_valid_near_prefix_trial/` に独立して保存した。後続比較も同一試合内の開発であり、未知動画への精度改善を証明しない。
''',encoding='utf-8')
    dump(OUT/'final_verification.json',dict(status='passed',contexts=10,requested_times=310,
        changed_requested_times=20,changed_contexts=['006'],added_requested_native_boxes=40,
        additional_prefix_contexts=['005','006'],compatible_observed_raw_ID_handoffs=2,
        no_independent_early_initialization_in_remaining_contexts=8,
        all_original_native_observations_and_far_paths_unchanged=True,
        all_protected_hashes_unchanged=True,synthetic_invariant_tests_passed=7,
        media_decode_verified=True,current22_and_published_reference_evaluation_unchanged=True,
        independent_accuracy_claim=False))
    for dest in [OUT,FAILED]:
        (dest/'executed_finalize_near_prefix_trials.py').write_text(Path(__file__).read_text(),encoding='utf-8')
        hashes={str(n.relative_to(dest)):sha(n) for n in sorted(dest.rglob('*')) if n.is_file() and n.name!='publication_hashes.json'}
        dump(dest/'publication_hashes.json',hashes)
        assert all(sha(dest/n)==h for n,h in hashes.items())
    print('Published both trials; failed variant preserved; original inputs and evaluation unchanged.',flush=True)


if __name__=='__main__':{'render':render,'finalize':finalize}[sys.argv[1]]()
