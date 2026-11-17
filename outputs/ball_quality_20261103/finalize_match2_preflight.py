"""Publish source/cadence verification, explicitly before independent tracking."""
import sys,json,subprocess,shutil
from pathlib import Path
from collections import Counter
import cv2
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,write,sha,dump
from prepare_match2_external_validation import OUT,VIDEO,load


def main():
    p=load(OUT/'preflight_protocol.json');s=load(OUT/'source_registration.json');cadence=load(OUT/'cadence_result.json')
    assert sha(VIDEO)==s['source_video_sha256'] and all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    assert len(cadence['windows'])==5 and all(r['decoded_frames']==120 for r in cadence['windows'])
    assert all(r['exact_thumbnail_duplicate_pairs']==0 for r in cadence['windows'])
    manifest=load(OUT/'raw_reference_pilot/manifest.json')['items'];assert len(manifest)==9
    for r in manifest:
        path=OUT/'raw_reference_pilot'/r['filename'];im=cv2.imread(str(path));assert im.shape[:2]==(1080,1920) and sha(path)==r['image_sha256']
    camera_states=['closeup','closeup','closeup','alternate_oblique_view','alternate_oblique_view','main_view_candidate','main_view_candidate','main_view_candidate','closeup']
    reviews=[dict(clip_id=r['clip_id'],reference_index=r['reference_index'],source_frame=r['source_frame'],AI_view_state=state,
        independent_human_view_label=False,identity_GT_created=False,position_GT_created=False) for r,state in zip(manifest,camera_states)]
    dump(OUT/'raw_camera_review_AI_only.json',dict(reviewer='AI assistant, not independent human GT',items=reviews,
        counts=dict(Counter(camera_states)),
        notes='Three fixed time clips contain close-ups, another camera angle and frontal wide views. Keep all frames and cases. These preflight samples are not automatically valid in-play rallies or nine usable four-player reference frames. Even frontal shots need camera/zoom consistency checks before sharing a homography.'))
    tests=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'test_match_frame_coordinate_adapter.py')],capture_output=True,text=True)
    assert tests.returncode==0,tests.stderr
    dump(OUT/'coordinate_adapter_protocol.json',dict(source_dimensions=[1920,1080],canonical_dimensions=[3840,2160],
        coordinate_scale=2,aspect_changes_rejected=True,source_frame_and_timestamps_preserved=True,
        missing_body_only_coordinates_preserved=True,source_video_not_resampled_or_overwritten=True,
        image_upscale_adds_no_detail=True,
        new_native_camera_calibration_available=False,old_camera_calibration_reused=False,
        homogeneous_coordinate_conversion_is_not_camera_calibration=True,
        legacy_image_region_gates_still_camera_dependent=True,
        tracker_run_ready=False,synthetic_tests_passed=7,test_output=tests.stdout+tests.stderr,
        frozen_before_tracking_outputs=True,
        hashes={str(n):sha(n) for n in [ROOT/'match_frame_coordinate_adapter.py',ROOT/'test_match_frame_coordinate_adapter.py']}))
    gates=[dict(file='tracking_quality_20261102/four_player_association.py',condition='Image foot y divided by2160; near/far recovery gates are camera dependent.'),
        dict(file='tracking_quality_20261102_stage4/temporal_tracker.py',condition='Initial selector called with3840,2160; court geometry plus overlap/past8 confirmation required.'),
        dict(file='tracking_quality_20261102_stage9/adaptive_near.py',condition='Reference-height floor multiplied by2160; native1080 boxes cannot be used directly.'),
        dict(file='ball_quality_20261103/run_body_part_localization_probe.py',condition='Court/far-context image crops use canonical4K pixel coordinates; a new camera calibration is distinct from resolution conversion.')]
    dump(OUT/'legacy_compatibility_audit.json',dict(findings=gates,
        conclusion='Coordinate unit mismatch has an explicit tested adapter. New-camera/zoom/alternate-view eligibility and seed/recovery geometry are unresolved prerequisites. Do not equate1080 adaptation with full cross-camera validity.'))
    (OUT/'README.md').write_text('''# match2：別試合評価の取込・事前確認

元ファイルは `softtennis_frontcourt_study/data/raw/match2.mp4`。2024年全日本選手権男子決勝（内本隆文・内田理久 対 船水颯人・上松俊貴）で、開発中の2025年男子決勝とは全4選手が異なる。元ファイルは変更せずSHA-256を固定した。

## 取込確認

- 1920×1080、16:9、AV1。動画は182506フレーム、3044.808秒（約50分45秒）。
- 名目・平均フレームレートは60000/1001≒59.94fps。OpenCVでも読み出せる。
- 固定した30・300・900・1800・2700秒付近で各120枚、計600枚を読み出した。ROI付き縮小グレースケールの隣接595ペアに、画素が完全一致するペアは0。
- 偶奇フレームごとの変化量の中央値比は約0.94〜1.00。抽出範囲では毎2枚の規則的な重複を示していない。撮影時の真のfps、フレーム補間の有無、映像全体の更新頻度を証明する測定ではない。接写・カメラ移動も含むため、競技動作だけの精密な時間評価とも区別する。

## 追跡結果を見る前に固定した区間

60〜75秒、300〜315秒、900〜915秒の3つの固定時間窓。各31時点の出力有無と、各+1・7.5・14秒の原画像参照候補を固定した。9画像は元の1920×1080 PNGで、検出枠やモデルの点を表示していない。ラリー3本と認定したものではなく、接触・得点側も未確認。1試合の3区間であり、独立した3試合とは数えない。

原画像を点検すると、最初の窓の参照候補は3枚とも接写。2番目は横からの画角2枚と主画面候補1枚、3番目は主画面候補2枚と接写1枚だった。これはAIの画角所見であり、人物・座標の独立正解ではない。成功しそうな画像へ置き換えず保存する。プレー前の画像や別画角を、有効なラリー内の4人追跡データへ自動昇格させない。

## 解像度とカメラの対応

既存コードには3840×2160と、旧カメラの画面内y範囲・切り出し範囲が残る。1080pの枠をそのまま渡すと、人物選択・高さの基準・位置が崩れる。

1920×1080と既存の内部座標の間を2倍／1/2倍で明示変換する小さなアダプターを作り、7件の合成検査を通した。ID・元フレーム番号・時刻・欠測を維持し、画像比率が違う場合は停止する。姿勢点の信頼度は座標と一緒に拡大しない。画像を拡大しても解像情報は増えない。

この変換は新カメラの校正ではない。旧試合のホモグラフィや画面内の前後判定を、新しい正面・横画角へそのまま流用しない。主画面でもズーム差の確認が必要。まだ新動画の人物検出・姿勢推定・追跡・独立精度評価を実行したわけではない。

## 次の手順

1. 試合開始後の主画面とカメラ切替を原映像で確認し、新カメラの幾何対応を記録する。固定時間窓は事前点検として残し、ラリー分析には別途プレー区間を固定する。
2. 新映像で必要な解像度・画角の処理と比較する凍結済み方式を、追跡出力を見る前に決める。
3. 初期化失敗・欠測・IDの混同を含めて出力し、人手参照で評価する。新試合の失敗を見て方法を修正した場合、その結果を最終的な独立評価として扱わない。

得点側ラベル・元のmatch1入力・current22・これまでの公開済み比較は変更していない。取り込めたこと、候補が出ること、本人同定や位置が正しいことを分けて記録する。
''',encoding='utf-8')
    dump(OUT/'processing_decision.json',dict(status='source_and_preflight_verified_before_tracking',
        file_received=True,decoded_and_hashed=True,new_match_tracking_outputs_created=False,
        independent_accuracy_evaluation_completed=False,
        next='Resolve main-camera/zoom eligibility and new-camera calibration before applying the frozen tracker. Review actual in-play chronology before rally selection; preserve the preflight windows including closeups.',
        original_videos_or_old_labels_changed=False))
    dump(OUT/'final_verification.json',dict(status='passed',source_sha256=s['source_video_sha256'],source_dimensions=[1920,1080],
        source_frames=182506,source_nominal_fps='60000/1001',cadence_windows=5,cadence_frames=600,
        raw_reference_pngs_verified=9,reference_model_overlays=False,human_ground_truth_created=False,
        frozen_fixed_windows=3,availability_times=93,coordinate_adapter_tests_passed=7,
        new_video_tracking_outputs_created=False,protected_previous_work_unchanged=True))
    for name in ['prepare_match2_external_validation.py','finalize_match2_preflight.py','match_frame_coordinate_adapter.py','test_match_frame_coordinate_adapter.py']:
        shutil.copy2(ROOT/name,OUT/('executed_'+name))
    hashes={str(n.relative_to(OUT)):sha(n) for n in sorted(OUT.rglob('*')) if n.is_file() and n.name!='publication_hashes.json'}
    dump(OUT/'publication_hashes.json',hashes)
    assert all(sha(OUT/n)==h for n,h in hashes.items()) and all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    assert sha(VIDEO)==s['source_video_sha256']
    print('Published match2 preflight; source and old research inputs unchanged; calibration remains explicit prerequisite.',flush=True)


if __name__=='__main__':main()
