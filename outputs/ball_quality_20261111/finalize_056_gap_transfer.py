"""Diagnose the fixed transfer failure without retuning or inventing observations."""
from pathlib import Path
import shutil
import numpy as np
import build_044_spatiotemporal_pilot as b
from transfer_056_reacquisition_other_gaps import OUT,FROZEN

def main():
    assert not (OUT/'publication_hashes.json').exists()
    method=b.load(FROZEN/'protocol.json');radius=method['native_search_radius_px'];threshold=method['candidate_threshold']
    cache=np.load(OUT/'full_heatmaps.npz')
    maps={int(f):hm for f,hm in zip(cache['source_frames'],cache['heatmaps'],strict=True)}
    # Continuous heatmap-pixel centers in native coordinates; diagnostic only.
    yy,xx=np.mgrid[:288,:512];xx=(xx+.5)*1920/512;yy=(yy+.5)*1080/288
    diagnostic=[]
    for r in b.read(OUT/'proposed_candidates.csv'):
        f=int(r['source_frame']);hm=maps[f];gx=float(r['search_reference_x']);gy=float(r['search_reference_y'])
        mask=(xx-gx)**2+(yy-gy)**2<=radius**2
        peak=float(hm[mask].max());global_peak=float(hm.max())
        diagnostic.append(dict(source_frame=f,search_disk_peak=peak,global_peak=global_peak,
            search_disk_pixels_above_frozen_threshold=int(np.count_nonzero(hm[mask]>threshold)),
            failure_description='no_above_threshold_heatmap_pixels_in_search_disk' if peak<=threshold else
                'heatmap_pixels_present_but_no_component_center_passed_fixed_gate',
            diagnostic_is_not_a_ball_candidate=1))
    b.write(OUT/'failure_diagnostics.csv',diagnostic)
    result=b.load(OUT/'result.json')
    protected=b.load(OUT/'protocol.json')['protected_input_hashes']
    assert all(b.sha(Path(p))==digest for p,digest in protected.items())
    assert result['new_unconfirmed_candidates']==0
    b.dump(OUT/'verification.json',dict(
        result,protected_inputs_unchanged=True,exact_frozen_parameters_reused=True,
        searched_settings=1,additional_threshold_or_radius_tuning=False,
        failure_diagnostics_count=len(diagnostic),
        no_above_threshold_pixels_in_search_disk=sum(r['search_disk_pixels_above_frozen_threshold']==0 for r in diagnostic),
        global_output_is_not_ball_accuracy=True,missing_is_not_physical_absence=True))
    b.dump(OUT/'decision.json',dict(
        status='fixed_transfer_complete_no_additional_candidates',
        retain_previous_four_human_confirmed_points=True,
        adopt_as_general_recovery_method=False,
        reason='The unchanged setting produced no candidates in all10eligible other gaps. Previously confirmed four points remain useful local repairs, not generalization evidence.',
        next_technical_question='Does the ball still have usable same-frame image evidence at native resolution when the full-image512x288 model has no usable local heatmap response?',
        next_comparison='Keep this failed setting as baseline. A separately declared native-resolution crop trial would require checks on both missing targets and already correct/known wrong candidates, with immutable source mappings and no automatic adoption.',
        no_further_same_trial_parameter_search=True))
    (OUT/'README.md').write_text(
        '# 056：同じ設定を他の欠測へ適用した結果\n\n'
        '結論：追加候補なし。以前の終盤4点の本人確認は保持するが、一般的な再取得方法としての有効性は確認できない。\n\n'
        '既存の376時点の注釈にあった未取得42時点と、別物体と確認されたF154780の1時点を合わせた43時点を全件整理した。'
        '接触・バウンドのAI暫定区間をまたぐ14時点、両端間隔が12フレームを超える18時点、'
        '既知の誤検出1時点を除き、同一飛行区間内の10時点を探索した。\n\n'
        '検出成分閾値0.1・探索半径32px・最大間隔12フレームを変更していない。'
        '前後の入力位置には人手でクリック修正された点も含み、完全自動追跡の独立評価ではない。'
        '元モデル36フレームを同じバッチ・AMPで再推論し、座標と信頼度を完全再現した。'
        '対象10時点のうち、条件を満たす新候補は0時点。追加の調整や位置補間は実施しなかった。\n\n'
        'failure_diagnostics.csv は探索範囲内の検出反応を数値で確認する診断で、球位置出力ではない。'
        '原画像で本当に球が見えないとは結論しない。'
        '次に調べるべき点は、全体を512×288へ縮小した入力では消える球の情報が、元画像の局所範囲に残るかである。'
        '低解像度の反応が不足している場所へ、時系列の参考線だけで位置を埋めることはしない。\n',encoding='utf-8')
    shutil.copy2(__file__,OUT/'executed_finalize.py')
    b.dump(OUT/'publication_hashes.json',{p.name:b.sha(p) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='publication_hashes.json'})
    print(b.load(OUT/'verification.json'))

if __name__=='__main__':main()
