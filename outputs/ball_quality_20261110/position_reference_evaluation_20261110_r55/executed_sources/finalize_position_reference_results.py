"""Retain the completed manual-reference evaluation and evidence-bounded decision."""
import json,math,shutil,sys
from pathlib import Path
from collections import Counter
import numpy as np,cv2,matplotlib
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,sha,dump

STAGE=BASE/'position_reference_evaluation_20261110_r55'

def main():
    assert not (STAGE/'final_verification.json').exists();p=json.loads((STAGE/'protocol.json').read_text(encoding='utf-8'));assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items());labels=json.loads((STAGE/'human_reference_labels.json').read_text(encoding='utf-8'));assert labels['revision']==55
    count=Counter(e['state'] for group in labels['labels'].values() for e in group.values());assert count=={'point':48,'unplaceable':4};assert len(labels['labels'])==13
    rows=read(STAGE/'evaluation/per_context_side_results.csv');assert len(rows)==48;summary=read(STAGE/'evaluation/coverage_and_conditional_error_summary.csv');points=read(STAGE/'per_matched_reference_point_diagnostic.csv');a=json.loads((STAGE/'analysis_result.json').read_text(encoding='utf-8'));assert a['independently_checked_numeric_values']==210
    for policy in ['retain_native_proxy','withhold_body_warning_proxy']:
        uniform=[r for r in rows if r['sampling_group']=='uniform_one_per_context' and r['position_policy']==policy];assert len(uniform)==20 and sum(r['evaluation_state']=='scored' for r in uniform)==17 and sum(r['position_pair_evaluable']=='1' for r in uniform)==18 and sum(r['evaluation_state']=='prediction_pair_incomplete' for r in uniform)==1 and sum(r['evaluation_state']=='reference_positions_unavailable' for r in uniform)==2
        scored=[r for r in points if r['sampling_group']=='uniform_one_per_context' and r['position_policy']==policy];assert len(scored)==34;assert math.isclose(sum(float(r['distance_px']) for r in scored)/34,21.50783238431649,abs_tol=1e-12)
    for row in rows:
        if row['evaluation_state']!='scored':assert all(row[k]=='' for k in ['mean_point_error_px','mean_point_error_over_image_height','centroid_x_error_norm','centroid_y_error_norm','x_gap_error_norm','y_gap_error_norm'])
    before={(r['image_id'],r['camera_side']):r for r in rows if r['sampling_group']=='uniform_one_per_context' and r['position_policy']=='retain_native_proxy'};after={(r['image_id'],r['camera_side']):r for r in rows if r['sampling_group']=='uniform_one_per_context' and r['position_policy']=='withhold_body_warning_proxy'}
    for k in before:
        if before[k]['evaluation_state']=='scored':assert before[k]['mean_point_error_px']==after[k]['mean_point_error_px']
    known=[r for r in rows if r['sampling_group']=='known_body_warning_challenge'];assert len(known)==8
    assert sum(r['evaluation_state']=='reference_positions_unavailable' for r in known if r['position_policy']=='retain_native_proxy')==3
    counter=next(r for r in known if r['image_id']=='P012' and r['position_policy']=='retain_native_proxy');removed=next(r for r in known if r['image_id']=='P012' and r['position_policy']=='withhold_body_warning_proxy');assert counter['evaluation_state']=='scored' and math.isclose(float(counter['mean_point_error_px']),7.560126491683436,abs_tol=1e-12) and removed['evaluation_state']=='prediction_pair_incomplete'
    diagnostic=json.loads((STAGE/'diagnostic_selection.json').read_text(encoding='utf-8'));assert len(diagnostic['items'])==3 and all(sha(STAGE/r['image'])==r['image_sha256'] for r in diagnostic['items'])
    previous_files=0
    for folder in [BASE/'current22_player_inputs',BASE/'cohort_tracking/case_019',BASE/'pair_initialization_transfer_019',BASE/'far_identity_guard_development',BASE/'far_spatial_guard_development',BASE/'body_part_localization_probe',BASE/'torso_temporal_association_trial',BASE/'position_proxy_support_trial',BASE/'identity_position_bundle']:
        for name,h in json.loads((folder/'publication_hashes.json').read_text(encoding='utf-8')).items():assert sha(folder/name)==h;previous_files+=1
    current=BASE/'current22_player_inputs';cr=json.loads((current/'result.json').read_text(encoding='utf-8'));assert len(read(current/'confirmed_human_outcomes.csv'))==11 and cr['fixed30_constructed_inputs']==19 and cr['next_candidate']=='match1_rally_window_019'
    dump(STAGE/'runtime.json',dict(python=sys.version,numpy=np.__version__,opencv=cv2.__version__,matplotlib=matplotlib.__version__,new_model_inference_performed=False))
    decision=dict(status='manual_reference_pilot_complete_warning_only_policy_retained',automatic_position_mask_adopted=False,reason='Uniform sample has identical conditional error under both policies. In the only evaluable known-warning pair the original agrees with manual reference and masking makes the pair unavailable. Other warning examples have unplaceable references, not proven coordinate errors.',warning_layer_retained_as_diagnostic=True,known_warning_outcomes_not_called_precision=True,coarse_manual_reference_not_surveyed_ground_truth=True,next_priorities=['Investigate006 initialization delay while keeping originally requested input times.','Diagnose the bbox-bottom versus shoe-reference definition and direction of differences before any correction.','Use separate data/references for later method selection validation; do not tune an offset or mask threshold to these13 images.'],canonical_tracking_replaced=False,current22_inputs_unchanged=True,confirmed_human_winner_labels=11,new_position_reference_fields=52,position_reference_points=48,position_unplaceable_fields=4,fixed30_constructed_inputs=19,fixed30_pending_inputs=11,next_unresolved_candidate='match1_rally_window_019',ID_continuity_accuracy_established=False,independent_match_accuracy_established=False,outcome_model_fitted=False)
    dump(STAGE/'processing_decision.json',decision)
    report=dict(status='passed',scope='Frozen human reference completeness, independent numeric recalculation, denominators and prior-artifact preservation; not independent-match tracking validation',reference_revision=55,reference_images=13,reference_fields=52,point_fields=48,unplaceable_fields=4,pending_fields=0,uniform_contexts=10,uniform_context_side_pairs=20,uniform_reference_position_pairs=18,uniform_scored_pairs=17,uniform_scored_points=34,uniform_scored_contexts=9,uniform_conditional_mean_distance_px=21.50783238431649,uniform_conditional_median_distance_px=18.200536626275877,known_warning_pairs=4,known_warning_pairs_with_unplaceable_reference=3,known_warning_pairs_with_two_reference_points=1,uniform_error_identical_between_policies=True,known_warning_counterexample_case019_preserved=True,numeric_values_independently_checked=210,diagnostic_examples_AI_inspected=3,scientific_figure_AI_inspected=True,previous_publication_files_unchanged=previous_files,manual_references_modified_by_agent=False,no_label_based_numeric_tuning=True,no_model_training=True,current22_inputs_and11_winner_labels_unchanged=True,automatic_position_mask_adopted=False)
    dump(STAGE/'final_verification.json',report)
    text='''# 手入力の画像位置参照による予備評価：自動除外は採用しない

入力完了後の参照revision55を別保存し、既に固定した2条件を評価した。13画像×4欄の52項目が完了し、点を置いたもの48、位置を置けないもの4、未入力0。参照や「位置不明」をモデルに合わせて修正していない。元の人手参照ファイルも変更していない。

無作為群は同じ10区間から各1画像。既知の身体重なり警告4例を別の問題群とし、1画像は両群に含まれるので、画像は13枚。両群を混ぜた精度や4例の成功率は出さない。画面には推定点・推定枠を表示しなかったが、注釈者は以前の開発例を見ているため完全な観察者盲検ではない。対象は同一試合内の開発データであり、独立試合のテストではない。

参照は左右の靴の中央x・低い方の靴底yを目安にした粗い画像上の代表点。測量的な接地点の正解ではなく、厳密な接地判定も要求していない。1人による参照で、反復注釈や注釈者間のばらつきは未測定。以下の値はこの参照点に対する距離であり、物理的なコート位置精度や本人同定率とは異なる。

## 無作為10画像

|項目|結果|
|---|---:|
|手前・奥の2人組|20組|
|参照位置を2人とも置けた組|18組|
|参照と予測が2点ともそろい、距離を計算できた組|17組・34選手点・9区間|
|参照はそろったが予測が不足|1組（006の手前側）|
|参照が片方位置不明|2組（006・017の奥側）|
|計算できた34点の平均距離|21.51px|
|同じ34点の中央値|18.20px|

画像サイズは3840×2160。34点は34個の独立試合ではなく、同一画像・同一区間の選手点を含む。未取得を0pxとして埋めず、不明・欠測を残した。006は初期化前のため手前も奥も予測0点で、奥側は参照も片方不明。誤差を計算できない3組を無かったことにしていない。

元の位置維持と身体警告位置の除外で、計算対象17組の平均距離は全く同じ。完全な予測ペア数自体は18組から17組へ減るが、減った017の奥側は参照も片方不明で、元から数値誤差の対象ではない。条件付き誤差だけから警告除外が有効とも無効とも一般化しない。

参考診断として、手前18点の平均距離は30.17px、奥16点は11.76px。手前は18点中17点で元の枠下端が参照より画像下方向にあり、平均y差は+22.85px。奥の平均y差は+6.04px。撮影上の大きさ、検出枠の下端定義、参照を靴のどこへ置いたかの違いも含まれる。奥の方が物理的に高精度と断定せず、この値をそのまま差し引く補正もしない。

## 既知警告4場面

3場面では奥側の参照の片方が「位置不明」だった。これは警告と参照可能性が整合する例だが、推定座標の誤りを数値確認できた3件ではない。また、対象本人についての警告精度3/4を示す結果としては扱わない。

残る019F46396では2人とも参照位置があり、元のペア平均距離は7.56px、警告対象の点単独では約4.3pxだった。自動除外するとこのペアは位置不足になり、参照に近い点まで失う反例が得られた。したがって、身体近似への重なりをそのまま自動欠測へ変換する処理は採用しない。注意情報の別表として保持する。

警告4例は警告が出た後に選んだ既知の問題例で、代表性のある精度評価ではない。017F42138は無作為群にも含まれるが、独立した追加成功例として重複計上しない。全13画像の参照を使ったF1/F2の継続ID精度は評価していない。同じ側の2点を順序なしで対応付けた配置位置の評価である。

位置距離とペアの中心・間隔の差を210項目独立計算で照合した。17組の距離を無作為群の両条件で比較し、同じことを確認。diagnostic_examples.jpgは、無作為群の手前・奥で最大の点距離となった例と、019の除外反例を保存したもの。結果を見た後の原因診断用であり、精度標本ではない。参照点は変更していない。position_reference_diagnostic.png/.pdfは区間別の条件付き距離と符号付き差の図。

この評価は、前段の人物追跡用補助コスト自体の有効性を直接検証したものではない。比較したのは、同じ人物候補経路から位置を残すか警告位置だけ欠測にするかの2条件。13枚の小標本から別試合での精度、試合全体の本人同定、得点側との関係や失点理由は結論づけない。

次の改善対象は、006の初期化遅れと、枠下端・靴の代表点の位置定義差である。今の13枚に合わせて閾値や座標オフセットを調整せず、新方式を試すなら参照に使ったデータと後の確認データを区別する。注釈した位置を正解として入力表へ直接上書きすることもしない。

元の追跡・current22_player_inputs・確認済み11得点ラベルは保持。固定30群は入力作成19件、未完了11件で019未解決。今回は52件の位置参照を得たのであり、得点ラベルを52件へ増やしたわけではない。モデル学習、得点予測性能や球の追加価値比較は行っていない。
'''
    (STAGE/'README.md').write_text(text,encoding='utf-8');archive=STAGE/'executed_sources';archive.mkdir()
    for name in ['evaluate_position_reference_pilot.py','evaluate_completed_position_reference.py','analyze_position_reference_results.py','finalize_position_reference_results.py']:shutil.copy2(ROOT/name,archive/name)
    day=BASE/'README.md';content=day.read_text(encoding='utf-8');marker='手入力の位置参照revision55を評価';assert marker not in content;day.write_text(content+'\n\n'+marker+'。13画像52欄は48点・位置不明4件で完了。固定済み2条件を評価し、無作為10画像の20組中17組34点で条件付き平均距離21.51px（3840×2160）・中央値18.20px。両条件の計算対象は同じで距離差なし。既知警告4場面は3場面で参照不明、1場面019F46396では元が参照と近いのに除外される反例となったため、自動欠測は採用せず注意情報に限定。006の初期化前予測0も欠測として保持。参照は粗い画像位置で、独立試合・継続ID・接地位置の精度ではない。210数値を独立照合。current22・11得点ラベル・19/30保持。詳細position_reference_evaluation_20261110_r55/README.md。\n',encoding='utf-8')
    dump(BASE/'latest_position_reference_evaluation.json',dict(stage=str(STAGE),label_revision=55,status='completed_development_pilot',automatic_position_filter_adopted=False,uniform_conditional_mean_distance_px=21.50783238431649,uniform_scored_points=34,current22_unchanged=True))
    dump(STAGE/'publication_hashes.json',{str(n.relative_to(STAGE)):sha(n) for n in sorted(STAGE.rglob('*')) if n.is_file() and n.name!='publication_hashes.json'});print(report)

if __name__=='__main__':main()
