"""Verify the matched soft-cost trial without adopting unvalidated positions."""
import math,shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,write,sha,dump
from run_body_part_localization_probe import load
from run_torso_temporal_trial_cached import OUT,MODES,load_pose

FIELDS=['source_frame','player_slot','track_id','x1','y1','x2','y2','bottom_center_x','bottom_center_y']
key=lambda r:tuple(r[k] for k in FIELDS)

def main():
    assert not (OUT/'final_verification.json').exists();p=load(OUT/'protocol.json');r=load(OUT/'result.json');assert len(r['cases'])==len(p['cases'])==10 and r['baseline_exact_replays']==10
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    transitive=load(BASE/'far_spatial_guard_development/protocol.json')['protected_input_hashes'];assert all(sha(Path(n))==h for n,h in transitive.items());assert sha(Path(p['source_video']))==p['source_video_sha256']
    amendment=load(OUT/'execution_amendment.json');assert sha(ROOT/'run_torso_temporal_trial.py')==amendment['original_runner_sha256'];assert sha(ROOT/'run_torso_temporal_trial_cached.py')==amendment['cached_runner_sha256'];assert sha(OUT/'protocol.json')==amendment['source_policy_protocol_sha256'];assert all(sha(Path(n))==h for n,h in amendment['prior_output_hashes'].items())
    outputs=checked=pose_rows=diagnostic_rows=positive_cost_rows=0;changes=[];controls=[];prefix_reports=[];metadata={r['rally_candidate_id'][-3:]:r for r in read(BASE/'current22_player_inputs/rally_metadata.csv')};pose_proof=[];summaries=[]
    for e in p['cases']:
        tag=e['tag'];dest=OUT/f'case_{tag}';native={(r['source_frame'],r['track_id']):r for r in read(Path(e['raw_csv']))};baseline=read(dest/'baseline/far_pair_path.csv');bk={key(r) for r in baseline};assert [key(r) for r in baseline]==[key(r) for r in read(Path(e['reference_far_path']))]
        near=[r for r in read(Path(e['temporal_folder'])/'selected_four.csv') if r['player_slot'].startswith('near_')];near_keys={key(r) for r in near};near_native={(r['source_frame'],r['track_id']):r for r in near};mode_sets=[]
        poses=load_pose(tag)
        data=np.load(dest/'pose_tensors.npz');boxes=data['boxes'];kp=data['keypoints'];features=data['torso_features'];pm=read(dest/'pose_metadata.csv');assert len(pm)==len(boxes)==len(kp)==len(features);assert boxes.shape[1:]==(5,) and kp.shape[1:]==(17,3) and features.shape[1:]==(192,);assert np.isfinite(boxes).all() and np.isfinite(kp).all()
        for row in pm:
            f=int(row['source_frame']);i=int(row['array_row']);assert row['mode'] in e['pose_schedule'][str(f)];assert np.all((kp[i,:,2]>=0)&(kp[i,:,2]<=1))
            available=row['torso_available']=='1';assert available==bool(np.isfinite(features[i]).all())
            if available:assert row['eligible']=='1' and int(row['feature_pixels'])>=12 and np.all(kp[i,[5,6,11,12],2]>=.5) and abs(float(features[i].sum())-1)<1e-5
            else:assert np.isnan(features[i]).all()
        pose_rows+=len(pm);gallery=load(dest/'gallery.json');proof=read(dest/'initial_pose_anchor_correspondence.csv');assert sum(gallery['anchors'].values())==len(proof)
        for row in proof:
            assert e['initialization']['window_start']<=int(row['source_frame'])<=e['initialization']['frame'];assert int(row['native_seed_id'])==e['initialization']['seed_ids'][row['slot']];assert float(row['iou'])>=.5
        pose_proof.append(dict(tag=tag,**gallery['anchors'],negative_initial_pose_features=gallery['unselected_initial_pose_features']))
        diags=read(dest/'candidate_torso_cost_diagnostics.csv');lookup={(row['source_frame'],row['native_track_id']):row for row in diags};assert len(lookup)==len(diags)
        for row in diags:
            assert (row['source_frame'],row['native_track_id']) in native;value=float(row['extra_cost']);assert 0<=value<=.8 and row['hard_veto']==row['pose_coordinates_exported']=='0'
            if row['state']=='compared':assert math.isclose(value,.8*max(0,float(row['far_team_distance'])-float(row['negative_distance'])-.04),abs_tol=1e-12)
            else:assert value==0
            if row['state']=='unsampled':assert str(row['source_frame']) not in e['pose_schedule']
            if value>0:positive_cost_rows+=1
        diagnostic_rows+=len(diags)
        for mode in ['baseline',*MODES]:
            rows=read(dest/mode/'far_pair_path.csv');seen=defaultdict(set);seen_slot=set();samepose=defaultdict(list)
            for row in rows:
                original=native[row['source_frame'],row['track_id']];assert all(row[k]==original[k] for k in ['x1','y1','x2','y2','bottom_center_x','bottom_center_y','confidence']);assert (row['source_frame'],row['player_slot']) not in seen_slot;seen_slot.add((row['source_frame'],row['player_slot']));assert row['track_id'] not in seen[row['source_frame']];seen[row['source_frame']].add(row['track_id'])
                assert row['position_source']=='detected_box'
                if mode.endswith('pose_exclusive'):
                    d=lookup[row['source_frame'],row['track_id']]
                    if d['pose_index']!='':samepose[row['source_frame']].append(d['pose_index'])
            assert all(len(v)==len(set(v)) for v in samepose.values());outputs+=1;checked+=len(rows)
            nk={key(row) for row in rows}
            if mode!='baseline':
                assert not nk-bk;mode_sets.append(nk)
                for row in baseline:
                    if key(row) not in nk:changes.append(dict(tag=tag,mode=mode,**row))
        assert mode_sets[0]==mode_sets[1]
        if tag!='019':assert mode_sets[0]==bk;controls.append(tag)
        cp=BASE/f'cohort_tracking/case_{tag}/protocol.json'
        times=load(cp)['requested_input_frames'] if cp.exists() else list(range(int(metadata[tag]['input_first_source_frame']),int(metadata[tag]['input_end_source_frame'])+1,2));assert len(times)==31
        assert {row for row in bk if int(row[0]) in times}=={row for row in mode_sets[0] if int(row[0]) in times}
        prefix_reports.append(dict(tag=tag,requested_frames=times,native_far_boxes_unchanged=True,near_path_unchanged=True,feature_window_shifted=False,features_adopted=False))
        summaries.append(dict(tag=tag,candidate_rows=len(diags),positive_cost_rows=sum(float(row['extra_cost'])>0 for row in diags),near_rear_anchor_count=gallery['anchors']['near_rear'],baseline_far_rows=len(baseline),trial_far_rows=len(mode_sets[0]),newly_selected_rows=0))
    assert len(controls)==9 and outputs==30 and checked==37506 and pose_rows==27913 and positive_cost_rows==112
    expected={('019',46501),('019',46502),('019',46503),('019',46505),('019',46665),('019',46666)}
    assert len(changes)==12 and {(row['tag'],int(row['source_frame'])) for row in changes}==expected
    write(OUT/'all_removed_native_observations.csv',changes);write(OUT/'per_case_cost_and_output_diagnostic.csv',summaries);write(OUT/'initial_pose_anchor_counts.csv',pose_proof);dump(OUT/'all10_prefix_native_box_verification.json',dict(cases=prefix_reports,observed_ground_position_accuracy_certified=False))
    review=load(OUT/'all_change_review/manifest.json');assert len(review['items'])==16 and all(sha(OUT/'all_change_review'/row['image'])==row['sha256'] for row in review['items']);findings=[]
    for row in review['items']:
        tag,f=row['tag'],row['source_frame'];note=('Official/player contaminated rectangle removed; no replacement coordinate.' if f in [46501,46502,46503,46505] and tag=='019' else 'Near white/red player wrongly assigned to far F2 removed; no replacement coordinate.' if tag=='019' and f in [46665,46666] else 'Official and actual dark player still share the native rectangle; body correspondence does not certify its lower-edge position.' if tag=='019' and f==46511 else 'Dark player visible behind orange player; native lower edge remains ambiguous, no hidden-foot estimate accepted.' if tag=='019' and f==46634 else 'Actual server visible and preserved; previous hard-veto loss avoided.' if tag=='016' else 'Visible server fragment preserved; its native lower edge is not certified ground contact.' if tag=='019' and f==46352 else 'Mixed official/orange-player native rectangle preserved; torso evidence follows the actual player, single-person box quality remains unvalidated.' if tag=='006' else 'Visibly present player retained at a prior torso-color rejection counterexample. Independent identity and ground-position truth not added.')
        findings.append(dict(tag=tag,source_frame=f,slot=row['slot'],removed=int(row['removed']),AI_observation=note,image=str(OUT/'all_change_review'/row['image']),image_sha256=row['sha256'],independent_human_GT=0))
    write(OUT/'AI_visual_findings.csv',findings)
    ex=OUT/'export019_torso_temporal';ep=load(ex/'protocol.json');assert all(sha(Path(n))==h for n,h in ep['protected_input_hashes'].items());ev=load(ex/'verification.json');assert ev['matched_native_observations']==3262 and ev['recalculated_feature_values']==379 and ev['prefix_features_equal_original'] and ev['video_frames_decoded']==930
    comparison=OUT/'comparison_review';cv=load(comparison/'verification.json');assert cv['video_decoded_frames']==666 and cv['aspect_preserved'];assert sha(comparison/'before_after_torso_soft_cost_h264.mp4')==cv['video_sha256'] and sha(comparison/'video_QA.jpg')==cv['QA_image_sha256']
    previous_files=0
    for previous in [BASE/'current22_player_inputs',BASE/'cohort_tracking/case_019',BASE/'pair_initialization_transfer_019',BASE/'far_identity_guard_development',BASE/'far_spatial_guard_development',BASE/'body_part_localization_probe']:
        for name,value in load(previous/'publication_hashes.json').items():assert sha(previous/name)==value;previous_files+=1
    current=BASE/'current22_player_inputs';cr=load(current/'result.json');assert len(read(current/'confirmed_human_outcomes.csv'))==11 and cr['fixed30_constructed_inputs']==19 and cr['next_candidate']=='match1_rally_window_019'
    decision=dict(status='partial_temporal_candidate_retained_not_adopted',primary_candidate=MODES[0],secondary_pose_exclusivity_added_effect_not_demonstrated=True,comparison='Both modes remove only the same six019 native observations. All nine entire control paths remain exact. Unlike prior short-tracklet hard veto,016F40625/F40626 and019F46352 remain present.',known_failures_removed_source_frames=[46501,46502,46503,46505,46665,46666],known_or_ambiguous_mixed_boxes_remaining=[46511,46634],new_positions_recovered=False,known_true_player_loss_not_seen_in_changed_rows=True,not_independent_accuracy_estimate=True,canonical_pipeline_replaced=False,current22_unchanged=True,confirmed_human_outcomes=11,fixed30_constructed_inputs=19,fixed30_pending_inputs=11,next_unresolved_candidate='match1_rally_window_019',next_action='Keep this partial candidate as the tracking comparator. Separate native-box lower-edge validity from identity presence in remaining mixed/occluded frames; do not try to force hidden ground positions or enlarge the color penalty to eliminate a particular frame.')
    dump(OUT/'processing_decision.json',decision)
    pr=load(OUT/'pose_result.json');assert pr['new_inference_calls']==3536 and pr['reused_predictions']==95
    report=dict(status='passed',scope='Native/provenance/cost/feature/video validation and bounded same-match AI inspection; not identity accuracy',contexts=10,search_frames=7019,scheduled_far_pose_frames=3551,scheduled_court_pose_frames=80,new_model_inference_calls=3536,reused_predictions=95,pose_proposal_rows=pose_rows,baseline_path_and_cost_exact_replays=10,player_path_csvs=outputs,native_output_rows_verified=checked,candidate_cost_rows=diagnostic_rows,positive_extra_cost_candidate_rows=positive_cost_rows,primary_removed_rows=6,primary_added_rows=0,secondary_native_paths_equal_primary_cases=10,all9_complete_control_paths_unchanged=True,all10_fixed_prefix_native_boxes_unchanged=True,AI_reviewed_native_observations=16,AI_changed_observations_reviewed=6,missing_initial_near_rear_pose_anchors_cases=['009','012'],all_output_positions_from_native_detector_boxes=True,predicted_ground_positions_exported=False,export019_native_rows=3262,export019_recalculated_feature_values=379,export019_decoded_video_frames=930,comparison_decoded_video_frames=666,comparison_QA_frames_inspected=5,previous_publication_files_unchanged=previous_files,cached_loader_amendment_numerical_policy_unchanged=True,current22_inputs_and11_outcomes_unchanged=True,canonical_pipeline_replaced=False,independent_accuracy_established=False)
    dump(OUT/'final_verification.json',report)
    text='''# 胴体情報を時間方向の対応付けへ追加：019の誤対応6行を欠測化

前段の姿勢診断で胴体を分けられる例と、胴体色だけでは実選手まで拒否する例の両方を確認した。そのため、色による強制拒否をやめ、既存の奥2人のビーム探索へ補助コストとして追加した。同じ10区間を初期化から各既存区間末尾まで計7019フレームで比較。元検出・元候補・初期ID・手前経路・探索条件は固定した。全10区間で追加コストなしの経路と探索コストを元処理に完全一致させてから比較した。

奥2人の胴体参照はまとめて同じ側の参照とし、もう一方の奥選手に服装が近いことだけでは不利にしない。相手側または初期の未選択人物に近い場合だけ、0.8×max(0,奥側への距離−負例への距離−0.04)を観測コストへ加える。両IDへ同じ値を加え、個人IDを服装だけで決めない。0.8・0.04・IoU0.5・身体部位確信度0.5などは既存値を使用し、結果を見た数値調整はしていない。初期未選択人物は正解の審判ラベルではない。

初期8フレームは全体・奥拡大の両方で姿勢を取得し、その後は原則2元フレームごとの奥拡大画像を使う。推論予定は新しい出力を見る前に固定。合計3551時点の奥拡大と80時点の全体画像を使い、既存95出力を再利用、3536回を追加推論した。27913の姿勢提案を保存。未取得時点や身体部位の不足には追加コスト0を与え、特徴や足位置を補間しない。009・012は初期の手前N2に対応する有効な姿勢参照が得られず、そこは欠けた参照のまま処理した。

|条件|元枠から除去|追加選択した枠|対照9区間の全経路|
|---|---:|---:|---|
|胴体の側別矛盾を補助コストへ追加|019の6行|0行|全9区間が元と同じ|
|上記に同一姿勢を2IDへ使わない制約も追加|同じ6行|0行|全9区間が元と同じ|

両条件の経路は全10区間で同一。同一姿勢の重複利用防止による追加効果はこの比較では示されなかった。出力行数や経路一致は本人同定精度・検出精度・独立試行数ではない。対照全経路の正しさを新たに保証した結果でもない。

019で消えたのはF46501・46502・46503・46505の審判を含むF1枠と、F46665・46666の手前選手をF2にした枠。変更6行すべてをAI画像点検した。直接の胴体追加コストがある変更行はF46501とF46665で、残る4行はコスト0でも時系列経路の選び直しにより欠測になった。未推論フレームへ色特徴を補間したわけではない。探索は将来の情報を使う近似的なビーム探索であり、オンライン処理や確率的な本人確信度ではない。

以前の不安定断片だけを拒否する処理が消した016F40625・40626と019F46352は保持した。以前の胴体色の反例002F2319、009F21945・21975も保持した。残る問題2行、これらの保持例と006の混合枠も含め計16行をAI点検した。006の混合枠はそのままであり、経路が不変だから単一人物の位置が正確という意味にはしない。

019F46511の審判＋濃色選手の混合枠と、F46634の濃色・橙色選手の重なりは残る。本人の一部が見えること、枠が単一人物を囲むこと、下端が本人の接地点を示すことは別々に判断する必要がある。混合枠だから下端が必ず誤っているとも断定しないが、今回それを独立な正解で確認したものでもない。隠れた足の座標を推測して入力へ追加していない。

comparison_review/before_after_torso_soft_cost_h264.mp4は左が元、右が補助コスト版。019の変更近傍と残る混合枠、016と019の保持例を含む。動きは1/3速度、保持例の末尾2場面は各2秒の静止画。縦横比を保持し666フレームを復号確認。export019_torso_temporalには別保存した4人経路3262行と930フレームの比較用映像を置いた。

全30経路CSVの37506行が元検出座標と一致し、補間・姿勢の足首座標が正式位置へ混入していないことを検証。全10区間の指定序盤窓の元枠は不変。019は元の指定31時点のうち4候補21時点・F1欠測10時点のままで、配置値379項目も独立計算と元処理に一致した。入力窓後の改善と序盤値の正誤は区別する。

計算途中でNPZの各配列を行ごとに再展開する無駄が見つかり、元処理を006の読み込み中に中断した。配列を一度だけ展開する実行版へ変更し、完了済み002・005と006の元経路をハッシュで保持して再開。条件・数式・モデル・数値は変更していない。元の実行コード、修正版、execution_amendment.jsonを保存した。

同じ試合の既知失敗を使った逐次開発であり、独立精度の向上率を主張しない。6行は隣接フレームを含む2つの失敗近傍で、6個の独立成功例ではない。この段階の改善は誤対応の欠測化であり、正しい位置の再取得ではない。混合枠が残るため全面採用は保留するが、以前の正しい観測を失う拒否案に対し、次の比較用候補として保持する。

current22_player_inputsの全ファイル、確認済み11得点ラベル、以前の全失敗・試作を保持。固定30群は入力作成19件・未完了11件のまま、019を除外せず未解決に残す。得点予測の学習、性能評価、球の追加価値比較は行っていない。

次は混合・遮蔽時の位置の扱いを本人IDの有無と分けて整理する。本人に対応する胴体があっても、配置に使える下端が確認できるとは限らない。既知の1フレームを消すために色の重みを上げず、見えていない接地点を無理に補うことも避ける。
'''
    (OUT/'README.md').write_text(text,encoding='utf-8');(ex/'README.md').write_text('019の胴体補助コスト版。元の審判・手前選手への誤対応6行を欠測化。F46511・46634の混合／重なりが残るため未採用。元検出座標のみ出力し、指定序盤配置値は元と同一。親README.md参照。\n',encoding='utf-8')
    archive=OUT/'executed_sources';archive.mkdir()
    for name in ['torso_temporal_association.py','run_torso_temporal_trial.py','run_torso_temporal_trial_cached.py','prepare_torso_temporal_review.py','export_torso_temporal019.py','render_torso_temporal_comparison.py','finalize_torso_temporal_trial.py']:shutil.copy2(ROOT/name,archive/name)
    dump(OUT/'frozen_partial_candidate.json',dict(status='experimental_comparator_not_adopted',candidate=MODES[0],parameters=p['parameters'],protocol_sha256=sha(OUT/'protocol.json'),solver_sha256=sha(ROOT/'torso_temporal_association.py'),known_removed_examples=[46501,46502,46503,46505,46665,46666],retained_mixed_or_ambiguous_examples=[46511,46634],same_match_development=True,independent_GT=False))
    day=BASE/'README.md';content=day.read_text(encoding='utf-8');marker='胴体情報を時間方向の追跡へ追加した10区間比較';assert marker not in content
    day.write_text(content+'\n\n'+marker+'。7019探索フレームで元経路・コストを全10区間完全再現後、奥2人をまとめた胴体色の補助コストを追加。対照9区間は全経路不変、019の審判・手前選手への誤枠6行だけ欠測化し追加枠なし。以前落とした実選手の枠3行と色の反例も保持。変更6行・残る問題2行・保持例8行をAI点検。姿勢の重複利用防止の追加効果はなく、F46511・46634の混合／位置不明確な重なりは残るため未採用。比較用候補として固定しcurrent22・11ラベル・19/30を保持。詳細torso_temporal_association_trial/README.md。\n',encoding='utf-8')
    dump(OUT/'publication_hashes.json',{str(n.relative_to(OUT)):sha(n) for n in sorted(OUT.rglob('*')) if n.is_file() and n.name!='publication_hashes.json'});print(report)

if __name__=='__main__':main()
