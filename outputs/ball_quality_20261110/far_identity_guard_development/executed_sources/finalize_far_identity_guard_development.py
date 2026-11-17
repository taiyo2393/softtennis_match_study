"""Verify and retain all sequential development attempts without pipeline adoption."""
import json,shutil
from pathlib import Path
from collections import defaultdict
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,write,sha,dump
from diagnose_far_identity_failures import OUT,load

def main():
    assert not (OUT/'final_verification.json').exists();first=OUT/'matched_trial';v2=OUT/'matched_trial_v2';v3=OUT/'matched_trial_v3';p=load(first/'protocol.json');cases=p['cases'];assert len(cases)==10
    for folder in [first,v2,v3]:
        proto=load(folder/'protocol.json');assert all(sha(Path(n))==h for n,h in proto['protected_input_hashes'].items());assert len(load(folder/'result.json')['cases'])==10
    assert load(first/'result.json')['baseline_exact_replays']==10;assert load(first/'case_019/copied_solver_baseline_verification.json')['path_and_cost_exact_without_veto']
    fields=['source_frame','player_slot','track_id','x1','y1','x2','y2','bottom_center_x','bottom_center_y'];key=lambda r:tuple(r[k] for k in fields);total_rows=0;outputs=0;changes=[];controls_unchanged=[]
    for e in cases:
        tag=e['tag'];native={(r['source_frame'],r['track_id']):r for r in read(Path(e['raw_csv']))};base=read(first/f'case_{tag}/baseline/far_pair_path.csv');bk={key(r) for r in base};assert [key(r) for r in base]==[key(r) for r in read(Path(e['reference_far_path']))]
        paths=[first/f'case_{tag}/{mode}/far_pair_path.csv' for mode in p['modes']]+[v2/f'case_{tag}/far_pair_path.csv',v3/f'case_{tag}/far_pair_path.csv']
        for path in paths:
            rows=read(path);seen=defaultdict(set);seen_slots=set()
            for r in rows:
                raw=native[r['source_frame'],r['track_id']];assert all(raw[k]==r[k] for k in ['x1','y1','x2','y2','bottom_center_x','bottom_center_y','confidence']);assert (r['source_frame'],r['player_slot']) not in seen_slots;seen_slots.add((r['source_frame'],r['player_slot']));assert r['track_id'] not in seen[r['source_frame']];seen[r['source_frame']].add(r['track_id'])
            outputs+=1;total_rows+=len(rows)
        new=read(v3/f'case_{tag}/far_pair_path.csv');nk={key(r) for r in new};assert not nk-bk
        if tag!='019' and nk==bk:controls_unchanged.append(tag)
        c=BASE/f'cohort_tracking/case_{tag}/protocol.json'
        if c.exists():
            times=set(load(c)['requested_input_frames']);assert {r for r in bk if int(r[0]) in times}=={r for r in nk if int(r[0]) in times}
        else:assert nk==bk
        for r in base:
            if key(r) not in nk:changes.append(dict(tag=tag,**r))
    assert len(controls_unchanged)==8 and len(changes)==9;assert {(r['tag'],int(r['source_frame'])) for r in changes}=={('016',40625),('016',40626),('019',46352),('019',46501),('019',46502),('019',46503),('019',46505),('019',46665),('019',46666)}
    write(OUT/'v3_all_removed_observations.csv',changes)
    probes=read(BASE/'pair_initialization_transfer_019/AI_failure_probes.csv');probe_keys={(r['source_frame'],r['player_slot'],r['track_id']) for r in probes};kept={(r['source_frame'],r['player_slot'],r['track_id']) for r in read(v3/'case_019/far_pair_path.csv')};removed_probes=probe_keys-kept;retained_probes=probe_keys&kept;assert {int(r[0]) for r in removed_probes}=={46501,46503,46505,46665,46666} and {int(r[0]) for r in retained_probes}=={46511,46634}
    oldreview=load(first/'changed_observation_review/manifest.json');newreview=load(v3/'all_change_review/manifest.json');assert len(oldreview['items'])==29 and len(newreview['items'])==11
    for folder,manifest in [(first/'changed_observation_review',oldreview),(v3/'all_change_review',newreview)]:assert all(sha(folder/e['image'])==e['sha256'] for e in manifest['items'])
    assessments=[]
    for r in newreview['items']:
        f=r['source_frame'];assessment=('Visible actual dark server; rejection loses a valid-person observation, ground position not independently measured.' if r['tag']=='016' else 'Visible dark server fragment behind near player; rejection loses a visible fragment, lower edge was not ground contact.' if f==46352 else 'Official/contaminated box removed; no replacement coordinate.' if f in [46501,46502,46503,46505] else 'Near white/red player wrongly used as far F2; removed without replacement.' if f in [46665,46666] else 'Mixed official+dark-player box remains; unresolved single-person localization.' if f==46511 else 'Dark/orange overlap remains; identity localization unresolved.')
        assessments.append(dict(tag=r['tag'],source_frame=f,slot=r['slot'],removed=int(r['removed']),AI_assessment=assessment,image=str(v3/'all_change_review'/r['image']),image_sha256=r['sha256'],independent_human_GT=0))
    write(OUT/'AI_all_v3_changes_and_remaining_probes.csv',assessments)
    dump(OUT/'AI_visual_findings.json',dict(reviewer='Codex AI saved diagnostic image inspection',first_trial_images_inspected=29,v3_images_inspected=11,selection='All nine changed rows in final trial plus both retained known failure/ambiguity probes; first trial longest removed runs and all new additions were also inspected.',observations='Broad per-frame color veto removes visibly correct players in controls. Narrow unstable-tracklet veto preserves8/9 complete control paths, but loses016 actual-server frames40625/40626 and019 visible server fragment46352.019 official/contaminated boxes46501/46502/46503/46505 and near-player F2 boxes46665/46666 become missing. Mixed boxes46511 and46634 remain. No newly created coordinates in narrow trial.',prior_AI_probes_rejected_source_frames=[46501,46503,46505,46665,46666],newly_inspected_extra_official_row_rejected=46502,known_or_ambiguous_remaining_source_frames=[46511,46634],individual_ID_binding=False,independent_identity_or_position_accuracy_established=False,model_predictions_not_used_as_human_GT=True,not_an_accuracy_estimate=True,not_all_rows_or_all_clips_independently_annotated=True,comparison_video_includes_true_player_loss=True))
    export=OUT/'export019_v3';ep=load(export/'protocol.json');assert all(sha(Path(n))==h for n,h in ep['protected_input_hashes'].items());ev=load(export/'verification.json');assert ev['status']=='passed' and ev['matched_native_observations']==3261 and ev['prefix_features_equal_original'];assert all(sha(export/n)==h for n,h in load(export/'case_result.json')['artifact_hashes'].items())
    comparison=OUT/'comparison_review';cv=load(comparison/'verification.json');assert cv['video_decoded_frames']==546 and cv['aspect_preserved'];assert sha(comparison/'before_after_with_counterexample_h264.mp4')==cv['video_sha256'] and sha(comparison/'video_QA.jpg')==cv['QA_image_sha256']
    current=BASE/'current22_player_inputs';assert all(sha(current/n)==h for n,h in load(current/'publication_hashes.json').items());assert len(read(current/'confirmed_human_outcomes.csv'))==11;current_result=load(current/'result.json');assert current_result['fixed30_constructed_inputs']==19 and current_result['next_candidate']=='match1_rally_window_019'
    cp=load(BASE/'cohort_and_baselines/protocol.json');score={n:h for n,h in cp['protected_input_hashes'].items() if Path(n).name=='score_suggestion_validation_sample.csv'};assert len(score)==1 and all(sha(Path(n))==h for n,h in score.items())
    decision=dict(status='partial_rejection_candidate_not_adopted',selected_for_diagnostic_export='unstable_tracklet_identity_guard_v3',selection_after_same_match_development=True,known_case_failures_not_all_resolved=True,retained_mixed_box_examples=[46511,46634],true_player_loss_control016=[40625,40626],no_new_observed_or_inferred_coordinates=True,current_input_snapshot=str(current),current_input_files_unchanged=True,confirmed_human_outcomes=11,fixed30_constructed_inputs=19,fixed30_pending_inputs=11,next_unresolved_candidate='match1_rally_window_019',canonical_pipeline_replaced=False,next_action='Investigate stable mixed-person boxes separately from short-lived false reacquisitions. Keep v3 as a comparator; do not relax thresholds to remove one known frame or promote same-match probe rejection to independent accuracy.')
    dump(OUT/'processing_decision.json',decision)
    report=dict(status='passed',scope='Structural/provenance checks and same-match development comparison, not tracking accuracy',baseline_exact_replays=10,matched_output_csvs=outputs,matched_native_observation_rows=total_rows,v3_controls_unchanged=controls_unchanged,v3_removed_rows_all_cases=9,v3_added_rows_all_cases=0,prior019_AI_probe_rows_rejected=5,prior019_AI_probe_rows_remaining=2,adjacent_probe_frames_not_independent_trials=True,all_v3_requested_prefix_native_boxes_unchanged=True,export019_recalculated_feature_values=379,export019_decoded_video_frames=930,comparison_decoded_frames=546,current22_inputs_unchanged=True,confirmed11_outcomes_preserved=True,canonical_pipeline_replaced=False,independent_identity_or_position_accuracy_established=False)
    dump(OUT/'final_verification.json',report)
    text='''# 奥側の誤再取得を欠測へ戻す試験：019と対照9区間

019の初期化3条件が同じ誤対応を残したため、同じ検出枠・初期対象・手前経路・ビーム探索を使い、奥候補の選別だけを比較した。以前の処理は「奥のどちらかに近い」外観で候補を通し、他方のIDへの割当ても許していた。また上半身・下半身を平均した特徴では、上に審判、下に選手を含む枠が候補に残る。019の全既存奥観測を採点して原因を診断した。

比較対象は019と002・005・006・009・012・015・016・017・018の10区間。設定と対象区間を各試作の実行前に保存し、前試作を上書きしていない。既存閾値0.70・外観差0.04・安定断片8観測を流用し、数値の網羅探索はしていない。ただし前結果を見て規則を改めた逐次的な同一試合内の開発であり、独立テストではない。

|試作|元枠から除去した行|追加選択した行|解釈|
|---|---:|---:|---|
|選手別の外観判定＋手前選手の外観を負例に追加|408|2|手前選手への誤対応は抑えるが、正しい選手も失う|
|上半身の負例比較も追加|869|5|019の明白な誤枠を落とす一方、対照の正しい観測を多く失う|
|上半身と全身で同じ負例の裏付けを要求|527|2|過剰除去は減るが、まだ正しい観測を失う|
|上記の拒否を既存判定で不安定な短い断片に限定（v3）|9|0|8/9対照区間の全経路を維持、016の2フレームと019の7フレームだけ欠測へ変更|

数字は同じ10区間内の出力行数の差であり、誤り数・精度・独立試行数ではない。探索コストも本人同定の確信度や性能指標ではない。元の全10経路を保存済み特徴から完全再現し、ビーム探索の拒否なし条件も019で経路とコストの一致を確認した。

最初の広い拒否では、対照の最大欠落区間と全追加枠を29画像でAI点検し、実際の選手まで消えることを確認した。v3では変更された9行すべてと、以前の問題・曖昧な例のうち残る2行を11画像で点検した。016のF40625・40626は実際の濃色サーバーが見える枠であり、その除去は副作用。019のF46352は手前選手に重なる濃色サーバーの可視断片で、これも観測を失う。いずれも指定序盤入力窓の外で、今回の10区間の指定窓の元枠は変わっていない。

019では、審判を含むF1枠F46501・46502・46503・46505、手前選手を奥F2にした枠F46665・46666が欠測になる。以前保存した問題例では5行を選ばなくなり、結果を見て全変更行を点検した際に追加のF46502も確認した。これは隣接フレームを含む2つの失敗近傍の観察で、精度率へ換算しない。人物を正しく再取得して座標を回復したわけではない。混合枠F46511と、奥2人が重なるF46634は残るので019は解決済みにしない。

v3は元経路の部分集合で、新しい座標も補間も出さない。安定性の判定は元のオフラインの全区間断片長を使うため、オンライン処理ではない。8フレーム以上続く誤枠や混合枠はこの限定策で防げない。色の特徴だけでは同じ服装・前後の見え方・照明・他人の混入を十分に識別できず、独立した本人同定・接地点の正解を追加したものでもない。

export019_v3に追跡と比較用動画を別保存した。元枠3261行と配置値379項目を照合し、930動画フレームを復号確認。指定序盤の21/31時点・10欠測・配置値は元処理と同じである。comparison_reviewの比較動画は左が元処理、右がv3。019は1/3速度、最後に016の実選手を消す副作用を静止画で示した。縦横比を保持し、残る混合枠も表示している。

全面採用はしない。原経路・以前の失敗記録・current22_player_inputsの全ファイル・確認済み11得点ラベルを保持。固定30群は入力作成19件、未完了11件のまま、019を未解決に残す。候補枠の出現率を本人同定精度に置き換えず、AI点検を独立した人手正解に昇格させない。学習・得点予測・球の追加価値比較は行っていない。

次は、安定断片になってしまう複数人物の混合枠を、短い誤再取得と分けて扱う。v3を部分的な拒否の比較対象として保持し、既知の1フレームを消すための閾値調整や、正しい観測を大量に落とす全面適用は避ける。
'''
    (OUT/'README.md').write_text(text,encoding='utf-8');(export/'README.md').write_text('019の部分的な誤枠拒否の試験版。元人物への再取得ではなく欠測化。混合枠F46511・F46634が残り、入力表にも正規処理にも未採用。指定序盤の配置値は元と同一。全体の比較と副作用は親README.mdを参照。\n',encoding='utf-8')
    scripts=['diagnose_far_identity_failures.py','far_identity_guard.py','far_identity_guard_v2.py','run_far_identity_guard_trial.py','run_far_identity_guard_trial_v2.py','run_far_identity_guard_trial_v3.py','prepare_far_guard_change_review.py','prepare_far_guard_v3_review.py','export_far_guard019_v3.py','render_far_guard_comparison.py','finalize_far_identity_guard_development.py']
    archive=OUT/'executed_sources';archive.mkdir()
    for name in scripts:shutil.copy2(ROOT/name,archive/name)
    dump(OUT/'frozen_partial_candidate.json',dict(status='experimental_not_adopted',policy='v2 corroborated identity veto only for retained unstable_tracklet flag',stable_tracklet_threshold=8,threshold_source='retained original far parameter',parameters=load(v3/'protocol.json')['guard_parameters'],code_hashes={str(ROOT/name):sha(ROOT/name) for name in ['far_identity_guard.py','far_identity_guard_v2.py','run_far_identity_guard_trial_v3.py']},known_residual_failures=[46511,46634],known_control_observation_loss=[40625,40626],candidate_needing_resolution='match1_rally_window_019',independent_GT=False))
    day=BASE/'README.md';marker='019の奥側誤対応を拒否する10区間比較';assert marker not in day.read_text(encoding='utf-8');day.write_text(day.read_text(encoding='utf-8')+'\n\n'+marker+'。選手別外観と負例による広い拒否は対照の正しい枠も除去したため、逐次開発の全試作を保持。不安定断片だけに限定したv3は対照9中8区間が元経路と同一、016の2フレームと019の7フレームを欠測にし追加枠なし。全変更9行と残る問題例2行をAI点検。019の審判・手前選手への誤枠の一部は消えるが、混合枠が残り全面採用しない。指定序盤配置値は不変。current22全ファイル・11ラベル保持、19/30作成のまま019未解決。比較動画は副作用も表示。詳細far_identity_guard_development/README.md。\n',encoding='utf-8')
    dump(OUT/'publication_hashes.json',{str(n.relative_to(OUT)):sha(n) for n in sorted(OUT.rglob('*')) if n.is_file() and n.name!='publication_hashes.json'})
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
