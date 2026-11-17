"""Finalize same-match development, preserve provenance, and freeze next transfer."""
import json,shutil
from pathlib import Path
from collections import defaultdict
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,sha,dump

def load(p):return json.loads(p.read_text(encoding='utf-8'))
def verified(folder,result):
    p=load(folder/'protocol.json');assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    assert load(folder/'verification.json')['status']=='passed'
    r=load(folder/result);assert all(sha(folder/n)==h for n,h in r['artifact_hashes'].items())
    return p,r

def main():
    stage=BASE/'pair_initialization_development';trial=stage/'matched_trial';old=BASE/'current21_player_inputs';out=BASE/'current22_player_inputs';p=load(trial/'protocol.json');r=load(trial/'result.json')
    assert not (stage/'final_verification.json').exists();assert len(r['cases'])==18 and r['court_filter_exact_replays']==18
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items());assert load(trial/'invariance_verification.json')['comparisons']==36
    unchanged=[];temporal_rows=0
    for entry in r['cases']:
        f=trial/f'case_{entry["tag"]}';assert (f/'pair_rank/selected_four.csv').read_bytes()==(f/'pair_rank_abstain/selected_four.csv').read_bytes()
        for mode in p['modes']:temporal_rows+=len(read(f/mode/'selected_four.csv'))
        if entry['modes']['pair_rank']['same_temporal_output_as_court_filter']:unchanged.append(entry['tag'])
    assert len(unchanged)==16 and set(x['tag'] for x in r['cases'])-set(unchanged)=={'012','018'}
    for tag in ['009','015']:
        entry=next(x for x in r['cases'] if x['tag']==tag);assert entry['modes']['pair_rank']['same_temporal_output_as_court_filter']
    matched=checked=decoded=0
    for tag in ['012','018']:
        case=BASE/f'cohort_tracking/case_{tag}/near_pair_development';q,cr=verified(case,'case_result.json');v=load(case/'verification.json');assert not q['AI_identity_binding_added'] and not v['interpolation'];assert not load(case/'AI_visual_findings.json')['known_wrong_person_assignment']
        matched+=v['matched_native_boxes'];checked+=v['recalculated_feature_values'];decoded+=v['video_frames_decoded']
        if tag=='012':
            assert v['prefix_feature_values_equal_original'] and v['removed_original_assignments']==0 and v['added_assignments']==4
            keys=['source_frame','player_slot','track_id','x1','y1','x2','y2'];sig=lambda row:tuple(row[k] for k in keys)
            added={sig(row) for row in read(case/'four_player_path.csv')}-{sig(row) for row in read(case.parent/'four_player_path.csv')};assert {int(row[0]) for row in added}=={31935}
        else:
            assert q['requested_input_frames']==list(range(43838,43899,2)) and v['four_observation_times']==31
            identity=load(case/'fixed_input_identity/manifest.json');assert all(sha(case/'fixed_input_identity'/n)==h for n,h in identity['images'].items())
            findings=load(case/'AI_visual_findings.json');assert all(sha(Path(n))==h for n,h in findings['inspected_image_hashes'].items())
            gaps=read(case/'gap_diagnostics/all_final_path_gap_runs.csv');assert sorted(int(row['missing_frames']) for row in gaps)==[11,14,17]
            assert load(case.parent/'AI_visual_findings.json')['known_wrong_person_assignment']
    init=load(trial/'initialization_evidence/manifest.json');assert {e['tag'] for e in init['images']}=={'012','018'};assert all(sha(trial/'initialization_evidence'/e['image'])==e['sha256'] for e in init['images'])
    for previous in ['cohort_tracking/case_018','court_initialization_transfer_018','current21_player_inputs']:
        folder=BASE/previous;assert all(sha(folder/n)==h for n,h in load(folder/'publication_hashes.json').items())
    op,orr=verified(out,'result.json');assert op['selected_case_variant']=='near_pair_development' and op['development_after_known_failure'];assert orr['fixed30_constructed_inputs']==19 and orr['fixed30_pending_inputs']==11 and orr['next_candidate']=='match1_rally_window_019'
    names=['B1_server_side.csv','B2_snapshot.csv','B2_window_means.csv','diagnostic_snapshot_masks_only.csv','diagnostic_window_masks_only.csv','human_outcomes.csv','source_snapshot_features.csv','source_window_summary_features.csv','rally_metadata.csv','processing_interventions.csv','outcome_label_status.csv'];cid='match1_rally_window_018'
    for name in names:
        before=read(old/name);after=read(out/name);assert len(before)==21 and len(after)==22;assert [row for row in after if row['rally_candidate_id']!=cid]==before
    assert (old/'confirmed_human_outcomes.csv').read_bytes()==(out/'confirmed_human_outcomes.csv').read_bytes();assert len(read(out/'confirmed_human_outcomes.csv'))==11
    label=next(row for row in read(out/'human_outcomes.csv') if row['rally_candidate_id']==cid);assert label['winner_camera_side']==label['human_side_review_id']==''
    meta=next(row for row in read(out/'rally_metadata.csv') if row['rally_candidate_id']==cid);assert meta['tracking_source']==str(BASE/'cohort_tracking/case_018/near_pair_development/four_player_path.csv')
    intervention=next(row for row in read(out/'processing_interventions.csv') if row['rally_candidate_id']==cid);assert intervention['intervention']=='automatic_near_pair_initialization_development'
    before=read(old/'fixed30_processing_progress.csv');after=read(out/'fixed30_processing_progress.csv');assert [row for row in before if row['candidate_id']!=cid]==[row for row in after if row['candidate_id']!=cid]
    cp=load(BASE/'cohort_and_baselines/protocol.json');score={n:h for n,h in cp['protected_input_hashes'].items() if Path(n).name=='score_suggestion_validation_sample.csv'};assert len(score)==1 and all(sha(Path(n))==h for n,h in score.items())
    video=stage/'corrected_018_review_h264.mp4';cap=cv2.VideoCapture(str(video));n=0;cells=[]
    while True:
        ok,im=cap.read()
        if not ok:break
        assert im.shape[:2]==(1080,1920)
        if n in [0,60,150,517]:
            head=np.zeros((32,960,3),np.uint8);cv2.putText(head,f'018 F{43748+n}: pair-ranking development candidate',(8,24),0,.65,(255,255,255),1);cells.append(np.vstack([head,cv2.resize(im,(960,540))]))
        n+=1
    cap.release();assert n==518 and len(cells)==4;assert cv2.imwrite(str(stage/'video_QA.jpg'),np.vstack([np.hstack(cells[:2]),np.hstack(cells[2:])]))
    dump(stage/'excerpt_manifest.json',dict(first_source_frame=43748,last_source_frame=44265,frames=n,dimensions=[1920,1080],aspect_preserved=True,video_sha256=sha(video),source=str(BASE/'cohort_tracking/case_018/near_pair_development/tracking_quality_20261102_stage7/players_018_global/global_review_h264.mp4'),scope='Known-case development variant, includes inherited context end/broadcast dissolve; not independently certified exact rally bounds.'))
    frozen={name:value for name,value in p['protected_input_hashes'].items() if Path(name).suffix=='.py' and Path(name).name not in ['diagnose_initial_pair_appearance.py','run_pair_initialization_trial.py']};frozen[p['calibration']]=sha(Path(p['calibration']))
    dump(stage/'next_transfer_protocol.json',dict(next_candidate='match1_rally_window_019',fixed_before_019_inspection=True,threshold_tuning_on_019=False,selection='Next chronological pending fixed cohort candidate. Keep all failures and missing; no visibility-driven substitution.',scope='Additional same-match transfer after018-informed development, not independent other-match or never-seen research footage.',source_video_sha256=p['source_video_sha256'],calibration=p['calibration'],calibration_transfer_accuracy_established=False,selected_pair_policy='pair_rank',extra_ambiguity_margin_applied=False,selection_reason='Pair rank and pair rank+0.06 abstention produced identical temporal outputs in18 development cases; select simpler policy. No extra-gate effectiveness established.',modes=['baseline','court_seed_only','pair_rank'],pair_parameters=p['parameters'],court_parameters=p['court_parameters'],initialization_confirmation_frames=8,tracking_and_recovery_parameters_changed=False,new_person_ID_binding_allowed=False,uniform_assumption=p['uniform_assumption'],policy_uses_future_frames=False,far_association_uses_future_frames=True,tracking_mask_after_initialization=False,frozen_code_and_calibration_hashes=frozen,previous_development_protocol=str(trial/'protocol.json'),previous_development_protocol_sha256=sha(trial/'protocol.json'),previous_development_result=str(trial/'result.json'),previous_development_result_sha256=sha(trial/'result.json'),current_input_snapshot=str(out),canonical_pipeline_replaced=False,phase_policy='Keep inherited contact or freeze raw-image AI bracket before person tracking. Never use camera entry or first tracker initialization as contact. Fixed offsets +30..+90 stride2 stay unchanged.',identity_review_policy='Candidate count is not identity success. Reject known wrong-person input, preserve original failure and any experimental variant provenance. AI inspection is not independent human GT.'))
    report=dict(status='passed',scope='Same-match development, not independent tracking accuracy evaluation',matched_temporal_cases=18,court_filter_exact_replays=18,temporal_rows_across_all_three_modes=temporal_rows,pair_rank_exact_to_court_filter_cases=unchanged,changed_cases=['012','018'],pair_rank_and_extra_abstention_outputs_exact_all18=True,extra_abstention_benefit_established=False,initial_selection_invariance_comparisons=36,matched_native_boxes_in_two_materialized_cases=matched,recalculated_feature_values_in_two_materialized_cases=checked,decoded_materialized_video_frames=decoded,review_video_frames=n,previous21_input_rows_exactly_preserved=True,confirmed11_outcomes_preserved=True,original018_failure_preserved=True,individual_ID_binding_added=False,development_variant_provenance_preserved=True,fixed30_constructed_inputs=19,fixed30_pending_inputs=11,next_candidate='match1_rally_window_019',canonical_pipeline_replaced=False,independent_identity_or_position_accuracy_established=False)
    dump(stage/'final_verification.json',report);dump(out/'additional_verification.json',report)
    text='''# 初期化の人物ペア選択：018の既知失敗からの開発

018で審判を初期選択した失敗を受け、コート領域内の手前候補が3人以上いる場合、服装の色ヒストグラム距離が近い2人の組合せを選ぶ処理を試作した。従来の8フレーム連続確認・検出枠・追跡・再取得の数値設定は保持する。選手IDや結果ラベルを規則に埋め込んでいない。同じ服装のペアという仮定があり、学習済みの人物ReIDではない。

診断では正常に見えるペアの特徴距離にも幅があったため、絶対距離による新たな足切り閾値は設けなかった。新規則の出力を見る前に18区間と比較条件を固定した。領域制限のみ・組合せ順位・順位に差0.06での保留を加えた条件を、同じ保存済み検出枠と特徴で比較。0.06は既存の別用途の値を工学的に流用した比較用設定であり、確率的根拠や最適性はない。追跡出力が18区間全てで順位のみと同一だったので、追加保留の効果は主張せず、単純な順位付けを次の比較候補として固定した。

18区間すべてで領域制限付きの既存出力を完全再現。新規則は16区間の時間方向の枠・対応・数値コストを変えなかった。009・015で既に領域制限によって修正された出力も維持した。012は同じ4人を1フレーム早いF31935で選択し、実枠4個だけ増えた。012の指定序盤の配置値は不変。018は3フレーム早いF43643で、審判ではなく実選手を選択した。検出候補の順序を反転し、IDの数値を一括変更しても、両方式×18区間の初期対象と初期化時刻は変わらなかった。

018の指定31時点すべてのID付き画像では、N1が手前右の黒い帽子の白赤選手、N2が手前左の白い帽子の選手、F1が奥の濃色選手、F2が奥の橙色選手に対応して見える。初期対象・時系列の文脈画像・欠測前後もAI点検した。奥側には11・17・14フレームの欠測が残り、補間せず保持した。初期時刻と外観テンプレートも変わるので、奥側の欠測減少を再取得性能の改善とは断定しない。末尾は放送切替を含み、厳密なラリー終端は未確定。

これは018の失敗を見て方法を作った同一試合内の開発結果であり、独立評価の成功例ではない。同じ服装でないペア、審判が2人同時に候補になる場面、服装の前後差、照明、遮蔽、別カメラの校正ずれには限界がある。接地点の誤差や本人同定精度も独立した正解で測っていない。候補枠31/31を正解率へ読み替えない。奥側の対応付けは未来のフレームを使う。

原処理の失敗・過去の検証記録は保持し、018のnear_pair_developmentだけを開発処理由来と明記してcurrent22_player_inputsへ追加した。旧21行・確認済み11得点ラベルを保持し、012の既存入力も置換していない。固定30群は19件作成、11件未処理。これは分析適格ラリーや自動追跡成功の数ではない。得点側未確認11件は空欄。正規処理は置換していない。

次は019を、原処理・領域制限のみ・固定した組合せ順位の3条件で比較する。next_transfer_protocol.jsonでコード、校正、比較条件を固定した。失敗時は元の出力を保持し、設定を黙って調整しない。
'''
    (stage/'README.md').write_text(text,encoding='utf-8')
    (out/'README.md').write_text('''# 配置入力22候補：018の開発版を追加

旧21件の入力値・欠測・介入履歴、確認済み11得点ラベル（手前7・奥4）を保持した。固定30群内では19件作成・11件未処理、次は019。22候補や19件を、独立した正解を持つ分析適格ラリー数や自動追跡成功数として扱わない。008〜018の得点側未確認11件は空欄。

018の原処理は審判を追っているため不採用のまま。既知失敗を使って開発したnear_pair_development版を別由来として追加した。初期化で服装特徴の近い2人を選ぶ規則で、個別IDの人手指定はない。指定31時点は4候補あり、AI点検画像では4選手への対応と整合する。独立精度は未評価。同一試合での開発結果であり、汎化性能の成功例ではない。processing_interventions.csvとprotocol.jsonに開発処理の由来を明示した。

サーブ接触候補F43804〜43808と入力F43838〜43898は追跡前の設定を保持した。サーブ有効性・何本目か・厳密な終端は人手未確定。終盤の欠測を補間せず、球の入力も追加していない。human_outcomes.csvは未確認込みの台帳、confirmed_human_outcomes.csvは確認済み11件のみ。学習・性能推定・球の追加価値比較は未実施。
''',encoding='utf-8')
    for tag in ['012','018']:
        case=BASE/f'cohort_tracking/case_{tag}/near_pair_development';(case/'README.md').write_text(text if tag=='018' else '012の比較対照。初期化が1フレーム早まりF31935の4枠を追加した。旧枠の除去・変更はなく、指定序盤の配置値は同一。AIが新しい初期画像を点検し、後続は既存枠と完全照合した。既存の入力表の012は置換していない。詳細はpair_initialization_development/README.md。\n',encoding='utf-8')
        for name in ['materialize_pair_initialization_case.py','pair_initialization_guard.py','record_pair_development_review.py','finalize_pair_initialization_development.py']:shutil.copy2(ROOT/name,case/('executed_'+name))
    for name in ['verify_pair_initialization_invariance.py','record_pair_development_review.py','finalize_pair_initialization_development.py','prepare_fixed_input_identity_sheet_v3.py','prepare_player_gap_diagnostics_v2.py']:shutil.copy2(ROOT/name,trial/('executed_'+name))
    shutil.copy2(ROOT/'append_fixed_cohort_player_inputs_v5.py',out/'executed_source.py')
    day=BASE/'README.md';marker='018を人物ペアの外観順位で再処理';assert marker not in day.read_text(encoding='utf-8');day.write_text(day.read_text(encoding='utf-8')+'\n\n'+marker+'。18区間の同条件比較で16区間は時間方向出力が不変、012は1フレーム早い実枠4個のみ追加し序盤配置値は不変。018は審判から実選手へ初期対象が変わり、指定31時点と欠測前後をAI点検。個別ID指定なし。既知失敗を用いた開発であり独立精度の成功例ではない。元失敗を保持し、開発処理由来を明記してcurrent22_player_inputsへ追加。旧21件・11ラベル保持、固定30群19件作成・11件未処理。追加の保留条件の効果は示されず、順位のみを019へ事前固定。詳細はpair_initialization_development/README.md。\n',encoding='utf-8')
    for folder in [BASE/'cohort_tracking/case_012/near_pair_development',BASE/'cohort_tracking/case_018/near_pair_development',out,stage]:dump(folder/'publication_hashes.json',{str(n.relative_to(folder)):sha(n) for n in sorted(folder.rglob('*')) if n.is_file() and n.name!='publication_hashes.json'})
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
