"""Retain the negative/partial result, provenance, and bounded AI inspection."""
import json,shutil
from pathlib import Path
from collections import defaultdict
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,write,sha,dump
from run_far_identity_guard_trial import load
import far_spatial_identity_guard as guard

OUT=BASE/'far_spatial_guard_development'
FIELDS=['source_frame','player_slot','track_id','x1','y1','x2','y2','bottom_center_x','bottom_center_y']
key=lambda r:tuple(r[k] for k in FIELDS)

def main():
    assert not (OUT/'final_verification.json').exists()
    p=load(OUT/'protocol.json');result=load(OUT/'result.json');assert len(result['cases'])==len(p['cases'])==10
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items())
    outputs=rows_checked=predictions_checked=prefix_cases=0;changes=[];unchanged=[]
    for e in p['cases']:
        tag=e['tag'];native={(r['source_frame'],r['track_id']):r for r in read(Path(e['raw_csv']))}
        baseline=read(BASE/f'far_identity_guard_development/matched_trial/case_{tag}/baseline/far_pair_path.csv');bk={key(r) for r in baseline};mode_keys=[]
        for mode in p['modes']:
            rows=read(OUT/f'case_{tag}/{mode}/far_pair_path.csv');seen=defaultdict(set);slots=set()
            for r in rows:
                raw=native[r['source_frame'],r['track_id']]
                assert all(raw[k]==r[k] for k in ['x1','y1','x2','y2','bottom_center_x','bottom_center_y','confidence'])
                assert (r['source_frame'],r['player_slot']) not in slots;slots.add((r['source_frame'],r['player_slot']))
                assert r['track_id'] not in seen[r['source_frame']];seen[r['source_frame']].add(r['track_id'])
            nk={key(r) for r in rows};assert not nk-bk;mode_keys.append(nk);outputs+=1;rows_checked+=len(rows)
        assert mode_keys[0]==mode_keys[1],'Near exclusion ablation changed the path: document if this assertion changes.'
        nk=mode_keys[-1]
        if nk==bk:unchanged.append(tag)
        for r in baseline:
            if key(r) not in nk:changes.append(dict(tag=tag,**r))
        cp=BASE/f'cohort_tracking/case_{tag}/protocol.json'
        if cp.exists():
            times=set(load(cp)['requested_input_frames']);assert {r for r in bk if int(r[0]) in times}=={r for r in nk if int(r[0]) in times}
        else:assert bk==nk
        prefix_cases+=1
        # Reconstruct only from past observed near rectangles, and check every saved prediction.
        near=defaultdict(list)
        for r in read(Path(e['temporal_folder'])/'selected_four.csv'):
            if r['player_slot'].startswith('near_'):near[int(r['source_frame'])].append(r)
        ev=next(r for r in load(Path(e['temporal_folder'])/'events.json') if r['event']=='initialized');history={};computed={};obs={(r['player_slot'],int(r['source_frame'])):r for group in near.values() for r in group}
        for f in range(ev['frame'],e['last']+1):
            for reg in guard.near_regions(f,near[f],history):
                if not reg['age']:continue
                last=reg['row'];source=int(last['source_frame']);assert source<f and reg['age']==f-source and 1<=reg['age']<=8
                assert (reg['slot'],source) in obs
                computed[f,reg['slot']]=reg
        file=OUT/f'case_{tag}/predicted_exclusion_regions_not_output_positions.csv';saved=read(file) if file.exists() else []
        assert len(saved)==len(computed)
        for r in saved:
            reg=computed[int(r['source_frame']),r['near_slot']];assert int(r['source_observation_frame'])==int(reg['row']['source_frame'])
            for k in ['x1','y1','x2','y2']:assert abs(float(r[k])-float(reg['row'][k]))<1e-9
            predictions_checked+=1
    assert {(r['tag'],int(r['source_frame'])) for r in changes}=={('006',12421),('006',12422),('019',46501),('019',46502),('019',46503),('019',46505)}
    write(OUT/'all_removed_native_observations.csv',changes)
    review=load(OUT/'all_change_review/manifest.json');assert len(review['items'])==12
    assert all(sha(OUT/'all_change_review'/r['image'])==r['sha256'] for r in review['items'])
    findings=[]
    for r in review['items']:
        f=r['source_frame']
        assessment=('Official and actual orange player share the native rectangle; removed without recovering a single-person box.' if r['tag']=='006' else 'Actual dark server visible and preserved; previous v3 false rejection avoided.' if r['tag']=='016' else 'Official/actual-player contaminated rectangle removed without replacement.' if f in [46501,46502,46503,46505] else 'Official+dark-player mixed rectangle remains; not a reliable single-person location.' if f==46511 else 'Dark player visible behind orange player; single-person lower-edge localization remains ambiguous.' if f==46634 else 'Near white/red player still assigned to far F2; unresolved wrong-person assignment.')
        findings.append(dict(tag=r['tag'],source_frame=f,slot=r['slot'],removed=int(r['removed']),AI_assessment=assessment,image=str(OUT/'all_change_review'/r['image']),image_sha256=r['sha256'],independent_human_GT=0))
    write(OUT/'AI_all_changes_and_remaining_probes.csv',findings)
    dump(OUT/'AI_visual_findings.json',dict(reviewer='Codex AI saved-image inspection',diagnostic_images_inspected=12,selection='Every changed native row across ten cases, all retained prior019 probes, and the two previous016 false rejections.',near_prediction_diagnostic_frames_inspected=[46658,46659,46660,46665],comparison_QA_frames_inspected=5,observations='006 removes two mixed official/player rectangles.019 removes four contaminated rectangles, retains mixed46511 and overlapping46634 plus near-player-as-far assignments46665/46666.01640625/40626 actual-server boxes are preserved. Near-rear last box lower edge moves downward by14.57 native pixels from46659 to46660; extrapolating that box bottom is not extrapolating measured ground contact.',independent_identity_or_position_accuracy_established=False,not_an_accuracy_estimate=True,new_human_labels=False))
    diag=load(OUT/'residual_mechanism_diagnostic.json');assert len(diag['rows'])==8
    mixed=next(r for r in diag['rows'] if r['source_frame']==46511);assert any(r['spatial_veto'] and not r['appearance_veto'] for r in mixed['overlapping_initial_negative_regions'])
    for f in [46665,46666]:
        r=next(r for r in diag['rows'] if r['source_frame']==f);assert all(not r['spatial_veto'] for r in r['near_regions'])
    ep=OUT/'export019_spatial';ev=load(ep/'verification.json');assert ev['matched_native_observations']==3264 and ev['recalculated_feature_values']==379 and ev['video_frames_decoded']==930 and ev['prefix_features_equal_original']
    assert all(sha(Path(n))==h for n,h in load(ep/'protocol.json')['protected_input_hashes'].items())
    cv=load(OUT/'comparison_review/verification.json');assert cv['video_decoded_frames']==615 and cv['aspect_preserved']
    assert sha(OUT/'comparison_review/before_after_spatial_guard_h264.mp4')==cv['video_sha256']
    assert sha(OUT/'comparison_review/video_QA.jpg')==cv['QA_image_sha256']
    protected_publications=0
    for old in [BASE/'current22_player_inputs',BASE/'cohort_tracking/case_019',BASE/'pair_initialization_transfer_019',BASE/'far_identity_guard_development']:
        for n,h in load(old/'publication_hashes.json').items():assert sha(old/n)==h;protected_publications+=1
    current=BASE/'current22_player_inputs';cr=load(current/'result.json');assert len(read(current/'confirmed_human_outcomes.csv'))==11 and cr['fixed30_constructed_inputs']==19 and cr['next_candidate']=='match1_rally_window_019'
    decision=dict(status='partial_result_not_adopted',reason='Spatial support preserves prior016 observations but fails to reject remaining mixed/wrong-person boxes in019. Near-exclusion ablation has no output effect in any of the ten cases.',canonical_pipeline_replaced=False,current22_unchanged=True,confirmed_human_outcomes=11,fixed30_constructed_inputs=19,fixed30_pending_inputs=11,next_unresolved_candidate='match1_rally_window_019',do_not_skip019_or_change_input_windows=True,remaining_known_or_ambiguous_examples=[46511,46634,46665,46666],independent_accuracy_established=False,next_action='Address single-person localization and identity evidence separately. Use a controlled masked-appearance or body-part localization prototype before another numerical threshold change; compare the retained failures and all nine controls, and keep native observations distinct from inferred positions.')
    dump(OUT/'processing_decision.json',decision)
    report=dict(status='passed',scope='Native-box/provenance/feature/video and past-only exclusion checks; not identity accuracy',matched_cases=10,baseline_exact_replays_reused=10,output_csvs=outputs,native_output_rows_checked=rows_checked,unchanged_complete_paths=unchanged,removed_rows_all_cases=6,added_rows_all_cases=0,background_vs_background_and_near_paths_identical_cases=10,all10_requested_prefix_native_boxes_unchanged=True,predicted_exclusion_rows_checked=predictions_checked,predicted_positions_exported_to_player_paths=False,AI_reviewed_cells=12,export019_native_rows=3264,export019_recalculated_values=379,export019_decoded_video_frames=930,comparison_video_decoded_frames=615,prior_publication_files_verified=protected_publications,current22_inputs_and11_confirmed_outcomes_unchanged=True,canonical_pipeline_replaced=False,independent_accuracy_established=False)
    dump(OUT/'final_verification.json',report)
    text='''# 外観拒否へ位置の裏付けを加える試験：部分的な効果、採用は保留

019と従来の対照9区間を、同じ検出枠・初期対象・手前経路・ビーム探索で比較した。上半身の外観が初期の未選択人物に近く、その人物の初期位置とも重なる場合だけ拒否する。手前人物への外観一致には、現在の手前枠または過去2観測から最大8フレームだけ延ばした除外用矩形との重なりも要求した。予測矩形は選別の内部情報で、選手位置として出力しない。

0.04の外観差、0.55の重なり、8フレーム、身長相当の0.10倍の速度上限は既存値を流用。新たな数値探索はしていない。ただし既知の019失敗と前試作の結果を見た後の同一試合内開発であり、独立テストではない。初期の未選択人物を正解の審判と見なさず、固定カメラという適用条件も新たに自動検証したものではない。奥側の探索には将来の断片情報を使う既存のオフライン処理が残る。

|比較|元枠から除去|追加枠|全経路が不変の区間|
|---|---:|---:|---:|
|初期負例の位置と上半身外観を併用|6行|0行|8/10区間|
|上記に手前の観測・短期予測矩形を追加|6行|0行|8/10区間|

両条件の出力経路は全10区間で同一。手前の除外用予測矩形を追加する効果は、この比較では示されなかった。行数は出力の差であり、精度・誤り数・独立試行数ではない。

変更した全6行、残る019の問題・曖昧な4行、前試作が落とした016の2行を計12画像でAI点検した。006のF12421・12422は審判と実選手が混ざる枠で、削除しても実選手の座標が回復したわけではない。019のF46501・46502・46503・46505も審判を含む枠が欠測になる。一方、016のF40625・40626は実際の濃色サーバーを保持し、前試作の誤除去を回避した。

019のF46511は審判と濃色選手の混合枠のまま。初期負例位置との上半身領域の重なりは条件を満たすが、同じ負例との上半身外観差の条件は満たさない。F46634は濃色選手の頭部が橙色選手の後ろに見え、単純な別人化と断定できないが、枠下端を本人の接地位置として扱えない。F46665・46666では手前の白赤選手が奥F2へ割り当てられたままである。

手前N2はF46661〜46666で欠測。直前F46659→46660の枠下端は14.57画素下へ動き、そのまま外挿するとF46665の矩形は実際の白赤選手から下へずれる。F46658・46659・46660・46665の画像と元枠座標、拒否に至らなかった条件を保存した。これは枠下端の移動から選手の接地位置を直接予測できない例であり、速度上限の緩和で解決したと扱わない。

comparison_review/before_after_spatial_guard_h264.mp4は左が元処理、右が今回。006の変更近傍、019の改善部分と残る失敗、016の保持例を同じフレーム・同じ縮尺で表示。動きの区間は1/3速度、016は静止画。縦横比を保持した。615フレームを復号確認した。

export019_spatialは別保存した試験版。元検出と一致する3264観測、指定序盤配置の再計算379値、930動画フレームを検証。指定31時点では4候補21時点・F1欠測10時点のままで、元配置値と同一。全10区間の指定序盤窓の元枠も不変。入力窓後の失敗と序盤値の正誤は区別する。

全面採用はしない。019は固定群から除外せず未解決。current22_player_inputs全体・確認済み11得点ラベル・以前の失敗と試作を保持し、固定30群は入力作成19件、未完了11件のまま。AI画像点検は独立した本人同定・接地点の正解ではない。得点予測の学習・性能評価・球の追加価値比較は行っていない。

次は、色や枠の重なりの閾値をさらに調整する前に、「本人を示す上半身」と「配置に使える本人の位置」を分ける。人物領域を限定した外観比較または身体部位の位置推定を小規模に試し、混合枠と実選手の消失を同じ対照で比較する。隠れた足の補間を実測値として入力しない。
'''
    (OUT/'README.md').write_text(text,encoding='utf-8')
    (ep/'README.md').write_text('019の位置と外観を併用した拒否の試験版。F46511の混合枠、F46634の位置不明確な重なり、F46665・46666の別人割当てが残るため未採用。元観測にない座標は出力せず、指定序盤配置値は元と同一。親README.mdを参照。\n',encoding='utf-8')
    archive=OUT/'executed_sources';archive.mkdir()
    names=['far_spatial_identity_guard.py','run_far_spatial_guard_trial.py','prepare_far_spatial_guard_review.py','diagnose_far_spatial_guard_residuals.py','export_far_spatial_guard019.py','render_far_spatial_guard_comparison.py','finalize_far_spatial_guard_development.py']
    for name in names:shutil.copy2(ROOT/name,archive/name)
    day=BASE/'README.md';marker='019の誤対応へ位置の裏付けを加えた10区間比較';content=day.read_text(encoding='utf-8');assert marker not in content
    day.write_text(content+'\n\n'+marker+'。初期負例の位置と上半身外観の併用で006の2行と019の4行を欠測化。016の前試作による実選手2行の誤除去は回避したが、019の混合枠と手前選手への乗り換えは残った。手前の短期予測矩形を加えた条件は全10区間で経路に効果なし。変更全6行・残る問題4行・016の保持2行をAI点検し、予測が枠下端の変動でずれる例も保存。全面採用せずcurrent22と11ラベル保持、19/30のまま019未解決。詳細far_spatial_guard_development/README.md。\n',encoding='utf-8')
    dump(OUT/'publication_hashes.json',{str(n.relative_to(OUT)):sha(n) for n in sorted(OUT.rglob('*')) if n.is_file() and n.name!='publication_hashes.json'})
    print(json.dumps(report,ensure_ascii=False))

if __name__=='__main__':main()
