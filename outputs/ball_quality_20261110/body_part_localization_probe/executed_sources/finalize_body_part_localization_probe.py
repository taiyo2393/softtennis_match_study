"""Verify the body-part feasibility artifacts and preserve the negative controls."""
import math,shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,write,sha,dump
from run_body_part_localization_probe import OUT,CROPS,SLOTS,load,associations,verify_matching

def main():
    assert not (OUT/'final_verification.json').exists();p=load(OUT/'protocol.json');result=load(OUT/'result.json');pred=load(OUT/'pose_predictions.json');assert len(p['cases'])==10 and len(p['selected_frames'])==len(pred['frames'])==74;assert result['inference_calls']==148
    assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items());verify_matching()
    selected={(r['tag'],r['source_frame']):r for r in p['selected_frames']};assert len(selected)==74
    associations_expected=[];pose_count=0
    for r in pred['frames']:
        item=selected[r['tag'],r['source_frame']]
        assert len({x['player_slot'] for x in item['native_observations']})==len(item['native_observations'])
        for mode,people in r['modes'].items():
            assert mode in CROPS and len({x['pose_index'] for x in people})==len(people)
            for person in people:
                kp=np.asarray(person['keypoints']);bb=np.asarray(person['box']);assert kp.shape==(17,3) and np.isfinite(kp).all() and np.isfinite(bb).all();assert bb[2]>bb[0] and bb[3]>bb[1];assert np.all((kp[:,2]>=0)&(kp[:,2]<=1));assert 0<=person['confidence']<=1;pose_count+=1
            rows=associations(item,mode,people);assigned=[x['pose_index'] for x in rows if x['pose_matched']];assert len(assigned)==len(set(assigned));associations_expected.extend(rows)
    actual=read(OUT/'native_box_to_pose_diagnostics.csv');assert len(actual)==len(associations_expected)==548
    for a,b in zip(actual,associations_expected):
        assert a.keys()==b.keys()
        for k,v in b.items():
            if isinstance(v,(int,float,np.number)):assert math.isclose(float(a[k]),float(v),rel_tol=1e-10,abs_tol=1e-9)
            else:assert a[k]==str(v)
        assert a['observed_ground_position_certified']==a['identity_certified']=='0'
    counts={}
    for mode in CROPS:
        rows=[r for r in actual if r['mode']==mode and r['player_slot'].startswith('far_')];counts[mode]=dict(selected_native_far_boxes=len(rows),matched_pose_candidates=sum(r['pose_matched']=='1' for r in rows),torso_sets_above_threshold=sum(r['torso_points_above_threshold']=='1' for r in rows),ankle_pairs_above_threshold=sum(r['both_ankles_above_threshold']=='1' for r in rows))
    assert counts['court']['selected_native_far_boxes']==counts['far_context']['selected_native_far_boxes']==133 and counts['court']['matched_pose_candidates']==47 and counts['far_context']['matched_pose_candidates']==124
    torso=OUT/'torso_appearance_diagnostic';tp=load(torso/'protocol.json');assert all(sha(Path(n))==h for n,h in tp['protected_input_hashes'].items());colors=read(torso/'appearance_comparison_diagnostic.csv');assert len(colors)==532;policies=defaultdict(dict)
    for r in colors:
        policies[r['tag'],r['source_frame'],r['mode'],r['player_slot']][r['policy']]=r
        if r['comparison_available']=='1':
            assert int(r['feature_pixels'])>=12 and r['own_anchor_available']=='1';own=float(r['own_distance']);comp=float(r['best_competing_distance']);assert math.isclose(float(r['own_minus_competing_distance']),own-comp,abs_tol=1e-9);assert int(r['would_disagree_with_native_slot'])==int(comp+.04<own)
        else:assert r['own_distance']==r['would_disagree_with_native_slot']==''
        assert r['identity_certified']=='0'
    disagreements=[]
    for key,group in policies.items():
        a,b=[group[x] for x in ['pose_box_upper_rectangle','pose_torso_polygon']]
        assert a['pose_index']==b['pose_index'] and a['pose_matched']==b['pose_matched'],'Mask comparison must use the same pose geometry.'
        if a['would_disagree_with_native_slot']!=b['would_disagree_with_native_slot']:disagreements.append(dict(tag=key[0],source_frame=int(key[1]),mode=key[2],player_slot=key[3],rectangle_disagrees=a['would_disagree_with_native_slot'],torso_disagrees=b['would_disagree_with_native_slot']))
    assert len(disagreements)==10 and len({(r['tag'],r['source_frame']) for r in disagreements})==7
    write(torso/'all_policy_disagreements.csv',disagreements)
    manifests=[]
    for folder,n in [(OUT/'AI_review',27),(torso/'policy_difference_review',7)]:
        m=load(folder/'manifest.json');assert len(m['items'])==n and all(sha(folder/r['image'])==r['image_sha256'] for r in m['items']);manifests.extend(m['items'])
    reviewed={(r['tag'],r['source_frame']) for r in manifests};assert len(reviewed)==30 and {(r['tag'],r['source_frame']) for r in disagreements}<=reviewed
    assessments=[]
    for tag,f in sorted(reviewed):
        note=('Pose torso follows actual orange player rather than cyan official within the native mixed box. Rectangle-color veto disagrees with native slot; torso-color veto does not. No measured ground location recovered.' if tag=='006' and f in [12421,12422] else 'Actual dark server has a pose at far_context scale and is not rejected by either color region in this diagnostic.' if tag=='016' and f in [40625,40626] else 'Torso-only color rule newly disagrees on a visibly present far player. At002F2319 the competing color anchor is the other far slot; at009F21945/F21975 it is near_front. Do not use this as a hard rejection.' if (tag,f) in {('002',2319),('009',21945),('009',21975)} else 'Far-context model follows cyan official for some frames; pose presence does not establish a player identity.' if tag=='019' and f in [46501,46502] else 'Official/dark-player mixed native rectangle remains without a separating pose proposal at this frame.' if tag=='019' and f in [46503,46505] else 'Far-context pose torso follows visible dark player inside official+player mixed native rectangle; original native ID/box remains unchanged.' if tag=='019' and f==46511 else 'Orange player receives the pose; rear dark player is not separately localized. Do not infer hidden ground position.' if tag=='019' and f in [46633,46634] else 'White/red near player is still given native far F2 label; pose-derived color disagrees and favors a near-side appearance anchor. No identity reassignment made.' if tag=='019' and f in [46665,46666] else 'Visible head/body fragment is not separately recovered as a full person by either pose scale.' if tag=='019' and f==46352 else 'Selected control/overlap image inspected for body-part feasibility. Keypoints and native IDs are not independently verified truth.')
        assessments.append(dict(tag=tag,source_frame=f,AI_observation=note,independent_human_GT=0,ground_position_certified=0))
    write(OUT/'AI_visual_findings.csv',assessments)
    video=OUT/'temporal_demo';vp=load(video/'protocol.json');assert all(sha(Path(n))==h for n,h in vp['protected_input_hashes'].items());v=load(video/'verification.json');assert v['source_frames']==152 and v['decoded_video_frames']==456 and v['reused_probe_frames']==17 and v['inference_calls']==135
    assert sha(video/'body_part_feasibility_h264.mp4')==v['video_sha256'] and sha(video/'video_QA.jpg')==v['QA_sha256']
    cached={(r['tag'],r['source_frame']):r['modes']['far_context'] for r in pred['frames']};temporal=load(video/'pose_predictions.json')['frames'];assert len(temporal)==152
    for r in temporal:
        if r['reused_frozen_prediction']:assert r['people']==cached[r['tag'],r['source_frame']]
    previous_files=0
    for prior in [BASE/'current22_player_inputs',BASE/'cohort_tracking/case_019',BASE/'pair_initialization_transfer_019',BASE/'far_identity_guard_development',BASE/'far_spatial_guard_development']:
        for n,h in load(prior/'publication_hashes.json').items():assert sha(prior/n)==h;previous_files+=1
    current=BASE/'current22_player_inputs';cr=load(current/'result.json');assert len(read(current/'confirmed_human_outcomes.csv'))==11 and cr['fixed30_constructed_inputs']==19 and cr['next_candidate']=='match1_rally_window_019'
    decision=dict(status='body_part_candidate_promising_not_adopted',evidence='Pose can separate the actual torso from the official at019F46511 and006F12421/12422. Pose-derived color raises a near-side appearance conflict for019F46665/F46666. Rear hidden player at019F46634 is not recovered.',limitations='Mask-only color comparison has visibly present control-player rejection counterexamples002F2319 and009F21945/F21975. Identical clothing, small torso regions, pose contamination and one-frame anchors remain limitations. Keypoint confidence is not identity confidence or measured ground contact.',no_improvement_rate_claim=True,canonical_pipeline_replaced=False,current22_unchanged=True,confirmed_human_outcomes=11,fixed30_constructed_inputs=19,fixed30_pending_inputs=11,next_unresolved_candidate='match1_rally_window_019',next_action='Use body-part evidence as a supplementary association signal and distinguish same-team ID conflicts from opposite-team/nonplayer conflicts. Test a fixed temporal association comparison on full retained cases before adoption; do not replace foot coordinates or reject on torso color alone.')
    dump(OUT/'processing_decision.json',decision)
    report=dict(status='passed',scope='Scientific artifact/coordinate/assignment/provenance validation, not tracking accuracy',selected_frames=74,pose_scale_comparisons=148,native_box_pose_rows_checked=548,pose_proposals_checked=pose_count,small_matching_problems_verified_exhaustively=100,pose_output_availability_not_accuracy=counts,appearance_diagnostic_rows=532,appearance_rows_available=342,feature_policy_disagreement_rows=10,all7_feature_disagreement_frames_AI_inspected=True,AI_initial_review_frames=27,AI_difference_review_frames=7,AI_unique_reviewed_frames=30,video_QA_frames_inspected=6,explanatory_temporal_source_frames=152,extra_inference_calls=135,reused_frozen_pose_frames=17,decoded_video_frames=456,previous_published_files_unchanged=previous_files,current22_inputs_and11_outcomes_unchanged=True,canonical_pipeline_changed=False,new_human_labels=False,independent_identity_or_ground_position_accuracy_established=False)
    dump(OUT/'final_verification.json',report)
    text='''# 身体部位による混合人物枠の診断：胴体は有望、IDと接地点は未解決

前段の位置＋外観拒否が混合枠を解消できなかったため、既存のYOLO11m-poseを追加学習なしで試した。同じ10区間から、初期化、指定序盤の先頭・中央・末尾、元経路の奥同士／奥と手前の最大重なり、および既知の問題場面を推論前に固定。計74フレームで、従来のコート全体切り出しと奥側拡大切り出しの2条件を独立に保存した。既知失敗を見た後の同一試合内開発で、独立テストではない。

重なり最大点は元枠から選び、得点ラベルを使っていない。将来の全体文脈から選んだ診断用時点であり、オンライン動作や得点予測入力の追加ではない。指定序盤の範囲と既存入力表は変更していない。

|条件|選択時点に元の奥枠がある行|姿勢枠との対応候補が得られた行|
|---|---:|---:|
|コート全体|133|47|
|奥側拡大|133|124|

これは元の奥枠が存在する行に条件づけた出力数であり、追跡率・再現率・本人同定精度ではない。元経路の奥枠自体が欠ける時点はこの分母に含まれない。隣接フレームや2条件を独立試行として扱わない。

モデルは17点の身体部位を推定する。確信度0.5以上の肩・腰で胴体領域を作り、足首も別に記録した。元枠と姿勢枠はIoU0.5以上の1対1対応で比較。1個の姿勢を2選手に使わないが、この幾何対応自体は本人確認ではない。既存の推論条件と閾値を流用し、数値探索をしていない。SciPyのない環境に合わせ、4人までの厳密な割当てを動的計画法で実装し、100個の小問題を全探索解と照合した。

019のF46511では、審判と濃色選手を含む元枠に対して、奥側拡大の姿勢は濃色選手側の胴体を示した。006のF12421・12422でも、審判に近い実選手の胴体を抽出できる例を確認した。一方、019のF46501・46502では審判にも姿勢が出ており、「骨格があるから実選手」とは言えない。F46633・46634の後ろの濃色選手は、前の橙色選手と分離できなかった。F46352の頭部断片も全身として回復していない。

さらに、同じ姿勢枠について上半身の長方形と肩腰の胴体多角形からHSV色特徴を計算し、領域の違いだけを比較した。両条件で同じ初期姿勢・同じ候補姿勢を使う。初期参照は各区間の初期化1フレームで、従来の8フレーム参照の追跡処理を再現した比較ではない。奥参照には奥拡大、手前参照には全体の姿勢を使う。この色比較の条件は姿勢画像を見た後、色比較の出力前に別途固定した。

019のF46665・46666で、元F2に割り当てられた白赤選手の胴体色は、奥F2より手前側の参照に近いと判定された。006のF12421・12422では長方形から胴体領域へ変えると負例寄りの判定が消えた。ただし002のF2319、009のF21945・21975では、実選手が見える枠にも胴体色による新たな拒否判定が出た。002では同じ奥側の他方の選手の服装が競合する。服装だけで個人IDを確定する規則にはできない。

色比較は532行、特徴と参照がそろった342行を保存。2領域の拒否判定が異なる全10行・7フレームを追加表示し、すべてAI点検した。最初の既知例＋各区間の奥ペア最大重なり27フレームと合わせて、重複を除き30フレームを点検。これは人手の独立正解でも、全74フレームの座標注釈でもない。胴体を使うほうが常に正しいという結果は得られていない。

temporal_demo/body_part_feasibility_h264.mp4には006・016・019の連続した5区間、計152元フレームを表示した。左は元枠と元ID、右は同じ元枠・IDに奥拡大の骨格を重ねたもの。右もIDを修正した追跡結果ではない。1/3速度で456フレーム、縦横比を保持し復号確認。17フレームは固定済みの姿勢出力を再利用し、残り135フレームを同条件で追加推論した。映像は説明用の事後選択で、独立した時間方向の性能評価ではない。

足首はモデル推定値で、確信度が高くても本人の見えている足や接地点を保証しない。ジャンプや遮蔽を含むため、そのままコート平面上の実測位置へ置き換えない。姿勢の胴体中心と足首は診断CSVに分離し、正式な選手位置や配置特徴には出力していない。

この段階では全面採用しない。019は未解決。既存の配置入力と11件の確認済み得点ラベル、以前の失敗と比較結果を保持し、固定30群は入力作成19件・未完了11件のまま。学習、得点予測性能の評価、球の追加価値比較は行っていない。

次に進める価値があるのは、混合枠全体の服装ではなく本人の胴体を追跡の補助情報として使うこと。ただし同じチーム内のID競合と相手チーム・審判への乗り換えを分け、胴体色だけの強制拒否を避ける。全区間の時間方向比較を固定条件で行い、正しい観測を失う副作用を確認するまで元経路を置き換えない。
'''
    (OUT/'README.md').write_text(text,encoding='utf-8');(torso/'README.md').write_text('同じ姿勢提案について長方形と胴体多角形の色特徴を比較した診断。単一の初期化フレーム参照で、既存の8フレーム参照の追跡性能比較ではない。対照の拒否反例を含め、変更は未採用。親README.md参照。\n',encoding='utf-8')
    archive=OUT/'executed_sources';archive.mkdir()
    for name in ['run_body_part_localization_probe.py','prepare_body_part_probe_review.py','diagnose_pose_torso_appearance.py','prepare_body_part_policy_difference_review.py','render_body_part_temporal_probe.py','finalize_body_part_localization_probe.py']:shutil.copy2(ROOT/name,archive/name)
    day=BASE/'README.md';content=day.read_text(encoding='utf-8');marker='019の混合枠を身体部位で分ける試験';assert marker not in content
    day.write_text(content+'\n\n'+marker+'。同じ10区間の固定74フレームを2尺度で姿勢推定。019F46511・006の混合枠で実選手側の胴体が得られる例があるが、019F46634の後方人物は未分離。胴体色は019F46665/F46666の手前服装への一致を示す一方、002・009に拒否の反例が残るため採用しない。色領域を比較した全判定差7フレームを含む計30フレームをAI点検。連続152フレームの説明映像も保存、右側もID未修正。足首推定を接地位置へ昇格させずcurrent22・11ラベル・19/30を保持。詳細body_part_localization_probe/README.md。\n',encoding='utf-8')
    dump(OUT/'publication_hashes.json',{str(n.relative_to(OUT)):sha(n) for n in sorted(OUT.rglob('*')) if n.is_file() and n.name!='publication_hashes.json'});print(report)

if __name__=='__main__':main()
