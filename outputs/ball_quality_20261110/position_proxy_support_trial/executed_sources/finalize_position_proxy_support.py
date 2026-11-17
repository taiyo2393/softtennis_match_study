"""Verify warning/sensitivity artifacts without certifying model-inferred feet."""
import math,shutil
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,write,sha,dump
import run_position_proxy_support as old
import run_position_proxy_support_head_independent as new

OUT=old.OUT

def geometric_checks():
    # A back-facing torso must not require an inferred face; an unsupported head adds no disc.
    kp=np.zeros((17,3),float);kp[[5,6,12,11]]=[[40,40,.9],[60,40,.9],[60,65,.9],[40,65,.9]];kp[0]=[50,15,.1];person=dict(box=[20,0,80,100],keypoints=kp.tolist())
    assert not old.body_supported(person) and new.body_supported(person)
    assert new.inside_body([50,50],person) and not new.inside_body([50,15],person)
    assert not new.inside_body([90,50],person)
    missing=dict(person,keypoints=np.zeros((17,3)).tolist());assert not new.body_supported(missing) and not new.inside_body([50,50],missing)

def complete_prefix(stage,module,p):
    added=[];total=0
    for e in p['cases']:
        tag=e['tag'];dest=stage/f'case_{tag}';original=read(dest/'fixed_prefix_position_support.csv');by={(int(r['source_frame']),r['player_slot']):r for r in original};table=module.native_table(e);poses=module.all_pose_geometry(tag);rows=[]
        for f in e['requested_input_frames']:
            assert all((f,m) in poses for m in module.CROPS),'Every requested input time must have both inference views, even if identity is missing.'
            if (f,module.SLOTS[0]) not in by:
                assert f<e['initialization']['frame'] and not table[f]
                generated=module.support_for_frame(tag,f,{},poses)
                for r in generated:r['position_support_state']='native_missing_before_initialization';by[f,r['player_slot']]=r
                added.append(dict(tag=tag,source_frame=f,missing_native_identity_slots=4,reason='Fixed input time precedes retained four-person initialization; no pose identity or proxy substituted.'))
            for slot in module.SLOTS:rows.append(by[f,slot])
        assert len(rows)==124 and len({(str(r['source_frame']),r['player_slot']) for r in rows})==124
        write(dest/'fixed_prefix_position_support_complete.csv',rows);total+=len(rows)
    assert total==1240 and len(added)==20 and {r['tag'] for r in added}=={'006'}
    dump(stage/'fixed_prefix_completion.json',dict(total_rows=1240,requested_times=310,before_initialization_times_explicitly_added=added,additional_missing_slot_rows=80,original_partial_sidecar_files_retained=True,feature_values_changed=False));return added

def main():
    assert not (OUT/'final_verification.json').exists();geometric_checks();p=old.load(OUT/'protocol.json');np_=new.load(new.OUT/'protocol.json')
    for protocol in [p,np_]:assert all(sha(Path(n))==h for n,h in protocol['protected_input_hashes'].items())
    assert sha(Path(p['source_video']))==p['source_video_sha256'];model_result=old.load(OUT/'pose_result.json');assert model_result['new_inference_calls']==566 and model_result['reused_predictions']==178
    complete_prefix(OUT,old,p);complete_prefix(new.OUT,new,np_)
    total_rows=pose_rows=observed_prefix=box_risks=body_risks=0;changes=[];coverage=[];max_mean_difference={};snapshot_changes={};risk_summary=[]
    for e in p['cases']:
        tag=e['tag'];native=old.native_table(e);left=read(OUT/f'case_{tag}/position_proxy_support.csv');right=read(new.OUT/f'case_{tag}/position_proxy_support.csv');assert len(left)==len(right)==4*e['full_search_frames'];total_rows+=len(right)
        for a,b in zip(left,right):
            assert (a['source_frame'],a['player_slot'])==(b['source_frame'],b['player_slot']);f=int(b['source_frame']);slot=b['player_slot'];r=native[f].get(slot)
            assert a['ground_position_certified']==a['identity_certified']==b['ground_position_certified']==b['identity_certified']=='0'
            assert a['native_observation_present']==b['native_observation_present']==str(int(r is not None));assert a['geometry_box_overlap_risk']==b['geometry_box_overlap_risk']
            assert int(a['geometry_body_overlap_risk'])<=int(b['geometry_body_overlap_risk'])
            if r is not None:
                assert b['native_track_id']==r['track_id'] and b['native_proxy_x']==r['bottom_center_x'] and b['native_proxy_y']==r['bottom_center_y'];point=(float(r['bottom_center_x']),float(r['bottom_center_y']));expected={s for s,n in native[f].items() if s!=slot and float(n['x1'])<=point[0]<=float(n['x2']) and float(n['y1'])<=point[1]<=float(n['y2'])};assert expected==set(filter(None,b['native_proxy_in_other_boxes'].split('|')))
                assert slot not in set(filter(None,b['native_proxy_in_other_pose_bodies'].split('|')))
            else:assert b['native_proxy_x']==b['native_proxy_y']==b['native_track_id']==''
        tensors=np.load(OUT/f'case_{tag}/pose_geometry.npz');boxes=tensors['boxes'];joints=tensors['keypoints'];meta=read(OUT/f'case_{tag}/pose_metadata.csv');assert len(meta)==len(boxes)==len(joints);pose_rows+=len(meta);assert boxes.shape[1:]==(5,) and joints.shape[1:]==(17,3) and np.isfinite(boxes).all() and np.isfinite(joints).all()
        prefix=read(new.OUT/f'case_{tag}/fixed_prefix_position_support_complete.csv');assert len(prefix)==124 and {int(r['source_frame']) for r in prefix}==set(e['requested_input_frames'])
        for r in prefix:
            assert r['pose_frame_evaluated']=='1'
            if r['native_observation_present']=='1':observed_prefix+=1
            box_risks+=int(r['geometry_box_overlap_risk']);body_risks+=int(r['geometry_body_overlap_risk'])
            overlap=set(filter(None,r['native_proxy_in_other_boxes'].split('|')));evaluated=set(filter(None,r['other_pose_bodies_evaluated'].split('|')))
            if overlap:coverage.append(dict(tag=tag,source_frame=r['source_frame'],player_slot=r['player_slot'],overlapping_native_slots='|'.join(sorted(overlap)),overlapping_slots_with_body_support='|'.join(sorted(overlap&evaluated)),overlapping_slots_without_body_support='|'.join(sorted(overlap-evaluated)),all_overlapping_bodies_evaluated=int(overlap<=evaluated),ground_position_certified=0))
            if r['geometry_body_overlap_risk']=='1':changes.append(r)
        for policy in old.POLICIES:
            # Rectangle policy and no-mask policy must remain bit-identical between versions.
            if policy!='mask_pose_body_overlap':
                for name in ['diagnostic_prefix_frame_features.csv','diagnostic_snapshot_features.csv','diagnostic_prefix_summary.csv']:assert read(OUT/f'case_{tag}/{policy}/{name}')==read(new.OUT/f'case_{tag}/{policy}/{name}')
    assert total_rows==28076 and observed_prefix==1121 and box_risks==30 and body_risks==4 and len(coverage)==30 and all(r['all_overlapping_bodies_evaluated'] for r in coverage)
    assert {(r['tag'],int(r['source_frame']),r['player_slot']) for r in changes}=={('002',2303,'far_front'),('017',42132,'far_front'),('017',42138,'far_front'),('019',46396,'far_rear')}
    write(new.OUT/'all_fixed_prefix_body_warning_rows.csv',changes);write(new.OUT/'rectangle_risk_body_coverage.csv',coverage)
    for policy in ['mask_native_box_overlap','mask_pose_body_overlap']:
        delta=[];snapshot=[]
        for e in p['cases']:
            tag=e['tag'];baseline=read(new.OUT/f'case_{tag}/retain_native_proxies/diagnostic_prefix_summary.csv')[0];variant=read(new.OUT/f'case_{tag}/{policy}/diagnostic_prefix_summary.csv')[0]
            for k in baseline:
                if k.endswith('_mean') and baseline[k]!='' and variant[k]!='':delta.append(dict(tag=tag,feature=k,absolute_difference=abs(float(baseline[k])-float(variant[k]))))
            if read(new.OUT/f'case_{tag}/retain_native_proxies/diagnostic_snapshot_features.csv')!=read(new.OUT/f'case_{tag}/{policy}/diagnostic_snapshot_features.csv'):snapshot.append(tag)
        max_mean_difference[policy]=max(delta,key=lambda r:r['absolute_difference']);snapshot_changes[policy]=snapshot
    assert snapshot_changes=={'mask_native_box_overlap':['002'],'mask_pose_body_overlap':[]}
    dump(new.OUT/'feature_sensitivity_diagnostic.json',dict(max_absolute_mean_feature_difference=max_mean_difference,snapshot_changed_cases=snapshot_changes,interpretation='Sensitivity to these masks only, not measurement error or accuracy. Fewer retained points also change missingness and can select particular geometries.'))
    manifest=old.load(new.OUT/'AI_review/manifest.json');assert len(manifest['items'])==14 and all(sha(new.OUT/'AI_review'/r['image'])==r['image_sha256'] for r in manifest['items']);reviewed={(r['tag'],r['source_frame'],r['slot']) for r in manifest['items']};assert {(r['tag'],int(r['source_frame']),r['player_slot']) for r in changes}<=reviewed
    findings=[]
    for r in manifest['items']:
        tag,f=r['tag'],r['source_frame'];note=('Rear player remains visible but the native lower-edge point lies on the foreground person. Both model ankles can still be confident; they are not independent visible-foot truth.' if (tag,f) in {('002',2303),('017',42132),('017',42138),('019',46396)} else 'Only a head/body fragment is visible behind the near player; the rectangle bottom lies on the near player. Identity observation may be kept while the image-position proxy is uncertain.' if tag=='019' and f==46352 else 'Mixed official/dark-player rectangle; native proxy overlaps another native rectangle but not its estimated body approximation. This is not a certificate of correct position.' if tag=='019' and f==46511 else 'Dark player visible behind orange teammate; native proxy lies on teammate body approximation. Own pose is not separately matched, no hidden-foot coordinate recovered.' if tag=='019' and f==46634 else 'Native proxy and inferred ankles remain uncertain despite no positive body-overlap flag; models do not establish ground contact.' if tag=='019' and f in [46406,46436] else 'Rectangle includes empty image space. Rectangle-overlap warning is broader than pose-body overlap; no independent position truth added.')
        findings.append(dict(tag=tag,source_frame=f,slot=r['slot'],box_risk=r['box_risk'],body_risk=r['body_risk'],AI_observation=note,independent_human_GT=0,image=str(new.OUT/'AI_review'/r['image']),image_sha256=r['image_sha256']))
    write(new.OUT/'AI_visual_findings.csv',findings)
    vv=old.load(new.OUT/'comparison_video/verification.json');assert vv['decoded_frames']==504 and vv['aspect_preserved'];assert sha(new.OUT/'comparison_video/position_proxy_support_h264.mp4')==vv['video_sha256'] and sha(new.OUT/'comparison_video/video_QA.jpg')==vv['QA_sha256']
    previous_files=0
    for stage in [BASE/'current22_player_inputs',BASE/'cohort_tracking/case_019',BASE/'pair_initialization_transfer_019',BASE/'far_identity_guard_development',BASE/'far_spatial_guard_development',BASE/'body_part_localization_probe',PRIOR]:
        for name,h in old.load(stage/'publication_hashes.json').items():assert sha(stage/name)==h;previous_files+=1
    for protocol in [p,np_]:assert all(sha(Path(n))==h for n,h in protocol['protected_input_hashes'].items())
    current=BASE/'current22_player_inputs';cr=old.load(current/'result.json');assert len(read(current/'confirmed_human_outcomes.csv'))==11 and cr['fixed30_constructed_inputs']==19 and cr['next_candidate']=='match1_rally_window_019'
    decision=dict(status='position_warning_sidecar_retained_masking_not_adopted',identity_paths_unchanged=True,diagnostic_layer='Position warning rows separate from native person ID observations; no model foot position is promoted to an observed coordinate.',head_independent_variant='Shoulder/hip support allows back-facing people; the head approximation itself still needs nose confidence.',known019_F46511='Native proxy overlaps a box only; no positive body warning in this model. Mixed-box quality and position remain unverified.',known019_F46634='Native proxy lies on other-player body approximation; keep identity observation separate from position warning. Hidden player position not recovered.',fixed_prefix_body_warning_native_rows=4,masking_policy_adopted=False,absence_of_warning_is_not_accuracy=True,current22_unchanged=True,confirmed_human_outcomes=11,fixed30_constructed_inputs=19,fixed30_pending_inputs=11,next_unresolved_candidate='match1_rally_window_019',next_action='Retain warnings and evaluate any position filter with independent image-position reference points and predeclared tolerances on a small sample. Avoid contact-frame microlabeling, blanket overlap removal, and silently dropping unassessed or pre-initialization times.')
    dump(OUT/'processing_decision.json',decision)
    report=dict(status='passed',scope='Position warning geometry/provenance/feature-sensitivity checks and bounded AI inspection, not positional accuracy',contexts=10,search_frame_slot_rows_per_variant=28076,fixed_input_times=310,complete_fixed_prefix_slot_rows_per_variant=1240,before_initialization_missing_times006=20,before_initialization_missing_slot_rows_per_variant=80,observed_native_prefix_rows=1121,fixed_prefix_native_box_warning_rows=30,all30_box_overlap_occluders_have_body_support_in_head_independent_variant=True,fixed_prefix_body_warning_rows=4,body_mask_affected_cases=['002','017','019'],body_mask_snapshots_unchanged_cases=10,max_absolute_mean_feature_difference_body_mask=max_mean_difference['mask_pose_body_overlap']['absolute_difference'],new_pose_inference_calls=566,reused_pose_predictions=178,additional_variant_inference_calls=0,new_cached_pose_proposals=pose_rows,independently_recalculated_feature_values_per_variant=11370,AI_reviewed_frames=14,all4_fixed_prefix_body_mask_changes_AI_inspected=True,video_decoded_frames=504,video_QA_frames_inspected=5,previous_published_files_unchanged=previous_files,identity_paths_changed=False,model_coordinates_adopted=False,mask_policy_adopted=False,current22_inputs_and11_outcomes_unchanged=True,independent_identity_or_position_accuracy_established=False)
    dump(OUT/'final_verification.json',report)
    text='''# 人物のIDと位置の注意情報を分ける試験

前段で019の別人・審判への6枠を欠測に戻せたが、混合枠と遮蔽時の位置が残った。今回は追跡IDを変更せず、検出枠の下端中央を配置の代理点として使う際の注意情報を別保存した。足の接地タイミングの細かな注釈や、隠れた足の補間は行っていない。位置は画像上の代理点であり、コート平面上の実測接地点ではない。

同じ10区間で各指定序盤31時点を維持し、初期化から末尾まで7019フレーム×4枠の診断表を保存した。指定時点を姿勢のある近いフレームへずらしていない。全310指定時点を両尺度で確認できるよう372組の時点を事前固定し、既存178出力を再利用、566回の姿勢推論を追加。2つの019近傍と対照時点も含む。

006の指定31時点のうち20時点は元の4人追跡初期化より前で、ID付き観測がない。最初の診断表は初期化以後だけを持っていたため、元ファイルを残しfixed_prefix_position_support_complete.csvへ4枠分の明示的な欠測を加えた。各条件で全1240行がそろう。姿勢が見つかっても、この80枠を同定済みの選手位置へ置き換えていない。配置特徴の計算は元から31時点を分母にしており、表の補完で数値は変わらない。

2種類の注意を比較した。第1は代理点が別選手の長方形枠に入る場合。第2は代理点が、別選手と幾何対応する姿勢の身体近似形状に入る場合。身体形状は肩腰の多角形・頭部円・腕脚のカプセルで、044の既存比率を使用。可視画素のセグメンテーションや本物の遮蔽判定ではない。姿勢や身体が得られない場合は未評価として扱い、注意なしを正しい座標と認定しない。

最初は旧処理に合わせ鼻・肩・腰の確信度を必須にしたが、背中向きの選手では鼻が不確かでも肩腰は推定されるため、重要な身体を評価できなかった。元結果を保持し、head_independent_body_supportに肩腰を胴体の条件とする別試作を固定して比較した。頭部円だけは引き続き鼻の確信度を要求する。距離の新しい閾値や数値探索は加えず、追加モデル推論もしていない。ただし最初の結果を見た後の逐次的な同一試合内開発で、独立テストではない。

|指定序盤の仮想的位置フィルタ|除外される代理点|4人分の代理値がそろう時点の合計|
|---|---:|---:|
|元の代理点を維持|0|251/310|
|別選手の長方形枠と重なる点を除外|30|221/310|
|肩腰に基づく身体近似形状と重なる点を除外|4|247/310|

この数は診断上の出力数であり、正しい位置の数・追跡精度ではない。長方形枠は身体以外の空間も含むので、重なりだけの一律除外は厳しすぎる。002では22/31時点から5/31時点まで減る。身体形状の4点は002F2303、017F42132・42138、019F46396。全4点と枠だけの重なり例、019の残る混合／断片例など14画像をAI点検した。4点では背後の選手が見えても、元の下端代理点が前方選手に重なる例が確認できる。IDを消す理由と位置に注意する理由を分ける必要がある。

019F46634はF1の代理点がF2の身体近似形状に重なり、F1自身の姿勢は別に対応しない。F46511は混合枠があるが、代理点は別枠と重なるだけで、このモデルでは別身体への重なりを検出しない。後者の代理点が正しいと認定したものではない。F46406・46436なども、推定足首の確信度が高いことから本当に見える足・接地点とは断定しない。警告に陽性がないことと位置の正しさは区別する。

この310時点では、長方形重なりの30点すべてで相手側の身体近似も評価できたことを別表で確認した。単に身体推定が欠けたため4点になったわけではない。ただしそれ以外の全区間には姿勢を取得していない時点や対応しない人物もあり、全区間の警告なしを安全判定へ使わない。身体近似形状自体の妥当性も独立評価していない。

仮想マスクごとに配置値を独立計算と照合した。肩腰条件の4点を除外した場合、全10区間の指定末尾スナップショットは変わらず、平均配置特徴の最大差は正規化画像座標で約0.000656だった。この限定された感度分析は位置誤差や得点側との関係を評価した結果ではない。他の未検出誤差は残り、遮蔽・接近に依存する欠測を増やすと研究対象の配置分布を偏らせる可能性がある。

head_independent_body_support/comparison_video/position_proxy_support_h264.mp4に、元画像と身体近似を重ねた画像を同じ縮尺で表示した。橙点は元の代理点、赤い×はモデル足首で正解点ではない。3場面は静止画、019の2近傍は1/6速度。504フレームを復号し縦横比を確認した。表示は診断例の事後選択で、盲検・独立の位置評価ではない。

位置注意の別表は診断用として保持し、いずれの除外条件も正式入力に採用しない。元ID・元追跡・current22_player_inputs・11件の確認済み得点ラベルを維持。固定30群は入力作成19件・未完了11件のまま、019を未解決に残す。モデル位置を実測値へ昇格させず、得点予測の学習・精度評価・球の追加価値比較は行っていない。

次に位置除外を採用するなら、小さな独立の画像位置参照と許容誤差を先に定め、誤って残す点と不要に捨てる点の両方を調べる必要がある。全選手の接地フレームを精密注釈する必要はない。警告と未評価を保持して、追跡の出現数だけで位置の正しさを判断しない。
'''
    (OUT/'README.md').write_text(text,encoding='utf-8');(new.OUT/'README.md').write_text('背中向きの身体も扱う位置注意の診断版。肩腰で胴体を作り、頭部円には鼻の確信度を要求。元IDは保持し、位置マスクは未採用。正式入力にはモデル足首を追加しない。全固定時点の別表は各caseのfixed_prefix_position_support_complete.csv。親README.md参照。\n',encoding='utf-8')
    archive=OUT/'executed_sources';archive.mkdir()
    for name in ['run_position_proxy_support.py','run_position_proxy_support_head_independent.py','review_position_proxy_support.py','render_position_proxy_support.py','finalize_position_proxy_support.py']:shutil.copy2(ROOT/name,archive/name)
    dump(OUT/'frozen_warning_policy.json',dict(status='diagnostic_sidecar_only_not_input_filter',native_IDs_unchanged=True,preferred_variant='head_independent_body_support',parameters=new.BODY,protocol_sha256=sha(new.OUT/'protocol.json'),point_inside_body_is_warning_not_GT=True,no_warning_is_not_certified_position=True,contact_frame_annotation_required=False,independent_position_accuracy_established=False))
    day=BASE/'README.md';content=day.read_text(encoding='utf-8');marker='人物IDと位置代理点の注意情報を分離';assert marker not in content
    day.write_text(content+'\n\n'+marker+'。同じ10区間の固定310時点を動かさず姿勢情報を補い、別表に位置注意を保存。006の初期化前20時点も明示欠測として記録。長方形枠への重なりだけの除外は30点、背中向きも扱う肩腰の身体近似では4点が対象。全4点を含む14画像をAI点検。019F46634には身体重なり警告、F46511は枠だけの重なりで位置を正しいと認定しない。4点の仮想除外では末尾スナップショット不変だが、独立位置精度は未評価。位置フィルタは未採用、元ID・current22・11ラベル・19/30保持。詳細position_proxy_support_trial/README.md。\n',encoding='utf-8')
    dump(OUT/'publication_hashes.json',{str(n.relative_to(OUT)):sha(n) for n in sorted(OUT.rglob('*')) if n.is_file() and n.name!='publication_hashes.json'});print(report)

PRIOR=old.PRIOR
if __name__=='__main__':main()
