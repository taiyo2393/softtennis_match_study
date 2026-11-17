"""Freeze the separated export and reference-pilot design, not live manual labels."""
import json,random,shutil
from pathlib import Path
from collections import defaultdict,Counter
import cv2,numpy as np
from run_fixed_cohort_player_case_v3 import ROOT,BASE,read,write,sha,dump
import run_position_proxy_support_head_independent as support
from verify_player_prefix_expansion import close
from evaluate_position_reference_pilot import self_test

OUT=BASE/'identity_position_bundle'
PILOT=OUT/'reference_pilot'

def main():
    assert not (OUT/'final_verification.json').exists();p=support.load(OUT/'protocol.json');assert all(sha(Path(n))==h for n,h in p['protected_input_hashes'].items());assert self_test()['status']=='passed'
    native=read(OUT/'native_identity_observations.csv');variants=read(OUT/'position_policy_observations.csv');assert len(native)==28156 and len(variants)==56312;lookup={(r['candidate_id'],r['source_frame'],r['player_slot']):r for r in native};assert len(lookup)==len(native);seen=set();withheld=[];prefix_withheld=[];by_case={e['tag']:e for e in p['cases']}
    tables={e['tag']:support.native_table(e) for e in p['cases']};qualities={e['tag']:{(r['source_frame'],r['player_slot']):r for r in read(support.OUT/f'case_{e["tag"]}/position_proxy_support.csv')+read(support.OUT/f'case_{e["tag"]}/fixed_prefix_position_support_complete.csv')} for e in p['cases']}
    for r in variants:
        k=r['candidate_id'],r['source_frame'],r['player_slot'];assert (*k,r['position_policy']) not in seen;seen.add((*k,r['position_policy']));original=lookup[k];assert all(r[field]==original[field] for field in original);tag=r['candidate_id'][-3:];f=int(r['source_frame']);actual=tables[tag][f].get(r['player_slot']);present=actual is not None;assert r['candidate_box_present']==str(int(present));assert r['position_certified']==r['independent_identity_certified']=='0';q=qualities[tag][r['source_frame'],r['player_slot']]
        assert r['body_overlap_warning']==q['geometry_body_overlap_risk'] and r['position_warning_state']==q['position_support_state'];held=present and r['position_policy']=='withhold_body_warning_proxy' and r['body_overlap_warning']=='1';available=present and not held;assert r['analysis_proxy_present']==str(int(available))
        if available:assert r['analysis_proxy_x']==actual['bottom_center_x'] and r['analysis_proxy_y']==actual['bottom_center_y']
        else:assert r['analysis_proxy_x']==r['analysis_proxy_y']==''
        if present:assert all(r[field]==actual[field] for field in ['x1','y1','x2','y2']) and r['native_track_id']==actual['track_id']
        if held:
            withheld.append(r)
            if f in by_case[tag]['requested_input_frames']:prefix_withheld.append(r)
    assert len(prefix_withheld)==4 and len(seen)==2*len(lookup)
    frame_features=read(OUT/'fixed_prefix_frame_features.csv');assert len(frame_features)==620
    for e in p['cases']:
        tag=e['tag']
        for policy,prior_policy in [('retain_native_proxy','retain_native_proxies'),('withhold_body_warning_proxy','mask_pose_body_overlap')]:
            current=[r for r in frame_features if r['rally_candidate_id'].endswith('_'+tag) and r['position_policy']==policy];old=read(support.OUT/f'case_{tag}/{prior_policy}/diagnostic_prefix_frame_features.csv');assert len(current)==len(old)==31
            for a,b in zip(current,old):
                for k,v in b.items():
                    if k=='rally_candidate_id':assert a[k]==v
                    else:close(a[k],v)
            for filename,previous in [('fixed_prefix_snapshot_features.csv','diagnostic_snapshot_features.csv'),('fixed_prefix_summary_features.csv','diagnostic_prefix_summary.csv')]:
                row=next(r for r in read(OUT/filename) if r['rally_candidate_id'].endswith('_'+tag) and r['position_policy']==policy);expected=read(support.OUT/f'case_{tag}/{prior_policy}/{previous}')[0]
                for k,v in expected.items():
                    if k=='rally_candidate_id':assert row[k]==v
                    else:close(row[k],v)
    private=support.load(PILOT/'private_sampling_manifest.json');plan=support.load(PILOT/'private_sampling_protocol.json');public=support.load(PILOT/'public_manifest.json');assert len(private['items'])==len(public['items'])==13;assert {r['image_id'] for r in private['items']}=={r['image_id'] for r in public['items']}
    rng=random.Random(20261110);uniform={e['tag']:rng.choice(e['requested_input_frames']) for e in sorted(p['cases'],key=lambda e:e['tag'])};actual_uniform={r['tag']:r['source_frame'] for r in private['items'] if 'uniform_one_per_context' in r['sampling_groups']};assert uniform==actual_uniform and len(actual_uniform)==10
    challenge={(r['tag'],r['source_frame'],r['target_slot']) for r in private['items'] if 'known_body_warning_challenge' in r['sampling_groups']};assert challenge=={('002',2303,'far_front'),('017',42132,'far_front'),('017',42138,'far_front'),('019',46396,'far_rear')};assert len({(r['tag'],r['source_frame']) for r in private['items']})==13
    cap=cv2.VideoCapture(p['source_video'])
    for item in private['items']:
        path=PILOT/'images'/(item['image_id']+'.png');assert sha(path)==item['image_sha256'];cap.set(1,item['source_frame']);ok,raw=cap.read();assert ok and round(cap.get(1))==item['source_frame']+1;image=cv2.imread(str(path));assert image.shape==(2160,3840,3) and np.array_equal(raw,image)
    cap.release()
    for item in public['items']:assert set(item)=={'image_id','width','height','image_file'}
    assert uniform['006']<by_case['006']['initialization']['frame'],'Do not replace the sampled pre-initialization failure with an easier frame.'
    http=support.load(OUT/'server_http_qa/verification.json');assert http['status']=='passed' and http['real_manual_reference_file_created'] is False
    live=support.load(OUT/'live_http_verification.json');assert live['status']=='passed' and live['image_sha256_checks']==13 and live['private_routes_rejected']==3
    dump(OUT/'display_confirmation.json',dict(status='confirmed_by_user',source='User reply to HTTP-link display question',reported_answer='表示できた',components_requested='Match image plus four near/far-left/right selection fields',automated_browser_rendering_verified=False))
    # Reference data and runtime endpoints are mutable and deliberately outside this publication.
    manual=PILOT/'human_reference_labels.json';progress=dict(reference_entries=0,reviewed_images=0)
    if manual.exists():
        data=support.load(manual);assert data['label_origin']=='manual_reference_ui';progress['reference_entries']=sum(e.get('state','pending')!='pending' for r in data['labels'].values() for e in r.values());progress['reviewed_images']=sum(all(r.get(s,{}).get('state','pending')!='pending' for s in ['near_left','near_right','far_left','far_right']) for r in data['labels'].values())
    previous_files=0
    for previous in [BASE/'current22_player_inputs',BASE/'cohort_tracking/case_019',BASE/'pair_initialization_transfer_019',BASE/'far_identity_guard_development',BASE/'far_spatial_guard_development',BASE/'body_part_localization_probe',BASE/'torso_temporal_association_trial',BASE/'position_proxy_support_trial']:
        for name,h in support.load(previous/'publication_hashes.json').items():assert sha(previous/name)==h;previous_files+=1
    current=BASE/'current22_player_inputs';cr=support.load(current/'result.json');assert len(read(current/'confirmed_human_outcomes.csv'))==11 and cr['fixed30_constructed_inputs']==19 and cr['next_candidate']=='match1_rally_window_019'
    report=dict(status='passed',scope='Separated export/provenance/image/source/reference-tool validation, not positional accuracy',contexts=10,frame_slot_rows=28156,position_policy_rows=56312,fixed_prefix_frame_feature_rows=620,feature_values_independently_checked_at_construction=7580,withheld_proxy_rows_full_context=len(withheld),withheld_proxy_rows_fixed_prefix=4,identity_fields_preserved_between_policies=True,model_coordinates_substituted=False,reference_images=13,uniform_context_samples=10,known_warning_challenge_samples=4,one_image_in_both_sampling_groups=True,manual_reference_slots=52,source_PNGs_pixel_exact_to_video_frames=13,pre_initialization006_uniform_sample_retained=True,synthetic_HTTP_checks=11,synthetic_evaluator_cases=5,frontend_JS_syntax='passed with node24.22.1',live_image_assets_sha_checked=13,display_confirmed_by_user=True,automatic_browser_rendering_verified=False,agent_authored_real_reference_labels=False,reference_progress_at_finalization=progress,reference_accuracy_not_yet_evaluated=True,previous_publication_files_preserved=previous_files,current22_inputs_and11_outcomes_unchanged=True,position_filter_adopted=False)
    dump(OUT/'final_verification.json',report);dump(OUT/'processing_decision.json',dict(status='separated_candidate_export_ready_reference_collection_pending',canonical_pipeline_replaced=False,current22_unchanged=True,fixed30_constructed_inputs=19,fixed30_pending_inputs=11,confirmed_human_outcomes=11,reference_accuracy_evaluated=False,next_action='Collect independent-of-prediction approximate image-position references through the13-image UI. Keep unknowns, retain sampled missing predictions, and evaluate uniform/challenge groups separately. No ID-continuity or physical ground-contact claims.'))
    (OUT/'README.md').write_text('''# 人物候補と配置位置を分けた比較用データ

native_identity_observations.csvに元の人物候補・ID・検出枠を残し、position_policy_observations.csvに位置代理値の扱いを別保存した。28156枠には006の初期化前20時点も明示的な欠測で含む。元の位置を維持する条件と、身体重なり警告の位置だけを欠測にする条件を並べ、IDや元枠を削除しない。代理点は元検出の下端中央で、モデル足首や補間値は入力にしない。正式処理とcurrent22_player_inputsは置き換えていない。

固定310時点の620特徴行は以前の診断値と一致し、7580値を独立計算と照合した。序盤の位置だけを欠測にする対象は4枠。警告なしや未評価を位置の正解と見なさない。位置除外の有効性は独立参照がないため、まだ数値評価していない。

reference_pilotには、10区間から元の31指定時点の一つを固定seedで無作為抽出した10枚と、既知警告4例を用意。1枚が重複するため13枚・各4選手の計52欄。006の無作為画像は追跡初期化前で、予測がないからと画像を差し替えていない。PNGは元動画フレームと画素単位で一致する。画面には推定枠、モデル点、元のF1/F2、サンプルの種類を表示しない。

靴の位置が分かる選手について左右の靴の中央x・低い方の靴底yを目安に代表点を置く。跳躍中でも靴が見えればよく、接地瞬間の精密判定は不要。隠れて置けない／選手が見当たらない／判断できない場合はその状態を保存する。参照点は粗い画像位置の別参照で、測量的な接地点の正解ではない。参照を作る本人が以前の開発画像を見ている可能性は残る。

画面はHTMLファイルを直接開かず、起動したHTTPサーバーのリンクから使う。13枚の画像配信を読み取り検証し、ユーザーから表示できたとの確認を得た。前回はfile://タブへの操作がブラウザーのURLポリシーで拒否され、今回はその操作を再試行していない。ローカルサーバーをブラウザーから接続できる環境で起動し、HTTPリンクを案内した。

保存・再開・入力取り消しなど11項目は、実注釈と分離したautomated_qa_fixtureで検証した。実際の参照点をエージェントが付けるテストは行っていない。JavaScriptの構文と自動保存失敗時の未保存表示も整えた。実参照はreference_pilot/human_reference_labels.jsonへ自動保存され、書き込みは原子的に行う。実参照・サーバーURL・評価結果フォルダーは、後から変わるためpublication_hashes.jsonの対象外とする。

評価用コードは同じ側の2点を順序なしで対応付け、2参照点と2予測がそろう場合だけ位置誤差・ペアの中心と間隔の誤差を計算する。不足した予測、位置不明、未入力を分母から黙って消さず、別に報告する。無作為10区間と既知警告4例は混ぜて精度を出さない。これは同一試合内の開発位置評価で、個人ID継続精度、接地位置、別試合への汎化、得点予測や作戦効果の評価ではない。参照未入力の段階では精度数値を生成しない。

サーバーを再起動する場合は、このワークスペースで既存のPythonからoutputs/ball_quality_20261103/serve_position_reference_pilot.pyを実行し、表示されたhttp://127.0.0.1のURLを開く。ポートとURLは起動ごとに変わる。入力済みの参照ファイルは残り再開できる。

元追跡・旧入力・確認済み11得点ラベルを保持。固定30群の入力作成19件・未完了11件、019未解決の扱いは変更していない。参照がそろってから、位置の誤差と残せる観測数の両方を比較する。
''',encoding='utf-8')
    archive=OUT/'executed_sources';archive.mkdir()
    for name in ['build_identity_position_bundle.py','serve_position_reference_pilot.py','evaluate_position_reference_pilot.py','test_position_reference_http.py','verify_position_reference_live.py','finalize_identity_position_bundle.py']:shutil.copy2(ROOT/name,archive/name)
    shutil.copy2(PILOT/'index.html',archive/'reference_ui.html')
    day=BASE/'README.md';content=day.read_text(encoding='utf-8');marker='人物候補と配置位置を分けた共通出力';assert marker not in content;day.write_text(content+'\n\n'+marker+'を作成。28156枠の元IDを保持し、位置維持／身体警告位置のみ欠測の2条件を比較可能にした。固定620特徴行・7580値を確認。別参照を集めるため無作為10区間＋既知警告4例の13枚を元画素で抽出し、モデル結果を表示しない確認画面を用意。HTTP接続・保存再開の分離テストを通し、ユーザーから表示確認あり。参照点の入力・精度評価は未完了。current22・11ラベル・19/30保持。詳細identity_position_bundle/README.md。\n',encoding='utf-8')
    def immutable(n):
        rel=n.relative_to(OUT)
        if n.name in ['publication_hashes.json','human_reference_labels.json','human_reference_labels.tmp','server_runtime.json','server.log','server_error.log']:return False
        if str(rel).replace('\\','/').startswith('reference_pilot/evaluation/'):return False
        return True
    dump(OUT/'publication_hashes.json',{str(n.relative_to(OUT)):sha(n) for n in sorted(OUT.rglob('*')) if n.is_file() and immutable(n)});print(report)

if __name__=='__main__':main()
