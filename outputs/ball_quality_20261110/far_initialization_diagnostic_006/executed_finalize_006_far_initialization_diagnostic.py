"""Replay the inherited initialization and publish its exact blocking reasons."""
import json
import shutil
from pathlib import Path
from collections import Counter
import numpy as np
from run_fixed_cohort_player_case_v3 import ROOT, BASE, read, write, sha, dump
from run_near_prefix_initialization_trial import load
from diagnose_initial_pair_appearance import cache
from diagnose_006_far_initialization import OUT
from court_initialization_guard import eligible, old
from pair_initialization_guard import pair_initialization_class


def trace():
    p=load(OUT/'protocol.json');e=next(e for e in p['cases'] if e['tag']=='006')
    for n,h in p['protected_input_hashes'].items():assert sha(Path(n))==h
    protected={str(n):sha(n) for n in [Path(__file__),ROOT/'pair_initialization_guard.py',Path(old.__file__),Path(old.selector.__file__)]}
    assert not (OUT/'initialization_trace_protocol.json').exists()
    dump(OUT/'initialization_trace_protocol.json',dict(method='Replay only the inherited pair-ranked all-four initialize() on identical cached rows/features from context start until first initialization. Retain original8-frame/overlap conditions; no downstream tracking replay or modified policy.',protected_input_hashes=protected))
    H=np.array(load(Path(p['calibration']))['image_to_court_homography'])
    raw,array,offset=cache(e);tr=pair_initialization_class(old.TemporalTracker,H)();records=[]
    for i,f in enumerate(range(e['first'],e['last']+1)):
        rows=raw[offset[i]:offset[i+1]];features=[None if np.isnan(v).all() else v for v in array[offset[i]:offset[i+1]]]
        allowed=[r for r in rows if eligible(r,H)]
        near=[r for r in allowed if float(r['bottom_center_y'])/2160>=.61 and old.a.height(r)/2160>=.08]
        # Exact selector diagnostics here rely on the observed absence of near
        # ambiguity; the actual replay below still uses the complete pair rank.
        assert len(near)==2, 'Use the full ranked-selector instrumentation if the near set is ambiguous.'
        proposed={};maximum=None;reason=''
        try:
            proposed=old.selector.seed_by_court_geometry(allowed,f,3840,2160)
            selected=[next(r for r in rows if int(r['track_id'])==tid) for tid in proposed.values()]
            maximum=max(old.overlap_fraction(a,b) for j,a in enumerate(selected) for b in selected[j+1:])
            reason='selected_box_overlap' if maximum>old.PARAMETERS['occlusion_intersection_fraction'] else 'confirming_past8'
        except RuntimeError:reason='fewer_than_two_geometry_candidates_on_a_side'
        assignments=tr.initialize(f,rows,features)
        record=dict(source_frame=f,proposed_ids_json=json.dumps(proposed),
            maximum_selected_overlap=maximum if maximum is not None else '',
            blocking_stage='initialized' if assignments else reason,
            consecutive_confirmation_frames=len(tr.initial_window),emitted_seed_boxes=len(assignments))
        records.append(record)
        if assignments:break
    event=tr.events[0];baseline=load(Path(e['temporal_folder'])/'events.json')[0]
    assert event==baseline
    write(OUT/'inherited_initialization_trace.csv',records)
    dump(OUT/'initialization_trace_verification.json',dict(status='passed',initialization_event_exact=True,
        event=event,context_frames_replayed=len(records),full_tracking_replayed=False,
        numeric_thresholds_changed=False))
    assert all(sha(Path(n))==h for n,h in protected.items())
    print('Exact initialization replay:',event['frame'],event['seed_ids'],flush=True)


def finalize():
    p=load(OUT/'protocol.json');result=load(OUT/'result.json')
    tracep=load(OUT/'initialization_trace_protocol.json')
    protected=dict(p['protected_input_hashes']);protected.update(tracep['protected_input_hashes'])
    assert all(sha(Path(n))==h for n,h in protected.items())
    timeline=read(OUT/'frame_gate_audit.csv');fixed=[r for r in timeline if r['fixed_requested_time']=='1']
    assert len(fixed)==310 and len({(r['tag'],r['source_frame']) for r in fixed})==310
    six=[r for r in fixed if r['tag']=='006'];pre=[r for r in six if int(r['source_frame'])<10575]
    assert len(pre)==20
    counts=dict(Counter(int(r['far_seed_candidates']) for r in pre));assert counts=={1:13,2:7}
    trace={int(r['source_frame']):r for r in read(OUT/'inherited_initialization_trace.csv')}
    stages=dict(Counter(trace[int(r['source_frame'])]['blocking_stage'] for r in pre))
    assert stages=={'fewer_than_two_geometry_candidates_on_a_side':13,'selected_box_overlap':3,'confirming_past8':4}
    dump(OUT/'blocking_reason_summary.json',dict(fixed_requested_times_006=31,
        before_common_initialization_times=20,preinit_seed_candidate_count_histogram=counts,
        preinit_blocking_stage_histogram=stages,
        geometry_shortage_source_frames=[int(r['source_frame']) for r in pre if int(r['far_seed_candidates'])==1],
        selected_overlap_source_frames=[int(r['source_frame']) for r in pre if trace[int(r['source_frame'])]['blocking_stage']=='selected_box_overlap'],
        confirming_source_frames=[int(r['source_frame']) for r in pre if trace[int(r['source_frame'])]['blocking_stage']=='confirming_past8'],
        first_consecutive_accepted_common_seed_frame=10568,confirmation_output_frame=10575,
        candidate_counts_not_true_player_recall=True))
    review=dict(reviewer='AI assistant, not independent human GT',
        inspected_frames=[10430,10465,10536,10546,10574,10575,10596],
        observations=[
            'At10465, far-crop raw1000103 appears aligned to the orange far player behind the white near player; very high near-box overlap. Do not label it a false person solely because it was rejected.',
            'At10536/10546, the orange far player is partly hidden by the white near player; the small extra native detection near the head/arms does not support observed shoe coordinates. Nearby blue official contributes a separate-person ambiguity.',
            'At10574/10575, raw59 surrounds a much more exposed orange far player; near overlap has decreased and the inherited confirmation is finishing.',
            'Broader ROI also contains blue officials and the near player. Additional candidates do not establish two far-player locations.'],
        continuous_identity_or_visibility_ground_truth_added=False,
        P011_unplaceable_label_interpreted_as_missing_reference_not_automatic_occlusion_truth=True)
    dump(OUT/'AI_visual_review.json',review)
    dump(OUT/'processing_decision.json',dict(
        conclusion='Mixed limitation:13 fixed preinit times lack a second native candidate passing the current far-body seed definition;3 are rejected by selected-box overlap;4 await past8 confirmation. Partial visibility and detector-box quality must be separated from initialization latency.',
        not_adopted='No relaxed court/y/height/overlap gates, no forced early far pair, no hidden-foot/pose coordinate fill, no copying future raw59 backward.',
        retained='Near-prefix camera-conditioned development comparator retained separately. All existing tracking/native coordinates/labels/inputs unchanged.',
        next='Test whether earlier valid far-body evidence can initialize identity separately from current-frame position availability. Treat partially visible body evidence as an ID candidate with uncertain position; first compare all ten frozen contexts and retain the same output-time/coordinate rules. Do not call future/offline initialization online.',
        canonical_pipeline_replaced=False,independent_accuracy_claim=False))
    (OUT/'README.md').write_text('''# 006の奥側初期化遅れ：検出と待ち条件の切り分け

指定入力F10536〜10596の31時点を固定した。元の検出候補・外観キャッシュ・校正・初期化処理を読み、他9区間も含めて全310指定時点を調べた。新しい推論や人物IDの割当て、座標補間、閾値変更は行っていない。

## 未出力20時点の内訳

| 従来初期化が止まる段階 | 指定時点数 | ソースフレーム |
|---|---:|---|
| 奥の初期化条件を通る候補が1枠 | 13 | F10536〜10560、2フレーム間隔 |
| 2枠あるが、選択枠と手前選手枠の重なりが大きい | 3 | F10562・10564・10566 |
| 重なり条件も通るが過去8連続フレームの確認待ち | 4 | F10568・10570・10572・10574 |

F10568から8ソースフレーム連続で条件を満たし、F10575に出力される。従来の初期化のみを同じキャッシュで再実行し、開始フレーム・確認窓・4つのseed IDを含む初期化イベントが完全一致した。下流の全追跡を再実行したという意味ではない。

13時点の「1枠」は従来の足元y・x・高さ・コート領域の条件を通る数であり、真の人物検出再現率ではない。小さい人物枠や頭・腕付近の候補は存在する。新しい検出器が回復できないという証明でもない。

## 画像で確認できたことと限界

AIが事前指定の7画像を点検した。F10536/F10546では橙服の奥選手が白服の手前選手と重なっている。小さい追加枠は頭・腕・隣接する青服の係員の付近にあり、靴位置の実観測としては使えない。F10574付近には奥選手の露出が増え、raw59の大きな人物枠が出る。これはAIの画像所見であり、独立した本人同定・可視性の正解は追加していない。

F10465には、4人を要求せず奥2候補だけ8フレーム確認すれば初期化できそうなraw4/raw1000103がある。raw1000103は橙服の実選手に対応して見えるため、重なりで落ちることだけを理由に誤検出とは扱わない。一方、手前枠との重なりが大きい。存在・本人同定の候補と、足元座標の信頼性は分ける必要がある。

以前の人手位置参照P011/F10546では奥の片方がunplaceableである。このラベルは参照点が置けないという意味で、検出器の失敗や完全遮蔽の正解を自動的に意味しない。既知参照を見た同一試合内の診断であり、未知試合への性能評価でもない。

## 判断

領域や高さの条件を緩めると、奥の選手候補に加えて審判・係員・手前選手も入る。4枠を出すために緩める変更、未来のraw59を過去へコピーする処理、見えない靴位置の補間は採用しない。

次に試す価値があるのは「早い時点の身体情報で人物候補を初期化し、足元が測れないフレームは位置を欠測にする」比較である。既存F1/F2との対応・再取得を全10区間で比較し、未来情報を使う場合はオフライン方式と明記する。これは次の実験案であり、今回実装して改善を確認したものではない。

current22、得点側と区間ラベル、指定入力時点、公開済み位置参照評価、従来の選手座標は変更していない。奥側の追加出力は0。4人がそろう時点数も11/31のままである。

## 成果物

- `006_far_candidate_audit.mp4`：0.5倍速。上は奥側全体、下左は重なり箇所の原画像、下中央は同じ拡大範囲の検出枠。すべて縦横比を保存。
- `candidate_gate_audit.csv`：検出候補ごとの従来条件と除外理由。
- `frame_gate_audit.csv`：全10区間×31時点と006の連続区間。候補数は正解率ではない。
- `inherited_initialization_trace.csv`：初期化待ちを再現した各フレームの理由。
- `blocking_reason_summary.json`：指定20時点の13・3・4の内訳。

動画の緑は初期化の幾何条件を通る枠、橙は広い奥側領域に入る他の検出枠。色は正誤を表さない。
''',encoding='utf-8')
    assert load(OUT/'media_verification.json')['status']=='passed'
    dump(OUT/'final_verification.json',dict(status='passed',fixed_contexts=10,fixed_times=310,
        native_candidate_rows=len(read(OUT/'candidate_gate_audit.csv')),
        initialization_event_exact_replay=True,preinit_fixed_times=20,
        blocking_counts={'candidate_shortage_under_current_definition':13,'selected_overlap':3,'past8_confirmation':4},
        protected_inputs_unchanged=True,additional_far_ID_or_position_outputs=0,
        media_decode_verified=True,independent_accuracy_established=False))
    for name in ['diagnose_006_far_initialization.py','render_006_far_initialization_audit.py','finalize_006_far_initialization_diagnostic.py']:
        shutil.copy2(ROOT/name,OUT/('executed_'+name))
    hashes={str(n.relative_to(OUT)):sha(n) for n in sorted(OUT.rglob('*')) if n.is_file() and n.name!='publication_hashes.json'}
    dump(OUT/'publication_hashes.json',hashes)
    assert all(sha(OUT/n)==h for n,h in hashes.items())
    assert all(sha(Path(n))==h for n,h in protected.items())
    print('Published diagnosis. Existing coordinates, labels and fixed windows unchanged.',flush=True)


if __name__=='__main__':
    import sys
    {'trace':trace,'finalize':finalize}[sys.argv[1]]()
