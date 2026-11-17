"""Summarize the frozen comparison, without selecting further settings."""
import json
import csv
import math
from compare_totnet import OUT,read


def main():
    result=json.loads((OUT/'test_results.json').read_text())
    selection=json.loads((OUT/'selection.json').read_text())
    fallback=json.loads((OUT/'fallback_test_results.json').read_text())
    b=result['baseline'];t=result['totnet'];f=fallback['fallback'];paired=result['paired_visible']
    criterion=result['provisional_adoption_criterion_met']
    failures=[]
    for r in read(OUT/'test_frame_comparison.csv'):
        if r['truth_visibility']!='visible':continue
        statuses={}
        for name in ('baseline','totnet'):
            if int(r[f'{name}_visible'])==0:statuses[name]='missing'
            else:
                error=math.hypot(float(r[f'{name}_x_px'])-float(r['truth_x_px']),float(r[f'{name}_y_px'])-float(r['truth_y_px']))
                statuses[name]='correct_within16px' if error<=16 else 'wrong_location_over16px'
        if all(s!='correct_within16px' for s in statuses.values()):
            failures.append(dict(candidate_id=r['candidate_id'],local_frame=r['local_frame'],
                source_frame=r['source_frame'],source_timestamp_sec=r['source_timestamp_sec'],
                baseline_status=statuses['baseline'],totnet_status=statuses['totnet'],
                existing_truth_x_px=r['truth_x_px'],existing_truth_y_px=r['truth_y_px'],
                review_status='unreviewed_existing_label_not_independent_truth'))
    assert len(failures)==paired['neither_correct_visible']==27
    with (OUT/'joint_failure_review_candidates.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(failures[0]));writer.writeheader();writer.writerows(failures)
    decision=dict(model='TOTNet tennis pretrained',standalone_criterion_met=criterion,
        fallback_criterion_met=fallback['development_candidate_criterion_met'],active_baseline_changed=False,
        reason='No automatic adoption from sampled-frame evaluation; retain current aligned baseline.',
        evidence=['protocol.json','selection.json','test_results.json','fallback_selection.json','fallback_test_results.json'],
        review_scope='300 labeled samples, no continuous-track validation or independent external test')
    (OUT/'decision.json').write_text(json.dumps(decision,indent=2),encoding='utf-8')
    def value(m,key):return m['counts'].get(key,0)
    def row(label,fn):return '| '+label+' | '+' | '.join(str(fn(m)) for m in (b,t,f))+' |'
    lines=[
        '# TOTNetの比較実験（2026-11-02）','',
        '現行の時刻整合済みMS-TrackNetV3を保持する。TOTNetの単独置換・未検出時の併用を同一標本で比較した。',
        '出力数と位置正解数を区別し、今回の標本評価だけで連続追跡の品質を確定しない。','',
        '## テスト結果','',
        '| 指標 | 現行版 | TOTNet単独 | 未検出時だけ補う |',
        '|---|---:|---:|---:|',
        row('16px以内の位置正解数 / 可視注釈128件',lambda m:f"{value(m,'localized_tp_16px')} / 128"),
        row('位置再現率',lambda m:f"{100*m['localization_recall_16px']:.1f}%"),
        row('位置正解数 / 座標出力数',lambda m:f"{value(m,'localized_tp_16px')} / {value(m,'pred_visible')}"),
        row('双方が可視とした標本での誤位置数',lambda m:value(m,'wrong_localization')),
        row('非可視注釈に対する座標出力数',lambda m:value(m,'presence_fp')),
        row('可視注釈での未検出数',lambda m:value(m,'presence_fn')),'',
        f"可視標本の内訳：両方正解{paired['both_correct']}、現行版だけ正解{paired['only_baseline_correct']}、TOTNetだけ正解{paired['only_totnet_correct']}、両方不正解{paired['neither_correct_visible']}。",'',
        '## 条件と採否','',
        '- 001–010の150標本で条件を選択し、011–020の150標本をテストした。元画像は1920×1080、位置許容誤差は16px。',
        '- 学習コードに合わせてBGR・Lanczos4縮小512×288・指定平均/標準偏差で正規化。5連続フレームの末尾が対象時刻。',
        '- 公式テニス重みを厳密に読み込み、float32・評価モードで推論。ソースと重みのハッシュを保存した。',
        f"- 単独版のピーク下限は検証用標本だけで{result['selected_threshold']}に固定した。原点(0,0)は未検出として扱う。ピーク値は可視確率ではない。",
        f"- 検証用の位置正解数は現行版80/121、TOTNet {value(selection['selected']['metrics'],'localized_tp_16px')}/121。未検出時だけ補う案もテスト推論前に条件を固定した。",
        f"- 併用案の下限は{fallback['selected_threshold']}。1.0は今回の候補では全て不採用となり、現行版の出力をそのまま保持する設定を意味する。",
        f"- 事前のテスト採用条件（位置正解数増加・誤出力数増加なし）：単独版={criterion}、併用案={fallback['development_candidate_criterion_met']}。",
        '- テストを見た条件の変更・再選択は行わない。既存の採用版ファイルは変更していない。','',
        '## 比較コードの修正記録','',
        '初回の比較コードはピーク値だけで可視判定し、TOTNetが出す原点の未検出座標を誤出力として数えていた。',
        '検証用結果を確認した段階で公開コードの座標規約に合わせて修正し、テスト推論前に再選択した。',
        '修正前の条件と結果は initial_peak_only_decoder/ に保存した。推論そのものは再実行せず、元の生出力を使った。',
        '検証時の推論記録は修正前プロトコル、テスト時は修正後プロトコルを参照する。入力・重み・座標推定は同一で、変更は原点の解釈だけである。','',
        '## 研究上の限界','',
        '- 同一試合内の既使用テストであり、新しい独立外部テストではない。別試合での汎化性能を主張できない。',
        '- 既存注釈の作成者・独立性は未確認。可視・非可視の定義を含め、卒論の最終精度評価には独立した正解データが必要。',
        '- 各区間15標本の評価であり、連続映像の途切れ、再取得の遅れ、打球・バウンド周辺の全フレーム精度を測っていない。',
        '- BGRの入力条件での移植比較であり、TOTNet全般の優劣やソフトテニス向け再学習後の性能を示すものではない。','',
        '## 記録','',
        '- protocol.json / selection.json / test_results.json：単独比較の条件と結果。',
        '- fallback_protocol.json / fallback_selection.json / fallback_test_results.json：併用案の条件と結果。',
        '- validation_raw.csv / test_raw.csv：注釈の座標を参照せず推論した生座標・ピーク。',
        '- test_frame_comparison.csv：標本ごとの現行版・TOTNet・既存注釈の照合表。',
        '- joint_failure_review_candidates.csv：既存可視注釈に対し両方式が失敗した27標本の連続映像確認候補。未確認の原因分類や正解ラベルは書き込まない。',
        '- fallback_test_sample_predictions.csv：注釈対象時刻だけの実験出力。連続軌跡ファイルではない。',
        '- validation_001_all_samples.png：最初の検証区間の全15標本。256×160の等倍切り出しで縮尺の変形なし。',
        '- ../compare_totnet.py、../compare_totnet_fallback.py、../totnet_diagnostics.py：実行コード。','',
        '27標本の前後フレーム点検は joint_failure_review/README.md を参照。',
        '元の注釈と評価値は保持し、原因候補の診断を独立した正解注釈とは扱わない。','']
    (OUT/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    print(OUT/'README.md')


if __name__=='__main__':main()
