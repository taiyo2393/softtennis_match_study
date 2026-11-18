"""Before/after preservation audit and statistics over displayed source intervals."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_court_view as b


def near_digest(table, columns):
    table = table[table.player_slot.str.startswith('near_')][columns]
    return hashlib.sha256(table.to_csv(index=False, lineterminator='\n').encode()).hexdigest()


def metrics(point, positions, segments):
    selected = pd.Series(False, index=positions.index)
    for segment in segments:
        selected |= positions.source_frame.between(segment['start_frame'], segment['end_frame'])
    positions = positions[selected]
    result = []
    for slot in b.SLOTS:
        group = positions[positions.player_slot == slot].sort_values('source_frame')
        valid = group[group.status.isin(b.VALID)]
        quantiles = (valid.court_y * 23.77).quantile([.05,.5,.95]).tolist()
        steps = []
        for segment in segments:
            rally = group[group.source_frame.between(segment['start_frame'],segment['end_frame'])]
            for a,z in zip(rally.itertuples(),list(rally.itertuples())[1:]):
                if z.source_frame-a.source_frame == 1 and a.status in b.VALID and z.status in b.VALID:
                    steps.append((b.distance((a.court_x,a.court_y),(z.court_x,z.court_y)),int(a.source_frame),int(z.source_frame)))
        largest = max((s[0] for s in steps),default=None)
        ties = [s for s in steps if s[0] == largest]
        result.append(dict(point=point,player_slot=slot,name=b.NAMES[slot],
                           valid_source_frames=len(valid),total_source_frames=len(group),
                           court_y_p05_m=quantiles[0],court_y_p50_m=quantiles[1],court_y_p95_m=quantiles[2],
                           max_step_m_per_source_frame=largest,
                           max_step_from_frame=ties[0][1] if ties else None,
                           max_step_to_frame=ties[0][2] if ties else None,
                           max_step_ties=[dict(from_frame=a,to_frame=z) for _,a,z in ties],
                           status_counts={s:int((group.status==s).sum()) for s in b.STATUSES},
                           filter_status_counts={s:int((group.filter_status==s).sum()) for s in b.STATUSES}))
    return result


def snapshot():
    out = b.OUT
    target = out/'a_height_revision_before.json'
    assert not target.exists(), 'Preserve the original baseline; do not overwrite it'
    summary = json.loads((out/'cleaning_summary.json').read_text(encoding='utf-8'))
    frozen = [out/'identity_summary.json',out/'hitter_auto.csv',out/'hitter_auto_tracking_v1.csv',
              out/'wrist_identity.csv',out/'ball_candidates.csv',out/'ball_manifest.json']
    frozen += [out/f'{prefix}_p{point:02d}.csv' for prefix in ('identity_map','identity_positions') for point in range(7,12)]
    protected = [p for dirname in ('tracking','tracking_v2','pose') for p in (b.PIPE/dirname).rglob('*') if p.is_file()]
    near = {}; before_metrics = []
    for point in range(7,12):
        table = pd.read_csv(out/f'cleaned_player_positions_p{point:02d}.csv')
        near[str(point)] = dict(columns=list(table.columns),sha256=near_digest(table,list(table.columns)))
        before_metrics += metrics(point,table,summary['points'][str(point)]['rendering']['segments'])
    ball = {str(p.relative_to(b.ROOT)):[p.stat().st_size,p.stat().st_mtime_ns]
            for directory in (b.ROOT/'outputs').glob('ball_quality_*') for p in directory.rglob('*') if p.is_file()}
    b.dump(target,dict(frozen_sha256={str(p.relative_to(b.ROOT)):b.sha(p) for p in frozen},
                       protected_sha256={str(p.relative_to(b.ROOT)):b.sha(p) for p in protected},
                       ball_quality_metadata=ball,near=near,before_metrics=before_metrics))
    print(f'Baseline saved: {len(protected)} protected inputs, {len(frozen)} B/C files, {len(ball)} ball_quality files',flush=True)


def verify():
    out = b.OUT
    before = json.loads((out/'a_height_revision_before.json').read_text(encoding='utf-8'))
    for section in ('frozen_sha256','protected_sha256'):
        for name,digest in before[section].items():
            assert b.sha(b.ROOT/name)==digest, f'Changed {name}'
    ball = {str(p.relative_to(b.ROOT)):[p.stat().st_size,p.stat().st_mtime_ns]
            for directory in (b.ROOT/'outputs').glob('ball_quality_*') for p in directory.rglob('*') if p.is_file()}
    assert ball == before['ball_quality_metadata']
    summary = json.loads((out/'cleaning_summary.json').read_text(encoding='utf-8'))
    after = []
    for point in range(7,12):
        table = pd.read_csv(out/f'cleaned_player_positions_p{point:02d}.csv')
        reference = before['near'][str(point)]
        assert near_digest(table,reference['columns']) == reference['sha256'], f'Near player changed p{point}'
        after += metrics(point,table,summary['points'][str(point)]['rendering']['segments'])
    report = dict(scope='All source frames within the rendered rally intervals, including frames omitted by the 2F video sampling; no spatial clipping. Only valid final statuses contribute to y quantiles. Steps use Euclidean court distance over consecutive source frames within one rally; missing frames are not bridged.',
                  frozen_B_C_files_sha256_unchanged=len(before['frozen_sha256']),
                  protected_files_sha256_unchanged=len(before['protected_sha256']),
                  ball_quality_files_metadata_unchanged=len(ball),near_player_outputs_unchanged=True,
                  before_metrics=before['before_metrics'],after_metrics=after)
    b.dump(out/'a_height_revision_report.json',report)
    flat = [{k:v for k,v in r.items() if k not in ('status_counts','filter_status_counts','max_step_ties')} for r in after]
    pd.DataFrame(flat).to_csv(out/'displayed_position_metrics.csv',index=False,encoding='utf-8-sig',lineterminator='\n')
    counts = [dict(point=r['point'],player_slot=r['player_slot'],scope='displayed_source_intervals',**r['status_counts']) for r in after if r['player_slot'].startswith('far_')]
    counts += [dict(point=p,player_slot=slot,scope='full_export',**summary['points'][str(p)]['status_counts_by_player'][slot]) for p in range(7,12) for slot in b.SLOTS[2:]]
    pd.DataFrame(counts).to_csv(out/'far_status_counts.csv',index=False,encoding='utf-8-sig',lineterminator='\n')
    filter_counts = [dict(point=r['point'],player_slot=r['player_slot'],scope='displayed_source_intervals',**r['filter_status_counts']) for r in after if r['player_slot'].startswith('far_')]
    filter_counts += [dict(point=p,player_slot=slot,scope='full_export',**summary['points'][str(p)]['filter_status_counts_by_player'][slot]) for p in range(7,12) for slot in b.SLOTS[2:]]
    pd.DataFrame(filter_counts).to_csv(out/'far_filter_status_counts.csv',index=False,encoding='utf-8-sig',lineterminator='\n')
    summary['a_height_revision_report']='a_height_revision_report.json'
    summary['displayed_position_metrics']=after
    summary['preservation_audit']={k:v for k,v in report.items() if k not in ('scope','before_metrics','after_metrics')}
    b.dump(out/'cleaning_summary.json',summary)
    b.write_readme(summary,summary['hitter_decisions'])
    lines=['# A：奥側の高さ規則の修正結果','',
           '5本のMP4を再生成し、全編復号・フレーム数・指定元フレームとの一致を検証した。B/Cの16ファイルは修正前とSHA-256一致。tracking/・tracking_v2/・pose/の215ファイルはSHA-256一致。ball_quality_*の14,256ファイルは一覧・サイズ・更新時刻が一致。手前側は全従来列の出力が一致。git commitは行っていない。','',
           '規則はREADME.mdとcleaning_summary.jsonに記載。奥側は人物固定後・同一ラリー内の観測枠から±180元Fの高さ80%点（30観測未満ならラリー全体）を求め、0.88未満は頭基準、0.88〜1.35は下端、1.35超は高さ棄却。重なりは記録のみ、奥側の高さリセットは無効。速度・平滑化・補間は維持した。','',
           '## 集計範囲','',
           '以下は表示区間内の全元フレーム。動画の2F間引きで省いたFも含む。空間的な切り取りなし。y百分位は最終statusが採用・補間の点のみ。移動量は同じラリーで連続する元Fの両方が有効な場合のコート平面ユークリッド距離。欠測をまたがない。N1=near_front、N2=near_rear、F1=far_rear、F2=far_front。','',
           '## 奥側の最終status件数','',
           '補間で置換された棄却は最終statusではinterpolatedとなる。補間前の棄却件数はfar_filter_status_counts.csv。全エクスポート範囲と表示区間内の両方をfar_status_counts.csvに保存。','',
           '|点|選手|observed|head_anchored|after_height_reset|rejected_box_height|rejected_speed|interpolated|missing|',
           '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in after:
        if r['player_slot'].startswith('far_'):
            c=r['status_counts']
            lines.append(f'|{r["point"]}|{r["name"]}|{c["observed"]}|{c["observed_head_anchored"]}|{c["observed_after_height_reset"]}|{c["rejected_box_height"]}|{c["rejected_speed"]}|{c["interpolated"]}|{c["missing"]}|')
    lines += ['','## 4人の位置・最大移動','',
              '|点|選手|有効F/対象F|y5% m|y50% m|y95% m|最大移動 m/元F|元Fの遷移|',
              '|---|---|---:|---:|---:|---:|---:|---|']
    for r in after:
        lines.append(f'|{r["point"]}|{r["name"]}|{r["valid_source_frames"]}/{r["total_source_frames"]}|{r["court_y_p05_m"]:.3f}|{r["court_y_p50_m"]:.3f}|{r["court_y_p95_m"]:.3f}|{r["max_step_m_per_source_frame"]:.6f}|F{r["max_step_from_frame"]}→F{r["max_step_to_frame"]}|')
    lines += ['','## 問題として挙げられた箇所','',
              '第7点far_frontの最大移動は0.894532→0.472205 m/元F。F23548は高さ122.504px、href129.722pxでobserved（重なりありでも頭基準にしない）。第11点F32999のtrack4（F2）は高さ76.716px、href101.400pxでobserved_head_anchored（重なりなしでも補正）。','',
              '第7点far_rearのy95%は29.292m、第11点far_frontは19.422m。提示された前回値は各30.4m、30.6m。今回と同じ全元F集計で修正直前を再集計すると各30.581m、30.578m。提示値との微差があるため、厳密な比較にはこの再集計値を用いる。第7点far_rearの大きな奥行きは残っており、規則適合を位置精度の検証とはみなさない。','',
              '第7点F23422のF1は頭基準を試みるが速度棄却され、最終表示は補間。第7点F23548のF2はobserved、第11点F32999のF2はobserved_head_anchored。','',
              '## 動画','']
    for p in range(7,12):
        v=summary['points'][str(p)]['rendering']
        lines.append(f'- [第{p}点](point_{p:02d}.mp4)：{v["frames"]}コマ、{v["duration_sec"]:.3f}秒、全編復号済み。')
    lines += ['','## 確認画像','']
    for p,f in [(7,23422),(7,23548),(11,32999)]:
        lines.append(f'![第{p}点 F{f}](check_abc_p{p:02d}_F{f}.jpg)')
    (out/'a_height_revision_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    pd.DataFrame(b.abc.inventory()).to_csv(out/'file_inventory.csv',index=False,encoding='utf-8-sig',lineterminator='\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('before_metrics','after_metrics')},ensure_ascii=False,indent=2))
    print(pd.DataFrame(flat).to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['snapshot','verify'])
    args=parser.parse_args()
    snapshot() if args.action=='snapshot' else verify()
