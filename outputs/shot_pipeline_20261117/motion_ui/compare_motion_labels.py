"""motion_review.csv の確認結果と本人回答との一致を集計（標準ライブラリのみ）。"""
from __future__ import annotations
import argparse
import csv
import io
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
SOURCES = ('v1', 'v2', 'claude')
ASPECTS = ('motion', 'direction3', 'complexity')
JUDGES = {'正しい', '違う', '分からない'}
DIR8 = ('straight_front','right_front','right_lateral','right_back','straight_back','left_back','left_lateral','left_front')
DIR3 = {'front','lateral','back'}
FIELDS = ['id','point','contact_frame','corrected_contact_frame','contact_judgment','player_judgment',
          *[f'{s}_{a}_judgment' for s in SOURCES for a in ASPECTS],
          'motion','single_compound','screen_direction','player_direction8','player_direction3','feint','answer_origin',
          'confidence','note','seconds','annotator','saved_at']


def load_sources():
    data = json.loads((HERE / 'data.json').read_text(encoding='utf-8'))
    items = {i['id']: i for i in data['items']}
    return items, {s: {k: i['predictions'][s] for k, i in items.items() if s in i['predictions']} for s in SOURCES}


def load_labels(path, items):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        missing = set(FIELDS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'必要な列がありません: {sorted(missing)}')
        rows = list(reader)
    if not rows:
        raise ValueError('CSVにデータ行がありません')
    result = {}
    for line, r in enumerate(rows, 2):
        ident = r['id']
        if ident not in items:
            raise ValueError(f'{line}行: 対象外ID {ident}')
        if ident in result:
            raise ValueError(f'{line}行: ID重複 {ident}')
        i = items[ident]
        if int(r['point']) != i['point'] or int(r['contact_frame']) != i['contact_frame']:
            raise ValueError(f'{line}行: メタデータ不一致')
        for k in ('contact_judgment', 'player_judgment', *[f'{s}_{a}_judgment' for s in SOURCES for a in ASPECTS]):
            allowed = {'','正しい','違う'} if k == 'player_judgment' else JUDGES | {''}
            if r[k] not in allowed:
                raise ValueError(f'{line}行: 不正な判定 {k}')
        if 'claude' not in i['predictions'] and any(r[f'claude_{a}_judgment'] for a in ASPECTS):
            raise ValueError(f'{line}行: Claude参照のないIDに判定があります')
        if r['corrected_contact_frame']:
            frame = int(r['corrected_contact_frame'])
            if r['contact_judgment'] != '違う' or not i['start_frame'] <= frame <= i['stop_frame']:
                raise ValueError(f'{line}行: 不正な修正フレーム')
        if r['motion'] not in {'','motion','none','unknown'} or r['single_compound'] not in {'','single','compound'}:
            raise ValueError(f'{line}行: 不正な本人回答')
        if r['player_direction3'] not in DIR3 | {''} or r['player_direction8'] not in set(DIR8) | {''}:
            raise ValueError(f'{line}行: 不正な方向')
        if r['feint'] not in {'','yes','no','unknown'} or r['confidence'] not in {'','high','medium','low'}:
            raise ValueError(f'{line}行: 不正なフェイント／自信')
        if r['motion'] in {'none','unknown'} and (r['single_compound'] or r['player_direction8'] or r['player_direction3']):
            raise ValueError(f'{line}行: 動きなし／不明に方向・単複があります')
        seconds = float(r['seconds'])
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError(f'{line}行: 不正な作業時間')
        result[ident] = r
    return result


def review_accuracy(labels, predictions, source):
    out = {}
    for a in ASPECTS:
        counts = {j: 0 for j in (*sorted(JUDGES), '未入力', '対象外')}
        for ident, r in labels.items():
            if ident not in predictions:
                continue
            counts[r[f'{source}_{a}_judgment'] or '未入力'] += 1
        out[a] = counts
    return out


def agreement(labels, predictions, items):
    metrics = {a: [0, 0] for a in ASPECTS}
    counts = dict(shared=0, player_rejected=0, player_unchecked=0, target_mismatch=0,
                  human_unknown=0, prediction_missing=0, corrected_contact=0, contact_unchecked=0)
    for ident, h in labels.items():
        if ident not in predictions:
            continue
        counts['shared'] += 1
        p, i = predictions[ident], items[ident]
        if h['contact_judgment'] == '違う':
            counts['corrected_contact'] += 1
        elif h['contact_judgment'] != '正しい':
            counts['contact_unchecked'] += 1
        if h['player_judgment'] != '正しい':
            counts['player_rejected' if h['player_judgment'] == '違う' else 'player_unchecked'] += 1
            continue
        if (not i['front_slot'] or (p['front_slot'], p['front_side']) != (i['front_slot'], i['front_side'])
                or 'wrong_player' in p.get('claude_issue', '')):
            counts['target_mismatch'] += 1
            continue
        if h['motion'] not in {'none','motion'}:
            counts['human_unknown'] += 1
            continue
        # 自動の棄却予測は不一致。Claudeの未記入は予測欠損として除外する。
        if not p['motion']:
            counts['prediction_missing'] += 1
            continue
        metrics['motion'][1] += 1
        metrics['motion'][0] += int(p['motion'] == h['motion'])
        if h['motion'] == 'motion':
            for a, field in (('direction3','player_direction3'), ('complexity','single_compound')):
                if not h[field]:
                    continue
                if not p[a] and 'claude_motion' in p:
                    continue
                metrics[a][1] += 1
                metrics[a][0] += int(h[field] == p[a])
    return metrics, counts


def rate(pair):
    m, n = pair
    return f'{100*m/n:.1f}% ({m}/{n})' if n else '— (0/0)'


def contact_stats(labels):
    shifted = [r for r in labels.values() if r['contact_judgment'] == '違う']
    deltas = [(r['id'], int(r['corrected_contact_frame'])-int(r['contact_frame']))
              for r in shifted if r['corrected_contact_frame']]
    return dict(shifted=len(shifted), corrected=len(deltas), missing=len(shifted)-len(deltas), deltas=deltas)


def report(labels, items, sources, stream=None):
    out = stream or sys.stdout
    print(f'確認式レビュー集計：入力 {len(labels)}行 / 本人回答未確定 {sum(r["motion"] not in {"none","motion"} for r in labels.values())}件', file=out)
    print('本人による項目別の正解率（正しい / (正しい+違う)）。分からない・未入力・対象外は分母外。', file=out)
    print('| 判定 | 項目 | 正解率 | 分からない | 未入力 | 対象外 |', file=out)
    print('|---|---|---:|---:|---:|---:|', file=out)
    results = {}
    for s, predictions in sources.items():
        reviews = review_accuracy(labels, predictions, s)
        metrics, counts = agreement(labels, predictions, items)
        results[s] = dict(review=reviews, agreement=metrics, audit=counts)
        for a, c in reviews.items():
            print(f'| {s} | {a} | {rate([c["正しい"],c["正しい"]+c["違う"]])} | {c["分からない"]} | {c["未入力"]} | {c["対象外"]} |', file=out)
    print('\n本人の答えとの一致（対象選手が同じで、本人が枠を正しいと確認した行）', file=out)
    print('| 判定 | 動きの有無 | 3方向 | 単・複 |', file=out)
    print('|---|---:|---:|---:|', file=out)
    for s, r in results.items():
        print('| '+s+' | '+' | '.join(rate(r['agreement'][a]) for a in ASPECTS)+' |', file=out)
    for s, r in results.items():
        print('監査 '+s+': '+json.dumps(r['audit'], ensure_ascii=False), file=out)
    stats = contact_stats(labels)
    print(f'\n打球の瞬間：ずれている {stats["shifted"]}件 / 修正済み {stats["corrected"]}件 / 修正フレーム未入力 {stats["missing"]}件', file=out)
    if stats['deltas']:
        ds = [abs(d) for _, d in stats['deltas']]
        print(f'ずれ（修正−元ラベル）：絶対値の平均 {statistics.mean(ds):.2f}、中央値 {statistics.median(ds):.2f}、最大 {max(ds)} フレーム', file=out)
        for ident, delta in stats['deltas']:
            print(f'  {ident}: {delta:+d} フレーム', file=out)
    print('\n一致の方向・単複は本人が動きありで該当回答がある行。予測がなし／棄却なら不一致、Claude未記入は除外。', file=out)
    print('打球修正行も元の分類との比較に含む（分類の再計算なし）。監査欄に件数を示す。', file=out)
    print('注意：予測を提示した確認作業であり、独立した盲検の正解率ではない。採用した予測と本人回答の一致には循環性がある。', file=out)
    print('30件はClaude既確認20件＋追加10件。開発に使用済みの例を含み、未知試合への汎化精度にはできない。', file=out)
    return results


class ComparisonTests(unittest.TestCase):
    def test_previous_contact_is_front_side(self):
        items, _ = load_sources()
        self.assertEqual(items['p08_B_18']['previous_contact_frame'], 25458)
        self.assertEqual(items['p08_B_18']['start_frame'], 25446)
        for item in items.values():
            with (HERE.parent/f'labels/point_{item["point"]:02d}.csv').open(encoding='utf-8-sig', newline='') as f:
                shots = list(csv.DictReader(f))
            earlier = [int(s['contact_frame']) for s in shots if s['rally'] == item['rally']
                       and s['hitter_side'] == item['front_side'] and int(s['contact_frame']) < item['contact_frame']]
            self.assertEqual(item['previous_contact_frame'], max(earlier) if earlier else None)

    def test_roundtrip_real_sources(self):
        items, sources = load_sources()
        self.assertEqual(len(items), 30)
        self.assertEqual(len(sources['claude']), 20)
        # 実際のJS exporterで30件のダミーCSVを作り、Python loaderとCLIまで通す。
        proc = subprocess.run(['node', str(HERE/'test_motion_ui.js'), '--csv'], capture_output=True, text=True, encoding='utf-8', check=True)
        with tempfile.TemporaryDirectory(prefix='motion-review-') as tmp:
            path = Path(tmp)/'motion_review.csv'
            path.write_bytes(proc.stdout.encode('utf-8'))
            labels = load_labels(path, items)
            self.assertEqual(len(labels), 30)
            self.assertIn('改行\n', next(iter(labels.values()))['note'])
            result = report(labels, items, sources, io.StringIO())
            self.assertEqual(result['claude']['audit']['shared'], 20)
            self.assertEqual(contact_stats(labels)['shifted'], 10)
            self.assertEqual(contact_stats(labels)['deltas'][0][1], -3)
            cli = subprocess.run([sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),str(path)], capture_output=True,text=True,encoding='utf-8')
            self.assertEqual(cli.returncode, 0, cli.stderr)
            self.assertIn('循環性', cli.stdout)
            with path.open('a', encoding='utf-8', newline='') as f:
                csv.DictWriter(f,FIELDS).writerow(next(iter(labels.values())))
            with self.assertRaisesRegex(ValueError,'ID重複'):
                load_labels(path,items)
        self.assertFalse(path.exists())

    def test_denominators_and_target_guard(self):
        base = dict(front_slot='far_front',front_side='F')
        items = {str(i):base.copy() for i in range(6)}
        labels = {str(i):dict(player_judgment='正しい',contact_judgment='正しい',motion='motion',single_compound='single',player_direction3='front',
                   **{f'v1_{a}_judgment':'正しい' for a in ASPECTS}) for i in range(6)}
        pred = {str(i):dict(**base,motion='motion',complexity='single',direction3='front') for i in range(6)}
        labels['1']['v1_motion_judgment']='違う'
        labels['2']['v1_motion_judgment']='分からない'
        labels['3']['v1_motion_judgment']=''
        labels['1']['motion']='unknown'
        pred['2']['front_slot']='far_rear'
        labels['3']['player_judgment']='違う'
        pred['4'].update(motion='none',complexity='',direction3='')
        pred['5'].update(motion='unclassifiable',complexity='',direction3='')
        m,c = agreement(labels,pred,items)
        self.assertEqual(m, {a:[1,3] for a in ASPECTS})
        self.assertEqual((c['human_unknown'],c['target_mismatch'],c['player_rejected']),(1,1,1))
        review = review_accuracy(labels,pred,'v1')
        self.assertEqual(review['motion']['正しい'],3)
        self.assertEqual(review['motion']['分からない'],1)
        self.assertEqual(review['direction3']['正しい'],6)

    def test_missing_and_negative_contact_shift(self):
        r = contact_stats({'a':dict(id='a',contact_judgment='違う',contact_frame='100',corrected_contact_frame='92'),
                           'b':dict(id='b',contact_judgment='違う',contact_frame='100',corrected_contact_frame='')})
        self.assertEqual(r,dict(shifted=2,corrected=1,missing=1,deltas=[('a',-8)]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('labels', nargs='?', type=Path)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        return 0 if unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ComparisonTests)).wasSuccessful() else 1
    if not args.labels:
        parser.error('motion_review.csv のパス、または --self-test が必要です')
    try:
        items,sources = load_sources()
        report(load_labels(args.labels,items),items,sources)
    except (OSError,ValueError,KeyError) as exc:
        print(f'比較失敗: {exc}',file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
