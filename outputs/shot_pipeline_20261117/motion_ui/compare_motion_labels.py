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
SOURCES = ('bbox', 'ankle')
SOURCE_NAMES = {'bbox':'箱の下端', 'ankle':'足首'}
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
        if not i['front_slot'] or (p['front_slot'], p['front_side']) != (i['front_slot'], i['front_side']):
            counts['target_mismatch'] += 1
            continue
        if h['motion'] not in {'none','motion'}:
            counts['human_unknown'] += 1
            continue
        # 自動の判定不能は不一致。本人不能は分母から除く。
        if not p['motion']:
            counts['prediction_missing'] += 1
            continue
        metrics['motion'][1] += 1
        metrics['motion'][0] += int(p['motion'] == h['motion'])
        if h['motion'] == 'motion':
            for a, field in (('direction3','player_direction3'), ('complexity','single_compound')):
                if not h[field]:
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


def composition(labels, items, metric, exclude_p8_near=False):
    """Use human answers only; missing detail never becomes a negative answer."""
    field = {'presence':'motion', 'complexity':'single_compound',
             'single_3dir':'player_direction3', 'single_8dir':'player_direction8',
             'final_3dir':'player_direction3'}[metric]
    allowed = {'presence':{'motion','none'}, 'complexity':{'single','compound'},
               'single_3dir':DIR3, 'single_8dir':set(DIR8), 'final_3dir':DIR3}[metric]
    values = []
    for ident, r in labels.items():
        if r['player_judgment'] != '正しい' or not items[ident]['front_slot']:
            continue
        if exclude_p8_near and items[ident]['exclude_p8_near']:
            continue
        if metric != 'presence' and r['motion'] != 'motion':
            continue
        if metric.startswith('single_') and r['single_compound'] != 'single':
            continue
        if r[field] in allowed:
            values.append(r[field])
    return {v: values.count(v) for v in allowed}, len(values)


def composition_report(labels, items, out):
    paper = json.loads((HERE/'data.json').read_text(encoding='utf-8'))['paper_reference']
    print('\n本人の答えの構成比と井箟ほかの値（サーブを含む）', file=out)
    print('前衛の枠を正しいと確認し、対象slotがあり、該当回答が確定した行だけを各分母に含む。', file=out)
    print('有無＝確定した有無、単複＝動きありで単複確定、単3/8方向＝単で方向確定、最終3方向＝動きありで方向確定。', file=out)
    print('本人不能・未入力は分母外。項目別に分母を明記し、元の自動判定から補完しない。', file=out)
    print('| 項目 | 本人の分類 | 本人・全件 % (n/N) | 本人・第8点手前除外 % (n/N) | 井箟ほか % |', file=out)
    print('|---|---|---:|---:|---:|', file=out)
    names = {'presence':'動きの有無', 'complexity':'単・複', 'single_3dir':'単・3方向',
             'single_8dir':'単・8方向', 'final_3dir':'最終・3方向'}
    for p in paper['rows']:
        pairs = []
        for exclude in (False, True):
            counts, n = composition(labels, items, p['metric'], exclude)
            pairs.append(rate([counts.get(p['category'], 0), n]))
        print(f'| {names[p["metric"]]} | {p["category_ja"]} | {pairs[0]} | {pairs[1]} | {p["percent"]:.1f} |', file=out)
    accepted = sum(r['answer_origin'] == 'accepted' for r in labels.values())
    unknown = sum(r['motion'] == 'unknown' for r in labels.values())
    blank = sum(not r['motion'] for r in labels.values())
    rejected = sum(r['player_judgment'] != '正しい' or not items[k]['front_slot'] for k, r in labels.items())
    print(f'監査：予測から採用 {accepted}件 / 本人判定できない {unknown}件 / 有無未入力 {blank}件 / 前衛未確認・誤選択・未選択 {rejected}件。', file=out)
    print('井箟ほかの値は v_human/net_motion/paper_comparison.csv の継承値（原論文未再照合）。単8方向は合計101.9%で、再正規化しない。', file=out)
    print('論文とは対象・区間定義・判定方法が異なるため、構成比の参考比較であり、正解率や再現性の証明ではない。', file=out)


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
            print(f'| {SOURCE_NAMES[s]} | {a} | {rate([c["正しい"],c["正しい"]+c["違う"]])} | {c["分からない"]} | {c["未入力"]} | {c["対象外"]} |', file=out)
    print('\n本人の答えとの一致（対象選手が同じで、本人が枠を正しいと確認した行）', file=out)
    print('| 判定 | 動きの有無 | 3方向 | 単・複 |', file=out)
    print('|---|---:|---:|---:|', file=out)
    for s, r in results.items():
        print('| '+SOURCE_NAMES[s]+' | '+' | '.join(rate(r['agreement'][a]) for a in ASPECTS)+' |', file=out)
    for s, r in results.items():
        print('監査 '+s+': '+json.dumps(r['audit'], ensure_ascii=False), file=out)
    stats = contact_stats(labels)
    print(f'\n打球の瞬間：ずれている {stats["shifted"]}件 / 修正済み {stats["corrected"]}件 / 修正フレーム未入力 {stats["missing"]}件', file=out)
    if stats['deltas']:
        ds = [abs(d) for _, d in stats['deltas']]
        print(f'ずれ（修正−元ラベル）：絶対値の平均 {statistics.mean(ds):.2f}、中央値 {statistics.median(ds):.2f}、最大 {max(ds)} フレーム', file=out)
        for ident, delta in stats['deltas']:
            print(f'  {ident}: {delta:+d} フレーム', file=out)
    print('\n一致の方向・単複は本人が動きありで該当回答がある行。予測がなし／判定不能なら不一致。', file=out)
    print('打球修正行も元の分類との比較に含む（分類の再計算なし）。監査欄に件数を示す。', file=out)
    print('注意：予測を提示した確認作業であり、独立した盲検の正解率ではない。採用した予測と本人回答の一致には循環性がある。', file=out)
    print(f'without_candidates の本人打球から第8点手前を除いた{len(items)}件を対象とする。同一試合の少数例であり、未知試合への汎化精度にはできない。', file=out)
    composition_report(labels, items, out)
    return results


class ComparisonTests(unittest.TestCase):
    def test_human_contacts_neighbors_roles_and_predictions(self):
        from build_motion_ui import read_csv
        import hashlib
        import random
        data = json.loads((HERE/'data.json').read_text(encoding='utf-8'))
        items, sources = load_sources()
        base = HERE.parent
        human = [r for r in read_csv(base/'labels_human/human_labels_20261118.csv')
                 if r['condition'] == 'without_candidates']
        main = read_csv(base/'v_human/net_motion/motions_main.csv')
        roles = {(int(r['point']), int(r['contact_frame'])):r for r in read_csv(base/'v_human/net_motion/role_selection.csv')}
        self.assertEqual({(i['point'], i['contact_frame']) for i in items.values()},
                         {(int(h['point']), int(h['frame'])) for h in human
                          if roles[int(h['point']), int(h['frame'])]['exclude_p8_near'] != 'True'})
        self.assertEqual(set(data['excluded_ids']), {'p08_1_01','p08_1_03','p08_1_05','p08_1_07'})
        self.assertEqual(len(items), len(human) - len(data['excluded_ids']))
        self.assertTrue(all(not i['exclude_p8_near'] for i in items.values()))
        order = sorted({r['id'] for r in main})
        random.Random(20261118).shuffle(order)
        self.assertEqual(list(items), [ident for ident in order if ident not in data['excluded_ids']])
        for i in items.values():
            group = sorted((h for h in human if int(h['point']) == i['point'] and int(h['rally_no']) == i['rally']), key=lambda h:int(h['frame']))
            at = next(k for k,h in enumerate(group) if int(h['frame']) == i['contact_frame'])
            earlier = [int(h['frame']) for h in group[:at] if h['side'] == i['front_side']]
            self.assertEqual(i['previous_contact_frame'], max(earlier) if earlier else None)
            self.assertEqual(i['next_contact_frame'], int(group[at+1]['frame']) if at+1 < len(group) else None)
            self.assertEqual(i['role_selection'], roles[i['point'], i['contact_frame']])
            self.assertTrue((HERE/i['clip']).is_file())
            for method in SOURCES:
                r = next(r for r in main if r['id'] == i['id'] and r['coordinate_method'] == method)
                self.assertEqual(i['predictions'][method]['classification'], r['classification'])
                self.assertEqual(i['predictions'][method]['direction3'], r['final_direction3'])
                self.assertEqual(i['predictions'][method]['direction8'], r['final_direction8'])
        for method in SOURCES:
            self.assertEqual(sum(p['classification'] == 'unclassifiable' for p in sources[method].values()), 3)
        for rel, digest in data['input_sha256'].items():
            self.assertEqual(hashlib.sha256((base.parent.parent/rel).read_bytes()).hexdigest(), digest)

    def test_box_suspect_threshold_window_and_missing(self):
        from build_motion_ui import box_suspect_check
        def track(x, y=.5):
            return dict(court_proxy_x=str(x), court_proxy_y=str(y))
        tracks = {(99,'front'):track(-1), (100,'front'):track(-.15),
                  (101,'front'):track(1.15), (102,'front'):track(.5, 3),
                  (104,'front'):track(-1), (102,'other'):track(-1)}
        check = box_suspect_check(tracks,'front',100,103)
        self.assertEqual(check, dict(outside_frames=2, observed_frames=3, interval_frames=4))
        self.assertTrue(check['outside_frames'] * 2 >= check['interval_frames'])  # ちょうど半分
        tracks[101,'front'] = track(1.149999)
        check = box_suspect_check(tracks,'front',100,103)
        self.assertEqual(check['outside_frames'],1)
        self.assertFalse(check['outside_frames'] * 2 >= check['interval_frames'])
        tracks[100,'front'] = track(-.149999)
        self.assertEqual(box_suspect_check(tracks,'front',100,103)['outside_frames'],0)
        tracks[100,'front'] = track('nan')
        tracks[101,'front'] = track(-1,'')
        self.assertEqual(box_suspect_check(tracks,'front',100,101),
                         dict(outside_frames=0, observed_frames=0, interval_frames=2))
        self.assertEqual(box_suspect_check({},'',100,103),
                         dict(outside_frames=0, observed_frames=0, interval_frames=4))

    def test_box_suspect_matches_tracking_csv(self):
        from build_motion_ui import read_csv
        items, _ = load_sources()
        tracking = {}
        for i in items.values():
            point = i['point']
            if point not in tracking:
                tracking[point] = read_csv(HERE.parent/f'tracking/point_{point:02d}/player_court_proxies.csv')
            coords = [(float(t['court_proxy_x']),float(t['court_proxy_y'])) for t in tracking[point]
                      if t['player_slot'] == i['front_slot'] and i['contact_frame'] <= int(t['source_frame']) <= i['end_frame']
                      and t['court_proxy_x'] and t['court_proxy_y']]
            finite = [(x,y) for x,y in coords if math.isfinite(x) and math.isfinite(y)]
            outside = sum(x <= -.15 or x >= 1.15 for x,y in finite)
            total = i['end_frame'] - i['contact_frame'] + 1
            self.assertEqual(i['box_suspect_check'], dict(outside_frames=outside,
                             observed_frames=len(finite), interval_frames=total))
            self.assertEqual(i['box_suspect'], outside * 2 >= total)

    def test_composition_uses_human_answers_and_explicit_denominators(self):
        items = {str(k):dict(front_slot='near_front', exclude_p8_near=k == 0) for k in range(6)}
        labels = {str(k):dict(player_judgment='正しい', motion='motion', single_compound='single',
                  player_direction3='front', player_direction8='straight_front') for k in range(6)}
        labels['1']['motion'] = 'unknown'
        labels['2'].update(motion='none', single_compound='', player_direction3='', player_direction8='')
        labels['3']['player_judgment'] = '違う'
        labels['4'].update(single_compound='compound', player_direction3='back', player_direction8='straight_back')
        labels['5'].update(single_compound='', player_direction3='', player_direction8='')
        self.assertEqual(composition(labels, items, 'presence'), ({'motion':3,'none':1}, 4))
        self.assertEqual(composition(labels, items, 'complexity'), ({'single':1,'compound':1}, 2))
        counts, n = composition(labels, items, 'single_3dir')
        self.assertEqual((counts['front'], n), (1, 1))
        self.assertEqual(composition(labels, items, 'single_8dir', True)[1], 0)
        counts, n = composition(labels, items, 'final_3dir')
        self.assertEqual((counts['front'], counts['back'], n), (1, 1, 2))

    def test_roundtrip_real_sources(self):
        items, sources = load_sources()
        n = len(items)
        self.assertEqual(n, 28)
        self.assertEqual(set(sources), {'bbox','ankle'})
        self.assertTrue(all(len(s) == n for s in sources.values()))
        # 実際のJS exporterで全件のダミーCSVを作り、Python loaderとCLIまで通す。
        proc = subprocess.run(['node', str(HERE/'test_motion_ui.js'), '--csv'], capture_output=True, text=True, encoding='utf-8', check=True)
        with tempfile.TemporaryDirectory(prefix='.motion-review-test-', dir=HERE) as tmp:
            path = Path(tmp)/'motion_review.csv'
            path.write_bytes(proc.stdout.encode('utf-8'))
            labels = load_labels(path, items)
            self.assertEqual(len(labels), n)
            self.assertIn('改行\n', next(iter(labels.values()))['note'])
            result = report(labels, items, sources, io.StringIO())
            self.assertEqual(result['bbox']['audit']['shared'], n)
            self.assertEqual(contact_stats(labels)['shifted'], len(range(0,n,3)))
            self.assertEqual(contact_stats(labels)['deltas'][0][1], -3)
            cli = subprocess.run([sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),str(path)], capture_output=True,text=True,encoding='utf-8')
            self.assertEqual(cli.returncode, 0, cli.stderr)
            self.assertIn('循環性', cli.stdout)
            self.assertIn('井箟ほか', cli.stdout)
            self.assertNotIn('claude_', cli.stdout)
            with path.open('a', encoding='utf-8', newline='') as f:
                csv.DictWriter(f,FIELDS).writerow(next(iter(labels.values())))
            with self.assertRaisesRegex(ValueError,'ID重複'):
                load_labels(path,items)
        self.assertFalse(path.exists())

    def test_denominators_and_target_guard(self):
        base = dict(front_slot='far_front',front_side='F')
        items = {str(i):base.copy() for i in range(6)}
        labels = {str(i):dict(player_judgment='正しい',contact_judgment='正しい',motion='motion',single_compound='single',player_direction3='front',
                   **{f'bbox_{a}_judgment':'正しい' for a in ASPECTS}) for i in range(6)}
        pred = {str(i):dict(**base,motion='motion',complexity='single',direction3='front') for i in range(6)}
        labels['1']['bbox_motion_judgment']='違う'
        labels['2']['bbox_motion_judgment']='分からない'
        labels['3']['bbox_motion_judgment']=''
        labels['1']['motion']='unknown'
        pred['2']['front_slot']='far_rear'
        labels['3']['player_judgment']='違う'
        pred['4'].update(motion='none',complexity='',direction3='')
        pred['5'].update(motion='unclassifiable',complexity='',direction3='')
        m,c = agreement(labels,pred,items)
        self.assertEqual(m, {a:[1,3] for a in ASPECTS})
        self.assertEqual((c['human_unknown'],c['target_mismatch'],c['player_rejected']),(1,1,1))
        review = review_accuracy(labels,pred,'bbox')
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
