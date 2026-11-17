"""Render source-frame brackets from the match2 points 4–6 visual review."""
from collections import defaultdict

import build_044_spatiotemporal_pilot as common
import sync_match2_points_4_6 as sync


OUT = common.HERE / 'match2_points_4_6_stroke_review'
META = OUT / 'stroke_observations.json'
STATUS = {
    'candidate': '打球者候補',
    'user_confirmed': '映像レビューで確認',
    'offscreen': '画面外・接触保留',
    'unresolved': '打球の有無を保留',
}
COLORS = {common.NAMES[slot]: common.RGB[slot] for slot in common.SLOTS}


def event_name(event):
    hitter = event['hitter']
    if event['status'] == 'candidate':
        return hitter
    if event.get('result') == 'net':
        return f'{hitter}（ネット）'
    if event['status'] == 'offscreen':
        return f'画面外（{hitter}?）'
    return f'{hitter}?' if hitter else '保留'


def display_footer(events):
    def draw_footer(draw, point, frame, fonts):
        point_events = events[point]
        active = next(event for event in reversed(point_events) if event['frames'][0] <= frame)
        first, last = active['frames']
        suffix = '区間内' if frame <= last else '直前の場面'
        dim = (167, 192, 209)
        draw.line((20, 776, 1580, 776), fill=(74, 108, 129))
        draw.text((20, 783), f'{suffix}    F{first}–F{last}    {STATUS[active["status"]]}',
                  font=fonts[17], fill=dim)
        draw.text((596, 783), active['note'], font=fonts[17], fill=dim)
        x = 20
        draw.text((x, 815), '打球の並び', font=fonts[17], fill=dim)
        x += 122
        for event in point_events:
            label = event_name(event)
            color = COLORS.get(event['hitter'], dim)
            width = int(draw.textlength(label, font=fonts[17])) + 20
            if event is active:
                draw.rounded_rectangle((x - 6, 810, x + width - 10, 841), radius=5,
                                       fill=(29, 46, 61), outline=color)
            draw.text((x, 815), label, font=fonts[17], fill=color)
            x += width
        draw.text((1120, 815), '元映像のフレーム範囲　接触時刻は暫定', font=fonts[17], fill=dim)
    return draw_footer


def write_report(points):
    court_positions = defaultdict(set)
    for row in common.load(OUT / 'guarded_positions.json'):
        if row['court_xy'] is not None:
            court_positions[row['source_frame']].add(row['display_id'])
    lines = [
        '# match2 第4〜6点の打球候補',
        '',
        '原映像のスイングと球の動きを照合した記録。フレーム範囲は接触を探す区間。',
        'Nは手前側、Fは奥側。得点側はスコア表示から照合した。',
        '「守備側位置あり」は、候補区間内で相手側2人の足元推定値が両方あるフレーム数。',
        '位置の正確さと打球順の完全性は評価していない。',
    ]
    for point in points:
        lines += [
            '',
            f'## 第{point["point"]}点（{point["winner_side"]}が得点）',
            '',
            '| 元映像のフレーム | 選手 | 判定 | 根拠 | 守備側位置あり |',
            '| --- | --- | --- | --- | --- |',
        ]
        for event in point['events']:
            first, last = event['frames']
            available = '—'
            if event['status'] == 'candidate':
                opponent = 'F' if event['hitter'].startswith('N') else 'N'
                pair = {f'{opponent}1', f'{opponent}2'}
                frames = range(first, last + 1)
                count = sum(pair.issubset(court_positions[f]) for f in frames)
                available = f'{count}/{len(frames)}' if any(court_positions[f] for f in frames) else '追跡範囲外'
            lines.append(f'| F{first}–F{last} | {event_name(event) or "—"} | {STATUS[event["status"]]} | '
                         f'{event["note"]}。[画像]({event["evidence"]}) | {available} |')
        lines.extend(('', point['terminal_note']))
    (OUT / '打球候補.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main():
    data = common.load(META)
    events = {row['point']: row['events'] for row in data['points']}
    assert events.keys() == sync.scopes().keys()
    for point, windows in events.items():
        start, end = sync.scopes()[point][1:]
        assert windows[0]['frames'][0] < start <= windows[1]['frames'][0]
        previous_end = 0
        for item in windows:
            first, last = item['frames']
            assert previous_end < first <= last <= end
            assert (OUT / item['evidence']).is_file(), item['evidence']
            previous_end = last
    for entry in data['ball_exclusions']:
        assert (OUT / entry['evidence']).is_file()

    sync.review(split_abrupt_ball_links=True, visual_exclusions=True, visual_recovery=True,
                destination_override=OUT, additional_ball_exclusions=data['ball_exclusions'],
                event_footer=display_footer(events))
    write_report(data['points'])


if __name__ == '__main__':
    main()
