"""Extract receiving-pair geometry at the existing stroke contact windows."""
from collections import Counter, defaultdict
from math import dist

from PIL import Image, ImageDraw, ImageFont

import build_044_spatiotemporal_pilot as common
from check_match2_serve_positions import OUT as SERVE
from review_match2_stroke_routes import METRES


BASE = common.HERE / 'match2_first_three_player_ball_v4'
OUT = common.HERE / 'match2_receiving_position_trial'
POSITION = BASE / 'guarded_player_positions.csv'
STROKE = BASE / 'stroke_candidates.csv'
ROUTES = BASE / 'schematic_routes.json'
UPDATED_ROUTES = BASE.with_name('match2_first_three_player_ball_v5') / 'schematic_routes.json'
REVISED_BASE = BASE.with_name('match2_first_three_player_ball_v6')
SERVE_POS = SERVE / 'contact_positions.json'
CONTACT_REVIEW = common.HERE / 'match2_raw_stroke_review/point3_contact_review.json'


def index_positions():
    frames = defaultdict(dict)
    for row in common.read(POSITION):
        if row['position_status'] == 'proxy_shown':
            frames[int(row['point']), int(row['source_frame'])][row['display_id']] = (
                float(row['shown_court_x']), float(row['shown_court_y']))
    for row in common.load(SERVE_POS):
        coords = row['shown_court_xy']
        if coords is not None:
            frame = frames[row['point'], row['source_frame']]
            assert row['player'] not in frame
            frame[row['player']] = coords
    return frames


def metres(xy):
    return tuple(v * scale for v, scale in zip(xy, METRES))


def summarize(stroke, route, frames):
    point, ordinal = (int(stroke[key]) for key in ('point', 'ordinal'))
    start, end = (int(stroke[key]) for key in ('first_candidate_frame', 'last_candidate_frame'))
    players = ('N1', 'N2') if stroke['hitter'].startswith('F') else ('F1', 'F2')
    bounce = route['bounce_point'] if route is not None else None
    samples, missing = [], []
    for f in range(start, end + 1):
        positions = frames[point, f]
        if not all(name in positions for name in players):
            missing.append(dict(source_frame=f, unavailable=[name for name in players if name not in positions]))
            continue
        a, b = (metres(positions[name]) for name in players)
        samples.append(dict(source_frame=f, pair_court_xy=[positions[name] for name in players],
                            lateral_spacing_m=abs(a[0] - b[0]), depth_difference_m=abs(a[1] - b[1]),
                            nearest_to_bounce_at_contact_m=min(dist(a, metres(bounce)), dist(b, metres(bounce)))
                            if bounce is not None else None))
    complete = len(samples) == end - start + 1
    columns = ('lateral_spacing_m', 'depth_difference_m', 'nearest_to_bounce_at_contact_m')
    features = {}
    if complete:
        for key in columns:
            values = [sample[key] for sample in samples if sample[key] is not None]
            features[key] = [min(values), max(values)] if values else None
    middle = (start + end) // 2
    return dict(point=point, stroke=ordinal, hitter_candidate=stroke['hitter'],
                receiving_players=players, candidate_first=start, candidate_last=end,
                candidate_frames=end - start + 1, pair_frames=len(samples), whole_window_pair=complete,
                missing=missing, bounce_candidate_frame=route['bounce_frame'] if route is not None else None,
                bounce_candidate_court_xy=bounce,
                bounce_evidence=route.get('bounce_evidence') if route is not None else None,
                middle_candidate_frame_positions=next((s for s in samples if s['source_frame'] == middle), None),
                features_min_max=features, per_frame=samples)


def render_preview(paired, destination):
    columns = 4 if len(paired) > 6 else 3
    height_px = 135 + 320 * max(1, (len(paired) + columns - 1) // columns)
    image = Image.new('RGB', (40 + 400 * columns, height_px), (14, 23, 33))
    draw = ImageDraw.Draw(image)
    fonts = {size: ImageFont.truetype(str(common.FONT), size) for size in (16, 18, 21, 31)}
    ink, dim, line = (233, 241, 247), (171, 193, 208), (68, 104, 124)
    amber = (255, 174, 104)
    colors = {common.NAMES[slot]: color for slot, color in common.RGB.items()}
    draw.text((25, 12), 'match2　打球候補時のレシーブ側配置と着地点候補', font=fonts[31], fill=ink)
    draw.text((25, 59), '● 接触候補区間の中間フレームの選手位置　◇ その後の着地点候補　　奥↑　手前↓',
              font=fonts[18], fill=dim)
    for idx, row in enumerate(paired):
        px, py = (26 + idx % columns * 400, 104 + idx // columns * 320)
        draw.rounded_rectangle((px, py, px + 380, py + 304), 8, fill=(20, 34, 46), outline=line)
        heading = f'第{row["point"]}点・{row["stroke"]}打目　{row["hitter_candidate"]}候補'
        draw.text((px + 15, py + 8), heading, font=fonts[21], fill=ink)
        height = 205
        width = height * METRES[0] / METRES[1]
        cx = px + 101
        top = py + 54

        def court(x, y):
            return (cx + (x - .5) * width, top + (1 - y) * height)

        draw.rectangle((*court(0, 1), *court(1, 0)), outline=ink, width=2)
        for y in (.231, .5, .769):
            draw.line([court(0, y), court(1, y)], width=1, fill=ink if y == .5 else line)
        for x in (.125, .875):
            draw.line([court(x, 0), court(x, 1)], width=1, fill=line)
        draw.line([court(.5, .231), court(.5, .769)], width=1, fill=line)
        sample = row['middle_candidate_frame_positions']
        for name, coords in zip(row['receiving_players'], sample['pair_court_xy']):
            x, y = court(*coords)
            color = colors[name]
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
            draw.text((x + 7, y - 10), name, font=fonts[16], fill=color)
        x, y = court(*row['bounce_candidate_court_xy'])
        draw.line([(x, y - 6), (x + 6, y), (x, y + 6), (x - 6, y), (x, y - 6)], fill=amber, width=2)
        side_x = px + 182
        draw.text((side_x, py + 59), f'配置 F{sample["source_frame"]}', font=fonts[16], fill=dim)
        draw.text((side_x, py + 82), f'着地候補 F{row["bounce_candidate_frame"]}', font=fonts[16], fill=dim)
        draw.text((side_x, py + 124), f'左右間隔 {sample["lateral_spacing_m"]:.1f} m', font=fonts[18], fill=ink)
        draw.text((side_x, py + 153), f'前後差    {sample["depth_difference_m"]:.1f} m', font=fonts[18], fill=ink)
        if row['bounce_evidence']:
            draw.text((side_x, py + 206), '奥の着地は暫定', font=fonts[16], fill=amber)
            draw.text((side_x, py + 229), '球の像のみ確認済み', font=fonts[16], fill=dim)
    footer = height_px - 30
    draw.text((25, footer), '選手位置は検出枠の下端から推定。着地点は候補。', font=fonts[16], fill=dim)
    image.save(destination / 'match2_receiving_positions_preview.png')


def main():
    routes_path = UPDATED_ROUTES if UPDATED_ROUTES.exists() else ROUTES
    strokes_path = STROKE
    destination = OUT.with_name('match2_receiving_position_trial_v2') if routes_path == UPDATED_ROUTES else OUT
    if (REVISED_BASE / 'schematic_routes.json').exists():
        routes_path = REVISED_BASE / 'schematic_routes.json'
        strokes_path = REVISED_BASE / 'stroke_candidates.csv'
        destination = OUT.with_name('match2_receiving_position_trial_v3')
    destination.mkdir(exist_ok=True)
    frames = index_positions()
    stroke = common.read(strokes_path)
    source_routes = common.load(routes_path)
    route_index = {(r['point'], r['ordinal']): r for group in source_routes.values() for r in group}
    rows = [summarize(s, route_index[int(s['point']), int(s['ordinal'])], frames) for s in stroke]
    for row in rows:
        assert row['pair_frames'] + len(row['missing']) == row['candidate_frames']
        if row['bounce_candidate_frame'] is not None:
            assert row['bounce_candidate_frame'] > row['candidate_last']
    paired = [row for row in rows if row['whole_window_pair'] and row['bounce_candidate_frame'] is not None]
    n_complete = sum(r['whole_window_pair'] for r in rows)
    complete = [r for r in rows if r['whole_window_pair']]
    contact = common.load(CONTACT_REVIEW)
    common.dump(destination / 'stroke_receiving_positions.json', rows)
    common.dump(destination / 'result.json', dict(
        stroke_candidates=len(rows),
        receiving_pair_full_contact_window=n_complete,
        bounce_candidates=sum(r['bounce_candidate_frame'] is not None for r in rows),
        receiving_pair_and_bounce=len(paired),
        paired_bounces_by_receiving_side=dict(Counter(
            'near' if r['receiving_players'][0].startswith('N') else 'far' for r in paired)),
        incomplete_pairs=[dict(point=r['point'], stroke=r['stroke'], pair_frames=r['pair_frames'],
                               contact_frames=r['candidate_frames']) for r in rows if not r['whole_window_pair']],
        lateral_spacing_max_within_window_change_m=max(
            r['features_min_max']['lateral_spacing_m'][1] - r['features_min_max']['lateral_spacing_m'][0]
            for r in complete),
        depth_difference_max_within_window_change_m=max(
            r['features_min_max']['depth_difference_m'][1] - r['features_min_max']['depth_difference_m'][0]
            for r in complete),
        terminal_review_point3=contact,
        method='At the provisional hitter contact window, use available opposite-side box-bottom proxies. '
               'No extrapolation. For complete windows report min and max over every candidate time.',
        distance_definition='Nearest receiving player at impact proxy to later candidate bounce. '
                            'It is a geometric feature, not distance run or a reaction-time measurement.',
        scope='Three rallies; provisional shot contacts and landing points; no new independent position '
              'references or return labels. No claims about tactical effect.',
        source_hashes={str(p): common.sha(p) for p in (POSITION, strokes_path, routes_path, SERVE_POS, CONTACT_REVIEW)},
    ))
    render_preview(paired, destination)
    print('strokes', len(rows), 'receiving pair', n_complete, 'pair and bounce', len(paired))


if __name__ == '__main__':
    main()
