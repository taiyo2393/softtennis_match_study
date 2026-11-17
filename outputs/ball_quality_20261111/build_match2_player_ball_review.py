"""Synchronize fixed match2 rally windows with existing ball candidates."""
import os
import subprocess
import sys
from collections import Counter, defaultdict

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import build_044_spatiotemporal_pilot as common


OUT = common.HERE / 'match2_first_three_player_ball'
BALL = common.HERE / 'match2_first_three_reviewed_paths/ball_path.csv'
PILOT = common.HERE.parent / 'ball_quality_20261110/match2_camera_transfer_pilot'
SCOPES = {1: (12288, 12327, 12569), 2: (13560, 13657, 14055), 3: (15150, 15223, 15505)}
VIDEO_NAME = 'match2_first_three_players_and_ball.mp4'


def track_players(destination=OUT, scopes=SCOPES, ball_path=BALL):
    sys.path.insert(0, str(common.OLD))
    import fit_match2_court_geometry as geometry
    from fit_match2_court_geometry_centers import fit_line_centers
    from calibrated_camera_temporal_tracker import calibrated_temporal_class
    from confirmed_continuity_guard import adaptive, old
    from match_frame_coordinate_adapter import FrameCoordinates

    destination.mkdir(exist_ok=True)
    os.environ['YOLO_CONFIG_DIR'] = str(destination / 'ultralytics_config')
    from ultralytics import YOLO
    import torch
    import ultralytics

    protocol = common.load(PILOT / 'protocol.json')
    detector_config = protocol['detector']
    homography = np.asarray(protocol['image_to_court_homography_canonical'])
    legacy_homography = np.asarray(common.load(common.CAL)['image_to_court_homography'])
    adapter = FrameCoordinates(1920, 1080)
    make_tracker = calibrated_temporal_class(adaptive.AdaptiveNearTracker, homography, legacy_homography)
    geometry.fit_line = fit_line_centers
    assert torch.cuda.is_available()
    common.dump(destination / 'protocol.json', {
        'windows_preroll_core_first_core_last': scopes,
        'source_video': protocol['source_video'],
        'source_fps': '60000/1001',
        'ball_path_sha256': common.sha(ball_path),
        'camera_transfer_protocol_sha256': common.sha(PILOT / 'protocol.json'),
        'detector': detector_config,
        'runtime': {'torch': torch.__version__, 'ultralytics': ultralytics.__version__},
        'association': 'Frozen court-mapped tracker; restart semantic slots after camera-line failure.',
        'position': 'Native bounding box bottom center projected using fixed camera homography.',
    })
    rows, frames, events = [], [], []
    for point, (preroll, start, end) in scopes.items():
        detector = YOLO(detector_config['model'])
        capture = cv2.VideoCapture(protocol['source_video'])
        assert capture.isOpened() and capture.set(cv2.CAP_PROP_POS_FRAMES, preroll)
        assert abs(capture.get(cv2.CAP_PROP_FPS) - common.FPS) < .001
        tracker, segment = None, 0
        for frame in range(preroll, end + 1):
            ok, image = capture.read()
            assert ok and image.shape[:2] == (1080, 1920)
            assert round(capture.get(cv2.CAP_PROP_POS_FRAMES)) == frame + 1
            fit, _ = geometry.geometry(image)
            pred = detector.track(adapter.image(image), persist=True, classes=[0],
                                  conf=detector_config['confidence'], imgsz=detector_config['imgsz'],
                                  tracker=detector_config['tracker'], device=detector_config['device'],
                                  verbose=False)[0]
            assigned = []
            if fit is None:
                tracker = None
            else:
                if tracker is None:
                    tracker = make_tracker()
                    segment += 1
                    events.append(dict(point=point, segment=segment, start=frame, events=tracker.events))
                candidates, appearance = [], []
                for box in pred.boxes:
                    if box.id is None:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].cpu().tolist()
                    row = dict(source_frame=frame, timestamp_sec=frame / common.FPS,
                               track_id=int(box.id.item()), x1=x1, y1=y1, x2=x2, y2=y2,
                               bottom_center_x=(x1 + x2) / 2, bottom_center_y=y2,
                               confidence=float(box.conf.item()), candidate_origin='full_frame_only')
                    candidates.append(row)
                    appearance.append(old.a.appearance(image, row))
                assigned = tracker.step(frame, candidates, appearance)
                if frame >= start:
                    for slot, candidate, margin, cost, origin in assigned:
                        row = candidates[candidate]
                        u, v = common.project(homography, row['bottom_center_x'], row['bottom_center_y'])
                        rows.append(dict(point=point, segment=segment,
                                         **adapter.row(row, to_canonical=False),
                                         player_slot=old.a.SLOTS[slot], court_proxy_x=u, court_proxy_y=v,
                                         assignment_margin=float(margin), assignment_cost=float(cost),
                                         identity_source=origin))
            frames.append(dict(point=point, source_frame=frame, in_core=int(frame >= start),
                               camera_line_supported=int(fit is not None),
                               segment=segment if fit is not None else '',
                               assigned_slots=len(assigned)))
            if (frame - preroll) % 120 == 0:
                print(point, frame, end, 'line', fit is not None, 'slots', len(assigned), flush=True)
        capture.release()
        del detector
        common.write(destination / 'player_court_proxies.csv', rows)
        common.write(destination / 'player_frame_availability.csv', frames)
        common.dump(destination / 'player_segment_events.json', events)
        print(point, Counter(row['assigned_slots'] for row in frames
                             if row['point'] == point and row['in_core']), flush=True)


def render(review=None, *, tracking_directory=OUT, ball_path=BALL, scopes=SCOPES):
    protocol = common.load(tracking_directory / 'protocol.json')
    assert common.sha(ball_path) == protocol['ball_path_sha256']
    people = common.grouped(common.read(tracking_directory / 'player_court_proxies.csv'))
    coverage = {int(r['source_frame']): r for r in common.read(tracking_directory / 'player_frame_availability.csv')
                if r['in_core'] == '1'}
    ball = {int(r['source_frame']): r for r in common.read(ball_path)}
    suppressed_ball_frames = set() if review is None else review.get('suppressed_ball_frames', set())
    replacement_ball = {} if review is None else review.get('replacement_ball', {})
    for f in suppressed_ball_frames:
        assert ball[f]['current_candidate_present'] == '1'
        ball[f] = dict(ball[f], current_candidate_present='0')
    for f, (x, y) in replacement_ball.items():
        assert f not in suppressed_ball_frames and ball[f]['current_candidate_present'] == '1'
        ball[f] = dict(ball[f], current_x_px=x, current_y_px=y,
                       current_position_source='visual_recovery_candidate')
    frames = {f for _, start, end in scopes.values() for f in range(start, end + 1)}
    assert frames == coverage.keys() == ball.keys()
    slots = common.SLOTS
    for f, row in coverage.items():
        assigned = people[f]
        assert len(assigned) == int(row['assigned_slots'])
        assert len({r['player_slot'] for r in assigned}) == len({r['track_id'] for r in assigned}) == len(assigned)
        assert all(r['point'] == row['point'] == ball[f]['point'] for r in assigned)

    destination = OUT if review is None else review['destination']
    routes = {} if review is None else review['routes']
    stable_names = {} if review is None else review['stable_names']
    masked_positions = set() if review is None else review['masked_positions']
    stroke_groups = {} if review is None else review.get('strokes', {})
    event_footer = None if review is None else review.get('event_footer')
    guard_reasons = {} if review is None else review.get('guard_reasons', {})
    contact_review = None if review is None else review.get('contact_review')
    unavailable_reasons = {} if review is None else review.get('unavailable_reasons', {})
    id_legend = ('選手IDはラリー内の暫定ラベル' if not review else
                 review.get('id_legend', 'IDは3点で統一（ウェア・靴で照合）'))
    ball_has_reviewed_points = any(
        row['current_candidate_present'] == '1' and
        (row['current_position_source'] != 'baseline_model_candidate'
         or row['baseline_identity_confirmed_in_followup'] == '1')
        for row in ball.values())
    analysis_ends = {} if review is None else review.get('analysis_ends', {})
    trail_break_frames = set() if review is None else review.get('trail_break_frames', set())

    def vertices(route):
        if 'vertices' in route:
            return route['vertices']
        return [(route['start_x'], route['start_y']), (route['end_x'], route['end_y'])]

    # Common court scale and margins for all three points, including outside the lines.
    positions = np.asarray([(float(r['court_proxy_x']), float(r['court_proxy_y']))
                            for f, rows in people.items() for r in rows
                            if (f, r['player_slot']) not in masked_positions]
                           + [xy for group in routes.values() for r in group for xy in vertices(r)])
    assert len(positions) and np.isfinite(positions).all()
    xmin, ymin = np.minimum(positions.min(axis=0) - .08, (-.25, -.30))
    xmax, ymax = np.maximum(positions.max(axis=0) + .08, (1.25, 1.22))
    ratio = 250 / 540
    plot_top = 184 if review else 108
    plot_bottom = 630
    court_h = min(298 / ((xmax - xmin) * ratio), (plot_bottom - plot_top) / (ymax - ymin))
    court_w = court_h * ratio

    def court(u, v):
        return (1400 + (u - (xmin + xmax) / 2) * court_w,
                (plot_top + plot_bottom) / 2 - (v - (ymin + ymax) / 2) * court_h)

    W, VIEW_W, VIEW_H, TOP = 1600, 1200, 675, 52
    H = 850 if event_footer else 828 if stroke_groups else 780
    bg, panel, ink = (11, 20, 30), (18, 32, 44), (231, 242, 249)
    dim, border = (167, 192, 209), (74, 108, 129)
    candidate_color, assisted_color = (255, 248, 224), (255, 152, 82)
    fonts = {n: ImageFont.truetype(str(common.FONT), n) for n in (15, 17, 20, 24)}

    def xy(x, y):
        return float(x) * VIEW_W / 1920, float(y) * VIEW_H / 1080 + TOP

    def ball_xy(row):
        return xy(row['current_x_px'], row['current_y_px'])

    video = destination / (review.get('video_name', VIDEO_NAME) if review else VIDEO_NAME)
    with (destination / 'encoding.log').open('wb') as log:
        proc = subprocess.Popen([
            'C:/ffmpeg/bin/ffmpeg.exe', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt',
            'rgb24', '-s', f'{W}x{H}', '-r', '15000/1001', '-i', 'pipe:0', '-an', '-c:v',
            'libx264', '-preset', 'fast', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags',
            '+faststart', str(video)], stdin=subprocess.PIPE, stderr=log)
        mapping = []
        try:
            for point, (_, start, end) in scopes.items():
                capture = cv2.VideoCapture(protocol['source_video'])
                assert capture.isOpened() and capture.set(cv2.CAP_PROP_POS_FRAMES, start)
                point_routes = routes.get(point, [])
                strokes = stroke_groups.get(point, [])
                display_end = max([end] + [route['bounce_frame'] for route in point_routes if route['bounce_frame']])
                for f in range(start, display_end + 1):
                    ok, raw = capture.read()
                    assert ok and round(capture.get(cv2.CAP_PROP_POS_FRAMES)) == f + 1
                    if (f - start) % 2 and f != display_end:
                        continue
                    main = cv2.cvtColor(cv2.resize(raw, (VIEW_W, VIEW_H), interpolation=cv2.INTER_AREA),
                                        cv2.COLOR_BGR2RGB)
                    canvas = Image.new('RGB', (W, H), bg)
                    canvas.paste(Image.fromarray(main), (0, TOP))
                    draw = ImageDraw.Draw(canvas)

                    def label(x, y, value, size=17, fill=ink):
                        draw.text((x, y), value, font=fonts[size], fill=fill)

                    assigned = {stable_names.get(point, {}).get(r['player_slot'], r['player_slot']): r
                                for r in people.get(f, [])}
                    target = ball.get(f, {})
                    anchor = next((r for r in point_routes if r['bounce_frame'] == f), None)
                    stroke = next((s for s in reversed(strokes) if s['reference_frame'] <= f), None)
                    present = target.get('current_candidate_present') == '1'
                    assisted = present and (target['current_position_source'] != 'baseline_model_candidate'
                                            or target['baseline_identity_confirmed_in_followup'] == '1')
                    label(20, 9, f'match2 第{point}点', 24)
                    label(264, 14, f'F{f}    {f / common.FPS:.3f} 秒    0.5倍速', 17, dim)
                    if stroke:
                        label(610, 13, f'第{stroke["ordinal"]}打  {stroke["hitter"]} 候補', 17)
                    if f > end:
                        label(840, 13, '追跡区間の直後・着地確認', 17)
                    else:
                        ball_status = '除外' if f in suppressed_ball_frames else 'あり' if present else '欠測'
                        label(840, 13, f'選手枠 {len(assigned)}/4   球候補 {ball_status}', 17)
                    draw.rectangle((1202, TOP, W - 1, TOP + VIEW_H - 1), fill=panel)
                    label(1222, 66, 'コート上の選手配置', 20)
                    label(1222, 94, '枠下端の推定位置', 15, dim)
                    draw.rectangle((*court(xmin, ymax), *court(xmax, ymin)), outline=border)
                    draw.rectangle((*court(0, 1), *court(1, 0)), outline=ink, width=2)
                    for v in (.231, .5, .769):
                        draw.line([court(0, v), court(1, v)], fill=ink if v == .5 else border, width=2)
                    for u in (.125, .875):
                        draw.line([court(u, 0), court(u, 1)], fill=border, width=1)
                    draw.line([court(.5, .231), court(.5, .769)], fill=border, width=1)
                    if stroke:
                        active_route = point_routes[stroke['ordinal'] - 1]
                    else:
                        active_route = next((r for r in reversed(point_routes)
                                             if r['first_frame'] <= f <= r['bounce_frame'] + 30), None)
                    route_color = (255, 174, 104)
                    if routes:
                        label(1222, 114, '破線  打球地点・着地点の概略経路', 15, route_color)
                        label(1222, 135, '◇  バウンド推定点', 15, route_color)
                        if active_route:
                            points = [np.array(court(*xy)) for xy in vertices(active_route)]
                            for a, z in zip(points, points[1:]):
                                distance = float(np.linalg.norm(z - a))
                                if distance == 0:
                                    continue
                                direction = (z - a) / distance
                                for i in range(0, int(distance), 11):
                                    draw.line([tuple(a + i * direction),
                                               tuple(a + min(i + 6, distance) * direction)],
                                              fill=route_color, width=3)
                            bounce = active_route.get('bounce_point')
                            if 'vertices' not in active_route:
                                bounce = (active_route['end_x'], active_route['end_y'])
                            if bounce:
                                x, y = court(*bounce)
                                draw.line([(x, y - 6), (x + 6, y), (x, y + 6), (x - 6, y), (x, y - 6)],
                                          fill=route_color, width=2)
                                if active_route.get('bounce_evidence') == 'conditional_fit_with_user_reviewed_ball_identity':
                                    label(1222, 156, '奥の着地は暫定（球のみ確認済み）', 15, dim)
                            elif stroke:
                                label(1222, 156, 'この打球のバウンド点は未特定', 15, dim)
                            if stroke and not active_route['has_start']:
                                label(1222, 156, '打球地点は足元不明', 15, dim)
                        else:
                            label(1222, 156, 'この区間の球経路は未推定', 15, dim)
                    label(1222, 637, '奥 ↑　手前 ↓　周囲の余白はコート外', 15, dim)
                    for i, slot in enumerate(slots):
                        color, name = common.RGB[slot], common.NAMES[slot]
                        row = assigned.get(slot)
                        masked = row is not None and (f, row['player_slot']) in masked_positions
                        reason = guard_reasons.get((f, row['player_slot']), '') if row else ''
                        status = '範囲外' if f > end else '足元不明' if masked else '表示中' if row else '欠測'
                        if row is None and (f, slot) in unavailable_reasons:
                            status = unavailable_reasons[f, slot]
                        if reason in ('unstable_foot_position', 'insufficient_visible_run'):
                            status = '位置保留'
                        elif reason == 'airborne_serve_visual_review':
                            status = '空中・位置保留'
                        label(1222 + (i % 2) * 165, 665 + (i // 2) * 23,
                              f'{name}  {status}', 15, color if row else dim)
                        if row is None:
                            continue
                        x0, y0 = xy(row['x1'], row['y1'])
                        x1, y1 = xy(row['x2'], row['y2'])
                        draw.rectangle((x0, y0, x1, y1), outline=color, width=2)
                        draw.rectangle((x0, max(TOP, y0 - 22), x0 + 29, max(TOP, y0 - 22) + 22), fill=bg)
                        label(x0 + 3, max(TOP, y0 - 22), name, 15, color)
                        if masked:
                            continue
                        x, y = court(float(row['court_proxy_x']), float(row['court_proxy_y']))
                        assert 1208 <= x <= 1588 and 106 <= y <= 634
                        draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline=bg, width=2)
                        draw.text((x + 8, y - 9), name, font=fonts[15], fill=color,
                                  stroke_width=2, stroke_fill=panel)

                    # Stop at missing detections, stroke boundaries, or a flagged link.
                    tail = []
                    trail_start = stroke['reference_frame'] if stroke else start
                    for t in range(f, max(trail_start - 1, f - 12), -1):
                        row = ball.get(t, {})
                        if row.get('current_candidate_present') != '1':
                            break
                        tail.append(ball_xy(row))
                        if t in trail_break_frames:
                            break
                    if len(tail) > 1:
                        draw.line(tail, fill=bg, width=5)
                        draw.line(tail, fill=candidate_color, width=2)
                    if present:
                        x, y = ball_xy(target)
                        draw.ellipse((x - 7, y - 7, x + 7, y + 7), outline=bg, width=4)
                        draw.ellipse((x - 7, y - 7, x + 7, y + 7),
                                     outline=assisted_color if assisted else candidate_color, width=2)
                    if anchor and f > end:
                        x, y = xy(anchor['bounce_image_x'], anchor['bounce_image_y'])
                        draw.line([(x, y - 9), (x + 9, y), (x, y + 9), (x - 9, y), (x, y - 9)],
                                  fill=route_color, width=3)
                    ball_legend = '白線・白丸  球候補の画像上の軌跡'
                    if not ball_has_reviewed_points:
                        ball_legend += '（誤検出の一部を除外）' if suppressed_ball_frames else '（照合前）'
                    label(20, 744, ball_legend, 17, candidate_color)
                    if ball_has_reviewed_points:
                        assisted_legend = ('橙丸  拡大映像で照合した候補' if replacement_ball else
                                           '橙丸  人手の確認または注釈と照合済み')
                        label(386, 744, assisted_legend, 17, assisted_color)
                    label(860, 744, id_legend, 17, dim)
                    if stroke:
                        label(20, 788, '打球順（暫定）', 17, dim)
                        stroke_colors = {common.NAMES[s]: color for s, color in common.RGB.items()}
                        for i, item in enumerate(strokes):
                            left = 173 + 177 * i
                            selected = item['ordinal'] == stroke['ordinal']
                            color = stroke_colors[item['hitter']]
                            draw.rounded_rectangle((left, 779, left + 160, 815), radius=5,
                                                   fill=panel, outline=color if selected else border,
                                                   width=2 if selected else 1)
                            label(left + 10, 786, f'{i + 1}打目   {item["hitter"]}', 17, color)
                        if point == 3 and f >= 15494:
                            note = contact_review['display_note'] if contact_review else '終盤のN2の接触は判定保留'
                            label(1082, 788, note, 17, dim)
                    if event_footer:
                        event_footer(draw, point, f, fonts)
                    repeats = 12 if anchor and f > end else 1
                    data = canvas.tobytes()
                    for _ in range(repeats):
                        proc.stdin.write(data)
                        mapping.append(dict(video_frame=len(mapping), point=point, source_frame=f,
                                            source_seconds=f / common.FPS, player_slots=len(assigned),
                                            ball_candidate_present=int(present), ball_trail_samples=len(tail),
                                            ball_candidate_excluded=int(f in suppressed_ball_frames),
                                            ball_candidate_recovered=int(f in replacement_ball),
                                            ball_trail_break=int(f in trail_break_frames),
                                            masked_player_positions=sum((f, r['player_slot']) in masked_positions
                                                                        for r in assigned.values()),
                                            shown_route_id=active_route['id'] if active_route else '',
                                            hitter_candidate=stroke['hitter'] if stroke else '',
                                            analysis_core=int(f <= analysis_ends.get(point, end)),
                                            terminal_hold=int(repeats > 1)))
                capture.release()
        finally:
            proc.stdin.close()
            code = proc.wait()
    assert code == 0, (destination / 'encoding.log').read_text()

    common.write(destination / 'video_frame_map.csv', mapping)
    capture = cv2.VideoCapture(str(video))
    assert abs(capture.get(cv2.CAP_PROP_FPS) - 15000 / 1001) < .001
    for item in mapping:
        ok, frame = capture.read()
        assert ok and frame.shape[:2] == (H, W)
        stills = {12505, 12571, 14033, 14074, 15271, 15477} if stroke_groups else (
            {12571, 14033, 15477} if review else {13857})
        if contact_review:
            stills.add(15499)
        if any(r.get('bounce_evidence') for group in routes.values() for r in group):
            stills.add(13979)
        if review and 'preview_frames' in review:
            stills = review['preview_frames']
        if item['source_frame'] in stills:
            assert cv2.imwrite(str(destination / f'decoded_preview_F{item["source_frame"]}.jpg'), frame)
    assert not capture.read()[0]
    capture.release()

    summary = []
    for point, (_, start, end) in scopes.items():
        span = [coverage[f] for f in range(start, end + 1)]
        both = sum(int(r['assigned_slots']) == 4 and ball[int(r['source_frame'])]['current_candidate_present'] == '1'
                   for r in span)
        summary.append(dict(point=point, first_frame=start, last_frame=end, core_frames=len(span),
                            line_supported_frames=sum(int(r['camera_line_supported']) for r in span),
                            four_player_candidate_frames=sum(int(r['assigned_slots']) == 4 for r in span),
                            ball_candidate_frames=sum(int(ball[int(r['source_frame'])]['current_candidate_present'])
                                                      for r in span),
                            both_candidate_frames=both))
    common.write(destination / 'availability_summary.csv', summary)
    common.dump(destination / 'verification.json', {
        'decoded_frames': len(mapping), 'source_core_frames': len(frames),
        'output_fps': '15000/1001', 'source_frame_stride': 2, 'playback_speed': .5,
        'court_view_limits': {'x': [float(xmin), float(xmax)], 'y': [float(ymin), float(ymax)]},
        'all_shown_player_proxies_fit_viewport': True,
        'ball_geometry': 'image candidate trace and manually anchored schematic routes' if routes else 'image pixels only',
        'route_count': sum(map(len, routes.values())),
        'occluded_foot_positions_hidden': sum((f, r['player_slot']) in masked_positions
                                              for f, rows in people.items() for r in rows),
        'metrics': 'Candidate availability. Player identity and physical accuracy have no independent ground truth here.',
        'video_sha256': common.sha(video), 'rallies': summary,
    })
    print(summary, flush=True)


if __name__ == '__main__':
    {'track': track_players, 'render': render}[sys.argv[1] if len(sys.argv) > 1 else 'track']()
