"""Rebuild reviewed 056 frames and align left boxes with filter_status.

Only this renderer and files in 056_cleaned_wide_court may change. Cleaning
rules, input tables and reviewed ball/stroke/bounce/route drawings are retained.
Run with study/.venv/Scripts/python.exe (use -B to avoid import cache writes).
"""
from pathlib import Path, PureWindowsPath
import argparse
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import build_044_spatiotemporal_pilot as b
from render_044_estimated_routes import dashed, diamond, GREEN, YELLOW, CYAN, ORANGE
from render_056_wide_court_view import XLIM, YLIM
from clean_056_player_positions import OBSERVED_STATUSES

HERE = Path(__file__).resolve().parent
PARENT = HERE / '056_reviewed_consolidated'
OUT = HERE / '056_cleaned_wide_court'
FRAMES = HERE.parent / 'ball_quality_20261109/rally_repair_056/frames'
CHECK_FRAMES = (154376, 154690, 154762)
REJECTED_STATUSES = ('rejected_box_height', 'rejected_speed')
# New note ends above the unchanged stroke strip, whose first row is y=921.
RIGHT_PANEL = (1320, 100, 1920, 921)  # half-open
MAE_LIMIT = 3.0  # RGB intensity levels (0..255), frozen before rendering


def render_reviewed_frame(source, f, people, filters, strokes, path, phase, times, fonts,
                          routes, bounces, *, cleaned):
    """Reproduce the consolidation main-loop drawing; only left boxes may differ.

    The original right court is drawn too, then replaced by the existing wide
    court in main. Ball phase checks, coordinates, text and stroke strip retain
    the original code and drawing order. No decoded video pixels are reused.
    """
    raw=Image.open(source).convert('RGB');image=Image.new('RGB',(1920,1080),(12,22,32));d=ImageDraw.Draw(image)
    def txt(x,y,msg,size=20,color=(225,235,245)):d.text((x,y),msg,font=fonts[size],fill=color)
    txt(24,12,'056：確認済みの修正をまとめた選手・球・配球表示',34)
    txt(24,60,'13打球の順序は確認済み ｜ 接触時刻・選手位置・バウンド点は暫定 ｜ 半分の速度',22)
    image.paste(raw.resize((1280,720),Image.Resampling.LANCZOS),(16,104));d=ImageDraw.Draw(image)
    for person in people[f]:
        color=b.RGB[person['player_slot']]
        x1,y1,x2,y2=[float(person[k])*2/3 for k in ['x1_px','y1_px','x2_px','y2_px']]
        decision = filters[f][person['player_slot']]['filter_status']
        assert decision in (*OBSERVED_STATUSES, *REJECTED_STATUSES), decision
        if cleaned and decision in REJECTED_STATUSES:
            corners = [(16+x1,104+y1), (16+x2,104+y1),
                       (16+x2,104+y2), (16+x1,104+y2)]
            for a, z in zip(corners, corners[1:] + corners[:1]):
                dashed(d, a, z, color, width=1, dash=6, gap=4)
            label = b.NAMES[person['player_slot']] + ' 除外'
        else:
            d.rectangle((16+x1,104+y1,16+x2,104+y2),outline=color,width=2)
            label = b.NAMES[person['player_slot']]
        txt(int(16+x1),int(78+y1),label,20,color)
    for g in range(max(times[0]+2,f-36),f+1,2):
        a,z=path[g-2],path[g]
        if a['kind']=='missing' or z['kind']=='missing':continue
        if not(phase[g].startswith('flight_') and phase[g-2]==phase[g-1]==phase[g]):continue
        points=[(16+float(r['display_image_x_px'])*2/3,104+float(r['display_image_y_px'])*2/3) for r in [a,z]]
        if a['kind']==z['kind']=='reviewed_observation':d.line(points,fill=GREEN,width=3)
        else:dashed(d,points[0],points[1],YELLOW,3,5,4)
    r=path[f]
    if r['kind']!='missing':
        x=16+float(r['display_image_x_px'])*2/3;y=104+float(r['display_image_y_px'])*2/3
        if r['kind']=='estimated':diamond(d,x,y,YELLOW,9,3)
        else:d.ellipse((x-8,y-8,x+8,y+8),outline=GREEN,width=3)
    status={'reviewed_observation':'球位置：人手整備・実球確認を反映','estimated':'球位置：補間・モデル推定（未確認）',
            'missing':'球位置：未取得' if r['annotation_available'] else '球位置：この時点は注釈なし'}[r['kind']]
    txt(32,780,status,20,YELLOW if r['kind']!='reviewed_observation' else GREEN)
    current=max((j for j,s in enumerate(strokes) if int(s['source_frame_first'])<=f),default=0)
    sid=strokes[current]['stroke_id'];left,top,w,h=1455,228,250,540
    def cp(u,v):return left+w*u,top+h*(1-v)
    txt(1335,109,'選手配置と配球模式線',30)
    txt(1335,157,'○ バウンド候補　◇ 予測点（未確認）',18,ORANGE)
    d.rectangle((left,top,left+w,top+h),outline=(145,180,200),width=2)
    for v in [.231,.5,.769]:d.line([cp(0,v),cp(1,v)],fill=(205,220,235) if v==.5 else (80,120,145),width=3 if v==.5 else 1)
    for u in [.125,.875]:d.line([cp(u,0),cp(u,1)],fill=(80,120,145))
    d.line([cp(.5,0),cp(.5,.231)],fill=(80,120,145));d.line([cp(.5,.769),cp(.5,1)],fill=(80,120,145))
    txt(left+95,186,'奥側');txt(left+95,776,'手前側')
    for j in range(max(0,current-1),current+1):
        points=[cp(float(v['court_x']),float(v['court_y'])) for v in routes[strokes[j]['stroke_id']] if v['court_x']!='']
        for a,z in zip(points,points[1:]):
            ok,aa,zz=cv2.clipLine((1330,180,575,625),tuple(round(v) for v in a),tuple(round(v) for v in z))
            if ok:dashed(d,aa,zz,CYAN if j==current else (65,100,125),2,7,5)
    for point in bounces:
        if f<int(point['source_frame']):continue
        x,y=cp(float(point['court_x']),float(point['court_y']));active=point['after_stroke']==sid
        if point['status'].startswith('AI_visible'):d.ellipse((x-8,y-8,x+8,y+8),outline=ORANGE,width=3 if active else 1)
        else:diamond(d,x,y,YELLOW,9,3 if active else 1)
        if active:txt(int(x+12),int(y-14),point['id'],18,ORANGE)
    for person in people[f]:
        x,y=cp(float(person['court_proxy_x']),float(person['court_proxy_y']));color=b.RGB[person['player_slot']]
        if 1320<x<1905 and 180<y<804:d.ellipse((x-6,y-6,x+6,y+6),fill=color);txt(int(x+8),int(y-13),b.NAMES[person['player_slot']],18,color)
    txt(1335,811,'人物：既存入力の範囲外' if f<154086 or f>154855 else '人物点は枠下端の代理位置',18,YELLOW)
    txt(1335,847,'破線は模式線。実際の3D軌道ではない。',16)
    txt(24,837,f'元フレーム {f} ｜ 表示中の打球 {current+1}/13：'+b.NAMES[strokes[current]['hitter_candidate']],22)
    footer='緑○＝人手整備・実球確認　黄◇・黄破線＝推定　打球・欠測をまたぐ線は切っています。'
    if abs(f-154324)<=6:footer='F154324：誤検出の青候補は除外。黄◇は従来の補間推定で、実球として未確認。'
    if abs(f-154780)<=6:footer='F154780：別物体の旧候補を除外し、別位置で実球と確認された点を表示。'
    txt(24,874,footer,20)
    start=max(0,min(len(strokes)-5,current-2))
    for k,s in enumerate(strokes[start:start+5]):
        j=start+k;x=24+k*376;color=b.RGB[s['hitter_candidate']]
        d.rounded_rectangle((x,921,x+358,1006),radius=7,fill=(29,49,66) if j==current else (18,31,43),outline=color if j==current else (50,70,85),width=2)
        txt(x+12,929,f'{j+1}：'+b.NAMES[s['hitter_candidate']]+'（順序確認済み）',22,color)
        txt(x+12,970,f'暫定 F{s["source_frame_first"]}〜{s["source_frame_last"]}',18)
    txt(24,1030,'球の高さ・速度・最終2バウンドの位置は測定していません。',20)
    return image


def main():
    assert OUT.is_dir(), 'Run clean_056_player_positions.py first'
    video = OUT / '056_cleaned_wide_court.mp4'
    before = b.load(OUT / 'existing_files_sha256_before.json')
    renderer_name = Path(__file__).name
    protected = {name: digest for name, digest in before.items() if name != renderer_name}
    assert all(b.sha(HERE / name) == digest for name, digest in protected.items())
    # Protect cleaning inputs and all other existing output files as well.
    mutable_names = {video.name, 'cleaning_summary.json', 'README.md', 'verification.json',
                     *(f'check_F{f}.jpg' for f in CHECK_FRAMES)}
    output_protected = {p.name: b.sha(p) for p in OUT.iterdir()
                        if p.is_file() and p.name not in mutable_names}
    summary = b.load(OUT / 'cleaning_summary.json')
    people = b.grouped(b.read(OUT / 'cleaned_player_positions.csv'))
    raw_people = b.grouped(b.read(PARENT / 'player_court_proxies.csv'))
    filters = {f: {p['player_slot']: p for p in rows} for f, rows in people.items()}
    strokes = b.read(PARENT / 'stroke_order_confirmed.csv')
    bounces = b.read(PARENT / 'bounce_court_points.csv')
    route_rows = b.read(PARENT / 'schematic_court_routes.csv')
    routes = {s['stroke_id']: [r for r in route_rows if r['stroke_id'] == s['stroke_id']] for s in strokes}
    mapping = b.read(PARENT / 'video_source_frame_map.csv')
    times = [int(row['source_frame']) for row in mapping]
    assert len(times) == len(set(times)) == 401 and times == sorted(times)
    assert set(filters) == set(times)
    path = {int(r['source_frame']): r for r in b.read(PARENT / 'display_ball_path.csv')}
    # The original loop builds this field as an int; CSV strings '0' are truthy.
    for ball in path.values():
        ball['annotation_available'] = int(ball['annotation_available'])
    phase = {int(r['source_frame']): r['phase'] for r in
             b.read(HERE / '056_post_hit_trial/event_segmented_model_path.csv')}
    manifest = b.load(PARENT / 'source_image_manifest.json')
    sources = {int(r['source_frame']): FRAMES / PureWindowsPath(r['path']).name for r in manifest}
    assert len(sources) == len(manifest) == 401 and set(sources) == set(times)
    assert all(b.sha(sources[int(r['source_frame'])]) == r['sha256'] for r in manifest)
    for f in times:
        assert set(filters[f]) == set(b.SLOTS) and len(people[f]) == 4
        raw_slots = [r['player_slot'] for r in raw_people[f]]
        assert len(raw_slots) == len(set(raw_slots))
        for slot in b.SLOTS:
            decision = filters[f][slot]['filter_status']
            assert (slot in raw_slots) == (decision in (*OBSERVED_STATUSES, *REJECTED_STATUSES))
            if slot not in raw_slots:
                assert decision == 'missing'
    plot_top, plot_height, center_x = 200., 630., 1620.
    height = plot_height / (YLIM[1] - YLIM[0])
    width = height * 250 / 540
    court_left = center_x - width / 2
    court_top = plot_top + (YLIM[1] - 1) * height

    def cp(u, v):
        return court_left + width * u, court_top + height * (1 - v)

    view_left, view_top = cp(XLIM[0], YLIM[1])
    view_right, view_bottom = cp(XLIM[1], YLIM[0])
    fonts = {s: ImageFont.truetype(str(b.FONT), s) for s in [16, 18, 20, 22, 25, 30, 34]}
    cap = cv2.VideoCapture(str(PARENT / '056_reviewed_players_ball_and_court.mp4'))
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert abs(fps - b.FPS / 4) < 1e-6
    proc = subprocess.Popen([
        'C:/ffmpeg/bin/ffmpeg.exe', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', '1920x1080', '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
        '-preset', 'fast', '-crf', '17', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    outside = []
    drawn = {slot: {status: 0 for status in (*OBSERVED_STATUSES, 'interpolated')} for slot in b.SLOTS}
    dashed_counts = {slot: {status: 0 for status in REJECTED_STATUSES} for slot in b.SLOTS}
    dashed_interpolated = {slot: 0 for slot in b.SLOTS}
    rejected_frames = []
    equality_frames = []
    comparisons = []
    sample_frames = {times[0], times[3], times[len(times)//2], times[-1], *CHECK_FRAMES}
    count = 0
    try:
        for row in mapping:
            ok, frame = cap.read()
            assert ok
            f = int(row['source_frame'])
            # Both complete images are drawn from the original JPEG in memory.
            legacy = render_reviewed_frame(sources[f], f, raw_people, filters, strokes,
                                           path, phase, times, fonts, routes, bounces, cleaned=False)
            image = render_reviewed_frame(sources[f], f, raw_people, filters, strokes,
                                          path, phase, times, fonts, routes, bounces, cleaned=True)
            legacy_array = np.asarray(legacy)
            rejected = [p for p in raw_people[f]
                        if filters[f][p['player_slot']]['filter_status'] in REJECTED_STATUSES]
            if rejected:
                rejected_frames.append(f)
            else:
                assert np.array_equal(np.asarray(image), legacy_array), f
            for p in rejected:
                slot = p['player_slot']
                decision = filters[f][slot]
                dashed_counts[slot][decision['filter_status']] += 1
                dashed_interpolated[slot] += int(decision['status'] == 'interpolated')
            if f in sample_frames:
                decoded_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                diff = np.abs(legacy_array.astype(np.int16) - decoded_rgb.astype(np.int16))
                left_mae = float(diff[:, :1320].mean())
                comparisons.append(dict(
                    source_frame=f, left_rgb_mean_absolute_difference=left_mae,
                    video_tile_rgb_mean_absolute_difference=float(diff[104:824, 16:1296].mean()),
                    stroke_strip_rgb_mean_absolute_difference=float(diff[921:].mean()),
                    left_rgb_95th_percentile_absolute_difference=float(np.percentile(diff[:, :1320], 95)),
                    passed=left_mae < MAE_LIMIT))
                assert left_mae < MAE_LIMIT, comparisons[-1]
            d = ImageDraw.Draw(image)
            d.rectangle((1320, 100, 1919, 920), fill=(12, 22, 32))

            def txt(x, y, s, size=18, color=(225, 235, 245)):
                d.text((x, y), s, font=fonts[size], fill=color)

            txt(1335, 109, 'コート外まで含む選手配置', 30)
            txt(1335, 153, '白線＝コート　周囲の余白＝コート外', 18)
            d.rectangle((view_left, view_top, view_right, view_bottom), fill=(16, 31, 43), outline=(45, 70, 85), width=1)
            x0, y1 = cp(0, 1)
            x1, y0 = cp(1, 0)
            d.rectangle((x0, y1, x1, y0), outline=(205, 225, 235), width=2)
            for v in [.231, .5, .769]:
                d.line([cp(0, v), cp(1, v)], fill=(205, 220, 235) if v == .5 else (80, 120, 145), width=2 if v == .5 else 1)
            for u in [.125, .875]:
                d.line([cp(u, 0), cp(u, 1)], fill=(80, 120, 145), width=1)
            d.line([cp(.5, 0), cp(.5, .231)], fill=(80, 120, 145))
            d.line([cp(.5, .769), cp(.5, 1)], fill=(80, 120, 145))
            txt(x1 + 14, y1 - 12, '奥ベースライン', 16)
            txt(x1 + 14, y0 - 12, '手前ベースライン', 16)
            current = max((j for j, r in enumerate(strokes) if int(r['source_frame_first']) <= f), default=0)
            sid = strokes[current]['stroke_id']
            for j in range(max(0, current - 1), current + 1):
                points = [cp(float(r['court_x']), float(r['court_y'])) for r in routes[strokes[j]['stroke_id']] if r['court_x'] != '']
                for a, z in zip(points, points[1:]):
                    ok, aa, zz = cv2.clipLine(
                        (round(view_left), round(view_top), round(view_right - view_left), round(view_bottom - view_top)),
                        tuple(round(v) for v in a), tuple(round(v) for v in z))
                    if ok:
                        dashed(d, aa, zz, CYAN if j == current else (65, 100, 125), 2, 6, 4)
            for point in bounces:
                if f < int(point['source_frame']):
                    continue
                x, y = cp(float(point['court_x']), float(point['court_y']))
                active = point['after_stroke'] == sid
                if point['status'].startswith('AI_visible'):
                    d.ellipse((x - 5, y - 5, x + 5, y + 5), outline=ORANGE, width=2 if active else 1)
                else:
                    diamond(d, x, y, YELLOW, 6, 2 if active else 1)
                if active:
                    txt(x + 8, y - 12, point['id'], 16, ORANGE)
            frame_drawn = 0
            for p in people[f]:
                if p['status'] not in (*OBSERVED_STATUSES, 'interpolated'):
                    continue
                u, v = float(p['court_x']), float(p['court_y'])
                x, y = cp(u, v)
                # Same circle margin as the original view; skip without clamping.
                inside = view_left + 7 < x < view_right - 7 and view_top + 7 < y < view_bottom - 7
                if not inside:
                    outside.append(dict(source_frame=f, player_slot=p['player_slot'], status=p['status'], court_x=u, court_y=v))
                    continue
                color = b.RGB[p['player_slot']]
                if p['status'] in OBSERVED_STATUSES:
                    d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color, outline=(12, 22, 32), width=1)
                else:
                    d.ellipse((x - 6, y - 6, x + 6, y + 6), outline=color, width=2)
                txt(x + 8, y - 12, b.NAMES[p['player_slot']], 18, color)
                drawn[p['player_slot']][p['status']] += 1
                frame_drawn += 1
            txt(1335, 840, '人物点は枠下端の代理位置。隠れ等による異常点を除外。', 16)
            txt(1335, 862, '採用点間30元フレーム以内は補間（白抜き、約0.5秒）。', 16)
            txt(1335, 884, '左の破線枠＝隠れなどで除いた枠', 16)
            if not frame_drawn:
                txt(1335, 815, '人物：採用・補間点なし', 16, YELLOW)
            array = np.asarray(image)
            # Exclude only the right panel, including the added note. No-rejection
            # frames must retain every other pixel, including title and strip.
            if not rejected:
                assert np.array_equal(array[:, :1320], legacy_array[:, :1320]), f
                assert np.array_equal(array[:100], legacy_array[:100]), f
                assert np.array_equal(array[921:], legacy_array[921:]), f
                equality_frames.append(f)
            # These regions cannot be affected by left player boxes on any frame.
            assert np.array_equal(array[:100], legacy_array[:100]), f
            assert np.array_equal(array[921:], legacy_array[921:]), f
            proc.stdin.write(array.tobytes())
            count += 1
            if f in CHECK_FRAMES:
                image.save(OUT / f'check_F{f}.jpg', quality=95)
            if count % 50 == 0:
                print(f'Rendered {count}/401; no-rejection pixel checks {len(equality_frames)}', flush=True)
        ok, _ = cap.read()
        assert not ok
        proc.stdin.close()
        err = proc.stderr.read()
        assert proc.wait() == 0, err
    finally:
        cap.release()
        if proc.poll() is None:
            proc.kill()
            proc.wait()
    check = cv2.VideoCapture(str(video))
    assert check.isOpened()
    decoded = 0
    while True:
        ok, frame = check.read()
        if not ok:
            break
        assert frame.shape == (1080, 1920, 3)
        decoded += 1
    check.release()
    assert decoded == count == len(mapping) == 401
    assert len(equality_frames) + len(rejected_frames) == count
    assert len(comparisons) == len(sample_frames) and all(c['passed'] for c in comparisons)
    assert all((OUT / f'check_F{f}.jpg').exists() for f in CHECK_FRAMES)
    assert all(b.sha(HERE / name) == digest for name, digest in protected.items())
    assert all(b.sha(OUT / name) == digest for name, digest in output_protected.items())
    for slot in b.SLOTS:
        for status in REJECTED_STATUSES:
            assert dashed_counts[slot][status] == sum(
                p['filter_status'] == status for rows in people.values() for p in rows
                if p['player_slot'] == slot)
    # Keep all previous right-panel rendering statistics, checking them for drift.
    previous = summary.get('rendering', {})
    for key, value in [('drawn_counts_by_player', drawn), ('outside_view_skipped_total', len(outside)),
                       ('outside_view_skipped_points', outside)]:
        if key in previous:
            assert previous[key] == value, key
    dashed_by_player = {slot: sum(c.values()) for slot, c in dashed_counts.items()}
    summary['rendering'] = dict(
        encoded_frames_decoded=decoded, display_fps=fps, width=1920, height=1080,
        x_normalized_limits=list(XLIM), y_normalized_limits=list(YLIM),
        outside_view_skipped_total=len(outside),
        outside_view_skipped_by_player={slot: sum(r['player_slot'] == slot for r in outside) for slot in b.SLOTS},
        outside_view_skipped_points=outside, drawn_counts_by_player=drawn,
        outside_policy='Skip points beyond original view including original 7px circle margin; never clamp',
        existing_files_verified_unchanged=len(protected),
        authorized_modified_existing_script=renderer_name,
        other_existing_output_files_verified_unchanged=output_protected,
        source_images_verified_against_manifest=len(manifest),
        source_image_resolution_policy='Use only the Windows path basename in the registered rally_repair_056/frames directory',
        left_render_policy='Raw observed/observed_after_height_reset: original solid box and name; raw rejected_box_height/rejected_speed: same-color 1px dashed box and name plus 除外, even when final status is interpolated; no raw box: no overlay',
        left_dashed_box_frames_by_player=dashed_by_player,
        left_dashed_box_frames_by_player_and_filter_status=dashed_counts,
        left_dashed_box_with_final_interpolated_by_player=dashed_interpolated,
        frames_with_rejected_boxes=len(rejected_frames),
        source_frames_with_rejected_boxes=rejected_frames,
        all_frames_rendered_both_original_and_cleaned=count,
        no_rejected_box_pixel_equality_asserted_frames=len(equality_frames),
        pixel_equality_excluded_rectangle_xyxy_half_open=list(RIGHT_PANEL),
        no_rejected_box_outside_right_panel_exact_match=True,
        all_frame_title_and_stroke_strip_exact_match=True,
        left_panel_and_stroke_strip_unchanged_before_video_encoding=False,
        original_render_vs_source_video_comparison=dict(
            region='All RGB pixels with x in [0,1320), y in [0,1080)',
            mean_absolute_difference_limit=MAE_LIMIT,
            range_0_to_255=True, sample_count=len(comparisons), samples=comparisons,
            maximum_sample_mean_absolute_difference=max(c['left_rgb_mean_absolute_difference'] for c in comparisons),
            all_samples_passed=True,
            note='Original-style JPEG re-render vs decoded original H.264; small codec differences are expected, not exact pixel equality'),
        existing_routes_and_bounces_reused=True,
        right_player_drawing_counts_unchanged=True,
        existing_ball_and_stroke_drawing_reproduced=True,
        renderer_sha256=b.sha(Path(__file__)),
        output_video_sha256=b.sha(video))
    b.dump(OUT / 'cleaning_summary.json', summary)
    b.dump(OUT / 'verification.json', summary['rendering'])
    readme_path = OUT / 'README.md'
    readme = readme_path.read_text(encoding='utf-8')
    readme = readme.replace(
        '右側コート図の人物点だけを変更した。左側映像・打球表は再エンコード前に画素一致を確認し、'
        '球位置、配球線、バウンドは元の動画・入力を維持。既存ファイルのSHA-256一致を確認した。',
        '右側コート図の人物点を清掃し、左側の選手枠も同じfilter_statusの判定に合わせた。'
        '元JPEGから元の描画を再現し、球位置・軌跡・打球表・バウンド・配球線の入力と描画を維持。'
        '今回の変更を認められたレンダラ以外の既存スクリプト・保護対象ファイルはSHA-256一致を確認した。')
    marker = '\n<!-- left-filter-render-verification -->\n'
    readme = readme.split(marker)[0].rstrip()
    report = [
        '', '左側の枠表示の更新と検証：', '',
        '- observed / observed_after_height_reset：従来の実線枠と名前。',
        '- rejected_box_height / rejected_speed：同じ色の細い1px破線枠と「F2 除外」等の名前。'
        '最終statusがinterpolatedでも生の枠の棄却を示す。',
        '- 生の枠がないmissing、および生の枠がないinterpolated：左側には何も追加しない。',
        '- 右の人物点は従来の最終statusで表示。右下に「左の破線枠＝隠れなどで除いた枠」を追記。',
        '', '|選手|破線枠のフレーム数|高さ棄却|速度棄却|うち最終interpolated|',
        '|---|---:|---:|---:|---:|']
    for slot in b.SLOTS:
        report.append(f"|{b.NAMES[slot]}|{dashed_by_player[slot]}|"
                      f"{dashed_counts[slot]['rejected_box_height']}|"
                      f"{dashed_counts[slot]['rejected_speed']}|{dashed_interpolated[slot]}|")
    report.extend([
        '', f'棄却枠があるのは{len(rejected_frames)}フレーム（選手延べ{sum(dashed_by_player.values())}枠）。',
        f'全{count}フレームで元の全実線画像と変更後の画像をメモリ上で生成。'
        f'棄却枠がない{len(equality_frames)}フレームでは右パネル以外の画素が完全一致することをassertした。'
        'タイトルと打球帯は棄却の有無を問わず全401フレームで完全一致。'
        '右の人物点の状態別描画件数と範囲外省略も既存summaryと一致した。',
        '右パネルの検証除外範囲はx=[1320,1920)、y=[100,921)。追加注記を含み、打球帯は含まない。',
        f'元動画左側との比較は{len(comparisons)}フレーム。元の全実線描画と復号した元動画のRGB平均絶対差を測定した。'
        f'許容値は事前固定の{MAE_LIMIT}未満（0〜255の画素値）。動画の再圧縮による差を含む。',
        '', '|元フレーム|左全体の平均絶対差|映像タイルの平均絶対差|打球帯の平均絶対差|',
        '|---|---:|---:|---:|'])
    for c in comparisons:
        report.append(f"|F{c['source_frame']}|{c['left_rgb_mean_absolute_difference']:.6f}|"
                      f"{c['video_tile_rgb_mean_absolute_difference']:.6f}|"
                      f"{c['stroke_strip_rgb_mean_absolute_difference']:.6f}|")
    report.extend([
        '', 'JPEG401枚のSHA-256はsource_image_manifest.jsonとも再照合済み。'
        '旧絶対パスはファイル名だけ使用し、rally_repair_056/framesから読み込む。',
        f'生成したMP4を最後まで復号し、{decoded}フレーム・1920×1080を確認した。'
        '確認画像はcheck_F154376.jpg、check_F154690.jpg、check_F154762.jpg。'
        '数値・フレーム一覧・保護対象ハッシュはverification.jsonとcleaning_summary.jsonのrenderingを参照。',
        'この検証は表示の整合性を確認するもので、隠れ判定・選手位置・球位置の独立した精度検証ではない。', ''])
    readme_path.write_text(readme + marker + '\n'.join(report), encoding='utf-8')
    print(dict(encoded_frames=decoded, dashed_box_frames_by_player=dashed_by_player,
               frames_with_rejected_boxes=len(rejected_frames),
               pixel_equal_no_rejection_frames=len(equality_frames),
               original_left_max_sample_mae=max(c['left_rgb_mean_absolute_difference'] for c in comparisons)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT,
                        help='Directory produced by clean_056_player_positions.py; rendered files are overwritten.')
    OUT = parser.parse_args().output_dir.resolve()
    assert OUT == (HERE / '056_cleaned_wide_court').resolve(), 'Only the authorized output directory may be changed'
    main()
