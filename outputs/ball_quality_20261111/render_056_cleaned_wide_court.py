"""Render the existing wide-court view with filtered player points only."""
from pathlib import Path
import argparse
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import build_044_spatiotemporal_pilot as b
from render_044_estimated_routes import dashed, diamond, YELLOW, CYAN, ORANGE
from render_056_wide_court_view import XLIM, YLIM
from clean_056_player_positions import STATUSES, OBSERVED_STATUSES

HERE = Path(__file__).resolve().parent
PARENT = HERE / '056_reviewed_consolidated'
OUT = HERE / '056_cleaned_wide_court'
CHECK_FRAMES = (154376, 154690, 154762)


def main():
    assert OUT.is_dir(), 'Run clean_056_player_positions.py first'
    video = OUT / '056_cleaned_wide_court.mp4'
    assert not video.exists(), f'Refusing to overwrite {video}'
    before = b.load(OUT / 'existing_files_sha256_before.json')
    assert all(b.sha(HERE / name) == digest for name, digest in before.items())
    summary = b.load(OUT / 'cleaning_summary.json')
    people = b.grouped(b.read(OUT / 'cleaned_player_positions.csv'))
    strokes = b.read(PARENT / 'stroke_order_confirmed.csv')
    bounces = b.read(PARENT / 'bounce_court_points.csv')
    route_rows = b.read(PARENT / 'schematic_court_routes.csv')
    routes = {s['stroke_id']: [r for r in route_rows if r['stroke_id'] == s['stroke_id']] for s in strokes}
    mapping = b.read(PARENT / 'video_source_frame_map.csv')
    plot_top, plot_height, center_x = 200., 630., 1620.
    height = plot_height / (YLIM[1] - YLIM[0])
    width = height * 250 / 540
    court_left = center_x - width / 2
    court_top = plot_top + (YLIM[1] - 1) * height

    def cp(u, v):
        return court_left + width * u, court_top + height * (1 - v)

    view_left, view_top = cp(XLIM[0], YLIM[1])
    view_right, view_bottom = cp(XLIM[1], YLIM[0])
    fonts = {s: ImageFont.truetype(str(b.FONT), s) for s in [16, 18, 20, 30]}
    cap = cv2.VideoCapture(str(PARENT / '056_reviewed_players_ball_and_court.mp4'))
    assert cap.isOpened()
    fps = cap.get(cv2.CAP_PROP_FPS)
    proc = subprocess.Popen([
        'C:/ffmpeg/bin/ffmpeg.exe', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
        '-s', '1920x1080', '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
        '-preset', 'fast', '-crf', '17', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)],
        stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    outside = []
    drawn = {slot: {status: 0 for status in (*OBSERVED_STATUSES, 'interpolated')} for slot in b.SLOTS}
    count = 0
    try:
        for row in mapping:
            ok, frame = cap.read()
            assert ok
            f = int(row['source_frame'])
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            d = ImageDraw.Draw(image)
            d.rectangle((1320, 100, 1919, 900), fill=(12, 22, 32))

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
            txt(1335, 847, '人物点は枠下端の代理位置。隠れ等による異常点を除外。', 16)
            txt(1335, 875, '採用点間30元フレーム以内は補間（白抜き、約0.5秒）。', 16)
            if not frame_drawn:
                txt(1335, 815, '人物：採用・補間点なし', 16, YELLOW)
            array = np.asarray(image)
            assert np.array_equal(array[:, :1320], rgb[:, :1320])
            assert np.array_equal(array[:100], rgb[:100])
            assert np.array_equal(array[901:], rgb[901:])
            proc.stdin.write(array.tobytes())
            count += 1
            if f in CHECK_FRAMES:
                image.save(OUT / f'check_F{f}.jpg', quality=95)
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
    decoded = 0
    while True:
        ok, frame = check.read()
        if not ok:
            break
        assert frame.shape == (1080, 1920, 3)
        decoded += 1
    check.release()
    assert decoded == count == len(mapping) == 401
    assert all((OUT / f'check_F{f}.jpg').exists() for f in CHECK_FRAMES)
    assert all(b.sha(HERE / name) == digest for name, digest in before.items())
    summary['rendering'] = dict(
        encoded_frames_decoded=decoded, display_fps=fps,
        x_normalized_limits=list(XLIM), y_normalized_limits=list(YLIM),
        outside_view_skipped_total=len(outside),
        outside_view_skipped_by_player={slot: sum(r['player_slot'] == slot for r in outside) for slot in b.SLOTS},
        outside_view_skipped_points=outside, drawn_counts_by_player=drawn,
        outside_policy='Skip points beyond original view including original 7px circle margin; never clamp',
        existing_files_verified_unchanged=len(before),
        left_panel_and_stroke_strip_unchanged_before_video_encoding=True,
        existing_routes_and_bounces_reused=True)
    b.dump(OUT / 'cleaning_summary.json', summary)
    table = ['|選手|' + '|'.join(STATUSES) + '|',
             '|---|' + '---:|' * len(STATUSES)]
    for slot in b.SLOTS:
        c = summary['status_counts_by_player'][slot]
        table.append('|' + b.NAMES[slot] + ' (' + slot + ')|' + '|'.join(str(c[s]) for s in STATUSES) + '|')
    readme = (
        '# 056：隠れによる選手代理位置の飛びを抑制\n\n'
        '右側コート図の人物点だけを変更した。左側映像・打球表は再エンコード前に画素一致を確認し、'
        '球位置、配球線、バウンドは元の動画・入力を維持。既存ファイルのSHA-256一致を確認した。'
        '動画は1920×1080、401フレーム。再圧縮による画素差は生じうる。\n\n'
        '4選手に同じ規則を適用。枠高さが直前の採用点（最大15点）の中央値の0.75倍未満・1.35倍超なら棄却。'
        '採用が5点に達するまでは当該選手の入力全体の中央値を参照する。'
        'ただし、高さ棄却が生の観測点の順序で連続し、先頭と現在点の差が30元フレームを超えたら、'
        'その区間を先頭から再判定する（ちょうど30では戻さない。元から欠測の点は連続を切らず、復帰対象にしない）。'
        '区間の高さの中央値をリセット時の参照値とし、古い高さ履歴を破棄して復帰点の高さ（直近最大15点）で作り直す。'
        'その後の点は新しい基準で通常判定する。復帰点はobserved_after_height_resetとする。'
        '復帰時も直前の採用点からの速さ規則を維持し、さらに直前の生の点からの移動が同じ速さ規則を破る点を'
        'rejected_speedにする（区間への入り口および異常点から戻る移動も検査）。'
        '復帰できる点がない場合にも区間の高さで基準を更新し、速度判定は継続する。'
        '直前の採用点からの距離が8 m/s×経過秒＋1.0 mを超える場合も棄却（高さ判定を優先）。'
        '元動画59.94 fps、メートル換算はx×10.97、y×23.77。'
        '両端の採用点が30元フレーム以内（約0.5005秒）の欠けだけを表示フレーム上で線形補間する。'
        '端点のない区間は補わない。\n\n'
        '塗りつぶし円は採用点、同色の白抜き円は補間点。棄却・欠測・表示範囲外の点は描かない。'
        f'表示範囲は元と同じx={XLIM}、y={YLIM}。範囲外等による描画省略は{len(outside)}点。\n\n'
        'CSVのstatusは補間後の表示状態。filter_status（および互換用input_status）にリセット後・補間前の判定を残し、'
        'rejected_speedの点も既存の短区間補間規則を満たせば最終statusはinterpolatedとなる。'
        'raw_previous_frame、raw_jump_m、raw_speed_limit_mは生の点どうしの移動の監査用で、'
        'height_reset_start_frame/trigger_frameは再判定した区間を示す。'
        'court_proxy_x/yは入力をそのまま保存した。件数は各選手401行、計1604行について集計。\n\n'
        f"高さリセットによる復帰は合計{summary['height_reset_restored_points_total']}点。"
        + '、'.join(f"{b.NAMES[slot]}：{summary['height_reset_restored_points_by_player'][slot]}点"
                   for slot in b.SLOTS) + '。区間の開始・発動フレームと速度棄却点はsummaryのheight_resets_by_playerに記録した。\n\n'
        + '\n'.join(table) + '\n\n'
        '限界：補間点は観測ではない。両端の採用点が30元フレームより離れた欠測は補間しない。'
        '長い高さ棄却を大きさの変化と扱う規則は仮定であり、長い隠れも復帰させる可能性がある。'
        '棄却した行のstatusは棄却理由のまま、元から欠測の行はmissingのまま残す。'
        '足元の位置精度は検証していない。隠れの検出器ではなく、枠高さと移動量によるヒューリスティックであり、'
        '正常点の棄却や異常点の通過、ID誤りの残存はありうる。速度閾値は推定された真の走速度を保証しない。\n')
    (OUT / 'README.md').write_text(readme, encoding='utf-8')
    shutil.copy2(PARENT / 'video_source_frame_map.csv', OUT / 'video_source_frame_map.csv')
    b.dump(OUT / 'verification.json', summary['rendering'])
    print(summary['rendering'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=OUT,
                        help='Directory produced by clean_056_player_positions.py.')
    OUT = parser.parse_args().output_dir.resolve()
    main()
