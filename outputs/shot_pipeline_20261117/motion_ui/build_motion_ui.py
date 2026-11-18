"""本人の32打球すべてから確認画面を生成。書込みは motion_ui/data.json のみ。"""
from __future__ import annotations
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess

import numpy as np  # study/.venv に導入済み

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
ROOT = BASE.parent.parent
ORDER_SEED = 20261118
METHODS = ('bbox', 'ankle')
CAL = ROOT / 'outputs/ball_quality_20261110/match2_camera_transfer_pilot/protocol.json'


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def probe(path):
    executable = shutil.which('ffprobe')
    if not executable:
        raise RuntimeError('既存の ffprobe が PATH に必要です')
    result = subprocess.run([executable, '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,nb_frames,r_frame_rate', '-of', 'json', str(path)],
        check=True, capture_output=True, text=True, encoding='utf-8')
    return json.loads(result.stdout)['streams'][0]


def prediction(r):
    c = r['classification']
    return dict(classification=c, motion='motion' if c in ('single', 'compound') else c,
        complexity=c if c in ('single', 'compound') else '',
        direction8=r['final_direction8'], direction3=r['final_direction3'],
        feint='unknown' if c == 'unclassifiable' else 'yes' if r['feint'] == 'True' else 'no',
        front_slot=r['front_slot'], front_side=r['front_side'], end_frame=int(r['end_frame']),
        issue=r['unclassifiable_reason'], uses_bbox_fallback=r['uses_bbox_fallback'] == 'True',
        foot_source_counts=json.loads(r['foot_source_counts_json']))


def role_note(r):
    if r['receiver_basis'] == 'farthest_net_at_next_contact':
        return 'サーブ：次の打球の側でネットから遠い方をレシーバーとみなし、もう一方を前衛とした（レシーバー未確認）。'
    if r['role_issue'] == 'serve_next_receiving_contact_missing':
        return 'サーブ：次の相手側の打球がないため、レシーバーと前衛を選択できない。'
    if r['role_issue'] == 'serve_receiver_positions_missing':
        return 'サーブ：次の打球時の位置が不足し、レシーバーと前衛を選択できない。'
    assert r['receiver_basis'] == 'not_serve', r
    note = '通常の打球：相手の打球時、守備側でネットに近い方を前衛とした。'
    if r['role_ambiguous'] == 'True':
        note += ' 2人の距離が近く、選択は曖昧。'
    if r['role_issue']:
        note += ' 注意：' + r['role_issue']
    return note


def build():
    human_path = BASE / 'labels_human/human_labels_20261118.csv'
    motion_path = BASE / 'v_human/net_motion/motions_main.csv'
    if not motion_path.exists():
        motion_path = BASE / 'v_human/net_motion/motions.csv'
    trajectory_path = BASE / 'v_human/net_motion/trajectories_main.csv'
    role_path = BASE / 'v_human/net_motion/role_selection.csv'
    paper_path = BASE / 'v_human/net_motion/paper_comparison.csv'
    clip_path = BASE / 'label_ui/data.json'
    inputs = {human_path, motion_path, trajectory_path, role_path, paper_path, clip_path, CAL}
    human = [r for r in read_csv(human_path) if r['condition'] == 'without_candidates']
    assert len(human) == 32 and all(r['status'] == 'completed' for r in human)
    shots = {}
    for r in human:
        key = (int(r['point']), int(r['rally_no']))
        shots.setdefault(key, []).append(r)
    for g in shots.values():
        g.sort(key=lambda r: int(r['frame']))
        assert len({r['frame'] for r in g}) == len(g)
    human_by_key = {(int(r['point']), int(r['frame'])): r for r in human}
    assert len(human_by_key) == len(human)
    motions = {m: {} for m in METHODS}
    for r in read_csv(motion_path):
        if not (math.isclose(float(r['W_sec']), .4) and math.isclose(float(r['D_m']), .3)):
            continue
        method = r['coordinate_method']
        assert method in METHODS and r['id'] not in motions[method]
        h = human_by_key[int(r['point']), int(r['contact_frame'])]
        assert (int(r['rally_no']), r['hitter_side'], r['is_serve'].lower()) == (int(h['rally_no']), h['side'], h['is_serve'].lower())
        group = shots[int(h['point']), int(h['rally_no'])]
        assert r['id'] == f"p{int(h['point']):02d}_{h['rally_no']}_{group.index(h)+1:02d}"
        motions[method][r['id']] = r
    selected = sorted(motions['bbox'])
    assert len(selected) == len(human) and set(selected) == set(motions['ankle'])
    assert {(int(r['point']), int(r['contact_frame'])) for r in motions['bbox'].values()} == set(human_by_key)
    random.Random(ORDER_SEED).shuffle(selected)
    roles = {(int(r['point']), int(r['contact_frame'])): r for r in read_csv(role_path)}
    assert len(roles) == len(human) and set(roles) == set(human_by_key)
    clips = {int(c['point']): c for c in json.loads(clip_path.read_text(encoding='utf-8'))['clips']}
    # Inverse-project the stored raw court metres to native video pixels.
    cal = json.loads(CAL.read_text(encoding='utf-8-sig'))
    assert cal['canonical_dimensions'] == [3840, 2160]
    inverse = np.linalg.inv(np.asarray(cal['image_to_court_homography_canonical'], float))
    trajectory, seen = {}, set()
    for t in read_csv(trajectory_path):
        key = (t['id'], t['coordinate_method'])
        unique = (*key, int(t['source_frame']))
        assert unique not in seen and key[0] in motions[key[1]]
        seen.add(unique)
        trajectory.setdefault(key, []).append(t)
    tracks, probes, items = {}, {}, []
    for ident in selected:
        r = motions['bbox'][ident]
        point, contact = int(r['point']), int(r['contact_frame'])
        role = roles[point, contact]
        for method in METHODS:
            p = motions[method][ident]
            assert (p['front_slot'], p['front_side']) == (role['front_slot'], role['front_side'])
        clip = clips[point]
        assert clip['file'] == f'clips/point_{point:02d}.mp4' and clip['rally'] == 'all'
        path = BASE / 'label_ui' / clip['file']
        if path not in probes:
            probes[path] = probe(path)
        meta = probes[path]
        fps = clip['fps_num'] / clip['fps_den']
        num, den = map(int, meta['r_frame_rate'].split('/'))
        assert math.isclose(num / den, fps, rel_tol=1e-9)
        first, count = clip['source_first_frame'], clip['frame_count']
        assert int(meta['nb_frames']) == count
        assert (int(meta['width']), int(meta['height'])) == (1920, 1080)
        if point not in tracks:
            tp = BASE / f'tracking/point_{point:02d}/player_court_proxies.csv'
            inputs.add(tp)
            tracks[point] = {}
            for t in read_csv(tp):
                key = (int(t['source_frame']), t['player_slot'])
                assert key not in tracks[point]
                tracks[point][key] = t
        group = shots[point, int(r['rally_no'])]
        at = next(i for i, h in enumerate(group) if int(h['frame']) == contact)
        prev = next((int(h['frame']) for h in reversed(group[:at]) if h['side'] == r['front_side']), None)
        nxt = int(group[at+1]['frame']) if at+1 < len(group) else None
        start = max(first, min(contact-round(fps), prev-round(.2*fps) if prev is not None else contact-round(fps)))
        end = max(int(motions[m][ident]['end_frame']) for m in METHODS)
        stop = min(first+count-1, max(contact+round(fps), end, nxt+round(.2*fps) if nxt is not None else contact+round(fps)))
        assert first <= start <= contact <= end <= stop < first+count, ident
        assert 0 <= (start-first)/fps and (stop-first+.5)/fps < count/fps, ident
        boxes = []
        for frame in range(start, stop+1):
            t = tracks[point].get((frame, r['front_slot']))
            if t:
                box = [round(float(t[k])) for k in ('x1', 'y1', 'x2', 'y2')]
                assert all(math.isfinite(v) for v in box) and box[0] < box[2] and box[1] < box[3]
                boxes.append([frame, *box])
        paths, foot_counts = {}, {}
        for method in METHODS:
            points, feet = [], Counter()
            p = motions[method][ident]
            for t in sorted(trajectory.get((ident, method), []), key=lambda t: int(t['source_frame'])):
                frame = int(t['source_frame'])
                assert t['front_slot'] == r['front_slot']
                if not contact <= frame <= int(p['end_frame']):
                    continue
                feet[t['foot_source'] or 'missing'] += 1
                xy = [float(t[k]) if t[k] else math.nan for k in ('raw_x_m', 'raw_y_m')]
                if not all(math.isfinite(v) for v in xy):
                    continue
                pixel = inverse @ [xy[0]/10.97, xy[1]/23.77, 1]
                assert abs(pixel[2]) > 1e-9
                pixel = pixel[:2]/pixel[2]/2
                assert np.isfinite(pixel).all()
                # Box pixel coordinates independently guard scale/axis mistakes.
                if t['foot_source'].startswith('bbox'):
                    track = tracks[point][frame, r['front_slot']]
                    np.testing.assert_allclose(pixel, [float(track[k]) for k in ('bottom_center_x', 'bottom_center_y')], rtol=0, atol=1e-5)
                points.append([frame, *[round(float(v), 1) for v in pixel]])
            paths[method] = points
            foot_counts[method] = dict(feet)
        items.append(dict(id=ident, point=point, rally=int(r['rally_no']), contact_frame=contact,
            previous_contact_frame=prev, next_contact_frame=nxt,
            front_slot=r['front_slot'], front_side=r['front_side'], end_frame=end,
            role_note=role_note(role), role_selection=role, exclude_p8_near=role['exclude_p8_near'] == 'True',
            clip=f"../label_ui/{clip['file']}", source_first_frame=first, frame_count=count,
            fps_num=clip['fps_num'], fps_den=clip['fps_den'], box_width=1920, box_height=1080,
            start_frame=start, stop_frame=stop, boxes=boxes, trajectories=paths,
            trajectory_foot_sources=foot_counts,
            predictions={m: prediction(motions[m][ident]) for m in METHODS}))
    paper = {}
    for p in read_csv(paper_path):
        key = (p['metric'], p['category'])
        value = dict(metric=key[0], category=key[1], category_ja=p['category_ja'], percent=float(p['paper_percent']))
        assert key not in paper or paper[key] == value
        paper[key] = value
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(inputs)}
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    payload = dict(schema_version=3, dataset_id=f'motion-review-human-20261118-{digest}',
        label_condition='without_candidates', order_seed=ORDER_SEED, condition=dict(W_sec=.4, D_m=.3),
        trajectory_note='trajectories_main raw court metres inverse-projected to native pixels; missing frames not interpolated; ankle may use bbox fallback',
        paper_reference=dict(source='v_human/net_motion/paper_comparison.csv',
            provenance='inherited_v2_PAPER_not_rechecked_against_original', rows=list(paper.values())),
        input_sha256=hashes, items=items)
    assert len(items) == len({i['id'] for i in items}) == 32
    raw = (json.dumps(payload, ensure_ascii=False, separators=(',', ':'))+'\n').encode('utf-8')
    assert len(raw) < 256_000, f'data.json は256KB未満にする: {len(raw)} bytes'
    (HERE/'data.json').write_bytes(raw)
    counts = {m: dict(Counter(i['predictions'][m]['classification'] for i in items)) for m in METHODS}
    print(f'生成成功: 全{len(items)}件、seed={ORDER_SEED}、全範囲clip内assert済み、{len(raw):,} bytes')
    print(json.dumps(counts, ensure_ascii=False))


if __name__ == '__main__':
    build()
