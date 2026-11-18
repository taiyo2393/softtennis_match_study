"""確認式UI用の固定30件を生成。書込み先は motion_ui/data.json のみ。"""
from __future__ import annotations
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
SAMPLE_SEED, ORDER_SEED = 1986, 20261118


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main_rows(path, method=None):
    rows = [r for r in read_csv(path) if math.isclose(float(r["W_sec"]), .4)
            and math.isclose(float(r["D_m"]), .3)
            and (method is None or r["coordinate_method"] == method)]
    result = {r["id"]: r for r in rows}
    assert len(rows) == len(result), path
    return result


def probe(path):
    executable = shutil.which("ffprobe")
    if not executable:
        raise RuntimeError("既存の ffprobe が PATH に必要です")
    result = subprocess.run([executable, "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,nb_frames,r_frame_rate", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, encoding="utf-8")
    return json.loads(result.stdout)["streams"][0]


def prediction(r):
    c = r["classification"]
    return dict(classification=c, motion="motion" if c in ("single", "compound") else c,
        complexity=c if c in ("single", "compound") else "",
        direction8=r["final_direction8"], direction3=r["final_direction3"],
        feint="yes" if r["feint"] == "True" else "no" if c != "unclassifiable" else "unknown",
        front_slot=r["front_slot"], front_side=r["front_side"], end_frame=int(r["end_frame"]),
        issue=r["unclassifiable_reason"])


def build():
    inputs = {BASE / s for s in ("net_motion/motions.csv", "net_motion/v2/motions.csv",
        "net_motion/review_claude_v1.csv", "label_ui/data.json")}
    v1 = main_rows(BASE / "net_motion/motions.csv")
    v2 = main_rows(BASE / "net_motion/v2/motions.csv", "ankle")
    bbox = main_rows(BASE / "net_motion/v2/motions.csv", "bbox")
    refs = {r["id"]: r for r in read_csv(BASE / "net_motion/review_claude_v1.csv")}
    reviewed = list(refs)
    assert len(reviewed) == 20 and set(reviewed) <= v2.keys()
    selected = reviewed + random.Random(SAMPLE_SEED).sample(sorted(v2.keys() - refs.keys()), 10)
    random.Random(ORDER_SEED).shuffle(selected)
    old = HERE / "data.json"
    if old.exists():
        assert [i["id"] for i in json.loads(old.read_text(encoding="utf-8"))["items"]] == selected, "順序変更"
    clips = {(int(c["point"]), c["rally"]): c for c in
             json.loads((BASE / "label_ui/data.json").read_text(encoding="utf-8"))["clips"]}
    tracks, labels, poses, probes, items = {}, {}, {}, {}, []
    for ident in selected:
        r = v2[ident]
        point, contact = int(r["point"]), int(r["contact_frame"])
        clip = clips[point, r["rally"]]
        path = BASE / "label_ui" / clip["file"]
        if path not in probes:
            probes[path] = probe(path)
        meta = probes[path]
        fps = clip["fps_num"] / clip["fps_den"]
        num, den = map(int, meta["r_frame_rate"].split("/"))
        assert math.isclose(num / den, fps, rel_tol=1e-9)
        first, count = clip["source_first_frame"], clip["frame_count"]
        assert int(meta["nb_frames"]) == count
        assert (int(meta["width"]), int(meta["height"])) == (1920, 1080)
        if point not in labels:
            lp = BASE / f"labels/point_{point:02d}.csv"
            tp = BASE / f"tracking/point_{point:02d}/player_court_proxies.csv"
            inputs.update((lp, tp))
            labels[point] = read_csv(lp)
            tracks[point] = {}
            for t in read_csv(tp):
                key = (int(t["source_frame"]), t["player_slot"])
                assert key not in tracks[point]
                tracks[point][key] = t
        shots = sorted((s for s in labels[point] if s["rally"] == r["rally"]), key=lambda s: int(s["shot_index"]))
        at = next(i for i, s in enumerate(shots) if int(s["shot_index"]) == int(r["shot_index"]))
        shot = shots[at]
        assert int(shot["contact_frame"]) == contact
        # 相手側のラベルが連続する例がある。前衛側の最後の既知打球まで戻る。
        # 間の見えない打球を推定して追加しない。
        prev = next((int(s["contact_frame"]) for s in reversed(shots[:at])
                     if s["hitter_side"] == r["front_side"]), None)
        nxt = int(shots[at+1]["contact_frame"]) if at+1 < len(shots) else None
        start = max(first, min(contact-round(fps), prev-round(.2*fps) if prev is not None else contact-round(fps)))
        stop = min(first+count-1, max(contact+round(fps), nxt+round(.2*fps) if nxt is not None else contact+round(fps)))
        assert first <= start <= contact <= stop < first+count, ident
        assert 0 <= (start-first)/fps and (stop-first+.5)/fps < count/fps
        boxes = []
        for frame in range(start, stop+1):
            t = tracks[point].get((frame, r["front_slot"]))
            if t:
                box = [round(float(t[k]), 1) for k in ("x1", "y1", "x2", "y2")]
                assert all(math.isfinite(v) for v in box) and box[0] < box[2] and box[1] < box[3]
                boxes.append([frame, *box])
        pk = (point, r["rally"])
        if pk not in poses:
            pp = BASE / f"pose/pose_cache/point_{point:02d}_rally_{r['rally']}.csv"
            inputs.add(pp)
            pose_rows = read_csv(pp) if pp.exists() else []
            poses[pk] = {(int(t["source_frame"]), t["player_slot"]): t for t in pose_rows}
            assert len(poses[pk]) == len(pose_rows)
        paths = {}
        # 生の画面座標。v2の実際の終点（次打球によるcapを含む）内だけ描く。
        for name, source in (("v1", v1[ident]), ("v2", r)):
            trajectory = []
            for frame in range(contact, int(r["end_frame"])+1):
                t = tracks[point].get((frame, source["front_slot"]))
                xy = None
                if name == "v1" and t:
                    xy = [float(t[k]) for k in ("bottom_center_x", "bottom_center_y")]
                elif name == "v2":
                    p = poses[pk].get((frame, source["front_slot"]))
                    if p:
                        try:
                            ankles = [[float(p[f"kp_{k}_{a}"]) for a in ("x", "y")] for k in (15,16)]
                            if all(float(p[f"kp_{k}_conf"]) >= .3 for k in (15,16)):
                                xy = [(ankles[0][a]+ankles[1][a])/2 for a in (0,1)]
                        except ValueError:
                            pass
                if xy and all(math.isfinite(v) for v in xy):
                    trajectory.append([frame, *[round(v, 1) for v in xy]])
            paths[name] = trajectory
        predictions = dict(v1=prediction(v1[ident]), v2=prediction(r), v2_bbox=prediction(bbox[ident]))
        if ident in refs:
            ref = refs[ident]
            predictions["claude"] = dict(motion=ref["claude_motion"], complexity=ref["claude_single_compound"],
                direction3=ref["claude_direction3"], direction8="", feint="unknown",
                front_slot=v1[ident]["front_slot"], front_side=v1[ident]["front_side"],
                **{k: ref[k] for k in ("claude_motion", "claude_direction3", "claude_single_compound", "claude_issue", "claude_note")})
        items.append(dict(id=ident, point=point, rally=r["rally"], contact_frame=contact,
            contact_confidence=shot["confidence"], previous_contact_frame=prev, next_contact_frame=nxt,
            front_slot=r["front_slot"], front_side=r["front_side"], end_frame=int(r["end_frame"]),
            clip=f"../label_ui/{clip['file']}", source_first_frame=first, frame_count=count,
            fps_num=clip["fps_num"], fps_den=clip["fps_den"], box_width=1920, box_height=1080,
            start_frame=start, stop_frame=stop, boxes=boxes, trajectories=paths, predictions=predictions))
    assert len(items) == len({x["id"] for x in items}) == 30
    assert sum("claude" in i["predictions"] for i in items) == 20
    payload = dict(schema_version=2, dataset_id="motion-review-confirm-1986-20261118-v2",
        sample_seed=SAMPLE_SEED, order_seed=ORDER_SEED, condition=dict(W_sec=.4, D_m=.3),
        trajectory_note="raw screen coordinates; ankle midpoint only, both conf>=0.3; no interpolation; v2 end cap",
        input_sha256={str(p.relative_to(BASE)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
                      if p.exists() else "missing" for p in sorted(inputs)}, items=items)
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n"
    assert len(raw.encode("utf-8")) < 256*1024, "data.json は256KB未満にする"
    (HERE / "data.json").write_bytes(raw.encode("utf-8"))
    print(f"生成成功: 30件（20+10）、順序固定、全範囲clip内assert済み、{len(raw.encode('utf-8')):,} bytes")


if __name__ == "__main__":
    build()
