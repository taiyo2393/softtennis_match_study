"""固定標本と追跡枠を生成。入力は読取専用、出力は隣の data.json のみ。"""
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
SAMPLE_SEED = 1986
ORDER_SEED = 20261118


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main_rows(path, method=None):
    rows = [r for r in read_csv(path)
            if math.isclose(float(r["W_sec"]), .4)
            and math.isclose(float(r["D_m"]), .3)
            and (method is None or r["coordinate_method"] == method)]
    result = {r["id"]: r for r in rows}
    assert len(result) == len(rows), f"主条件のID重複: {path}"
    return result


def probe(path):
    """既存の ffprobe で動画そのものの寸法・フレーム数を照合する。"""
    executable = shutil.which("ffprobe")
    if not executable:
        raise RuntimeError("既存の ffprobe が PATH に必要です。パッケージの追加は行いません。")
    result = subprocess.run(
        [executable, "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,nb_frames,r_frame_rate", "-of", "json", str(path)],
        check=True, capture_output=True, text=True, encoding="utf-8")
    return json.loads(result.stdout)["streams"][0]


def build():
    inputs = [BASE / "net_motion/v2/motions.csv", BASE / "net_motion/review_claude_v1.csv",
              BASE / "label_ui/data.json"]
    rows = main_rows(inputs[0], "ankle")
    reference = read_csv(inputs[1])
    reviewed = [r["id"] for r in reference]
    assert len(reviewed) == len(set(reviewed)) == 20
    assert set(reviewed) <= rows.keys()
    extras = random.Random(SAMPLE_SEED).sample(sorted(rows.keys() - set(reviewed)), 10)
    selected = reviewed + extras
    random.Random(ORDER_SEED).shuffle(selected)
    clip_data = json.loads(inputs[2].read_text(encoding="utf-8"))
    clips = {(int(c["point"]), c["rally"]): c for c in clip_data["clips"]}
    tracks, probes = {}, {}
    items = []
    for ident in selected:
        r = rows[ident]
        point, contact = int(r["point"]), int(r["contact_frame"])
        clip = clips[point, r["rally"]]
        clip_path = BASE / "label_ui" / clip["file"]
        assert clip_path.is_file(), clip_path
        if clip_path not in probes:
            probes[clip_path] = probe(clip_path)
        meta = probes[clip_path]
        fps = clip["fps_num"] / clip["fps_den"]
        num, den = map(int, meta["r_frame_rate"].split("/"))
        assert math.isclose(num / den, fps, rel_tol=1e-9)
        assert int(meta["nb_frames"]) == clip["frame_count"]
        first, count = clip["source_first_frame"], clip["frame_count"]
        start, end = contact - round(.5 * fps), contact + round(.8 * fps)
        assert first <= start <= contact < end <= first + count - 1, (ident, start, end, clip)
        # 時間でも照合（フレーム中心へのシークを含む）。端を切り詰めて通さない。
        assert 0 <= (start - first) / fps and (end - first + .5) / fps < count / fps
        if point not in tracks:
            track_path = BASE / f"tracking/point_{point:02d}/player_court_proxies.csv"
            inputs.append(track_path)
            indexed = {}
            for t in read_csv(track_path):
                key = (int(t["source_frame"]), t["player_slot"])
                assert key not in indexed, f"追跡枠重複: {point}, {key}"
                indexed[key] = t
            tracks[point] = indexed
        boxes = {}
        for frame in range(start, end + 1):
            t = tracks[point].get((frame, r["front_slot"]))
            if t:
                box = [round(float(t[k]), 3) for k in ("x1", "y1", "x2", "y2")]
                assert all(math.isfinite(v) for v in box) and box[0] < box[2] and box[1] < box[3]
                boxes[str(frame)] = box
        assert r["front_side"] in ("N", "F")
        items.append(dict(
            id=ident, point=point, rally=r["rally"], contact_frame=contact,
            front_slot=r["front_slot"], front_side=r["front_side"], end_frame=int(r["end_frame"]),
            auto_classification=r["classification"],
            clip=f"../label_ui/{clip['file']}", source_first_frame=first, frame_count=count,
            fps_num=clip["fps_num"], fps_den=clip["fps_den"],
            video_width=int(meta["width"]), video_height=int(meta["height"]),
            # CSV の箱は canonical の半分 = 1920x1080。動画も同じ寸法であることを検証。
            box_width=1920, box_height=1080,
            start_frame=start, stop_frame=end, marker_04_frame=contact + round(.4 * fps), boxes=boxes))
        assert (int(meta["width"]), int(meta["height"])) == (1920, 1080)
    assert len(items) == len({x["id"] for x in items}) == 30
    assert len(set(selected) & set(reviewed)) == 20
    assert len(set(selected) - set(reviewed)) == 10
    for name in ("index.html", "app.js", "style.css", "README.md", "compare_motion_labels.py"):
        assert (HERE / name).is_file(), name
    hashes = {str(p.relative_to(BASE)).replace("\\", "/"): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in inputs}
    payload = dict(schema_version=1, dataset_id="motion-ui-1986-20261118-v1",
                   sample_seed=SAMPLE_SEED, order_seed=ORDER_SEED,
                   condition=dict(W_sec=.4, D_m=.3, coordinate_method="ankle"),
                   input_sha256=hashes, items=items)
    (HERE / "data.json").write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
    missing = sum(not x["front_slot"] for x in items)
    print(f"生成成功: {len(items)}件 / 20+10件 / 全件の表示範囲・動画実体をassert済み")
    print(f"slot未確定={missing}件（補完せず画面で明示）。抽出seed={SAMPLE_SEED}, 順序seed={ORDER_SEED}")


if __name__ == "__main__":
    build()
