"""CPU scene segmentation and provisional fixed-main-camera screening (0-based F).

All writes stay below this script's directory. No tracking/label input is edited.
"""
import argparse
import json
import math
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent
ROOT = HERE.parents[2]
POINTS = (7, 8, 9, 10, 11)
SMALL = (160, 90)
LAG = 12
ABRUPT_HIST = 0.28
ABRUPT_GRAY = 0.16
ABRUPT_JOINT_HIST = 0.12
ABRUPT_JOINT_GRAY = 0.085
SLOW_HIST = 0.18
SLOW_GRAY = 0.045
SLOW_GRAY_ONLY = 0.095
GROUP_GAP = 6
MIN_SCENE = 10
CORR_MAIN = 0.92
CORR_OTHER = 0.65
BLUE_MAIN = 0.45
BLUE_OTHER = 0.10
SCENE_MAIN_FRACTION = 0.98
SCENE_OTHER_FRACTION = 0.70
ROWS_PER_SHEET = 10


def write_csv(name, data):
    data.to_csv(HERE / name, index=False, encoding="utf-8-sig", float_format="%.6f")


def intervals():
    inv = pd.read_csv(ROOT / "outputs/match2_point_inventory_20261117/point_inventory.csv")
    return [(int(r.point), int(r.display_from_frame), int(r.next_display_from_frame) - 1)
            for r in inv.itertuples() if r.point in POINTS]


def source_info():
    p = json.loads((PIPELINE / "tracking/point_07/protocol.json").read_text(encoding="utf-8-sig"))
    return Path(p["source_video"]), float(Fraction(p["source_fps"]))


def sequential_frames(source, fps, first, last, targets=None):
    """One initial seek per interval/pass; all subsequent movement uses grab.

    POS_FRAMES is the index of the *next* frame; assert it before/after retrieve.
    targets only suppresses retrieval, never seeks across intervening frames.
    """
    cap = cv2.VideoCapture(str(source))
    try:
        assert cap.isOpened(), source
        assert abs(cap.get(cv2.CAP_PROP_FPS) - fps) < 0.001
        assert cap.set(cv2.CAP_PROP_POS_FRAMES, first)
        assert round(cap.get(cv2.CAP_PROP_POS_FRAMES)) == first
        for frame in range(first, last + 1):
            assert round(cap.get(cv2.CAP_PROP_POS_FRAMES)) == frame
            assert cap.grab(), f"grab F{frame}"
            assert round(cap.get(cv2.CAP_PROP_POS_FRAMES)) == frame + 1
            if targets is None or frame in targets:
                ok, raw = cap.retrieve()
                assert ok and raw is not None, f"retrieve F{frame}"
                assert round(cap.get(cv2.CAP_PROP_POS_FRAMES)) == frame + 1
                assert raw.shape[:2] == (1080, 1920), raw.shape
                yield frame, raw
    finally:
        cap.release()


def features(raw):
    small = cv2.resize(raw, SMALL, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256]).ravel()
    hist /= hist.sum()
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    # Ignore the top 1/6 (scoreboard/crowd) for camera-geometry correlation.
    vec = cv2.resize(gray[15:], (80, 38), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    vec -= vec.mean()
    vec /= max(float(np.linalg.norm(vec)), 1e-8)
    lower = hsv[30:]
    blue = np.mean((lower[:, :, 0] >= 90) & (lower[:, :, 0] <= 130)
                   & (lower[:, :, 1] >= 60) & (lower[:, :, 2] >= 40))
    return hist, gray, vec, float(blue)


def load_references(source, fps):
    old = pd.read_csv(ROOT / "outputs/shot_table_20261117/shot_table.csv")
    refs = [dict(point=int(r.point), frame=int(r.contact_frame_ref),
                 origin="shot_table_4_6") for r in old.itertuples() if r.point in (4, 5, 6)]
    for point in (9, 10):
        labels = pd.read_csv(PIPELINE / f"labels/point_{point:02d}.csv")
        refs += [dict(point=point, frame=int(r.contact_frame), origin="rally_B_clear_9_10")
                 for r in labels.itertuples() if r.rally == "B" and r.confidence == "clear"]
    assert refs and len({r["frame"] for r in refs}) == len(refs)
    by_frame = {r["frame"]: r for r in refs}
    vectors = []
    for frame, raw in sequential_frames(source, fps, min(by_frame), max(by_frame), set(by_frame)):
        _, _, vec, blue = features(raw)
        by_frame[frame]["blue_ratio"] = blue
        vectors.append(vec)
    write_csv("reference_frames.csv", pd.DataFrame([by_frame[f] for f in sorted(by_frame)]))
    return np.stack(vectors)


def extract_metrics(source, fps, refs):
    rows = []
    for point, first, last in intervals():
        started = time.perf_counter()
        history = []
        for frame, raw in sequential_frames(source, fps, first, last):
            hist, gray, vec, blue = features(raw)
            def difference(previous):
                h, g = previous
                return (float(cv2.compareHist(hist, h, cv2.HISTCMP_BHATTACHARYYA)),
                        float(np.mean(cv2.absdiff(gray, g))) / 255)
            h1, g1 = difference(history[-1]) if history else (0., 0.)
            hl, gl = difference(history[-LAG]) if len(history) >= LAG else (0., 0.)
            # einsum avoids multithreaded BLAS launch overhead on tiny vectors.
            corr = float(np.einsum("ij,j->i", refs, vec).max())
            rows.append(dict(point=point, frame=frame, hist_adj=h1, gray_adj=g1,
                             hist_lag=hl, gray_lag=gl, ref_corr=corr, blue_ratio=blue))
            history.append((hist, gray))
            history = history[-LAG:]
        print(f"features point {point}: {last-first+1} frames, {time.perf_counter()-started:.1f}s", flush=True)
    result = pd.DataFrame(rows)
    write_csv("frame_features.csv", result)
    return result


def build_scenes(metrics):
    cuts, scene_rows = [], []
    for point, first, last in intervals():
        data = metrics[metrics.point == point].set_index("frame", drop=False)
        abrupt = ((data.hist_adj >= ABRUPT_HIST) | (data.gray_adj >= ABRUPT_GRAY)
                  | ((data.hist_adj >= ABRUPT_JOINT_HIST) & (data.gray_adj >= ABRUPT_JOINT_GRAY)))
        slow = (((data.hist_lag >= SLOW_HIST) & (data.gray_lag >= SLOW_GRAY))
                | (data.gray_lag >= SLOW_GRAY_ONLY))
        candidates = sorted(data.index[abrupt | slow].tolist())
        groups = []
        for f in candidates:
            if not groups or f - groups[-1][-1] > GROUP_GAP:
                groups.append([f])
            else:
                groups[-1].append(f)
        boundaries = {first: "point_start", last + 1: "point_end"}
        candidate_meta = {}
        for group_id, group in enumerate(groups, 1):
            hard = [f for f in group if abrupt.loc[f]]
            gradual = [f for f in group if slow.loc[f]]
            selected = []
            if hard:
                # Keep each sudden local maximum, even within one slow envelope.
                hard_groups = []
                for f in hard:
                    if not hard_groups or f - hard_groups[-1][-1] > GROUP_GAP:
                        hard_groups.append([f])
                    else:
                        hard_groups[-1].append(f)
                selected += [max(g, key=lambda f: data.loc[f, "hist_adj"] + data.loc[f, "gray_adj"])
                             for g in hard_groups]
            if gradual:
                # Lag evidence starts LAG frames after onset. Preserve the
                # whole uncertain transition as its own scene, not its midpoint.
                selected += [max(first + 1, gradual[0] - LAG + 1), min(last, gradual[-1] + 1)]
            selected = sorted(set(selected))
            kind = "abrupt_and_gradual" if hard and gradual else "abrupt_peak" if hard else "gradual_envelope"
            for b in selected:
                boundaries[b] = kind
            for f in group:
                candidate_meta[f] = dict(group_id=group_id, proposed_boundaries=";".join(map(str, selected)),
                                         group_rule=kind)
        # Merge short scenes deterministically; log every removed boundary.
        removed = {}
        while True:
            bounds = sorted(boundaries)
            short = next((i for i in range(len(bounds)-1) if bounds[i+1]-bounds[i] < MIN_SCENE), None)
            if short is None or len(bounds) == 2:
                break
            i = short
            remove = bounds[i+1] if i == 0 else bounds[i]
            removed[remove] = f"absorbed_short_{bounds[i]}_{bounds[i+1]-1}"
            del boundaries[remove]
        for f, meta in candidate_meta.items():
            r = data.loc[f]
            selected = [int(v) for v in meta["proposed_boundaries"].split(";")]
            cuts.append(dict(point=point, candidate_frame=f, abrupt_candidate=bool(abrupt.loc[f]),
                             gradual_candidate=bool(slow.loc[f]), hist_adj=r.hist_adj, gray_adj=r.gray_adj,
                             hist_lag=r.hist_lag, gray_lag=r.gray_lag, **meta,
                             retained_boundaries=";".join(str(b) for b in selected if b in boundaries),
                             absorption=";".join(f"F{b}:{removed[b]}" for b in selected if b in removed)))
        # Explicit boundary records include point edges and removed boundaries.
        for b, reason in {**boundaries, **removed}.items():
            cuts.append(dict(point=point, candidate_frame=b, group_rule=reason,
                             retained_boundaries=str(b) if b in boundaries else "",
                             absorption=removed.get(b, "")))
        bounds = sorted(boundaries)
        for i, (a, b) in enumerate(zip(bounds, bounds[1:]), 1):
            sample = data.loc[a:b-1]
            main_fraction = float(((sample.ref_corr >= CORR_MAIN) & (sample.blue_ratio >= BLUE_MAIN)).mean())
            other_fraction = float(((sample.ref_corr < CORR_OTHER) | (sample.blue_ratio < BLUE_OTHER)).mean())
            label = ("True" if main_fraction >= SCENE_MAIN_FRACTION else
                     "False" if other_fraction >= SCENE_OTHER_FRACTION else "uncertain")
            scene_rows.append(dict(point=point, scene_id=f"p{point:02d}_s{i:03d}", start_frame=a,
                                   end_frame=b-1, n_frames=b-a, is_main_camera=label,
                                   corr_min=sample.ref_corr.min(), corr_p10=sample.ref_corr.quantile(.1),
                                   corr_median=sample.ref_corr.median(), corr_max=sample.ref_corr.max(),
                                   blue_p10=sample.blue_ratio.quantile(.1), blue_median=sample.blue_ratio.median(),
                                   main_frame_fraction=main_fraction, other_frame_fraction=other_fraction,
                                   start_reason=boundaries[a]))
        assert sum(s["n_frames"] for s in scene_rows if s["point"] == point) == last-first+1
    scenes = pd.DataFrame(scene_rows)
    write_csv("scene_cuts.csv", pd.DataFrame(cuts))
    write_csv("scenes.csv", scenes)
    template = scenes[["point", "scene_id", "start_frame", "end_frame", "is_main_camera"]].rename(columns={"is_main_camera": "auto_label"})
    template["claude_label"] = ""
    template["review_note"] = ""
    # A manually edited scene_review.csv is never overwritten.
    write_csv("scene_review_template.csv", template)
    return scenes


def make_sheets(source, fps, scenes):
    folder = HERE / "scene_sheets"
    folder.mkdir(exist_ok=True)
    font = ImageFont.truetype("C:/Windows/Fonts/meiryo.ttc", 15)
    index = []
    expected_files = set()
    for point, first, last in intervals():
        rows = scenes[scenes.point == point]
        targets = set()
        for r in rows.itertuples():
            targets.update((r.start_frame, (r.start_frame+r.end_frame)//2, r.end_frame))
        thumbs = {}
        for f, raw in sequential_frames(source, fps, first, last, targets):
            small = cv2.resize(raw, (320, 180), interpolation=cv2.INTER_AREA)
            thumbs[f] = Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
        for offset in range(0, len(rows), ROWS_PER_SHEET):
            block = rows.iloc[offset:offset+ROWS_PER_SHEET]
            sheet = Image.new("RGB", (1220, 210*len(block)), (22, 28, 36))
            draw = ImageDraw.Draw(sheet)
            for j, r in enumerate(block.itertuples()):
                y = j*210
                text = (f"{r.scene_id} / point {point}\nF{r.start_frame}-{r.end_frame}\n"
                        f"{r.n_frames} frames\nauto: {r.is_main_camera}\n"
                        f"corr med {r.corr_median:.3f}\nblue med {r.blue_median:.3f}")
                draw.multiline_text((8, y+8), text, font=font, fill="white", spacing=3)
                for k, f in enumerate((r.start_frame, (r.start_frame+r.end_frame)//2, r.end_frame)):
                    x = 260 + 320*k
                    draw.text((x+5, y+2), f"F{f}", font=font, fill="white")
                    sheet.paste(thumbs[f], (x, y+27))
            name = f"point_{point:02d}_sheet_{offset//ROWS_PER_SHEET+1:02d}.jpg"
            sheet.save(folder/name, quality=90)
            expected_files.add(name)
            index.append(dict(point=point, sheet=name, scenes=len(block),
                              scene_ids=";".join(block.scene_id)))
    for old in folder.glob("point_??_sheet_??.jpg"):
        if old.name not in expected_files:
            old.unlink()  # Only obsolete sheets generated by this script.
    write_csv("scene_sheet_index.csv", pd.DataFrame(index))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reuse-features", action="store_true", help="reuse generated CSV; skip source feature pass")
    args = parser.parse_args()
    source, fps = source_info()
    refs = load_references(source, fps) if not args.reuse_features else None
    metrics = (pd.read_csv(HERE/"frame_features.csv") if args.reuse_features
               else extract_metrics(source, fps, refs))
    for point, first, last in intervals():
        assert metrics.loc[metrics.point == point, "frame"].tolist() == list(range(first, last+1))
    scenes = build_scenes(metrics)
    make_sheets(source, fps, scenes)
    settings = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, tuple))}
    (HERE/"detection_protocol.json").write_text(json.dumps(dict(source_video=str(source), source_fps=fps,
        frame_index="zero_based_inclusive", reference_rule="points4-6 contact_frame_ref + points9-10 rallyB clear",
        settings=settings, reused_features=args.reuse_features), indent=2), encoding="utf-8")
    import recount
    recount.recount()
    for point, group in scenes.groupby("point"):
        main_frames = int(group.loc[group.is_main_camera == "True", "n_frames"].sum())
        print(f"point {point}: {len(group)} scenes, main {main_frames}/{group.n_frames.sum()} "
              f"({main_frames/group.n_frames.sum():.1%}), sheets {math.ceil(len(group)/ROWS_PER_SHEET)}", flush=True)


if __name__ == "__main__":
    main()
