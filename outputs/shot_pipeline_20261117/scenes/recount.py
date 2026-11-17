"""Recount existing outputs by provisional/reviewed scene membership; no edits to inputs."""
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PIPELINE = HERE.parent
LOW_Y, HIGH_Y = 11.885, 26.0
LABELS = {"true": "True", "false": "False", "uncertain": "uncertain"}


def normalize(value):
    key = str(value).strip().lower()
    if key not in LABELS:
        raise ValueError(f"Invalid camera label: {value!r}; use True/False/uncertain")
    return LABELS[key]


def effective_scenes(review_path=None):
    scenes = pd.read_csv(HERE/"scenes.csv", dtype={"is_main_camera": str})
    assert scenes.scene_id.is_unique
    scenes["auto_label"] = scenes.is_main_camera.map(normalize)
    scenes["effective_label"] = scenes.auto_label
    scenes["label_source"] = "auto"
    review_path = Path(review_path) if review_path is not None else HERE/"scene_review.csv"
    if review_path.exists():
        review = pd.read_csv(review_path, dtype=str).fillna("")
        assert {"scene_id", "claude_label"} <= set(review.columns)
        assert review.scene_id.is_unique
        unknown = set(review.scene_id) - set(scenes.scene_id)
        if unknown:
            raise ValueError(f"Stale/unknown scene IDs: {unknown}")
        for r in review.itertuples():
            if r.claude_label.strip():
                mask = scenes.scene_id == r.scene_id
                # Guard against reusing reviews after changed segmentation.
                for col in ("start_frame", "end_frame"):
                    value = getattr(r, col, "")
                    if value and int(value) != int(scenes.loc[mask, col].iloc[0]):
                        raise ValueError(f"Stale review boundary: {r.scene_id} {col}")
                scenes.loc[mask, "effective_label"] = normalize(r.claude_label)
                scenes.loc[mask, "label_source"] = "scene_review.csv"
    for point, group in scenes.groupby("point"):
        g = group.sort_values("start_frame")
        assert np.all(g.end_frame.to_numpy()[:-1]+1 == g.start_frame.to_numpy()[1:])
        assert np.all(g.end_frame-g.start_frame+1 == g.n_frames)
    return scenes


def locate(scenes, point, frame):
    found = scenes[(scenes.point == point) & (scenes.start_frame <= frame) & (scenes.end_frame >= frame)]
    if len(found) != 1:
        return dict(scene_id="", camera_class="outside", label_source="outside")
    r = found.iloc[0]
    return dict(scene_id=r.scene_id, camera_class=r.effective_label, label_source=r.label_source)


def attach(data, scenes, frame_col):
    membership = pd.DataFrame([locate(scenes, int(r.point), int(getattr(r, frame_col)))
                               for r in data.itertuples()], index=data.index)
    return pd.concat([data, membership], axis=1)


def save(name, data):
    data.to_csv(HERE/name, index=False, encoding="utf-8-sig", float_format="%.6f")


def recount():
    """Public rerunnable function. Returns summary; reviewed nonblank labels win."""
    scenes = effective_scenes()
    save("effective_scenes.csv", scenes)
    shots = attach(pd.read_csv(PIPELINE/"shot_table.csv"), scenes, "contact_frame")
    shots["high_row"] = (shots.F1_y_m > HIGH_Y) | (shots.F2_y_m > HIGH_Y)
    shots["low_row"] = (shots.F1_y_m < LOW_Y) | (shots.F2_y_m < LOW_Y)
    shots["abnormal_row"] = shots.high_row | shots.low_row
    abnormal_rows = shots[shots.abnormal_row].copy()
    save("abnormal_shot_rows.csv", abnormal_rows)
    positions = []
    for r in shots.itertuples():
        for slot in ("F1", "F2"):
            y = getattr(r, f"{slot}_y_m")
            if pd.notna(y):
                positions.append(dict(point=r.point, rally=r.rally, shot_index=r.shot_index,
                    contact_frame=r.contact_frame, player=slot, y_m=y, high=y>HIGH_Y,
                    low=y<LOW_Y, abnormal=y>HIGH_Y or y<LOW_Y,
                    scene_id=r.scene_id, camera_class=r.camera_class, label_source=r.label_source))
    pos = pd.DataFrame(positions)
    save("abnormal_positions.csv", pos[pos.abnormal])
    audio = attach(pd.read_csv(PIPELINE/"audio/candidates.csv"), scenes, "candidate_frame")
    save("audio_scene_membership.csv", audio)
    labels = attach(pd.concat([pd.read_csv(PIPELINE/f"labels/point_{p:02d}.csv")
                              for p in (7, 8, 9, 10, 11)], ignore_index=True), scenes, "contact_frame")
    save("label_scene_membership.csv", labels)
    save("labels_other_camera.csv", labels[labels.camera_class == "False"])
    save("labels_uncertain_camera.csv", labels[labels.camera_class == "uncertain"])
    # The user's requested closed range is retained. Audio README uses [22776,23032).
    overlap = scenes[(scenes.point == 7) & (scenes.start_frame <= 23032) & (scenes.end_frame >= 22776)].copy()
    overlap["overlap_start"] = overlap.start_frame.clip(lower=22776)
    overlap["overlap_end"] = overlap.end_frame.clip(upper=23032)
    overlap["overlap_frames"] = overlap.overlap_end-overlap.overlap_start+1
    assert overlap.overlap_frames.sum() == 257
    save("point07_rally_A_scenes.csv", overlap)
    rows = []
    for point in (7, 8, 9, 10, 11, "all"):
        def subset(data, camera):
            d = data if point == "all" else data[data.point == point]
            return d if camera == "all" else d[d.camera_class == camera]
        for camera in ("True", "False", "uncertain", "outside", "all"):
            s = subset(shots, camera)
            p = subset(pos, camera)
            a = subset(audio, camera)
            l = subset(labels, camera)
            sc = scenes if point == "all" else scenes[scenes.point == point]
            if camera != "all":
                sc = sc[sc.effective_label == camera]
            rows.append(dict(point=point, camera_class=camera, scenes=len(sc), frames=int(sc.n_frames.sum()),
                shot_rows=len(s), available_far_positions=len(p), abnormal_shot_rows=int(s.abnormal_row.sum()),
                high_shot_rows=int(s.high_row.sum()), low_shot_rows=int(s.low_row.sum()),
                abnormal_positions=int(p.abnormal.sum()), high_positions=int(p.high.sum()), low_positions=int(p.low.sum()),
                audio_candidates=len(a), labels=len(l), clear_labels=int((l.confidence == "clear").sum())))
    summary = pd.DataFrame(rows)
    save("recount_summary.csv", summary)
    assert not (shots.camera_class == "outside").any()
    assert not (audio.camera_class == "outside").any()
    assert not (labels.camera_class == "outside").any()
    update_readme(summary, scenes, overlap, labels)
    print(summary[summary.point == "all"].to_string(index=False), flush=True)
    return summary


def update_readme(summary, scenes, overlap, labels):
    path = HERE/"README.md"
    if not path.exists():
        return
    reviewed = int((scenes.label_source == "scene_review.csv").sum())
    lines = [f"実効判定はレビュー適用 {reviewed} 場面。未修正部分は自動仮判定。",
             "", "| 点 | 場面数 | True F / 全F | True割合 | False F | uncertain F | シート数 |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    index = pd.read_csv(HERE/"scene_sheet_index.csv") if (HERE/"scene_sheet_index.csv").exists() else None
    for point in (7, 8, 9, 10, 11, "all"):
        s = summary[summary.point == point].set_index("camera_class")
        n = int(s.loc["all", "frames"])
        main = int(s.loc["True", "frames"])
        sheets = (0 if index is None else len(index) if point == "all"
                  else len(index[index.point == point]))
        lines.append(f"| {point} | {s.loc['all','scenes']} | {main}/{n} | {main/n:.2%} | "
                     f"{s.loc['False','frames']} | {s.loc['uncertain','frames']} | {sheets} |")
    lines += ["", "| 実効カメラ判定 | 異常打球行（上限超/下限未満） | 異常選手座標 | 音v1候補 | ラベル（clear） |",
              "|---|---:|---:|---:|---:|"]
    for r in summary[summary.point == "all"].itertuples():
        lines.append(f"| {r.camera_class} | {r.abnormal_shot_rows} ({r.high_shot_rows}/{r.low_shot_rows}) | "
                     f"{r.abnormal_positions} | {r.audio_candidates} | {r.labels} ({r.clear_labels}) |")
    lines += ["", "第7点 rally A（閉区間F22776–23032）："]
    for r in overlap.itertuples():
        lines.append(f"- {r.scene_id} F{r.start_frame}–{r.end_frame}、判定 {r.effective_label}、"
                     f"交差 F{r.overlap_start}–{r.overlap_end}（{r.overlap_frames}F）。")
    lines += ["", "ディゾルブ指定フレームの所属（境界の正確さを保証するものではない）："]
    for point, f in ((7,23994),(8,25962)):
        m = locate(scenes, point, f)
        r = scenes[scenes.scene_id == m["scene_id"]].iloc[0]
        lines.append(f"- 第{point}点 F{f}：{r.scene_id} F{r.start_frame}–{r.end_frame}、{r.effective_label}。")
    lines += ["", "26m超は全27打球行。このうち第7点rally Aが4行、rally Bのみで合計23行。"
              "元の『23件』と全ラベルの集計母数はこの4行分異なる。",
              "", "別カメラFalseのラベル一覧は [labels_other_camera.csv](labels_other_camera.csv)、"
              "uncertain一覧は [labels_uncertain_camera.csv](labels_uncertain_camera.csv)。"
              "Trueに残る異常も [abnormal_shot_rows.csv](abnormal_shot_rows.csv) で確認する。"]
    text = path.read_text(encoding="utf-8")
    start, end = "<!-- RECOUNT_RESULTS_START -->", "<!-- RECOUNT_RESULTS_END -->"
    assert start in text and end in text
    prefix, remainder = text.split(start, 1)
    _, suffix = remainder.split(end, 1)
    path.write_text(prefix+start+"\n"+"\n".join(lines)+"\n"+end+suffix, encoding="utf-8")


if __name__ == "__main__":
    recount()
