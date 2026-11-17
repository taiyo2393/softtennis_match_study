"""Join existing Claude labels and player proxies; never rerun upstream processing.

Run with study/.venv/Scripts/python.exe. Only the four documented outputs in
this directory are written. Input label cells, including blanks, stay intact.
"""

import csv
import hashlib
import io
import json
import math
import re
from collections import Counter
from datetime import datetime
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
POINTS = (7, 8, 9, 10, 11)
EXPECTED_LABEL_ROWS = 91
TARGET_POINTS = 66
MAX_FRAME_OFFSET = 3
WIDTH_M, LENGTH_M = 10.97, 23.77
NET_Y_M = 11.885
# Same constants/formula as outputs/shot_table_20261117/build_match2_shot_table.py:
# x_m = court_proxy_x * 10.97, y_m = court_proxy_y * 23.77. Do not clamp.
SLOTS = {
    "N1": "near_front", "N2": "near_rear",
    "F1": "far_rear", "F2": "far_front",
}
SIDE_SLOTS = {"N": ("N1", "N2"), "F": ("F1", "F2")}
PAIR_NAMES = {"U": "内本・内田", "F": "船水・上松"}
# Evidence: point_07 final note says far=内本・内田, point_08 final note
# says near=船水・上松. In the previous points 4--6 table, point_winner_side
# is far/near/near, while inventory point_winner is U/F/F: same mapping.
# point_11 first note explicitly says ends change and near=内本・内田.
# Keep these assignments explicit; do not infer sides from a point's winner.
POINT_CONTEXT = {
    7: (1, "F", "U"), 8: (1, "F", "U"),
    9: (1, "F", "U"), 10: (1, "F", "U"), 11: (2, "U", "F"),
}
LABEL_COLUMNS = [
    "point", "rally", "shot_index", "contact_frame", "hitter_side",
    "hitter_slot", "is_serve", "shot_outcome", "confidence", "tracking_issue", "note",
]
ISSUES = ("wrong_person", "missing", "swapped")
ADDED_COLUMNS = [
    "game", "near_pair", "far_pair", "hitter_pair", "point_winner", "hitter_won_point",
    *[f"{slot}_{axis}_m" for slot in SLOTS for axis in ("x", "y")],
    *[f"{slot}_frame_offset" for slot in SLOTS],
    "players_available", "near_positions_trusted", "far_positions_trusted",
    "hitter_pair_lateral_m", "hitter_pair_depth_m",
    "opponent_pair_lateral_m", "opponent_pair_depth_m",
]
RESULTS_BEGIN = "<!-- BEGIN GENERATED SHOT RESULTS -->"
RESULTS_END = "<!-- END GENERATED SHOT RESULTS -->"


class Inputs:
    """Read inputs once and retain hashes for provenance and mutation checks."""

    def __init__(self):
        self.hashes = {}

    def text(self, path):
        path = Path(path)
        data = path.read_bytes()
        self.hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(data).hexdigest()
        return data.decode("utf-8-sig")

    def csv(self, path, required=()):
        reader = csv.DictReader(io.StringIO(self.text(path)))
        if not set(required).issubset(reader.fieldnames or []):
            raise ValueError(f"Missing CSV columns: {path}: {required}")
        rows = list(reader)
        if any(None in row or None in row.values() for row in rows):
            raise ValueError(f"Malformed CSV row: {path}")
        return rows

    def by_point(self, path):
        rows = self.csv(path, ("point",))
        result = {int(row["point"]): row for row in rows}
        assert len(result) == len(rows), f"Duplicate points: {path}"
        assert set(POINTS) <= result.keys(), f"Missing target points: {path}"
        return result

    def assert_unchanged(self):
        for relative, expected in self.hashes.items():
            assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected, relative


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def seconds(value):
    value = float(value)
    assert math.isfinite(value) and value >= 0
    return value


def position_index(inputs, point, summary):
    folder = HERE / "tracking" / f"point_{point:02d}"
    availability = inputs.csv(folder / "player_frame_availability.csv", (
        "source_frame", "camera_line_supported", "assigned_slots",
    ))
    frames = [int(row["source_frame"]) for row in availability]
    first, last = int(summary["first_frame"]), int(summary["last_frame"])
    assert frames == list(range(first, last + 1)), f"Availability frame gap/duplicate: {point}"
    assert len(frames) == int(summary["interval_frames"])
    supported = [row for row in availability if int(row["camera_line_supported"]) == 1]
    assert len(supported) == int(summary["camera_line_supported_frames"])
    four = sum(int(row["assigned_slots"]) == 4 for row in supported)
    assert four == int(summary["four_player_frames"])
    expected_ratio = fraction(four, len(supported))
    if expected_ratio is not None:
        assert abs(float(summary["four_player_ratio"]) - expected_ratio) <= 0.00000051
    assigned = {int(row["source_frame"]): int(row["assigned_slots"]) for row in availability}
    proxies = inputs.csv(folder / "player_court_proxies.csv", (
        "source_frame", "player_slot", "court_proxy_x", "court_proxy_y",
    ))
    by_slot = {slot: {} for slot in SLOTS}
    reverse = {long: short for short, long in SLOTS.items()}
    seen, proxy_counts = set(), Counter()
    for proxy in proxies:
        frame, slot = int(proxy["source_frame"]), reverse[proxy["player_slot"]]
        assert first <= frame <= last
        assert (frame, slot) not in seen, f"Duplicate position: {point}, {frame}, {slot}"
        seen.add((frame, slot))
        proxy_counts[frame] += 1
        if not proxy["court_proxy_x"] or not proxy["court_proxy_y"]:
            continue
        u, v = float(proxy["court_proxy_x"]), float(proxy["court_proxy_y"])
        if math.isfinite(u) and math.isfinite(v):
            by_slot[slot][frame] = (u * WIDTH_M, v * LENGTH_M)
    assert all(proxy_counts[frame] == assigned[frame] for frame in frames), point
    return by_slot


def nearest_position(index, contact_frame):
    if contact_frame is None:
        return None
    candidates = [f for f in range(contact_frame - MAX_FRAME_OFFSET,
                                   contact_frame + MAX_FRAME_OFFSET + 1) if f in index]
    if not candidates:
        return None
    # Inclusive +/-3 search, ties earlier. No interpolation/extrapolation.
    frame = min(candidates, key=lambda f: (abs(f - contact_frame), f))
    return (*index[frame], frame - contact_frame)


def pair_metrics(row, side):
    trusted = row["near_positions_trusted" if side == "N" else "far_positions_trusted"]
    a, b = SIDE_SLOTS[side]
    if not trusted or row[f"{a}_x_m"] is None or row[f"{b}_x_m"] is None:
        return None, None
    return (abs(row[f"{a}_x_m"] - row[f"{b}_x_m"]),
            abs(row[f"{a}_y_m"] - row[f"{b}_y_m"]))


def build_shots(inputs, inventory, tracking_summary):
    shots, labels = [], []
    for point in POINTS:
        by_slot = position_index(inputs, point, tracking_summary[point])
        point_labels = inputs.csv(HERE / "labels" / f"point_{point:02d}.csv", LABEL_COLUMNS)
        game, near, far = POINT_CONTEXT[point]
        assert int(inventory[point]["game"]) == game
        winner = inventory[point]["point_winner"]
        assert winner in PAIR_NAMES
        seen = set()
        for label in point_labels:
            assert list(label) == LABEL_COLUMNS, "Preserve every original label column"
            assert int(label["point"]) == point
            key = (label["rally"], int(label["shot_index"]))
            assert key not in seen, f"Duplicate shot: {point}, {key}"
            seen.add(key)
            side = label["hitter_side"]
            assert side in SIDE_SLOTS
            assert label["hitter_slot"] in ("", *SIDE_SLOTS[side])
            assert label["confidence"] in ("clear", "inferred")
            assert label["tracking_issue"] in ("", *ISSUES)
            assert label["shot_outcome"] in ("continue", "winner", "unknown")
            contact = int(label["contact_frame"]) if label["contact_frame"] else None
            if contact is not None:
                assert int(tracking_summary[point]["first_frame"]) <= contact <= int(
                    tracking_summary[point]["last_frame"])
            row = dict(label)
            row.update(game=game, near_pair=near, far_pair=far,
                       hitter_pair=near if side == "N" else far, point_winner=winner)
            # For rally A this is only the eventual point context, not that flow's result.
            row["hitter_won_point"] = row["hitter_pair"] == winner
            row["players_available"] = 0
            for slot, index in by_slot.items():
                selected = nearest_position(index, contact)
                row[f"{slot}_x_m"] = selected[0] if selected else None
                row[f"{slot}_y_m"] = selected[1] if selected else None
                row[f"{slot}_frame_offset"] = selected[2] if selected else None
                row["players_available"] += selected is not None
            # point_08 first note: N1 tracks the line judge throughout; pink is absent.
            # Retain raw N1/N2 proxies for audit, suppress both near-pair metrics.
            # True means only 'not excluded by this rule', never independently validated.
            row["near_positions_trusted"] = point != 8
            row["far_positions_trusted"] = True
            for prefix, pair_side in (("hitter", side), ("opponent", "F" if side == "N" else "N")):
                lateral, depth = pair_metrics(row, pair_side)
                row[f"{prefix}_pair_lateral_m"] = lateral
                row[f"{prefix}_pair_depth_m"] = depth
            shots.append(row)
            labels.append(label)
    assert len(shots) == len(labels) == EXPECTED_LABEL_ROWS, f"Expected 91 labels, got {len(shots)}"
    assert [{column: row[column] for column in LABEL_COLUMNS} for row in shots] == labels
    return shots


def validate_end_mapping(inputs, inventory):
    previous = inputs.csv(ROOT / "outputs" / "shot_table_20261117" / "shot_table.csv",
                          ("point", "point_winner_side"))
    expected_sides = {4: "far", 5: "near", 6: "near"}
    for point, expected_side in expected_sides.items():
        assert {row["point_winner_side"] for row in previous if int(row["point"]) == point} == {expected_side}
        assert inventory[point]["point_winner"] == {"near": "F", "far": "U"}[expected_side]


def build_point_summary(inputs, shots, tracking_summary):
    timing = inputs.by_point(HERE / "tracking" / "timing.csv")
    normal_timing = inputs.by_point(HERE / "review" / "timing.csv")
    far_timing = inputs.by_point(HERE / "review_far" / "timing.csv")
    sheets_summary = inputs.by_point(HERE / "review" / "sheets_summary.csv")
    rows = []
    for point in POINTS:
        all_rows = [row for row in shots if int(row["point"]) == point]
        included = [row for row in all_rows if row["rally"] != "A"]
        n = len(included)
        clear = sum(row["confidence"] == "clear" for row in included)
        inferred = sum(row["confidence"] == "inferred" for row in included)
        four_raw = sum(row["players_available"] == 4 for row in included)
        four_trusted = sum(row["players_available"] == 4 and row["near_positions_trusted"]
                           and row["far_positions_trusted"] for row in included)
        issues = Counter(row["tracking_issue"] for row in included)
        summary = tracking_summary[point]
        normal_index = inputs.csv(HERE / "review" / f"point_{point:02d}" / "index.csv", ("sheet",))
        far_index = inputs.csv(HERE / "review_far" / f"point_{point:02d}" / "index.csv", ("sheet",))
        normal_count, far_count = len(normal_index), len(far_index)
        assert normal_count == int(normal_timing[point]["sheets"]) == int(sheets_summary[point]["sheets"])
        assert far_count == int(far_timing[point]["sheets"])
        for key in ("first_frame", "last_frame"):
            assert int(summary[key]) == int(timing[point][key])
        assert int(summary["interval_frames"]) == int(timing[point]["frames"])
        track_s = seconds(timing[point]["seconds"])
        normal_s = seconds(normal_timing[point]["seconds"])
        far_s = seconds(far_timing[point]["seconds"])
        row = {"point": point, "game": included[0]["game"],
               "near_pair": included[0]["near_pair"], "far_pair": included[0]["far_pair"],
               "point_winner": included[0]["point_winner"],
               "labels_rows_total": len(all_rows), "labels_rows_excluded_rally_a": len(all_rows) - n}
        for key in ("interval_frames", "camera_line_supported_frames", "four_player_frames"):
            row[key] = int(summary[key])
        # Keep the upstream ratio as published (six decimals); counts are available too.
        row["four_player_ratio"] = float(summary["four_player_ratio"]) if summary["four_player_ratio"] else None
        row.update(shots=n, clear_shots=clear, inferred_shots=inferred,
                   clear_ratio=fraction(clear, n), inferred_ratio=fraction(inferred, n),
                   contact_frame_shots=sum(bool(r["contact_frame"]) for r in included),
                   shots_four_players_available_raw=four_raw,
                   four_players_available_raw_ratio=fraction(four_raw, n),
                   shots_four_players_available_trusted=four_trusted,
                   four_players_available_trusted_ratio=fraction(four_trusted, n),
                   tracking_issue_none=issues[""])
        row.update({f"tracking_issue_{issue}": issues[issue] for issue in ISSUES})
        row.update(sheets_available_normal=normal_count, sheets_available_far=far_count,
                   sheets_available_total=normal_count + far_count,
                   tracking_seconds=track_s, normal_sheets_seconds=normal_s,
                   far_sheets_seconds=far_s, automatic_seconds=round(track_s + normal_s + far_s, 6))
        assert clear + inferred == n
        rows.append(row)
    assert len(rows) == 5
    return rows


def labeling_reference(inputs):
    lines = inputs.text(HERE / "labels" / "labeling_log.txt").splitlines()
    pattern = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}[+-]\d{2}:\d{2}"
    start = datetime.fromisoformat(re.search(pattern, lines[0]).group())
    end_line = next(line for line in lines if line.startswith("end all labeling"))
    end = datetime.fromisoformat(re.search(pattern, end_line).group())
    return {
        "start": start.isoformat(), "end": end.isoformat(),
        "elapsed_seconds": (end - start).total_seconds(), "log_lines": lines,
        "interpretation": "Claude が画像を読んだ時間で、人の作業時間の目安にはならない。ログの開始から終了までの経過時間で、純粋な画像読取時間の分離計測ではない。",
        "human_work_time_measured": False, "human_work_seconds": None,
    }


def build_workload(inputs, rows):
    additive = [
        "labels_rows_total", "labels_rows_excluded_rally_a", "interval_frames",
        "camera_line_supported_frames", "four_player_frames", "shots", "clear_shots",
        "inferred_shots", "contact_frame_shots", "shots_four_players_available_raw",
        "shots_four_players_available_trusted", "tracking_issue_none",
        *[f"tracking_issue_{issue}" for issue in ISSUES],
        "sheets_available_normal", "sheets_available_far", "sheets_available_total",
        "tracking_seconds", "normal_sheets_seconds", "far_sheets_seconds", "automatic_seconds",
    ]
    totals = {key: sum(row[key] for row in rows) for key in additive}
    for key in additive:
        if key.endswith("_seconds"):
            totals[key] = round(totals[key], 6)
    totals.update(points=len(rows), clear_ratio=fraction(totals["clear_shots"], totals["shots"]),
                  inferred_ratio=fraction(totals["inferred_shots"], totals["shots"]),
                  four_players_available_raw_ratio=fraction(totals["shots_four_players_available_raw"], totals["shots"]),
                  four_players_available_trusted_ratio=fraction(totals["shots_four_players_available_trusted"], totals["shots"]),
                  four_player_frame_ratio=fraction(totals["four_player_frames"], totals["camera_line_supported_frames"]))
    averages = {key: round(totals[key] / len(rows), 7) for key in additive}
    # Point-weighted means differ from total ratios, which weight each shot equally.
    for key in ("clear_ratio", "inferred_ratio", "four_players_available_raw_ratio",
                "four_players_available_trusted_ratio"):
        values = [row[key] for row in rows if row[key] is not None]
        averages[f"{key}_mean_of_point_ratios"] = sum(values) / len(values) if values else None
    extrapolation = {"target_points": TARGET_POINTS, "method": "5点の平均を掛けただけ（1点あたり平均 × 66）。"}
    for source, target in (("tracking_seconds", "tracking_minutes"),
                           ("normal_sheets_seconds", "normal_sheets_minutes"),
                           ("far_sheets_seconds", "far_sheets_minutes"),
                           ("automatic_seconds", "automatic_minutes")):
        extrapolation[target] = round(averages[source] * TARGET_POINTS / 60, 8)
    for key in ("shots", "sheets_available_normal", "sheets_available_far", "sheets_available_total"):
        extrapolation[key] = round(averages[key] * TARGET_POINTS, 6)
    log = labeling_reference(inputs)
    return {
        "scope": {"match": "match2", "points": list(POINTS),
                  "shot_aggregation_excludes_rally": "A", "point_count": len(rows)},
        "definitions": {
            "total_ratios": "合計の率は合計件数から算出（打球数で加重）。",
            "mean_of_point_ratios": "1点あたりの率は5点の率の単純平均。合計の率とは異なる。",
            "four_player_shot_denominator": "rally Aを除く全打球。contact_frame欠測も分母に残す。",
            "positions_trusted": "第8点nearのみFalse。他のTrueは独立検証済みを意味しない。",
            "frame_counts_and_processing": "フレーム集計・自動時間・利用可能シートは点区間全体の記録。rally Aの分離記録がないため、打球集計のみAを除外する。",
            "automatic_seconds": "既存の追跡＋通常シート＋拡大シートの記録秒数の和。表作成時間、導入、再試行、ラベル付け、人の確認は含まない。",
            "sheet_counts": "index.csvの行数。Claudeの読了枚数ではない。",
            "coordinates": {"width_m": WIDTH_M, "length_m": LENGTH_M, "net_y_m": NET_Y_M,
                            "frame_tolerance": MAX_FRAME_OFFSET, "tie_break": "earlier frame"},
        },
        "totals": totals, "per_point_mean": averages,
        "extrapolation_66_points": extrapolation, "claude_labeling_reference": log,
        "limitations": [
            "ラベルはClaudeが1回付けただけで、人の確認も二人目の確認もない。clearも独立に確認した正解ではない。",
            "inferredは前後の打球からの推定。読めなかった中間打球がnoteにあり、打球数の完全性は未確認。",
            "接触フレームは±6フレーム程度の精度という作業上の目安。独立検証した誤差分布ではない。位置探索の±3とは別。",
            "出力率・clear割合から打球ラベルや位置の精度は算出できず、66点の精度も推定できない。",
            "5点は無作為標本ではなく、外挿にラリー長・カメラ・エンド・修正負担の変動を反映していない。",
            "人の作業時間は測っていない。",
        ],
        "source_sha256": dict(sorted(inputs.hashes.items())),
    }


def percent(value):
    return f"{value:.1%}" if value is not None else "—"


def markdown_table(headers, data):
    return "\n".join([
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
        *["| " + " | ".join(map(str, row)) + " |" for row in data],
    ])


def render_results(rows, workload):
    total, mean = workload["totals"], workload["per_point_mean"]
    projected, log = workload["extrapolation_66_points"], workload["claude_labeling_reference"]
    all_rows = [*[(str(row["point"]), row) for row in rows], ("合計", total)]
    lines = ["## 実行結果（スクリプト生成）", "",
             "`shot_table.csv`・`point_summary.csv`・`workload_estimate.json`と同じ計算結果から生成する。",
             f"全ラベル{total['labels_rows_total']}行を表に保存。第7点のrally A {total['labels_rows_excluded_rally_a']}行を打球集計から除き、{total['points']}点・{total['shots']}打球を対象とする。", "",
             "### 打球と位置の取得状況", "",
             markdown_table(["点", "打球", "clear", "inferred", "clear率", "接触Fあり", "4人取得・未考慮", "同割合", "4人取得・信頼考慮", "同割合"], [
                 [label, r["shots"], r["clear_shots"], r["inferred_shots"], percent(r["clear_ratio"]),
                  r["contact_frame_shots"], r["shots_four_players_available_raw"], percent(r["four_players_available_raw_ratio"]),
                  r["shots_four_players_available_trusted"], percent(r["four_players_available_trusted_ratio"])] for label, r in all_rows]), "",
             "割合の分母はrally Aを除く全打球。第8点はnear全体の信頼フラグがFalseなので、4人取得の信頼考慮後は0打球。Trueの点も精度検証済みではない。", "",
             "### 点区間全体のフレーム集計", "",
             markdown_table(["点", "区間F", "試合カメラ代理F", "4人出力F", "4人出力率"], [
                 [label, r["interval_frames"], r["camera_line_supported_frames"], r["four_player_frames"],
                  percent(r["four_player_ratio"] if label != "合計" else r["four_player_frame_ratio"])] for label, r in all_rows]), "",
             "`tracking/summary.csv`の値を使用。試合カメラは`camera_line_supported=1`の代理条件。4人出力率の分母はこのフレーム数であり、全区間Fではない。第8点の線審も人数に含む未補正の出力率。", "",
             "### 追跡問題のラベル件数", "",
             markdown_table(["点", "wrong_person", "missing", "swapped", "空欄"], [
                 [label, *[r[f"tracking_issue_{issue}"] for issue in ISSUES], r["tracking_issue_none"]] for label, r in all_rows]), "",
             "noteで区間全体に及ぶ問題が記されても、件数はtracking_issueセルが記入された行だけを数える。空欄は問題がないと確認済みという意味ではない。", "",
             "### 自動処理の時間と利用可能シート", "",
             markdown_table(["点", "追跡秒", "通常秒", "拡大秒", "合計秒", "通常枚数", "拡大枚数"], [
                 [label, *[f"{r[key]:.6f}" for key in ("tracking_seconds", "normal_sheets_seconds", "far_sheets_seconds", "automatic_seconds")],
                  r["sheets_available_normal"], r["sheets_available_far"]] for label, r in all_rows]), "",
             "フレーム・時間・シートは点区間全体の既存記録で、rally Aを切り分けられない。打球集計のA除外とは分けて解釈する。自動時間は3工程の記録値の和で、今回の表作成時間・環境導入・ラベル付け・人の確認は含まない。シート枚数はindex.csvの行数で、Claudeが読んだ枚数ではない。", "",
             "### 5点の合計と1点あたり平均", "",
             markdown_table(["指標", "5点合計", "1点あたり平均"], [
                 ["自動処理（秒）", f"{total['automatic_seconds']:.6f}", f"{mean['automatic_seconds']:.7f}"],
                 ["打球数", total["shots"], f"{mean['shots']:.1f}"],
                 ["通常シート枚数", total["sheets_available_normal"], f"{mean['sheets_available_normal']:.1f}"],
                 ["拡大シート枚数", total["sheets_available_far"], f"{mean['sheets_available_far']:.1f}"],
                 ["clear割合", percent(total["clear_ratio"]), percent(mean["clear_ratio_mean_of_point_ratios"])],
                 ["4人取得割合・未考慮", percent(total["four_players_available_raw_ratio"]), percent(mean["four_players_available_raw_ratio_mean_of_point_ratios"])],
                 ["4人取得割合・信頼考慮", percent(total["four_players_available_trusted_ratio"]), percent(mean["four_players_available_trusted_ratio_mean_of_point_ratios"])],
             ]), "",
             "率の合計欄は合計件数の比（打球加重）、平均欄は5点の率の単純平均。人数の出力率、clear率、信頼フラグ考慮後の取得率はいずれも正答率ではない。", "",
             "### 全66点への単純外挿", "",
             "**5点の平均を掛けただけ（1点あたり平均 × 66）**。対象5点の処理条件と打球・シート数が同程度に続くという仮定で、66点の実測結果ではない。小数は外挿値をそのまま示す。", "",
             markdown_table(["指標", "66点の単純外挿"], [
                 ["自動処理（分）", f"{projected['automatic_minutes']:.6f}"],
                 ["打球数", f"{projected['shots']:.1f}"],
                 ["通常シート枚数", f"{projected['sheets_available_normal']:.1f}"],
                 ["拡大シート枚数", f"{projected['sheets_available_far']:.1f}"],
                 ["シート合計枚数", f"{projected['sheets_available_total']:.1f}"],
             ]), "",
             "5点は無作為標本ではない。ラリー長、カメラ、エンド、追跡失敗と修正負担の変動は反映していない。全66点での精度や人の必要時間を、この外挿からは推定できない。", "",
             "### Claudeのラベル付けログ（参考）", "",
             f"開始 `{log['start']}`、終了 `{log['end']}`、経過 **{log['elapsed_seconds']:.0f}秒（{log['elapsed_seconds'] / 60:.2f}分）**。", "",
             "**Claude が画像を読んだ時間で、人の作業時間の目安にはならない。人の作業時間は測っていない。** ログ全体の開始・終了差であり、画像読取だけの純時間を分離計測してはいない。途中ログは第7点で通常18枚中11枚を読んだと記録し、すべてのシートを読んだわけではない。", "",
             "### 分かったことと根拠", "",
             "- 作業経緯（依頼文で提供）：通常シートでは奥の打球を読めず、拡大シートが必要だった。各ラベルのnoteにこの経緯自体が明記されているわけではない。",
             "- 第7点：rally Aの最後のnoteに、約9秒後に再びサーブがあり、先の流れはレットかフォルトの可能性とある。Bは区間内の2回目のサーブで始まる流れ。第2サービスと確定したという意味ではない。Aは表に残し打球集計から外す。",
             "- 第8点：先頭noteに、手前のピンクが一度も追跡されず、N1の印が右の線審に付いていたとある。4人分の値が出ることだけでは、正しい4人の位置が得られたとは言えない。",
             "- 第7点Bの4打目は別人への重なり、14・16・18打目は画面外の記録がある。第7点Bの19打目と第11点7打目以降にはF1/F2の印の入れ替わりが記されている。",
             "- 第8点の複数noteには間の手前・奥の打球が読めないとある。表の打球数は今回ラベルが付いた数で、全打球の捕捉を確認した値ではない。", "",
             "### この集計で残る限界", "",
             *[f"- {limitation}" for limitation in workload["limitations"]], "",
             "11/20には、自動処理の所要時間と、未検証ラベル・取得率・既知の失敗例を提示できる。精度と人の作業量を示すには、元動画による独立確認と、人による確認・修正時間の計測が別途必要。",
             ""]
    return "\n".join(lines)


def update_readme(results):
    path = HERE / "README.md"
    original = path.read_text(encoding="utf-8")
    if RESULTS_BEGIN in original:
        before, rest = original.split(RESULTS_BEGIN, 1)
        _, after = rest.split(RESULTS_END, 1)
    else:
        before, separator, _ = original.partition("## 実行結果（実行後に記入）")
        assert separator, "README has no result section/markers; refusing to replace other content"
        after = "\n"
    path.write_text(before.rstrip() + "\n\n" + RESULTS_BEGIN + "\n" + results.rstrip()
                    + "\n" + RESULTS_END + after, encoding="utf-8")


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    inputs = Inputs()
    inventory = inputs.by_point(ROOT / "outputs" / "match2_point_inventory_20261117" / "point_inventory.csv")
    tracking_summary = inputs.by_point(HERE / "tracking" / "summary.csv")
    validate_end_mapping(inputs, inventory)
    shots = build_shots(inputs, inventory, tracking_summary)
    points = build_point_summary(inputs, shots, tracking_summary)
    workload = build_workload(inputs, points)
    results = render_results(points, workload)
    inputs.assert_unchanged()
    write_csv(HERE / "shot_table.csv", LABEL_COLUMNS + ADDED_COLUMNS, shots)
    write_csv(HERE / "point_summary.csv", list(points[0]), points)
    (HERE / "workload_estimate.json").write_text(
        json.dumps(workload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    update_readme(results)
    # Check the saved artifact, not just the in-memory list.
    with (HERE / "shot_table.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == EXPECTED_LABEL_ROWS, "shot_table.csv must contain 91 data rows"
    with (HERE / "point_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
        assert len(list(csv.DictReader(stream))) == 5
    inputs.assert_unchanged()
    print(f"Wrote {len(shots)} label rows; summarized {len(points)} points / {workload['totals']['shots']} shots.")
    print("Read-only input SHA256 checks passed.")
    print(json.dumps({key: workload[key] for key in ("totals", "per_point_mean", "extrapolation_66_points")},
                     ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
