"""Build a shot-candidate demo from existing JSON only; never open a video.

Run with Python 3.12, matplotlib and Pillow. Paths are relative to this file,
so the command works from any working directory. Existing inputs are read only.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from collections import Counter
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from PIL import Image


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
SOURCE = ROOT / "outputs" / "ball_quality_20261111"
REVIEW = SOURCE / "match2_points_4_6_stroke_review"
STROKES = REVIEW / "stroke_observations.json"
POSITIONS = REVIEW / "guarded_positions.json"
REFERENCE = SOURCE / "match2_points_4_6_contact_sensitivity" / "contact_sensitivity.json"
METHOD = SOURCE / "analyze_match2_points_4_6_contact_sensitivity.py"
GEOMETRY_METHOD = SOURCE / "summarize_match2_receiving_positions.py"
SCALE_METHOD = SOURCE / "review_match2_stroke_routes.py"
REVIEW_NOTES = (
    SOURCE / "match2_points_4_6_contact_sensitivity" / "接触時刻と守備配置.md",
    REVIEW / "打球候補.md",
)

# Copied, not imported: review_match2_stroke_routes.METRES, accessed by
# analyze_match2_points_4_6_contact_sensitivity via summarize_match2_receiving_positions.
# Its metres(xy) multiplies each normalized coordinate by the matching dimension.
METRES = (10.97, 23.77)
WIDTH, LENGTH = METRES
NET_Y = LENGTH * 0.5
# Same approximate schematic lines as summarize_match2_receiving_positions.render_preview.
SERVICE_Y = (LENGTH * 0.231, LENGTH * 0.769)
SINGLES_X = (WIDTH * 0.125, WIDTH * 0.875)
PLAYERS = ("N1", "N2", "F1", "F2")
PAIRS = {"near": ("N1", "N2"), "far": ("F1", "F2")}
SIDES = {"手前": "near", "奥": "far"}
OUTCOMES = ("continue", "error_net", "unknown")
COLORS = {"N1": "#2166ac", "N2": "#61a5d2", "F1": "#ce6415", "F2": "#e99b45"}
PNG_TARGET_BYTES = 256_000
COLUMNS = [
    "point", "shot_index", "is_serve", "hitter", "hitter_side", "status",
    "contact_frame_first", "contact_frame_last", "contact_frame_ref",
    *[f"{p}_{axis}_m" for p in PLAYERS for axis in ("x", "y")],
    *[f"{p}_frame_offset" for p in PLAYERS],
    "players_available", "hitter_pair_lateral_m", "hitter_pair_depth_m",
    "opponent_pair_lateral_m", "opponent_pair_depth_m", "hitter_x_m", "hitter_y_m",
    "hitter_net_distance_m", "shot_outcome", "outcome_source", "point_winner_side",
    "hitter_won_point", "evidence",
]


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def side_of(player):
    return "near" if player.startswith("N") else "far"


def load_inputs():
    observations = load_json(STROKES)
    positions = load_json(POSITIONS)
    reference = load_json(REFERENCE)
    if tuple(reference["reference_court_size_m"]) != METRES:
        raise ValueError("換算寸法が既存 contact_sensitivity.json と一致しません。")
    hashes = {relative(path): sha256(path) for path in (
        STROKES, POSITIONS, REFERENCE, METHOD, GEOMETRY_METHOD, SCALE_METHOD, *REVIEW_NOTES
    )}
    for path in (STROKES, POSITIONS):
        expected = reference.get("source_sha256", {}).get(path.name)
        if expected is not None and expected != hashes[relative(path)]:
            raise ValueError(f"既存の感度分析と入力が異なります: {path.name}")
    points = observations["points"]
    if sorted(p["point"] for p in points) != [4, 5, 6]:
        raise ValueError("入力は第4・5・6点が各1件のデータを想定しています。")
    for point in points:
        if point["winner_side"] not in SIDES or not point["events"]:
            raise ValueError(f"得点側またはeventsが不正です: 第{point['point']}点")
        previous_end = -1
        for event in point["events"]:
            first, last = event["frames"]
            if (type(first) is not int or type(last) is not int
                    or first > last or first <= previous_end):
                raise ValueError(f"接触区間の順序または値が不正です: {event}")
            previous_end = last
            if event["hitter"] not in PLAYERS:
                raise ValueError(f"未知の打者ID: {event['hitter']}")
            if not isinstance(event["status"], str):
                raise ValueError("statusは文字列である必要があります。")
            # evidence is preserved verbatim in CSV; its base is REVIEW, not HERE.
            if not (REVIEW / event["evidence"]).is_file():
                raise FileNotFoundError(REVIEW / event["evidence"])
    # The sole error label must remain supported by the human-reviewed source.
    final = next(p for p in points if p["point"] == 6)["events"][-1]
    if not (final["hitter"] == "F2" and final["status"] == "user_confirmed"
            and final.get("result") == "net"):
        raise ValueError("第6点末尾のF2・人手確認済みネットと入力が一致しません。")

    by_player = {player: {} for player in PLAYERS}
    seen = set()
    for position in positions:
        frame, player = position["source_frame"], position["display_id"]
        if player not in PLAYERS or type(frame) is not int:
            raise ValueError(f"位置データのフレーム/IDが不正: {position}")
        if (frame, player) in seen:
            raise ValueError(f"位置データの重複: F{frame} {player}")
        seen.add((frame, player))
        xy = position["court_xy"]
        if xy is None:
            continue  # Preserve the upstream guard; never fill a hidden position.
        if len(xy) != 2 or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in xy
        ):
            raise ValueError(f"有限な2次元座標ではありません: {position}")
        by_player[player][frame] = {
            "xy_m": (xy[0] * METRES[0], xy[1] * METRES[1]),
            "position_status": position["position_status"],
        }
    return observations, by_player, hashes


def closest_position(index, first, last, ref):
    candidates = [frame for frame in range(first, last + 1) if frame in index]
    if not candidates:
        return None
    # Deterministic tie rule: choose the earlier frame. Never leave the interval.
    frame = min(candidates, key=lambda f: (abs(f - ref), f))
    return frame, index[frame]


def pair_geometry(row, players):
    a, b = players
    if row[f"{a}_x_m"] is None or row[f"{b}_x_m"] is None:
        return None, None
    # Exactly the definitions in summarize_match2_receiving_positions.summarize.
    return (abs(row[f"{a}_x_m"] - row[f"{b}_x_m"]),
            abs(row[f"{a}_y_m"] - row[f"{b}_y_m"]))


def build_rows(observations, by_player):
    rows, selection_audit = [], []
    for point in sorted(observations["points"], key=lambda p: p["point"]):
        events = point["events"]
        for shot_index, event in enumerate(events, 1):
            first, last = event["frames"]
            ref = (first + last) // 2
            hitter = event["hitter"]
            side = side_of(hitter)
            winner = SIDES[point["winner_side"]]
            row = {column: None for column in COLUMNS}
            row.update(point=point["point"], shot_index=shot_index, is_serve=shot_index == 1,
                       hitter=hitter, hitter_side=side, status=event["status"],
                       contact_frame_first=first, contact_frame_last=last, contact_frame_ref=ref,
                       players_available=0, point_winner_side=winner, hitter_won_point=side == winner,
                       evidence=event["evidence"])
            selected = {}
            for player in PLAYERS:
                choice = closest_position(by_player[player], first, last, ref)
                if choice is None:
                    selected[player] = None
                    continue
                frame, position = choice
                row[f"{player}_x_m"], row[f"{player}_y_m"] = position["xy_m"]
                row[f"{player}_frame_offset"] = frame - ref
                row["players_available"] += 1
                selected[player] = {"source_frame": frame, "frame_offset": frame - ref,
                                    "position_status": position["position_status"]}
            for prefix, pair_side in (("hitter", side), ("opponent", "far" if side == "near" else "near")):
                row[f"{prefix}_pair_lateral_m"], row[f"{prefix}_pair_depth_m"] = pair_geometry(row, PAIRS[pair_side])
            row["hitter_x_m"], row["hitter_y_m"] = row[f"{hitter}_x_m"], row[f"{hitter}_y_m"]
            if row["hitter_y_m"] is not None:
                row["hitter_net_distance_m"] = abs(row["hitter_y_m"] - NET_Y)
            # A next candidate is evidence of the recorded sequence, not a verified return.
            if shot_index < len(events):
                row.update(shot_outcome="continue", outcome_source="next_shot_exists")
            elif point["point"] == 6:
                row.update(shot_outcome="error_net", outcome_source="human_review")
            else:
                row.update(shot_outcome="unknown", outcome_source="undetermined")
            rows.append(row)
            selection_audit.append({"point": point["point"], "shot_index": shot_index,
                                    "contact_frame_ref": ref, "selected_players": selected})
    return rows, selection_audit


def counts(rows):
    return {
        "shot_candidates": len(rows),
        "excluding_serves": sum(not r["is_serve"] for r in rows),
        "four_players_available": sum(r["players_available"] == 4 for r in rows),
        "four_players_available_excluding_serves": sum(
            r["players_available"] == 4 and not r["is_serve"] for r in rows),
        "outcomes": {key: sum(r["shot_outcome"] == key for r in rows) for key in OUTCOMES},
        "statuses": dict(sorted(Counter(r["status"] for r in rows).items())),
    }


def group_medians(rows):
    result = []
    for won, group in ((True, "hitter_won"), (False, "hitter_lost")):
        samples = [r for r in rows if r["hitter_won_point"] == won]
        entry = {"group": group, "hitter_won_point": won, "n_shots": len(samples),
                 "n_points": len({r["point"] for r in samples})}
        for feature in ("lateral", "depth"):
            values = [r[f"opponent_pair_{feature}_m"] for r in samples
                      if r[f"opponent_pair_{feature}_m"] is not None]
            entry[f"opponent_pair_{feature}_n"] = len(values)
            entry[f"opponent_pair_{feature}_median_m"] = median(values) if values else None
        result.append(entry)
    return result


def configure_font():
    # Use Yu Gothic explicitly, including installations missed by matplotlib's cache.
    font_path = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "YuGothR.ttc"
    if font_path.is_file():
        font_manager.fontManager.addfont(str(font_path))
    found = font_manager.findfont(font_manager.FontProperties(family="Yu Gothic"), fallback_to_default=False)
    plt.rcParams.update({"font.family": "Yu Gothic", "axes.unicode_minus": False})
    return str(found)


def render_figure(rows):
    points = (4, 5, 6)
    ncols = max(sum(r["point"] == p for r in rows) for p in points)
    fig, axes = plt.subplots(3, ncols, figsize=(2.3 * ncols, 11.2), squeeze=False)
    xs = [r[f"{p}_x_m"] for r in rows for p in PLAYERS if r[f"{p}_x_m"] is not None]
    ys = [r[f"{p}_y_m"] for r in rows for p in PLAYERS if r[f"{p}_y_m"] is not None]
    xlim = (min([0.0, *xs]) - 1.6, max([WIDTH, *xs]) + 1.6)
    ylim = (min([0.0, *ys]) - 1.6, max([LENGTH, *ys]) + 1.6)
    labels = {"continue": "継続候補", "error_net": "ネット(確認済)", "unknown": "結果不明"}
    try:
        for ridx, point in enumerate(points):
            point_rows = [r for r in rows if r["point"] == point]
            for cidx, ax in enumerate(axes[ridx]):
                if cidx >= len(point_rows):
                    ax.set_axis_off()
                    continue
                row = point_rows[cidx]
                ax.add_patch(Rectangle((0, 0), WIDTH, LENGTH, fill=False, edgecolor="#555555", linewidth=0.8))
                ax.hlines(NET_Y, 0, WIDTH, colors="#333333", linewidth=1.4)
                ax.hlines(SERVICE_Y, SINGLES_X[0], SINGLES_X[1], colors="#999999", linewidth=0.6)
                ax.vlines(SINGLES_X, 0, LENGTH, colors="#999999", linewidth=0.6)
                ax.vlines(WIDTH / 2, SERVICE_Y[0], SERVICE_Y[1], colors="#999999", linewidth=0.6)
                for player in PLAYERS:
                    x, y = row[f"{player}_x_m"], row[f"{player}_y_m"]
                    if x is None:
                        continue
                    is_hitter = player == row["hitter"]
                    ax.scatter([x], [y], c=[COLORS[player]], marker="*" if is_hitter else "o",
                               s=85 if is_hitter else 26, edgecolors="#333333", linewidths=0.35, zorder=3)
                    ax.annotate(player, (x, y), xytext=(4, 3), textcoords="offset points",
                                fontsize=7, color=COLORS[player], zorder=4)
                hitter_label = row["hitter"] + ("?" if row["status"] == "offscreen" else "")
                ax.set_title(f"第{point}点・{row['shot_index']}打目 {hitter_label}\n{labels[row['shot_outcome']]}",
                             fontsize=9, pad=5)
                missing = [p for p in PLAYERS if row[f"{p}_x_m"] is None]
                shifted = [f"{p}:{row[f'{p}_frame_offset']:+d}" for p in PLAYERS
                           if row[f"{p}_frame_offset"] not in (None, 0)]
                footer = f"参照F{row['contact_frame_ref']}｜位置{row['players_available']}/4"
                if missing:
                    footer += "\n欠測 " + "/".join(missing)
                if shifted:
                    footer += "\n差 " + ", ".join(shifted) + " frame"
                ax.text(0.5, -0.10, footer, transform=ax.transAxes, ha="center", va="top", fontsize=7)
                ax.set(xlim=xlim, ylim=ylim, aspect="equal")
                ax.set_xticks([0, WIDTH], ["0", "10.97"])
                ax.set_yticks([0, NET_Y, LENGTH], ["0", "11.885", "23.77"])
                ax.tick_params(labelsize=6, length=2, pad=1)
                for spine in ax.spines.values():
                    spine.set_visible(False)
        fig.suptitle("match2 第4〜6点：打球候補ごとの推定配置と結果", fontsize=16, y=0.985)
        legend = [Line2D([], [], color=COLORS["N1"], marker="o", linestyle="", label="手前 N1/N2"),
                  Line2D([], [], color=COLORS["F1"], marker="o", linestyle="", label="奥 F1/F2"),
                  Line2D([], [], color="#444444", marker="*", markersize=10, linestyle="", label="打者候補（位置あり時）")]
        fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(0.5, 0.96), ncol=3, frameon=False, fontsize=10)
        fig.text(0.5, 0.015, "単位 m・上が奥／下が手前。区間中央に近い選手別位置。欠測は描画せず、コート外座標も保持。\n"
                 "打球順・配置は暫定。継続候補＝次の候補あり。第4・5点の終了事象は未判定。", ha="center", fontsize=9)
        fig.subplots_adjust(left=0.025, right=0.99, top=0.88, bottom=0.10, wspace=0.36, hspace=0.62)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=120, facecolor="white")
        buffer.seek(0)
        # Indexed PNG reduces size without reducing DPI or dropping any plotted points.
        encoded = []
        with Image.open(buffer) as original:
            rgb = original.convert("RGB")
            for colors in (256, 128, 96):
                compressed = io.BytesIO()
                rgb.quantize(colors=colors, method=Image.Quantize.MEDIANCUT).save(
                    compressed, format="PNG", optimize=True, compress_level=9, dpi=(120, 120))
                encoded.append(compressed.getvalue())
                if len(encoded[-1]) < PNG_TARGET_BYTES:
                    break
        data = min(encoded, key=len)
        (HERE / "shot_positions_points_4_6.png").write_bytes(data)
        return {"dpi": 120, "bytes": len(data), "target_bytes": PNG_TARGET_BYTES,
                "under_target": len(data) < PNG_TARGET_BYTES, "indexed_png": True}
    finally:
        plt.close(fig)


def make_summary(rows, audit, observations, hashes, image_info, font_path):
    return {
        "scope": "match2 第4〜6点・打球候補デモ。独立な打球結果/座標精度の検証ではない。",
        "source_fps": observations.get("source_fps"),
        "source_sha256": hashes,
        "method": {
            "coordinate_conversion": "Copied constants/formula; no imports of existing analysis modules.",
            "reference_court_size_m": list(METRES), "net_y_m": NET_Y,
            "service_line_y_m_schematic": list(SERVICE_Y),
            "singles_line_x_m_schematic": list(SINGLES_X),
            "reference_frame": "(first + last) // 2; nearest available per player inside inclusive window; ties earlier",
            "frame_offset": "selected_source_frame - contact_frame_ref",
            "pair_metrics": "abs(x1-x2), abs(y1-y2); per-player frames may differ",
            "evidence_base": relative(REVIEW),
            "missing": "CSV empty cell / JSON null; no interpolation, extrapolation or coordinate clamping",
        },
        "total": counts(rows),
        "by_point": {str(p): counts([r for r in rows if r["point"] == p]) for p in (4, 5, 6)},
        "opponent_pair_by_hitter_point_result": {
            "interpretation": "傾向の例。点の勝敗で群分けし、同一点の複数打球を含む。検定なし。",
            "all_shots": group_medians(rows),
            "excluding_serves": group_medians([r for r in rows if not r["is_serve"]]),
        },
        "position_selection_audit": audit,
        "terminal_notes": {str(p["point"]): p["terminal_note"] for p in observations["points"]},
        "figure": {**image_info, "font": "Yu Gothic", "font_path": font_path},
        "outputs": ["shot_table.csv", "shot_positions_points_4_6.png", "shot_summary.json", "README.md"],
    }


def median_table(groups):
    lines = ["| 打者側の点の勝敗 | 打球候補数 | 含まれる点数 | 横間隔の有効件数 | 横間隔中央値 (m) | 前後差の有効件数 | 前後差中央値 (m) |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for group in groups:
        label = "得点" if group["hitter_won_point"] else "失点"
        lat, dep = (group[f"opponent_pair_{key}_median_m"] for key in ("lateral", "depth"))
        lat_text = f"{lat:.3f}" if lat is not None else "—"
        dep_text = f"{dep:.3f}" if dep is not None else "—"
        lines.append(f"| {label} | {group['n_shots']} | {group['n_points']} | "
                     f"{group['opponent_pair_lateral_n']} | {lat_text} | "
                     f"{group['opponent_pair_depth_n']} | {dep_text} |")
    return "\n".join(lines)


def make_readme(summary):
    total = summary["total"]
    count_lines = ["| 点 | 打球候補 | サーブ除外 | 4人の位置あり | continue | error_net | unknown |",
                   "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for label, count in [*summary["by_point"].items(), ("合計", total)]:
        out = count["outcomes"]
        count_lines.append(f"| {label} | {count['shot_candidates']} | {count['excluding_serves']} | "
                           f"{count['four_players_available']} | {out['continue']} | {out['error_net']} | {out['unknown']} |")
    shifts = []
    for sample in summary["position_selection_audit"]:
        for player, selected in sample["selected_players"].items():
            if selected is not None and selected["frame_offset"] != 0:
                shifts.append(f"- 第{sample['point']}点・{sample['shot_index']}打目 {player}: "
                              f"F{selected['source_frame']}（参照F{sample['contact_frame_ref']}、差{selected['frame_offset']:+d}）")
    geometry = summary["opponent_pair_by_hitter_point_result"]
    return f'''# match2 第4〜6点 打球単位の戦術データ（試作）

11/20の指導教員向けデモ。「既存の試合映像レビューと推定位置から、打球候補ごとに配置と記録済みの結果を対応づける」ことを示す。

## 実行と出力

必要環境: Python 3.12、matplotlib、Pillow、Yu Gothic。動画・画像の再解析、追跡・検出は行わない。根拠画像は存在だけを確認し、開かない。
作業ディレクトリに依存せず、このスクリプトの場所から入力を解決する。既存分析モジュールはimportしない。既存入力は変更しない。

```powershell
& 'C:\\Users\\ta239\\AppData\\Local\\Programs\\Python\\Python312\\python.exe' 'C:\\Users\\ta239\\Research\\softtennis_match_study\\outputs\\shot_table_20261117\\build_match2_shot_table.py'
```

- `shot_table.csv`: 1行=1打球候補、UTF-8 BOMあり。欠測は空欄、真偽値はTrue/False。
- `shot_positions_points_4_6.png`: 点ごとに1行・候補順に1列、120 dpi・Yu Gothic。欠測は描画せず、全パネル共通の軸範囲。青が手前、橙が奥、位置がある打者は星。画面外打者はタイトルに`?`を付す。
- `shot_summary.json`: 集計、中央値と有効件数、入力SHA256、選手別採用フレームと位置status、描画サイズ。JSON欠測はnull。
- `README.md`: この説明と実行時の集計。再実行で本フォルダの上記4ファイルを上書きする。

PNGは256 KB（256,000 bytes）未満を目標に256/128/96色のパレット圧縮を行う。今回の出力は{summary['figure']['bytes']:,} bytes、目標達成={summary['figure']['under_target']}。サイズと描画内容は実行後に確認する。

## 入力と換算

入力は`outputs/ball_quality_20261111/`内の以下。全て読み取り専用。

- `match2_points_4_6_stroke_review/stroke_observations.json`: 候補区間・打者・status・得点側・根拠。
- `match2_points_4_6_stroke_review/guarded_positions.json`: 既存の推定位置とガードによる欠測記録。位置の正しさを確認済みという意味ではない。
- `analyze_match2_points_4_6_contact_sensitivity.py`、その`contact_sensitivity.json`、`接触時刻と守備配置.md`、`打球候補.md`を方法・寸法・人手確認結果の参照に使用。

既存のimport連鎖は`analyze_match2_points_4_6_contact_sensitivity.py` → `summarize_match2_receiving_positions.py` → `review_match2_stroke_routes.py`。
このスクリプトでは後二者の`metres(xy)`の式と`METRES=(10.97, 23.77)`を読み、同じ定数・式を複写した。パス依存や既存モジュールの副作用を避けるためimportしない。
`contact_sensitivity.json`の寸法と入力2ファイルのSHA256を照合し、不一致なら中止する。

`court_xy=(u,v)`から`x_m=u×10.97`, `y_m=v×23.77`。xは画面左から右、yは手前ベースライン0から奥ベースライン23.77へ増加する。
ネットは正規化y=0.5、すなわち`y_m=11.885`。ネット距離は`abs(hitter_y_m - 11.885)`で、ネット線までのコート平面上の距離。球の高さや飛距離ではない。
外枠は10.97×23.77 mで指定寸法と一致する。コート外の推定位置は切り詰めない。
サービスラインは既存`render_preview`と同じ正規化y=0.231/0.769（5.49087/18.27913 m）、シングルス線はx=0.125/0.875（1.37125/9.59875 m）を使う。これらは既存模式図に合わせた近似線で、公式寸法の再校正ではない。センターサービス線は2本のサービスライン間に描く。
メートル表記は既存の正規化平面に規格寸法を掛けた値であり、実距離の測量精度を保証しない。

## 位置の決め方と列の定義

入力の`events`の順序を保ち、statusにかかわらず1候補1行とする。最初の候補をサーブとして扱う。
`contact_frame_ref=(first+last)//2`（中央が半整数なら小さい方）。選手ごとに、両端を含む候補区間内で`court_xy`がnullでないフレームのうち、参照フレームに最も近いものを採用する。同距離なら早いフレーム。
区間内になければ座標とoffsetを空欄にする。区間外探索、補間、外挿、ガードで除かれた足元の復元はしない。利用可能とは既存推定値があることを指す。

| 列 | 定義 |
| --- | --- |
| `point`, `shot_index`, `is_serve` | 点番号、点内の候補順（1始まり）、先頭候補か |
| `hitter`, `hitter_side`, `status` | 元打者ID、N=near（手前）/F=far（奥）、元statusをそのまま保存 |
| `contact_frame_first/last/ref` | 元映像の候補開始・終了・中央の整数フレーム。接触確定時刻ではない |
| `N1_x_m/N1_y_m`〜`F2_x_m/F2_y_m` | 上記規則で選んだ各選手の推定位置（m） |
| `N1_frame_offset`〜`F2_frame_offset` | 採用source_frame − contact_frame_ref。負は参照より前、正は後。0と欠測は区別 |
| `players_available` | 位置のある人数（0〜4）。同一フレームに4人そろうという意味ではない |
| `hitter_pair_lateral_m/depth_m` | 打者側2人のabs(x差)/abs(y差)。2人そろわなければ空欄 |
| `opponent_pair_lateral_m/depth_m` | 相手側2人のabs(x差)/abs(y差)。同上 |
| `hitter_x_m`, `hitter_y_m`, `hitter_net_distance_m` | 打者の採用位置とネットまでの平面距離。打者位置なしなら空欄 |
| `shot_outcome`, `outcome_source` | 下記の記録規則による結果と根拠種別 |
| `point_winner_side`, `hitter_won_point` | 元winner_sideをnear/far化、打者側と得点側が一致するか。打球自体の成功判定ではない |
| `evidence` | 元画像パスを文字列のまま保持。相対基準は`outputs/ball_quality_20261111/match2_points_4_6_stroke_review/`であり出力フォルダではない |

横間隔=左右方向の絶対差、前後差=ベースラインに垂直な方向の絶対差で、既存感度分析と同じ定義。2人間のユークリッド距離や前衛・後衛の役割差ではない。
既存感度分析は区間全体の同一フレームでの最小・最大を調べるが、本表は選手別に選んだ1位置から計算する。集計対象数と時間の扱いは異なる。

参照フレームからずれた採用位置:

{chr(10).join(shifts) if shifts else '- なし'}

特に第6点最後は接触フレーム未特定で、F2と他選手の位置時刻も一致しない。ペア指標は同時刻の配置とは限らず、正確なインパクト時の守備配置として解釈できない。

## 結果ラベル

- 最後以外は`continue` / `next_shot_exists`。次の**候補**があるという記録で、返球成功の独立確認ではない。
- 第6点最後はF2のネットが映像レビューで確認済みなので`error_net` / `human_review`。正確な接触フレームは未特定。
- 第4点最後（画面外候補）と第5点最後は`unknown` / `undetermined`。点の得点側から打球結果を逆算しない。

## 件数

{chr(10).join(count_lines)}

サーブ3候補の区間内には位置データがなく、4人とも空欄となる。

## 得失点別の相手側配置中央値（傾向の例）

主表はサーブ・画面外候補を含む全候補。欠測は指標ごとに除外し有効件数を併記する。1候補を1件として重みづけする。

{median_table(geometry['all_shots'])}

サーブ除外:

{median_table(geometry['excluding_serves'])}

わずか3点の記述例であり、同一点の打球は独立標本ではない。群は打者側の**点の勝敗**で分けており、直後の返球結果ではない。検定はしない。配置の良否、得失点への因果効果、一般的戦術傾向は評価できない。
第6点最後のF2のネットに対する`opponent_pair_*`は手前側の配置である。奥側の返球失敗前の守備配置を調べるなら直前のN1打球時の相手側配置と区別する必要がある。

## 限界と今後

- 打球候補の多くは未確定。順序・打者・打球数の完全性を独立検証していない。`offscreen`のN1も仮の打者。
- 足元は検出箱の下端からの推定値。遮蔽などでnullにされた位置は欠測を保つ。位置が得られることと位置・人物対応が正しいことは別である。
- 候補区間中央は実接触時刻とは限らず、区間内の移動と推定の揺れを含む。選手別採用時刻のずれがある。
- 3点だけであり、第4点・第5点の終わり方は未判定。得点側の記録は終了事象の確定を意味しない。
- 今後: 第1〜3点は同形式の打球候補がないため今回の対象外。同形式の人手レビューを整備してから追加する。独立な接触・結果・位置の参照と、映像のみとの比較評価が必要。

## このデモで示せること・示せないこと

示せること: 既存の映像レビューと推定位置を打球候補ごとの表・コート図へ統合し、配置、既知の結果、欠測、不確実性と根拠を対応づけて記録できること。

示せないこと: 映像からの完全自動抽出、全打球の正確な接触時刻・座標・結果、戦術の有効性や新規性、映像レビュー支援の効果。これらは別途検証が必要。
'''


def main():
    observations, by_player, hashes = load_inputs()
    rows, audit = build_rows(observations, by_player)
    font_path = configure_font()
    HERE.mkdir(parents=True, exist_ok=True)
    image_info = render_figure(rows)
    summary = make_summary(rows, audit, observations, hashes, image_info, font_path)
    with (HERE / "shot_table.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (HERE / "shot_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (HERE / "README.md").write_text(make_readme(summary), encoding="utf-8")
    print(f"Wrote {len(rows)} shot candidates to {HERE}")
    for filename in summary["outputs"]:
        print(f"  {filename}: {(HERE / filename).stat().st_size:,} bytes")
    if not image_info["under_target"]:
        print("PNGは256 KB以上です。内容を確認し、必要に応じて図のサイズを調整してください。")


if __name__ == "__main__":
    main()
