"""Render Japanese figures/report for court-normalized pooled motion summaries.

The script deliberately makes the threshold sensitivity visible.  A high rate
at a single threshold or timepoint is not promoted to an anticipation finding.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


plt.rcParams["font.family"] = ["Yu Gothic"]
plt.rcParams["axes.unicode_minus"] = False

TIMEPOINTS = [-0.5, -0.3, -0.2, -0.1]
COLORS = {0.010: "#0072B2", 0.015: "#D55E00", 0.020: "#009E73"}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value: str) -> float:
    return float(value)


def threshold_of(rows: list[dict[str, str]]) -> float:
    values = {f(row["movement_threshold_court_width"]) for row in rows}
    if len(values) != 1:
        raise SystemExit("Each input summary must contain exactly one threshold")
    return values.pop()


def plot_main(rows: list[dict[str, str]], output: Path) -> None:
    rows = sorted(rows, key=lambda row: f(row["timepoint_sec"]))
    x = [f(row["timepoint_sec"]) for row in rows]
    y = [f(row["conditional_match_rate"]) for row in rows]
    low = [value - f(row["conditional_match_wilson95_low"]) for value, row in zip(y, rows)]
    high = [f(row["conditional_match_wilson95_high"]) - value for value, row in zip(y, rows)]
    figure, axis = plt.subplots(figsize=(10, 6.6))
    axis.axhline(0.5, color="#666666", linestyle="--", linewidth=1, label="0.5の基準線")
    axis.errorbar(x, y, yerr=[low, high], color="#009E73", marker="o", linewidth=2.3, capsize=5)
    for position, value, row in zip(x, y, rows):
        axis.annotate(
            f"{row['matching_direction_events']}/{row['directional_events']}",
            (position, value), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=11,
        )
    axis.set(xlim=(-0.54, -0.06), ylim=(0, 1.05), xticks=TIMEPOINTS)
    axis.grid(axis="y", alpha=0.25)
    axis.set_xlabel("相手後衛のインパクトからの時点（秒）")
    axis.set_ylabel("方向一致率（動いた試技に条件付け）")
    axis.set_title(
        "コート平面正規化後の前衛横方向動作と実打球方向\n"
        "変位閾値：コート幅の2.0%／ラベル＝一致動作数／方向判定動作数"
    )
    figure.text(
        0.5, 0.015,
        "手前・奥は同一固定カメラのコート平面へ変換後に統合。停止（stay）は分母に含めない。\n"
        "誤差棒は95% Wilson信頼区間。1試合内の探索的記述値であり、予測能力・先読みの証拠ではない。",
        ha="center", va="bottom", fontsize=9,
    )
    figure.tight_layout(rect=(0.04, 0.10, 1, 0.94))
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_sensitivity(rows_by_threshold: dict[float, list[dict[str, str]]], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(10, 6.6))
    offsets = {0.010: -0.006, 0.015: 0.0, 0.020: 0.006}
    axis.axhline(0.5, color="#666666", linestyle="--", linewidth=1, label="0.5の基準線")
    for threshold in sorted(rows_by_threshold):
        rows = sorted(rows_by_threshold[threshold], key=lambda row: f(row["timepoint_sec"]))
        x = [f(row["timepoint_sec"]) + offsets.get(threshold, 0) for row in rows]
        y = [f(row["conditional_match_rate"]) for row in rows]
        axis.plot(x, y, marker="o", linewidth=2, color=COLORS.get(threshold, "black"), label=f"{100 * threshold:.1f}%")
        for position, value, row in zip(x, y, rows):
            axis.annotate(
                row["directional_events"], (position, value), xytext=(0, -13),
                textcoords="offset points", ha="center", fontsize=9, color=COLORS.get(threshold, "black"),
            )
    axis.set(xlim=(-0.54, -0.06), ylim=(0, 1.05), xticks=TIMEPOINTS)
    axis.grid(axis="y", alpha=0.25)
    axis.set_xlabel("相手後衛のインパクトからの時点（秒）")
    axis.set_ylabel("方向一致率（動いた試技に条件付け）")
    axis.legend(title="コート幅に対する変位閾値")
    axis.set_title("変位閾値への感度分析（点の下の数字＝方向判定動作数）")
    figure.text(
        0.5, 0.015,
        "閾値で一致率と分母が変わるため、単一時点・単一閾値の高い率は結論に採用しない。",
        ha="center", va="bottom", fontsize=9,
    )
    figure.tight_layout(rect=(0.04, 0.08, 1, 0.94))
    figure.savefig(output, dpi=220)
    plt.close(figure)


def player_rows(event_rows: list[dict[str, str]], threshold: float) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in event_rows:
        grouped[(row["target_front_player_id"], row["timepoint_sec"])].append(row)
    result: list[dict[str, object]] = []
    for (player, timepoint), group in sorted(grouped.items(), key=lambda item: (item[0][0], float(item[0][1]))):
        directional = [row for row in group if row["movement_is_directional"] == "1"]
        matched = sum(row["direction_matches_actual"] == "1" for row in directional)
        result.append({
            "player": player,
            "timepoint": float(timepoint),
            "events": len(group),
            "directional": len(directional),
            "matches": matched,
            "rate": None if not directional else matched / len(directional),
            "threshold": threshold,
        })
    return result


def write_report(
    rows_by_threshold: dict[float, list[dict[str, str]]],
    primary_event_rows: list[dict[str, str]],
    calibration: dict[str, object],
    output: Path,
) -> None:
    primary = rows_by_threshold[0.020]
    validation = calibration["validation"]
    mapped = validation["mapped_held_out_net_points"]
    errors = validation["errors_in_normalized_court_units"]
    lines = [
        "# コート平面正規化後の前衛動作と実打球方向の対応",
        "",
        "## 手前・奥の統合方法",
        "",
        "- 固定主カメラのコート四隅を人手指定し、画像上の両足位置を正規化コート平面へ射影した。",
        "- x座標は画面左サイドラインを0、画面右サイドラインを1とした。各前衛のカメラ側（手前／奥）を用いて、x方向を本人から見た左右へ変換した。",
        "- 四隅は変換の計算に用い、ネット両端は計算に使わない検証点とした。",
        f"- 検証点の推定値は左ネット端 ({mapped[0][0]:.3f}, {mapped[0][1]:.3f})、右ネット端 ({mapped[1][0]:.3f}, {mapped[1][1]:.3f})。誤差はそれぞれ {errors[0]:.3f}、{errors[1]:.3f} コート単位で、最大 {validation['maximum_error_in_normalized_court_units']:.3f} だった。",
        "- これにより画面pxの遠近差は補正できる。ただし、同一試合・同一カメラにしか使えず、P1とP2の個人差を消すものではない。",
        "",
        "## 統合表：変位閾値をコート幅の2.0%とした場合",
        "",
        "|時点|全試技|方向判定試技|一致動作|条件付き一致率（95% Wilson CI）|",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(primary, key=lambda value: f(value["timepoint_sec"])):
        lines.append(
            f"|{f(row['timepoint_sec']):.1f} s|{row['events']}|{row['directional_events']}|"
            f"{row['matching_direction_events']}|{100 * f(row['conditional_match_rate']):.1f}% "
            f"({100 * f(row['conditional_match_wilson95_low']):.1f}–{100 * f(row['conditional_match_wilson95_high']):.1f}%)|"
        )
    lines.extend([
        "",
        "## 閾値感度（方向一致率、分母は方向判定試技数）",
        "",
        "|変位閾値|−0.5 s|−0.3 s|−0.2 s|−0.1 s|",
        "|---:|---:|---:|---:|",
    ])
    for threshold in sorted(rows_by_threshold):
        values = {f(row["timepoint_sec"]): row for row in rows_by_threshold[threshold]}
        cells = [
            f"{100 * f(values[time]['conditional_match_rate']):.1f}% ({values[time]['matching_direction_events']}/{values[time]['directional_events']})"
            for time in TIMEPOINTS
        ]
        lines.append(f"|{100 * threshold:.1f}%|" + "|".join(cells) + "|")
    lines.extend([
        "",
        "## 選手別の確認（2.0%閾値）",
        "",
        "|選手|時点|全試技|方向判定試技|一致動作|一致率|",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in player_rows(primary_event_rows, 0.020):
        rate = "—" if row["rate"] is None else f"{100 * float(row['rate']):.1f}%"
        lines.append(
            f"|{row['player']}|{float(row['timepoint']):.1f} s|{row['events']}|{row['directional']}|"
            f"{row['matches']}|{rate}|"
        )
    at_minus_02 = next(row for row in primary if f(row["timepoint_sec"]) == -0.2)
    at_minus_02_low = next(row for row in rows_by_threshold[0.010] if f(row["timepoint_sec"]) == -0.2)
    lines.extend([
        "",
        "## 解釈上の結論",
        "",
        f"- 2.0%閾値の−0.2 sでは {at_minus_02['matching_direction_events']}/{at_minus_02['directional_events']} = {100 * f(at_minus_02['conditional_match_rate']):.1f}% となった。しかし方向判定試技は{at_minus_02['directional_events']}件だけである。",
        f"- 同じ時点でも1.0%閾値では {at_minus_02_low['matching_direction_events']}/{at_minus_02_low['directional_events']} = {100 * f(at_minus_02_low['conditional_match_rate']):.1f}% となる。閾値により率・分母が変わるため、90%という単独の値を研究結論にしてはならない。",
        "- 現時点で主張できるのは、固定主カメラに対する手動コート較正とAI姿勢推定を組み合わせ、手前・奥を正規化座標で統合して時点別対応を記述できたことである。",
        "- 前衛が何秒前から打球方向を予測していたか、または前衛動作が打球を生じさせたという因果的・一般的な結論は、この1試合の結果からは出せない。",
        "",
        "## 次に必要な検証",
        "",
        "1. 変位閾値を、方向ラベルを参照しない開発用データまたは人手初動ラベルで固定する。",
        "2. 複数試合で同じ較正・追跡・監査処理を繰り返し、選手・打者・サーブ／レシーブ・開始位置を記録する。",
        "3. 最終評価では、開発時に固定した閾値と時点だけを用いる。",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", required=True)
    parser.add_argument("--primary-event-timepoints", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inputs = [*args.summary, args.primary_event_timepoints, args.calibration]
    if any(not path.is_file() for path in inputs):
        raise SystemExit(f"Required input missing: {[str(path) for path in inputs if not path.is_file()]}")
    rows_by_threshold: dict[float, list[dict[str, str]]] = {}
    for path in args.summary:
        rows = read(path)
        if not rows:
            raise SystemExit(f"Summary is empty: {path}")
        threshold = threshold_of(rows)
        if threshold in rows_by_threshold:
            raise SystemExit(f"Duplicate threshold {threshold}")
        rows_by_threshold[threshold] = rows
    if set(COLORS) - set(rows_by_threshold):
        raise SystemExit("Summaries at 1.0%, 1.5%, and 2.0% court-width thresholds are required")
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    if calibration.get("validation", {}).get("status") != "usable":
        raise SystemExit("Calibration is not usable")
    primary_events = read(args.primary_event_timepoints)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main_figure = args.output_dir / "match1_court_plane_directional_motion_t0p020.png"
    sensitivity_figure = args.output_dir / "match1_court_plane_directional_motion_threshold_sensitivity.png"
    report = args.output_dir / "match1_court_plane_directional_motion_report.md"
    plot_main(rows_by_threshold[0.020], main_figure)
    plot_sensitivity(rows_by_threshold, sensitivity_figure)
    write_report(rows_by_threshold, primary_events, calibration, report)
    print(f"Wrote {main_figure}")
    print(f"Wrote {sensitivity_figure}")
    print(f"Wrote {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
