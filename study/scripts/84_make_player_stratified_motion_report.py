"""Create descriptive figures and a cautious report for pre-impact motion.

The inputs are the player-by-camera-end summaries made by script 83.  This is
not a prediction-model evaluation: it visualises the conditional descriptive
quantity P(movement direction = actual shot direction | movement != stay).
Camera ends stay separated because a pixel threshold has a different physical
meaning in the near and far parts of the broadcast image.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# This study is run on Windows and renders Japanese figure labels.  Yu Gothic is
# present on the research workstation; falling back to DejaVu would silently
# turn Japanese text into missing-glyph boxes.
plt.rcParams["font.family"] = ["Yu Gothic"]
plt.rcParams["axes.unicode_minus"] = False


TIMEPOINTS = [-0.5, -0.3, -0.2, -0.1]
COLORS = {10.0: "#0072B2", 15.0: "#D55E00", 20.0: "#009E73"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def f(value: str) -> float:
    return float(value)


def median(values: list[float]) -> float:
    values = sorted(values)
    midpoint = len(values) // 2
    return values[midpoint] if len(values) % 2 else (values[midpoint - 1] + values[midpoint]) / 2


def percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    return values[math.floor((len(values) - 1) * fraction)]


def camera_scale_diagnostic(near_pose: Path, far_pose: Path) -> list[dict[str, float | int | str]]:
    """Use ankle span only as a scale diagnostic, never as a court calibration."""
    results: list[dict[str, float | int | str]] = []
    for camera_end, path in (("near", near_pose), ("far", far_pose)):
        spans = []
        for row in read_csv(path):
            if row.get("ankle_pair_visible") != "1":
                continue
            spans.append(math.hypot(
                f(row["left_ankle_x_px"]) - f(row["right_ankle_x_px"]),
                f(row["left_ankle_y_px"]) - f(row["right_ankle_y_px"]),
            ))
        if not spans:
            raise SystemExit(f"No visible bilateral-ankle frames in {path}")
        width = median(spans)
        results.append({
            "camera_end": camera_end,
            "ankle_pair_frames": len(spans),
            "median_ankle_span_px": width,
            "p25_ankle_span_px": percentile(spans, 0.25),
            "p75_ankle_span_px": percentile(spans, 0.75),
            "raw_15px_as_ankle_span_ratio": 15 / width,
        })
    return results


def key(row: dict[str, str]) -> tuple[str, str]:
    return row["target_front_player_id"], row["target_front_camera_end"]


def describe_rate(row: dict[str, str]) -> str:
    return (
        f"{row['matching_direction_events']}/{row['directional_events']}"
        f" = {100 * f(row['conditional_match_rate']):.1f}%"
        f" (95% Wilson CI {100 * f(row['conditional_match_wilson95_low']):.1f}"
        f"–{100 * f(row['conditional_match_wilson95_high']):.1f}%)"
    )


def plot_main(rows: list[dict[str, str]], output: Path) -> None:
    strata = sorted({key(row) for row in rows})
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), sharex=True, sharey=True)
    for axis, stratum in zip(axes.flat, strata):
        group = sorted((row for row in rows if key(row) == stratum), key=lambda row: f(row["timepoint_sec"]))
        x = [f(row["timepoint_sec"]) for row in group]
        y = [f(row["conditional_match_rate"]) for row in group]
        low = [value - f(row["conditional_match_wilson95_low"]) for value, row in zip(y, group)]
        high = [f(row["conditional_match_wilson95_high"]) - value for value, row in zip(y, group)]
        axis.axhline(0.5, color="#666666", linestyle="--", linewidth=1, label="0.5の基準線")
        axis.errorbar(x, y, yerr=[low, high], color="#D55E00", marker="o", linewidth=2, capsize=4)
        for position, value, row in zip(x, y, group):
            axis.annotate(
                f"{row['matching_direction_events']}/{row['directional_events']}",
                (position, value), xytext=(0, 8), textcoords="offset points",
                ha="center", fontsize=9,
            )
        player, camera_end = stratum
        end_label = "奥側" if camera_end == "far" else "手前側"
        axis.set_title(f"{player}・{end_label}（全{group[0]['events']}件）")
        axis.set_xlim(-0.54, -0.06)
        axis.set_ylim(0, 1.05)
        axis.set_xticks(TIMEPOINTS)
        axis.grid(axis="y", alpha=0.25)
    figure.supylabel("方向一致率（動いた試技に条件付け）")
    figure.suptitle(
        "打球前における前衛の横方向動作と実打球方向の対応\n"
        "画面上の変位15 px：ラベル＝一致動作数／方向判定動作数",
        y=0.98,
    )
    figure.text(0.5, 0.085, "相手後衛のインパクトからの時点（秒）", ha="center", va="bottom")
    figure.text(
        0.5, 0.025,
        "停止（stay）は条件付き一致率の分母に含めない。誤差棒は95% Wilson信頼区間。\n"
        "1試合内の記述値であり、一般的な予測能力・先読みを示すものではない。",
        ha="center", va="bottom", fontsize=9,
    )
    figure.tight_layout(rect=(0.03, 0.13, 1, 0.91))
    figure.savefig(output, dpi=220)
    plt.close(figure)


def plot_sensitivity(rows_by_threshold: dict[float, list[dict[str, str]]], output: Path) -> None:
    strata = sorted({key(row) for rows in rows_by_threshold.values() for row in rows})
    figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.8), sharex=True, sharey=True)
    offsets = {10.0: -0.012, 15.0: 0.0, 20.0: 0.012}
    for axis, stratum in zip(axes.flat, strata):
        axis.axhline(0.5, color="#666666", linestyle="--", linewidth=1)
        for threshold in sorted(rows_by_threshold):
            group = sorted(
                (row for row in rows_by_threshold[threshold] if key(row) == stratum),
                key=lambda row: f(row["timepoint_sec"]),
            )
            x = [f(row["timepoint_sec"]) + offsets[threshold] for row in group]
            y = [f(row["conditional_match_rate"]) for row in group]
            axis.plot(x, y, marker="o", linewidth=1.8, color=COLORS.get(threshold, "black"), label=f"{threshold:.0f} px")
            for position, value, row in zip(x, y, group):
                axis.annotate(
                    row["directional_events"], (position, value), xytext=(0, -12),
                    textcoords="offset points", ha="center", fontsize=7,
                    color=COLORS.get(threshold, "black"),
                )
        player, camera_end = stratum
        events = next(row["events"] for row in rows_by_threshold[15.0] if key(row) == stratum)
        end_label = "奥側" if camera_end == "far" else "手前側"
        axis.set_title(f"{player}・{end_label}（全{events}件）")
        axis.set_xlim(-0.54, -0.06)
        axis.set_ylim(0, 1.05)
        axis.set_xticks(TIMEPOINTS)
        axis.grid(axis="y", alpha=0.25)
    axes.flat[0].legend(title="変位閾値", loc="lower left")
    figure.supylabel("方向一致率（動いた試技に条件付け）")
    figure.suptitle(
        "画面上の変位閾値に対する感度分析\n"
        "点の下の数字＝方向判定動作となった試技数",
        y=0.98,
    )
    figure.text(0.5, 0.085, "相手後衛のインパクトからの時点（秒）", ha="center", va="bottom")
    figure.text(
        0.5, 0.025,
        "閾値を変えると一致率と分母の両方が変化する。これは閾値を選ぶためではなく、結論の頑健性を確認する分析である。",
        ha="center", va="bottom", fontsize=9,
    )
    figure.tight_layout(rect=(0.03, 0.13, 1, 0.91))
    figure.savefig(output, dpi=220)
    plt.close(figure)


def pooled_reference_rows(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    """Create a clearly marked raw-pixel pooled table for descriptive inventory."""
    output = []
    for timepoint in TIMEPOINTS:
        group = [row for row in rows if f(row["timepoint_sec"]) == timepoint]
        events = sum(int(row["events"]) for row in group)
        directional = sum(int(row["directional_events"]) for row in group)
        matches = sum(int(row["matching_direction_events"]) for row in group)
        if directional:
            rate = matches / directional
            # Wilson interval duplicated here to keep this report independent of a pooled model.
            z = 1.96
            denom = 1 + z * z / directional
            centre = (rate + z * z / (2 * directional)) / denom
            radius = z * math.sqrt(rate * (1 - rate) / directional + z * z / (4 * directional * directional)) / denom
            low, high = max(0, centre - radius), min(1, centre + radius)
        else:
            rate = low = high = None
        output.append({
            "timepoint_sec": timepoint,
            "events": events,
            "directional_events": directional,
            "matching_direction_events": matches,
            "conditional_match_rate": rate,
            "wilson95_low": low,
            "wilson95_high": high,
        })
    return output


def write_pooled_reference_csv(rows: list[dict[str, object]], output: Path) -> None:
    fields = list(rows[0])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_scale_diagnostic_csv(rows: list[dict[str, float | int | str]], output: Path) -> None:
    fields = list(rows[0])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    rows_by_threshold: dict[float, list[dict[str, str]]],
    scale_rows: list[dict[str, float | int | str]],
    pooled_rows: list[dict[str, object]],
    output: Path,
) -> None:
    primary = rows_by_threshold[15.0]
    identity_counts: dict[str, int] = defaultdict(int)
    end_counts: dict[tuple[str, str], int] = {}
    for player, camera_end in sorted({key(row) for row in primary}):
        group = [row for row in primary if key(row) == (player, camera_end)]
        end_counts[(player, camera_end)] = int(group[0]["events"])
        identity_counts[player] += int(group[0]["events"])
    scale_by_end = {str(row["camera_end"]): row for row in scale_rows}
    near_scale = scale_by_end["near"]
    far_scale = scale_by_end["far"]
    relative_difference = float(far_scale["raw_15px_as_ankle_span_ratio"]) / float(near_scale["raw_15px_as_ankle_span_ratio"])
    lines = [
        "# 前衛横方向動作と実打球方向：選手・カメラ位置別の記述結果",
        "",
        "## 対象と定義",
        "",
        "- 固定主カメラの1試合から、人手確認済みの相手後衛インパクト98件を対象とした。",
        "- 対象前衛の選手IDはP1（42件）とP2（56件）として人手確認した。",
        "- 各時点の直前100 msにおける横方向変位が閾値を超えた場合だけ、前衛本人から見た左右と実打球方向の一致率を算出した。",
        "- 主指標は `P(動作方向 = 実打球方向 | 動作方向が判定可能)`。停止（stay）は正解に含めない。",
        "- 画面pxの変位は遠近で物理量が異なるため、主解析では手前・奥およびP1・P2を混合しない。",
        "",
        "## 手前・奥の統合可否の点検",
        "",
        f"- 両足間距離の中央値は、手前 {float(near_scale['median_ankle_span_px']):.1f} px、奥 {float(far_scale['median_ankle_span_px']):.1f} pxだった。",
        f"- 生の15 pxは、手前で足間距離の {100 * float(near_scale['raw_15px_as_ankle_span_ratio']):.1f}%、奥で {100 * float(far_scale['raw_15px_as_ankle_span_ratio']):.1f}% に相当する。奥側では相対的に約 {relative_difference:.2f} 倍の変位を要求することになる。",
        "- 足間距離は姿勢にも依存するため、これは正式な正規化ではなく縮尺差の診断である。この診断だけで正規化係数は決められないが、生の15 pxを共通の運動閾値と仮定する根拠もない。",
        "- よって以下の全体表はデータ量を示す参考集計に留め、主結果・統計的推論には使わない。統合した主解析には、コート平面への射影（ホモグラフィ）等で変位を正規化する必要がある。",
        "",
        "## 15 px の記述表",
        "",
        "|選手|カメラ位置|時点|全試技|方向判定試技|一致率（95% Wilson CI）|",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in sorted(primary, key=lambda value: (value["target_front_player_id"], value["target_front_camera_end"], f(value["timepoint_sec"]))):
        end_label = "奥側" if row["target_front_camera_end"] == "far" else "手前側"
        lines.append(
            f"|{row['target_front_player_id']}|{end_label}|"
            f"{f(row['timepoint_sec']):.1f} s|{row['events']}|{row['directional_events']}|{describe_rate(row)}|"
        )
    p1_far_late = next(row for row in primary if key(row) == ("P1", "far") and f(row["timepoint_sec"]) == -0.1)
    p2_far_late = next(row for row in primary if key(row) == ("P2", "far") and f(row["timepoint_sec"]) == -0.1)
    p2_near_late_t20 = next(row for row in rows_by_threshold[20.0] if key(row) == ("P2", "near") and f(row["timepoint_sec"]) == -0.1)
    lines.extend([
        "",
        "## 全体表（参考：手前・奥を生の15 pxで統合）",
        "",
        "|時点|全試技|方向判定試技|一致動作|条件付き一致率（95% Wilson CI）|",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in pooled_rows:
        lines.append(
            f"|{float(row['timepoint_sec']):.1f} s|{row['events']}|{row['directional_events']}|"
            f"{row['matching_direction_events']}|{100 * float(row['conditional_match_rate']):.1f}% "
            f"({100 * float(row['wilson95_low']):.1f}–{100 * float(row['wilson95_high']):.1f}%)|"
        )
    lines.extend([
        "",
        "## 解釈上の結論",
        "",
        f"- 例として、P1・奥側・-0.1 sは{describe_rate(p1_far_late)}である一方、P2・奥側・同時点は{describe_rate(p2_far_late)}である。選手間で方向が一貫しない。",
        f"- 閾値を20 pxへ変更するとP2・手前側・-0.1 sは{describe_rate(p2_near_late_t20)}となるが、方向判定試技は4件だけである。高い率を単独で結論に採用してはならない。",
        "- よって本データは、打球前の前衛横方向動作と実打球方向を時点別に記述する処理の実現可能性を示す。一方で、前衛が何秒前から方向を予測していたか、または一般的に方向を読んでいたかを示す証拠にはならない。全体表もこの結論を変えない。",
        "",
        "## この段階での限界",
        "",
        "1. 1試合のみであり、同じ選手・対戦相手・戦術配置への依存を分離できない。",
        "2. 打者位置、サーブ／レシーブ、球種、直前のラリー、前衛の開始位置を統制していない。これらは方向と前衛動作の両方に関係し得る。",
        "3. 10・15・20 pxの感度分析で率と方向判定試技数が変化した。閾値の事後選択はできない。",
        "4. 初動時刻を成果にするなら、別途、人手初動ラベルとの誤差評価およびコート座標または身体寸法による正規化が必要である。",
        "",
        "## 次の検証計画",
        "",
        "- 複数試合を追加し、選手ID・カメラ位置・打者・サーブ／レシーブ・開始位置を記録する。",
        "- 開発用データで動作定義を固定し、最終評価データでは定義や閾値を変更しない。",
        "- 初動時刻を扱う場合は、少なくとも20件程度の独立した人手初動ラベルに対して、AI追跡からの推定誤差を報告する。",
    ])
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, action="append", required=True, help="One script-83 summary CSV per threshold")
    parser.add_argument("--near-pose", type=Path, required=True, help="Near-camera audited ankle-pose CSV")
    parser.add_argument("--far-pose", type=Path, required=True, help="Far-camera audited ankle-pose CSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if len(args.summary) < 2:
        raise SystemExit("Provide at least two threshold summaries for sensitivity checking")
    rows_by_threshold: dict[float, list[dict[str, str]]] = {}
    for path in [*args.summary, args.near_pose, args.far_pose]:
        if not path.is_file():
            raise SystemExit(f"Required input not found: {path}")
    for path in args.summary:
        rows = read_csv(path)
        if not rows:
            raise SystemExit(f"Summary has no rows: {path}")
        threshold = f(rows[0]["movement_threshold_px"])
        if any(f(row["movement_threshold_px"]) != threshold for row in rows):
            raise SystemExit(f"More than one threshold is present in {path}")
        if threshold in rows_by_threshold:
            raise SystemExit(f"Duplicate threshold: {threshold}")
        rows_by_threshold[threshold] = rows
    if 15.0 not in rows_by_threshold:
        raise SystemExit("A 15 px summary is required as the baseline descriptive display")
    expected = {(player, end, time) for player in ("P1", "P2") for end in ("far", "near") for time in TIMEPOINTS}
    found = {
        (row["target_front_player_id"], row["target_front_camera_end"], f(row["timepoint_sec"]))
        for rows in rows_by_threshold.values() for row in rows
    }
    missing = expected - found
    if missing:
        raise SystemExit(f"Missing player/camera/time cells: {sorted(missing)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    main_figure = args.output_dir / "match1_player_camera_directional_motion_t15.png"
    sensitivity_figure = args.output_dir / "match1_player_camera_directional_motion_threshold_sensitivity.png"
    report = args.output_dir / "match1_player_camera_directional_motion_report.md"
    pooled_reference = args.output_dir / "match1_overall_raw15px_reference_only.csv"
    scale_diagnostic = args.output_dir / "match1_camera_end_pixel_scale_diagnostic.csv"
    scale_rows = camera_scale_diagnostic(args.near_pose, args.far_pose)
    pooled_rows = pooled_reference_rows(rows_by_threshold[15.0])
    plot_main(rows_by_threshold[15.0], main_figure)
    plot_sensitivity(rows_by_threshold, sensitivity_figure)
    write_pooled_reference_csv(pooled_rows, pooled_reference)
    write_scale_diagnostic_csv(scale_rows, scale_diagnostic)
    write_report(rows_by_threshold, scale_rows, pooled_rows, report)
    print(f"Wrote {main_figure}")
    print(f"Wrote {sensitivity_figure}")
    print(f"Wrote {report}")
    print(f"Wrote {pooled_reference}")
    print(f"Wrote {scale_diagnostic}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
