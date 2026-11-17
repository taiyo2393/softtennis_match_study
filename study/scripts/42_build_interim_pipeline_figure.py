"""Create a rights-neutral pipeline figure for the interim preprint."""

from __future__ import annotations

from pathlib import Path

import matplotlib

# This script is run without a desktop display (for reproducible preprint figures).
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch


def main() -> int:
    project = Path(__file__).resolve().parents[1]
    output_dir = project / "analysis" / "interim_preprint"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "fig1_semi_automatic_pipeline.png"
    font_path = Path(r"C:\Windows\Fonts\YuGothM.ttc")
    font = FontProperties(fname=str(font_path)) if font_path.is_file() else None
    figure, axes = plt.subplots(figsize=(12, 3.5), dpi=300)
    axes.set_xlim(0, 12); axes.set_ylim(0, 4); axes.axis("off")

    boxes = [
        (0.25, 2.2, "固定主カメラ\n試合映像", "#e8f1fb"),
        (2.55, 2.2, "カット・別カメラ\n区間の除外", "#e8f1fb"),
        (4.85, 2.2, "MS-TrackNetV3\nボール候補・信頼度", "#eaf5ea"),
        (7.15, 2.2, "低信頼・欠損箇所の\n人手確認", "#fff5df"),
        (9.45, 2.2, "ボール候補付き\n分析用データ", "#eaf5ea"),
    ]
    for x, y, label, color in boxes:
        axes.add_patch(FancyBboxPatch((x, y), 1.8, 0.9, boxstyle="round,pad=0.04,rounding_size=0.08",
                                      linewidth=1.3, edgecolor="#333333", facecolor=color))
        axes.text(x + 0.9, y + 0.45, label, ha="center", va="center", fontsize=10.5, fontproperties=font)
    for left in (2.05, 4.35, 6.65, 8.95):
        axes.annotate("", xy=(left + 0.47, 2.65), xytext=(left, 2.65),
                      arrowprops=dict(arrowstyle="->", lw=1.4, color="#333333"))

    axes.add_patch(FancyBboxPatch((3.4, 0.45), 5.2, 0.9, boxstyle="round,pad=0.04,rounding_size=0.08",
                                  linewidth=1.2, linestyle="--", edgecolor="#555555", facecolor="#f7f7f7"))
    axes.text(6.0, 0.9, "ブラインド正解付与による予備評価\n（時間層化300フレーム：適合率・再現率・位置誤差）",
              ha="center", va="center", fontsize=10, fontproperties=font)
    axes.annotate("", xy=(5.75, 1.35), xytext=(5.75, 2.18), arrowprops=dict(arrowstyle="->", lw=1.2, color="#555555"))
    axes.annotate("", xy=(7.6, 1.35), xytext=(7.6, 2.18), arrowprops=dict(arrowstyle="->", lw=1.2, color="#555555"))
    axes.text(6.0, 3.65, "ソフトテニスダブルス映像における半自動ボール候補生成の処理フロー",
              ha="center", va="center", fontsize=13, fontweight="bold", fontproperties=font)
    figure.savefig(output, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
