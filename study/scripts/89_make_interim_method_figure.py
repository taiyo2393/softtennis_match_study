"""Render the method overview figure used in the interim-proceedings draft.

The figure makes the division between automated extraction and human
confirmation explicit.  It intentionally does not claim that either the
impact frame or shot direction is fully automatic.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.patches import FancyBboxPatch


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "analysis" / "interim_preprint" / "fig1_front_motion_direction_agreement.png"


def font_properties() -> FontProperties | None:
    """Use a Japanese font when this script is run on the study workstation."""
    font_path = Path(r"C:\Windows\Fonts\YuGothM.ttc")
    return FontProperties(fname=str(font_path)) if font_path.is_file() else None


def add_box(
    ax,
    x: float,
    y: float,
    label: str,
    color: str,
    font: FontProperties | None,
    width: float,
    height: float,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.018,rounding_size=0.035",
            linewidth=1.1, edgecolor="#333333", facecolor=color,
        )
    )
    ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=10.2, linespacing=1.18, fontproperties=font)


def arrow(ax, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.25, color="#333333"))


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    font = font_properties()
    # This layout is deliberately sized for one Word column rather than for a
    # two-column-wide figure in an A4 proceeding.
    fig = plt.figure(figsize=(3.80, 1.90), dpi=400)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 4.20)
    ax.set_ylim(0, 2.10)
    ax.axis("off")

    # Blue is the video/data path, yellow human confirmation, and green
    # automated position extraction.  The parallel boxes are combined later.
    add_box(ax, 0.46, 1.69, "コート後方からの試合映像", "#e8f1fb", font, 3.28, 0.32)
    add_box(ax, 0.05, 0.96, "インパクト・方向を\n人手で確認", "#fff5df", font, 1.96, 0.56)
    add_box(ax, 2.19, 0.96, "前衛の足首位置を\n自動で読み取る", "#eaf5ea", font, 1.96, 0.56)
    add_box(
        ax, 0.23, 0.25,
        "人手で確認し，コート上の位置へ変換\n時点別に方向の一致率を集計",
        "#e8f1fb", font, 3.74, 0.56,
    )

    arrow(ax, (1.62, 1.66), (1.02, 1.55))
    arrow(ax, (2.58, 1.66), (3.18, 1.55))
    arrow(ax, (1.02, 0.93), (1.62, 0.84))
    arrow(ax, (3.18, 0.93), (2.58, 0.84))
    ax.text(2.10, 0.075, "評価時点：−0.5，−0.3，−0.2，−0.1秒", ha="center", va="center", fontsize=9.2, fontproperties=font)
    fig.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(fig)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
