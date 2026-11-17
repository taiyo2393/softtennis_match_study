"""Render the one-column Table 1 image for the interim proceeding."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties


PROJECT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT / "analysis" / "interim_preprint" / "table1_direction_agreement.png"


def font_properties() -> FontProperties | None:
    """Use the Japanese font installed on the study workstation when present."""
    font_path = Path(r"C:\Windows\Fonts\YuGothM.ttc")
    return FontProperties(fname=str(font_path)) if font_path.is_file() else None


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    font = font_properties()
    figure = plt.figure(figsize=(3.75, 1.22), dpi=400)
    axis = figure.add_axes((0, 0, 1, 1))
    axis.axis("off")

    table = axis.table(
        cellText=[
            ["−0.5秒", "14", "6", "42.9％"],
            ["−0.3秒", "25", "16", "64.0％"],
            ["−0.2秒", "15", "11", "73.3％"],
            ["−0.1秒", "22", "15", "68.2％"],
        ],
        colLabels=["時点", "判定試技", "一致", "一致率"],
        colWidths=[0.26, 0.28, 0.20, 0.26],
        cellLoc="center",
        colLoc="center",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11.0)
    last_row = len(table.get_celld()) // 4 - 1
    for (row, _column), cell in table.get_celld().items():
        cell.set_edgecolor("black")
        cell.set_text_props(fontproperties=font)
        cell.set_facecolor("white")
        cell.set_linewidth(0.0)
        cell.visible_edges = ""
        if row == 0:
            cell.visible_edges = "TB"
            cell.set_linewidth(0.8)
        elif row == last_row:
            cell.visible_edges = "B"
            cell.set_linewidth(0.8)

    figure.savefig(OUTPUT, bbox_inches="tight", pad_inches=0.02, facecolor="white")
    plt.close(figure)
    print(OUTPUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
