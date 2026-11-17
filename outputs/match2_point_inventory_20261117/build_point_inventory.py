"""Turn the human transcription of the scoreboard sheets into a point list.

The transcription below was read from review_sheets/sheet_01..05.jpg on
2026-11-17 (Claude, reviewer). Row 1 of the display is 内本隆文・内田理久 (U),
row 2 is 船水颯人・上松俊貴 (F). Each entry is
segment_id: (games_U, games_F, points_U, points_F), where points may be
"A" (advantage) or "" (blank beside an advantage). None = no scoreboard.
"""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent

NO_BOARD = {1, 2, 3, 4, 5, 6}
T = {
    7: (0, 0, "0", "0"), 8: (0, 0, "0", "1"), 9: (0, 0, "1", "1"), 10: (0, 0, "1", "1"),
    11: (0, 0, "1", "1"), 12: (0, 0, "2", "1"), 13: (0, 0, "2", "1"), 14: (0, 0, "3", "1"),
    15: (0, 0, "3", "2"), 16: (0, 0, "3", "3"), 17: (0, 0, "3", "3"), 18: (0, 0, "A", ""),
    19: (0, 0, "4", "4"), 20: (0, 0, "4", "4"), 21: (0, 0, "A", ""), 22: (0, 0, "A", ""),
    23: (0, 0, "A", ""),
    24: (1, 0, "0", "0"), 25: (1, 0, "1", "0"), 26: (1, 0, "1", "0"), 27: (1, 0, "1", "1"),
    28: (1, 0, "1", "2"), 29: (1, 0, "2", "2"), 30: (1, 0, "2", "3"), 31: (1, 0, "3", "3"),
    32: (1, 0, "3", "3"), 33: (1, 0, "", "A"),
    34: (1, 1, "0", "0"), 35: (1, 1, "0", "0"), 36: (1, 1, "0", "0"), 37: (1, 1, "0", "1"),
    38: (1, 1, "0", "1"), 39: (1, 1, "0", "1"), 40: (1, 1, "2", "1"), 41: (1, 1, "2", "2"),
    42: (1, 1, "2", "3"), 43: (1, 1, "3", "3"), 44: (1, 1, "", "A"),
    45: (1, 2, "0", "0"), 46: (1, 2, "1", "0"), 47: (1, 2, "1", "1"), 48: (1, 2, "2", "1"),
    49: (1, 2, "2", "1"), 50: (1, 2, "2", "2"), 51: (1, 2, "2", "2"), 52: (1, 2, "3", "2"),
    53: (1, 2, "3", "2"),
    54: (2, 2, "0", "0"), 55: (2, 2, "0", "0"), 56: (2, 2, "0", "1"), 57: (2, 2, "1", "1"),
    58: (2, 2, "2", "1"), 59: (2, 2, "2", "2"), 60: (2, 2, "3", "2"),
    61: (3, 2, "0", "0"), 62: (3, 2, "0", "0"), 63: (3, 2, "0", "0"), 64: (3, 2, "0", "1"),
    65: (3, 2, "0", "1"), 66: (3, 2, "1", "1"), 67: (3, 2, "2", "1"), 68: (3, 2, "2", "1"),
    69: (3, 2, "2", "1"), 70: (3, 2, "2", "2"), 71: (3, 2, "2", "2"), 72: (3, 2, "3", "2"),
    73: (3, 2, "3", "2"),
    74: (4, 2, "0", "0"), 75: (4, 2, "0", "0"), 76: (4, 2, "0", "1"), 77: (4, 2, "0", "2"),
    78: (4, 2, "1", "2"), 79: (4, 2, "1", "3"), 80: (4, 2, "1", "3"), 81: (4, 2, "1", "3"),
    82: (4, 3, "0", "0"), 83: (4, 3, "1", "0"), 84: (4, 3, "1", "1"), 85: (4, 3, "1", "1"),
    86: (4, 3, "1", "2"), 87: (4, 3, "1", "3"), 88: (4, 3, "1", "3"),
    89: (4, 4, "0", "0"), 90: (4, 4, "0", "0"), 91: (4, 4, "0", "1"), 92: (4, 4, "0", "1"),
    93: (4, 4, "0", "2"), 94: (4, 4, "0", "3"), 95: (4, 4, "0", "3"), 96: (4, 4, "0", "3"),
    97: (4, 4, "0", "4"), 98: (4, 4, "0", "4"), 99: (4, 4, "1", "4"), 100: (4, 4, "1", "4"),
    101: (4, 4, "1", "4"), 102: (4, 4, "2", "4"), 103: (4, 4, "2", "5"), 104: (4, 4, "3", "5"),
    105: (4, 4, "3", "6"), 106: (4, 4, "4", "6"), 107: (4, 4, "5", "6"), 108: (4, 4, "5", "6"),
    109: (4, 4, "5", "6"), 110: (4, 4, "5", "7"),
}
# States that must have occurred but were never shown long enough to be captured.
INFERRED = {(1, 1, "1", "1"): "between segment 39 (0-1) and 40 (2-1)"}
FINAL = {"games_U": 4, "games_F": 5, "winner": "F"}  # final game ended 5-7


def main() -> None:
    segments = list(csv.DictReader((HERE / "stable_segments.csv").open(encoding="utf-8-sig")))
    assert {int(s["segment_id"]) for s in segments} == NO_BOARD | set(T), "transcription incomplete"
    by_id = {int(s["segment_id"]): s for s in segments}

    # Filled transcription table.
    with (HERE / "score_transcription.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["segment_id", "start_frame", "rep_frame", "is_scoreboard_visible",
                    "games_U", "games_F", "points_U", "points_F"])
        for sid in sorted(by_id):
            s = by_id[sid]
            if sid in NO_BOARD:
                w.writerow([sid, s["start_frame"], s["rep_frame"], False, "", "", "", ""])
            else:
                w.writerow([sid, s["start_frame"], s["rep_frame"], True, *T[sid]])

    # Distinct consecutive states, with the inferred state inserted.
    states: list[tuple[tuple, int | None]] = []
    for sid in sorted(T):
        if not states or states[-1][0] != T[sid]:
            if T[sid] == (1, 1, "2", "1"):
                states.append(((1, 1, "1", "1"), None))
            states.append((T[sid], sid))
    # The last captured state (final game 5-7) is already the result of the last point.
    assert states[-1][0] == (4, 4, "5", "7")

    rows = []
    for number, ((before, sid), (after, next_sid)) in enumerate(zip(states, states[1:]), 1):
        gu, gf = before[0], before[1]
        game_number = gu + gf + 1
        if after[0] > gu:
            winner = "U"
        elif after[1] > gf:
            winner = "F"
        else:
            winner = point_winner(before, after)
        start = int(by_id[sid]["start_frame"]) if sid else ""
        end = int(by_id[next_sid]["start_frame"]) if next_sid else ""
        rows.append({
            "point": number, "game": game_number, "is_final_game": game_number == 9,
            "games_before": f"{gu}-{gf}", "score_before": f"{before[2] or '_'}-{before[3] or '_'}",
            "score_after": "game" if after[0] != gu or after[1] != gf or after[2] == "-" else f"{after[2] or '_'}-{after[3] or '_'}",
            "point_winner": winner,
            "display_from_frame": start, "next_display_from_frame": end,
            "before_state_source": f"segment_{sid:04d}" if sid else "inferred: " + INFERRED.get(before, ""),
        })
    with (HERE / "point_inventory.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    per_game = {}
    for r in rows:
        per_game[r["game"]] = per_game.get(r["game"], 0) + 1
    print(f"points={len(rows)}  U won={sum(r['point_winner'] == 'U' for r in rows)}  "
          f"F won={sum(r['point_winner'] == 'F' for r in rows)}  per game={per_game}")


ORDER = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}


def point_winner(before, after) -> str:
    bu, bf, au, af = before[2], before[3], after[2], after[3]
    if au == "A":
        return "U"
    if af == "A":
        return "F"
    if bu == "A" and au == "4":
        return "F"  # advantage U lost -> deuce
    if bf == "A" and af == "4":
        return "U"
    du, df = ORDER[au] - ORDER[bu], ORDER[af] - ORDER[bf]
    assert (du, df) in ((1, 0), (0, 1)), (before, after)
    return "U" if du == 1 else "F"


if __name__ == "__main__":
    main()
