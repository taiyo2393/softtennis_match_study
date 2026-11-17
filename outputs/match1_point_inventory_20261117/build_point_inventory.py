"""Turn the human transcription of match1 scoreboard sheets into a point list.

Read from review_sheets/sheet_01..08.jpg on 2026-11-17 (Claude, reviewer).
Row 1 of the display is 橋場柊一郎・菊山太陽 (H), row 2 is 上岡俊介・丸山海斗 (K).
Points are stored as counts (an "A" for K after 3-3 is written as 3-4).

Unlike match2, match1 briefly shows the game-winning score (e.g. 0-4, 6-4)
before the games column changes. Such a "finished" display is not a new point.
The video ends at 6-5 in the final game; the last point is counted, but its
winner is not shown on the scoreboard.
"""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (first_segment, last_segment, games_H, games_K, points_H, points_K)
RUNS = [
    (1, 1, 0, 0, 0, 0), (2, 4, 0, 0, 0, 1), (5, 7, 0, 0, 0, 2), (8, 9, 0, 0, 0, 3), (10, 10, 0, 0, 0, 4),
    (11, 11, 0, 1, 0, 0), (12, 12, 0, 1, 0, 1), (13, 15, 0, 1, 0, 2), (16, 17, 0, 1, 0, 3),
    (18, 21, 0, 1, 1, 3), (22, 24, 0, 1, 2, 3), (25, 27, 0, 1, 3, 3), (28, 29, 0, 1, 3, 4),
    (30, 30, 0, 1, 4, 4), (31, 32, 0, 1, 5, 4), (33, 36, 0, 1, 6, 4),
    (37, 37, 1, 1, 0, 0), (38, 40, 1, 1, 0, 1), (41, 41, 1, 1, 1, 1), (42, 43, 1, 1, 2, 1),
    (44, 46, 1, 1, 2, 2), (47, 48, 1, 1, 3, 2),
    (49, 49, 2, 1, 0, 0), (50, 50, 2, 1, 0, 1), (51, 55, 2, 1, 0, 2), (56, 58, 2, 1, 0, 3),
    (59, 60, 2, 1, 0, 4),
    (61, 61, 2, 2, 0, 0), (62, 65, 2, 2, 0, 1), (66, 69, 2, 2, 1, 1), (70, 72, 2, 2, 1, 2),
    (73, 74, 2, 2, 2, 2), (75, 76, 2, 2, 2, 3), (77, 79, 2, 2, 3, 3), (80, 83, 2, 2, 3, 4),
    (84, 84, 2, 2, 3, 5),
    (85, 89, 2, 3, 0, 0), (90, 92, 2, 3, 0, 1), (93, 95, 2, 3, 1, 1), (96, 98, 2, 3, 2, 1),
    (99, 100, 2, 3, 3, 1), (101, 102, 2, 3, 3, 2), (103, 103, 2, 3, 4, 2),
    (104, 106, 3, 3, 0, 0), (107, 109, 3, 3, 0, 1), (110, 112, 3, 3, 1, 1), (113, 114, 3, 3, 1, 2),
    (115, 116, 3, 3, 1, 3), (117, 117, 3, 3, 1, 4),
    (118, 119, 3, 4, 0, 0), (120, 121, 3, 4, 0, 1), (122, 124, 3, 4, 1, 1), (125, 127, 3, 4, 2, 1),
    (128, 134, 3, 4, 2, 2), (135, 138, 3, 4, 3, 2), (139, 142, 3, 4, 3, 3), (143, 145, 3, 4, 4, 3),
    (146, 146, 3, 4, 5, 3),
    (147, 151, 4, 4, 0, 0), (152, 153, 4, 4, 1, 0), (154, 154, 4, 4, 1, 1), (155, 160, 4, 4, 2, 1),
    (161, 164, 4, 4, 3, 1), (165, 166, 4, 4, 3, 2), (167, 168, 4, 4, 3, 3), (169, 170, 4, 4, 4, 3),
    (171, 172, 4, 4, 4, 4), (173, 175, 4, 4, 4, 5), (176, 176, 4, 4, 5, 5), (177, 180, 4, 4, 6, 5),
]


def finished(state) -> bool:
    gh, gk, ph, pk = state
    target = 7 if gh == gk == 4 else 4
    return max(ph, pk) >= target and abs(ph - pk) >= 2


def main() -> None:
    segments = {int(s["segment_id"]): s for s in csv.DictReader((HERE / "stable_segments.csv").open(encoding="utf-8-sig"))}
    covered = [sid for first, last, *_ in RUNS for sid in range(first, last + 1)]
    assert covered == sorted(segments), "every segment must be transcribed exactly once, in order"

    with (HERE / "score_transcription.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["segment_id", "start_frame", "rep_frame", "games_H", "games_K", "points_H", "points_K"])
        for first, last, *state in RUNS:
            for sid in range(first, last + 1):
                w.writerow([sid, segments[sid]["start_frame"], segments[sid]["rep_frame"], *state])

    rows = []
    for (first, _, *before), (next_first, _, *after) in zip(RUNS, RUNS[1:]):
        if finished(before):
            continue  # the game was already decided; the games column is just catching up
        new_game = after[0] != before[0] or after[1] != before[1]
        if new_game:
            winner = "H" if after[0] > before[0] else "K"
            after_text = "game"
        else:
            dh, dk = after[2] - before[2], after[3] - before[3]
            assert (dh, dk) in ((1, 0), (0, 1)), (before, after)
            winner = "H" if dh else "K"
            after_text = f"{after[2]}-{after[3]}"
        rows.append(dict(game=before[0] + before[1] + 1, games_before=f"{before[0]}-{before[1]}",
                         score_before=f"{before[2]}-{before[3]}", score_after=after_text,
                         point_winner=winner, winner_source="scoreboard",
                         display_from_frame=int(segments[first]["start_frame"]),
                         next_display_from_frame=int(segments[next_first]["start_frame"]),
                         before_state_source=f"segment_{first:04d}"))
    last_first, _, *last_state = RUNS[-1]
    assert not finished(last_state)
    rows.append(dict(game=9, games_before="4-4", score_before="6-5", score_after="video ends",
                     point_winner="", winner_source="not shown (video ends before the next display)",
                     display_from_frame=int(segments[last_first]["start_frame"]), next_display_from_frame="",
                     before_state_source=f"segment_{last_first:04d}"))
    for number, row in enumerate(rows, 1):
        row["point"] = number
        row["is_final_game"] = row["game"] == 9
    columns = ["point", "game", "is_final_game", "games_before", "score_before", "score_after", "point_winner",
               "winner_source", "display_from_frame", "next_display_from_frame", "before_state_source"]
    with (HERE / "point_inventory.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    per_game = {}
    for r in rows:
        per_game[r["game"]] = per_game.get(r["game"], 0) + 1
    print(f"points={len(rows)}  H won={sum(r['point_winner'] == 'H' for r in rows)}  "
          f"K won={sum(r['point_winner'] == 'K' for r in rows)}  unknown=1  per game={per_game}")


if __name__ == "__main__":
    main()
