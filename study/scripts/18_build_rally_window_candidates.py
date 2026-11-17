"""Pair confirmed rally-end candidates with preceding main-camera entries.

Output rows are review windows, not confirmed rallies or trials.  A window is
excluded when two purported rally ends are implausibly close and need review.
"""

from __future__ import annotations

import argparse, csv
from pathlib import Path


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file: return list(csv.DictReader(file))


def write(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer=csv.DictWriter(file,fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--video-id",required=True); parser.add_argument("--min-end-gap-sec",type=float,default=5.0); parser.add_argument("--min-enter-to-end-sec",type=float,default=3.0,help="Ignore a transient main-camera-enter candidate this many seconds or less before a confirmed rally end.")
    args=parser.parse_args(); directory=Path(__file__).resolve().parents[1]/"data"/"segments"; rows=read(directory/f"{args.video_id}_rally_boundary_candidates.csv")
    # Rebuilding after a boundary correction must not discard reviewed starts.
    # The end-boundary ID is stable even if a window number changes.
    previous_path=directory/f"{args.video_id}_rally_window_candidates.csv"
    previous={row["end_boundary_id"]: row for row in read(previous_path)} if previous_path.exists() else {}
    ends=sorted([row for row in rows if row["human_status"]=="confirmed" and row["human_boundary_type"]=="rally_end"],key=lambda r:float(r["human_timestamp_sec"] or r["timestamp_sec"]))
    entries=sorted([row for row in rows if row["candidate_type"]=="main_camera_enter_candidate"],key=lambda r:float(r["timestamp_sec"]))
    conflicts=[]; blocked=set()
    for earlier,later in zip(ends,ends[1:]):
        first=float(earlier["human_timestamp_sec"] or earlier["timestamp_sec"]); second=float(later["human_timestamp_sec"] or later["timestamp_sec"])
        if second-first<args.min_end_gap_sec:
            conflicts.append({"video_id":args.video_id,"first_boundary_id":earlier["boundary_id"],"first_timestamp_sec":f"{first:.6f}","second_boundary_id":later["boundary_id"],"second_timestamp_sec":f"{second:.6f}","gap_sec":f"{second-first:.6f}","human_status":"unresolved","note":"Two confirmed rally ends are too close to define separate rally windows."})
            blocked.update((earlier["boundary_id"],later["boundary_id"]))
    output=[]
    for number,end in enumerate(ends,start=1):
        if end["boundary_id"] in blocked: continue
        end_time=float(end["human_timestamp_sec"] or end["timestamp_sec"])
        # A false departure/re-entry pair can occur within the same fixed-camera
        # scene.  Such an entry cannot be the start of the rally that ends almost
        # immediately after it, so fall back to the preceding entry instead.
        candidates=[entry for entry in entries if end_time-float(entry["timestamp_sec"])>args.min_enter_to_end_sec]
        if not candidates: continue
        start=candidates[-1]; start_time=float(start["timestamp_sec"])
        if end_time<=start_time: continue
        old=previous.get(end["boundary_id"], {})
        # A review belongs to a particular proposed start.  If suppression
        # changes that proposal, retain no human decision silently.
        if old.get("start_candidate_id") != start["boundary_id"]:
            old={}
        start_confirmation=old.get("start_confirmation", "unreviewed")
        if start_confirmation == "confirmed": window_status="complete_boundary"
        elif start_confirmation == "not_rally_start": window_status="excluded_or_unresolved"
        else: window_status="needs_start_confirmation"
        output.append({"window_id":f"{args.video_id}_rally_window_{number:03d}","video_id":args.video_id,"source_start_timestamp_sec":f"{start_time:.6f}","source_end_timestamp_sec":f"{end_time:.6f}","start_source_frame":round(start_time*60000/1001),"end_source_frame":round(end_time*60000/1001),"start_candidate_id":start["boundary_id"],"end_boundary_id":end["boundary_id"],"end_boundary_type":"rally_end","start_confirmation":start_confirmation,"human_start_timestamp_sec":old.get("human_start_timestamp_sec", ""),"start_review_note":old.get("start_review_note", ""),"end_confirmation":"confirmed","window_status":window_status,"notes":"Candidate window only; do not use as rally_id or trial_id before start review."})
    fields=["window_id","video_id","source_start_timestamp_sec","source_end_timestamp_sec","start_source_frame","end_source_frame","start_candidate_id","end_boundary_id","end_boundary_type","start_confirmation","human_start_timestamp_sec","start_review_note","end_confirmation","window_status","notes"]
    write(directory/f"{args.video_id}_rally_window_candidates.csv",fields,output)
    write(directory/f"{args.video_id}_rally_boundary_conflicts.csv",["video_id","first_boundary_id","first_timestamp_sec","second_boundary_id","second_timestamp_sec","gap_sec","human_status","note"],conflicts)
    print(f"Wrote {len(output)} rally-window candidates; {len(conflicts)} boundary conflicts need review")
    return 0


if __name__=="__main__": raise SystemExit(main())
