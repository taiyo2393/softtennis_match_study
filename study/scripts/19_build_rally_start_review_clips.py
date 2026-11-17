"""Create short clips around the provisional start of each rally-window candidate."""

from __future__ import annotations

import argparse, csv, subprocess
from pathlib import Path


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--input",type=Path,required=True); parser.add_argument("--video-id",required=True); parser.add_argument("--window-id",action="append",default=[],help="Create only this window ID; may be given more than once."); parser.add_argument("--overwrite",action="store_true")
    args=parser.parse_args(); project=Path(__file__).resolve().parents[1]; directory=project/"data"/"segments"
    with (directory/f"{args.video_id}_rally_window_candidates.csv").open(newline="",encoding="utf-8") as file: rows=list(csv.DictReader(file))
    out=directory/"rally_start_review"/args.video_id; out.mkdir(parents=True,exist_ok=True); made=0
    for row in rows:
        if args.window_id and row["window_id"] not in args.window_id: continue
        video=out/f"{row['window_id']}.mp4"
        if video.exists() and not args.overwrite: continue
        start=max(0.0,float(row["source_start_timestamp_sec"])-3.0)
        subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-y","-ss",f"{start:.6f}","-i",str(args.input),"-t","6","-map","0:v:0","-map","0:a?","-vf","scale=960:-2,drawbox=x=0:y=0:w=iw:h=ih:color=lime@0.95:t=12:enable='between(t,2.8,3.2)'","-r","15","-c:v","libx264","-pix_fmt","yuv420p","-c:a","aac","-movflags","+faststart",str(video)],check=True)
        made+=1
    print(f"Created {made} rally-start review clips in {out}")


if __name__=="__main__": main()
