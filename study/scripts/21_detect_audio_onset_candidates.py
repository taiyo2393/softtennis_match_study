"""Extract broadcast-audio onset candidates and relate them to view departures.

This detects abrupt audio-energy events, not verified racket-ball impacts.  It
is intentionally a candidate generator because commentary, applause, and
replay audio can create identical peaks.
"""

from __future__ import annotations

import argparse, csv, subprocess
from pathlib import Path

import numpy as np


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--input",type=Path,required=True); parser.add_argument("--video-id",required=True)
    parser.add_argument("--sample-rate",type=int,default=16000); parser.add_argument("--frame-ms",type=float,default=20); parser.add_argument("--hop-ms",type=float,default=10); args=parser.parse_args()
    project=Path(__file__).resolve().parents[1]; directory=project/"data"/"segments"; sr=args.sample_rate; frame=round(sr*args.frame_ms/1000); hop=round(sr*args.hop_ms/1000)
    completed=subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-nostdin","-i",str(args.input),"-vn","-ac","1","-ar",str(sr),"-f","s16le","pipe:1"],capture_output=True,check=True)
    audio=np.frombuffer(completed.stdout,dtype=np.int16).astype(np.float32)/32768.0
    if len(audio)<frame: raise SystemExit("Audio is too short")
    # Broadband RMS and first-difference RMS serve as inexpensive energy and
    # high-frequency-change proxies. They are not a learned hit classifier.
    sq=np.concatenate(([0.0],np.cumsum(audio*audio))); diff=np.diff(audio,prepend=audio[0]); dsq=np.concatenate(([0.0],np.cumsum(diff*diff)))
    starts=np.arange(0,len(audio)-frame+1,hop); rms=np.sqrt((sq[starts+frame]-sq[starts])/frame+1e-12); drms=np.sqrt((dsq[starts+frame]-dsq[starts])/frame+1e-12)
    energy=20*np.log10(rms+1e-8); change=20*np.log10(drms+1e-8); score=(energy-np.median(energy))/max(np.std(energy),1e-6)+(change-np.median(change))/max(np.std(change),1e-6)
    # Local maxima above an empirical high percentile; threshold is a proposal
    # parameter and must be evaluated against manual events.
    threshold=float(np.percentile(score,97)); radius=max(1,round(0.08/(hop/sr))); peaks=[]; last=-10**9
    for index in np.flatnonzero(score>=threshold):
        if index-last<radius: continue
        lo,hi=max(0,index-radius),min(len(score),index+radius+1)
        best=lo+int(np.argmax(score[lo:hi]))
        if best==index: peaks.append(index); last=index
    onset_rows=[{"video_id":args.video_id,"onset_id":f"{args.video_id}_audio_{n:05d}","timestamp_sec":f"{(starts[i]+frame/2)/sr:.6f}","energy_db":f"{energy[i]:.4f}","change_db":f"{change[i]:.4f}","onset_score":f"{score[i]:.4f}","candidate_status":"unreviewed","note":""} for n,i in enumerate(peaks,start=1)]
    onset_path=directory/f"{args.video_id}_audio_onset_candidates.csv"; fields=list(onset_rows[0]) if onset_rows else ["video_id","onset_id","timestamp_sec","energy_db","change_db","onset_score","candidate_status","note"]
    with onset_path.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(onset_rows)
    with (directory/f"{args.video_id}_rally_boundary_candidates.csv").open(newline="",encoding="utf-8") as f: boundaries=[r for r in csv.DictReader(f) if r["candidate_type"]=="main_camera_departure_candidate"]
    times=np.array([float(row["timestamp_sec"]) for row in onset_rows]); features=[]
    for row in boundaries:
        t=float(row["timestamp_sec"]); after=times[(times>t)&(times<=t+3.0)]; before=times[(times>=t-3.0)&(times<t)]
        features.append({"boundary_id":row["boundary_id"],"video_id":args.video_id,"boundary_timestamp_sec":f"{t:.6f}","onset_count_3s_before":len(before),"last_onset_before_sec":"" if not len(before) else f"{before[-1]:.6f}","onset_count_3s_after":len(after),"first_onset_after_sec":"" if not len(after) else f"{after[0]:.6f}","audio_continues_after":"true" if len(after) else "false","interpretation":"candidate_only"})
    feature_path=directory/f"{args.video_id}_boundary_audio_features.csv";fields=list(features[0]) if features else []
    with feature_path.open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(features)
    print(f"Wrote {len(onset_rows)} audio-onset candidates and {len(features)} boundary audio summaries")


if __name__=="__main__": main()
