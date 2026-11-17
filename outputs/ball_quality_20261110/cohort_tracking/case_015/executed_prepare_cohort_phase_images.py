"""Render unannotated chronological evidence for a fixed new cohort input window."""
import argparse,json
from pathlib import Path
import cv2
import numpy as np
from prepare_prefix_phase_audit import grid,sha,dump

ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'ball_quality_20261110/cohort_tracking'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',required=True);tag=ap.parse_args().tag
    case=BASE/f'case_{tag}';protocol=json.loads((case/'protocol.json').read_text(encoding='utf-8'))
    out=case/'phase_evidence';out.mkdir();assert 'requested_input_frames' in protocol
    times=protocol['requested_input_frames'];first=protocol['contact_candidate_first_source_frame']-60;last=times[-1]+180
    frames=sorted(set(round(f) for f in np.linspace(first,last,24))|{protocol['contact_candidate_first_source_frame'],protocol['contact_candidate_last_source_frame'],times[0],times[-1]})
    requested=set(frames+times);images={}
    cap=cv2.VideoCapture(protocol['source_video']);cap.set(1,first)
    for f in range(first,last+1):
        ok,im=cap.read();assert ok and int(round(cap.get(1)))==f+1
        if f in requested:
            images[f]=cv2.resize(im,(1920,1080),interpolation=cv2.INTER_AREA)
            assert cv2.imwrite(str(out/f'F{f}.jpg'),images[f],[cv2.IMWRITE_JPEG_QUALITY,95])
    cap.release()
    meta=dict(rally_candidate_id=protocol['candidate_id'],input_first_source_frame=times[0],input_end_source_frame=times[-1])
    grid(frames,images,meta,out/'context_grid.jpg');grid(times,images,meta,out/'input_grid.jpg')
    dump(out/'manifest.json',dict(candidate_id=protocol['candidate_id'],source_video=protocol['source_video'],
        source_video_sha256=protocol['source_video_sha256'],context_decoded_frames=[first,last],
        context_grid_source_frames=frames,input_grid_source_frames=times,
        context_is_not_point_boundary_ground_truth=True,images={p.name:sha(p) for p in out.glob('*.jpg')},
        script_sha256=sha(Path(__file__)),grid_helper_sha256=sha(ROOT/'prepare_prefix_phase_audit.py')))
    print(tag,'raw chronological grids saved; no phase decision made.')

if __name__=='__main__':main()
