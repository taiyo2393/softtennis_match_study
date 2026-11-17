"""Compare new tracker output with an existing human-audited front-player track.

This validates one player in one clip, not all four players or other matches.
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TRUTH = ROOT / 'data/tracking/front_target_highres/match1_full_match_far_front_reseed_02_exact_tracks_yolo11m.csv'
DEFAULT_EVENT = 'match1_far_front_direction_01_002'


def read(path):
    with path.open(newline='', encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def iou(a, b):
    ax1, ay1, ax2, ay2 = [float(a[k]) for k in ('x1_px', 'y1_px', 'x2_px', 'y2_px')]
    bx1, by1, bx2, by2 = [float(b[k]) for k in ('x1', 'y1', 'x2', 'y2')]
    overlap = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = (ax2-ax1)*(ay2-ay1) + (bx2-bx1)*(by2-by1) - overlap
    return overlap/union if union > 0 else 0


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--pilot', type=Path, required=True)
    p.add_argument('--event', default=DEFAULT_EVENT)
    args = p.parse_args()
    truth = [r for r in read(TRUTH) if r['event_id'] == args.event and r['visible'] == '1']
    if not truth:
        raise SystemExit('No audited reference rows')
    output = []
    for mode in ('bytetrack', 'botsort_reid'):
        frames = defaultdict(list)
        for r in read(args.pilot/f'{mode}.csv'):
            frames[int(r['source_frame'])].append(r)
        ids, scores, misses = [], [], []
        for target in truth:
            frame = int(target['source_frame'])
            options = frames.get(frame, [])
            best = max(options, key=lambda r: iou(target, r)) if options else None
            score = iou(target, best) if best else 0
            scores.append(score)
            if score >= .5:
                ids.append(best['track_id'])
            else:
                misses.append(frame)
        result = dict(mode=mode, reference_event=args.event, reference_frames=len(truth),
                      detections_iou_at_least_050=len(ids), mean_best_iou=round(sum(scores)/len(scores),3),
                      matched_track_ids=dict(Counter(ids)), missed_frames=misses)
        output.append(result)
        print(json.dumps(result, ensure_ascii=False))
    (args.pilot/f'audited_front_comparison_{args.event}.json').write_text(json.dumps(output,ensure_ascii=False,indent=2),encoding='utf-8')


if __name__ == '__main__':
    main()
