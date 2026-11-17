"""Audit full-rally ByteTrack candidates without inferring player identity."""
import csv
import importlib.util
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

ROOT = Path(r'C:\Users\ta239\Research\softtennis_match_study\study')
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('select_four', ROOT/'scripts/96_select_four_players_pilot.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as file:
        return list(csv.DictReader(file))

def runs(values):
    result = []
    if not values:
        return result
    start = previous = values[0]
    for value in values[1:]:
        if value != previous + 1:
            result.append((start, previous))
            start = value
        previous = value
    result.append((start, previous))
    return result

summary = []
bins = []
queue = []
for rally in ('044','056'):
    folder = OUT/f'full_rally_{rally}'
    raw = read(folder/'bytetrack.csv')
    first = int(raw[0]['source_frame'])
    last = int(raw[-1]['source_frame'])
    seeds = module.seed_by_court_geometry(raw, first, 3840, 2160)
    by_frame = defaultdict(list)
    for row in raw:
        by_frame[int(row['source_frame'])].append(row)
    selected_ids = set(seeds.values())
    roi_counts = []
    four_raw = 0
    four_seeded = 0
    for frame in range(first,last+1):
        rows = by_frame.get(frame, [])
        roi_count = sum(int(row['in_initial_roi']) for row in rows)
        roi_counts.append(roi_count)
        four_raw += roi_count >= 4
        four_seeded += len({int(row['track_id']) for row in rows} & selected_ids) == 4
    missing_runs = {}
    for slot, track_id in seeds.items():
        observed = {frame for frame, rows in by_frame.items()
                    if any(int(row['track_id']) == track_id for row in rows)}
        missing = [frame for frame in range(first,last+1) if frame not in observed]
        missing_runs[slot] = runs(missing)
        for start,end in missing_runs[slot]:
            if end-start+1 >= 6:
                queue.append(dict(candidate_id=f'match1_rally_window_{rally}',
                                  player_slot=slot, seed_track_id=track_id,
                                  missing_start_frame=start,missing_end_frame=end,
                                  missing_frames=end-start+1,
                                  suggested_review_start=max(first,start-30),
                                  suggested_review_end=min(last,end+30),
                                  identity_review_status='not_reviewed',
                                  warning='seed_id_missing_not_proof_player_absent'))
    for start in range(first,last+1,60):
        end = min(last,start+59)
        frames = list(range(start,end+1))
        bins.append(dict(candidate_id=f'match1_rally_window_{rally}',
                         start_frame=start,end_frame=end,frames=len(frames),
                         frames_with_4plus_roi_candidates=sum(
                             sum(int(row['in_initial_roi']) for row in by_frame.get(frame,[]))>=4
                             for frame in frames),
                         frames_with_all_four_seed_ids=sum(
                             len({int(row['track_id']) for row in by_frame.get(frame,[])} & selected_ids)==4
                             for frame in frames),
                         interpretation='candidate_availability_only'))
    summary.append(dict(candidate_id=f'match1_rally_window_{rally}',
                        first_frame=first,last_frame=last,frames=last-first+1,
                        initial_slot_ids=seeds,
                        frames_with_4plus_roi_candidates=four_raw,
                        frames_with_exactly_four_roi_candidates=sum(n==4 for n in roi_counts),
                        median_roi_candidate_count=median(roi_counts),
                        maximum_roi_candidate_count=max(roi_counts),
                        frames_with_all_four_seed_ids=four_seeded,
                        missing_runs_per_slot={slot:len(value) for slot,value in missing_runs.items()},
                        longest_seed_id_missing_run_per_slot={
                            slot:max((b-a+1 for a,b in value),default=0)
                            for slot,value in missing_runs.items()},
                        warning='No ID or foot-position ground truth; seed ID presence is not identity accuracy.'))

def write_csv(path, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

write_csv(OUT/'full_rally_seed_coverage.csv',bins)
write_csv(OUT/'full_rally_identity_review_queue.csv',queue)
(OUT/'full_rally_tracking_summary.json').write_text(
    json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
print('review_queue_items',len(queue))
