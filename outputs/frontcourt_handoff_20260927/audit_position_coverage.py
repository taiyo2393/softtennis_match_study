"""Describe available pilot positions without treating a candidate box as a correct identity."""
import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(r'C:\Users\ta239\Research\softtennis_match_study\study')
OUT = Path(__file__).resolve().parent
PILOTS = {
    'pilot_011': ('match1_rally_window_044', 115380, 115559),
    'pilot_013': ('match1_rally_window_056', 154096, 154275),
}
SLOTS = {'near_front', 'near_rear', 'far_front', 'far_rear'}

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))

labels = {row['candidate_id']: row for row in read(OUT / 'rally_label_review.csv')}
result = []
for pilot, (candidate_id, first, last) in PILOTS.items():
    rows = read(ROOT / 'analysis/person_tracker_comparison' / pilot /
                'bytetrack_four_player_reconnection.csv')
    assert {int(r['source_frame']) for r in rows} <= set(range(first, last + 1))
    by_frame = defaultdict(dict)
    source_counts = Counter()
    for row in rows:
        frame = int(row['source_frame'])
        slot = row['player_slot']
        assert slot in SLOTS and slot not in by_frame[frame]
        by_frame[frame][slot] = row
        source_counts[row['position_source']] += 1
    all_four_observed = sum(
        len(by_frame[frame]) == 4 and all(
            item['position_source'] != 'interpolated'
            for item in by_frame[frame].values())
        for frame in range(first, last + 1))
    all_four_including_interpolation = sum(
        len(by_frame[frame]) == 4 for frame in range(first, last + 1))
    label = labels[candidate_id]
    # The 59.94 fps source has a frame clock near, but not exactly, 60 fps.
    # These are approximate candidate boundaries, not event labels.
    candidate_first = round(float(label['candidate_start_sec']) * 60000 / 1001)
    candidate_last = round(float(label['candidate_end_sec']) * 60000 / 1001)
    result.append({
        'pilot': pilot, 'candidate_id': candidate_id,
        'pilot_first_frame': first, 'pilot_last_frame': last,
        'pilot_frames': last - first + 1,
        'candidate_first_frame_approx': candidate_first,
        'candidate_last_frame_approx': candidate_last,
        'pilot_fraction_of_candidate_approx':
            f'{(last-first+1)/(candidate_last-candidate_first+1):.4f}',
        'all_four_observed_candidate_boxes': all_four_observed,
        'all_four_with_interpolation': all_four_including_interpolation,
        'observed_original_slots': source_counts['observed'],
        'observed_relinked_slots': source_counts['observed_relinked'],
        'interpolated_slots': source_counts['interpolated'],
        'outside_pilot_position_frames':
            (candidate_last-candidate_first+1) - (last-first+1),
        'identity_ground_truth_frames': 0,
        'position_ground_truth_frames': 0,
        'analysis_eligible': label['analysis_eligible'],
    })

path = OUT / 'position_coverage_audit.csv'
with path.open('w', encoding='utf-8-sig', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(result[0]))
    writer.writeheader()
    writer.writerows(result)
for row in result:
    print(row)
