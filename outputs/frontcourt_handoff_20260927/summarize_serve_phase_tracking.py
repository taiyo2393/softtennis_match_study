"""Summarize serve-phase candidate availability, without claiming identity accuracy."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

OUT = Path(__file__).resolve().parent
events = {}
with (OUT / 'event_review_ranges.csv').open(encoding='utf-8-sig', newline='') as file:
    for row in csv.DictReader(file):
        if row['event'] == 'serve_contact':
            events[row['candidate_id'][-3:]] = (int(row['first_possible_source_frame']),
                                                  int(row['last_possible_source_frame']))
seed_rows = {r['candidate_id'][-3:]: r for r in json.loads(
    (OUT / 'serve_seed_availability.json').read_text(encoding='utf-8'))}
results = []
for suffix in ('044', '056'):
    pilot = OUT / f'serve_phase_{suffix}'
    summary = json.loads((pilot / 'summary.json').read_text(encoding='utf-8'))['results'][0]
    first = int(seed_rows[suffix]['first_frame'])
    contact_first, contact_last = events[suffix]
    selected = pilot / 'bytetrack_selected_four.csv'
    if selected.exists():
        with selected.open(encoding='utf-8-sig', newline='') as file:
            rows = list(csv.DictReader(file))
        by_frame = defaultdict(set)
        for row in rows:
            by_frame[int(row['source_frame'])].add(row['player_slot'])
        selected_four = sum(len(by_frame[f]) == 4 for f in range(first, first+180))
        contact_four = sum(len(by_frame[f]) == 4 for f in range(contact_first, contact_last+1))
    else:
        selected_four = None
        contact_four = None
    results.append(dict(candidate_id=f'match1_rally_window_{suffix}',
                        tracker_first_frame=first, tracker_last_frame=first+179,
                        roi_frames_with_four_or_more_candidate_boxes=summary['frames_with_at_least_four_roi_candidates'],
                        geometric_seed_first_frame=seed_rows[suffix]['first_valid_seed_frame'],
                        geometric_seed_available_frames=seed_rows[suffix]['frames_with_four_seed_candidates'],
                        selected_four_candidate_frames=selected_four,
                        provisional_contact_first_frame=contact_first,
                        provisional_contact_last_frame=contact_last,
                        selected_four_during_contact_candidate_frames=contact_four,
                        identity_ground_truth_available=False,
                        status='selection_candidate_only' if selected.exists() else 'initial_seed_failed'))

deferred = OUT / 'serve_phase_056_deferred/bytetrack_four_player_reconnection.csv'
if deferred.exists():
    with deferred.open(encoding='utf-8-sig', newline='') as file:
        rows = list(csv.DictReader(file))
    by_frame = defaultdict(dict)
    source_counts = Counter()
    for row in rows:
        frame = int(row['source_frame'])
        by_frame[frame][row['player_slot']] = row
        source_counts[row['position_source']] += 1
    four_observed = sum(len(by_frame[f]) == 4 and all(
        r['position_source'] != 'interpolated' for r in by_frame[f].values())
        for f in range(154086, 154231))
    four_with_interpolation = sum(len(by_frame[f]) == 4 for f in range(154086, 154231))
    results[1]['deferred_seed'] = dict(
        first_frame=154086, last_frame=154230, frames=145,
        initial_frames_unselected=35,
        four_observed_candidate_frames=four_observed,
        four_with_interpolation_candidate_frames=four_with_interpolation,
        original_observed_slots=source_counts['observed'],
        provisionally_relinked_observed_slots=source_counts['observed_relinked'],
        interpolated_unobserved_slots=source_counts['interpolated'],
        provisional_relink='far_rear ID42 to ID61 at F154136',
        warning='Four candidate boxes can be selected; identity and foot positions remain unverified')
(OUT / 'serve_phase_tracking_audit.json').write_text(
    json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
for result in results:
    print(json.dumps(result, ensure_ascii=False))
