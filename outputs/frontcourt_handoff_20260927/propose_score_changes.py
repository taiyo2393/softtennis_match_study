"""OCR-free score-digit suggestions for triage, never authoritative rally labels."""
import csv
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).resolve().parent
with (OUT / 'scoreboard_scan_queue.csv').open(encoding='utf-8-sig', newline='') as stream:
    queue = list(csv.DictReader(stream))
assert len(queue) == 63
pages = {}
def tile(index, sample):
    page = index // 3
    if page not in pages:
        pages[page] = cv2.imread(str(OUT / f'scoreboard_scan_{page:02d}.jpg'))
        assert pages[page] is not None
    y = (index % 3) * 540
    if sample in ('start', 'end_minus_1s'):
        y += 36
    else:
        y += 288
    x = 0 if sample in ('start', 'end_plus_1s') else 900
    return pages[page][y:y+252, x:x+900]

def digit_mask(index, sample, team):
    image = tile(index, sample)
    y = 62 if team == 'top' else 130
    crop = image[y:y+65, 770:825]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = ((gray > 160) & (hsv[:, :, 1] < 100)).astype(np.uint8)
    yy, xx = np.where(mask)
    if len(xx) < 20:
        return None
    glyph = mask[yy.min():yy.max()+1, xx.min():xx.max()+1]
    return cv2.resize(glyph, (32, 48), interpolation=cv2.INTER_NEAREST)

# Visible point digits in the already-reviewed image pages. Their labels only
# train a fixed-font recognizer; all predictions remain review candidates.
seeds = {
    0: (43, 'start', 'top'),
    1: (43, 'end_plus_1s', 'bottom'),
    2: (55, 'start', 'top'),
    3: (55, 'end_plus_1s', 'top'),
    4: (42, 'end_plus_1s', 'bottom'),
}
templates = {digit: digit_mask(*locator) for digit, locator in seeds.items()}
assert all(v is not None for v in templates.values())

def recognize(index, sample, team):
    glyph = digit_mask(index, sample, team)
    if glyph is None:
        return '', 0.0
    distances = sorted((float(np.mean(glyph != template)), digit)
                       for digit, template in templates.items())
    best_error, best_digit = distances[0]
    margin = distances[1][0] - best_error
    if best_error > .23 or margin < .045:
        return '', round(margin, 3)
    return str(best_digit), round(margin, 3)

out = []
for index, row in enumerate(queue):
    digits = {}
    margins = {}
    for sample in ('start', 'end_minus_1s', 'end_plus_1s', 'end_plus_3s'):
        for team in ('top', 'bottom'):
            digits[f'{sample}_{team}'], margins[f'{sample}_{team}'] = recognize(index, sample, team)
    before = (digits['start_top'], digits['start_bottom'])
    pre_end = (digits['end_minus_1s_top'], digits['end_minus_1s_bottom'])
    post1 = (digits['end_plus_1s_top'], digits['end_plus_1s_bottom'])
    post3 = (digits['end_plus_3s_top'], digits['end_plus_3s_bottom'])
    suggestion = ''
    reason = 'unreadable_or_ambiguous'
    if all(before + pre_end + post1 + post3):
        if before == pre_end and post1 == post3:
            delta = (int(post1[0]) - int(pre_end[0]), int(post1[1]) - int(pre_end[1]))
            if delta == (1, 0):
                suggestion, reason = 'top', 'one_point_increment_consistent'
            elif delta == (0, 1):
                suggestion, reason = 'bottom', 'one_point_increment_consistent'
            else:
                reason = 'no_simple_one_point_increment'
        else:
            reason = 'score_changes_outside_expected_pair_or_unstable'
    out.append(dict(candidate_id=row['candidate_id'],
                    legacy_window_status=row['legacy_window_status'],
                    page_image=row['page_image'],
                    score_start='-'.join(before), score_before_end='-'.join(pre_end),
                    score_after_1s='-'.join(post1), score_after_3s='-'.join(post3),
                    suggested_scoreboard_row=suggestion, suggestion_reason=reason,
                    min_digit_margin=min(margins.values()),
                    independent_video_review_status='pending',
                    confirmed_rally_winner=''))

with (OUT / 'score_change_suggestions.csv').open('w', encoding='utf-8-sig', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(out[0]))
    writer.writeheader()
    writer.writerows(out)
print(Counter(r['suggestion_reason'] for r in out))
for key in ('match1_rally_window_044', 'match1_rally_window_056'):
    print(next(r for r in out if r['candidate_id'] == key))
