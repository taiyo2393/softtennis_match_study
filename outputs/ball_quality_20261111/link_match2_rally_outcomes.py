"""Link reviewed scoreboard transitions to provisional receiving positions."""
from collections import Counter, defaultdict

import build_044_spatiotemporal_pilot as common


OUT = common.HERE / 'match2_rally_outcome_review'
RECEIVING = common.HERE / 'match2_receiving_position_trial_v3/stroke_receiving_positions.json'
POINT1 = common.HERE / 'match2_ball_first_point/protocol.json'
POINTS23 = common.HERE / 'match2_points_2_3/protocol.json'
POINT3_REVIEW = common.HERE / 'match2_raw_stroke_review/point3_contact_review.json'
SIDE_BY_SCORE_ROW = ['far', 'near']
SCORE1 = dict(point=1, scoreboard_before=[0, 0], scoreboard_after=[0, 1],
              scoreboard_frames=[12327, 12647])
ENDING = {
    1: dict(label='hitting_side_out', evidence='Original-video user response アウト',
            visual_review_frames=[12554, 12577], exact_terminal_frame=None),
    2: dict(label='no_return_observed', evidence='Source footage: no return seen before camera cut; '
            'point awarded to the hitting side. Racket contact and line judgment remain unverified.',
            visual_review_frames=[14019, 14087], exact_terminal_frame=None),
    3: dict(label='no_return_observed', evidence='Source footage: no return seen. User saw no N2 '
            'racket contact at F15496–15500; the end event and frame were not independently reviewed.',
            visual_review_frames=[15461, 15522], exact_terminal_frame=None),
}


def score_side(before, after):
    deltas = [b - a for a, b in zip(before, after, strict=True)]
    assert sorted(deltas) == [0, 1]
    return SIDE_BY_SCORE_ROW[deltas.index(1)], deltas.index(1)


def main():
    OUT.mkdir(exist_ok=True)
    source = common.load(POINTS23)
    contact = common.load(POINT3_REVIEW)
    assert contact['racket_contact'] == 'no_contact'

    by_rally = defaultdict(list)
    for row in common.load(RECEIVING):
        by_rally[row['point']].append(row)
    scoreboard_rows = [SCORE1, *source['points']]
    terminal_examples, rallies, linked_strokes = [], [], []
    for record in scoreboard_rows:
        point = record['point']
        before, after = record['scoreboard_before'], record['scoreboard_after']
        side, row_id = score_side(before, after)
        cases = sorted(by_rally[point], key=lambda row: row['stroke'])
        assert [r['stroke'] for r in cases] == list(range(1, len(cases) + 1))
        final = cases[-1]
        ending = dict(ENDING[point])
        if point == 3:
            ending['confirmed_local_contact_assessment'] = contact
        serves = cases[0]
        terminal_bracket = record.get('end_bracket') or common.load(POINT1)['provisional_termination']
        rally = dict(point=point,
                     serve_contact_candidate_interval=[serves['candidate_first'], serves['candidate_last']],
                     terminal_candidate_interval=terminal_bracket,
                     scoreboard_before=before, scoreboard_after=after,
                     scoreboard_frames=record['scoreboard_frames'],
                     scoreboard_order=source['score_order'], scoring_side=side,
                     scoring_team=source['score_order'][row_id],
                     score_evidence='scoreboard changes; team sides checked using near-team jerseys',
                     near_team_source_frames={1: [12485, 12510], 2: [13997, 14018], 3: [15232, 15260]}[point],
                     stroke_candidates=len(cases), final_hitter_candidate=final['hitter_candidate'],
                     terminal_review=ending)
        rallies.append(rally)
        for row, following in zip(cases, cases[1:] + [None], strict=True):
            linked_strokes.append(dict(
                point=point, stroke=row['stroke'], hitter_candidate=row['hitter_candidate'],
                contact_interval=[row['candidate_first'], row['candidate_last']],
                receiving_pair_full_contact_window=row['whole_window_pair'],
                bounce_candidate_frame=row['bounce_candidate_frame'],
                receiving_players=row['receiving_players'],
                features_min_max=row['features_min_max'],
                next_hitter_candidate=following['hitter_candidate'] if following else None,
                terminal_review_label=ending['label'] if following is None else None,
                scoring_side=side,
                outcome_role='rally_level_label_shared_across_strokes',
            ))
        assert final['whole_window_pair'] and final['bounce_candidate_court_xy'] is not None
        terminal_examples.append(dict(
            point=point, scoring_side=side, scoring_team=rally['scoring_team'],
            final_hitter_candidate=final['hitter_candidate'],
            terminal_review_label=ending['label'], receiving_players=final['receiving_players'],
            bounce_candidate_frame=final['bounce_candidate_frame'],
            bounce_candidate_court_xy=final['bounce_candidate_court_xy'],
            geometry_midpoint=final['middle_candidate_frame_positions'],
            geometry_min_max=final['features_min_max'],
        ))
    assert all(a['scoreboard_after'] == b['scoreboard_before'] for a, b in zip(rallies, rallies[1:]))
    assert len({(r['point'], r['stroke']) for r in linked_strokes}) == len(linked_strokes) == 13
    common.dump(OUT / 'rally_labels.json', rallies)
    common.dump(OUT / 'stroke_geometry_and_outcome_links.json', linked_strokes)
    common.dump(OUT / 'terminal_placement_examples.json', terminal_examples)
    common.dump(OUT / 'result.json', dict(
        rallies=len(rallies), stroke_candidates=len(linked_strokes),
        scoring_side_counts=dict(Counter(r['scoring_side'] for r in rallies)),
        terminal_records_with_receiving_pair_and_bounce=len(terminal_examples),
        terminal_labels=dict(Counter(r['terminal_review_label'] for r in terminal_examples)),
        terminal_receiving_side='near in all three examples',
        evidence_images=['score_source_review.jpg', 'near_team_side_review.jpg'],
        limitation='A single match with three points. Repeated scoreboard outcomes at the stroke level '
                   'remain three rally outcomes. Shot timing, landing coordinates and causal diagnosis are provisional.',
        source_hashes={str(p): common.sha(p) for p in (RECEIVING, POINT1, POINTS23, POINT3_REVIEW)},
    ))
    print([(r['point'], r['scoring_side'], r['terminal_review']['label']) for r in rallies])


if __name__ == '__main__':
    main()
